import torch
import torch.nn as nn
from math import pi
import numpy as np
from io import StringIO
import numpy as np

import torch
from torch import nn

class SIFLayerMaskOSIRIS(nn.Module):
    def __init__(self, polar_height, polar_width, filter_mat1, filter_mat2, osiris_filters, device, channels = 1):
        super().__init__()
        with torch.no_grad():
            self.device = device
            self.polar_height = polar_height
            self.polar_width = polar_width
            self.channels = channels
            
            self.filter1_size = filter_mat1.shape[0]
            self.num_filters1 = filter_mat1.shape[2]
            self.filter1 = torch.FloatTensor(filter_mat1).to(self.device)
            self.filter_mat1 = filter_mat1
            self.filter1 = torch.moveaxis(self.filter1.unsqueeze(0), 3, 0)
            self.filter1 = torch.flip(self.filter1, dims=[0]).detach().requires_grad_(False)
            #print(self.filter1.shape)
            
            self.filter2_size = filter_mat2.shape[0]
            self.num_filters2 = filter_mat2.shape[2]
            self.filter2 = torch.FloatTensor(filter_mat2).to(self.device)
            self.filter_mat2 = filter_mat2
            self.filter2 = torch.moveaxis(self.filter2.unsqueeze(0), 3, 0)
            self.filter2 = torch.flip(self.filter2, dims=[0]).detach().requires_grad_(False)
            #print(self.filter2.shape)
            
            self.np_filters = []
            with open(osiris_filters, 'r') as osirisFile:
                string_inp = ''
                for line in osirisFile:
                    if not line.strip() == 'end':
                        string_inp += line.strip() + '\n'
                    else:
                        c = StringIO(string_inp)
                        self.np_filters.append(np.loadtxt(c))
                        string_inp = ''
            
            self.torch_filters = []
            
            self.torch_filters.append(self.filter1)
            self.torch_filters.append(self.filter2)
            
            self.max_filter_size = max(self.filter1.shape[2], self.filter2.shape[2])
            
            for np_filter in self.np_filters:
                torch_filter = torch.FloatTensor(np_filter).to(self.device)
                torch_filter = torch_filter.unsqueeze(0).unsqueeze(0).detach().requires_grad_(False)
                self.max_filter_size = max(self.max_filter_size, torch_filter.shape[2])
                #print(torch_filter.shape)
                self.torch_filters.append(torch_filter)
            
            #print('Max Filter Size:', self.max_filter_size)
            self.n_filters = len(self.torch_filters)
        
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        #print(f'grid_sample input shape: {input.shape}, grid shape: {grid.shape}, interp_mode: {interp_mode}')
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1).requires_grad_(False)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, padding_mode='zeros')

    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        with torch.no_grad():
            #image shape: torch.Size([16, 1, 256, 256]) mask shape: torch.Size([16, 1, 256, 256]) pupil_xyr shape: torch.Size([16, 3]) iris_xyr shape: torch.Size([16, 3])f
            batch_size = image.shape[0]
            width = image.shape[3]
            height = image.shape[2]
    
            polar_height = self.polar_height
            polar_width = self.polar_width
    
            pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
            iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
            
            theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).requires_grad_(False).to(self.device)
    
            # image shape: torch.Size([16, 1, 256, 256]) mask shape: torch.Size([16, 1, 256, 256]) pupil_xyr shape: torch.Size([16, 1, 3]) iris_xyr shape: torch.Size([16, 1, 3])
            pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
    
            radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False).to(self.device)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
    
            x = (pxCoords + ixCoords).float()
            x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512
    
            y = (pyCoords + iyCoords).float()
            y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512
    
            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).detach().requires_grad_(False)
            
            if mask is not None:
                if torch.is_tensor(mask):
                    mask_t = mask.clone().detach().to(self.device)
                else: 
                    mask_t = torch.tensor(mask).float().to(self.device)
                mask_polar = self.grid_sample(mask_t, grid_sample_mat, interp_mode='nearest')
                mask_polar = (mask_polar - mask_polar.min()) / mask_polar.max()
                mask_polar = torch.gt(mask_polar, 0.5)
            else:
                mask_polar = None
                
        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
        return image_polar, mask_polar
    
    def getCodes(self, image_polar):
        codes_list = []
        for filter in self.torch_filters:
            r1 = int(filter.shape[2]/2)
            r2 = int(filter.shape[3]/2)
            imgWrap = nn.functional.pad(image_polar, (r2, r2, 0, 0), mode='circular')
            imgWrap = nn.functional.pad(imgWrap, (0, 0, r1, r1), mode='replicate')
            code = nn.functional.conv2d(imgWrap, filter, stride=1, padding='valid')
            codes_list.append(code)
        codes = torch.cat(codes_list, dim=1)
        return codes

    def forward(self, image, pupil_xyr, iris_xyr, mask=None):
        image_polar, mask_polar = self.cartToPol(image, mask, pupil_xyr, iris_xyr)
        codes = self.getCodes(image_polar)
        return codes, image_polar, mask_polar