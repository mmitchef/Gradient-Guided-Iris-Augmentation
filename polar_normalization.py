import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from math import pi
import math
from torchvision import models

class fclayer(nn.Module):
    def __init__(self, in_h = 8, in_w = 10, out_n = 6):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.out_n = out_n
        self.fc_list = []
        for i in range(out_n):
            self.fc_list.append(nn.Linear(in_h*in_w, 1))
        self.fc_list = nn.ModuleList(self.fc_list)
    def forward(self, x):
        x = x.reshape(-1, 6, self.in_h, self.in_w)
        outs = []
        for i in range(self.out_n):
            outs.append(self.fc_list[i](x[:, i, :, :].reshape(-1, self.in_h*self.in_w)))
        out = torch.cat(outs, 1)
        return out

class conv(nn.Module):
    def __init__(self, in_channels=512, out_n = 6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_n, kernel_size=1, stride=1, padding='same')
    def forward(self, x):
        x = self.conv(x)
        return x

class SharedAtrousConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(int(out_channels/2), in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weights, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(int(out_channels/2)))
            self.bias2 = nn.Parameter(torch.zeros(int(out_channels/2)))
        else:
            self.bias1 = None
            self.bias2 = None
    def forward(self, x):
        x1 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', bias=self.bias1)
        x2 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', dilation=2, bias=self.bias2)
        x3 = torch.cat([x1, x2], 1)
        return x3

class SharedAtrousResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, middle_channels, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            SharedAtrousConv2d(middle_channels, out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.relu(x)
        return x
    
class Resize(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)
        
class NestedSharedAtrousResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=64, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        self.up = Resize(scale_factor=2, mode='bilinear')

        self.conv0_0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class PolarNormalization(object):
    def __init__(self, polar_height = 64, polar_width = 512, mask_net_path = './nestedsharedatrousresunet-135-0.026591-maskIoU-0.942362.pth', circle_net_path = './resnet34-1583-0.045002-maskIoU-0.93717.pth', device='cpu'):
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.circle_net_path = circle_net_path
        self.mask_net_path = mask_net_path
        self.device = torch.device(device)
        self.NET_INPUT_SIZE = (320,240)
        #with torch.inference_mode():
        with torch.enable_grad():
            self.circle_model = models.resnet34()
            self.circle_model.avgpool = conv(in_channels=512, out_n=6)
            self.circle_model.fc = fclayer(out_n=6)
            try:
                self.circle_model.load_state_dict(torch.load(self.circle_net_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.circle_model.load_state_dict(torch.load(self.circle_net_path,
                        map_location = lambda storage, loc: storage))
            self.circle_model = self.circle_model.to(self.device)
            self.circle_model.eval()
            self.mask_model = NestedSharedAtrousResUNet(1, 1, width=64, resolution=(self.NET_INPUT_SIZE[1], self.NET_INPUT_SIZE[0]))
            try:
                self.mask_model.load_state_dict(torch.load(self.mask_net_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.mask_model.load_state_dict(torch.load(self.mask_net_path,
                        map_location = lambda storage, loc: storage))
            self.mask_model = self.mask_model.to(self.device)
            self.mask_model.eval()
        self.input_transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,))
        ])
        
        self.inp_transform = Compose([
            Normalize(mean=(0.5,), std=(0.5,))
        ])
        
    
    #@torch.inference_mode()
    def getMask(self, image):
        #w,h = image.size
        #image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_CUBIC)
        #mask_logit_t = self.mask_model(Variable(self.input_transform(image).unsqueeze(0).to(self.device)))[0]
        #assert image.requires_grad, "image doesnt have gradient in getMask"
        w,h = 640, 480

        with torch.enable_grad():
            #mask_logit_t = self.mask_model(self.inp_transform(image).to(self.device))[0]
            mask_logit_t = self.mask_model(image.to(self.device))[0]

        assert  mask_logit_t.requires_grad, " mask_logit_t doesnt have gradient"
        
        mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)
        mask = mask_t.cpu().numpy()[0]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        #print('Mask Shape: ', mask.shape)

        return mask, mask_logit_t

    #@torch.inference_mode()
    def circApprox(self, image):
        w,h = 640, 480
        #assert image.requires_grad, "image doesnt have gradient in circApprox"
        #with torch.no_grad():
        #    inp_xyr_t = self.circle_model(Variable(self.input_transform(image).unsqueeze(0).repeat(1,3,1,1).to(self.device)))
        with torch.enable_grad():
            #inp_xyr_t = self.circle_model(self.inp_transform(image).repeat(1,3,1,1).to(self.device))
            inp_xyr_t = self.circle_model(image.repeat(1,3,1,1).to(self.device))

        #Circle params
        diag = torch.tensor(math.sqrt(w**2 + h**2))
        pupil_x = inp_xyr_t[0][0] * w                
        pupil_y = inp_xyr_t[0][1]* h                  
        pupil_r = inp_xyr_t[0][2]  * 0.5 * 0.8 * diag 
        iris_x = inp_xyr_t[0][3] * w         
        iris_y = inp_xyr_t[0][4] * h           
        iris_r = inp_xyr_t[0][5] * 0.5 * diag   
       
        pupil_xyr = (pupil_x, pupil_y, pupil_r)
        iris_xyr = (iris_x, iris_y, iris_r)

        return pupil_xyr, iris_xyr

    #@torch.inference_mode()
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, align_corners=False)

    #@torch.inference_mode()
    def visualize_image(self, image):
        mask = self.getMask(image)
        pupil_xyr, iris_xyr = self.circApprox(image)
        imVis = np.stack((np.array(image),)*3, axis=-1)
        #print(imVis.shape)
        try:
            imVis[:,:,1] = np.clip(imVis[:,:,1] + 0.25*mask,0,255)
        except:
            print("Mask could not be visualized.")
            pass
        try:
            imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        except:
            print("Pupil circle could not be visualized, values are: ", pupil_xyr)
            pass
        try:
            imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)
        except:
            print("Iris circle could not be visualized, values are: ", iris_xyr)
            pass
        imVis_pil = Image.fromarray(imVis)
        
        return imVis_pil

    # Rubbersheet model-based Cartesian-to-polar transformation using bilinear interpolation from torch grid sample
    #@torch.inference_mode()
    def cartToPol(self, image, pupil_xyr, iris_xyr, mask=None):

        if pupil_xyr is None or iris_xyr is None:
            return None, None
        
        image = ToTensor()(image).unsqueeze(0) * 255
        if mask is not None:
            mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
        width = image.shape[3]
        height = image.shape[2]

        polar_height = self.polar_height
        polar_width = self.polar_width

        pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float()
        iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float()
        
        theta = (2*pi*torch.linspace(1,polar_width,polar_width)/polar_width)
        pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
        pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
        
        ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
        iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

        radius = (torch.linspace(0,polar_height,polar_height)/polar_height).reshape(-1, 1)  #64 x 1
        
        pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        
        ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

        x = torch.clamp(pxCoords + ixCoords, 0, width-1).float()
        x_norm = (x/(width-1))*2 - 1 #b x 64 x 512

        y = torch.clamp(pyCoords + iyCoords, 0, height-1).float()
        y_norm = (y/(height-1))*2 - 1  #b x 64 x 512

        grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
        image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
        if mask is not None:
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')
            mask_polar = (mask_polar>0.5).long() * 255
            return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)
        else:
            return (image_polar[0][0].cpu().numpy()).astype(np.uint8), None

    #@torch.inference_mode()
    def convert_to_polar(self, image, mask=None): #PIL image as input
        pupil_xyr, iris_xyr = self.circApprox(image)
        image_polar, mask_polar = self.cartToPol(image, pupil_xyr, iris_xyr, mask)
        return image_polar, mask_polar