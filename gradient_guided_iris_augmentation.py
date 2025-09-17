
import os
from typing import Optional
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import legacy
from torch.optim import Adam, AdamW
from polar_normalization import PolarNormalization
from scipy import io
from IdentityLoss import SIFLayerMaskOSIRIS
from IdentityLoss import *
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------    

def calc_id_loss(out_imgs, init_img, init_img_mask, init_img_pupil_xyr, init_img_iris_xyr, device, sifLossModel):

        id_losses = 0
        init_img_pxyr_t = init_img_pupil_xyr
        init_img_ixyr_t = init_img_iris_xyr
        init_img_mask_t = init_img_mask.unsqueeze(0).unsqueeze(0).to(device) ## type: <class 'torch.Tensor'>, torch.Size([1, 1, 240, 320])

        
        with torch.no_grad():
            sif_code_init_img, init_img_polar, init_img_mask_polar = sifLossModel((init_img + 1)/2, init_img_pxyr_t, init_img_ixyr_t, init_img_mask_t)
            init_img_mask_polar = init_img_mask_polar.repeat(1, sif_code_init_img.shape[1], 1, 1)
            sif_code_init_img_masked = sif_code_init_img * init_img_mask_polar.requires_grad_(False)
        
        sif_code_gen, gen_img_polar, _ = sifLossModel((out_imgs+1)/2, init_img_pxyr_t, init_img_ixyr_t, None)
        sif_code_gen_masked = sif_code_gen * init_img_mask_polar.requires_grad_(False)
        id_losses += torch.mean(torch.abs(sif_code_gen_masked - sif_code_init_img_masked.requires_grad_(False)), dim=(1,2,3))
                
        return id_losses 


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------    
def pupilIrisRatio(PUPIL_RADIUS,IRIS_RADIUS):
    
    PUPIL_IRIS_RATIO = (100 * (PUPIL_RADIUS/IRIS_RADIUS))
    return PUPIL_IRIS_RATIO

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def eyelid_opening(binary_mask):
    
    # Find the row indices where there is at least one white pixel (find the mask area)
    rows_with_white = np.where(binary_mask.sum(axis=1) > 0)[0] 

    # Calculate mask height = (lower right- Upper left) + 1
    if rows_with_white.size > 0:
        return rows_with_white[-1] - rows_with_white[0] + 1 
    else:
        return 0    
    
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def sharpness(IRIS_IMAGE, MASK_t_b, device, use_on_iris_mask=False):    
    '''
    F: filter
    I: image
    IF = I*g
    ss = sum(IF**2)
    Power = ss/(Wf, Hf) # where wF and hF are the width and height of IF(x,y)
    c = 1800000
    Sharpness = 100 * (Power**2 / (Power**2 + c**2))
    '''
    F_np = np.array([[0, 1, 1, 2, 2, 2, 1, 1, 0],
                    [1, 2, 4, 5, 5, 5, 4, 2, 1],
                    [1, 4, 5, 3, 0, 3, 5, 4, 1],
                    [2, 5, 3, -12, -24, -12, 3, 5, 2],
                    [2, 5, 0, -24, -40, -24, 0, 5, 2],
                    [2, 5, 3, -12, -24, -12, 3, 5, 2],
                    [1, 4, 5, 3, 0, 3, 5, 4, 1],
                    [1, 2, 4, 5, 5, 5, 4, 2, 1],
                    [0, 1, 1, 2, 2, 2, 1, 1, 0]])
    # convert filter to tensor
    F_t = torch.from_numpy(F_np).float().unsqueeze(0).unsqueeze(0).to(device) #F_t shape torch.Size([1, 1, 9, 9])
    IRIS_IMAGE_t = IRIS_IMAGE.float().to(device)                            # type torch.float32, torch.Size([1, 1, 488, 648])
    assert IRIS_IMAGE_t.requires_grad, "IRIS_IMAGE_t doesnt have grad"
    IF = nn.functional.conv2d(IRIS_IMAGE_t, F_t, padding='same') # type IF torch.float32 , shape torch.Size([1, 1, 480, 640])
    assert IF.requires_grad, "IF doesnt have grad"
    # Extract iris area only 
    IF_Square = torch.square(IF)
    
   
    if use_on_iris_mask:
        IF_masked = torch.mul(IF_Square, MASK_t_b)

    # Calculate the sharpness only on iris area 
    mask_pixel_count = torch.sum(MASK_t_b).item()
    
    POWER = torch.sum(IF_masked)/(mask_pixel_count) # Note: power should be number    
    # sharpness calculation
    C = 1800000
    SHARPNESS = 100 * (torch.square(POWER)/(torch.square(POWER)+C**2))
    assert SHARPNESS.requires_grad, "Sharpness doesnt have gradient"

    return SHARPNESS
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def generate_img_z(z, G, label, truncation_psi, noise_mode,  device):
        
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode) #img shape torch.Size([1, 1, 512, 512])
    # Resize the image to be torch.Size([1, 1, 480, 640])
    img_t_resized = F.interpolate(img, size=(480, 640), mode='bilinear', align_corners=False)
    # Normalize the image to [0, 255] for extracting the mask. Also, Sharpness loss needs a normalized image
    img_norm_t_resized = 255 * ((img_t_resized-torch.min(img_t_resized)) / (torch.max(img_t_resized)-torch.min(img_t_resized)))
    assert img_norm_t_resized.requires_grad, "img_norm_t_resized doesnt have gradient (image_gen())"
   
    # Steps to Extract image mask and extracting only iris area
    img_norm = (img_norm_t_resized.permute(0, 2, 3, 1)).to(torch.uint8) #(b,c,w,h) --> (b,w,h,c)
    # Squeeze the image, convert it to pil image (mask_circle module needs PIL image)
    pil_img = PIL.Image.fromarray(img_norm.squeeze(0).squeeze(2).cpu().numpy(), 'L') #(h,w)

    # Extract the Mask and Circle (Resize the image to be torch.Size([1, 1, 240, 320]))
    polar_normalizer = PolarNormalization(device=device)
    img_t_resized_240_320 = F.interpolate(img, size=(240, 320))#, mode='bilinear')# mode='nearest') 
    assert img_t_resized_240_320.requires_grad, "img_t_norm_resized_240_320 doesnt have gradient (image_gen())"

    mask, mask_logit_t = polar_normalizer.getMask(img_t_resized_240_320) 
    pupil_xyr, iris_xyr = polar_normalizer.circApprox(img_t_resized_240_320) 
    
    
    # Convert mask to a PyTorch tensor and binarize it
    MASK_t = torch.tensor(np.array(mask), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # mask shape torch.Size([1, 1, 480, 640]) max: 255., min: 0, type: torch.float32
    Mask_t_binary = (MASK_t /255.0).float() # Mask Binary Max: 1.0, Min: 0.0, Shape: torch.Size([1, 1, 480, 640]), type: torch.float32

    return img_norm_t_resized, pil_img, img_t_resized, Mask_t_binary, pupil_xyr, iris_xyr, mask_logit_t, mask

# ---------------------------------------------------------------------------
#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default='./network.pkl')
@click.option('--steps', 'steps', type=int, help='Number of iterations for optimization', required=True, default=2000)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--target_val', type=float, help='value desire to reach', default=1.0, required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, default='./generated_samples', metavar='DIR')
@click.option('--seed', type=int, help='Define a random seed to generate an image-If you want a new image enter a different seed from what you set before', required=True)
@click.option('--use_on_iris_mask', type=bool, help='Apply on iris mask or iris center', default=True)
@click.option('--use_identity', type=bool, help='if you want to use identity loss', default=False)
@click.option('--iris_dec', type=bool, help='if you want to decrease the iris size', default=False)
@click.option('--iris_inc', type=bool, help='if you want to increase the iris size', default=False)
@click.option('--pir_dec', type=bool, help='if you want to decrease the pupil to iris ratio', default=False)
@click.option('--pir_inc', type=bool, help='if you want to increase the pupil to iris ratio', default=False)
@click.option('--pupil_dec', type=bool, help='if you want to decrease the pupil size', default=False)
@click.option('--pupil_inc', type=bool, help='if you want to increase the pupil size', default=False)
@click.option('--sharp_dec', type=bool, help='if you want to decrease the sharpness', default=False)
@click.option('--sharp_inc', type=bool, help='if you want to increase the sharpness', default=False)


def generate_images(
    ctx: click.Context,
    network_pkl: str,
    steps: int,
    truncation_psi: float,
    target_val: float,
    noise_mode: str,
    outdir: str,
    seed: int,
    class_idx: Optional[int],
    pupil_dec: bool,
    pupil_inc: bool,
    iris_dec: bool,
    iris_inc: bool,
    pir_dec: bool,
    pir_inc: bool,
    use_on_iris_mask: bool,
    sharp_dec: bool,
    sharp_inc: bool,
    use_identity: bool):

  
    outdir = os.path.join(
    outdir,
    f'{"idLoss" if use_identity else "noID_Loss"}',
    f'{seed}',f'{"iris_dec" if iris_dec else ""}'
    f'{"iris_inc" if iris_inc else ""}'
    f'{"pupil_dec" if pupil_dec else ""}'
    f'{"pupil_inc" if pupil_inc else ""}'
    f'{"pir_dec" if pir_dec else ""}'
    f'{"pir_inc" if pir_inc else ""}'
    f'{"sharp_dec" if sharp_dec else ""}'
    f'{"sharp_inc" if sharp_inc else ""}')                      
    os.makedirs(outdir, exist_ok=True)
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    
    
    if use_identity:
        # initialize the SIF loss model
        filter_mat1 = io.loadmat('./filters/ICAtextureFilters_15x15_7bit.mat')['ICAtextureFilters']
        filter_mat2 = io.loadmat('./filters/ICAtextureFilters_17x17_5bit.mat')['ICAtextureFilters']
        sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = './filters/osiris_filters.txt', device=device).to(device)

    with dnnlib.util.open_url(network_pkl) as f:    
        G = legacy.load_network_pkl(f)['G_ema'].to(device) 
   
    
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    # ---------------------------------------------------------------------------------- #
    #               1. Generate an image using a random z
    # ---------------------------------------------------------------------------------- #
    # Generate images from z (Z_init)
    rand_seed = torch.Generator().manual_seed(seed)
    z_init = torch.randn(1, G.z_dim, generator=rand_seed).to(device=device).requires_grad_(True)
    assert z_init.requires_grad, "Uploaded z_init doesnt have gradient"

    # Generate image using z_init     #mask_log_t_init torch.Size([1, 240, 320])
    init_gen_img_norm, gen_img_pil, init_gen_img, Mask_t_binary, PUPIL_xyr, IRIS_xyr, mask_log_t_init, mask = generate_img_z(z_init, G, label, truncation_psi=truncation_psi, noise_mode=noise_mode, device=device)  
    assert mask_log_t_init.requires_grad, "mask_log_t_init doesnt have gradient"
    
    # if The mask is not None
    if np.all(mask == 0):
        # Generate a new z
        rand_seed = torch.Generator().manual_seed(seed+1)
        z_init = torch.randn(1, G.z_dim, generator=rand_seed).to(device=device).requires_grad_(True)
        assert z_init.requires_grad, "Uploaded z_init doesnt have gradient"

        # Generate image using z_init     #mask_log_t_init torch.Size([1, 240, 320])
        init_gen_img_norm, gen_img_pil, init_gen_img, Mask_t_binary, PUPIL_xyr, IRIS_xyr, mask_log_t_init, mask = generate_img_z(z_init, G, label, truncation_psi=truncation_psi, noise_mode=noise_mode, device=device)  
        assert mask_log_t_init.requires_grad, "mask_log_t_init doesnt have gradient"
    
    else:   
                
        init_img_iris_xyr = torch.tensor(PUPIL_xyr)
        init_img_iris_xyr = init_img_iris_xyr.unsqueeze(0).to(device)
        
        init_img_pupil_xyr = torch.tensor(IRIS_xyr)
        init_img_pupil_xyr= init_img_pupil_xyr.unsqueeze(0).to(device)
        
        init_img_mask = Mask_t_binary.squeeze(0).squeeze(0).to(device)  
        mask_t = torch.where(torch.sigmoid(mask_log_t_init)> 0.5, 1.0, 0.0).requires_grad_(False)
        
        init_pir = pupilIrisRatio(PUPIL_xyr[2], IRIS_xyr[2])
        init_elp = eyelid_opening(mask)
        init_sharpness_value = sharpness(init_gen_img_norm, Mask_t_binary, device, use_on_iris_mask) 
        
        
        gen_img_pil.save(f'{outdir}/init_Sharp_{init_sharpness_value.item():.2f}_seed_{seed}_pr_{int(PUPIL_xyr[2])}_ir_{int(IRIS_xyr[2])}_elop_{init_elp}_pir_{init_pir:.3f}.png')

        # Define Optimizer
        if use_identity:
            learning_rate = 0.03
            optimizer = Adam([z_init], lr=learning_rate)
          
        else:
            learning_rate = 0.03
            optimizer = AdamW([z_init], lr=learning_rate)
        
        # Optimize z_init
        for i in range(steps):    
            optimizer.zero_grad()
            
            gen_img_norm, gen_img_pil, gen_img, Mask_t_binary, PUPIL_xyr, IRIS_xyr, mask_log_t, mask = generate_img_z(z_init, G, label, truncation_psi=truncation_psi, noise_mode=noise_mode, device=device) 
            assert mask_log_t.requires_grad, "mask_log_t doesnt have gradient"
            assert PUPIL_xyr[2].requires_grad, "PUPIL_xyr[2] doesnt have gradient"
            
            # if The mask is not empty
            if np.all(mask == 0):
                print("The mask is empty and can't optimize the image...")
    
            else:
                
                # Penalty for eyelid opening
                elp_value = eyelid_opening(mask)
                penalty_weight_elp = 20.0
                penalty_elp = np.abs(elp_value - init_elp) 
                penalty_term_elp = penalty_weight_elp * penalty_elp

                # Mask Loss 
                mask_loss = nn.BCEWithLogitsLoss()(mask_log_t, mask_t)
                
                # Sharpness
                sharpness_value = sharpness(gen_img_norm, Mask_t_binary, device, use_on_iris_mask)
                
                pr_value = PUPIL_xyr[2]
                ir_value = IRIS_xyr[2]
                pir_value =  pupilIrisRatio(pr_value, ir_value)
                
                
                # ---------------------------------------------------------------------------------- #
                #                                  Total Loss                                        #             
                # ---------------------------------------------------------------------------------- #
                z = z_init.clone().detach()
                if use_identity:
                    id_loss = calc_id_loss(gen_img_norm, init_gen_img_norm, init_img_mask, init_img_pupil_xyr, init_img_iris_xyr, device, sifLossModel)
                else:
                    id_loss = 0.0
                    
                if iris_dec:
                    total_loss = torch.abs(ir_value - target_val) + id_loss  
                if iris_inc:
                    total_loss = torch.abs(ir_value - target_val) + id_loss  + penalty_term_elp
                # Gradient clipping
                if pupil_dec or pupil_inc:
                    total_loss = torch.abs(pr_value - target_val) + id_loss 
                
                if pir_dec or pir_inc:
                    total_loss = torch.abs(pir_value - target_val) + id_loss  
                
                if sharp_dec or sharp_inc:
                    total_loss = torch.abs(sharpness_value - target_val) + id_loss 

                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_([z_init], max_norm=1.0) 
                optimizer.step()

                if use_identity:
                    # Generate a new image using optimized z_init) ,
                    print(f'Step {i}: total_loss {total_loss.item():.3f}, sharpness {sharpness_value:.3f}, mask_loss {mask_loss:.3f}, id_loss {id_loss.item():.3f}, , pr {pr_value:.3f}, ir {ir_value:.3f}, elp {elp_value:.3f}, pir {pir_value:.3f}') #contrast {contrast_value.item():.3f}, brightness {brightness_value.item():.3f}')
                    gen_img_pil.save(f'{outdir}/step{i}_seed_{seed}_sharp_{sharpness_value.item():.2f}_mask_loss_{mask_loss:.3f}_pr_{int(PUPIL_xyr[2])}_ir_{int(IRIS_xyr[2])}_elop_{elp_value}_id_loss_{id_loss.item():.3f}_pir_{pir_value:.2f}.png')
                else:
                    print(f'Step {i}: total_loss {total_loss.item():.3f}, sharpness {sharpness_value:.3f}, mask_loss {mask_loss:.3f}, pr {pr_value:.3f}, ir {ir_value:.3f}, elp {elp_value:.3f}, pir {pir_value:.3f}')
                    gen_img_pil.save(f'{outdir}/step{i}_seed_{seed}_sharp_{sharpness_value.item():.2f}_mask_loss_{mask_loss:.3f}_pr_{int(PUPIL_xyr[2])}_ir_{int(IRIS_xyr[2])}_elop_{elp_value}_pir_{pir_value:.2f}.png')

                
    #----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() 