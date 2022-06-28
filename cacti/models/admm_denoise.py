from cacti.utils.utils import A,At
import torch 
import einops
import cv2 
import numpy as np 
from cacti.utils.demosaic import demosaicing_CFA_Bayer_Menon2007 as demosaicing_bayer

def ADMM_denoise(denoiser,theta,y,b_hat,mask,mask_s,sigma,
                _lambda=1,gamma=0.01,
                color_denoiser=False,
                bayer=None,
                use_cv2_demosaic=True
                ):
    batch_size,frames,height,width = mask.shape
    src_theta =  torch.zeros_like(theta).to(theta.device)
    yb = A(theta+b_hat,mask)
    x_hat = (theta+b_hat) + _lambda*At((y-yb)/(mask_s+gamma),mask)
    x = x_hat-b_hat
    #denoise
    if color_denoiser:
        assert bayer is not None,"Bayer is None"
        x_rgb = torch.zeros([frames,height*2, width*2,3]).to(mask.device)
        x_bayer = torch.zeros([frames,height*2, width*2]).to(mask.device)
        for ib in range(len(bayer)): 
            b = bayer[ib]
            x_bayer[:,b[0]::2, b[1]::2] = x[ib,:]
        for imask in range(frames):
            np_x_bayer = x_bayer[imask].cpu().numpy()
            if not use_cv2_demosaic:
                np_x_bayer = demosaicing_bayer(np_x_bayer)
            else:
                np_x_bayer = cv2.cvtColor(np.uint8(np.clip(np_x_bayer,0,1)*255), cv2.COLOR_BAYER_RG2BGR)
                np_x_bayer = np_x_bayer.astype(np.float32)
                np_x_bayer /= 255.
            x_rgb[imask] = torch.from_numpy(np_x_bayer).to(mask.device)
        x = einops.rearrange(x_rgb,"f h w c->1 f h w c")
    else:
        x = einops.rearrange(x,"b f h w->b f h w 1")
    meas = einops.rearrange(x,"b f h w c->(b f) c h w")
    sigma = sigma.expand(meas.shape[0],1)
    with torch.no_grad():
        theta = denoiser(meas,sigma)
    theta = einops.rearrange(theta,"(b f) c h w->b f h w c",f=frames)
    if color_denoiser:
        src_theta[0] = theta[0,:,0::2,0::2,0] 
        src_theta[1] = theta[0,:,0::2,1::2,1] 
        src_theta[2] = theta[0,:,1::2,0::2,1] 
        src_theta[3] = theta[0,:,1::2,1::2,2]
        theta = src_theta 
    else:
        theta = einops.reduce(theta,"b f h w c->b f h w","max")
    b_hat = b_hat - (x_hat-theta)
    return theta,b_hat