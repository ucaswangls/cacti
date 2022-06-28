from cacti.utils.utils import A,At
import torch 
import einops
import cv2 
import numpy as np 
from cacti.utils.demosaic import demosaicing_CFA_Bayer_Menon2007 as demosaicing_bayer

def GAP_denoise(denoiser,x,y,y1,mask,mask_s,sigma,
                _lambda=1,accelerate=True,
                color_denoiser=False,bayer=None,
                use_cv2_demosaic=True
                ):
    batch_size,frames,height,width = mask.shape
    src_x = torch.zeros_like(x).to(x.device)
    yb = A(x,mask)
    if accelerate: # accelerated version of GAP
        y1 = y1 + (y-yb)
        x = x + _lambda*(At((y1-yb)/mask_s,mask)) # GAP_acc
    else:
        x = x + _lambda*(At((y-yb)/mask_s),mask) # GAP
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
        x = denoiser(meas,sigma)
    
    x = einops.rearrange(x,"(b f) c h w->b f h w c",f=frames)
    # x = x.view(batch_size,frames,height,width)
    if color_denoiser:
        src_x[0] = x[0,:,0::2,0::2,0] 
        src_x[1] = x[0,:,0::2,1::2,1] 
        src_x[2] = x[0,:,1::2,0::2,1] 
        src_x[3] = x[0,:,1::2,1::2,2]
        x = src_x 
    else:
        x = einops.reduce(x,"b f h w c->b f h w","max")
    return x,y1