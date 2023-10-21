from skimage.restoration import denoise_tv_chambolle
from torch import nn 
import torch 
import einops
from .builder import MODELS

@MODELS.register_module
class TV(nn.Module):
    def __init__(self,tv_weight,tv_iter_max):
        super(TV,self).__init__()
        self.tv_weight = tv_weight
        self.tv_iter_max = tv_iter_max 

    def forward(self,x,sigma):
        device = x.device
        b,c,height,width = x.shape
        x = x.view(-1,height,width)
        x = x.cpu().numpy()
        
        x = einops.rearrange(x,"b h w-> h w b")
        # multichannel=False
        # if b*c>1:
        #     multichannel=True
        # x = denoise_tv_chambolle(x, self.tv_weight, n_iter_max=self.tv_iter_max,multichannel=multichannel)
        x = denoise_tv_chambolle(x, self.tv_weight, max_num_iter=self.tv_iter_max,channel_axis=2)
        x = einops.rearrange(x,"h w b-> b h w")
        
        x = torch.from_numpy(x).to(device)
        x = x.view(b,c,height,width)
        return x
