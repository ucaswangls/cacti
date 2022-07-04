from torch import nn 
import torch 
import einops 
import numpy as np 
from .builder import MODELS 

@MODELS.register_module
class DeSCI(nn.Module):
    def __init__(self,
                image_size,
                local_win_size,
                patch_size,
                local_win_step,
                patch_step,
                sim_num,
        ):
        super(DeSCI,self).__init__()
        self.fold1 = nn.Fold(output_size=(image_size,image_size),kernel_size=(patch_size, patch_size),stride=patch_step)
        patch_num = (image_size-patch_size)//patch_step+1
        self.fold2 = nn.Fold(output_size=(patch_num,patch_num),kernel_size=(local_win_size, local_win_size),stride=local_win_step)
        self.unfold1 = nn.Unfold(kernel_size=(patch_size, patch_size),stride=patch_step)
        self.unfold2 = nn.Unfold(kernel_size=(local_win_size, local_win_size),stride=local_win_step)
        self.patch_size = patch_size
        self.local_win_size = local_win_size
        self.sim_num = sim_num
        
    def forward(self,input,sigma):
        sigma = sigma[0].cpu()
        f,c,h,w = input.shape
        input = input.squeeze(1)
        input = einops.repeat(input,"f h w->b f h w",b=1)
        patch_image = self.unfold1(input)
        patch_image = einops.rearrange(patch_image,"b (l f) n->b l f n",l=self.patch_size*self.patch_size)

        mask_1 = torch.ones_like(input)
        mask_1 = self.unfold1(mask_1)
        # mask_1 = einops.rearrange(mask_1,"b (l f) n->b l f n",l=self.patch_size*self.patch_size)
        
        local_win_img_list = []
        mask_2_list = []
        for i in range(f):
            single_patch_image = patch_image[:,:,i,:]
            h = int(np.sqrt(single_patch_image.shape[-1]))
            single_patch_image = einops.rearrange(single_patch_image,"b c (h w)->b c h w",h=h)
            single_local_win_img = self.unfold2(single_patch_image)
            single_local_win_img = einops.rearrange(single_local_win_img,"b (c n) l -> b c n l",n=self.local_win_size*self.local_win_size) 
            local_win_img_list.append(single_local_win_img)

            mask_2 = torch.ones_like(single_patch_image)
            mask_2 = self.unfold2(mask_2)
            mask_2_list.append(mask_2)
        local_win_img = torch.stack(local_win_img_list,dim=1)
        # local_win_img = einops.rearrange(local_win_img,"b (c n) l -> b c n l",n=self.local_win_size*self.local_win_size) 
        center_index = self.local_win_size*self.local_win_size//2
        center_img = local_win_img[:,:,:,center_index,:].unsqueeze(3)
        temp = torch.sqrt(torch.sum(torch.square(local_win_img-center_img),dim=2))
        #sing img
        values,indices = torch.topk(temp,k=self.sim_num,dim=2)
        indices = einops.repeat(indices,"b f n l->b f c n l",c=local_win_img.shape[2])
        # vec = last_out[:,:,image_index,patch_index]
        # output = torch.gather(local_win_img,dim=3,index=indices)
        # output = einops.rearrange(output,"b f c n l->(b f l) n c")
        # wnnm_out = self.WNNM(output,nsigma=sigma)
        # out = einops.rearrange(wnnm_out,"(b f l) n c->b f c n l",b=1,f=8)
        output = torch.gather(local_win_img,dim=3,index=indices)
        output = einops.rearrange(output,"b f c n l->(b f l) n c")
        output_mean = torch.mean(output,dim=1)
        output_mean = einops.repeat(output_mean,"b c->b n c",n=self.sim_num)
        wnnm_in = output - output_mean
        wnnm_input = einops.rearrange(wnnm_in,"b n c->b c n")
        wnnm_out = self.WNNM(wnnm_input,nsigma=sigma)
        
        max = wnnm_out.max()

        min = wnnm_out.min()
        wnnm_out = einops.rearrange(wnnm_out,"b n c->b c n")
        # wnnm_out = wnnm_input
        wnnm_out += output_mean
        # output = wnnm_in+wnnm_out
        # output = wnnm_out
        output += wnnm_out
        # output += output_mean
        out = einops.rearrange(output,"(b f l) n c->b f c n l",b=1,f=8)
        local_win_img.scatter_(dim=3,index=indices,src=out)
        local_win_img = einops.rearrange(local_win_img,"b f c n p -> b f (c n) p") 

        patch_img_list = []
        for i in range(f):
            single_local_win_img = local_win_img[:,i,:,:]
            patch_img = self.fold2(single_local_win_img)
            mask_2 = self.fold2(mask_2_list[i])
            mask_2[mask_2==0]=1
            patch_img = torch.div(patch_img,mask_2)
            patch_img_list.append(patch_img)

        patch_img = torch.stack(patch_img_list,dim=1)

        patch_img = einops.rearrange(patch_img,"b f c w h -> b (c f) (w h)") 
        patch_img = self.fold1(patch_img)
        mask_1 = self.fold1(mask_1)
        mask_1[mask_1==0]=1
        out = torch.div(patch_img,mask_1)
        out = out.squeeze(0)
        out = einops.repeat(out,"f h w->f c h w",c=1)
        p_i = out[0,0,:].detach().cpu().numpy().reshape([256,256])
        # import matplotlib.pyplot as plt
        # plt.figure("1")
        # plt.imshow(p_i,cmap="gray")
        # plt.show()
        return out
    # def WNNM(self,input,nsigma=100/255,c=1.14):
    #     U,S,Vh,= torch.linalg.svd(input,full_matrices=False)
    #     S[S<1.14]==0
    #     out = U@torch.diag_embed(S)@Vh
    #     return out
    def WNNM(self,input,nsigma=100/255,c=1.41):
        U,S,Vh,= torch.linalg.svd(input,full_matrices=False)
        patch_num = self.sim_num
        C0 = c*torch.sqrt(torch.tensor([patch_num]))*2*torch.square(nsigma)
        C0 = C0.to(input.device)
        e =torch.tensor([0.00001]).to(input.device)
        Delta = torch.square(S+e)-4*C0 
        S[Delta<0] = 0
        Delta[Delta<0] = 0
        Sx = S-e+torch.sqrt(Delta)/2
        Sx[Sx<0]=0
        out = U@torch.diag_embed(Sx)@Vh
        return out