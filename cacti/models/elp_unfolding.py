import einops
import torch
from torch import nn
from .builder import MODELS
from .elp_function import Visual_Transformer,SCIbackwardresem

class AL_net(nn.Module):
    def __init__(self,in_ch,pres_ch,init_channels,number1, dropout = 0.1):
        super(AL_net,self).__init__()
       
                
        self.pres_ch= pres_ch     
        self.gamma1 = torch.nn.Parameter(torch.Tensor([1]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([1]))
        
        self.visualtransformer=Visual_Transformer(C_dim=init_channels, in_ch=in_ch,initial_judge=number1)
                     

    def forward(self,v,x, lamda_1,lamda_2,mask,mask_sum,measurement,measurement_mean,iter_,x1=None,x2=None,x3=None,x4=None,x5=None):

        if iter_==0:
            mid_b=(lamda_2-mask*lamda_1)/self.gamma2            
            gamma=self.gamma2/self.gamma1
            x=(v+mid_b)+mask*(torch.div((measurement-torch.sum(mask*(v+mid_b),1,keepdim=True)),(mask_sum+gamma)))
        batch, C, H, W = mask.shape
        noise_map=self.gamma2.expand((batch, 1, H, W))
        img=x-lamda_2/self.gamma2
        x0=torch.cat((noise_map,measurement_mean),1)
        # x0=torch.cat((img,noise_map),1)
        if iter_==0:
            v,x1,x2,x3,x4,x5=self.visualtransformer(iter_,img,x0,self.pres_ch)
        else:
            v,x1,x2,x3,x4,x5=self.visualtransformer(iter_,img,x0,self.pres_ch,x1,x2,x3,x4,x5)
        lamda_1=lamda_1-self.gamma1*(measurement-torch.sum(mask*x,1,keepdim=True))
        lamda_2=lamda_2-self.gamma2*(x-v)
        mid_b=(lamda_2-mask*lamda_1)/self.gamma2            
        gamma=self.gamma2/self.gamma1
        x=(v+mid_b)+mask*(torch.div((measurement-torch.sum(mask*(v+mid_b),1,keepdim=True)),(mask_sum+gamma)))

        return	v,x,x1,x2,x3,x4,x5,lamda_1,lamda_2,self.gamma1,self.gamma2

class SCIbackwardinit(nn.Module):
    def __init__(self,in_ch,pres_ch,init_channels,iter_number, dropout = 0.1):
        super().__init__()       
        
        alnet=[]
        self.iter_number = iter_number      
       
        for i in range(self.iter_number):
            if i==0:
                alnet.append(AL_net(in_ch,pres_ch,init_channels,1))
            else:
                alnet.append(AL_net(in_ch,pres_ch,init_channels,2))	
           # up_samples.append(Upsample(BasicBlock,3,3*ntemp,4*ntemp))

        self.AL=nn.ModuleList(alnet)

    def forward(self, mask, measurement,img_out_ori):
        #nrow, ncol, ntemp, batch_size,iter_number=args['patchsize'],args['patchsize'],args['temporal_length'],args['batch_size'],args['iter__number']
       #T_or=torch.ones(batch_size,head_num,L_num)	
       #  yb = A(theta+b); 	v = (theta+b)+At((y-yb)./(mask_sum+gamma)) 
       #  yb = A(v+b); 	x = (v+b)+At((y-yb)./(mask_sum+gamma)), b=λ_2−𝐴^𝑇 λ_1, gamma=γ_2/γ_1        
        mask_sum=torch.sum(mask,1,keepdim=True)
        mask_sum[mask_sum==0]=1
        measurement_mean=measurement/mask_sum
        x_list,v_list = [],[]
        batch, C, H, W = mask.shape
        lamda_1=torch.zeros_like(measurement)
        lamda_2=torch.zeros_like(mask)
        v=img_out_ori.clone()      
        x=v
       
        for iter_ in range(self.iter_number):
            if iter_==0:
                v,x,x1,x2,x3,x4,x5,lamda_1,lamda_2,gamma1,gamma20=self.AL[iter_](v,x,lamda_1,lamda_2,mask,mask_sum,measurement,measurement_mean,iter_)

            else:
                v,x,x1,x2,x3,x4,x5,lamda_1,lamda_2,gamma1,gamma20=self.AL[iter_](v,x,lamda_1,lamda_2,mask,mask_sum,measurement,measurement_mean,iter_,x1,x2,x3,x4,x5)

            x_list.append(x),v_list.append(v)
        

        #output = v_list[-3:]
        gamma2=gamma20.unsqueeze(0).repeat(batch,1)
        gamma1=gamma1.unsqueeze(0).repeat(batch,1)

        return	x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_2

@MODELS.register_module
class ELPUnfolding(nn.Module):
    def __init__(self, 
                in_ch = 8,
                pres_ch = 8,
                init_channels = 512,
                iter_number = 8,
                priors = 6):
        super().__init__()       
        SCI_backward=[]
        SCI_backwardinit=SCIbackwardinit(in_ch,pres_ch,init_channels,iter_number)        #
        SCI_backward.append(SCI_backwardinit)
         
        self.prior=priors
        for i in range(0,self.prior-1):             
            SCI_backwardresem=SCIbackwardresem(in_ch,pres_ch,init_channels,i+1)          
            SCI_backward.append(SCI_backwardresem)
        self.all=nn.ModuleList(SCI_backward)
        

    def forward(self, measurement,mask,mask_s):
        #nrow, ncol, ntemp, batch_size,iter_number=args['patchsize'],args['patchsize'],args['temporal_length'],args['batch_size'],args['iter__number']
        v0,gamma20,lamda_20=[],[],[]
        img_out_ori = torch.ones_like(measurement).squeeze(1)
        img_out_ori = einops.repeat(img_out_ori,"b h w->b cr h w",cr=mask.shape[1])
        for prior in range(self.prior):
            if prior==0:
                x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_2=self.all[prior](mask,measurement,img_out_ori)
            else:
                x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_2=self.all[prior](x,v1,lamda_1,lamda_21,gamma1,gamma21,mask,measurement,x1,x2,x3,x4,x5)
            v0.append(v_list[-1]),gamma20.append(gamma2),lamda_20.append(lamda_2)
            v1,gamma21,lamda_21=torch.stack(v0, axis=4),torch.stack(gamma20, axis=2),torch.stack(lamda_20, axis=4)
            x=x_list[-1]        

        # return	x_list,v_list
        return	x_list