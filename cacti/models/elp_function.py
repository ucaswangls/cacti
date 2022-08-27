import torch 
from torch import nn 

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 4
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames, out_ch*self.interm_ch, \
					  kernel_size=3, padding=1, groups=1, bias=False),
			#nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)
        
class SCIbackwardresem(nn.Module):
    def __init__(self,in_ch,pres_ch,init_channels, number):
        super().__init__()       
        
        alnet=[]
        self.iter_number=1        
       
        for i in range(self.iter_number):            
            alnet.append(AL_net(in_ch,pres_ch,init_channels,2,number))	
           # up_samples.append(Upsample(BasicBlock,3,3*ntemp,4*ntemp))

        self.AL=nn.ModuleList(alnet)

    def forward(self, x0,v0,lamda_10,lamda_20,gamma10,gamma20,mask,measurement,x10,x20,x30,x40,x50):
        #nrow, ncol, ntemp, batch_size,iter_number=args['patchsize'],args['patchsize'],args['temporal_length'],args['batch_size'],args['iter__number']
       #T_or=torch.ones(batch_size,head_num,L_num)	
       #  yb = A(theta+b); 	v = (theta+b)+At((y-yb)./(mask_sum+gamma)) 
       #  yb = A(v+b); 	x = (v+b)+At((y-yb)./(mask_sum+gamma)), b=λ_2−𝐴^𝑇 λ_1, gamma=γ_2/γ_1        
        mask_sum=torch.sum(mask,1,keepdim=True)
        mask_sum[mask_sum==0]=1
        measurement_mean=measurement/mask_sum
        x_list,v_list = [],[]
        batch, C, H, W = mask.shape
        #lamda_1=torch.zeros_like(measurement)
        v,x=v0.clone(),x0.clone()      
        lamda_1,lamda_2=lamda_10.clone(),lamda_20.clone()      
        gamma1,gamma2=gamma10.clone(),gamma20.clone()      
        x1,x2,x3,x4,x5=x10.clone(),x20.clone(),x30.clone(),x40.clone(),x50.clone()      

        #lamda_2.append(lamda_2[-1])                
        #xv.append(xv[-1])
       
        for iter_ in range(8,8+self.iter_number):
            v_,x,x1,x2,x3,x4,x5,gamma20,lamda_1,lamda_21=self.AL[iter_-8](v,x,lamda_1,lamda_2,gamma1,gamma2,mask,mask_sum,measurement,measurement_mean,iter_,x1,x2,x3,x4,x5)
            x_list.append(x),v_list.append(v_)
            #lamda_2[:,:,:,:,-1]=lamda_21

        gamma2=gamma20.unsqueeze(0).repeat(batch,1)
        return	x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_21

class AL_net(nn.Module):
    def __init__(self,in_ch,pres_ch,init_channels,number1,number, dropout = 0.1):
        super(AL_net,self).__init__()
        
        self.pres_ch = pres_ch
        self.prior=number
             
        self.gamma1 = torch.nn.Parameter(torch.Tensor([1]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([1]))
        
        self.visualtransformer=Visual_Transformer(C_dim=init_channels,in_ch=in_ch,initial_judge=number1)
                     

    def forward(self,v0,x, lamda_1,lamda_20,gamma1,gamma20,mask,mask_sum,measurement,measurement_mean,iter_,x1=None,x2=None,x3=None,x4=None,x5=None):
        
        lamda_2=lamda_20[:,:,:,:,-1]
        img=x-lamda_2/self.gamma2
        batch, C, H, W = mask.shape
        noise_map=self.gamma2.expand((batch, 1, H, W))
        x0=torch.cat((noise_map,measurement_mean),1)
        v,x1,x2,x3,x4,x5=self.visualtransformer(iter_,img,x0,self.pres_ch,x1,x2,x3,x4,x5)

        lamda_1=lamda_1-gamma1[0,0]*(measurement-torch.sum(mask*x,1,keepdim=True))
        lamda_2=lamda_2-self.gamma2*(x-v)        
        gamma0=torch.zeros_like(self.gamma2)
        xb=-mask*lamda_1
        for p in range(self.prior-1):
            gamma0+=gamma20[0,:,p]
            xb=xb+lamda_20[:,:,:,:,p]+gamma20[0,:,p]*v0[:,:,:,:,p]
        gamma0+=self.gamma2
        xb=xb+lamda_2+self.gamma2*v
        gamma=gamma0/gamma1[0,0]
        x_b=xb/gamma0
        x=x_b+mask*(torch.div((measurement-torch.sum(mask*x_b,1,keepdim=True)),(mask_sum+gamma)))        

        return	v,x,x1,x2,x3,x4,x5,self.gamma2,lamda_1,lamda_2

class Visual_Transformer(nn.Module):
    def __init__(self, C_dim=64,in_ch=16,initial_judge=2): #dim = 128, num_tokens = 8
        super(Visual_Transformer, self).__init__()                
        
        C_dima=C_dim//2
        C_dimb=C_dima//2       
        self.number=in_ch
       
        self.inc = InputCvBlock(num_in_frames=self.number+2, out_ch=C_dimb)
        self.downc0 = DownBlock(in_ch=C_dimb*initial_judge, out_ch=C_dima)
        self.downc1 = DownBlock(in_ch=C_dima*initial_judge, out_ch=C_dim)
       # self.dence = OutputCvBlock(in_ch=C_dim, out_ch=C_dim)
        self.upc2 = UpBlock(in_ch=C_dim*initial_judge, out_ch=C_dima)
        self.upc1 = UpBlock(in_ch=C_dima*initial_judge, out_ch=C_dimb)
        self.outc = OutputCvBlock(in_ch=C_dimb*initial_judge, out_ch=self.number)       
        #self.transformer3a=Transformer(hw_number//16, attention_dim//4, nrow//4, mlp_dim//4, C_dim, depth,  dropout)
        #self.transformer3b=Transformer(hw_number//16, attention_dim//4, nrow//4, mlp_dim//4, C_dim, depth,  dropout)  
        
    def forward(self, iter_,img, x0,num,last1=None, last2 = None,last3=None, last4 = None,last5 = None): #100,3,32,32
        C=img.shape[1]
        count=num//C
        remain=num%C
        img1=img.repeat(1,count,1,1)
        img2=img[:,0:remain,:,:]
        img3=torch.cat((img1,img2,x0),1)
        x1=self.inc(img3)        
        if iter_==0:
            x2=self.downc0(x1)
            x3=self.downc1(x2)
            x4=self.upc2(x3)
            x5=self.upc1(x4+x2)
            x6=self.outc(x5+x1)
        else:
            x2=self.downc0(torch.cat((x1,last1),1))

            x3=self.downc1(torch.cat((x2,last2),1))
                       
            x4=self.upc2(torch.cat((x3,last3),1))

            x5=self.upc1(torch.cat((x4+x2,last4),1))

            x6=self.outc(torch.cat((x5+x1,last5),1))
            x1,x2,x3,x4,x5=x1+last1,x2+last2,x3+last3,x4+last4,x5+last5

        for i in range(C):
            z=torch.sum(x6[:,i::C,:,:],1,keepdim=False)  #sum(x6[:,0::C,:,:],)
            if i<remain:
                z1=(x6[:,count*C+i,:,:]+z)/(count+1)
            else:
                z1=z/count
            z1=z1.unsqueeze(1)
            if i==0:
                z2=z1
            else:
                z2=torch.cat((z2,z1),1)
        x=z2+img       
        return  x,x1,x2,x3,x4,x5