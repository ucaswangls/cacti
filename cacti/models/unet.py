import torch.nn as nn
from cacti.utils.utils import At
import torch 
from .builder import MODELS
class ResNetBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ResNetBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x+out

@MODELS.register_module
class Unet(nn.Module):

    def __init__(self,in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv2d(in_ch,out_ch,3,1,1)
        self.encode1 = ResNetBlock(out_ch,out_ch) 
        self.encode2 = ResNetBlock(out_ch,out_ch) 
        self.encode3 = ResNetBlock(out_ch,out_ch) 
        self.encode4 = ResNetBlock(out_ch,out_ch) 
        self.encode5 = ResNetBlock(out_ch,out_ch) 

        self.latent = nn.Sequential(
            nn.Conv2d(out_ch,out_ch,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,1,1),
            nn.LeakyReLU(inplace=True),
        )

        self.decode1 = ResNetBlock(out_ch,out_ch) 
        self.decode2 = ResNetBlock(out_ch,out_ch) 
        self.decode3 = ResNetBlock(out_ch,out_ch) 
        self.decode4 = ResNetBlock(out_ch,out_ch) 
        self.decode5 = ResNetBlock(out_ch,out_ch) 
        
        self.last_act = nn.Sequential(
            nn.Conv2d(out_ch,in_ch,1,1),
            nn.Tanh()
        )
        
    def forward(self,y,Phi,Phi_s):
        x = At(torch.div(y,Phi_s),Phi)

        conv1_out = self.conv1(x)
        en_out1 = self.encode1(conv1_out)
        en_out2 = self.encode2(en_out1)
        en_out3 = self.encode3(en_out2)
        en_out4 = self.encode4(en_out3)
        en_out5 = self.encode5(en_out4)

        latent_out = self.latent(en_out5)

        dec_out5 = self.decode5(latent_out+en_out5)
        dec_out4 = self.decode4(dec_out5+en_out4)
        dec_out3 = self.decode3(dec_out4+en_out3)
        dec_out2 = self.decode2(dec_out3+en_out2)
        dec_out1 = self.decode1(dec_out2+en_out1)

        out = self.last_act(dec_out1)

        out = out + x 
        out_list = []
        out_list.append(out)
        return out_list
