import torch
import torch.nn as nn
from cacti.utils.utils import A, At
from .builder import MODELS

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

class Unet(nn.Module):

    def __init__(self,in_ch, out_ch):
        super(Unet, self).__init__()
                
        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)       

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()
        
        
    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)

        
        x = self.upsample2(conv3)        
        x = torch.cat([x, conv2], dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample1(x)        
        x = torch.cat([x, conv1], dim=1)       

        x = self.dconv_up1(x)  
        
        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs
        
        return out
@MODELS.register_module
class GAPNet(nn.Module):
    def __init__(self,in_ch=8):
        super(GAPNet, self).__init__()
        self.unet1 = Unet(in_ch,in_ch)
        self.unet2 = Unet(in_ch,in_ch)
        self.unet3 = Unet(in_ch,in_ch)
        self.unet4 = Unet(in_ch,in_ch)
        self.unet5 = Unet(in_ch,in_ch)
        self.unet6 = Unet(in_ch,in_ch)
        self.unet7 = Unet(in_ch,in_ch)
        self.unet8 = Unet(in_ch,in_ch)
        self.unet9 = Unet(in_ch,in_ch)
        self.unet10 = Unet(in_ch,in_ch)
        self.unet11 = Unet(in_ch,in_ch)
        self.unet12 = Unet(in_ch,in_ch)
        
    def forward(self, y, Phi, Phi_s):
        x_list = []
        x = At(y,Phi)
        ### 1-3
        yb = A(x,Phi)
        
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet1(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet2(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet3(x)
        x_list.append(x)
        ### 4-6
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet4(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet5(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet6(x)
        x_list.append(x)
        # 7-9
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet7(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet8(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet9(x)
        x_list.append(x)

        # 10-12
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet10(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet11(x)
        x_list.append(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = self.unet12(x)
        x_list.append(x)

        output_list = x_list[-3:]
        return output_list