from torch import nn 
import torch
from cacti.utils.utils import A, At
from .builder import MODELS

try: 
    from .cotnet import CotLayer 
except:
    print("Please install cupy! (website: https://cupy.dev/)")

class ChannelAttention(nn.Module):
    def __init__(self,in_channels,reduction=16):
        super(ChannelAttention,self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels//reduction,1,bias=False),
            nn.Conv2d(in_channels//reduction,in_channels,1,bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        return input*self.attention(input)

class BasciConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2):
        super(BasciConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride,1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.channel_att = ChannelAttention(out_channels)
        
    def forward(self,input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.channel_att(out)
        out = self.relu(out)
        return out

class UpConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpConv,self).__init__()
        self.pixelshuffle= nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self,input):
        out = self.pixelshuffle(input)
        out = self.relu(out)
        out = self.conv(out)
        out = self.relu(out)
        return out

class CotBlock(nn.Module):
    def __init__(self,ratio,channels):
        super(CotBlock,self).__init__()
        
        self.down1 = BasciConv(ratio,channels[0],stride=1)
        self.cot1 = nn.Sequential(
            nn.Conv2d(ratio,channels[0],3,1,1,bias=False),
            CotLayer(channels[0],3),
            nn.Conv2d(channels[0],channels[0],1,1,bias=False)
        )
        self.down2 = BasciConv(channels[0]*2,channels[1])
        self.cot2 = nn.Sequential(
            nn.Conv2d(channels[0]*2,channels[1],3,2,1,bias=False),
            CotLayer(channels[1],3),
            nn.Conv2d(channels[1],channels[1],1,1,bias=False)
        )
        self.down3 = BasciConv(channels[1]*2,channels[2])
        self.cot3 = nn.Sequential(
            nn.Conv2d(channels[1]*2,channels[2],3,2,1,bias=False),
            CotLayer(channels[2],3),
            nn.Conv2d(channels[2],channels[2],1,1,bias=False)
        )
        self.conv1x1_list = nn.ModuleList(
            nn.Conv2d(channels[i]*2,channels[i],1,1) for i in range(len(channels)-1)
        )
        self.conv3x3_list = nn.ModuleList(
            nn.Conv2d(channels[i]*2,channels[i],3,1,1) for i in range(len(channels)-1)
        )
        self.up2 = UpConv(channels[2]*2//4,channels[1])
        self.up1 = UpConv(channels[1]//4,channels[0])
        self.conv = nn.Conv2d(channels[0],ratio,3,1,1)
        self.act = nn.Tanh()
    
    def forward(self,input):
        down1 = self.down1(input)
        cot1 = self.cot1(input)
        down1 = torch.cat([down1,cot1],dim=1)
        down2 = self.down2(down1)
        cot2 = self.cot2(down1)
        down2 = torch.cat([down2,cot2],dim=1)
        down3 = self.down3(down2)
        cot3 = self.cot3(down2)
        down3 = torch.cat([down3,cot3],dim=1)
        up2 = self.up2(down3)
        down2_conv = self.conv1x1_list[1](down2)
        up2_conv = self.conv3x3_list[1](torch.cat([up2,down2_conv],dim=1))
        up1 = self.up1(up2_conv)
        down1_conv = self.conv1x1_list[0](down1)
        up1_conv = self.conv3x3_list[0](torch.cat([up1,down1_conv],dim=1))
        out = self.conv(up1_conv)
        out = self.act(out)
        out = input+out
        return out
        
@MODELS.register_module
class GAP_CCoT(nn.Module):
    def __init__(self,cr=8,stage_num=12):
        super().__init__()
        channels_list=[32,64,128]
        self.unet_list = nn.ModuleList()
        for i in range(stage_num):
            self.unet_list.append(CotBlock(cr,channels_list))
        
    def forward(self, y, Phi, Phi_s):
        x_list = []
        x = At(y,Phi)
        for unet in self.unet_list:
            yb = A(x,Phi)
            x = x + At(torch.div(y-yb,Phi_s),Phi)
            x = unet(x)
            x_list.append(x)
        return x_list