from .utils import rev_3d_part
from torch import nn
import torch 

from .builder import MODELS 
class ResnetBlock(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch,ch,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch,ch,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch,ch,3,1,1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch,ch,1,1),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self,x):
        conv1_out = self.conv1(x)
        x = conv1_out+x
        out = self.conv2(x)
        return out

@MODELS.register_module
class MetaSCI(nn.Module):

    def __init__(self,):
        super(MetaSCI, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
        )

        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(ResnetBlock(128))

    def forward(self,meas,mask,mask_s):
        meas_re = torch.div(meas,mask_s)
        mask_meas_re = mask.mul(meas_re)
        data = meas_re+mask_meas_re

        out = self.conv1(data)

        for layer in self.layers:
            out = layer(out)

        out = self.conv2(out)
        return out