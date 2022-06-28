# from my_tools import *
from torch import nn 
import torch
from cacti.utils.utils import A, At
from .utils import rev_3d_part

from .builder import MODELS

class Re3dcnn_mask(nn.Module):

    def __init__(self, units):
        super(Re3dcnn_mask, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            # nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            # nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )
        self.fuse_r = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1)
        )

        self.layers = nn.ModuleList()
        for i in range(units):
            self.layers.append(rev_3d_part(128))

        # self.att1 = self_attention(128)

    def forward(self, x):

        out = self.conv1(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv2(out)

        return out

@MODELS.register_module
class GAP_net(nn.Module):

    def __init__(self,num_block):
        super(GAP_net, self).__init__()
        self.units = num_block
        self.cnn3d_net1 = Re3dcnn_mask(self.units)
        
        self.cnn3d_net2 = Re3dcnn_mask(self.units)
      

    def forward(self,y, Phi,Phi_s):
        x_list = []
        out = At(y, Phi)

        E_y = torch.div(y, Phi_s)
        # E_y = torch.unsqueeze(E_y, dim=1)
        data = E_y.mul(Phi)

        yb = A(out, Phi)
        out = out + At(torch.div(y - yb, Phi_s), Phi)
        out = torch.unsqueeze(out, 1)
        out = torch.cat([out, torch.unsqueeze(data, 1)], dim=1)
        out = self.cnn3d_net1(out)
        out = torch.squeeze(out, 1)
        x_list.append(out)

        yb = A(out, Phi)
        out = out + At(torch.div(y - yb, Phi_s), Phi)
        out = torch.unsqueeze(out, 1)
        out = torch.cat([out, torch.unsqueeze(data, dim=1)], dim=1)
        out = self.cnn3d_net2(out)
        out = torch.squeeze(out, 1)
        x_list.append(out)

        return x_list
