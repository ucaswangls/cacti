from .utils import rev_3d_part
from torch import nn
import torch 

from .builder import MODELS 

@MODELS.register_module
class re_3dcnn1(nn.Module):

    def __init__(self, num_block):
        super(re_3dcnn1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
        )

        self.layers = nn.ModuleList()
        for i in range(num_block):
            self.layers.append(rev_3d_part(32))

    def forward(self,meas,mask,mask_s):
        meas_re = torch.div(meas,mask_s)
        mask_meas_re = mask.mul(meas_re)
        data = meas_re+mask_meas_re

        out = self.conv1(torch.unsqueeze(data, 1))

        for layer in self.layers:
            out = layer(out)

        out = self.conv2(out)
        out = out.squeeze(1)
        return out

    def for_backward(self, mask, meas_re, gt, loss, opt, args):
        batch_size = meas_re.shape[0]
        maskt = mask.expand([batch_size, args.B, args.size[0], args.size[1]])
        maskt = maskt.mul(meas_re)
        data = meas_re + maskt
        data = torch.unsqueeze(data, 1)

        with torch.no_grad():
            out1 = self.conv1(data)
            out2 = out1
            for layer in self.layers:
                out2 = layer(out2)
        out3 = out2.requires_grad_()
        out4 = self.conv2(out3)

        loss1 = loss(torch.squeeze(out4), gt)
        loss1.backward()
        current_state_grad = out3.grad

        out_current = out3
        for layer in reversed(self.layers):
            with torch.no_grad():
                out_pre = layer.reverse(out_current)
            out_pre.requires_grad_()
            out_cur = layer(out_pre)
            torch.autograd.backward(out_cur, grad_tensors=current_state_grad)
            current_state_grad = out_pre.grad
            out_current = out_pre

        out1 = self.conv1(data)
        out1.requires_grad_()
        torch.autograd.backward(out1, grad_tensors=current_state_grad)
        if opt != 0:
            opt.step()

        return out4, loss1