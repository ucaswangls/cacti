import numpy as np
import torch.nn as nn
import torch
import einops
from collections import OrderedDict
from .builder import MODELS

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
"""
# --------------------------------------------
# FFDNet (15 or 12 conv layers)
# --------------------------------------------
Reference:
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
"""


# --------------------------------------------
# FFDNet
# --------------------------------------------
@MODELS.register_module
class FFDNetColor(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super().__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = PixelUnShuffle(upscale_factor=sf)

        m_head = conv(in_nc*sf*sf+1, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = conv(nc, out_nc*sf*sf, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        sigma = einops.rearrange(sigma,"b c->b c 1 1")
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        
        x = x[..., :h, :w]
        return x
