import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import MODELS

def A_operator(z, Phi):
    y = torch.sum(Phi * z, 1, keepdim=True)
    return y

def At_operator(z, Phi):
    y = z * Phi
    return y

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv3d(num_in_frames, num_in_frames * self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames, bias=True),
            # nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x) + x


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=(1, 2, 2), bias=True),
            # nn.BatchNorm2d(out_ch),
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
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return F.interpolate(self.convblock(x), scale_factor=(1, 2, 2))


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return self.convblock(x)


class BasicBlock(torch.nn.Module):
    def __init__(self, n_channels, layer):
        super(BasicBlock, self).__init__()

        self.eta_step = nn.Parameter(torch.Tensor([0.01]))
        # define head module

        self.n_channels = n_channels

        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128
        self.k = 3

        self.inc = InputCvBlock(num_in_frames=n_channels + 1, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        if layer == 0:
            self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
            self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
            self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
        else:
            self.upc2 = UpBlock(in_ch=self.chs_lyr2 * 2, out_ch=self.chs_lyr1)
            self.upc1 = UpBlock(in_ch=self.chs_lyr1 * 2, out_ch=self.chs_lyr0)
            self.outc = OutputCvBlock(in_ch=self.chs_lyr0 * 2, out_ch=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)
        self.generate_kernel1 = nn.Conv2d(in_channels=1, out_channels=self.chs_lyr2 * self.k * self.k, kernel_size=5, padding=1, stride=4, bias=None)
        self.generate_kernel2 = nn.Conv2d(in_channels=1, out_channels=self.chs_lyr1 * self.k * self.k, kernel_size=3, padding=1, stride=2, bias=None)
        self.generate_kernel3 = nn.Conv2d(in_channels=1, out_channels=self.chs_lyr0 * self.k * self.k, kernel_size=3, padding=1, stride=1, bias=None)



    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, v, y, y1, mask, Phisum, h_pre):

        yb = A_operator(v, mask)
        y1 = y1 + (y - yb)
        # x = v + At_operator((y1-yb)/Phisum, mask)
        x = v + At_operator((y1 - yb) / (Phisum + self.eta_step), mask)  ## torch.Size([1, 8, 1, 128, 128])

        Phisum[Phisum == 0] = 1  # torch.Size([1, 1, 128, 128])
        y_ = y / Phisum  # torch.Size([1, 1, 1, 128, 128])
        y_ = y_.expand_as(v)  # torch.Size([1, 8, 1, 128, 128])

        x_d0, x_d1, x_d2, x_d3, x_d4 = x.shape

        # # Spatial Denoising
        v = x.contiguous().view(x_d0, x_d2, x_d1, x_d3, x_d4)  # torch.Size([1, 1, 8, 128, 128])
        y_ = y_.contiguous().view(x_d0, x_d2, x_d1, x_d3, x_d4)  # torch.Size([1, 1, 8, 128, 128])

        h_curr = []

        x0 = self.inc(torch.cat((v, y_), dim=1))  # torch.Size([1, 32, 8, 128, 128])
        x1 = self.downc0(x0)  # torch.Size([1, 64, 8, 64, 64])
        x2 = self.downc1(x1)  # torch.Size([1, 128, 8, 32, 32])
        h_curr.append(x2)

        y_ = y_[:, :, 0, :, :]
        if len(h_pre) == 0:
            x2 = self.upc2(x2)
            h_curr.append(x2)
            x1 = self.upc1(x1 + x2)  # torch.Size([1, 32, 8, 128, 128])
            h_curr.append(x1)
            x = self.outc(x0 + x1)
        else:
            N, xC, xD, xH, xW = h_pre[0].size()
            kernel = self.relu(self.generate_kernel1(y_)).reshape([N, xC, self.k ** 2, xH, xW])
            similarity = []
            for i in range(xD):
                unfold_hpre = self.unfold(h_pre[0][:, :, i, :, :]).reshape([N, xC, -1, xH, xW])
                sim = self.sigmoid((kernel * unfold_hpre).sum(2))
                similarity.append(sim)
            similarity = torch.stack(similarity, dim=2) #torch.Size([1, 128, 8, 32, 32])
            h_pre[0] = similarity * h_pre[0]

            x2 = self.upc2(torch.cat((x2, h_pre[0]), dim=1))
            h_curr.append(x2)

            N, xC, xD, xH, xW = h_pre[1].size()
            kernel = self.relu(self.generate_kernel2(y_)).reshape([N, xC, self.k ** 2, xH, xW])
            similarity = []
            for i in range(xD):
                unfold_hpre = self.unfold(h_pre[1][:, :, i, :, :]).reshape([N, xC, -1, xH, xW])
                sim = self.sigmoid((kernel * unfold_hpre).sum(2))
                similarity.append(sim)
            similarity = torch.stack(similarity, dim=2)  # torch.Size([1, 128, 8, 32, 32])
            h_pre[1] = similarity * h_pre[1]
            x1 = self.upc1(torch.cat((x1 + x2, h_pre[1]), dim=1))
            h_curr.append(x1)

            N, xC, xD, xH, xW = h_pre[2].size()
            kernel = self.relu(self.generate_kernel3(y_)).reshape([N, xC, self.k ** 2, xH, xW])
            similarity = []
            for i in range(xD):
                unfold_hpre = self.unfold(h_pre[2][:, :, i, :, :]).reshape([N, xC, -1, xH, xW])
                sim = self.sigmoid((kernel * unfold_hpre).sum(2))
                similarity.append(sim)
            similarity = torch.stack(similarity, dim=2)  # torch.Size([1, 128, 8, 32, 32])
            h_pre[2] = similarity * h_pre[2]

            x = self.outc(torch.cat((x0 + x1, h_pre[2]), dim=1))  # torch.Size([1, 1, 8, 128, 128])

        x = v + x

        v = x.contiguous().view(x_d0, x_d1, x_d2, x_d3, x_d4)  # torch.Size([1, 8, 1, 128, 128])

        return v, y1, h_curr

@MODELS.register_module
class HQSNet(torch.nn.Module):
    def __init__(self, layer_num,n_channels):
        super(HQSNet, self).__init__()
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))
        onelayer = []
        self.LayerNo = layer_num

        for i in range(self.LayerNo):
            onelayer.append(BasicBlock(n_channels, i))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, y, mask, mask_s):
        y = y.unsqueeze(2)
        mask = mask.permute(1,0,2,3)
        # batch_size, nframes, ch, w, h = y.shape
        Phisum = torch.sum(mask * mask, 0, keepdim=True)
        # Phisum[Phisum == 0] = 1

        y1 = torch.zeros_like(y)
        hidden_state = []
        v = At_operator(y, mask)  # torch.Size([16, 1, 3, 64, 64])
        # h = [[None] * 4] * 8 * batch_size
        for i in range(self.LayerNo):
            v, y1, hidden_state = self.fcs[i](v, y, y1, mask, Phisum, hidden_state)

        y_pred = v
        y_pred = y_pred.squeeze(2)
        return y_pred
