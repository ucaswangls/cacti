import torch 
from torch import nn 
from .builder import MODELS 
import torch.nn.functional as F
class self_attention(nn.Module):
    def __init__(self, ch):
        super(self_attention, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch // 8, 1)
        self.conv2 = nn.Conv2d(ch, ch // 8, 1)
        self.conv3 = nn.Conv2d(ch, ch, 1)
        self.conv4 = nn.Conv2d(ch, ch, 1)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.ch = ch

    def forward(self, x):
        batch_size = x.shape[0]

        f = self.conv1(x)
        g = self.conv2(x)
        h = self.conv3(x)
        ht = h.reshape([batch_size, self.ch, -1])

        ft = f.reshape([batch_size, self.ch // 8, -1])
        n = torch.matmul(ft.permute([0, 2, 1]), g.reshape([batch_size, self.ch // 8, -1]))
        beta = F.softmax(n, dim=-1)

        o = torch.matmul(ht, beta)
        o = o.reshape(x.shape)  # [bs, C, h, w]

        o = self.conv4(o)

        x = self.gamma1 * o + x

        return x


class res_part(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_part, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1 + x
        x1 = self.conv2(x)
        x = x1 + x
        x1 = self.conv3(x)
        x = x1 + x
        return x


class down_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 30, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(30, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.Conv2d(20, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class forward_rnn(nn.Module):

    def __init__(self):
        super(forward_rnn, self).__init__()
        self.extract_feature1 = down_feature(1, 20)
        self.up_feature1 = up_feature(50, 1)
        self.conv_x = nn.Sequential(
            nn.Conv2d(2, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 80, 3, stride=2, padding=1),
            nn.Conv2d(80, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 40, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(40, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.h_h = nn.Sequential(
            nn.Conv2d(50, 30, 3, padding=1),
            nn.Conv2d(30, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
        )
        self.res_part1 = res_part(50, 50)
        self.res_part2 = res_part(50, 50)

    def forward(self, xt1, meas, mask, h, meas_re, block_size, cs_rate):
        ht = h
        xt = xt1

        out = xt1
        batch_size = meas.shape[0]
        for i in range(cs_rate - 1):
            d1 = torch.zeros(batch_size, block_size, block_size).to(meas.device)
            d2 = torch.zeros(batch_size, block_size, block_size).to(meas.device)
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask[ii, :, :], out[:, ii, :, :])
            for ii in range(i + 2, cs_rate):
                d2 = d2 + torch.mul(mask[ii, :, :], torch.squeeze(meas_re))
            x1 = self.conv_x(torch.cat([meas_re, torch.unsqueeze(meas - d1 - d2, 1)], dim=1))

            x2 = self.extract_feature1(xt)
            h = torch.cat([ht, x1, x2], dim=1)

            h = self.res_part1(h)
            h = self.res_part2(h)
            ht = self.h_h(h)
            xt = self.up_feature1(h)
            out = torch.cat([out, xt], dim=1)

        return out, ht


class cnn1(nn.Module):

    def __init__(self, in_ch):
        super(cnn1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.conv51 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu51 = nn.LeakyReLU(inplace=True)
        self.conv52 = nn.Conv2d(32, 16, kernel_size=1, stride=1)
        self.relu52 = nn.LeakyReLU(inplace=True)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.res_part1 = res_part(128, 128)
        self.res_part2 = res_part(128, 128)
        self.res_part3 = res_part(128, 128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.LeakyReLU(inplace=True)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=1, stride=1)

        self.att1 = self_attention(128)

    def forward(self, mask, meas_re, block_size, cs_rate):
        batch_size = meas_re.shape[0]
        maskt = mask.expand([batch_size, cs_rate, block_size, block_size])
        maskt = maskt.mul(meas_re)
        xt = torch.cat([meas_re, maskt], dim=1)
        data = xt
        out = self.conv1(data)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.res_part1(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.res_part2(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.conv10(out)
        out = self.res_part3(out)

        out = self.att1(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)
        out = self.relu52(out)
        out = self.conv6(out)

        return out


class backrnn(nn.Module):

    def __init__(self):
        super(backrnn, self).__init__()
        self.extract_feature1 = down_feature(1, 20)
        self.up_feature1 = up_feature(50, 1)
        self.conv_x = nn.Sequential(
            nn.Conv2d(2, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 80, 3, stride=2, padding=1),
            nn.Conv2d(80, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 40, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(40, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.h_h = nn.Sequential(
            nn.Conv2d(50, 30, 3, padding=1),
            nn.Conv2d(30, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
        )
        self.res_part1 = res_part(50, 50)
        self.res_part2 = res_part(50, 50)

    def forward(self, xt8, meas, mask, h, meas_re, block_size, cs_rate):
        ht = h

        xt = xt8[:, cs_rate - 1, :, :]
        xt = torch.unsqueeze(xt, 1)
        batch_size = meas.shape[0]
        out = torch.zeros(batch_size, cs_rate, block_size, block_size).to(meas.device)
        out[:, cs_rate - 1, :, :] = xt[:, 0, :, :]
        for i in range(cs_rate - 1):
            d1 = torch.zeros(batch_size, block_size, block_size).to(meas.device)
            d2 = torch.zeros(batch_size, block_size, block_size).to(meas.device)
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask[cs_rate - 1 - ii, :, :], out[:, cs_rate - 1 - ii, :, :].clone())
            for ii in range(i + 2, cs_rate):
                d2 = d2 + torch.mul(mask[cs_rate - 1 - ii, :, :], torch.squeeze(meas_re))
            x1 = self.conv_x(torch.cat([meas_re, torch.unsqueeze(meas - d1 - d2, 1)], dim=1))

            x2 = self.extract_feature1(xt)
            h = torch.cat([ht, x1, x2], dim=1)

            h = self.res_part1(h)
            h = self.res_part2(h)
            ht = self.h_h(h)
            xt = self.up_feature1(h)

            out[:, cs_rate - 2 - i, :, :] = xt[:, 0, :, :]

        return out

@MODELS.register_module
class BIRNAT(nn.Module):
    def __init__(self,in_ch=8+1):
        super(BIRNAT,self).__init__()
        self.first_frame_net = cnn1(in_ch)
        self.forward_rnn = forward_rnn()
        self.backrnn = backrnn()
    
    def forward(self,meas,mask,mask_s):
        outputs = []
        meas_re = torch.div(meas,mask_s)
        _,compress_rate,block_size,_ = mask.shape
        meas = meas.squeeze(1)
        mask = mask[0]
        h0 = torch.zeros(meas.shape[0], 20, block_size, block_size).to(meas.device)
        xt1 = self.first_frame_net(mask, meas_re, block_size, compress_rate)
        out_pic1, h1 = self.forward_rnn(xt1, meas, mask, h0,
                            meas_re, block_size, compress_rate)
        outputs.append(out_pic1)
        out_pic2 = self.backrnn(out_pic1, meas, mask, h1,
                        meas_re, block_size, compress_rate)
        outputs.append(out_pic2)
        return outputs