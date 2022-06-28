import torch
import torch.nn as nn

def split_feature(x):
    # x1,x2 = torch.chunk(x,chunks=2,dim=1)
    b,c,d,h,w = x.shape
    x1, x2 = x[:,:c//2],x[:,c//2:]
    return x1, x2

class rev_3d_part(nn.Module):

    def __init__(self, in_ch):
        super(rev_3d_part, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )
        self.g1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1, x2 = split_feature(x)
        y1 = x1 + self.f1(x2)
        y2 = x2 + self.g1(y1)
        y = torch.cat([y1, y2], dim=1)
        return y

    def reverse(self, y):
        y1, y2 = split_feature(y)
        x2 = y2 - self.g1(y1)
        x1 = y1 - self.f1(x2)
        x = torch.cat([x1, x2], dim=1)
        return x