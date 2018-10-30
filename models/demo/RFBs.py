import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.autograd import Variable



class FPN(nn.Module):
    def __init__(self, lat_inC, top_inC, outC, mode='nearest'):
        super(FPN, self).__init__()
        assert mode in ['nearest', 'bilinear']
        self.latlayer = nn.Conv2d(lat_inC, outC, 1, 1, padding=0)
        self.toplayer = nn.Conv2d(top_inC, outC, 1, 1, padding=0)
        self.up_mode = mode
        # self.bottom_smooth = nn.Sequential(
        #     nn.Conv2d(outC, outC, 3, 1, padding=1),
        #     nn.BatchNorm2d(outC),
        #     nn.ReLU(),
        # )
        self.bottom_smooth = nn.Conv2d(outC, outC, 3, 1, padding=1)

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        up_add = F.upsample(y, scale_factor=2, mode=self.up_mode) + x
        out = self.bottom_smooth(up_add)
        return out


class FPN2(nn.Module):
    def __init__(self, lat_inC, top_inC, outC, mode='nearest'):
        super(FPN2, self).__init__()
        assert mode in ['nearest', 'bilinear']
        self.latlayer = nn.Sequential(
            nn.Conv2d(lat_inC, outC, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )
        self.toplayer =nn.Sequential(
            nn.Conv2d(top_inC, outC, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )
        self.up_mode = mode
        # remove bias was tried in FPN2_b. It was worse.
        self.bottom_smooth = nn.Conv2d(outC, outC, 3, 1, padding=1)

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        up_add = F.upsample(y, scale_factor=2, mode=self.up_mode) + x
        out = self.bottom_smooth(up_add)
        return out


class Deconv(nn.Module):
    def __init__(self, lat_inC, top_inC, outC):
        super(Deconv, self).__init__()
        self.latlayer = nn.Sequential(
            nn.Conv2d(lat_inC, outC, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(outC),
        )
        self.toplayer = nn.Sequential(
            nn.ConvTranspose2d(top_inC, outC, 2, 2),
            nn.Conv2d(outC, outC, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(outC),
        )

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        out = F.relu(x*y)
        return out


class Deconv_s(nn.Module):
    def __init__(self, lat_inC, top_inC, outC):
        super(Deconv_s, self).__init__()
        self.latlayer = nn.Sequential(
            nn.Conv2d(lat_inC, outC, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(outC, outC, 3, 1, padding=1, bias=False),
            # nn.BatchNorm2d(outC),
        )
        self.toplayer = nn.Sequential(
            nn.ConvTranspose2d(top_inC, outC, 2, 2),
            # nn.Conv2d(outC, outC, 3, 1, padding=1, bias=False),
            # nn.BatchNorm2d(outC),
        )

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        out = F.relu(x*y)
        return out


class RON(nn.Module):
    def __init__(self, lat_inC, top_inC, outC):
        super(RON, self).__init__()
        self.latlayer = nn.Conv2d(lat_inC, outC, 3, 1, padding=1)
        self.toplayer = nn.ConvTranspose2d(top_inC, outC, 2, 2)

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        return x + y


class RON2(nn.Module):
    def __init__(self, lat_inC, top_inC, outC):
        super(RON2, self).__init__()
        self.latlayer = nn.Sequential(
            nn.Conv2d(lat_inC, outC, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

        self.toplayer = nn.Sequential(
            nn.ConvTranspose2d(top_inC, outC, 2, 2),
            # nn.BatchNorm2d(outC),
            # nn.ReLU()
        )

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        return x + y