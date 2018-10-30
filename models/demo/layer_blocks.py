import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.autograd import Variable


class DenseLayer(nn.Module):
    def __init__(self, inC, midC=192, growth_rate=48):
        super(DenseLayer, self).__init__()
        self.model_name = 'DenseLayer'
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, midC, 1, bias=False),
            nn.BatchNorm2d(midC),
            nn.ReLU(inplace=True),
            nn.Conv2d(midC, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x):
        y = self.conv(x)
        y = t.cat([x, y], 1)
        return y


class DenseBlock(nn.Module):
    def __init__(self, layer_num, inC, midC=192, growth_rate=48):
        super(DenseBlock, self).__init__()
        self.model_name = 'DenseBlock'
        layers = []
        layers.append(DenseLayer(inC, midC, growth_rate))
        for layer_idx in range(1, layer_num):
            layers.append(DenseLayer(inC + growth_rate * layer_idx, midC, growth_rate))
        self.dense = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense(x)


class TransitionLayer(nn.Module):

    def __init__(self, inC, outC, pool=False):
        super(TransitionLayer, self).__init__()
        self.model_name = 'TransitionLayer'
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, 1, bias=False)
        )
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True) if pool else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        return (x, self.pool(x))


class DenseSupervision1(nn.Module):
    def __init__(self, inC, outC=256):
        super(DenseSupervision1, self).__init__()
        self.model_name = 'DenseSupervision'

        self.right = nn.Sequential(
            # nn.BatchNorm2d(inC),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(inC,outC,1),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, 1, bias=False)
        )

    def forward(self, x1, x2):
        # x1 should be f1
        right = self.right(x1)
        return t.cat([x2, right], 1)


class DenseSupervision(nn.Module):
    def __init__(self, inC, outC=128, ceil=True):
        super(DenseSupervision, self).__init__()
        self.model_name = 'DenseSupervision'
        self.left = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=ceil),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, 1, bias=False)
        )
        self.right = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, 1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, 3, 2, 1*ceil, bias=False)
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return t.cat([left, right], 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


if __name__ == "__main__":
    input = t.autograd.Variable(t.randn(1, 10, 4, 4))
    m = SELayer(10)
    print(input)
    k = m(input)
    print(k)