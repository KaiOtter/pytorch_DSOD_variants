from models.demo.layer_blocks import *
from models.demo.RFBs import *


def stem():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, 1, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True)
    )


class DenseNet_64_16(nn.Module):
    # the output fea_size of dense_sup5 is 1x1
    def __init__(self):
        super(DenseNet_64_16, self).__init__()
        self.model_name = 'DenseNet'
        self.stem = stem()
        self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
        self.trans1 = TransitionLayer(224, 224, pool=True)
        self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
        self.trans2 = TransitionLayer(352, 352, pool=True)
        self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
        self.trans3 = TransitionLayer(480, 480)
        self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
        self.trans4 = TransitionLayer(608, 256)
        self.dense_sup1 = DenseSupervision1(352, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128, ceil=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)
        f2 = self.dense_sup1(f1, x)
        f3 = self.dense_sup2(f2)
        f4 = self.dense_sup3(f3)
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        return f1, f2, f3, f4, f5, f6


class DenseNet_64_16_FPN(nn.Module):
    '''
        Fea_Map Coverage: f1, f2, f3, f4
        1. FPN
    '''
    def __init__(self, upmode='nearest'):
        super(DenseNet_64_16_FPN, self).__init__()
        self.model_name = 'DenseNet'
        self.stem = stem()
        self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
        self.trans1 = TransitionLayer(224, 224, pool=True)
        self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
        self.trans2 = TransitionLayer(352, 352, pool=True)
        self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
        self.trans3 = TransitionLayer(480, 480)
        self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
        self.trans4 = TransitionLayer(608, 256)
        self.dense_sup1 = DenseSupervision1(352, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128, ceil=False)
        self.RBF1 = FPN(512, 256, 256, mode=upmode)
        self.RBF2 = FPN(512, 256, 256, mode=upmode)
        self.RBF3 = FPN(352, 256, 256, mode=upmode)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)  #40x40 352
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)
        f2 = self.dense_sup1(f1, x) #20x20 512
        f3 = self.dense_sup2(f2)  #10x10 512
        f4 = self.dense_sup3(f3)  #5x5 256
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        out3 = self.RBF1(f3, f4)  #10x10 256
        out2 = self.RBF2(f2, out3) #20x20 256
        out1 = self.RBF3(f1, out2) #40x40 256

        return out1, out2, out3, f4, f5, f6


# class DenseNet_64_16_FPN2(nn.Module):
#     '''
#         Fea_Map Coverage: f1, f2, f3, f4
#         1. FPN
#     '''
#     def __init__(self, upmode='nearest'):
#         super(DenseNet_64_16_FPN2, self).__init__()
#         self.model_name = 'DenseNet'
#         self.stem = stem()
#         self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
#         self.trans1 = TransitionLayer(224, 224, pool=True)
#         self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
#         self.trans2 = TransitionLayer(352, 352, pool=True)
#         self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
#         self.trans3 = TransitionLayer(480, 480)
#         self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
#         self.trans4 = TransitionLayer(608, 256)
#         self.dense_sup1 = DenseSupervision1(352, 256)
#         self.dense_sup2 = DenseSupervision(512, 256)
#         self.dense_sup3 = DenseSupervision(512, 128)
#         self.dense_sup4 = DenseSupervision(256, 128)
#         self.dense_sup5 = DenseSupervision(256, 128, ceil=False)
#         self.RBF1 = FPN2_b(512, 256, 256, mode=upmode)
#         self.RBF2 = FPN2_b(512, 256, 256, mode=upmode)
#         self.RBF3 = FPN2_b(352, 256, 256, mode=upmode)
#
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.dense1(x)
#         _, x = self.trans1(x)
#         x = self.dense2(x)
#         f1, x = self.trans2(x)  #40x40 352
#         x = self.dense3(x)
#         _, x = self.trans3(x)
#         x = self.dense4(x)
#         _, x = self.trans4(x)
#         f2 = self.dense_sup1(f1, x) #20x20 512
#         f3 = self.dense_sup2(f2)  #10x10 512
#         f4 = self.dense_sup3(f3)  #5x5 256
#         f5 = self.dense_sup4(f4)
#         f6 = self.dense_sup5(f5)
#         out3 = self.RBF1(f3, f4)  #10x10 256
#         out2 = self.RBF2(f2, out3) #20x20 256
#         out1 = self.RBF3(f1, out2) #40x40 256
#
#         return out1, out2, out3, f4, f5, f6


class DenseNet_64_16_DSSD(nn.Module):
    '''
        Fea_Map Coverage: f1, f2, f3, f4
        2. DSSD, Deconvolution module
    '''
    def __init__(self):
        super(DenseNet_64_16_DSSD, self).__init__()
        self.model_name = 'DenseNet'
        self.stem = stem()
        self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
        self.trans1 = TransitionLayer(224, 224, pool=True)
        self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
        self.trans2 = TransitionLayer(352, 352, pool=True)
        self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
        self.trans3 = TransitionLayer(480, 480)
        self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
        self.trans4 = TransitionLayer(608, 256)
        self.dense_sup1 = DenseSupervision1(352, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128, ceil=False)
        self.RBF1 = Deconv(512, 256, 256)
        self.RBF2 = Deconv(512, 256, 256)
        self.RBF3 = Deconv(352, 256, 256)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)  #40x40 352
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)
        f2 = self.dense_sup1(f1, x) #20x20 512
        f3 = self.dense_sup2(f2)  #10x10 512
        f4 = self.dense_sup3(f3)  #5x5 256
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        out3 = self.RBF1(f3, f4)  #10x10 256
        out2 = self.RBF2(f2, out3) #20x20 256
        out1 = self.RBF3(f1, out2) #40x40 256

        return out1, out2, out3, f4, f5, f6


class DenseNet_64_16_DSSD_s(nn.Module):
    '''
        Fea_Map Coverage: f1, f2, f3, f4
        A lite version of Deconv.
        Remove the redundant 3x3 conv in DSSD block
    '''
    def __init__(self):
        super(DenseNet_64_16_DSSD_s, self).__init__()
        self.model_name = 'DenseNet'
        self.stem = stem()
        self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
        self.trans1 = TransitionLayer(224, 224, pool=True)
        self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
        self.trans2 = TransitionLayer(352, 352, pool=True)
        self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
        self.trans3 = TransitionLayer(480, 480)
        self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
        self.trans4 = TransitionLayer(608, 256)
        self.dense_sup1 = DenseSupervision1(352, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128, ceil=False)
        self.RBF1 = Deconv_s(512, 256, 256)
        self.RBF2 = Deconv_s(512, 256, 256)
        self.RBF3 = Deconv_s(352, 256, 256)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        print(x.shape)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)  #40x40 352
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)
        f2 = self.dense_sup1(f1, x) #20x20 512
        f3 = self.dense_sup2(f2)  #10x10 512
        f4 = self.dense_sup3(f3)  #5x5 256
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        out3 = self.RBF1(f3, f4)  #10x10 256
        out2 = self.RBF2(f2, out3) #20x20 256
        out1 = self.RBF3(f1, out2) #40x40 256

        return out1, out2, out3, f4, f5, f6


class DenseNet_64_16_RON(nn.Module):
    '''
        Fea_Map Coverage: f1, f2, f3, f4
        3. RON, reverse connection block
    '''
    def __init__(self):
        super(DenseNet_64_16_RON, self).__init__()
        self.model_name = 'DenseNet'
        self.stem = stem()
        self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
        self.trans1 = TransitionLayer(224, 224, pool=True)
        self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
        self.trans2 = TransitionLayer(352, 352, pool=True)
        self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
        self.trans3 = TransitionLayer(480, 480)
        self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
        self.trans4 = TransitionLayer(608, 256)
        self.dense_sup1 = DenseSupervision1(352, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128, ceil=False)
        self.RBF1 = RON(512, 256, 256)
        self.RBF2 = RON(512, 256, 256)
        self.RBF3 = RON(352, 256, 256)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)  #40x40 352
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)
        f2 = self.dense_sup1(f1, x) #20x20 512
        f3 = self.dense_sup2(f2)  #10x10 512
        f4 = self.dense_sup3(f3)  #5x5 256
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        out3 = self.RBF1(f3, f4)  #10x10 256
        out2 = self.RBF2(f2, out3) #20x20 256
        out1 = self.RBF3(f1, out2) #40x40 256

        return out1, out2, out3, f4, f5, f6


class DenseNet_64_16_RON2(nn.Module):
    '''
        Fea_Map Coverage: f1, f2, f3, f4
        3. RON, reverse connection block
    '''
    def __init__(self):
        super(DenseNet_64_16_RON2, self).__init__()
        self.model_name = 'DenseNet'
        self.stem = stem()
        self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
        self.trans1 = TransitionLayer(224, 224, pool=True)
        self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
        self.trans2 = TransitionLayer(352, 352, pool=True)
        self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
        self.trans3 = TransitionLayer(480, 480)
        self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
        self.trans4 = TransitionLayer(608, 256)
        self.dense_sup1 = DenseSupervision1(352, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128, ceil=False)
        self.RBF1 = RON2(512, 256, 256)
        self.RBF2 = RON2(512, 256, 256)
        self.RBF3 = RON2(352, 256, 256)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)  #40x40 352
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)
        f2 = self.dense_sup1(f1, x) #20x20 512
        f3 = self.dense_sup2(f2)  #10x10 512
        f4 = self.dense_sup3(f3)  #5x5 256
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        out3 = self.RBF1(f3, f4)  #10x10 256
        out2 = self.RBF2(f2, out3) #20x20 256
        out1 = self.RBF3(f1, out2) #40x40 256

        return out1, out2, out3, f4, f5, f6


class DSSD_s_SE(nn.Module):
    '''
        Fea_Map Coverage: f1, f2, f3, f4
        Deconv_s with SE block
    '''
    def __init__(self):
        super(DSSD_s_SE, self).__init__()
        self.model_name = 'DenseNet'
        self.stem = stem()
        self.dense1 = DenseBlock(6, 128, midC=64, growth_rate=16)
        self.trans1 = TransitionLayer(224, 224, pool=True)
        self.dense2 = DenseBlock(8, 224, midC=64, growth_rate=16)
        self.trans2 = TransitionLayer(352, 352, pool=True)
        self.dense3 = DenseBlock(8, 352, midC=64, growth_rate=16)
        self.trans3 = TransitionLayer(480, 480)
        self.dense4 = DenseBlock(8, 480, midC=64, growth_rate=16)
        self.trans4 = TransitionLayer(608, 256)
        self.dense_sup1 = DenseSupervision1(352, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128, ceil=False)
        self.RBF1 = Deconv_s(512, 256, 256)
        self.RBF2 = Deconv_s(512, 256, 256)
        self.RBF3 = Deconv_s(352, 256, 256)
        self.SEblock1 = SELayer(256, reduction=8)
        self.SEblock2 = SELayer(256, reduction=8)
        self.SEblock3 = SELayer(256, reduction=8)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)  #40x40 352
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)
        f2 = self.dense_sup1(f1, x) #20x20 512
        f3 = self.dense_sup2(f2)  #10x10 512
        f4 = self.dense_sup3(f3)  #5x5 256
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        out3 = self.RBF1(f3, f4)  #10x10 256
        out3 = self.SEblock1(out3)
        out2 = self.RBF2(f2, out3) #20x20 256
        out2 = self.SEblock2(out2)
        out1 = self.RBF3(f1, out2) #40x40 256
        out1 = self.SEblock3(out1)

        return out1, out2, out3, f4, f5, f6


if __name__ == '__main__':
    m = DenseNet_64_16_DSSD_s()
    input = t.autograd.Variable(t.randn(1, 3, 320, 320))
    m.eval()
    o = m(input)
    print("xxxxxxxxxxxxxxxxxxx")
    for ii in o:
        print(ii.shape)

    # input = t.randn(1, 256, 3, 3)
    # maxp = nn.MaxPool2d(2, 2, ceil_mode=False)
    # conv = nn.Conv2d(256, 256, 3, 1*False, 0, bias=False)
    # output = conv(input)
    # print(output.shape)