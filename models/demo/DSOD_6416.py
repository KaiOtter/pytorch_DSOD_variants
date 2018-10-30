import torch
import torch.nn as nn
import torch.nn.init as init
from layers import PriorBox
from models.demo.DenseNet_6416 import *
from models.demo.layer_blocks import PredictBlock_A
from models.demo.final_idea import *

cfg_64_16 = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'min_dim': 300,
    'feature_maps': [38, 19, 10, 5, 3, 2],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30.0, 60.0, 111., 162., 213., 264.],
    'max_sizes': [60.0, 111., 162., 213., 264., 315.],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

cfg_64_16_1 = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'min_dim': 300,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30.0, 60.0, 111., 162., 213., 264.],
    'max_sizes': [60.0, 111., 162., 213., 264., 315.],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


cfg_320_64_16 = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'min_dim': 320,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 110, 320],
    'min_sizes': [32.0, 64.0, 118.4, 172.8, 227.2, 281.6],
    'max_sizes': [64.0, 118.4, 172.8, 227.2, 281.6, 336.0],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

channel_dict = {
    '6416': [352, 512, 512, 256, 256, 256],
    'FPN': [256, 256, 256, 256, 256, 256],
    'DSSD': [256, 256, 256, 256, 256, 256],
    'Ours': [256, 256, 256, 256, 256, 256],
    'RON': [256, 256, 256, 256, 256, 256],
    'shallow': [480, 256, 256, 256, 256, 256],
    'shallow2': [384, 256, 256, 256, 256, 256],
}


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class DSOD_64_16(nn.Module):
    def __init__(self, num_classes):
        super(DSOD_64_16, self).__init__()
        self.num_classes = num_classes
        self.extractor = DenseNet_64_16()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.cfg = cfg_320_64_16
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()

        in_channels = channel_dict['6416']
        num_anchors = (4, 6, 6, 6, 4, 4)
        for inC, num_anchor in zip(in_channels, num_anchors):
            # self.loc_layers += [nn.Conv2d(inC, num_anchor*4, kernel_size=3, padding=1)]
            # self.cls_layers += [nn.Conv2d(inC, num_anchor* num_classes, kernel_size=3, padding=1)
            #                                   ]
            self.loc_layers += [nn.Sequential(nn.Conv2d(inC,
                                                        num_anchor * 4, kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(num_anchor * 4)
                                              )]
            self.cls_layers += [nn.Sequential(nn.Conv2d(inC,
                                                        num_anchor * num_classes, kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(num_anchor * num_classes)
                                              )]
        self.normalize = nn.ModuleList([L2Norm(chan, 20) for chan in in_channels])

        self.reset_parameters()

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            x = self.normalize[i](x)
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(
                cls_pred.size(0), -1, self.num_classes))

        # loc_preds = torch.cat(loc_preds, 1)
        # cls_preds = torch.cat(cls_preds, 1)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in cls_preds], 1)

        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output

    def reset_parameters(self):
        for name, param in self.extractor.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.xavier_uniform(param.weight.data, gain=nn.init.calculate_gain('relu'))

        for name, param in self.loc_layers.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.normal(param.weight.data, std=0.01)

        for name, param in self.cls_layers.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.normal(param.weight.data, std=0.01)


class DSOD_64_16_1x1(nn.Module):
    def __init__(self, num_classes):
        super(DSOD_64_16_1x1, self).__init__()
        self.num_classes = num_classes
        self.extractor = DenseNet_64_16_DSSD_s_Pred_D()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.cfg = cfg_320_64_16
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()

        # in_channels = (768, 768, 768, 256, 256, 256) #pred C
        in_channels = (256, 256, 256, 256, 256, 256)  # pred D
        num_anchors = (4, 6, 6, 6, 4, 4)
        for inC, num_anchor in zip(in_channels, num_anchors):
            # self.loc_layers += [nn.Conv2d(inC, num_anchor*4, kernel_size=3, padding=1)]
            # self.cls_layers += [nn.Conv2d(inC, num_anchor* num_classes, kernel_size=3, padding=1)
            #                                   ]
            self.loc_layers += [nn.Sequential(nn.Conv2d(inC,
                                                        num_anchor * 4, kernel_size=1, padding=0, bias=False),
                                              nn.BatchNorm2d(num_anchor * 4)
                                              )]
            self.cls_layers += [nn.Sequential(nn.Conv2d(inC,
                                                        num_anchor * num_classes, kernel_size=1, padding=0, bias=False),
                                              nn.BatchNorm2d(num_anchor * num_classes)
                                              )]
        self.normalize = nn.ModuleList([L2Norm(chan, 20) for chan in in_channels])

        self.reset_parameters()

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            x = self.normalize[i](x)
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(
                cls_pred.size(0), -1, self.num_classes))

        # loc_preds = torch.cat(loc_preds, 1)
        # cls_preds = torch.cat(cls_preds, 1)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in cls_preds], 1)

        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output

    def reset_parameters(self):
        for name, param in self.extractor.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.xavier_uniform(param.weight.data, gain=nn.init.calculate_gain('relu'))

        for name, param in self.loc_layers.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.normal(param.weight.data, std=0.01)

        for name, param in self.cls_layers.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.normal(param.weight.data, std=0.01)


class DSOD_64_16_GN(nn.Module):
    def __init__(self, num_classes):
        super(DSOD_64_16_GN, self).__init__()
        self.num_classes = num_classes
        self.extractor = DSSD_s_GN()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.cfg = cfg_320_64_16
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()

        in_channels = channel_dict['DSSD']
        num_anchors = (4, 6, 6, 6, 4, 4)
        for inC, num_anchor in zip(in_channels, num_anchors):
            # self.loc_layers += [nn.Conv2d(inC, num_anchor*4, kernel_size=3, padding=1)]
            # self.cls_layers += [nn.Conv2d(inC, num_anchor* num_classes, kernel_size=3, padding=1)
            #                                   ]
            self.loc_layers += [nn.Sequential(nn.Conv2d(inC,
                                                        num_anchor * 4, kernel_size=3, padding=1, bias=False),
                                              nn.GroupNorm(4, num_anchor * 4)
                                              )]
            self.cls_layers += [nn.Sequential(nn.Conv2d(inC,
                                                        num_anchor * num_classes, kernel_size=3, padding=1, bias=False),
                                              nn.GroupNorm(num_classes, num_anchor * num_classes)
                                              )]
        self.normalize = nn.ModuleList([L2Norm(chan, 20) for chan in in_channels])
        self.reset_parameters()

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            x = self.normalize[i](x)
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(
                cls_pred.size(0), -1, self.num_classes))

        # loc_preds = torch.cat(loc_preds, 1)
        # cls_preds = torch.cat(cls_preds, 1)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in cls_preds], 1)

        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output

    def reset_parameters(self):
        for name, param in self.extractor.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.xavier_uniform(param.weight.data, gain=nn.init.calculate_gain('relu'))

        for name, param in self.loc_layers.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.normal(param.weight.data, std=0.01)

        for name, param in self.cls_layers.named_parameters():
            if hasattr(param, 'weight'):
                nn.init.normal(param.weight.data, std=0.01)


if __name__ == '__main__':
    m = DSOD_64_16(21)
    input = torch.autograd.Variable(torch.randn(1, 3, 320, 320))
    m.eval()
    o = m(input)
    for ii in o:
        print(ii.shape)
    s = 0
    for name, param in m.named_parameters():
        tmp = 1
        for k in param.shape:
            tmp *= k
        s += tmp
    print(s/1000000)
    for name, param in m.loc_layers.named_parameters():
        print(name)
        print(param.shape)