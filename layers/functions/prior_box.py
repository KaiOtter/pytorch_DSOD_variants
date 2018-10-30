from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        return output



class PriorBox2(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox2, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.max_rate = cfg['max_rate']
        self.min_rate = cfg['min_rate']
        self.feature_maps = cfg['feature_maps']
        self.wh_ratios = cfg['wh_ratios']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        self.feature_maps.sort()
        self.feature_maps.reverse()
        self.interval = (self.max_rate - self.min_rate)/sum(self.wh_ratios)

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            # for z in range(self.wh_ratios[k]):
            #     s_k = self.min_rate + (z + 1 + sum(self.wh_ratios[:k])) * self.interval
            #     print(s_k)
            for i, j in product(range(f), repeat=2):
                f_k = self.feature_maps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                if k == 0:
                    s_k = self.min_rate
                    mean += [cx, cy, s_k, s_k]
                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                        mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

                for z in range(self.wh_ratios[k]):
                    s_k = self.min_rate + (z + 1 + sum(self.wh_ratios[:k])) * self.interval
                    # s_k /= self.image_size
                    mean += [cx, cy, s_k, s_k]
                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        return output


if __name__ == '__main__':
    cfg_64_16_mod4 = {
        'num_classes': 21,
        'min_dim': 320,
        'max_rate': 0.9,
        'min_rate': 0.2,
        'feature_maps': [40, 20, 10],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3]],
        'wh_ratios': [1, 2, 3],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }
    p = PriorBox2(cfg_64_16_mod4)
    o = p.forward()
    # for k in o:
    #     print(k)