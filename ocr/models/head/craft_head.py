import math
import torch
import torch.nn as nn
from collections import OrderedDict
from .db_postprocess import DBPostProcessor


class CRAFTHead(nn.Module):
    def __init__(self, in_channels=256, upscale=4, bias=False, smooth=False, postprocess=None):
        super(CRAFTHead, self).__init__()
        assert postprocess is not None

        self.postprocess = DBPostProcessor(**postprocess)
        inner_channels = in_channels // 4
        upscale_count = int(math.log2(upscale))

        layers = [
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        ]
        for i in range(upscale_count):
            if i == upscale_count - 1:
                layers.append(nn.ConvTranspose2d(inner_channels, 1, 2, 2))
            else:
                layers.extend([
                    nn.ConvTranspose2d(inner_channels, inner_channels, 2, 2),
                    nn.BatchNorm2d(inner_channels),
                    nn.ReLU(inplace=True),
                ])
        layers.append(nn.Sigmoid())
        self.region_head = nn.Sequential(*layers)
        self.region_head.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features, return_loss=True):
        fuse = torch.cat(features, dim=1)
        region_map = self.region_head(fuse)
        return OrderedDict(prob_maps=region_map)

    def get_polygons_from_maps(self, gt, pred):
        return self.postprocess.represent(gt, pred)
