from torch import nn
import torch
import sys
from math import sqrt
sys.path.append('.')

# !pip install einops 필요

from coatnet import CoAtNet, coatnet_0

num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D
block_types=['C', 'T', 'T', 'T']        # 'C' for MBConv, 'T' for Transformer

coatnet = coatnet_0()

class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        # self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(1000, 7)
        self.gender_linear = nn.Linear(1000, 6)
        self.embel_linear = nn.Linear(1000, 3)

        self.coatnet = CoAtNet((224, 224), 3, num_blocks, channels, block_types=block_types)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.coatnet(x['image']) # output 노드 1000
        # flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(feat)
        out_gender = self.gender_linear(feat)
        out_embel = self.embel_linear(feat)

        return out_daily, out_gender, out_embel

