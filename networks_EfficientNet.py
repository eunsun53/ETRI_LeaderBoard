import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b6')


import torch
from torch import nn
from torch.nn import functional as F

class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        self.efficient = model
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(1000, 7)
        self.gender_linear = nn.Linear(1000, 6)
        self.embel_linear = nn.Linear(1000, 3)


    def forward(self, x):
        """ Forward propagation with input 'x' """
        # feat = self.encoder.front(x['image'])
        feat = self.efficient(x['image'])
        #flatten = self.avg_pool(feat).squeeze() efficient 자체가 풀링이 되어 있어서 문제 해결

        out_daily = self.daily_linear(feat)
        out_gender = self.gender_linear(feat)
        out_embel = self.embel_linear(feat)

        return out_daily, out_gender, out_embel


if __name__ == '__main__':
    pass
