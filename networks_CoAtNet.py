'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch import nn, sqrt
import torch
import sys
from math import sqrt

sys.path.append('.')
from model.conv.MBConv import MBConvBlock
from model.attention.SelfAttention import ScaledDotProductAttention

class CoAtNet(nn.Module):
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768]):
        super().__init__()
        self.out_chs=out_chs
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.s0=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1)
        )
        self.mlp0=nn.Sequential(
            nn.Conv2d(in_ch,out_chs[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[0],out_chs[0],kernel_size=1)
        )
        
        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)
        self.mlp1=nn.Sequential(
            nn.Conv2d(out_chs[0],out_chs[1],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[1],out_chs[1],kernel_size=1)
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)
        self.mlp2=nn.Sequential(
            nn.Conv2d(out_chs[1],out_chs[2],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[2],out_chs[2],kernel_size=1)
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//8,out_chs[2]//8,8)
        self.mlp3=nn.Sequential(
            nn.Linear(out_chs[2],out_chs[3]),
            nn.ReLU(),
            nn.Linear(out_chs[3],out_chs[3])
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//8,out_chs[3]//8,8)
        self.mlp4=nn.Sequential(
            nn.Linear(out_chs[3],out_chs[4]),
            nn.ReLU(),
            nn.Linear(out_chs[4],out_chs[4])
        )


    def forward(self, x) :
        B,C,H,W=x.shape
        #stage0
        y=self.mlp0(self.s0(x))
        y=self.maxpool2d(y)
        #stage1
        y=self.mlp1(self.s1(y))
        y=self.maxpool2d(y)
        #stage2
        y=self.mlp2(self.s2(y))
        y=self.maxpool2d(y)
        #stage3
        y=y.reshape(B,self.out_chs[2],-1).permute(0,2,1) #B,N,C
        y=self.mlp3(self.s3(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1)).permute(0,2,1)
        #stage4
        y=self.mlp4(self.s4(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1))
        N=y.shape[-1]
        y=y.reshape(B,self.out_chs[4],int(sqrt(N)),int(sqrt(N)))

        return y

###install to use coca### 
# !pip install coca-pytorch
# !pip install vit-pytorch>=0.35.8



class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        #self.encoder = ResExtractor('18')
        # self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(768, 7)
        self.gender_linear = nn.Linear(768, 6)
        self.embel_linear = nn.Linear(768, 3)

        self.coatnet = CoAtNet(3, 224)
        
    def forward(self, x):
        """ Forward propagation with input 'x' """
        #feat = self.encoder.front(x['image']) ##ResNet
       
        feat = self.coatnet(x['image']) #(b, 768, n, n)
        self.avg_pool = nn.AvgPool2d(kernel_size=feat.shape[-1]) #(b, 768, 1, 1)
        flatten = self.avg_pool(feat).squeeze() #(b, 768, 1, 1)

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel


if __name__ == '__main__':
    pass
