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

import torchvision

class ResExtractor(nn.Module):
    """Feature extractor based on ResNet structure
        Selectable from resnet18 to resnet152

    Args:
        resnetnum: Desired resnet version
                    (choices=['18','34','50','101','152'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, resnetnum='50', pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == '18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == '152':
            self.resnet = models.resnet152(pretrained=pretrained)

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)


    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)

class ResNext_Extractor(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNext_Extractor, self).__init__()

        self.resnext50 = models.resnext50_32x4d(pretrained=pretrained)

        self.modules_front = list(self.resnext50.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x) #output size (7, 7)



class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        #self.encoder = ResExtractor('18')
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.resnext50 = ResNext_Extractor()
        for param in self.resnext50.parameters(): #freeze model
            param.requires_grad = False 

        self.daily_linear = nn.Linear(2048, 7)
        self.gender_linear = nn.Linear(2048, 6)
        self.embel_linear = nn.Linear(2048, 3)

        #self.google =models.googlenet(pretrained=True)
        #self.resnext = models.resnext101_64x4d(pretrained = True)
        

    def forward(self, x):
        """ Forward propagation with input 'x' """
        #feat = self.encoder.front(x['image'])
        #effi = self.efficient_net(x['image'])
        #flatten = self.avg_pool(effi).squeeze() #coatnet, efficientnet 모두 fc 가 마지막에 포함되어있어 squeeze 안해줘도됨 
        feat = self.resnext50.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()
        
        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel



if __name__ == '__main__':
    pass
