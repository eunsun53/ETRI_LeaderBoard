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

        if resnetnum == '18': # output feat num: 512
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34': #output feat num: 512
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50': # output feat num: 2048
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == '152': # output feat num: 2048
            self.resnet = models.resnet152(pretrained=pretrained)


        for param in self.resnet.parameters():
            param.requires_grad = False #freezing 

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        self.encoder = ResExtractor('50') 
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(2048, 512)
        self.bn1 = norm_layer(512)
        self.conv2 = conv3x3(512, 512)
        self.bn2 = norm_layer(512)
        self.conv3 = conv1x1(512, 2048)
        self.bn3 = norm_layer(2048)
        self.relu = nn.ReLU(inplace=True)

        self.daily_linear = nn.Linear(2048, 7)
        self.gender_linear = nn.Linear(2048, 6)
        self.embel_linear = nn.Linear(2048, 3)


    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image']) # size: (7*7)

        ## fine-tuning conv layer-1 ##
        out = self.conv1(feat)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        ## fine-tuning conv layer-2 ##
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        ## fine-tuning conv layer-3 ##
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        ####################### out size : (7*7*2048)

        flatten = self.avg_pool(out).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel


if __name__ == '__main__':
    pass
