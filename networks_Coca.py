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
from setuptools import setup, find_packages

setup(
  name = 'CoCa-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.6',
  license='MIT',
  description = 'CoCa, Contrastive Captioners are Image-Text Foundation Models - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/CoCa-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'contrastive learning',
    'multimodal'
  ],
  install_requires=[
    'einops>=0.4',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

###install to use coca### 
# !pip install coca-pytorch
# !pip install vit-pytorch>=0.35.8

# import vision transformer

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor
from coca_pytorch.coca_pytorch import CoCa


vit = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

vit = Extractor(vit, return_embeddings_only = True, detach = False)

class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(512, 7)
        self.gender_linear = nn.Linear(512, 6)
        self.embel_linear = nn.Linear(512, 3)

        self.coca = CoCa(
                        dim = 512,                     # model dimension
                        img_encoder = vit,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
                        image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
                        num_tokens = 20000,            # number of text tokens
                        unimodal_depth = 6,            # depth of the unimodal transformer
                        multimodal_depth = 6,          # depth of the multimodal transformer
                        dim_head = 64,                 # dimension per attention head
                        heads = 8,                     # number of attention heads
                        caption_loss_weight = 1.,      # weight on the autoregressive caption loss
                        contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
        ).cuda()

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.coca(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel


if __name__ == '__main__':
    pass
