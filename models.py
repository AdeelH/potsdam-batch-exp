import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.utils
import torchvision as tv
from torchvision import transforms as tf

from fastai.vision import *
from fastai.metrics import error_rate



######################################
# utils
######################################
def freeze(m, recurse=True):
    for p in m.parameters(recurse=recurse):
        p.requires_grad = False
        p.grad = None

def unfreeze(m, recurse=True):
    for p in m.parameters(recurse=recurse):
        p.requires_grad = True

    
def attach_forward_hooks(m, callback, depth=0, max_depth=3, recurse_whitelist=[nn.Sequential, nn.ModuleList]):
    if depth > max_depth:
        return []

    hs = []
    for c in m.children():
        if isinstance(c, recurse_whitelist):
            hs += attach_forward_hooks(c, callback, depth=depth+1, max_depth=max_depth)
        else:
            hs.append(c.register_forward_hook(callback))
    return hs

def attach_backward_hooks(m, callback, depth=0, max_depth=3, recurse_whitelist=[nn.Sequential, nn.ModuleList]):
    if depth > max_depth:
        return []

    hs = []
    for c in m.children():
        if isinstance(c, recurse_whitelist):
            hs += attach_backward_hooks(c, callback, depth=depth+1, max_depth=max_depth)
        else:
            hs.append(c.register_forward_hook(callback))
    return hs

######################################
# Resnet
######################################
def get_resnet18(pretrained=True):
    return tv.models.resnet18(pretrained=pretrained)

def get_resnet18_custom(in_channels=3, pretrained=True):
    m = get_resnet18(pretrained=True)
    m._modules['conv1'] = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return m


######################################
# UNet
######################################
def get_resnet18(pretrained=True):
    return tv.models.resnet18(pretrained=pretrained)

def get_unet_custom(encoder, nclasses, last_cross=False):
    base_model_fn = lambda _: encoder
    return models.unet.DynamicUnet(create_body(base_model_fn), n_classes=nclasses, last_cross=last_cross).cuda()


######################################
# Deeplab
######################################
def get_deeplab(num_classes, pretrained=True):
    return tv.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True, num_classes=num_classes)

class DeepLabWrapper(nn.Module):
    def __init__(self, m, in_channels=3):
        super(DeepLabWrapper, self).__init__()
        m.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.m = m
    
    def forward(self, X):
        return self.m(X)['out']

def get_deeplab_custom(in_channels=3, pretrained=False):
    return DeepLabWrapper(get_deeplab(pretrained=pretrained), in_channels=in_channels).cuda()


