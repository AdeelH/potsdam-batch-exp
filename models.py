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

from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy

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

	
def attach_forward_hooks(m, callback, depth=0, max_depth=3, recurse_whitelist=(nn.Sequential, nn.ModuleList)):
	if depth >= max_depth:
		return []

	hs = []
	for c in m.children():
		if isinstance(c, recurse_whitelist):
			hs += attach_forward_hooks(c, callback, depth=depth+1, max_depth=max_depth)
		else:
			hs.append(c.register_forward_hook(callback))
	return hs

def attach_backward_hooks(m, callback, depth=0, max_depth=3, recurse_whitelist=(nn.Sequential, nn.ModuleList)):
	if depth > max_depth:
		return []

	hs = []
	for c in m.children():
		if isinstance(c, recurse_whitelist):
			hs += attach_backward_hooks(c, callback, depth=depth+1, max_depth=max_depth)
		else:
			hs.append(c.register_backward_hook(callback))
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
def get_deeplab(nclasses, pretrained=True):
	return tv.models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True, num_classes=nclasses)

class DeepLabWrapper(nn.Module):
	def __init__(self, m, in_channels=3):
		super(DeepLabWrapper, self).__init__()
		if in_channels != 3:
			m.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
		self.m = m

	def forward(self, X):
		return self.m(X)['out']

def get_deeplab_custom(nclasses, in_channels=3, pretrained=False):
	model = DeepLabWrapper(get_deeplab(21, pretrained=pretrained), in_channels=in_channels).cuda()
	if pretrained:
		model.m.aux_classifier[-1] = nn.Conv2d(256, nclasses, kernel_size=(1, 1), stride=(1, 1)).cuda()
	model.m.classifier[-1] = nn.Conv2d(256, nclasses, kernel_size=(1, 1), stride=(1, 1)).cuda()
	return model


######################################
# Modification Utils
######################################
class Split(nn.Module):
	''' Wrapper around `torch.split` '''
	def __init__(self, size, dim):
		super(Split, self).__init__()
		self.size = size
		self.dim = dim

	def forward(self, X):
		return X.split(self.size, self.dim)

class Parallel(nn.ModuleList):
	''' Passes inputs through multiple `nn.Module`s in parallel. Returns a tuple of outputs. '''

	def forward(self, Xs):
		if isinstance(Xs, torch.Tensor):
			return tuple(m(Xs) for m in self)
		assert len(Xs) == len(self)
		return tuple(m(X) for m, X in zip(self, Xs))

def Clone(n=2):
	return Parallel(nn.Identity() for _ in range(n))

class Concat(nn.Module):
	''' Concatenates an iterable input of tensors along `dim` '''
	def __init__(self, dim=1):
		super(Concat, self).__init__()
		self.dim = dim

	def forward(self, Xs):
		return torch.cat(Xs, dim=self.dim)

class Add(nn.Module):
	''' Sums an iterable input of tensors '''
	def forward(self, Xs):
		return sum(Xs)

class ConvProject(nn.Module):
	''' Projects `(N, in_channels, H, W)` to `(N, out_channels, H, W)` using a 1x1 convoluion '''
	def __init__(self, in_channels, out_channels, **kwargs):
		super(ConvProject, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, **kwargs)

	def forward(self, X):
		return self.conv(X)

class AddDict(nn.Module):
	def __init__(self, key='out'):
		super(AddDict, self).__init__()
		self.key = key

	def forward(self, Xs):
		out = OrderedDict()
		out[key] = sum(X[key] for X in Xs)
		return out

######################################
# Modifications
######################################
class ModifiedConv(nn.Module):

	def __init__(self, conv, new_conv_in_channels=1, new_conv_out_channels=1, out_channels=64):
		super(ModifiedConv, self).__init__()

		new_conv = nn.Conv2d(new_conv_in_channels, new_conv_out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
		self.net = nn.Sequential(
			Split((3, 1), dim=1),
			Parallel((conv, new_conv)),
			Concat(dim=1),
			nn.BatchNorm2d(conv.out_channels + new_conv_out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			ConvProject(conv.out_channels + new_conv_out_channels, out_channels, padding=(0, 0), bias=False)
		)

	def forward(self, X):
		out = self.net(X)
		return out

class ModifiedConv_alt(nn.Module):

	def __init__(self, conv, bn, new_conv_in_channels=1, new_conv_out_channels=64, out_channels=64):
		super(ModifiedConv_alt, self).__init__()

		original_conv = nn.Sequential(
			conv,
			deepcopy(bn),
			nn.ReLU()
		)
		new_conv = nn.Sequential(
			nn.Conv2d(new_conv_in_channels, new_conv_out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda(),
			nn.BatchNorm2d(new_conv_out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU()
		)
		self.net = nn.Sequential(
			Split((3, 1), dim=1),
			Parallel((original_conv, new_conv)),
			Concat(dim=1),
			nn.BatchNorm2d(conv.out_channels + new_conv_out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			ConvProject(conv.out_channels + new_conv_out_channels, out_channels, padding=(0, 0), bias=False)
		)

	def forward(self, X):
		out = self.net(X)
		return out

class ModifiedConv_add(nn.Module):

	def __init__(self, conv, new_conv_in_channels=1):
		super(ModifiedConv_add, self).__init__()

		new_conv = nn.Conv2d(new_conv_in_channels, conv.out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
		self.net = nn.Sequential(
			Split((3, 1), dim=1),
			Parallel((conv, new_conv)),
			Add()
		)

	def forward(self, X):
		out = self.net(X)
		return out

class ModifiedConv_alt_add(nn.Module):

	def __init__(self, conv, bn, new_conv_in_channels=1):
		super(ModifiedConv_alt_add, self).__init__()

		original_conv = nn.Sequential(
			conv,
			deepcopy(bn),
			nn.ReLU()
		)
		new_conv = nn.Sequential(
			nn.Conv2d(new_conv_in_channels, conv.out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda(),
			nn.BatchNorm2d(conv.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU()
		)
		self.net = nn.Sequential(
			Split((3, 1), dim=1),
			Parallel((original_conv, new_conv)),
			Concat(dim=1),
			Add()
		)

	def forward(self, X):
		out = self.net(X)
		return out

class DeeplabDoubleBackbone(nn.Module):

	def __init__(self, backbone1, backbone2):
		super(DeeplabDoubleBackbone, self).__init__()
		
		in_dims1 = backbone1.conv1.in_channels
		in_dims2 = backbone2.conv1.in_channels
		
		self.net = nn.Sequential(
			Split((in_dims1, in_dims2), dim=1),
			Parallel((backbone1, backbone2)),
			AddDict(key='out')
		)

	def forward(self, X):
		out = self.net(X)
		return out

class DeeplabDoubleASPP(nn.Module):

	def __init__(self, model1, model2):

		super(DeeplabDoubleASPP, self).__init__()
		
		in_dims1 = model1.backbone.conv1.in_channels
		in_dims2 = model2.backbone.conv1.in_channels
		
		self.backbone = nn.Sequential(
			Split((in_dims1, in_dims2), dim=1),
			Parallel((model1.backbone, model2.backbone)),
		)
		self.ASPP = Parallel((model1.classifier[0], model2.classifier[0]))

		self.classifier = nn.Sequential(
			self.ASPP,
			Add(),
			nn.Sequential(*model1.classifier)[1:]
		)
		self.aux_classifier = nn.Sequential(
			Parallel((model1.aux_classifier, model2.aux_classifier)),
			Add()
		)

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		out1, out2 = self.backbone(X)

		input_shape = X.shape[-2:]
		out = OrderedDict()
		out['out'] = self.classifier((out1['out'], out2['out']))
		out['out'] = F.interpolate(out['out'], size=input_shape, mode='bilinear', align_corners=False)

		out['aux'] = self.aux_classifier((out1['aux'], out2['aux']))
		out['aux'] = F.interpolate(out['aux'], size=input_shape, mode='bilinear', align_corners=False)

		return out
