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
# Modifications
######################################
class ModifiedConv(nn.Module):

	def __init__(self, conv, new_conv_in_channels=1, new_conv_out_channels=1, out_channels=64):
		super(ModifiedConv, self).__init__()

		self.orig_in = conv.in_channels
		self.orig_out = conv.out_channels

		self.new_in = new_conv_in_channels
		self.new_out = new_conv_out_channels

		self.original_conv = conv
		self.new_conv = nn.Conv2d(self.new_in, self.new_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
		self.bnrelu = nn.Sequential(
			nn.BatchNorm2d(self.orig_out + self.new_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU()
		)
		self.onexone = nn.Conv2d(self.orig_out + self.new_out, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		orig_out = self.original_conv(X[:, :self.orig_in])
		new_out = self.new_conv(X[:, -self.new_in:])
		concatenated = torch.cat((orig_out, new_out), dim=1)

		out = self.bnrelu(concatenated)
		out = self.onexone(out)

		return out

class ModifiedConv_alt(nn.Module):

	def __init__(self, conv, bn, new_conv_in_channels=1, new_conv_out_channels=64, out_channels=64):
		super(ModifiedConv_alt, self).__init__()

		self.orig_in = conv.in_channels
		self.orig_out = conv.out_channels

		self.new_in = new_conv_in_channels
		self.new_out = new_conv_out_channels

		self.original_conv = nn.Sequential(
			conv,
			deepcopy(bn),
			nn.ReLU()
		)
		self.new_conv = nn.Sequential(
			nn.Conv2d(self.new_in, self.new_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda(),
			nn.BatchNorm2d(self.new_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU()
		)
		self.onexone = nn.Conv2d(self.orig_out + self.new_out, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		orig_out = self.original_conv(X[:, :self.orig_in])
		new_out = self.new_conv(X[:, -self.new_in:])
		concatenated = torch.cat((orig_out, new_out), dim=1)

		out = self.onexone(concatenated)

		return out

class ModifiedConv_add(nn.Module):

	def __init__(self, conv, new_conv_in_channels=1):
		super(ModifiedConv_add, self).__init__()

		self.orig_in = conv.in_channels
		self.orig_out = conv.out_channels

		self.new_in = new_conv_in_channels
		self.new_out = self.orig_out

		self.original_conv = conv
		self.new_conv = nn.Conv2d(self.new_in, self.new_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		orig_out = self.original_conv(X[:, :self.orig_in])
		new_out = self.new_conv(X[:, -self.new_in:])
		out = orig_out + new_out

		return out

class ModifiedConv_alt_add(nn.Module):

	def __init__(self, conv, bn, new_conv_in_channels=1):
		super(ModifiedConv_alt_add, self).__init__()

		self.orig_in = conv.in_channels
		self.orig_out = conv.out_channels

		self.new_in = new_conv_in_channels
		self.new_out = self.orig_out

		self.original_conv = nn.Sequential(
			conv,
			deepcopy(bn),
			nn.ReLU()
		)
		self.new_conv = nn.Sequential(
			nn.Conv2d(self.new_in, self.new_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda(),
			nn.BatchNorm2d(self.new_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU()
		)

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		orig_out = self.original_conv(X[:, :self.orig_in])
		new_out = self.new_conv(X[:, -self.new_in:])
		out = orig_out + new_out

		return out

class RGB_E_ensemble(nn.Module):

	def __init__(self, rgb_model, e_model, nclasses=6):
		super(RGB_E_ensemble, self).__init__()

		self.rgb = nn.Sequential(
			rgb_model,
			nn.BatchNorm2d(nclasses, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.e = nn.Sequential(
			e_model,
			nn.BatchNorm2d(nclasses, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		rgb_out = self.rgb(X[:, :3])
		e_out = self.e(X[:, -1:])
		out = rgb_out + e_out

		return out

class DeeplabDoubleBackbone(nn.Module):

	def __init__(self, original_backbone, new_backbone):
		super(DeeplabDoubleBackbone, self).__init__()

		self.orig_in = original_backbone.conv1.in_channels
		self.orig_out = original_backbone.conv1.out_channels

		self.new_in = new_backbone.conv1.in_channels

		self.original_backbone = original_backbone
		self.new_backbone = new_backbone

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		orig_out = self.original_backbone(X[:, :self.orig_in])
		new_out = self.new_backbone(X[:, -self.new_in:])
		out = OrderedDict({k: orig_out[k] + new_out[k] for k in orig_out.keys()})

		return out

class DeeplabDoubleASPP(nn.Module):

	def __init__(self, model1, model2):
		super(DeeplabDoubleASPP, self).__init__()

		self.in1 = model1.backbone.conv1.in_channels
		self.out1 = model1.backbone.conv1.out_channels

		self.in2 = model2.backbone.conv1.in_channels

		self.model1 = model1
		self.model2 = model2

		self.model1.classifier = nn.Sequential(*model1.classifier)

	def forward(self, X):
		# X.shape = (N, Ch, H, W)
		input_shape = X.shape[-2:]
		out1 = self.model1.backbone(X[:, :self.in1])
		new_out = self.model2.backbone(X[:, -self.in2:])
		out = OrderedDict({k: out1[k] + new_out[k] for k in out1.keys()})

		out['out'] = self.model1.classifier[0](out['out']) + self.model2.classifier[0](out['out'])
		out['out'] = self.model1.classifier[1:](out['out'])
		out['out'] = F.interpolate(out['out'], size=input_shape, mode='bilinear', align_corners=False)

		out['aux'] = self.model1.aux_classifier[0](out['aux']) + self.model2.aux_classifier[0](out['aux'])
		out['aux'] = F.interpolate(out['aux'], size=input_shape, mode='bilinear', align_corners=False)

		return out
