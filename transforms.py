import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils
import torchvision as tv
from torchvision import transforms as tf


class DownsampleMulti(object):
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, arrs):
        return [np.ascontiguousarray(a[..., ::self.factor, ::self.factor]) for a in arrs]
    
    def __repr__(self):
        return self.__class__.__name__ + f'(factor={self.factor})'


class RandomFlipMulti(object):
    def __init__(self, h_prob=0.5, v_prob=0.5):
        self.h_prob = h_prob
        self.v_prob = v_prob
    
    def __call__(self, arrs):
        if torch.rand(1) < self.v_prob:
            arrs = [arr.flip(-2) for arr in arrs]
        if torch.rand(1) < self.h_prob:
            arrs = [arr.flip(-1) for arr in arrs]
        return arrs

    def __repr__(self):
        return self.__class__.__name__ + f'(h_prob={self.h_prob}, v_prob={self.v_prob})'


class ChannelSelect(object):
    def __init__(self, channels):
        self.channels = channels
    
    def __call__(self, arr):
        return arr[self.channels]

    def __repr__(self):
        return self.__class__.__name__ + f'(channels={self.channels})'


ch_R = 0
ch_G = 1
ch_B = 2
ch_IR = 3
ch_E = 4

def tfs_potsdam(channels=[ch_R, ch_G, ch_B], downsampling=2):

	train_transform = tf.Compose([
	    DownsampleMulti(downsampling),
	    tf.Lambda(lambda xs: [torch.tensor(x) for x in xs]),
	    RandomFlipMulti(),
	])

	val_transform = tf.Compose([
	    DownsampleMulti(downsampling),
	    tf.Lambda(lambda xs: [torch.tensor(x) for x in xs]),
	])

	x_transform = tf.Compose([
	    ChannelSelect(channels)
	])

	y_transform = tf.Compose([
	    tf.Lambda(lambda x: x.long())
	])

	return train_transform, val_transform, x_transform, y_transform
