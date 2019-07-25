import torch
import numpy as np
import torchvision as tv
from utils_ import *


class Potsdam(tv.datasets.VisionDataset):
    
    def __init__(self, d, chip_size=100, stride=1, tf=lambda x: x, x_tf=lambda x: x, y_tf=lambda x: x):
        super(Potsdam).__init__()

        self.data = list(d.values())
        self.fnames = list(d.keys())

        self.n = len(d)
        self.chip_size = chip_size
        self.stride = stride

        im_size = self.data[0][2].shape
        self.h = ((im_size[0] - chip_size) // stride) + 1
        self.w = ((im_size[1] - chip_size) // stride) + 1

        self.tf, self.x_tf, self.y_tf = tf, x_tf, y_tf


    def __len__(self):
        return self.n * self.w * self.h
    
    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return self._load(i)
        if isinstance(i, np.ndarray) or isinstance(i, torch.Tensor):
            i = i.squeeze()
            assert i.ndim in (0, 1), "too many dimensions"
            if i.ndim == 0:
                return self._load(i)
            return [self._load(j) for j in i]
        if isinstance(i, slice):
            return [self._load(j) for j in range(*i.indices(len(self.fnames)))]
        assert False, f"__getitem__(i): Invalid index"
        
    def _to_chip_idx(self, i):
        im_idx = i // (self.w * self.h)
        im_i = i % (self.w * self.h)
        chip_col = im_i % self.w
        chip_row = im_i // self.h
        return im_idx, chip_row, chip_col

    def _load(self, i):
        im_idx, chip_row, chip_col = self._to_chip_idx(i)
        rgbir, elevation, label  = self.data[im_idx]

        x_start, y_start = chip_col * self.stride, chip_row * self.stride
        x_end, y_end = x_start + self.chip_size, y_start + self.chip_size

        rgbir_chip     = rgbir[:,  y_start : y_end, x_start : x_end]
        elevation_chip = elevation[y_start : y_end, x_start : x_end]
        label_chip     = label[    y_start : y_end, x_start : x_end]

        x = torch.cat((rgbir_chip, elevation_chip.unsqueeze(0)), dim=0).float() / 255

        x, y = self.tf((x, label_chip))

        return self.x_tf(x), self.y_tf(y)
