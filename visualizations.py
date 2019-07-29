from utils_ import *
from models import *

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.utils
import torchvision as tv
from torchvision import transforms as tf
import matplotlib.pyplot as plt

from collections import OrderedDict


def imshow_chw(x, **kwargs):
    plt.imshow(x.permute(1, 2, 0).squeeze(), **kwargs)


def viz_conv_layer_output(module, input, output):
    if isinstance(output, OrderedDict): # handle deeplab
        output = output['out']
    fs = output.detach().cpu().permute(1, 0, 2, 3)

    print('Layer:', module)
    print('Output shape:', fs.shape)

    normalize = not (fs.shape[-1] == 1 and fs.shape[-2] == 1)
    grid = tv.utils.make_grid(fs, nrow=int(len(fs)**0.5), normalize=normalize, scale_each=normalize, padding=2).permute(1, 2, 0)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()


def model_debug(model, original_ds, val_ds, val_idx, flatten_seqs=False, channels=[0, 1, 2], max_depth=2, recurse_whitelist=(nn.Sequential, nn.ModuleList)):

    im, label = original_ds[val_idx]
    im_rgb = im.permute(1, 2, 0)[..., :3].squeeze()
    im_input = im.permute(1, 2, 0)[..., channels[:4]].squeeze()
    
    im_idx, row, col = original_ds._to_chip_idx(val_idx)
    c_sz = original_ds.chip_size
    fname = f'{original_ds.fnames[im_idx]}.tif'

    print('\n------------------------------------------- Input image -------------------------------------------\n')
    f = plt.figure(figsize=(18, 6))
    ax1 = f.add_subplot(131)
    ax1.imshow(im_rgb)
    ax1.set_title(f'{val_idx}, {fname}, ({row} : {row + c_sz}, {col} : {col + c_sz}), channels: {channels}')
    ax1.axis('off')

    ax2 = f.add_subplot(132)
    ax2.imshow(im_input)
    ax2.set_title(f'Model input')
    ax2.axis('off')

    ax3 = f.add_subplot(133)
    ax3.imshow(label)
    ax3.set_title(f'Ground truth labels')
    ax3.axis('off')
    plt.show()

    print('\n------------------------------------------- Layer-wise output -------------------------------------------\n')
    
    val_im = val_ds[val_idx][0].unsqueeze(0).cuda()
    out = viz_model_run(model, val_im, viz_conv_layer_output, max_depth=max_depth, recurse_whitelist=recurse_whitelist)

    print('\n------------------------------------------- Final prediction -------------------------------------------\n')
    yhat = out.cpu().squeeze().permute(1, 2, 0).argmax(dim=-1)
    
    f = plt.figure(figsize=(18, 6))
    ax1 = f.add_subplot(131)
    ax1.imshow(im_rgb)
    ax1.set_title(f'{val_idx}, {fname}, ({row} : {row + c_sz}, {col} : {col + c_sz}), channels: {channels}')
    ax1.axis('off')

    ax2 = f.add_subplot(132)
    ax2.imshow(label)
    ax2.set_title(f'Ground truth labels')
    ax2.axis('off')

    ax3 = f.add_subplot(133)
    ax3.imshow(yhat)
    ax3.set_title(f'Predicted labels')
    ax3.axis('off')
    plt.show()


def viz_model_run(model, model_input, hook_callback, max_depth=2, recurse_whitelist=(nn.Sequential, nn.ModuleList)):

    hs = []

    model.eval()
    try:
        hs = attach_forward_hooks(model, hook_callback, max_depth=max_depth, recurse_whitelist=recurse_whitelist)
        with torch.no_grad():
            out = model(model_input)
    finally:
        for h in hs:
            h.remove()
    return out


def viz_conv_layer_filters(fs, title='', normalize=True, scale_each=True, show=False, figsize=(14, 14), **kwargs):
    fs = fs.detach().cpu()
    grid = tv.utils.make_grid(fs, normalize=normalize, scale_each=scale_each, **kwargs).permute(1, 2, 0)

    fig = plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.title(title)
    plt.axis('off')

    if show:
        plt.show()
    else:
        return fig # remember to close figure after use


def viz_1x1_conv_filters(fs, title='', show=False, figsize=(12, 6), cmap='gray'):
    w = fs.squeeze().detach().cpu()
    assert w.ndim == 2

    fig = plt.figure(figsize=figsize)
    plt.imshow(w, cmap=cmap)
    plt.title(title)

    if show:
        plt.show()
    else:
        return fig # remember to close figure after use


def plot_epoch_wise(epochs, ys, title='', labels=None, show=False, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    for y in ys:
        plt.plot(epochs, y)
    plt.title(title)
    plt.xlabel('epoch')

    if labels is not None:
        plt.legend(labels=labels)

    if show:
        plt.show()
    else:
        return fig # remember to close figure after use


def plot_lr(epochs, lr, title='Learning rate', show=False, figsize=(10, 10)):
    return plot_epoch_wise(epochs, [lr], title=title, show=show, figsize=figsize)


def plot_losses(epochs, train_loss, val_loss, title='Loss', show=False, figsize=(10, 10)):
    return plot_epoch_wise(epochs, [train_loss, val_loss], title=title, show=show, figsize=figsize, labels=('train', 'val'))


def plot_accs(epochs, train_acc, val_acc, title='Accuracy', show=False, figsize=(10, 10)):
    return plot_epoch_wise(epochs, [train_acc, val_acc], title=title, show=show, figsize=figsize, labels=('train', 'val'))


def plot_class_stats(logs, stats=['precision', 'recall', 'fscore'], show=False, figsize=(10, 10)):
    figs = []
    for stat in stats:
        keys = [k for k in logs.keys() if k.startswith('class_') and k.endswith(f'_{stat}')]
        if len(keys) == 0:
            continue
        class_stats = [logs[k] for k in keys]
        fig = plot_epoch_wise(logs['epoch'], class_stats, title=f'{stat}', show=show, figsize=figsize, labels=keys)
        if not show:
            figs.append((stat, fig))
    if not show:
        return figs
