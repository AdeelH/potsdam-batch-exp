import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.utils
import torchvision as tv
from torchvision import transforms as tf
import matplotlib.pyplot as plt


def viz_conv_layer_output(module, input, output):
    fs = output.detach().cpu().permute(1, 0, 2, 3)

    print('Layer:', module)
    print('Output shape:', fs.shape)

    normalize = not (fs.shape[-1] == 1 and fs.shape[-2] == 1)
    grid = tv.utils.make_grid(fs, nrow=int(len(fs)**0.5), normalize=normalize, scale_each=normalize, padding=2).permute(1, 2, 0)

    plt.figure(figsize=(14, 14))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()


def imshow_chw(x, **kwargs):
    plt.imshow(x.permute(1, 2, 0).squeeze(), **kwargs)


def model_debug(model, original_ds, val_ds, val_idx, flatten_seqs=False, channels=channels, max_depth=2):

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
    viz_model_run(model, val_im, viz_conv_layer_output, max_depth=max_depth)

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


def viz_model_run(model, model_input, hook_callback, max_depth=2):

    hs = []

    model.eval()
    try:
        hs = attach_hooks(model, hook_callback, max_depth=max_depth)
        with torch.no_grad():
            out = model(model_input)
    finally:
        for h in hs:
            h.remove()


def viz_conv_layer_filters(fs, normalize=True, scale_each=True, file=None, **kwargs):
    fs = fs.detach().cpu()
    grid = tv.utils.make_grid(fs, **kwargs).permute(1, 2, 0)

    plt.figure(figsize=(14, 14))
    plt.imshow(grid)
    plt.axis('off')

    if file is not None:
        plt.savefig(file)
    else:
        plt.show()


def viz_1x1_conv_filters(model, file=None):
    w = model[0][0].after[2].weight.data.squeeze().detach().cpu()

    fig = plt.figure(figsize=())

    fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(311)
    ax1.imshow(w, cmap='gray')

    ax2 = fig.add_subplot(312)
    ax2.bar(np.arange(w.shape[1]), w.sum(dim=0))
    ax2.set_yscale('log')

    ax3 = fig.add_subplot(313)
    ax3.barh(np.arange(w.shape[0]), w.sum(dim=1))
    ax3.set_ylim(ax3.get_ylim()[::-1])
    ax3.set_xscale('log')
    plt.show()

    if file is not None:
        fig.savefig(file)

    return fig
