import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.utils
import torchvision as tv
from torchvision import transforms as tf
import matplotlib.pyplot as plt
import os

from io_ import *
from visualizations import *

def checkpoint(io_handler, model, model_name, logs={}):
    io_handler.save_model(model, f'checkpoints/{model_name}', info=logs)


def logs_to_str(logs):
        log_str = ''
        for key, log in logs.items():
            log_str += '%-20s: %.4f\n' % (key, log[-1])
        log_str += '-----------------------------------------------------------------------------------------------\n'

        return log_str


def get_epoch_monitor(io_handler, model_name='model', chkpt_interval=1, acc_tol=0.005, viz_root='visualizations/per_epoch'):
    assert chkpt_interval > 0 and acc_tol < 1

    filter_path = io_handler.to_local_path(f'{viz_root}/rgb_conv')
    os.makedirs(filter_path, exist_ok=True)

    def _monitor(model, logs):
        epoch = len(logs['epoch']) # epoch is now 1-indexed
        val_acc = logs['val_acc'][-1]
        last_best_acc = logs['best_acc'][-1] if epoch > 1 else -1

        if epoch % chkpt_interval == 0:
            io_handler.save_model(model, f'checkpoints/epoch_{epoch}', info=logs)

        if val_acc >= (last_best_acc + acc_tol):
            logs['best_acc'].append(val_acc)
            # io_handler.save_model(model, f'best_model/{model_name}', info=logs)
        else:
            logs['best_acc'].append(last_best_acc)

        log_str = logs_to_str(logs)
        print(log_str)

        io_handler.save_log('logs.pkl', logs)
        io_handler.save_log_str(f'logs.txt', logs)

        viz_conv_layer_filters(model[0][0].weight.data.cpu(), file=f'{filter_path}/epoch_%04d' % (epoch), padding=0)
        # viz_conv_layer_filters(model[0][0].original_conv1.weight.data, file='visualizations/rgb_conv/epoch_%04d' % epoch, padding=0)
        # viz_conv_layer_filters(model[0][0].new_conv1.weight.data, file='visualizations/e_conv/epoch_%04d' % epoch, padding=0)

    return _monitor

def get_batch_monitor(io_handler, viz_root='visualizations/per_batch', interval=4):

    filter_path = io_handler.to_local_path(f'{viz_root}/rgb_conv')
    grad_path = io_handler.to_local_path(f'{viz_root}/rgb_conv_grad')
    os.makedirs(filter_path, exist_ok=True)
    os.makedirs(grad_path, exist_ok=True)

    def _monitor(model, epoch, batch_idx, batch, labels):
        if (batch_idx + 1) % interval == 0:
            fs = model[0][0].weight
            viz_conv_layer_filters(fs.data.cpu(), file=f'{filter_path}/epoch_%04d_batch_%05d' % (epoch, batch_idx), scale_each=False, padding=0)
            viz_conv_layer_filters(fs.grad.data.cpu(), file=f'{grad_path}/epoch_%04d_batch_%05d' % (epoch, batch_idx), scale_each=False, padding=0)
            # viz_conv_layer_filters(model[0][0].original_conv1.weight.data, file='visualizations/rgb_conv/epoch_%04d' % epoch, padding=0)
            # viz_conv_layer_filters(model[0][0].new_conv1.weight.data, file='visualizations/e_conv/epoch_%04d' % epoch, padding=0)

            # viz_conv_layer_filters(model[0][0].original_conv1.weight.grad.data, file='visualizations/rgb_conv_grad/epoch_%04d' % epoch, padding=0)
            # viz_conv_layer_filters(model[0][0].new_conv1.weight.grad.data, file='visualizations/e_conv_grad/epoch_%04d' % epoch, padding=0)

    return _monitor
