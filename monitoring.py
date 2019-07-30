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
            log_str += f'%-20s: {log[-1]}\n' % (key)
        log_str += '-----------------------------------------------------------------------------------------------\n'

        return log_str

def track_best_model(io_handler, model, logs):

    last_epoch = logs['epoch'][-1]

    val_acc  = logs['val_acc' ][-1]
    val_loss = logs['val_loss'][-1]

    last_best_acc  = logs['best_acc' ][-1] if last_epoch > 0 else -1
    last_best_loss = logs['best_loss'][-1] if last_epoch > 0 else 1e8

    if val_acc > last_best_acc:
        logs['best_acc'].append(val_acc)
        io_handler.save_model(model, f'best_model/best_acc', info=logs)
    else:
        logs['best_acc'].append(last_best_acc)

    if val_loss < last_best_loss:
        logs['best_loss'].append(val_loss)
        io_handler.save_model(model, f'best_model/best_loss', info=logs)
    else:
        logs['best_loss'].append(last_best_loss)


def make_plots(io_handler, logs):
    fig = plot_lr(logs)
    io_handler.save_img(fig, f'visualizations/lr')
    plt.close(fig)

    fig = plot_losses(logs)
    io_handler.save_img(fig, f'visualizations/loss')
    plt.close(fig)

    fig = plot_accs(logs)
    io_handler.save_img(fig, f'visualizations/accuracy')
    plt.close(fig)

    stat_figs = plot_class_stats(logs)
    for stat, fig in stat_figs:
        io_handler.save_img(fig, f'visualizations/{stat}')
        plt.close(fig)
