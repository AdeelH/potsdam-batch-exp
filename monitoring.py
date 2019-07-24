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
