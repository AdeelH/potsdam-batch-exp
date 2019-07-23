import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.utils

from collections import defaultdict
import time

from utils_ import *


def validate(model, criterion, val_dl, nclasses):

    loss = 0.
    conf_matrix = torch.zeros((nclasses, nclasses))
    classes = torch.arange(nclasses)

    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for i, (batch, labels) in enumerate(val_dl):
            N = batch.shape[0]
            batch = batch.cuda()
            labels = labels.view(-1)

            preds = model(batch).detach().cpu().permute(0, 2, 3, 1).contiguous()
            preds = preds.view(-1, preds.shape[-1])

            loss += criterion(preds, labels).item()
            preds = preds.argmax(dim=-1)
            conf_matrix += ((preds == classes[:, None]) & (labels == classes[:, None, None])).sum(dim=2).float()

    acc = conf_matrix.trace() / conf_matrix.sum()
    class_precision = conf_matrix.diag() / replace_zeros(conf_matrix.sum(dim=0))
    class_recall = conf_matrix.diag() / replace_zeros(conf_matrix.sum(dim=1))
    class_f_score = fbeta(class_precision, class_recall, beta=2)
    
    # weighted sum over classes
    weights = conf_matrix.sum(dim=1) / conf_matrix.sum()
    precision = (weights * class_precision).sum()
    recall = (weights * class_recall).sum()
    f_score = fbeta(precision, recall, beta=2)
    
    return loss / conf_matrix.sum(), (acc, precision, recall, f_score), (class_precision, class_recall, class_f_score)


def train_epoch(epoch, model, train_dl, criterion, optimizer, batch_callback=identity):
    
    loss = 0.
    corrects = 0
    count = 0

    model.train()
    for i, (batch, labels_orig) in enumerate(train_dl):
        N = batch.shape[0]
        batch = batch.cuda()
        labels = labels_orig.view(-1).cuda()

        preds = model(batch).permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.shape[-1])
        
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss += loss.item()
        corrects += (preds.argmax(dim=-1) == labels).detach().cpu().sum()
        count += len(labels)

        batch_callback(model, epoch, i, batch, labels_orig)

    acc = corrects.float() / count
    
    return loss / count, acc


def train_seg(model, train_dl, val_dl, optimizer, sched, params, criterion=nn.CrossEntropyLoss(), 
    epoch_callback=identity, batch_callback=identity, logs=defaultdict(list)):
    '''Train a classification model.

    Args:
        - model: The model to be trained (must be a `nn.Module`).
        - train_dl: A `torch.utils.data.Dataloader` for training data.
        - val_dl: A `torch.utils.data.Dataloader` for validation data.
        - optimizer: The optimizer to use for training.
        - sched: The LR scheduler to use for training. `sched.step()` will be called after each epoch.
        - params: Additional training parameters.
        - criterion: Loss function.
        - epoch_callback: A callback that receives the model and logs as input. Called at the end of every epoch.
        - batch_callback: A callback that receives the model, epoch, batch number, batch, and labels. Called at the end of every epoch.
        - logs: dict of logged variables. Default: defaultdict(list).
    '''
    torch.cuda.empty_cache()
    for epoch in range(params['epochs']):
        
        # train (fwd pass and backprop)
        train_start_time = time.time()
        train_loss, train_acc = train_epoch(epoch, model, train_dl, criterion, optimizer, batch_callback=batch_callback)
        train_end_time = time.time()

        # validate
        val_start_time = time.time()
        val_loss, val_metrics, val_class_metrics = validate(model, criterion, val_dl, 6)
        val_end_time = time.time()

        logs['epoch'     ].append(epoch)
        logs['lr'        ].append(optimizer.param_groups[0]['lr'])
        logs['train_loss'].append(train_loss.item())
        logs['val_loss'  ].append(val_loss.item())
        logs['train_acc' ].append(train_acc.item())
        logs['val_acc'   ].append(val_metrics[0].item())
        logs['train_time'].append(train_end_time - train_start_time)
        logs['val_time'  ].append(val_end_time - val_start_time)

        cm = val_class_metrics
        for i in range(6):
            logs[f'class_{i}_precision'].append(cm[0][i].item())
            logs[f'class_{i}_recall'].append(cm[1][i].item())
            logs[f'class_{i}_fscore'].append(cm[2][i].item())

        sched.step()
        epoch_callback(model, logs)


def get_past_run_info(io_handler, epochs, optimizer, scheduler):
    logs = io_handler.load_pickled_file('logs.pkl')
    last_epoch = logs['epoch'][-1]
    last_lr = logs['lr'][-1]
    remaining_epochs = epochs - last_epoch

    return last_epoch, last_lr, remaining_epochs
