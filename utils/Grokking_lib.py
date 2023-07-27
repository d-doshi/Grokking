import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import sys

import math
import random
random.seed(0)
pair_seed = 420

dtype = torch.float32
complexdtype = torch.complex64


################ Real MLP models #################

class nnPower(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = power

    def forward(self, x):
        return torch.pow(x, self.power)


class fcn(nn.Module):
    def __init__(self, in_size, h_size, out_size, dp=0.0):
        super(fcn, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size

        self.fc1 = nn.Linear(in_size, h_size, bias=False)
        self.fc2 = nn.Linear(h_size, out_size, bias=False)
        self.act = nnPower(2)
        self.dp = nn.Dropout(dp)

        # torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0**0.5 / math.sqrt(self.h_size))
        # torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0**0.5 / math.sqrt(self.h_size))

        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std= 0.5 / np.power(2*self.h_size, 1/3)) #### standard initialization
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std= 0.5 / np.power(2*self.h_size, 1/3))

    def forward(self, x):
        x = x.flatten(1)
        # x = self.fc2( self.act( self.dp( self.fc1(x) ) ) )
        x = self.fc2( self.fc1(x)**2 )
        return x


######################### Training #########################

@torch.no_grad()
def check_accuracy_grokking(
    X:torch.Tensor, Y:torch.Tensor, model: nn.Module, dtype, device: str, scaler
) -> tuple:
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    x_wrong = []
    # X = X.to(dtype=dtype)
    Y = Y.to(dtype=torch.long)

    if scaler is None:
        scores = model(X)
    else:
        with torch.cuda.amp.autocast():
            scores = model(X)
            
    _, preds = scores.max(1)
    num_correct += (preds == Y).sum()
    num_samples += preds.size(0)
    x_wrong.append(X[Y != preds])
    acc = float(num_correct) / num_samples

    return num_correct, num_samples, acc


@torch.no_grad()
def test_loss_grokking(model: nn.Module, X:torch.Tensor, Y:torch.Tensor, 
              losstype: str, dtype, device: str, scaler):
    
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    
    loss = 0
    model.eval()  # put model to training mode
    # X = X.to(dtype=dtype)
    Y = Y.to(dtype=torch.long)
    
    if scaler is None:
        scores = model(X).squeeze()
        if losstype == 'MSE':
            loss += criterion(
                scores, 
                F.one_hot(Y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
            )
        elif losstype == 'CSE':
            loss += criterion(scores, Y)
    else:
        with torch.cuda.amp.autocast():
            scores = model(X).squeeze()
            if losstype == 'MSE':
                loss += criterion(
                    scores, 
                    F.one_hot(Y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
                )
            elif losstype == 'CSE':
                loss += criterion(scores, Y)
    return loss


def train_one_epoch_grokking(
    model: nn.Module, optimizer,  time: int, 
    X_train:torch.Tensor, Y_train:torch.Tensor, X_test:torch.Tensor, Y_test:torch.Tensor,
    dtype, device: str, stopwatch:int=0, losstype: str = 'MSE',
    scheduler=None, if_data:bool=True, verbose:bool=True,
    scaler=None
    ) -> list:

    # Make data dic, contains training data
    data = {'tr_acc': [], 'val_acc': [], 'loss': [], 'val_loss': [],
            'jac': [], 'grad': [], 'grad0': [], 'gradf': [],
            'time':[], 'grad1': [], 'grad2': []}

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError('Choose only MSE or CSE!')

    stopwatch = stopwatch

    # for t, (x,y) in enumaroate(loader_train):
    # for t, (x, y) in enumerate(zip(X_train, Y_train)):
        
    data['time'] = stopwatch
    # if stopwatch == time:
    #     break

    stopwatch += 1
    model.train()  # put model to training mode
    model = model.to(device=device)
    # x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
    # y = y.to(device=device, dtype=torch.long)


    optimizer.zero_grad()
    
    if scaler is None:
        scores = model(X_train).squeeze()
        if losstype == 'MSE':
            loss = criterion(
                scores, 
                F.one_hot(Y_train, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
            )
        elif losstype == 'CSE':
            loss = criterion(scores, Y_train)
        loss.backward()
        optimizer.step()
    else:
        with torch.cuda.amp.autocast():
            scores = model(X_train).squeeze()
            if losstype == 'MSE':
                loss = criterion(
                    scores, 
                    F.one_hot(Y_train, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
                )
            elif losstype == 'CSE':
                loss = criterion(scores, Y_train)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if if_data:
        data['loss'].append(loss.detach().cpu().numpy())
        data['grad1'].append(model.fc1.weight.grad.detach().cpu().numpy())
        data['grad2'].append(model.fc2.weight.grad.detach().cpu().numpy())
        
        
    if scheduler is not None:
        scheduler.step()
    
    if if_data:
        num_correct, num_samples, running_train = check_accuracy_grokking(
            X_train, Y_train, model, dtype, device, scaler
        )
        data['tr_acc'].append(running_train)
        num_correct, num_samples, running_val = check_accuracy_grokking(
            X_test, Y_test, model, dtype, device, scaler
        )
        data['val_acc'].append(running_val)
        data['val_loss'].append(test_loss_grokking(
            model, X_test, Y_test, losstype, dtype, device, scaler
        ).detach().clone().cpu().item())
    
    if verbose:
        print('TRAIN: {0:.2f},  TEST: {1:.2f}'.format(running_train, running_val))

    return data


def calculate_gradients(
    model: nn.Module, X, Y,
    dtype, device: str, losstype: str='MSE'
):
    grads = {}
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError('Choose only MSE or CSE!')

    model.train()  # put model to training mode
    model = model.to(device=device)
    X = X.to(dtype=dtype)  # move to device, e.g. GPU
    Y = Y.to(dtype=torch.long)
    
    scores = model(X).squeeze()
    if losstype == 'MSE':
        loss = criterion(
            scores, 
            F.one_hot(Y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
        )
    elif losstype == 'CSE':
        loss = criterion(scores, Y)
    loss.backward()

    for name, param in model.named_parameters():
        grads[name] = param.grad

    return grads


######################### Hooks #########################

class Hook():
    def __init__(self, module: nn.Module, backward: bool = False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module: nn.Module, input: Tensor, output: Tensor):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def fwd_hook_(model: nn.Module, module: nn.Module, n: int = 0) -> Hook:
    i = 1
    for id, layer in list(model.named_modules()):
        if isinstance(layer, module):
            if i == n:
                return Hook(layer, backward=False)
            else:
                i += 1


def bwd_hook_(model: nn.Module, module: nn.Module, n: int = 0) -> Hook:
    i = 1
    for id, layer in list(model.named_modules()):
        if isinstance(layer, module):
            if i == n:
                return Hook(layer, backward=True)
            else:
                i += 1


######################### Plot vertical lines at given freq #########################
def lines(k, p, col='grey'):
    plt.axvline(x=k%97, alpha=0.5, color=col)
    plt.axvline(x=97-k%97, alpha=0.5, color=col)

def line(k, p, col='grey'):
    kk = np.minimum(k%p, p - k%p)
    plt.axvline(x=kk, alpha=0.5, color=col)


######################### IPR etc. #########################
def calculate_ipr(array, r):
    return np.power(array / np.sqrt((array ** 2).sum()), 2*r).sum()

def ipr_test(array, r, chi_ipr):
    ipr_local = calculate_ipr(array, r)
    if ipr_local >= chi_ipr:
        return True
    else:
        return False

def calculate_gini(array):
    return np.abs(np.expand_dims(array, 0) - np.expand_dims(array, 1)).mean() / array.mean() / 2


######################### Custom contourplots #########################
def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    plt.show()