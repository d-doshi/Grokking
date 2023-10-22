import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models import fcn_norm, fcn_norm_only

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
    scaler=None, if_freeze:bool=False
    ) -> list:

    # Make data dic, contains training data
    data = {'tr_acc': [], 'val_acc': [], 'loss': [], 'val_loss': [],
            'jac': [], 'grad': [], 'grad0': [], 'gradf': [],
            'time':[], 'grad1': [], 'grad2': []}

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
    model.to(device=device, dtype=dtype)
    X_train = X_train.to(device=device, dtype=dtype)
    Y_train = Y_train.to(device=device, dtype=torch.long)
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
        if not if_freeze:
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
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError('Choose only MSE or CSE!')

    model.train()  # put model to training mode
    model.to(device=device)
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


def calculate_loss(net, input, target, p, losstype='MSE'):
    import torch.nn.functional as F
    if losstype == 'MSE':
        return F.mse_loss(net(input), F.one_hot(target, p)).item()
    if losstype == 'CSE':
        return F.cross_entropy(net(input), target).item()


def train_grokking_batchstep(
    model: nn.Module, optimizer, perm: torch.Tensor,
    X_train:torch.Tensor, Y_train:torch.Tensor, X_test:torch.Tensor, Y_test:torch.Tensor,
    dtype, device: str, stopwatch:int=0, losstype: str = 'MSE',
    scheduler=None, if_data:bool=True, verbose:bool=True,
    scaler=None
    ) -> list:

    # Make data dic, contains training data
    data = {'tr_acc': [], 'val_acc': [], 'loss': [], 'val_loss': [],
            'jac': [], 'grad': [], 'grad0': [], 'gradf': [], 'grad1': [], 'grad2': [], 'grad_norm1': [], 'norm1_means':[], 'norm1_vars': []}

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError('Choose only MSE or CSE!')

    model.train()  # put model to training mode
    model = model.to(device=device)
    # x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
    # y = y.to(device=device, dtype=torch.long)
    optimizer.zero_grad()
    X = X_train[perm, ...]
    Y = Y_train[perm, ...]
    if scaler is None:
        scores = model(X).squeeze()
        if losstype == 'MSE':
            loss = criterion(
                scores, 
                F.one_hot(Y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
            )
        elif losstype == 'CSE':
            loss = criterion(scores, Y)
        loss.backward()
        optimizer.step()
    else:
        with torch.cuda.amp.autocast():
            scores = model(X).squeeze()
            if losstype == 'MSE':
                loss = criterion(
                    scores, 
                    F.one_hot(Y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
                )
            elif losstype == 'CSE':
                loss = criterion(scores, Y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if scheduler is not None:
        scheduler.step()
            
    if if_data:
        data['loss'].append(loss.detach().cpu().numpy())
        if not isinstance(model, fcn_norm_only):
            data['grad1'].append(model.fc1.weight.grad.detach().cpu().numpy())
            data['grad2'].append(model.fc2.weight.grad.detach().cpu().numpy())
        if isinstance(model, fcn_norm):
            data['grad_norm1'].append(model.norm1.weight.grad.detach().cpu().numpy())
            data['norm1_means'].append(model.norm1.weight.mean().detach().cpu().numpy())
            data['norm1_vars'].append(model.norm1.weight.var().detach().cpu().numpy())

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