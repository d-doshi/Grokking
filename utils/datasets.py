import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ml_collections.config_dict import ConfigDict
import copy


######################### Generate noisy dataset #########################
def noisy_dataset(p:int, pair_seed:int, frac:float, noise_level:float, device, dtype, operation='addition', fixed_seed:bool=True):
    
    if operation.lower() not in ['addition', 'multiplication']:
        raise Exception('noisy_dataset function only makes Modular Addition and Multipllication datasets.')
    
    pairs = [(i,j) for i in range(p) for j in range(p)]
    X_og = torch.tensor(pairs)
    if operation.lower() == 'addition': Y_og = (( X_og[:,0]**1 + X_og[:,1]**1 )**1) % p
    elif operation.lower() == 'multiplication': Y_og = (( X_og[:,0]**1 * X_og[:,1]**1 )**1) % p
    X_og = F.one_hot(X_og, num_classes=p)
    
    #### Deterministic shuffle
    random.seed(pair_seed)
    orderlist = list(range(len(pairs)))
    random.shuffle(orderlist)
    pairs = [pairs[i] for i in orderlist]
    
    X = torch.tensor(pairs)
    
    if operation.lower() == 'addition': Y = (( X[:,0]**1 + X[:,1]**1 )**1) % p
    elif operation.lower() == 'multiplication': Y = (( X[:,0]**1 * X[:,1]**1 )**1) % p
    X = F.one_hot(X, num_classes=p)
    total_size = Y.shape[0]
    train_size = int(frac * total_size)
    test_size = total_size - train_size

    n_noise = int(noise_level * train_size)
    ids = torch.arange(n_noise)

    if fixed_seed == True:
        torch.random.manual_seed(0)
        random_labels = torch.randint(0, p, (n_noise,))
        labels_noise = copy.deepcopy(Y)
        labels_noise[:n_noise] = random_labels
    else:
        random_labels = torch.randint(0, p, (n_noise,))
        labels_noise = copy.deepcopy(Y)
        print(labels_noise.shape, random_labels.shape)
        # labels_noise[torch.randperm(train_size)[:n_noise]] = random_labels
        labels_noise[:n_noise] = copy.deepcopy(random_labels)
        # count = 0
        # for i in range(n_noise):
        #     if random_labels[i] == Y[i]:
        #         count += 1
        #         labels_noise[i] = torch.randint(0, p, (1,)).item()
        # print(f'# of changed corrupted labels = {count}')
        
    Y_noisy = copy.deepcopy(Y_og)
    for i in range(total_size):
        Y_noisy[orderlist[i]] = labels_noise[i]
    Y_noisy = Y_noisy.to(device=device, dtype=torch.long)
       
    X_og = X_og.to(device=device, dtype=dtype)
    Y_og = Y_og.to(device=device, dtype=torch.long)
    X_train = X[:train_size].to(device=device, dtype=dtype)
    Y_train = labels_noise[:train_size].to(device=device, dtype=torch.long)
    X_test = X[train_size:].to(device=device, dtype=dtype)
    Y_test = labels_noise[train_size:].to(device=device, dtype=torch.long)
    
    dataset_dict = {
        'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test,
        'X_og': X_og, 'Y_og': Y_og, 'Y_noisy': Y_noisy, 'orderlist': orderlist,
        'p': p, 'data_frac': frac, 'noise_level': noise_level
    }
    
    return dataset_dict



def noisy_dataset_int(p: int, pair_seed: int, frac: float, noise_level: float, device, dtype, operation='addition', fixed_seed: bool = True):
    
    if operation.lower() not in ['addition', 'multiplication']:
        raise Exception('noisy_dataset function only makes Modular Addition and Multipllication datasets.')
    
    pairs = [(i,j) for i in range(p) for j in range(p)]
    X_og = torch.tensor(pairs)
    if operation.lower() == 'addition': Y_og = (( X_og[:,0]**1 + X_og[:,1]**1 )**1) % p
    elif operation.lower() == 'multiplication': Y_og = (( X_og[:,0]**1 * X_og[:,1]**1 )**1) % p
    X_og = F.one_hot(X_og, num_classes=p)
    
    #### Deterministic shuffle
    random.seed(pair_seed)
    orderlist = list(range(len(pairs)))
    random.shuffle(orderlist)
    pairs = [pairs[i] for i in orderlist]
    
    X = torch.tensor(pairs)
    if operation.lower() == 'addition': Y = (( X[:,0]**1 + X[:,1]**1 )**1) % p
    elif operation.lower() == 'multiplication': Y = (( X[:,0]**1 * X[:,1]**1 )**1) % p
    
    total_size = Y.shape[0]
    train_size = int(frac * total_size)
    n_noise = int(noise_level * train_size)

    if fixed_seed == True:
        torch.random.manual_seed(0)
        random_labels = torch.randint(0, p-1, (n_noise,))
        labels_noise = Y
        labels_noise[:n_noise] = random_labels
    else:
        random_labels = torch.randint(0, p-1, (n_noise,))
        labels_noise = Y
        print(labels_noise.shape, random_labels.shape)
        labels_noise[torch.randperm(train_size)[:n_noise]] = random_labels
        # labels_noise[:n_noise] = random_labels
    
    Y_noisy = torch.zeros_like(labels_noise)
    for i in range(total_size):
        Y_noisy[orderlist[i]] = labels_noise[i]
    Y_noisy = Y_noisy.to(device=device, dtype=torch.long)
       
    X_og = X_og.to(device=device, dtype=torch.long)
    Y_og = Y_og.to(device=device, dtype=torch.long)
    X_train = X[:train_size].to(device=device, dtype=torch.long)
    Y_train = labels_noise[:train_size].to(device=device, dtype=torch.long)
    X_test = X[train_size:].to(device=device, dtype=torch.long)
    Y_test = labels_noise[train_size:].to(device=device, dtype=torch.long)
    
    dataset_dict = {
        'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test,
        'X_og': X_og, 'Y_og': Y_og, 'Y_noisy': Y_noisy, 'orderlist': orderlist,
        'p': p, 'data_frac': frac, 'noise_level': noise_level
    }
    
    return dataset_dict



def visualize_dataset(dataset_dict:dict):
    X_og = dataset_dict['X_og']; Y_og = dataset_dict['Y_og']
    Y_noisy = dataset_dict['Y_noisy']
    p = dataset_dict['p']
    total_size = X_og.shape[0]
    
    #####
    targets_og_2d = np.zeros((p,p))
    targets_noisy_2d = np.zeros((p,p))
    outputs_2d = np.zeros((p,p))
    for i in range(total_size):
        targets_og_2d[i//p,i%p] = Y_og[i]
        targets_noisy_2d[i//p,i%p] = Y_noisy[i]

    #####
    plt.figure(figsize=(5,4))

    plt.title(f'target')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.contourf(np.arange(p), np.arange(p), targets_og_2d, levels=np.arange(p))
    plt.colorbar()

    plt.show()