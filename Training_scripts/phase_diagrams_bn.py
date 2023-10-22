def main():
    """ Main entry point of the script """

    ########################
    ####### Setup #######
    ########################

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable

    import os
    import sys
    import copy
    import pickle
    import time
    
    module_path = os.path.abspath(os.path.join(".."))
    if module_path not in sys.path:
        sys.path.append(module_path)

    from utils.datasets import noisy_dataset
    from utils.models import fcn_norm
    from utils.optimization import train_grokking_batchstep
    from utils.data_processing import calculate_ipr
    import math
    import random
    random.seed(0)
    pair_seed = 420

    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    dtype = torch.float32
    complexdtype = torch.complex64
    USE_GPU = True
    if USE_GPU == True and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    print('using device:', device)


    ########################
    ####### Training #######
    ########################

    N = 500
    p = 97
    
    steps = 2000
    t = 10000000000
    wd = 0.0
    
    data_steps = np.array([steps])
    
    bss = np.array([8, 32, 64, 256])
    data_fracs = np.linspace(0.1, 0.9, 17)
    noise_levels = np.linspace(0.1, 0.9, 17)

    r = 2

    ########################### Data arrays ###########################
    train_losses = np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0]), dtype=float)
    test_losses = np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0]), dtype=float)
    train_accs = np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0]), dtype=float)
    test_accs = np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0]), dtype=float)

    iprs = {
        'U':np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0], N), dtype=float),
        'V':np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0], N), dtype=float),
        'W':np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0], N), dtype=float),
    }

    weight_norms = {
        'U':np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0]), dtype=float),
        'V':np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0]), dtype=float),
        'W':np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0]), dtype=float)
    }

    norm_weights = {
        'norm_weights': np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0], N), dtype=float),
        'norm_bias': np.empty((noise_levels.shape[0], data_fracs.shape[0], bss.shape[0], data_steps.shape[0], N), dtype=float)
    }
    ########################### Loop over noise levels ###########################
    for i_n in range(noise_levels.shape[0]):
        noise_level = noise_levels[i_n]

        ########################### Loop over weight decays ###########################
        for i_f in range(data_fracs.shape[0]):
            data_frac = data_fracs[i_f]
            
            ########################### Loop over weight decays ###########################
            for i_w in range(bss.shape[0]):
                bs = bss[i_w]
                
                standard_steps = np.array([2000])
                data_steps = standard_steps * 256 // bs
                steps = data_steps[-1]
                
                lr = 1e-2 * np.sqrt(bs / 256) if bs > 0 else 1e-2
                
                ########################### Generatre dataset ###########################
                dataset_dict = noisy_dataset(
                    p, pair_seed, data_frac, noise_level, operation='addition', device=device, dtype=dtype, fixed_seed=True
                )
                X_train = dataset_dict['X_train']; Y_train = dataset_dict['Y_train']
                X_test = dataset_dict['X_test']; Y_test = dataset_dict['Y_test']

                ########################### Initialize the model ###########################
                torch.manual_seed(1)
                model = fcn_norm(2*p, N, p, norm='bn')
                act = str(model.act)
                ########################### Optimizer ###########################
                optimizer = optim.AdamW(model.parameters(), lr=lr,  weight_decay=wd, betas=(0.9, 0.98), eps=1e-08)

                i_d = 0
                steps_per_epoch = len(X_train) // bs
                step = 1
                ########################### Training loop ###########################
                
                while step <= steps + 1:
                    if (step - 1) % steps_per_epoch == 0:
                        torch.manual_seed(step)
                        perms = torch.randperm(len(X_train), device=device)
                        perms = perms[:steps_per_epoch * bs]
                        perms = perms.view((steps_per_epoch, bs))
                    
                    for perm in perms:
                        if step in data_steps:
                            if_data = True
                        else:
                            if_data = False
                            
                        train_data = train_grokking_batchstep(
                            model, optimizer, perm, X_train, Y_train, X_test, Y_test,
                            dtype, device, losstype='MSE', if_data=if_data, verbose=False
                        )

                        ## collect data
                        if if_data:
                            # print(losses.shape, len(train_data['loss']), i_n, i_f, i_w, i_d)
                            train_losses[i_n, i_f, i_w, i_d] = train_data['loss'][-1]
                            test_losses[i_n, i_f, i_w, i_d]= train_data['val_loss'][-1]
                            train_accs[i_n, i_f, i_w, i_d] = train_data['tr_acc'][-1]
                            test_accs[i_n, i_f, i_w, i_d] = train_data['val_acc'][-1]
                            
                            U = copy.deepcopy(model.fc1.weight.data[:, :97]).detach().cpu().numpy()
                            V = copy.deepcopy(model.fc1.weight.data[:, 97:]).detach().cpu().numpy()
                            W = copy.deepcopy(model.fc2.weight.data).detach().cpu().numpy()
                                
                            
                            weight_norms['U'][i_n, i_f, i_w, i_d] = (np.abs(U)**2).sum()
                            weight_norms['V'][i_n, i_f, i_w, i_d] = (np.abs(V)**2).sum()
                            weight_norms['W'][i_n, i_f, i_w, i_d] = (np.abs(W)**2).sum()
                                
                            for k in range(N):
                                iprs['U'][i_n, i_f, i_w, i_d, k] = calculate_ipr( np.absolute(np.fft.rfft(U[k])), r ) 
                                iprs['V'][i_n, i_f, i_w, i_d, k] = calculate_ipr( np.absolute( np.fft.rfft(V[k])), r )
                                iprs['W'][i_n, i_f, i_w, i_d, k] = calculate_ipr( np.absolute( np.fft.rfft(W[:,k])), r )
                                  
                            norm_weights['norm_weights'][i_n, i_f, i_w, i_d, :] = model.norm1.weight.data.detach().clone().cpu().numpy()
                            norm_weights['norm_bias'][i_n, i_f, i_w, i_d, :] = model.norm1.bias.data.detach().clone().cpu().numpy()
                                    
                            i_d += 1

                            print(f'noise_level={noise_level}; data_frac={data_frac},  wd={wd} : \
                            {train_losses[i_n, i_f, i_w, i_d-1]}\t{test_losses[i_n, i_f, i_w, i_d-1]}\t{train_accs[i_n, i_f, i_w, i_d-1]}\t{test_accs[i_n, i_f, i_w, i_d-1]}')
                        step += 1


    ########################### print shapes of data arrays ###########################
    print(data_steps.shape)
    print(train_losses.shape, test_losses.shape, train_accs.shape, test_accs.shape)
    print(iprs['U'].shape, iprs['V'].shape, iprs['W'].shape)
    # print(counts['U'].shape, counts['V'].shape, counts['W'].shape)


    ##########################
    ####### Save stuff #######
    ##########################

    data = {
        'operation': 'addition',
        'model': 'fcn',
        'act': act,
        'loss': 'MSE',
        'p' : p,
        'N': N,
        'optimizer': 'AdamW',
        'lr': lr,
        'steps': steps,
        'noise_levels': noise_levels,
        'data_fracs': data_fracs,
        'bss': bss,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'iprs': iprs,
        'norm_pms': norm_weights,
    }

    location = './'

    with open(location + f'modular_addition_bn_3d_MSE(Quadratic,N={N})_long.pickle', 'wb') as f:
        pickle.dump(data, f)


###########################
####### Boilerplate #######
###########################

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
