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

    from utils.models import fcn
    from utils.datasets import noisy_dataset
    from utils.optimization import train_one_epoch_grokking
    from utils.data_processing import calculate_ipr

    import math
    import random
    random.seed(0)
    pair_seed = 420

    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    dtype = torch.float32
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
    
    epochs = 2000
    t = 10000000000
    lr = 1e-2

    data_epochs = np.array([epochs])

    data_fracs = np.linspace(0.1, 0.9, 17)
    noise_levels = np.linspace(0.1, 0.9, 17)
    wds = np.array([0., 5., 15., 20.])

    r = 2


    ########################### Data arrays ###########################
    train_losses = np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0]), dtype=float)
    test_losses = np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0]), dtype=float)
    train_accs = np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0]), dtype=float)
    test_accs = np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0]), dtype=float)

    iprs = {
        'U':np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0], N), dtype=float),
        'V':np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0], N), dtype=float),
        'W':np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0], N), dtype=float)
    }

    weight_norms = {
        'U':np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0]), dtype=float),
        'V':np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0]), dtype=float),
        'W':np.empty((noise_levels.shape[0], data_fracs.shape[0], wds.shape[0], data_epochs.shape[0]), dtype=float)
    }


    ########################### Loop over noise levels ###########################
    for i_n in range(noise_levels.shape[0]):
        noise_level = noise_levels[i_n]

        ########################### Loop over weight decays ###########################
        for i_f in range(data_fracs.shape[0]):
            data_frac = data_fracs[i_f]
            
            ########################### Loop over weight decays ###########################
            for i_w in range(wds.shape[0]):
                wd = wds[i_w]

                ########################### Generatre dataset ###########################
                dataset_dict = noisy_dataset(
                    p, pair_seed, data_frac, noise_level, operation='addition', device=device, dtype=dtype, fixed_seed=True
                )
                X_train = dataset_dict['X_train']; Y_train = dataset_dict['Y_train']
                X_test = dataset_dict['X_test']; Y_test = dataset_dict['Y_test']

                ########################### Initialize the model ###########################
                torch.manual_seed(1)
                model = fcn(2*p, N, p)
                # model.act = nn.ReLU()
                act = str(model.act)
                ########################### Optimizer ###########################
                optimizer = optim.AdamW(model.parameters(), lr=lr,  weight_decay=wd, betas=(0.9, 0.98), eps=1e-08)

                i_d = 0
                ########################### Training loop ###########################
                for epoch in range(1,epochs+1):
                    if epoch in data_epochs:
                        if_data = True
                    else:
                        if_data = False
                        
                    train_data = train_one_epoch_grokking(
                        model, optimizer, t, X_train, Y_train, X_test, Y_test,
                        dtype, device, losstype='MSE', if_data=if_data, verbose=False
                    )

                
                    ## collect data
                    if if_data:
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
                                
                        i_d += 1


                print(f'noise_level={noise_level}; data_frac={data_frac},  wd={wd} : \
                    {train_losses[i_n, i_f, i_w, -1]}\t{test_losses[i_n, i_f, i_w, -1]}\t{train_accs[i_n, i_f, i_w, -1]}\t{test_accs[i_n, i_f, i_w, -1]}')

                del model, U, V, W, optimizer


    ########################### print shapes of data arrays ###########################
    print(data_epochs.shape)
    print(train_losses.shape, test_losses.shape, train_accs.shape, test_accs.shape)
    print(iprs['U'].shape, iprs['V'].shape, iprs['W'].shape)


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
        'epochs': epochs,
        'noise_levels': noise_levels,
        'data_fracs': data_fracs,
        'wds': wds,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'iprs': iprs,
    }

    location = './'

    with open(location + f'modular_addition_wd(Quadratic,N={N}).pickle', 'wb') as f:
        pickle.dump(data, f)


###########################
####### Boilerplate #######
###########################

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
