import math
import numpy as np
import torch
import torch.nn as nn


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

        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.25**0.5 / np.power(2*self.h_size, 1/3)) #### standard initialization
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.25**0.5 / np.power(2*self.h_size, 1/3))

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc2( self.act( self.dp( self.fc1(x) ) ) )
        return x
    
    
class fcn_norm(nn.Module):
    def __init__(self, in_size, h_size, out_size, norm=None):
        super(fcn_norm, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size

        self.fc1 = nn.Linear(in_size, h_size, bias=False)
        self.fc2 = nn.Linear(h_size, out_size, bias=False)
        if norm == 'bn':
            self.norm1 = nn.BatchNorm1d(h_size)
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(h_size)
        self.act = nnPower(2)
        # self.dp = nn.Dropout(dp)

        # torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0**0.5 / math.sqrt(self.h_size))
        # torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0**0.5 / math.sqrt(self.h_size))

        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.25**0.5 / np.power(2*self.h_size, 1/3)) #### standard initialization
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.25**0.5 / np.power(2*self.h_size, 1/3))

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm1(x)
        x = self.fc2(x)
        # x = self.fc2( self.act( self.norm1( self.fc1(x) ) ) )
        return x
    

class fcn_norm_only(nn.Module):
    def __init__(self, in_size, h_size, out_size, depth, norm=None):
        super(fcn_norm_only, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size

        
        self.layers = nn.ModuleList([nn.Linear(in_size, h_size, bias=False)])
        for _ in range(depth - 1):
            if norm == 'bn':
                self.layers.append(nn.BatchNorm1d(h_size))
            elif norm == 'ln':
                self.norm1 = nn.LayerNorm(h_size)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(h_size, h_size, bias=False))
        
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(h_size, out_size, bias=False))
        

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a = 0, mode = 'fan_in', nonlinearity='relu')
                layer.weight.requires_grad_(False)
        self.layers[-1].weight.data *= 0.5

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        # x = self.fc2( self.act( self.norm1( self.fc1(x) ) ) )
        return x