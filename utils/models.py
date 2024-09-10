import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections.config_dict import ConfigDict
from timm.optim.optim_factory import param_groups_weight_decay


class nnPower(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = power

    def forward(self, x):
        return torch.pow(x, self.power)


################################
#### Fully connected models ####
################################

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
    

############################
#### Transformer models ####
############################

class EncoderOnlyTransformers(nn.Module):
    
    def __init__(self, Attn_Module: nn.Module, config: ConfigDict) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(Attn_Module, config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd) if config.if_ln else nn.Identity(),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.g_scale = 1.
        if config.weight_tying:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.apply(self._init_weights)
        self.register_buffer('scale', torch.sqrt(torch.tensor([1. / config.n_embd**(1. + config.s)]))) # Interpolate between SP and muP
        return
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.scale * self.lm_head(x)

        return logits
    
    def config_optimizers(model: nn.Module, config: ConfigDict) -> torch.optim.Optimizer:
        """
        Optimizer config following Sho's paper: https://arxiv.org/abs/2210.04909 
        """
        params = []
        if config.optim_name.lower() == 'adamw' and config.rescale_optim == True:
            for id, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    if not id.endswith(('c_fc', 'lm_head')):
                        params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad and p.ndim > 1],
                                    'lr': config.lr * config.n_embd**(config.s / 2. - 0.5) * m.g_scale,
                                    'weight_decay': config.weight_decay}]
                    elif id.endswith('c_fc'):
                        params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad and p.ndim > 1],
                                    'lr': config.lr * config.n_embd**(config.s / 2. - 0.5) * config.wide_factor**(-0.5) * m.g_scale,
                                    'weight_decay': config.weight_decay}]
                    elif id.endswith('lm_head') and config.weight_tying == False:
                        params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad and p.ndim > 1],
                                'lr': config.lr * config.n_embd**(config.s / 2. - 0.5),
                                'weight_decay': config.weight_decay}] 
                if isinstance(m, nn.LayerNorm):
                    params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad],
                                'lr': config.lr,
                                'weight_decay': 0.0}]
                if isinstance(m, nn.Embedding):
                    params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad and p.ndim > 1],
                                'lr': config.lr * config.n_embd**(config.s / 2. - 0.5),
                                'weight_decay': 0.0}]
            count = 0
            for p in params:
                for pm in p['params']:
                    count += pm.numel()
            assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = torch.optim.AdamW(params, eps=config.eps, betas=(config.beta1, config.beta2))
            
        elif config.optim_name.lower() == 'sgd' and config.rescale_optim == True:
            for id, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    if not id.endswith('lm_head'):
                        params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad and p.ndim > 1],
                                    'lr': config.lr * config.n_embd**(config.s) * m.g_scale,
                                    'weight_decay': config.weight_decay}]
                    elif id.endswith('lm_head') and config.weight_tying == False:
                        params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad and p.ndim > 1],
                                'lr': config.lr * config.n_embd**(config.s),
                                'weight_decay': config.weight_decay}] 
                if isinstance(m, nn.LayerNorm):
                    params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad],
                                'lr': config.lr,
                                'weight_decay': 0.0}]
                if isinstance(m, nn.Embedding):
                    params += [{'params': [p for n, p in m.named_parameters() if p.requires_grad and p.ndim > 1],
                                'lr': config.lr * config.n_embd**(config.s),
                                'weight_decay': 0.0}]
            count = 0
            for p in params:
                for pm in p['params']:
                    count += pm.numel()
            assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = torch.optim.SGD(params, momentum = config.momentum, nesterov = False)
            
        elif config.optim_name.lower() == 'adamw' and config.rescale_optim == False:
            params = param_groups_weight_decay(model, weight_decay=config.weight_decay)
            optimizer = torch.optim.AdamW(params, lr = config.lr, eps = config.eps, betas = (config.beta1, config.beta2))
        elif config.optim_name.lower() == 'sgd':
            params = param_groups_weight_decay(model, weight_decay = config.weight_decay)
            optimizer = torch.optim.SGD(params, lr = config.lr, momentum = config.momentum, nesterov = False)
        else:
            raise Exception("AdamW / SGD only")
        
        print((sum(p.numel() for p in model.parameters() if p.requires_grad) - (config.vocab_size + config.block_size) * config.n_embd) / 1e6, 'M non-embedding parameters')
        return optimizer


class Block(nn.Module):

    def __init__(self, Attn_Module: nn.Module, config) -> None:
        super().__init__()
        self.mu = config.mu
        self.ln_1 = nn.LayerNorm(config.n_embd) if config.if_ln else nn.Identity()
        self.attn = Attn_Module(config)
        self.ln_2 = nn.LayerNorm(config.n_embd) if config.if_ln else nn.Identity()
        self.mlp = MLP(config)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mu * x + self.attn(self.ln_1(x))
        x = self.mu * x + self.mlp(self.ln_2(x))
        return x


## Simple implementation of multihead attention module
class MultiHeadAttention(nn.Module):
    
    def __init__(self, config: ConfigDict) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.to_qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias = False)
        self.o = nn.Linear(self.n_embd, self.n_embd, bias = False)
        self.to_qkv.g_scale = self.n_embd**(-1)
        self.o.g_scale = self.n_embd**(-1)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.power = 0.5 * (1. + config.s) # s factor in Sho's setting for MuP
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.to_qkv(x).split(self.n_embd, dim = 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, dim_h)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / k.size(-1)**self.power
        att = F.softmax(att, dim = -1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.o(y))
        return y
    

## MLP block for Transformer
class MLP(nn.Module):

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.wide_factor * config.n_embd, bias = False)
        self.c_fc.g_scale = config.n_embd**(-1)
        if config.act_name == 'gelu':
            self.act_fn = nn.GELU(approximate = 'tanh')
        elif config.act_name == 'relu':
            self.act_fn = nn.ReLU()
        self.c_proj = nn.Linear(config.wide_factor * config.n_embd, config.n_embd, bias = False)
        self.c_proj.g_scale = (config.wide_factor * config.n_embd)**(-1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act_fn(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x