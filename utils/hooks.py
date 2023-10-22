import torch
import torch.nn as nn
from torch import Tensor


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