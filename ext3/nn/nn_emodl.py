from ext3.typing import *
from ext3.core   import EModl, EFuncMgr, EModlClsMgr1, EModlClsMgr2
from .nn_base    import _Add, _Cat, _Identity, _Mean, _Mul, _SplitHalf, _LogSoftmaxFunc, _SigmoidFunc

import torch, collections

__all__ = [
    'flatten', 'permute', 'reshape', 'view', # 'getitem',
    'Add', 'Cat', 'Input', 'Mean', 'Mul', 'SplitHalf', # 'Output',
    'AdaptiveAvgPool2d', 'AvgPool2d', 'BatchNorm2d', 'Conv2d',
    'Dropout', 'Linear', 'MaxPool2d', 'NLLLoss', 'ReLU',
    'LogSoftmax', 'Sigmoid',
    'Module', 'Sequential', 'CrossEntropyLoss',
]

# func(op with 1 input, 1 output, no fp op): wrap with efunc mgr.
# getitem: Callable = EFuncMgr.gen(torch.Tensor.__getitem__) # shfv2.
flatten: Callable = EFuncMgr.gen(torch.flatten)            # sqz.
permute: Callable = EFuncMgr.gen(torch.Tensor.permute)     # shfv2.
reshape: Callable = EFuncMgr.gen(torch.Tensor.reshape)     # shfv2.
view   : Callable = EFuncMgr.gen(torch.Tensor.view)        # shfv2, mblv2, resnet.

# modl(atomic op, new): wrap with emodl mgr1.
class Add      (_Add,       EModl): # shfv2, mblv2.
    pass
class Cat      (_Cat,       EModl): # sqz, shfv2.
    pass
class Input    (_Identity,  EModl): # ALL.
    pass
class Mean     (_Mean,      EModl): # shfv2 (imgnet).
    pass
class Mul      (_Mul,       EModl): # regnet.
    pass
class SplitHalf(_SplitHalf, EModl): # shfv2.
    pass

# modl(atomic op, bwd does not use y): wrap with emodl mgr1.
class AdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d, EModl):
    pass
class AvgPool2d        (torch.nn.AvgPool2d,         EModl):
    pass
class BatchNorm2d      (torch.nn.BatchNorm2d,       EModl):
    pass
class Conv2d           (torch.nn.Conv2d,            EModl):
    pass
class Dropout          (torch.nn.Dropout,           EModl):
    pass
class Linear           (torch.nn.Linear,            EModl):
    pass
class MaxPool2d        (torch.nn.MaxPool2d,         EModl):
    pass
class NLLLoss          (torch.nn.NLLLoss,           EModl):
    pass
class ReLU             (torch.nn.ReLU,              EModl):
    pass

# modl(atomic op, bwd does use y): wrap with emodl mgr2.
class LogSoftmax(torch.nn.LogSoftmax, EModl):
    def __init__(self, *args, **kwargs):
        super(LogSoftmax, self).__init__(*args, **kwargs)
        # remove {fwd,bwd}_hook(). contain them directly in forward().
        self._forward_hooks  = collections.OrderedDict()
        self._backward_hooks = collections.OrderedDict()
    def forward(self, x: TS) -> TS:
        # use forward() (and its implicit backward()) that contain {fwd,bwd}_hook().
        return _LogSoftmaxFunc.apply(x, self)

class Sigmoid(torch.nn.Sigmoid, EModl):
    def __init__(self, *args, **kwargs):
        super(Sigmoid, self).__init__(*args, **kwargs)
        # remove {fwd,bwd}_hook(). contain them directly in forward().
        self._forward_hooks  = collections.OrderedDict()
        self._backward_hooks = collections.OrderedDict()
    def forward(self, x: TS) -> TS:
        # use forward() (and its implicit backward()) that contain {fwd,bwd}_hook().
        return _SigmoidFunc.apply(x, self)

# modl(composite op): do not wrap.
Module     = torch.nn.Module
Sequential = torch.nn.Sequential
class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__(*args, **kwargs)
        self.logsoftmax = LogSoftmax(dim=-1)       # use wrapped op.
        self.nllloss    = NLLLoss(*args, **kwargs) # use wrapped op.

    def forward(self, input, target):
        assert(input.dim() == 2 and target.dim() == 1)
        y = self.logsoftmax(input)
        y = self.nllloss(y, target)
        return y
