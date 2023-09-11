import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

def getHC(Hin, Win, padding=(1, 1), dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1)):
    """
    Calculate Hout and Wout when using `nn.Conv2d` on an image. Supply Hin, Win, padding, dilation,
    kernel_size and stride to calculate.
    """
    Hout = np.floor(((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]) + 1)
    Wout = np.floor(((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]) + 1)
    return Hout, Wout


def getHCAdaptivePool(Hin, Win, padding=(1, 1), dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1)):
    """
    Calculate Hout and Wout when using `nn.Ada` on an image. Supply Hin, Win, padding, dilation,
    kernel_size and stride to calculate.
    """
    Hout = np.floor(((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]) + 1)
    Wout = np.floor(((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]) + 1)
    return Hout, Wout