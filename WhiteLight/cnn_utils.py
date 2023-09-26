import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

def getHC(Hin, Win, padding=(1, 1), dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1)):
    """
    Calculate Hout and Wout when using `nn.Conv2d` on an image. Supply Hin, Win, padding, dilation,
    kernel_size and stride to calculate.
    """
    Hout = np.floor(((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]) + 1)
    Wout = np.floor(((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]) + 1)
    return Hout, Wout


def getHCTransposed(Hin, Win, 
                    padding=(1, 1), 
                    dilation=(1, 1), 
                    kernel_size=(3, 3), 
                    stride=(1, 1),
                    output_padding=(0, 0)):
    """
    Calculate Hout and Wout when using `nn.ConvTranspose2d` on an image. Supply Hin, Win, padding, dilation,
    kernel_size and stride to calculate. Also supply output padding but we will keep it at zero so optional.
    """

    Hout = (Hin - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    Wout = (Win - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    return Hout, Wout


def getHCAdaptivePool(Hin, Win, padding=(1, 1), dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1)):
    """
    Calculate Hout and Wout when using `nn.AdaptiveAvgPool` on an image. Supply Hin, Win, padding, dilation,
    kernel_size and stride to calculate.
    """
    Hout = np.floor(((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]) + 1)
    Wout = np.floor(((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]) + 1)
    return Hout, Wout



def count_parameters(model):
    """
    Count number of parameters for instance of an ML model.
    Stolen from: https://github.com/rtqichen/torchdiffeq/blob/84e220ac9ea9367c14933d8f141fc2791034ec88/examples/odenet_mnist.py#L241
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def group_norm(dim):
    return nn.GroupNorm(min(16, dim), dim)


class Flatten(nn.Module):
    """
    Using same flattening as https://github.com/rtqichen/torchdiffeq/blob/84e220ac9ea9367c14933d8f141fc2791034ec88/examples/odenet_mnist.py#L137
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class ODEFuncCNN(nn.Module):
    """
    Define CNN and fully connected layers for passing through input images at some initial conditions and returning a sequence of images according to
    convolution dynamics.
    """
    def __init__(self):
        super(ODEFuncCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, 
                  kernel_size=(3, 3), 
                  stride=(1, 1),
                  padding=(1, 1),
                  dilation=(1, 1))

        self.norm1 = group_norm(32)
        
        self.tanh = nn.Tanh()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, 
                          kernel_size=(4, 4), 
                          stride=(2, 2),
                          padding=(1, 1),
                          dilation=(1, 1))
        
        self.norm2 = group_norm(32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, 
                          kernel_size=(4, 4), 
                          stride=(2, 2),
                          padding=(1, 1),
                          dilation=(1, 1))

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))

        self.flat1 = Flatten()

        self.lin1 = nn.Linear(32, 32 * 128)

    def forward(self, t, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.norm1(out)
        out = self.tanh(out)
        out = self.conv3(out)
        out = self.norm1(out)
        out = self.tanh(out)
        out = self.pool1(out)
        out = self.flat1(out)
        out = self.lin1(out)
        out = out.view(-1, 1, 32, 128)
        
        return out
    
    
    
def get_batch(torch_train_data, torch_train_time, batch_time=5, batch_size=10, device=None):
    """
    Get batches of images for given batch time and batch size.
    """
    s = torch.from_numpy(np.random.choice(np.arange(len(torch_train_time) - batch_time, dtype=np.int64),
                                          batch_size,
                                          replace=False))
    batch_y0 = torch_train_data[s]
    batch_t = torch.zeros((batch_size, batch_time))
    for i in range(batch_size):
        batch_t[i, :] = torch_train_time[s[i]:(s[i] + batch_time)]
        
    batch_y = torch.stack([torch_train_data[s + i] for i in range(batch_time)], dim=0)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)