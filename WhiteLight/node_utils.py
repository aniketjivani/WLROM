import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
import numpy as np
import os
import sys

def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def df_to_tensor(x: pd.DataFrame) -> torch:
    return torch.from_numpy(x.values).to(get_device())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr
        
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)

def get_ckpt_model(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    # # ckpt_args = checkpt['args']

    model_dict = model.state_dict()
    state_dict = checkpt['model']

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict) 
    # 3. load the new state dict
    model.load_state_dict(state_dict)

    model.to(device)

# Write conventional loss function without time-reversal component for ODENet. Use smooth L1 loss instead of MAE.
def oden_loss(y_true_f, y_pred_f, loss='smooth_l1'):
    """
    Conventional loss function for ODENet
    """
    # y_true and y_pred are both (batch_size, seq_len, input_dim)
    # y_true is the target data, y_pred is the model output

    # check that y_true and y_pred have the same shape
    assert y_true_f.shape == y_pred_f.shape
    
    if loss == 'smooth_l1':
        return torch.nn.SmoothL1Loss()(y_true_f, y_pred_f)
    elif loss == 'mse':
        return torch.nn.MSELoss()(y_true_f, y_pred_f)
    elif loss == 'mae':
        return torch.nn.L1Loss()(y_true_f, y_pred_f)
    else:
        raise ValueError('Invalid loss function')

# Write loss function for time-reversal symmetric ODE network (TRS-ODEN) based on https://github.com/inhuh/trs-oden/blob/main/train_duffing.py (original file uses Keras, convert to equivalent PyTorch code)
def trs_oden_loss(y_true_f, y_pred_f, y_pred_rev_f, lambda_trs=10, loss='smooth_l1'):
    """Loss function for time-reversal symmetric ODE network (TRS-ODEN) based on Huh et al. (2020): https://arxiv.org/abs/2007.11362
    """
    # y_true and y_pred are both (batch_size, seq_len, input_dim)
    # y_true is the target data, y_pred is the model output
    # lambda_trs is the weight for the time-reversal symmetric loss term
    
    # check that y_true and y_pred have the same shape
    assert y_true_f.shape == y_pred_f.shape

    # Compute the conventional loss term
    if loss == 'smooth_l1':
        conventional_loss = torch.nn.SmoothL1Loss()(y_true_f, y_pred_f)
    elif loss == 'mse':
        conventional_loss = torch.nn.MSELoss()(y_true_f, y_pred_f)
    elif loss == 'mae':
        conventional_loss = torch.nn.L1Loss()(y_true_f, y_pred_f)
    else:
        raise ValueError('Invalid loss function')
    
    # Compute the time-reversal symmetric loss term
    # The time-reversal symmetric loss term calculates loss between R(ODESolve(f, x, dt) and ODESolve(f, R(x), -dt)) where R is the reversing operator.
    # The reversing operator R is defined as R(x[1:n]) = x[n:1] where n is the length of the sequence.
    y_true_rev_f = torch.flip(y_true_f, [1])
    if loss == 'smooth_l1':
        trs_loss = torch.nn.SmoothL1Loss()(y_true_rev_f, y_pred_rev_f)
    elif loss == 'mse':
        trs_loss = torch.nn.MSELoss()(y_true_rev_f, y_pred_rev_f)
    elif loss == 'mae':
        trs_loss = torch.nn.L1Loss()(y_true_rev_f, y_pred_rev_f)
    else:
        raise ValueError('Invalid loss function')

    # add the two loss terms
    return conventional_loss + lambda_trs * trs_loss


# write function to create encoder-pnode-decoder model and transfer it to appropriate device. First define encoder, decoder, PNODE separately and take care of dimensions. Then combine them into a single model.
            
def init_network_weights_xavier_normal(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, val=0)


# Define Encoder-Decoder PNODE Now
class ODENet(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 param_dim,
                 device,
                 n_layers=2,
                 n_units=50,
                 n_units_q=100,
                 nonlinear=nn.ELU,
                 quadratic=False):
    
        super(ODENet, self).__init__()
        layers = [nn.Linear(latent_dim + param_dim, n_units)]
        for i in range(n_layers - 1):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, latent_dim))  

        q_dim = latent_dim * (latent_dim + 1) // 2

        layers_q = [nn.Linear(q_dim + param_dim, n_units_q)]
        for i in range(n_layers - 1):
            layers_q.append(nonlinear())
            layers_q.append(nn.Linear(n_units_q, n_units_q))

        layers_q.append(nonlinear())
        layers_q.append(nn.Linear(n_units_q, latent_dim))

        odenet = nn.Sequential(*layers)
        odenet_q = nn.Sequential(*layers_q)

        init_network_weights_xavier_normal(odenet)

        self.odenet = odenet
        self.odenet_q = odenet_q
        self.latent_dim = latent_dim
        self.quadratic = quadratic
        self.device = device

    def quadratic_y(self, y):
        # take lower triangular entries of outer product of y with itself.
        y_np = torch.squeeze(y[:, :, :self.latent_dim], axis=0)
        y_np_quad = torch.zeros((1, y_np.shape[0], int(y_np.shape[1] * (y_np.shape[1] + 1)/2)))
        for i in range(y_np.shape[0]):
            y_np_quad[0, i, :] = torch.outer(y_np[i, :], y_np[i, :])[np.tril_indices(y_np.shape[1])]

        return torch.cat((torch.Tensor(y_np_quad).to(self.device), 
                        torch.zeros_like(y[:, :, self.latent_dim:])),
                        -1).to(self.device)

    def forward(self, t, y):
        if self.quadratic==True:
            output = torch.cat((self.odenet(y),
                    torch.zeros_like(y[:, :, self.latent_dim:])), 
                    -1) + torch.cat((self.odenet_q(self.quadratic_y(y)),
                                     torch.zeros_like(y[:, :, self.latent_dim:])),
                                     -1)
        else:
            output = torch.cat((self.odenet(y),
                    torch.zeros_like(y[:, :, self.latent_dim:])), 
                    -1)
        return output


class PNODE_Conv(nn.Module):
    def __init__(self, 
                 input_dim,
                 latent_dim,
                 param_dim,
                 device,
                 n_layers=2, 
                 n_units=50,
                 nonlinear=nn.ELU,
                 quadratic=False):
        super(PNODE_Conv, self).__init__()
        
        encoder = nn.Sequential(nn.ZeroPad2d((7,8,7,8)),
                        nn.Conv2d(1,4,16,stride=(2,2),padding=(0,0)),
                        nn.ELU(),
                        nn.ZeroPad2d((3,4,3,4)),
                        nn.Conv2d(4,8,8,stride=(2,2),padding=(0,0)),
                        nn.ELU(),
                        nn.ZeroPad2d((1,2,1,2)),
                        nn.Conv2d(8,16,4,stride=(2,2),padding=(0,0)),
                        nn.ELU(),
                        nn.ZeroPad2d((0,1,0,1)),
                        nn.Conv2d(16,32,2,stride=(2,2),padding=(0,0)),
                        nn.ELU(),
                        nn.ZeroPad2d((0,0,0,0)),
                        nn.Conv2d(32,64,1,stride=(2,2),padding=(0,0)),
                        nn.ELU(),
                        nn.Flatten(),
                        nn.Linear(256,latent_dim), 
                        nn.ELU()) 
        

        
        decoder_mlp = nn.Sequential(nn.Linear(latent_dim, 256),
                                    nn.ELU())
        
        decoder_conv = nn.Sequential(nn.ConvTranspose2d(64,32,(4,4),stride=(2,2),padding=(1,1)),
                                     nn.ELU(),
                                     nn.ConvTranspose2d(32,16,(4,4),stride=(2,2),padding=(1,1)),
                                     nn.ELU(),
                                     nn.ConvTranspose2d(16,8,(8,8),stride=(2,2),padding=(3,3)),
                                     nn.ELU(),
                                     nn.ConvTranspose2d(8,4,(16,16),stride=(2,2),padding=(7,7)),
                                     nn.ELU(),
                                     nn.ConvTranspose2d(4,1,(16,16),stride=(2,2),padding=(7,7)),
                                     nn.ELU())        
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder=encoder
        self.pnode=ODENet(latent_dim, 
                          param_dim,
                          device,
                          n_layers=n_layers,
                          n_units=n_units,
                          quadratic=quadratic).to(device)
        
        if quadratic == True:
            print("Using lin-quad PNODE")
        else:
            print("Using regular PNODE")

        self.decoder_mlp = decoder_mlp
        self.decoder_conv = decoder_conv
        
        
        init_network_weights_xavier_normal(encoder)
        init_network_weights_xavier_normal(decoder_mlp)
        init_network_weights_xavier_normal(decoder_conv)
        
        
    def forward(self, t, y):
        init_state = y[:, :, :self.input_dim]
        nbatch, ntraj, nseq = init_state.shape
        init_state = init_state.reshape((nbatch * ntraj, 1, 32, 128))
        init_latent = self.encoder(init_state).reshape((nbatch, ntraj, self.latent_dim))
        init_latent = torch.cat((init_latent, y[:, :, self.input_dim:]),-1)
        latent_states = odeint(self.pnode, init_latent, t)
        ls_init = latent_states[:, :, :, :self.latent_dim]
        ntraj2, nbatch2, nseq2, _  = ls_init.shape
        latent_mlp = self.decoder_mlp(ls_init)
        decoder_features = latent_mlp.reshape((-1, 64, 1, 4))
        pred_sol = self.decoder_conv(decoder_features)
        pred_sol = pred_sol.reshape((ntraj2, nbatch2, nseq2, 4096))
        
        return pred_sol

# Credit for scheduler and early stopping pipeline: Hongfan Chen
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0,
                 path="/Users/ajivani/WLROM_new/model_stopping/model_state.pt",
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, optimizer, epoch, args):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, args)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, args):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.2f} --> {val_loss:.2f}).  Saving model ...')

        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': args,
                    }, 
                    self.path)
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
