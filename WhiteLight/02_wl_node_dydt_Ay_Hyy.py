# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Carrying the same basic concept as notebook 1.
#
#
# the idea in this notebook involves no prior model, just trying to put a feedforward neural network with reduced state model instead of OpInf. 
# More advanced version will not flatten anything and use convolutions.
#
# instead of linear RHS we also put quadratic term.

# %%
import os
import opinf
import scipy.signal
# import cv2
os.getcwd()

# %%
import numpy as np
from numpy import ogrid
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import opinf as op

# %%
import time

# %%
from sunpy.visualization import colormaps as cm

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# %%
adjoint=True

# %%
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
lc3 = cm.cmlist['soholasco3'].reversed()
lc3

# %%
lc3_reg = cm.cmlist['soholasco3']
lc3_reg

# %%
plt.rc("axes.spines", right=True, top=True)
plt.rc("figure", dpi=300,
       # figsize=(9, 3)
       )
plt.rc("font",
       family="serif",
       size=10
       )
plt.rc("legend", edgecolor="none", frameon=True)
plt.rc("text", usetex=False)
# plt.rc("text", usetex=True)

# %% [markdown]
# ### Setup training and testing data

# %%
m = 71

# %%
# %%
t_vec = (np.linspace(40, 180, m) - 40) / 60
t_vec

# %%
# %%
dt = t_vec[1] - t_vec[0]

# %%
nTrainTime = np.argwhere(t_vec == (120 - 40) / 60)[0, 0]

# %%
nTrainTime

# %%
nTrainTimeAll = np.argwhere(np.linspace(2, 180, 90) == 40)[0,0]

# %%
# %%
t_train = t_vec[:(nTrainTime + 1)]
t_train[-1] * 60 + 40

# %%
t_test = t_vec[(nTrainTime + 1):]
t_test

# %%
X_orig = np.load("./CR2161_tDecay2h_Polar_Compressed.npy")

# %%
X = X_orig.reshape((64 * 256, 90, 278))[:, (nTrainTimeAll):, :]

# %%
import netCDF4 as nc
ds = nc.Dataset("./20150315_CR2161_code_stable_tDecay2h.nc")
successfulRuns = ds['runs'].successfulRuns + 30
successfulRuns

# %%
train_test_sims = np.load("./CR2161_AWSoM2T_CME_tDecay2h_Polar_Clusters.npz")
train_sim = train_test_sims["training_id"]
test_sim = train_test_sims["test_id"]
train_sim, test_sim

# %%
tt_sim = np.sort(np.hstack((train_sim, test_sim)))

# %%
X_train_ttrain = X[:, :(nTrainTime + 1), train_sim]
X_train_ttrain.shape

# %%
X_test = X[:, :, test_sim]

# %%
X_train_test_ttrain = X[:, :(nTrainTime + 1), :]

# %%
X_train_ttrain.shape

# %%
rom_1 = op.ContinuousOpInfROM(modelform="AH")
basis_1 = op.pre.PODBasis().fit(X_train_ttrain[:, :, 0], residual_energy=1e-8)
basis_1.r

# %%
reduced_states = basis_1.encode(X_train_ttrain[:, :, 0])

# %%
reduced_states.shape

# %%
reduced_states[1, :]

# %% [markdown]
# ### Construction of NODE Architecture

# %%
device

# %%
t_train

# %%
data_size = t_train.shape[0]
batch_time = 5
batch_size = 10
print("Data size = ", data_size, " Batch time = ", batch_time, " Batch size = ", batch_size)

# %%
t = torch.tensor(np.float32(t_train)).to(device)

# %%
t.dtype

# %%
true_y0 = torch.from_numpy(np.float32(reduced_states[:, 0]))
true_y0 = true_y0.reshape((1, len(true_y0))).to(device)
true_y0.shape

# %%
true_y0.dtype

# %%
reduced_states.shape

# %%
yreduced = np.float32(reduced_states.T)
true_y = torch.from_numpy(np.expand_dims(yreduced, axis=1)).to(device)
true_y.shape

# %%
true_y.dtype


# %% [markdown]
# Need functions for getting batch, getting true x, getting ODE func - where we run the "loss" metric compared to the true simulation.

# %% [markdown]
# The batch is taking IC as values at different times in the training set, and the solution as the values following the IC upto some predefined batch size for time.

# %%
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64),
                                          batch_size, replace=False))
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


# %%
by0, bt, by = get_batch()

# %%
batch_time

# %%
by0.shape, bt.shape, by.shape


# %%
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# %%
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(32, 50),
            nn.Tanh(),
            nn.Linear(50, 32),
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(528, 700),
            nn.Tanh(),
            nn.Linear(700, 528),
        )

        for m in self.net1.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
        for m in self.net2.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    
    def quadratic_y(self, y):
            y_np = y.cpu().numpy().flatten()
            y_np_quad = np.outer(y_np, y_np)[np.tril_indices(len(y_np))]
            y_t_quad = torch.from_numpy(np.float32(y_np_quad))
            y_t_quad = y_t_quad.reshape((1, len(y_np_quad))).to(device)
            return y_t_quad
            
    def forward(self, t, y):
            return self.net1(y) + self.net2(self.quadratic_y(y))

# %%
ii = 0
func = ODEFunc().to(device)
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
end = time.time()

# %%
niters = 2000
test_freq = 5

# %%
func

# %%
func.forward(t, by0)

# %%
time_meter = RunningAverageMeter(0.97)
loss_meter = RunningAverageMeter(0.97)

for itr in range(1, niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()
    pred_y = odeint(func, batch_y0, batch_t).to(device)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()

    time_meter.update(time.time() - end)
    loss_meter.update(loss.item())

    if itr % test_freq == 0:
        with torch.no_grad():
            pred_y = odeint(func, true_y0, t)
            loss = torch.mean(torch.abs(pred_y - true_y))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
#             visualize(true_y, pred_y, func, ii)
            ii += 1

    end = time.time()

# %% [markdown]
# Effectively, we have two things to check over here:
#
# 1. What is the error within the training set, if we provide initial condition y0?
#
# 2. What is the error on the held out part of the data with the same IC?
#
# We could just plot the 2D images and check this qualitatively / with RMSE, with the caveat that RMSE is not really the metric we want to be optimizing for. Also is evaluating error between reduced _as opposed to_ full state appropriate?

# %% [markdown]
# If we tried everything with train times in first sim  i.e. `true_y` was derived from reduced states constructed off `X_train_ttrain[:, :, 0]` then `X_train_ttrain[:, :, 0]` is basically decoded shape for the `true_y`.

# %% [markdown]
# Predicted `y` i.e. `pred_y` needs to be decoded.

# %%
with torch.no_grad():
    pred_y = odeint(func, true_y0, t)

# %%
pred_y.shape

# %%
pred_y_np = pred_y.cpu().numpy()[:, 0, :].T

# %%
pred_y_decoded = basis_1.decode(pred_y_np)

# %%
true_y_decoded = X_train_ttrain[:, :, 0]

# %%
pred_y_decoded.shape

# %% [markdown]
# visualization for if we are able to replicate training properly.

# %%
64 * 256

# %%
# def visualize():
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
axs = ax.ravel()
im0 = axs[0].imshow(true_y_decoded[:, 40].reshape(64, 256),
            origin="lower",
            cmap=lc3_reg)
axs[0].set_title("True")
fig.colorbar(im0, fraction=0.046 * 1/3, pad=0.04)

im1 = axs[1].imshow(pred_y_decoded[:, 40].reshape(64, 256),
                   origin="lower",
                   cmap=lc3_reg)
axs[1].set_title("Predicted")
fig.colorbar(im1, fraction=0.046 * 1/3, pad=0.04)

# %% [markdown]
# by comparison to notebook 1?
#
#

# %% [markdown]
# testing times comparison to notebook 1?

# %%
t_full = torch.tensor(np.float32(t_vec)).to(device)

# %%
with torch.no_grad():
    pred_y_train_sim_fullt = odeint(func, true_y0, t_full)

# %%
pred_y_train_sim_fullt_decoded = basis_1.decode(pred_y_train_sim_fullt.cpu().numpy()[:, 0, :].T)

# %%
pred_y_train_sim_fullt_decoded.shape

# %%
true_y_decoded_fullt = X[:, :, 0]
true_y_decoded_fullt.shape

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
axs = ax.ravel()
im0 = axs[0].imshow(true_y_decoded_fullt[:, 70].reshape(64, 256),
            origin="lower",
            cmap=lc3_reg)
axs[0].set_title("True")
fig.colorbar(im0, fraction=0.046 * 1/3, pad=0.04)

im1 = axs[1].imshow(pred_y_train_sim_fullt_decoded[:, 70].reshape(64, 256),
                   origin="lower",
                   cmap=lc3_reg)
axs[1].set_title("Predicted")
fig.colorbar(im1, fraction=0.046 * 1/3, pad=0.04)

# %% [markdown]
# Try the same example but with prior learnt OpInf model.

# %% [markdown]
# #### Scratch Start

# %%
32 * (32 + 1) / 2

# %%
np.outer([1, 2, 3, 4], [2, 3, 4, 5])

# %%
np.tril_indices(4)

# %%
v_out_flat = np.outer([1, 2, 3, 4], [2, 3, 4, 5])[np.tril_indices(4)]
v_out_flat

# %%
torch.tensor(v_out_flat)

# %%

# %%
vquad_t = torch.from_numpy(np.float32(v_out_flat)).reshape((1, len(v_out_flat))).to(device)

# %%
v_np = vquad_t.cpu().numpy().flatten()

# %%
v_np

# %% [markdown]
# #### Scratch End
