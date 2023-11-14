# November 13, 2023: First pass for basic conformal prediction assuming exchangeable initial conditions and parameters.
import numpy as np
import os
import pandas as pd

# load model, generate scores for each testing point
import torch
import torch.nn as nn
import torch.optim as optim

import edge_utils as edut
import node_1d_utils as nut

import argparse


# run in Terminal first: pip install --user torchdiffeq
adjoint=True
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


parser = argparse.ArgumentParser("PNODE_CP")

parser.add_argument('--save',
                    type=str,
                    default='experiments/',
                    help="Path for save checkpoints")
parser.add_argument('--load', 
                    type=str, 
                    # default=None,
                    # default=96778,
                    default = 39492,
                    help="ID of the experiment to load for evaluation. If None, run a new experiment.")

parser.add_argument('--node-layers', 
                    type=int, 
                    default=3, 
                    help="number of layers in NODE")

parser.add_argument('-u', '--units', 
                    type=int, 
                    default=70,
                    # default=100, 
                    help="Number of units per layer in ODE func")

parser.add_argument('-ds',
                    type=int,
                    default=4,
                    help="Coarsening factor for position angles")

parser.add_argument('--nParamsToUse',
                    type=int,
                    default=10,
                    help="Number of CME params to use")


parser.add_argument('--alpha-cov',
                    type=float,
                    default=0.9,
                    help="Significance level for coverage")

args = parser.parse_args(args=()) # for running in interactive window
# args = parser.parse_args() # for running in terminal

vars(args)

## Get Sim IDs

ed_2161, sd_2161 = edut.load_edge_data_blobfree(2161)
# crude downsample for now
ed_2161 = ed_2161[:, ::args.ds, :]

sims_to_remove = np.array([33, 39, 63, 73, 113, 128, 131, 142, 193, 218, 253, 264, 273, 312, 313, 324])
sd_modified = np.setdiff1d(sd_2161, sims_to_remove)

from numpy.random import Generator, PCG64
# rng = Generator(PCG64())
rng = np.random.default_rng(seed=202310)

nTrain = int(np.floor(0.7 * len(sd_modified)))
nCalib = int(np.floor(0.15 * len(sd_modified)))
nTest = len(sd_modified) - nTrain - nCalib

print(nTrain, nCalib, nTest)

sd_train = np.sort(rng.choice(sd_modified, nTrain, replace=False))
sd_calib = np.sort(rng.choice(np.setdiff1d(sd_modified, sd_train), nCalib, replace=False))
sd_test = np.setdiff1d(sd_modified, np.sort(np.concatenate((sd_train, sd_calib), axis=0)))


# process:

# 1. split data into train, calibrate, test
# 2. Train model on time series from all training sets
# 3. Compute non-conformity scores on calibration set
# 4. find threshold q
# 5. Form prediction intervals for points on test set.

class PNODE(nn.Module):
    def __init__(self, input_dim, param_dim, 
                 n_layers=1, 
                 n_units=90,
                 nonlinear=nn.ELU):
        super(PNODE, self).__init__()

        layers = [nn.Linear(input_dim + param_dim, n_units)]
        
        for i in range(n_layers - 1):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
            
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, input_dim))
        
        self.net1 = nn.Sequential(*layers)
        
        # for m in self.net1.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        
        output = torch.cat((self.net1(y),
                    torch.zeros_like(y[:, input_dim:])), 
                    -1)
        
        return output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print statement as Device = ...
print("Device = ",
        device)

input_dim = 40
param_dim = 10

model = PNODE(input_dim,
              param_dim,
              n_layers=args.node_layers,
              n_units=args.units,
              nonlinear=nn.ELU).to(device)
print(model)

if args.load is not None:
    experimentID = args.load
else:
    experimentID = int(SystemRandom().random()*100000)
print(experimentID)

ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
print(ckpt_path)

if args.load is not None:
    edut.get_ckpt_model(ckpt_path, model, device)  

alpha_cov = args.alpha_cov

# %%

ed_2161, sd_2161 = edut.load_edge_data_blobfree(2161)
# crude downsample for now
ed_2161 = ed_2161[:, ::args.ds, :]

nTimes, nTheta_2161, nSims_2161 = ed_2161.shape
nTimes, nTheta_2161, nSims_2161

theta_s_2161 = np.linspace(0, 360, 512)[160] + 1.2 * 180 - 360
theta_e_2161 = np.linspace(0, 360, 512)[320] + 1.2 * 180 - 360

theta_grid = np.linspace(np.ceil(theta_s_2161), np.ceil(theta_e_2161), nTheta_2161 * args.ds)[::args.ds]

# %% 

# define augmented_r and other necessary normalizations.
def getRValuesAllSims(edge_data_matrix, sample_freq=2):
    """
    Return r values for all sims at once so we don't lose time in training processing r values repeatedly
    """
    r_data_matrix = np.zeros(edge_data_matrix.shape)
    nsims = edge_data_matrix.shape[2]
    for i in range(nsims):
        r_vals, theta_vals = edut.getRValues(edge_data_matrix, simIdx=i, minStartIdx=0, sample_freq=sample_freq)
        r_data_matrix[:, :, i] = r_vals

    return r_data_matrix

rd_2161 = getRValuesAllSims(ed_2161, sample_freq=4)


# %%
modified_sd_train_idx = np.array([np.where(sd_modified == i)[0][0] for i in sd_train])

orig_sd_train_idx = np.array([np.where(sd_2161 == i)[0][0] for i in sd_train])
orig_sd_test_idx = np.array([np.where(sd_2161 == i)[0][0] for i in sd_test])
orig_sd_calib_idx = np.array([np.where(sd_2161 == i)[0][0] for i in sd_calib])

# %%
y_train = rd_2161[:, :, orig_sd_train_idx]
y_test = rd_2161[:, :, orig_sd_test_idx]

y_train.shape, y_test.shape

y_calib = rd_2161[:, :, orig_sd_calib_idx]
y_calib.shape

# %%
# LOOP TO EXTRACT DT, TMIN, TMAX VALUES. BASED ON THESE, store min and max of training
# data and use that to normalize train and test sets.

tMinTrain = []
tMaxTrain = []

tMinTrainIdx=[]
tMaxTrainIdx=[]

dtTrain = []
yMinTrain = []
yMaxTrain = []

tMinTest = []
tMaxTest = []
dtTest = []

tMinCalib = []
tMaxCalib = []
dtCalib = []

for sidx in orig_sd_train_idx:
    
    r_sim = rd_2161[:, :, sidx]
    
    tMinIdx, tMin, tMaxIdx, tMax = edut.getTMinTMax(ed_2161, simIdx = sidx)

    r_sim_valid = r_sim[tMinIdx:(tMaxIdx + 1), :]
    
    tMinTrain.append(tMin)
    tMaxTrain.append(tMax)

    tMinTrainIdx.append(tMinIdx)
    tMaxTrainIdx.append(tMaxIdx)

    
    yMinTrain.append(r_sim_valid.min())
    yMaxTrain.append(r_sim_valid.max())
    
    tAllScaled = (np.arange(tMin, tMax + 2, step=2) - tMin) / (tMax - tMin)
    
    dtTrain.append(tAllScaled[1] - tAllScaled[0])
    
    
for sidx in orig_sd_test_idx:
    
    r_sim = rd_2161[:, :, sidx]
    
    tMinIdx, tMin, tMaxIdx, tMax = edut.getTMinTMax(ed_2161, simIdx = sidx)

    r_sim_valid = r_sim[tMinIdx:(tMaxIdx + 1), :]
    
    tMinTest.append(tMin)
    tMaxTest.append(tMax)
        
    tAllScaled = (np.arange(tMin, tMax + 2, step=2) - tMin) / (tMax - tMin)
    
    dtTest.append(tAllScaled[1] - tAllScaled[0])
    
    
    
for sidx in orig_sd_calib_idx:    
    r_sim = rd_2161[:, :, sidx]
    
    tMinIdx, tMin, tMaxIdx, tMax = edut.getTMinTMax(ed_2161, simIdx = sidx)

    r_sim_valid = r_sim[tMinIdx:(tMaxIdx + 1), :]
    
    tMinCalib.append(tMin)
    tMaxCalib.append(tMax)
        
    tAllScaled = (np.arange(tMin, tMax + 2, step=2) - tMin) / (tMax - tMin)
    
    dtCalib.append(tAllScaled[1] - tAllScaled[0])

# %%
# NOW NORMALIZE YTRAIN AND YTEST
yMinTrainAll = np.array(yMinTrain).min()
yMaxTrainAll = np.array(yMaxTrain).max()

print(yMinTrainAll, yMaxTrainAll)

# %%
y_train_normalized = (y_train - yMinTrainAll) / (yMaxTrainAll - yMinTrainAll)
y_test_normalized = (y_test - yMinTrainAll) / (yMaxTrainAll - yMinTrainAll)

# %%
y_calib_normalized = (y_calib - yMinTrainAll) / (yMaxTrainAll - yMinTrainAll)

# y_train_normalized.shape
# %%
cme_params_norm = pd.read_csv("./CMEParams2161_Scaled.csv", index_col=0)
cme_params_norm


cme_params_to_augment = cme_params_norm.to_numpy()
cme_params_to_augment.shape


cme_params_to_augment_final = np.zeros((cme_params_to_augment.shape[0],
                                       cme_params_to_augment.shape[1] + 1))
cme_params_to_augment_final[:, :(cme_params_to_augment.shape[1])] = cme_params_to_augment

if args.nParamsToUse == 2:
    param_dim = args.nParamsToUse
else:
    param_dim = cme_params_to_augment_final.shape[1]


cme_params_to_augment_final[orig_sd_train_idx, cme_params_to_augment.shape[1]] = dtTrain
cme_params_to_augment_final[orig_sd_test_idx, cme_params_to_augment.shape[1]] = dtTest
cme_params_to_augment_final[orig_sd_calib_idx, cme_params_to_augment.shape[1]] = dtCalib

# param_dim = cme_params_to_augment_final.shape[1]


input_dim = rd_2161.shape[1]
input_dim, param_dim

augmented_r = np.zeros((rd_2161.shape[0], input_dim + param_dim, rd_2161.shape[2]))
augmented_r[:, :input_dim, orig_sd_train_idx] = y_train_normalized
augmented_r[:, :input_dim, orig_sd_test_idx] = y_test_normalized
augmented_r[:, :input_dim, orig_sd_calib_idx] = y_calib_normalized
for iii in range(rd_2161.shape[2]):
    if args.nParamsToUse == 2:
        augmented_r[:, (input_dim):(input_dim + args.nParamsToUse), iii] = cme_params_to_augment_final[iii, [0, 4]]
    else:
        augmented_r[:, (input_dim):, iii] = cme_params_to_augment_final[iii, :]

aug_y_train = augmented_r[:, :, orig_sd_train_idx]
aug_y_test = augmented_r[:, :, orig_sd_test_idx]

aug_y_train.shape, aug_y_test.shape

# %%
# define get data for sim function
def get_data_for_sim(sidx, device="cpu"):
    """
    Supply sidx from either orig_sd_train_idx or orig_sd_test_idx
    Based on that, index augmented r dataset, and return relevant training data
    as well as training time.
    """
    tMinIdx, tMin, tMaxIdx, tMax = edut.getTMinTMax(ed_2161, simIdx = sidx)
    
    valid_times = np.arange(tMin, tMax + 2, step=2)
        
    tAllScaled = (valid_times - tMin) / (tMax - tMin)
    
    r_sim = augmented_r[tMinIdx:(tMaxIdx + 1), :, sidx]
    
    y0_train_torch = torch.from_numpy(np.float32(r_sim[0, :])).reshape((1, len(r_sim[0, :]))).to(device)
    
    t_train_torch = torch.tensor(np.float32(tAllScaled)).to(device)
    
    y_train_torch = torch.from_numpy(np.float32(r_sim)).to(device)
    

    return y0_train_torch, t_train_torch, y_train_torch


orig_sd_calib_idx = np.array([np.where(sd_2161 == i)[0][0] for i in sd_calib])

scores_val = []

with torch.no_grad():

    # yMinTrainAll, yMaxTrainAll
    
    sim_ids_all_calib = np.array([i for i in range(len(orig_sd_calib_idx))])
    
    for sim_id in sim_ids_all_calib:
    
        _, tt_data, ytt_data = get_data_for_sim(orig_sd_calib_idx[sim_id], device=device)
    
        pred_y_calib = torch.squeeze(odeint(model,
                                        ytt_data[[0], :],
                                        tt_data)) * (yMaxTrainAll - yMinTrainAll) + yMinTrainAll
    
        scores_val.append(torch.mean(torch.abs(pred_y_calib - (ytt_data * (yMaxTrainAll - yMinTrainAll) + yMinTrainAll))).item())
        # print(sim_id)
        print(sd_2161[orig_sd_calib_idx[sim_id]])


# COPIED from notebook, can ignore
# calibration_scores = np.array([0.506319522857666,
#  1.0012081861495972,
#  1.3690389394760132,
#  1.6391355991363525,
#  0.3319915235042572,
#  0.2783850133419037,
#  2.4236207008361816,
#  3.5876212120056152,
#  0.5302484035491943,
#  0.6874138116836548,
#  1.3181757926940918,
#  1.591491937637329,
#  0.6319116353988647,
#  1.4581226110458374,
#  2.415046453475952,
#  2.0321691036224365,
#  5.184138774871826,
#  0.9343780875205994,
#  4.962402820587158,
#  1.1896040439605713,
#  1.8247554302215576,
#  0.6201084852218628,
#  1.7047542333602905,
#  0.23206205666065216,
#  0.4924982786178589,
#  1.7790911197662354,
#  1.2335559129714966,
#  0.23391485214233398,
#  1.5346965789794922,
#  1.8672685623168945,
#  1.6008422374725342,
#  2.2860732078552246,
#  1.860539197921753,
#  3.269237518310547,
#  0.307586669921875,
#  3.1448137760162354,
#  2.5793702602386475,
#  0.9169696569442749,
#  0.42950841784477234])

# %%

scores_quant = np.quantile(scores_val, q=alpha_cov)
print(scores_quant)
# %%

# now plot test data with prediction interval width - we have finite calibration set size effects because the coverage of CP conditional on calibration set is a random quantity and will not be exactly 1 - alpha (just on average) (100 samples - original paper suggests 1000 calibration samples to get  coverage of 1 - alpha + epsilon with probability at least 1 - delta (0.9) where epsilon = 0.1

pred_test = []

with torch.no_grad():

    # yMinTrainAll, yMaxTrainAll
    
    sim_ids_all_test = np.array([i for i in range(len(orig_sd_test_idx))])
    
    for sim_id in sim_ids_all_test:
    
        _, tt_data, ytt_data = get_data_for_sim(orig_sd_test_idx[sim_id], device=device)
    
        pred_y_test = torch.squeeze(odeint(model,
                                        ytt_data[[0], :],
                                        tt_data)) * (yMaxTrainAll - yMinTrainAll) + yMinTrainAll
    
        # print(sim_id)
        print(sd_2161[orig_sd_test_idx[sim_id]])

        tMinIdx, tMin, tMaxIdx, tMax = edut.getTMinTMax(ed_2161, simIdx = orig_sd_test_idx[sim_id])
        r_true = augmented_r[tMinIdx:(tMaxIdx + 1), :input_dim, orig_sd_test_idx[sim_id]] * (yMaxTrainAll - yMinTrainAll) + yMinTrainAll

        edut.plotTrainPredData1ModelFixedInterval(r_true,
                                     pred_y_test.cpu()[:, :input_dim],
                                     ed_2161,
                                     sd_2161,
                                     theta=theta_grid,
                                     simIdx=orig_sd_test_idx[sim_id],
                                     intWidth=scores_quant,
                                     savefig=True,
                                     savedir="./test_sim_figs_fixed_CP")

# %%
