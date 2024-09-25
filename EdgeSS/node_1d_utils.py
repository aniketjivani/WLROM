import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import edge_utils as edut


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.losses = []
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
        self.losses.append(self.avg)
        
        
        
# def getDataForSim(edge_data_matrix, r_data_matrix, sim_data, sid):
#     """
#     Take in a randomly chosen sim from the training set and return the following:
#     y0_train_torch
#     y_train_torch
#     i.e. IC and data in torch tensor format on Device
#     t_train_torch
#     and correct sim_index from sim_data
#     """
    
#     sim_index = np.argwhere(sim_data == sid)[0][0]
    
#     r_sim = r_data_matrix[:, :, sim_index]
    
#     tMinIdx, tMin, tMaxIdx, tMax = edut.getTMinTMax(edge_data_matrix, simIdx=sim_index)
    
#     r_sim_valid = r_sim[tMinIdx:(tMaxIdx+1), :]
#     valid_times = np.arange(tMin, tMax + 2, step=2)
    
#     tTrainEnd = tMin + np.floor((2/3)*(tMax - tMin))
    
    
#     trainEndIdx = np.argmin(np.abs(valid_times - tTrainEnd))
#     #     trainEndIdx = np.argwhere(valid_times == tTrainEnd)[0][0]
    
#     tTrain = valid_times[:(trainEndIdx + 1)]
    
#     tTest = valid_times[(trainEndIdx + 1):]
    
#     tTrainScaled = (tTrain - tMin) / (tMax - tMin)
#     tTestScaled = (tTest - tMin) / (tMax - tMin)
    
#     tAllScaled = (valid_times - tMin) / (tMax - tMin)
    
#     y0_train_orig = r_sim_valid[0, :]
#     y0_train_torch = torch.from_numpy(np.float32(y0_train_orig))
#     y0_train_torch = y0_train_torch.reshape((1, len(y0_train_torch))).to(device)
    
    
#     y_train_orig = r_sim_valid[:(trainEndIdx + 1), :]
#     y_train_torch = torch.from_numpy(np.expand_dims(np.float32(y_train_orig), axis=1)).to(device)
    
#     y_full_torch = torch.from_numpy(np.expand_dims(np.float32(r_sim_valid), axis=1)).to(device)
    
#     t_train_torch = torch.tensor(np.float32(tTrainScaled)).to(device)
#     t_scaled_torch = torch.tensor(np.float32(tAllScaled)).to(device)
    
#     return y0_train_torch, y_train_torch, y_full_torch, t_train_torch, t_scaled_torch, sim_index


# def get_batch(torch_train_data, torch_train_time, batch_time=5, batch_size=10):
#     s = torch.from_numpy(np.random.choice(np.arange(len(torch_train_time) - batch_time, dtype=np.int64),
#                                           batch_size,
#                                           replace=False))
#     batch_y0 = torch_train_data[s]  # (M, D)
#     batch_t = torch.zeros((batch_size, batch_time))
#     for i in range(batch_size):
#         batch_t[i, :] = torch_train_time[s[i]:(s[i] + batch_time)]
        
#     batch_y = torch.stack([torch_train_data[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
#     return batch_y0.to(device), batch_t.to(device), batch_y.to(device)