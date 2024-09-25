import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
import os
import re
import sys
import logging
import torchvision
import torchvision.transforms as T

import node_utils as nut

# data_obj = dut.parse_datasets(validation_file, sim_file, param_file, args, device)
def collate_fn_wl(batch, input_dim, device):
    data = torch.zeros([len(batch), 1, batch[0][0].shape[2]]).to(device)
    target = torch.zeros([len(batch), batch[0][0].shape[1], input_dim]).to(device)
    
    for b, (snap, ts) in enumerate(batch):
        data[b, 0, :] = snap[:, 0, :]
        target[b, :, :] = snap[:, :, :input_dim]
        
        data_ts = ts[:1]
        
    data_dict = {"observed_data": data.permute(1, 0, 2),
                "observed_tp": data_ts,
                "data_to_predict": target,
                "tp_to_predict": ts}
    
    return data_dict

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def parse_datasets(raw_datafile, sim_datafile, param_datafile, args, device):
    """
    Take raw_data i.e. WL tensors, sim data and flux rope parameter information to construct dataloader objects for train, val and test sets depending on the chosen training mode i.e. 2-4 sims.
    """
    data = np.load(raw_datafile)
    files = np.load(sim_datafile)
    params = np.loadtxt(param_datafile)

    all_sim_ids = np.linspace(1, 24, 24, dtype=int)
    # extract simIDs from sims via regex matching
    # sims are of the form:
    # 'run001_AWSoM2T_restart_run004_AWSoM2T', 'run002_AWSoM2T_restart_run004_AWSoM2T' and so on. 
    # Get 001, 002 etc. from these strings
    successful_sims = np.array([int(re.search(r'\d+', f).group()) for f in files])


    # change simIDs below depending on event under consideration.
    if args["test_mode"]=="t2p1":
        train_idx = np.array([18, 22])
        val_idx = np.array([20])
        test_idx = np.array([14])
    elif args["test_mode"]=="t4p1":
        train_idx = np.array([18, 19, 21, 22])
        val_idx = np.array([20])
        test_idx = np.array([14])
    elif args["test_mode"]=="t5p4":
        train_idx = np.array([11, 12, 13, 18, 19, 20])
        val_idx = np.array([14, 15, 16, 21])
        test_idx = np.array([17, 22])
    elif args["test_mode"]=="t12v4p7":
        train_idx = np.array([0, 1, 2, 3, 7, 10, 11, 12, 13, 18, 19, 22])
        val_idx = np.array([4, 8, 14, 21])
        test_idx = np.array([5, 6, 9, 15, 16, 17, 20])
    
    # Load and scale params
    train_params_raw = params[successful_sims[train_idx] - 1, :]/args["param_scaling"]
    val_params_raw = params[successful_sims[val_idx] - 1, :]/args["param_scaling"]
    test_params_raw = params[successful_sims[test_idx] - 1, :]/args["param_scaling"]

    all_timesteps = np.linspace(2, 180, 90, dtype=int)
    tMinIdx = 46 # hardcoding for now, change later to accomodate variable length sequences
    tMaxIdx = len(all_timesteps) - 1
    nTimesteps = tMaxIdx - tMinIdx + 1

    # scale time appropriately
    tt = np.linspace(0, 1, tMaxIdx - tMinIdx + 1)
    tpredict = torch.Tensor(tt).to(device)

    # remove last 2 rows in vertical dimension as they also captured the edge of the end of the FOV (zero valued pixels)
    train_data_raw = data[:126, :, :, train_idx]
    val_data_raw = data[:126, :, :, val_idx]
    test_data_raw = data[:126, :, :, test_idx]

    # resize raw data to 32x128 tensors and transpose so that the order is (n, t, x, y)
    resize_dims = args["resize_dims"]
    input_dim = resize_dims[0] * resize_dims[1]

    train_data_resized = T.Resize(size=resize_dims,
                                  antialias=True
                                  )(torch.Tensor(train_data_raw.transpose(3, 2, 0, 1)))
    
    val_data_resized = T.Resize(size=resize_dims,
                                antialias=True
                                )(torch.Tensor(val_data_raw.transpose(3, 2, 0, 1)))
    
    test_data_resized = T.Resize(size=resize_dims,
                                antialias=True
                                )(torch.Tensor(test_data_raw.transpose(3, 2, 0, 1)))
    
    # calculate max of train data to scale all data
    max_train = torch.max(train_data_resized)
    min_train = torch.min(train_data_resized)

    # scale all data and index from tMinIdx to tMaxIdx
    train_data = (train_data_resized[:, tMinIdx:(tMaxIdx + 1), :, :] - min_train)/(max_train - min_train)
    val_data = (val_data_resized[:, tMinIdx:(tMaxIdx + 1), :, :] - min_train)/(max_train - min_train)
    test_data = (test_data_resized[:, tMinIdx:(tMaxIdx + 1), :, :] - min_train)/(max_train - min_train)


    # reshape above data to have dimensions (n, ntimesteps, nx*ny)
    train_snaps = train_data.reshape((train_data.shape[0], train_data.shape[1], -1))
    val_snaps = val_data.reshape((val_data.shape[0], val_data.shape[1], -1))
    test_snaps = test_data.reshape((test_data.shape[0], test_data.shape[1], -1))

    # repeat parameter values up to number of timesteps
    train_params_ts = np.repeat(train_params_raw.reshape((train_params_raw.shape[0], train_params_raw.shape[1], 1)), nTimesteps, axis=2)
    val_params_ts = np.repeat(val_params_raw.reshape((val_params_raw.shape[0], val_params_raw.shape[1], 1)), nTimesteps, axis=2)
    test_params_ts = np.repeat(test_params_raw.reshape((test_params_raw.shape[0], test_params_raw.shape[1], 1)), nTimesteps, axis=2)

    # concatenate snaps and params
    train_all = torch.cat((train_snaps, torch.Tensor(np.transpose(train_params_ts, (0, 2, 1)))), 2)
    val_all = torch.cat((val_snaps, torch.Tensor(np.transpose(val_params_ts, (0, 2, 1)))), 2)
    test_all = torch.cat((test_snaps, torch.Tensor(np.transpose(test_params_ts, (0, 2, 1)))), 2)

    # create lists
    train_dataset = [(train_all[i:(i+1), :, :], tpredict) for i in range(train_all.shape[0])]
    val_dataset = [(val_all[i:(i+1), :, :], tpredict) for i in range(val_all.shape[0])]
    test_dataset = [(test_all[i:(i+1), :, :], tpredict) for i in range(test_all.shape[0])]

    # create dataloaders
    bs = args["batch_size"] if args["batch_size"] < len(train_dataset) else len(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=bs,
                                #   batch_size=args["batch_size"],
                            #   batch_size = len(train_dataset), 
                              shuffle=True,
                            #   num_workers=1, 
                              collate_fn = lambda batch: collate_fn_wl(batch, input_dim, device))
    

    val_dataloader = DataLoader(val_dataset,
                            batch_size=bs,
                            # batch_size=args["batch_size"],
                            # batch_size = len(val_dataset),
                            shuffle=False,
                            # num_workers=1,
                            collate_fn = lambda batch: collate_fn_wl(batch, input_dim, device))
    
    test_dataloader = DataLoader(test_dataset,
                            batch_size=bs,
                            # batch_size=args["batch_size"],
                            # batch_size = len(test_dataset),
                            shuffle=False,
                            # num_workers=1,
                            collate_fn = lambda batch: collate_fn_wl(batch, input_dim, device))
    
    data_obj = {"train_dataloader": inf_generator(train_dataloader),
                "val_dataloader": inf_generator(val_dataloader),
                "test_dataloader": inf_generator(test_dataloader),
                "train_params": train_params_raw,
                "val_params": val_params_raw,
                "test_params": test_params_raw,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "test_dataset": test_dataset,
                "tpredict": tpredict,
                "max_train": max_train,
                "min_train": min_train,
                "input_dim": input_dim,
                "nTimesteps": nTimesteps
                }
    
    return data_obj

def get_next_batch(dataloader, device=torch.device("cuda:0")):
    return dataloader.__next__()


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger