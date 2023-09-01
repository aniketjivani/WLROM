import numpy as np
import matplotlib.pyplot as plt

import re
import os

plt.rc("axes.spines", right=True, top=True)
plt.rc("figure", dpi=300, 
       figsize=(9, 3)
      )
plt.rc("font", family="serif")
plt.rc("legend", edgecolor="none", frameon=True)
plt.style.use("dark_background")


def load_edge_data_blobfree(eventID):
    edge_data = np.load("../WhiteLight/edge_data/CR{}_stacked_edge_blobfree.npy".format(eventID))
    simID_data = np.load("../WhiteLight/edge_data/CR{}_SimID4edge.npy".format(eventID))
    
    all_sims = []
    
    for s in simID_data:
        mm = re.search("run\d\d\d_AWSoM2T", s)
        all_sims.append(int(mm.group().split("run")[1].split("_")[0]))
    
    return edge_data, np.array(all_sims)


def Polar_to_Cartesian(edge, start_angle, end_angle, height, width, circles_disk, circles_scope):
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / width)[start_angle:end_angle]
    # Coordinates of disk and telescope center
    circle_x = circles_disk[0]
    circle_y = circles_disk[1]
    # Radius of disk and telescope
    r_disk = circles_disk[2]
    r_scope = circles_scope[2]
    Xp = circle_x + r_disk * np.cos(theta + 1.2 * np.pi)
    Yp = circle_y + r_disk * np.sin(theta + 1.2 * np.pi)
    Xi = circle_x + r_scope * np.cos(theta + 1.2 * np.pi)
    Yi = circle_y + r_scope * np.sin(theta + 1.2 * np.pi)
    r = edge / height
    X = Xp + ( Xi - Xp ) * r
    Y = Yp + ( Yi - Yp ) * r
    return (X,Y)

def getRValues(edge_data_matrix, simIdx=0, minStartIdx=0):
    
    edges_sim = edge_data_matrix[minStartIdx:, :, simIdx]
    
    nTimes, nThetas = edges_sim.shape[0], edges_sim.shape[1]
    
    r_vals = np.zeros((nTimes, nThetas))
    theta_vals = np.zeros((nTimes, nThetas))
    
    for i in range(nTimes):
        xi, yi   = Polar_to_Cartesian(edges_sim[i, :], 
                   start_angle = 160, 
                   end_angle = 320, 
                   height=128, 
                   width=512, 
                   circles_disk=(149,149,19), 
                   circles_scope=(149,149,110))
        
        xi_norm, yi_norm = 64 * (xi/300) - 32, 64 * (yi/300) - 32
        theta_vals[i, :] = np.arctan2(yi_norm, xi_norm)
    
        r_vals[i, :] = np.sqrt(xi_norm**2 + yi_norm**2)
    
    return r_vals, theta_vals

def getTMinTMax(edge_data_matrix, simIdx=0):
    edges_sim = edge_data_matrix[:, :, simIdx]
    
    min_edge_by_time = np.min(edges_sim, axis=1)
    max_edge_by_time = np.max(edges_sim, axis=1)
    
    tMinIdx = np.where(min_edge_by_time > 0)[0][0]
    tMin = np.linspace(2, 180, 90)[tMinIdx]
    
    if np.max(edges_sim) > 120:
        tMaxIdx = np.where(max_edge_by_time > 120)[0][0]
        tMax = np.linspace(2, 180, 90)[tMaxIdx]
    else:
        tMaxIdx = 89
        tMax = 180

    return tMinIdx, tMin, tMaxIdx, tMax

def plotCartesianPolarEdges(edge_data_matrix, sim_data, theta=np.linspace(-31, 82, 160), simIdx=0, saveFig=False):
    
    # get Sim ID
    simID = sim_data[simIdx]
    
    # get minimum and max bounds (i.e. where edge appears and where edge is very close to boundary)
    tMinIdx, tMin, tMaxIdx, tMax = getTMinTMax(edge_data_matrix, simIdx=simIdx)
        
    # convert edge data to polar coordinates
    r_vals, theta_vals = getRValues(edge_data_matrix, simIdx=simIdx, minStartIdx=0)
    
    # sim times (list all times)
    all_times = np.linspace(2, 180, 90)
    
    # filter based on tMinIdx and tMaxIdx
    valid_times = all_times[tMinIdx:(tMaxIdx + 1)]
    
    valid_times_to_plot = np.arange(tMin, tMax + 2, step=6)
    
    valid_time_idx = np.array([np.where(all_times == i)[0][0] for i in valid_times_to_plot])
    
    #     return valid_times_to_plot
    
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, projection='polar')
        
    color = iter(plt.cm.jet(np.linspace(0, 1, len(valid_times_to_plot))))
    
    for i, j in enumerate(valid_time_idx):
        # ax1.plot(theta_vals[j, :] * (180 / np.pi), r_vals[j, :], color=next(color), label=fr"$u(t_{{{int(valid_times_to_plot[i])}}})$") # variable theta for each time??
        ax1.plot(theta, r_vals[j, :], color=next(color), label=fr"$u(t_{{{int(valid_times_to_plot[i])}}})$") # constant 
        # range of theta (usual plot)
        

    ax1.set_xlabel(r"$θ (deg)$")
    ax1.set_ylabel(r"$r_{edge}(t) \; R_s$")
    ax1.set_xlim(theta[0], theta[-1])
    ax1.set_ylim(4.05, 24)
    ax1.legend(loc=(1.05, .05))
    ax1.set_title("Sim {:03d}".format(simID))

    color = iter(plt.cm.jet(np.linspace(0, 1, len(valid_times_to_plot))))
    
    for i, j in enumerate(valid_time_idx):
        # ax2.plot(theta_vals[j, :], r_vals[j, :], color=next(color)) # variable theta for each time?
        ax2.plot(theta * np.pi/180, r_vals[j, :], color=next(color)) # constant range of theta (usual plot)

    ax2.set_rlabel_position(120)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_rmax(24)
    ax2.grid(True)


    fig.tight_layout()
    
    plt.style.use("dark_background")
    if saveFig:
        plt.savefig(os.path.join(edge_save_dir, "run_{:03d}.png".format(simID)),
                   bbox_inches="tight", pad_inches=0)
        print("Saved image for Sim {:03d}".format(simID))
        plt.close()


def plotTrainPredData(r_train, r_pred, edge_data_matrix, sim_data, theta=np.linspace(-31, 82, 160), simIdx=0, savefig=False):
    
    # get Sim ID
    simID = sim_data[simIdx]
    
    # get minimum and max bounds (i.e. where edge appears and where edge is very close to boundary)
    tMinIdx, tMin, tMaxIdx, tMax = getTMinTMax(edge_data_matrix, simIdx=simIdx)

    all_times = np.linspace(2, 180, 90)

    #     # filter based on tMinIdx and tMaxIdx
    valid_times = all_times[tMinIdx:(tMaxIdx + 1)]

    valid_times_to_plot = np.arange(tMin, tMax + 2, step=6)

    valid_time_idx = np.array([np.where(valid_times == i)[0][0] for i in valid_times_to_plot])
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121, projection='polar')
    ax2 = plt.subplot(122, projection='polar')
        
    color = iter(plt.cm.jet(np.linspace(0, 1, len(valid_times_to_plot))))
    
    for i, j in enumerate(valid_time_idx):
        cc = next(color)
        ax1.plot(theta * np.pi/180, r_train[j, :], color=cc, label=fr"$u(t_{{{int(valid_times_to_plot[i])}}})$")
        ax2.plot(theta * np.pi/180, r_pred[j, :], color=cc, label=fr"$u(t_{{{int(valid_times_to_plot[i])}}})$")
        

    ax1.set_rlabel_position(120)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_rmax(24)
    ax1.grid(True)

    ax1.set_title("Sim {:03d} Training".format(simID))
    ax1.legend(loc=(1.05, .05))
    
    ax2.set_rlabel_position(120)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_rmax(24)
    ax2.grid(True)

        
    ax2.set_title("Pred on Training")
    
    fig.tight_layout()
  
    
def plotTrainPredData2Models(r_train, r_pred1, r_pred2, edge_data_matrix, sim_data, theta=np.linspace(-31, 82, 160), simIdx=0,
                            savefig=False,
                            savedir="./train_2models_comparison"):
    
    # get Sim ID
    simID = sim_data[simIdx]
    
    # get minimum and max bounds (i.e. where edge appears and where edge is very close to boundary)
    tMinIdx, tMin, tMaxIdx, tMax = getTMinTMax(edge_data_matrix, simIdx=simIdx)

    all_times = np.linspace(2, 180, 90)

    #     # filter based on tMinIdx and tMaxIdx
    valid_times = all_times[tMinIdx:(tMaxIdx + 1)]

    valid_times_to_plot = np.arange(tMin, tMax + 2, step=12)

    valid_time_idx = np.array([np.where(valid_times == i)[0][0] for i in valid_times_to_plot])
    
    #     return valid_time_idx
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121, projection='polar')
    ax2 = plt.subplot(122, projection='polar')

    color = iter(plt.cm.jet(np.linspace(0, 1, len(valid_times_to_plot))))

    
    for i, j in enumerate(valid_time_idx):
        cc = next(color)
        ax1.plot(theta * np.pi/180, r_train[j, :], color=cc, label=fr"$u(t_{{{int(valid_times_to_plot[i])}}})$")
        ax1.plot(theta * np.pi/180, r_pred1[j, :], color=cc, linestyle='dashed', label="")

        ax2.plot(theta * np.pi/180, r_train[j, :], color=cc, label="")
        ax2.plot(theta * np.pi/180, r_pred2[j, :], color=cc, linestyle='dashed', label="")


    
    ax1.set_rlabel_position(120)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_rmax(24)
    ax1.grid(True)

    ax1.legend(loc=(1.05, 0.05))
    
    ax1.set_title("Sheeley".format(simID))
    
    ax2.set_rlabel_position(120)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_rmax(24)
    ax2.grid(True)


    ax2.set_title("Low")
    
    fig.tight_layout()
   
    if savefig:
        plt.savefig(os.path.join(savedir, "run_{:03d}.png".format(simID)),
                   bbox_inches="tight", pad_inches=0)
        print("Saved image for Sim {:03d}".format(simID))
        plt.close()
        
        
def plotTrainPredData1Model(r_train, r_pred1, edge_data_matrix, sim_data, theta=np.linspace(-31, 82, 160), simIdx=0,
                            savefig=False,
                            savedir="./train_1model_comparison"):
    
    # get Sim ID
    simID = sim_data[simIdx]
    
    # get minimum and max bounds (i.e. where edge appears and where edge is very close to boundary)
    tMinIdx, tMin, tMaxIdx, tMax = getTMinTMax(edge_data_matrix, simIdx=simIdx)

    all_times = np.linspace(2, 180, 90)

    #     # filter based on tMinIdx and tMaxIdx
    valid_times = all_times[tMinIdx:(tMaxIdx + 1)]

    valid_times_to_plot = np.arange(tMin, tMax + 2, step=12)

    valid_time_idx = np.array([np.where(valid_times == i)[0][0] for i in valid_times_to_plot])
    
    #     return valid_time_idx
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121, projection='polar')

    color = iter(plt.cm.jet(np.linspace(0, 1, len(valid_times_to_plot))))

    
    for i, j in enumerate(valid_time_idx):
        cc = next(color)
        ax1.plot(theta * np.pi/180, r_train[j, :], color=cc, label=fr"$u(t_{{{int(valid_times_to_plot[i])}}})$")
        ax1.plot(theta * np.pi/180, r_pred1[j, :], color=cc, linestyle='dashed', label="")


    
    ax1.set_rlabel_position(120)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_rmax(24)
    ax1.grid(True)

    ax1.legend(loc=(1.05, 0.05))
    
    ax1.set_title("Training and Predictions Sim {}".format(simID))
        
    fig.tight_layout()
   
    if savefig:
        plt.savefig(os.path.join(savedir, "run_{:03d}.png".format(simID)),
                   bbox_inches="tight", pad_inches=0)
        print("Saved image for Sim {:03d}".format(simID))
        plt.close()