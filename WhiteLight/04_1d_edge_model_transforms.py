# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# Load stacked edge data, and write basic viz functions to plot edges etc.

# %%
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import opinf

# %%
import re
import os

# %%
from multiprocessing import Pool
from itertools import repeat

# %%
plt.rc("axes.spines", right=True, top=True)
plt.rc("figure", dpi=300, 
       figsize=(9, 3)
      )
plt.rc("font", family="serif")
plt.rc("legend", edgecolor="none", frameon=True)
# plt.rc("text", usetex=True)

# %%
def load_edge_data(eventID):
    edge_data = np.load("./edge_data/CR{}_stacked_edge.npy".format(eventID))
    simID_data = np.load("./edge_data/CR{}_SimID4edge.npy".format(eventID))
    
    all_sims = []
    
    for s in simID_data:
        mm = re.search("run\d\d\d_AWSoM2T", s)
        all_sims.append(int(mm.group().split("run")[1].split("_")[0]))
    
    return edge_data, np.array(all_sims)


# %%
ed_2161, sd_2161 = load_edge_data(2161)
ed_2192, sd_2192 = load_edge_data(2192)
ed_2154, sd_2154 = load_edge_data(2154)

# %%
nTimes = ed_2161.shape[0]
nTimes

# %%
nTheta_2161 = ed_2161.shape[1]
nTheta_2192 = ed_2192.shape[1]
nTheta_2154 = ed_2154.shape[1]
nTheta_2161, nTheta_2192, nTheta_2154

# %%
nSims_2161 = ed_2161.shape[2]
nSims_2192 = ed_2192.shape[2]
nSims_2154 = ed_2154.shape[2]
nSims_2161, nSims_2192, nSims_2154

# %% [markdown]
# Workflow to filter data before it reaches boundary i.e. look for values going along indices above 120 , pick the minimum row index as cutoff for entire range of angles, and set that as the time up to which we have training and or testing data.

# %%
R_inner = 4.05
R_outer = 23.6

# %%
height_flattened=128
np.arange(0, height_flattened, step=21)

# %%
hts_pixel = np.linspace(0, height_flattened, height_flattened + 1)
hts_pixel

# %%
# is this correct?
radii = ((hts_pixel)/height_flattened)*(R_outer - R_inner) + R_inner
radii

# %%
ed_2161_scaled = (ed_2161/height_flattened)*(R_outer - R_inner) + R_inner
ed_2192_scaled = (ed_2192/height_flattened)*(R_outer - R_inner) + R_inner
ed_2154_scaled = (ed_2192/height_flattened)*(R_outer - R_inner) + R_inner

# %% [markdown]
# Adapted function from `plot_heat_data` in previous notebooks.

# %% [markdown]
# In plotting we need to convert axes indices to actual theta and R values (also control the ticks) - follow basic outline of `Kmeans_CR2154_SignedDist` from Hongfan.

# %% [markdown]
# **Calculation for angles**

# %%
theta_s_2161, theta_e_2161 = np.linspace(0, 360, 512)[160] + 1.2 * 180 - 360, np.linspace(0, 360, 512)[359] + 1.2 * 180 - 360
print("Range of angles for CR2161: {} {}".format(theta_s_2161, theta_e_2161))

theta_s_2192, theta_e_2192 = np.linspace(0, 360, 512)[90] + 1.2 * 180 - 360, np.linspace(0, 360, 512)[274] + 1.2 * 180 - 360
print("Range of angles for CR2161: {} {}".format(theta_s_2192, theta_e_2192))

theta_s_2154, theta_e_2154 = np.linspace(0, 360, 512)[50] + 1.2 * 180 - 360, np.linspace(0, 360, 512)[449] + 1.2 * 180 - 360
print("Range of angles for CR2161: {} {}".format(theta_s_2154, theta_e_2154))

# %% [markdown]
# ~~Can we also make a Cartesian plot? easier to see all edges progressing outward (with overlaid radius of occulting disk) and get a better sense of the shape.~~
#
# Save for all sims and events.

# %%
actualTimes = np.arange(2, 182, 2)

# %%
edge_save_dir = "./edge_img"

# %%
ed_2192.max()

# %% [markdown]
# **Threshold the edges**
#
# Set a cutoff time for plotting the edges, because if edge reaches the boundary very quickly, the algorithm breaks down after the cutoff and can result in incorrect edges (0 values, non-monotonic behaviour etc). so we need to filter the data correctly before using it for plotting / training purposes.

# %%
# cutoff_value = 120
# t_cutoff_idx = []
# for j in range(ed_2161.shape[2]):
#     for i in range(ed_2161.shape[0]):
#         boundary_idx = np.where(ed_2161[i, :, j])

# %%
plt.plot([1, 2, 4], [8, 9, 10])


# %%
def plot_edge_data(Z, titleSim, eventID=2161, plotAll=False, ax=None, theta=np.linspace(-32, 107, 200)):
    """Visualize edges data in space and time both in polar and original Cartesian coordinates."""
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Plot a few snapshots over the spatial domain.
    if plotAll:
        sample_columns = [i for i in range(Z.shape[1])]
    else:
        sample_columns = [0, 2, 4, 8, 12, 20, 40, 50, 60, 70, 74, 80, 84, 88, 89]
        
    sample_times = actualTimes[sample_columns]
    color = iter(plt.cm.viridis(np.linspace(0, 1, len(sample_columns))))
    while sample_columns[-1] > Z.shape[1]:
        sample_columns.pop()
    for i, j in enumerate(sample_columns):
        ax.plot(theta, Z[:, j], color=next(color), label=fr"$u(t_{{{sample_times[i]}}})$")

    ax.set_xlabel(r"$theta (deg)$")
    ax.set_ylabel(r"$r_{edge}(t)$")
    
    ax.set_xlim(theta[0], theta[-1])
    ax.set_yticks(np.arange(0, height_flattened, step=21), labels=['4R', '7R', '10R', '13R', '16R', '19R', '24R'])
    ax.legend(loc=(1.05, .05))
    ax.set_title("Edge data Sim {:03d}".format(titleSim))

    # %%
    # commenting out save functionality for now!!
    #     plt.savefig(os.path.join(edge_save_dir, "CR{}".format(eventID), "run_{:03d}.png".format(titleSim)),
    #                bbox_inches="tight", pad_inches=0)

    #     print("Saved image for Event {} Sim {:03d}".format(eventID, titleSim))

    #     plt.close()


# %%
def save_event_edge_data(eventID, plotAll=False, ax=None, theta_start=160, theta_end=360):
    # call data simwise, get sim labels
    edge_data, sim_data = load_edge_data(eventID)
    edge_data = (edge_data/height_flattened)*(R_outer - R_inner) + R_inner
    # make call to plotting function (also saves image) - MultiProcessing?
    nSims = edge_data.shape[2]
    
    theta_range = np.linspace(0, 360, 512)[theta_start:theta_end] + 1.2 * 180 - 360
    for i in range(nSims):
        plot_edge_data(edge_data[:, :, i].T, sim_data[i], eventID=eventID, 
                       plotAll=False, theta=theta_range)        


# %%
ed_2161.shape

# %%
plot_edge_data(ed_2161[:, :, 0].T, 31, eventID=2161, plotAll=False, ax=None, theta=np.linspace(-32, 107, 200))

# %% [markdown]
# Are our ylims getting set correctly in the figure?

# %% [markdown]
# A problem with the approach is that blindly plotting edges means that we will plot lines even where the algorithm is expected to stop working reliably i.e. where the edge has reached the boundary or is too slow and the lack of monotonicity means a lot of overlap with the edges at previous timesteps.

# %%
edge1 = ed_2161[20, :, 0]
edge2 = ed_2161[21, :, 0]

plt.plot(edge1, label="Edge t1")
plt.plot(edge2, label="Edge t2")
plt.legend()

# %% [markdown]
# For one sim, e.g. sim 31, find the min time at which min edge value is greater than zero. That has to be our starting point, else part of it is unobserved and we can't really use that for training.

# %%
ed31 = ed_2161[:, :, 0]

# %%
min_edge_by_time = np.min(ed31, axis=1)
min_edge_by_time

# %%
tMinIdx = np.where(min_edge_by_time > 0)[0][0]
tMinIdx

# %%
tMin = np.linspace(2, 180, 90)[tMinIdx]
tMin

# %% [markdown]
# Now set the values of selected points $(r/t)$ using `tMin` as a reference.

# %%
theta_samples = np.linspace(-32, 107, 200)[0:-1:2]
theta_all = np.linspace(-32, 107, 200)

# %%
r_by_t_ref = ed31[tMinIdx, 0:-1:2]/tMin

# %%
r_by_t_ref

# %%
t2 = tMin + 2
t2

# %%
r2_exact = r_by_t_ref * t2

# %%
r2_all = ed31[tMinIdx + 1, :]

# %%
plt.scatter(theta_samples, r2_exact, c=theta_samples, cmap='Greens')

# %% [markdown]
# Now set up interpolation to figure out the closest $\theta$ values where we can match to the calculated `r2`.

# %%
order = r2_all.argsort()
y_data = r2_all[order]
x_data = theta_all[order]

# %% [markdown]
# I think the above interpolation is going wrong. Theta should follow the parabolic curve but because of non-unique $r$ the sorting is not putting the $\theta$ in correct order. Maybe we should try piecewise sorting instead!!

# %%
plt.plot(np.diff(r2_all))

# %%
We really need to calculate th

# %%
theta_interp = np.interp(r2_exact, y_data, x_data)
theta_interp

# %% [markdown]
# Plot all interpolated $\theta$ and $r$ values.

# %%
plt.scatter(theta_interp, r2_exact, label="Interpolated")
plt.plot(theta_all, edge1, label="Edge t1")
plt.plot(theta_all, edge2, label="Edge t2")
plt.legend()

# %% [markdown]
# Plot $r$ on x-axis and $\theta$ on y-axis. Try `scipy.interpolate` method and see if the result is any different. Do we have to change the scaling to make this work?

# %%
plt.plot(r2_all, theta_all, label="theta = f(r)")
plt.xlabel("r")
plt.ylabel("θ")
plt.legend()

# %%
spl = CubicSpline(y_data, x_data)
plt.plot(r2_all, theta_all, label="theta = f(r)")
plt.plot(r2_all[0:-1:2], spl(r2_all[0:-1:2]), 'o', label="Interpolated for curve")
plt.plot(r2_exact, spl(r2_exact), 'o', label="Interpolated for calculated r")


plt.xlabel("r")
plt.ylabel("θ")
plt.legend()

# %% [markdown]
# **Approach 1**: Can we change sampling procedure too (non-uniform)? First, we need to deal with the non-monotonicity correctly. Some help thankfully from https://stackoverflow.com/questions/48028766/get-x-values-corresponding-to-y-value-on-non-monotonic-curves

# %%
import numpy as np
from scipy.interpolate import interpn


# y_data and x_data are sorted data.

theta_direction = np.sign(np.diff(x_data))

# %%
# import numpy as np
# from scipy.interpolate import interp1d

# # convert data lists to arrays
# x, y = np.array(x), np.array(y)

# # sort x and y by x value
# order_vals = np.argsort(x)
# xsort, ysort = x[order_vals], y[order_vals]

# # compute indices of points where y changes direction
# ydirection = np.sign(np.diff(ysort))
# changepoints = 1 + np.where(np.diff(ydirection) != 0)[0]

# # find groups of x and y within which y is monotonic
# xgroups = np.split(xsort, changepoints)
# ygroups = np.split(ysort, changepoints)
# interps = [interp1d(y, x, bounds_error=False) for y, x in zip(ygroups, xgroups)]

# # interpolate all y values
# yval = 100
# xvals = np.array([interp(yval) for interp in interps])

# print(xvals)

# %%

# %%

# %%

# %%

# %% [markdown]
# **Approach 2**: Nearest neighbours? Find closest matching $(r/t)$ values in edge at next timestep to the previous, and simply use those so that interpolation doesn't have to be optimized (interpolation goes the loong and possibly too complicated? way round trying to find the right $\theta$).

# %%
r_by_t_calc = 

# %%

# %%

# %%

# %%
# # commented out call for saving edge data
# save_event_edge_data(2161, plotAll=False, ax=None, theta_start=160, theta_end=360)
# save_event_edge_data(2192, plotAll=False, ax=None, theta_start=90, theta_end=275)
# save_event_edge_data(2154, plotAll=False, ax=None, theta_start=50, theta_end=450)

# %%
# POLAR AXES PLOT
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# # r = np.linspace(4.05, 23.6, height_flattened + 1)
# theta = np.linspace(-32, 107, 90) * np.pi / 180
# ax.plot(theta, ed_2161_scaled[:, :, 0].T[25, :])
# ax.plot(theta, ed_2161_scaled[:, :, 0].T[40, :])
# ax.plot(theta, ed_2161_scaled[:, :, 0].T[50, :])

# %%
# plot_edge_data(edge_data_cutoff.T, "Edge data Sim 31", plotAll=False)

# %% [markdown]
# **Scratch Begin**

# %%
aaa = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

# %%
np.min(aaa, axis=1)

# %%
edge_data_31 = ed_2161[:, :, 0]

# %%
tIdx = min(np.unique(np.where(edge_data_31 >= 120)[0]))
tIdx

# %%
tCutoff = actualTimes[tIdx]
tCutoff

# %%
edge_data_cutoff = edge_data_31[:tIdx, :]
edge_data_cutoff.shape

# %%
sample_columns = [0, 2, 4, 8, 12, 20, 40, 50, 60, 70]
sample_times = actualTimes[sample_columns]
sample_times

# %%
basis = opinf.pre.PODBasis().fit(edge_data_cutoff.T, residual_energy=1e-8)
print(basis)

# Check the decay of the singular values.
basis.plot_svdval_decay()
plt.xlim(0, 25)

# %%
plt.plot(basis.svdvals/basis.svdvals[1])
plt.xlim(0, 10)

# %% [markdown]
# **Scratch End**

# %%
