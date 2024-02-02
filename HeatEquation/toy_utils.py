# Copying over from `data_gen_utils` in bifidelity KLE paper.
import numpy as np
import matplotlib.pyplot as plt
import os


# %%
def getGridQuantities(nx, ny):
    x = np.linspace(0, 1, nx + 1)  # x-Grid (location of cell faces)
    y = np.linspace(0, 1, ny + 1)  # y-Grid (location of cell faces)
    xm = x[:-1] + (x[1] - x[0]) / 2  # x-Grid (location of cell centers)
    ym = y[:-1] + (y[1] - y[0]) / 2  # y-Grid (location of cell centers)
    XM, YM = np.meshgrid(xm, ym)  # Meshgrid for cell centers
    dx = xm[1] - xm[0]  # Grid spacing in x
    dy = ym[1] - ym[0]  # Grid spacing in y

    dxi = 1 / dx  # Inverse of dx
    dyi = 1 / dy  # Inverse of dy
    dxi2 = 1 / dx ** 2  # inverse of dx^2
    dyi2 = 1 / dy ** 2  # inverse of dy^2
    
    return x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2


# %%
def getVelocitiesGeneric(nx, ny):
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = getGridQuantities(nx, ny)
    u = np.zeros((nx + 1, ny))
    v = np.zeros((nx, ny + 1))
    for i in range(nx + 1):
        for j in range(ny):
            u[i, j] = 1 / 10 - (np.sin(np.pi * x[i])) ** 2 * (
                        np.sin(np.pi * (ym[j] - 0.05)) * np.cos(np.pi * (ym[j] - 0.05)) -
                        np.sin(np.pi * (ym[j] + 0.05)) * np.cos(np.pi * (ym[j] + 0.05)))
    for i in range(nx):
        for j in range(ny + 1):
            v[i, j] = np.sin(np.pi * xm[i]) * np.cos(np.pi * xm[i]) * (
                        (np.sin(np.pi * (y[j] - 0.05))) ** 2 -
                        (np.sin(np.pi * (y[j] + 0.05))) ** 2)
            
    return u, v


# %% 
def getPhiForThetaGeneric(nx, ny, u_vel, v_vel, theta_s=0.01, theta_h=0.05, theta_x=0.3, theta_y=0.55, alpha=1e-2):
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = getGridQuantities(nx, ny)
    
    omega = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            omega[i, j] = ((theta_s) / (2 * np.pi * theta_h**2)) * (np.exp(-((theta_x - xm[i]) ** 2 + (theta_y - ym[j]) ** 2) / (2 * theta_h ** 2)) - np.exp(-((xm[i] - theta_x + 0.05) ** 2 + (ym[j] - theta_y + 0.05) ** 2) / (2 * theta_h ** 2)))
            
    CFL = 0.8
    maxU = np.max(np.abs(u_vel))
    maxU = np.max([maxU, np.max(np.abs(v_vel))])
    dt_c = CFL * dx / maxU  # dt based on convective CFL
    dt_v = CFL * dx ** 2 / alpha / 4  # dt based on viscous CFL


    dt = min(dt_c, dt_v)   # Simulation timestep
    tf = 2.5               # Final time

    # Initialize phi
    phi = np.zeros((nx, ny))
    phi_old = phi.copy()


    # Output frequency
    import copy
    # Output frequency
    freq = tf / 8
    otime = copy.deepcopy(freq)

    # Loop through time
    t = 0
    tcount = 0
    while t <= tf:
        # Loop through space
        for i in range(nx):
            for j in range(ny):
                ip1 = i + 1
                if ip1 == nx:
                    ip1 = 0
                
                ip2 = i + 2
                if ip2 == nx:
                    ip2 = 0
                elif ip2 == nx + 1:
                    ip2 = 1

                im1 = i - 1
                if im1 == -1:
                    im1 = nx - 1
                
                im2 = i - 2
                if im2 == -1:
                    im2 = nx - 1
                elif im2 == -2:
                    im2 = nx - 2
                
                jp1 = j + 1
                if jp1 == ny:
                    jp1 = 0
                
                jp2 = j + 2
                if jp2 == ny:
                    jp2 = 0
                elif jp2 == ny + 1:
                    jp2 = 1

                jm1 = j - 1
                if jm1 == -1:
                    jm1 = ny - 1

                jm2 = j - 2
                if jm2 == -1:
                    jm2 = ny - 1
                elif jm2 == -2:
                    jm2 = ny - 2

                # Diffusion (explicit)
                diff = alpha * dxi2 * (phi_old[im1, j] - 2 * phi_old[i, j] + phi_old[ip1, j])  # in x
                diff += alpha * dyi2 * (phi_old[i, jm1] - 2 * phi_old[i, j] + phi_old[i, jp1])  # in y
                
                # Face velocities
                ue = u_vel[i + 1, j]
                uw = u_vel[i, j]
                un = v_vel[i, j + 1]
                us = v_vel[i, j]
                
                # QUICK reconstruction (explicit)
                # E
                if ue > 0:
                    phi_e = (-phi_old[im1, j] + 5 * phi_old[i, j] + 2 * phi_old[ip1, j]) / 6
                else:
                    phi_e = (2 * phi_old[i, j] + 5 * phi_old[ip1, j] - phi_old[ip2, j]) / 6
                # W
                if uw > 0:
                    phi_w = (-phi_old[im2, j] + 5 * phi_old[im1, j] + 2 * phi_old[i, j]) / 6
                else:
                    phi_w = (2 * phi_old[im1, j] + 5 * phi_old[i, j] - phi_old[ip1, j]) / 6
                # N
                if un > 0:
                    phi_n = (-phi_old[i, jm1] + 5 * phi_old[i, j] + 2 * phi_old[i, jp1]) / 6
                else:
                    phi_n = (2 * phi_old[i, j] + 5 * phi_old[i, jp1] - phi_old[i, jp2]) / 6
                # S
                if us > 0:
                    phi_s = (-phi_old[i, jm2] + 5 * phi_old[i, jm1] + 2 * phi_old[i, j]) / 6
                else:
                    phi_s = (2 * phi_old[i, jm1] + 5 * phi_old[i, j] - phi_old[i, jp1]) / 6

                # Convection (explicit)
                conv =      - dxi * (ue*phi_e - uw*phi_w) # in x
                conv = conv - dyi * (un*phi_n - us*phi_s) # in y
                
                # Update
            
                # 1st-order explicit
                phi[i,j] = phi_old[i,j] + dt * (conv + diff + omega[i,j])
        
        # Update time
        t = t + dt
        
        # Update old phi
        phi_old = phi
        
        tcount = tcount + dt
        
        
    return phi.T


# %%

def plotPhiForThetaGeneric(nx, ny, sid, u_vel, v_vel, 
                           theta_s=0.01, 
                           theta_h=0.05, 
                           theta_x=0.3, 
                           theta_y=0.55, 
                           savedatadir=None, 
                           savefigdir=None, 
                           savefig=False, 
                           savedata=False,
                           fidelity="LF",
                           alpha=1e-2):
    
    phi_data = getPhiForThetaGeneric(nx, ny, u_vel, v_vel, 
                                     theta_s=theta_s, 
                                     theta_h=theta_h, 
                                     theta_x=theta_x, 
                                     theta_y=theta_y,
                                     alpha=alpha)
    
    if savedata:
        if fidelity == "LF":
            np.savetxt(os.path.join(savedatadir, "LF_2D_Run{:04d}.txt".format(sid)), phi_data)
        elif fidelity == "HF":
            np.savetxt(os.path.join(savedatadir, "HF_2D_Run{:04d}.txt".format(sid)), phi_data)
    
    if savefig:
        fig, ax = plt.subplots()
        im = ax.imshow(phi_data,
                 origin="lower",
                 extent=[0, 1, 0, 1],
                 cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im)
        ax.set_title(r"θₛ={:.2f} θₕ={:.2f} θₓ={:.2f} θᵧ={:.2f}".format(theta_s, theta_h, theta_x, theta_y))
        fig.savefig(os.path.join(savefigdir, "{}_2D_Run{:04d}.png".format(fidelity, sid)))
        plt.close()
        
    print("processed data for Run {:04d}".format(sid))

# %% Function for purely plotting purposes
# save solution at all timesteps

def getPhiForThetaGenericAllTimesteps(nx, ny, u_vel, v_vel, 
                                      theta_s=0.01, 
                                      theta_h=0.05, 
                                      theta_x=0.3, 
                                      theta_y=0.55,
                                      tf=2.5,
                                      #savedatadir=None, 
                                      #savefigdir=None, 
                                      #savefig=False, 
                                      #savedata=False,
                                      #fidelity="HF",
                                      alpha=1e-2):
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = getGridQuantities(nx, ny)
    
    omega = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            omega[i, j] = ((theta_s) / (2 * np.pi * theta_h**2)) * (np.exp(-((theta_x - xm[i]) ** 2 + (theta_y - ym[j]) ** 2) / (2 * theta_h ** 2)) - np.exp(-((xm[i] - theta_x + 0.05) ** 2 + (ym[j] - theta_y + 0.05) ** 2) / (2 * theta_h ** 2)))

    CFL = 0.8
    maxU = np.max(np.abs(u_vel))
    maxU = np.max([maxU, np.max(np.abs(v_vel))])
    dt_c = CFL * dx / maxU  # dt based on convective CFL
    dt_v = CFL * dx ** 2 / alpha / 4  # dt based on viscous CFL


    dt = min(dt_c, dt_v)   # Simulation timestep
#     tf = 2.5               # Final time

    # Initialize phi
    phi = np.zeros((nx, ny))
    phi_old = phi.copy()

    phi_all = []

    # Output frequency
    import copy
    # Output frequency
    freq = tf / 8
    otime = copy.deepcopy(freq)

    # Loop through time
    t = 0
    tcount = 0
    while t <= tf:
        # Loop through space
        for i in range(nx):
            for j in range(ny):
                ip1 = i + 1
                if ip1 == nx:
                    ip1 = 0
                
                ip2 = i + 2
                if ip2 == nx:
                    ip2 = 0
                elif ip2 == nx + 1:
                    ip2 = 1

                im1 = i - 1
                if im1 == -1:
                    im1 = nx - 1
                
                im2 = i - 2
                if im2 == -1:
                    im2 = nx - 1
                elif im2 == -2:
                    im2 = nx - 2
                
                jp1 = j + 1
                if jp1 == ny:
                    jp1 = 0
                
                jp2 = j + 2
                if jp2 == ny:
                    jp2 = 0
                elif jp2 == ny + 1:
                    jp2 = 1

                jm1 = j - 1
                if jm1 == -1:
                    jm1 = ny - 1

                jm2 = j - 2
                if jm2 == -1:
                    jm2 = ny - 1
                elif jm2 == -2:
                    jm2 = ny - 2

                # Diffusion (explicit)
                diff = alpha * dxi2 * (phi_old[im1, j] - 2 * phi_old[i, j] + phi_old[ip1, j])  # in x
                diff += alpha * dyi2 * (phi_old[i, jm1] - 2 * phi_old[i, j] + phi_old[i, jp1])  # in y
                
                # Face velocities
                ue = u_vel[i + 1, j]
                uw = u_vel[i, j]
                un = v_vel[i, j + 1]
                us = v_vel[i, j]
                
                # QUICK reconstruction (explicit)
                # E
                if ue > 0:
                    phi_e = (-phi_old[im1, j] + 5 * phi_old[i, j] + 2 * phi_old[ip1, j]) / 6
                else:
                    phi_e = (2 * phi_old[i, j] + 5 * phi_old[ip1, j] - phi_old[ip2, j]) / 6
                # W
                if uw > 0:
                    phi_w = (-phi_old[im2, j] + 5 * phi_old[im1, j] + 2 * phi_old[i, j]) / 6
                else:
                    phi_w = (2 * phi_old[im1, j] + 5 * phi_old[i, j] - phi_old[ip1, j]) / 6
                # N
                if un > 0:
                    phi_n = (-phi_old[i, jm1] + 5 * phi_old[i, j] + 2 * phi_old[i, jp1]) / 6
                else:
                    phi_n = (2 * phi_old[i, j] + 5 * phi_old[i, jp1] - phi_old[i, jp2]) / 6
                # S
                if us > 0:
                    phi_s = (-phi_old[i, jm2] + 5 * phi_old[i, jm1] + 2 * phi_old[i, j]) / 6
                else:
                    phi_s = (2 * phi_old[i, jm1] + 5 * phi_old[i, j] - phi_old[i, jp1]) / 6

                # Convection (explicit)
                conv =      - dxi * (ue*phi_e - uw*phi_w) # in x
                conv = conv - dyi * (un*phi_n - us*phi_s) # in y
                
                # Update
            
                # 1st-order explicit
                phi[i,j] = phi_old[i,j] + dt * (conv + diff + omega[i,j])
        
        # Update time
        t = t + dt
        
        print("Generated solution for t = {:.3f} s".format(t))
        
        # Update old phi
        #         phi_old = phi
        phi_old = copy.deepcopy(phi)
        phi_all.append(phi_old.T)

        tcount = tcount + dt
        
    return phi_all