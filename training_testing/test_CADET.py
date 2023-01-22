# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tqdm, os, glob
import numpy as np
# import jax.numpy as np
from jax.numpy import stack
import pandas as pd
from scipy.ndimage import center_of_mass, rotate
from scipy.special import beta as betafun
from astropy.nddata import CCDData
from astropy.io import fits
# from sherpa.astro.ui import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, ListedColormap, Normalize
from matplotlib.ticker import FuncFormatter, ScalarFormatter
fsize, fsize2 = 16, 19
plt.rc('font', size=fsize)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{newtxtext}\usepackage{newtxmath}')

from sklearn.cluster import DBSCAN

from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.config.experimental import list_physical_devices, set_memory_growth, set_virtual_device_configuration, VirtualDeviceConfiguration, list_logical_devices

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = list_physical_devices('GPU')
set_memory_growth(gpus[0], False)
set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=2000)])
logical_gpus = list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

from os import environ
# environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.65'

import sys
sys.path.append("/home/plsek/Diplomka/CADET") 
from beta_model_fun import get_batch, run_vect
from functions import *

import warnings
warnings.filterwarnings('ignore') # :-)

q1, q3 = 0.15865525, 0.84134475

path_model = "CADET_size.hdf5"
# path_model = "/home/plsek/Diplomka/CADET/models/CADET_size.hdf5"
# path_model = "models/b16_lr0.0005_normal_rims_nosloshing_flatter_50_customunet.hdf5"
model = load_model(path_model, custom_objects = {"LeakyReLU": LeakyReLU, "ReLU": ReLU})

parnames = ["dx", "dy", "dx_2", "dy_2", "dx_3", "dy_3", "phi", "phi_2", "phi_3",
            "ampl", "r0", "beta", "ellip", "ampl_2", "r0_2", "beta_2", "ampl_3", "r0_3", "beta_3", "bkg", 
            "s_depth", "s_period", "s_dir", "s_angle",
            "r1", "r2", "phi1", "phi2", "theta1", "theta2", "R1", "R2", "e1", "e2", "varphi1", "varphi2",
            "rim_size", "rim_height", "rim_type"]

def error_heatmap(N=10, threshold=0.5, fac=3,
                  ampl=20, r0=10, alpha=1, bkg=0,
                  ampl_2=0, r0_2=0, alpha_2=0,
                  ampl_3=0, r0_3=0, alpha_3=0,
                  radii=[15, 20, 25, 30], sizes=[5, 10, 20], 
                  d1=0, d2=0, s1=0, s2=0):
    
    A = (ampl / r0 / betafun(alpha, 0.5))**0.5
    beta = (alpha + 0.5) / 3

    if ampl_2 > 0:
        A_2 = (ampl_2 / r0_2 / betafun(alpha_2, 0.5))**0.5
        beta_2 = (alpha_2 + 0.5) / 3
    else: A_2, beta_2 = 0, 0

    if ampl_3 > 0:
        A_3 = (ampl_3 / r0_3 / betafun(alpha_3, 0.5))**0.5
        beta_3 = (alpha_3 + 0.5) / 3
    else: A_3, beta_3 = 0, 0

    df_A = pd.DataFrame(columns=radii, index=sizes)
    df_Ae = pd.DataFrame(columns=radii, index=sizes)
    
    df_V = pd.DataFrame(columns=radii, index=sizes)
    df_Ve = pd.DataFrame(columns=radii, index=sizes)

    counts = []

    tvl = 0
    for rad in tqdm.tqdm(radii):
        for size in sizes:
            if rad <= size:
                df_A[rad].loc[size] = np.nan
                df_Ae[rad].loc[size] = np.nan
                df_V[rad].loc[size] = np.nan
                df_Ve[rad].loc[size] = np.nan
                continue
            
            n = stack([i for i in range(tvl,tvl+N)])
            tvl += N
            dx, dy, dx_2, dy_2, dx_3, dy_3 = 0, 0, 0, 0, 0, 0
            phi, phi_2, phi_3 = 0, 0, 0
            ellip =  0
            s_depth, s_period, s_dir, s_angle = 0, 0, 0, 0
            rim_size, rim_height, rim_type = 0, 0, 0
    
            phi1, phi2, theta1, theta2, varphi1, varphi2 = 0, 180, 0, 0, 0, 0
            r1, r2, R1, R2, e1, e2 = rad, rad, size, size, 0, 0

            # e1 = stack(exponential(0.15, 0.6, N))
            # e2 = stack(exponential(0.15, 0.6, N))

            phi1 = stack(np.random.uniform(0, 180, size=N))
            phi2 = phi1 + 180

            s = lambda x: stack([x]*(N))

            Xs, ys, vs = run_vect(n, s(dx), s(dy), s(dx_2), s(dy_2), s(dx_3), s(dy_3),
                                  s(phi), s(phi_2), s(phi_3),
                                  s(A), s(r0), s(beta), s(ellip),
                                  s(A_2), s(r0_2), s(beta_2),
                                  s(A_3), s(r0_3), s(beta_3),
                                  s(bkg), 
                                  s(s_depth), s(s_period), s(s_dir), s(s_angle),
                                  s(r1), s(r2), phi1, phi2, s(theta1), s(theta2), s(R1), s(R2), s(e1), s(e2), s(varphi1), s(varphi2),
                                  s(rim_size), s(rim_height), s(rim_type))

            # AUGMENTATION - ROTATION
            Xrot1 = np.array([np.rot90(X, k=1) for X in Xs])
            yrot1 = np.array([np.rot90(y, k=1) for y in ys])
            Xrot2 = np.array([np.rot90(X, k=2) for X in Xs])
            yrot2 = np.array([np.rot90(y, k=2) for y in ys])
            Xrot3 = np.array([np.rot90(X, k=3) for X in Xs])
            yrot3 = np.array([np.rot90(y, k=3) for y in ys])

            # # AUGMENTATION - INVERT
            # Xmir1 = np.array([X[::-1,:] for X in Xs])
            # ymir1 = np.array([y[::-1,:] for y in ys])
            # Xmir2 = np.array([X[:,::-1] for X in Xs])
            # ymir2 = np.array([y[:,::-1] for y in ys])
            # Xmir3 = np.array([X[::-1,::-1] for X in Xs])
            # ymir3 = np.array([y[::-1,::-1] for y in ys])
            # Xmir4 = np.array([X.T for X in Xs])
            # ymir4 = np.array([y.T for y in ys])

            Xs = np.concatenate((Xs, Xrot1, Xrot2, Xrot3))
            ys = np.concatenate((ys, yrot1, yrot2, yrot3))
            # Xs = np.concatenate((Xs, Xrot1, Xrot2, Xrot3, Xmir1, Xmir2, Xmir3, Xmir4))
            # ys = np.concatenate((ys, yrot1, yrot2, yrot3, ymir1, ymir2, ymir3, ymir4))

            # Xs = jnp.log10(Xs+1) / jnp.max(jnp.log10(Xs+1), axis=(1,2)).reshape((N,1,1))

            # y_pred = model.predict(image.reshape(N, 128, 128, 1)).reshape(N, 128 ,128)

            pred_A, errors_A, pred_V, errors_V = [], [], [], []
            for i in range(len(Xs)):
                image = np.log10(Xs[i]+1) / np.amax(np.log10(Xs[i]+1))
                pred = model.predict(image.reshape(1, 128, 128, 1)).reshape(128 ,128)
                # pred = y_pred[i]
                pred = np.where(pred > threshold, 1, 0)
                cavs = decompose_two(pred)
                
                img, cube, cube0 = 0, 0, 0
                for i1, cav in enumerate(cavs):
                    com = np.where((cav>0) & (ys[i]>0), 1, 0)
                    if com.any():
                        img += cav
                        cube += make_cube(cav, fac)

                pred_A.append((np.sum(img) / np.sum(ys[i])))
                errors_A.append(abs(np.sum(img) / np.sum(ys[i]) - 1) * 100)
                pred_V.append((np.sum(cube) / vs[i]))
                errors_V.append(abs(np.sum(cube) / vs[i] - 1) * 100)
                counts.append(np.sum(Xs[i]))

            df_A[rad].loc[size] = np.median(pred_A)
            df_Ae[rad].loc[size] = np.median(errors_A)
            df_V[rad].loc[size] = np.median(pred_V)
            df_Ve[rad].loc[size] = np.median(errors_V)

    # print(df_V)
    
    c = 0.8
    res = sizes[1] - sizes[0]
    
    fig, axs = plt.subplots(1, 2, figsize=(len(radii)*c*2, len(sizes)*c))
    if (ampl_2 == 0) & (ampl_3 == 0):
        fig.suptitle(f"A={ampl:.0f}, r0={r0:.2f}, beta={beta:.2f},\nbkg={bkg:.2f}, counts={np.mean(counts):.0f}", x=0.5, y=0.95)
    elif ampl_3 == 0:
        fig.suptitle(f"A={ampl:.0f}, r0={r0:.2f}, beta={beta:.2f}, A_2={ampl_2:.0f}, r0_2={r0_2:.2f}, beta_2={beta_2:.2f},\nbkg={bkg:.2f}, counts={np.mean(counts):.0f}", x=0.5, y=0.95)
    else:
        fig.suptitle(f"A={ampl:.0f}, r0={r0:.2f}, beta={beta:.2f}, A_2={ampl_2:.0f}, r0_2={r0_2:.2f}, beta_2={beta_2:.2f}, A_3={ampl_3:.0f}, r0_3={r0_3:.2f}, beta_3={beta_3:.2f},\nbkg={bkg:.2f}, counts={np.mean(counts):.0f}", x=0.5, y=0.95)

    df_1 = df_V
    limit = np.max([abs(np.min(df_1.to_numpy())-1), abs(np.max(df_1.to_numpy())-1)])
    axs[0].imshow(df_1.to_numpy().astype('float'),cmap="bwr",norm=Normalize(1-limit, 1+limit))
    for i, size in enumerate(sizes):
        for j, rad in enumerate(radii):
            if not np.isnan(df_1[rad].loc[size]):
                axs[0].text(j, i, round(df_1[rad].loc[size], 2), ha="center", va="center", size=fsize-2)
    
    axs[0].plot((d1-radii[0])/res, (s1-sizes[0])/res, "o", color="red")
    axs[0].plot((d2-radii[0])/res, (s2-sizes[0])/res, "o", color="red")
            
    axs[0].set_yticks(np.arange(len(sizes)))
    axs[0].set_yticklabels(df_1.index, fontsize=fsize-2)
    axs[0].set_xticks(np.arange(len(radii)))
    axs[0].set_xticklabels(df_1.columns, fontsize=fsize-2)
    
    axs[0].set_yticks(np.arange(len(sizes))-0.5, minor=True)
    axs[0].set_xticks(np.arange(len(radii))-0.5, minor=True)
    axs[0].grid(which="minor", color="gray", linestyle='-', linewidth=1)

    axs[0].tick_params(axis="both", which="major", length=4, width=1.0)
    axs[0].tick_params(axis="both", which="minor", length=0, width=0)
    
    axs[0].set_ylabel("cavity radius (pixels)")
    axs[0].set_xlabel("cavity distance (pixels)")

    df_1 = df_Ve
    axs[1].imshow(df_1.to_numpy().astype('float'),cmap="GnBu") #, norm=Normalize(1-limit, 1+limit))
    for i, size in enumerate(sizes):
        for j, rad in enumerate(radii):
            if not np.isnan(df_1[rad].loc[size]):
                axs[1].text(j, i, f"{df_1[rad].loc[size]:.0f}\,\%", ha="center", va="center", size=fsize-2)

    axs[1].plot((d1-radii[0])/res, (s1-sizes[0])/res, "o", color="red")
    axs[1].plot((d2-radii[0])/res, (s2-sizes[0])/res, "o", color="red")
            
    axs[1].set_yticks(np.arange(len(sizes)))
    axs[1].set_yticklabels(df_1.index, fontsize=fsize-2)
    axs[1].set_xticks(np.arange(len(radii)))
    axs[1].set_xticklabels(df_1.columns, fontsize=fsize-2)
    
    axs[1].set_yticks(np.arange(len(sizes))-0.5, minor=True)
    axs[1].set_xticks(np.arange(len(radii))-0.5, minor=True)
    axs[1].grid(which="minor", color="gray", linestyle='-', linewidth=1)
    
    axs[1].tick_params(axis="both", which="major", length=4, width=1.0)
    axs[1].tick_params(axis="both", which="minor", length=0, width=0)
    
    axs[1].set_ylabel("cavity radius (pixels)")
    axs[1].set_xlabel("cavity distance (pixels)")

    return fig, axs

def relative_error(galaxy, which, N=10, threshold=0.5, fac=3,
                   ampl=20, r0=10, alpha=1, bkg=0,
                   ampl_2=0, r0_2=0, alpha_2=0,
                   ampl_3=0, r0_3=0, alpha_3=0,
                   r1=0, r2=0, R1=0, R2=0):
    
    A = (ampl / r0 / betafun(alpha, 0.5))**0.5
    beta = (alpha + 0.5) / 3

    A_2 = np.where(ampl_2 > 0, (ampl_2 / r0_2 / betafun(alpha_2, 0.5))**0.5, 0)
    beta_2 = np.where(ampl_2 > 0, (alpha_2 + 0.5) / 3, 0)

    A_3 = np.where(ampl_3 > 0, (ampl_3 / r0_3 / betafun(alpha_3, 0.5))**0.5, 0)
    beta_3 = np.where(ampl_3 > 0, (alpha_3 + 0.5) / 3, 0)

    n = stack([i for i in range(N)])
    dx, dy, dx_2, dy_2, dx_3, dy_3 = 0, 0, 0, 0, 0, 0
    phi, phi_2, phi_3 = 0, 0, 0
    ellip =  0
    s_depth, s_period, s_dir, s_angle = 0, 0, 0, 0
    rim_size, rim_height, rim_type = 0, 0, 0

    theta1, theta2, varphi1, varphi2 = 0, 0, 0, 0
    e1, e2 = 0, 0

    # e1 = stack(exponential(0.15, 0.6, N))
    # e2 = stack(exponential(0.15, 0.6, N))

    phi1 = stack(np.random.uniform(0, 180, size=N))
    phi2 = phi1 + 180 + np.random.normal(0, 20, size=N)

    s1 = lambda x: stack([float(x)]*N)
    s2 = lambda x: stack(list(x))

    Xs, ys, vs = run_vect(n, s1(dx), s1(dy), s1(dx_2), s1(dy_2), s1(dx_3), s1(dy_3),
                            s1(phi), s1(phi_2), s1(phi_3),
                            s2(A), s2(r0), s2(beta), s1(ellip),
                            s2(A_2), s2(r0_2), s2(beta_2),
                            s2(A_3), s2(r0_3), s2(beta_3),
                            s2(bkg), 
                            s1(s_depth), s1(s_period), s1(s_dir), s1(s_angle),
                            s1(r1), s1(r2), phi1, phi2, s1(theta1), s1(theta2), s1(R1), s1(R2), s1(e1), s1(e2), s1(varphi1), s1(varphi2),
                            s1(rim_size), s1(rim_height), s1(rim_type))

    # # AUGMENTATION - ROTATION
    # Xrot1 = np.array([np.rot90(X, k=1) for X in Xs])
    # yrot1 = np.array([np.rot90(y, k=1) for y in ys])
    # Xrot2 = np.array([np.rot90(X, k=2) for X in Xs])
    # yrot2 = np.array([np.rot90(y, k=2) for y in ys])
    # Xrot3 = np.array([np.rot90(X, k=3) for X in Xs])
    # yrot3 = np.array([np.rot90(y, k=3) for y in ys])

    # Xs = np.concatenate((Xs, Xrot1, Xrot2, Xrot3))
    # ys = np.concatenate((ys, yrot1, yrot2, yrot3))

    # profs, pred_A, errors_A, pred_V, errors_V, counts = [], [], [], [], [], []
    # for i in range(len(Xs)):
    #     image = np.log10(Xs[i]+1) / np.amax(np.log10(Xs[i]+1))
    #     pred = model.predict(image.reshape(1, 128, 128, 1)).reshape(128 ,128)
    #     pred = np.where(pred > threshold, 1, 0)
    #     cavs = decompose_two(pred)
        
    #     img, cube, cube0 = 0, 0, 0
    #     for i1, cav in enumerate(cavs):
    #         com = np.where((cav>0) & (ys[i]>0), 1, 0)
    #         if com.any():
    #             img += cav
    #             cube += make_cube(cav, fac)

    #     pred_A.append((np.sum(img) / np.sum(ys[i])))
    #     errors_A.append(abs(np.sum(img) / np.sum(ys[i]) - 1) * 100)
    #     pred_V.append((np.sum(cube) / vs[i]))
    #     errors_V.append(abs(np.sum(cube) / vs[i] - 1) * 100)
    #     counts.append(np.sum(Xs[i]))
    #     profs.append(get_radial_profile(Xs[i])[1])

    profs, counts = [], []
    for i in range(len(Xs)):
        profs.append(get_radial_profile(Xs[i], center=(63.5, 63.5))[1])
        counts.append(np.sum(Xs[i]))

    # print(np.mean(counts), np.std(counts))
    # print(np.mean(errors_A), np.std(errors_A))
    # print(np.mean(errors_V), np.std(errors_V))

    real = fits.getdata(glob.glob(f"real_data/rescaled/{galaxy}*.fits")[0])

    x, y = get_radial_profile(real)

    Ymed = np.median(profs, axis=0)
    Y16,Y1,Y3,Y84 = np.quantile(profs, (0.1, 0.25, 0.75, 0.99), axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title(f"{galaxy}\ncounts: {np.sum(real):.0f}, sim counts: {np.mean(counts):.0f} +/- {np.std(counts):.0f}")

    ax.fill_between(x[:len(Y1)], Y16, Y84, color="k", alpha=0.3)
    ax.fill_between(x[:len(Y1)], Y1, Y3, color="k", alpha=0.4)
    ax.plot(x[:len(Y1)], Ymed, lw=1.5, ls="-", ms=0, c="k")

    # plt.plot(y1)
    ax.plot(x, y)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ylim = ax.get_ylim()
    ax.set_ylim(bottom=max(1e-2, ylim[0]))

    fig.savefig(f"profiles/{which}_{galaxy}.png")
    # plt.show()


def get_radial_profile(data, center=None):
    if not center: center = np.array(data.shape) / 2
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    R = np.linspace(np.min(r), np.max(r), len(radialprofile))
    return R, radialprofile



########################## BETA MODELS ##########################

df_beta = pd.read_csv("beta_models_2.csv", index_col=0)

# df_beta_ellip = pd.read_csv("beta_models_ellip.csv", index_col=0)

# for par in [".ellip", ".ellip+", ".ellip-", ".theta", ".theta+", ".theta-"]:
#     for i in range(1,4):
#         df_beta[f"b{i}{par}"] = df_beta_ellip[f"b{i}{par}"] 
        
for g in df_beta.index:
    if df_beta["model"].loc[g] == "Single beta": n = 24
    elif df_beta["model"].loc[g] == "Double beta": n = 45
    else: n = len(df_beta["model"])
        
    for col in df_beta.columns[n:-6]:
        df_beta[col].loc[g] = 0

galaxies = pd.read_csv("Galaxies.csv")
gals = galaxies["Galaxy"][galaxies["Sample_C22"] == 1]
galaxies.index = galaxies["Galaxy"]
D_SBF = galaxies["D_SBF"]

df_beta["Size"] = df_beta["size"] // 128

############################ CAVITIES ############################

cavities = pd.read_csv("cavities_new.csv", index_col=0)
fac = 0.492

def run_heatmap(galaxy, generation):
    df = df_beta.loc[galaxy]

    N = 20

    df.Size = 1

    ampl = np.random.normal(df["b1.ampl"], df["b1.ampl+"], size=N)
    r0 = np.random.normal(df["b1.r0"], df["b1.r0+"], size=N)
    alpha = np.random.normal(df["b1.alpha"], df["b1.alpha+"], size=N)
    bkg = np.random.normal(df["bkg.c0"], df["bkg.c0+"], size=N)

    ampl_2 = np.random.normal(df["b2.ampl"], df["b2.ampl+"], size=N)
    r0_2 = np.random.normal(df["b2.r0"], df["b2.r0+"], size=N)
    alpha_2 = np.random.normal(df["b2.alpha"], df["b2.alpha+"], size=N)

    ampl_3 = np.random.normal(df["b3.ampl"], df["b3.ampl+"], size=N)
    r0_3 = np.random.normal(df["b3.r0"], df["b3.r0+"], size=N)
    alpha_3 = np.random.normal(df["b3.alpha"], df["b3.alpha+"], size=N)

    try:
        cav = cavities.loc[galaxy]
        cav = cav[cav["Generation"] == generation]

        # data = fits.getdata(f"{galaxy}_1.fits")
        # print("Number of counts", np.sum(data))

        fig, axs = error_heatmap(N=N, threshold=0.5, fac=3,
                                ampl=df["b1.ampl"]/df.Size**2, r0=df["b1.r0"]*df.Size, alpha=df["b1.alpha"], bkg=df["bkg.c0"],
                                ampl_2=df["b2.ampl"]/df.Size**2, r0_2=df["b2.r0"]*df.Size, alpha_2=df["b2.alpha"],
                                ampl_3=df["b3.ampl"]/df.Size**2, r0_3=df["b3.r0"]*df.Size, alpha_3=df["b3.alpha"],
                                radii=[8,11,14,17,20,23,26,29,32,35],
                                sizes=[7,10,13,16,19,22,25,28],
                                d1=cav["R_1"]/fac, d2=cav["R_2"]/fac,
                                s1=cav["RL_1"]/fac, s2=cav["RL_2"]/fac)
                                
        fig.savefig(f"heatmap_{galaxy}_{generation}.png", dpi=300, bbox_inches="tight")
    except:
        pass


    # which = df.model.split()[0].lower()

    # relative_error(galaxy, which, N=N, threshold=0.5, fac=3,
    #                 ampl=ampl/df.Size**2, r0=r0*df.Size, alpha=alpha, bkg=bkg,
    #                 ampl_2=ampl_2/df.Size**2, r0_2=r0_2*df.Size, alpha_2=alpha_2,
    #                 ampl_3=ampl_3/df.Size**2, r0_3=r0_3*df.Size, alpha_3=alpha_3)#,
    #                 # r1=cav["R_1"]/fac, r2=cav["R_2"]/fac,
    #                 # R1=cav["RL_1"]/fac, R2=cav["RL_2"]/fac)

gals = df_beta.index[df_beta["model"] == "Single beta"]
# gals = ["NGC1553", "NGC4291"]

for galaxy in gals:
    run_heatmap(galaxy, 1)
