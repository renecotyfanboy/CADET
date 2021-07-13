import os, sys, time
import numpy as np
import pandas as pd
from concurrent import futures

path = 'path/to/beta_model.py'
sys.path.insert(1, path)
import beta_model

gen = "CADET_search_params.csv"
path = ""

os.system("mkdir -p {0}/{1}_images_search {0}/{1}_binmasks_search".format(gen))

df = pd.read_csv("{0}_params.csv".format(gen))
N = len(df["image"])
nums = np.arange(0,N)

def func(i):
    r0 = df["r0"].loc[i]
    ampl = df["ampl"].loc[i]
    beta = df["beta"].loc[i]
    r02 = df["r02"].loc[i]
    if df["double"].loc[i]: ampl2 = df["ampl2"].loc[i]
    else: ampl2 = 0
    beta2 = df["beta2"].loc[i]
    ellip = df["ellip"].loc[i]
    phi = df["phi"].loc[i]
    axes = (1, 2)

    if df["sloshing"].loc[i]: sloshing = [df[j].loc[i] for j in ["slosh_depth","slosh_period","slosh_direction","slosh_angle"]]
    else: sloshing = [0, 0, 0, 0]

    if df["rims"].loc[i]: rims = [df[j].loc[i] for j in ["rims_size", "rims_height", "rims_type"]]
    else: rims = [0, 0, 0]

    if df["point"].loc[i]: point_source = [df[j].loc[i] for j in ["point_radius","point_ampl"]]
    else: point_source = [0, 0]

    B = beta_model.beta_model(64, 1, r0, ampl, beta, ampl2, r02, beta2, ellip, phi, axes, sloshing, point_source)

    r1 = df["r1"].loc[i]
    r2 = df["r2"].loc[i]
    phi1 = df["phi1"].loc[i]
    phi2 = df["phi2"].loc[i]
    theta1 = df["theta1"].loc[i]
    theta2 = df["theta2"].loc[i]
    R1 = df["R1"].loc[i]
    R2 = df["R2"].loc[i]
    e1 = df["e1"].loc[i]
    e2 = df["e2"].loc[i]
    varphi1 = [0, 0, df["varphi1_z"].loc[i]]
    varphi2 = [0, 0, df["varphi2_z"].loc[i]]

    dx, dy = df["dx"].loc[i], df["dy"].loc[i]
    B.dither(dx, dy)

    if df["cavities"].loc[i]: 
        B.cavity_pair([r1, r2], [phi1, phi2], [theta1, theta2], [R1, R2], [e1, e2], [varphi1, varphi2], rims)

    B.apply_noise()

    counts = np.sum(B.noisy)
    if counts > 10000:
        B.save_image("{0}/{1}_images_search".format(path, gen), i)#, fits=True)
        B.save_binary_mask("{0}/{1}_binmasks_search".format(path, gen), i)#, fits=True)

    if i % 1000 == 0: print(i, time.asctime())

def generator(n_cores=1):
    a = time.time()
    
    # FOR DEBUGGING
    #for i in nums:
    #    func(i)

    with futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        executor.map(func, nums)

    b = time.time()
    t = (b - a) / 3600
    h, m, s = t//1, t%1*60//1, t%1*60%1*60 
    print("elapsed time: {0:02d}:{1:02d}:{2:2.2f}".format(int(h),int(m),s))

if __name__ == '__main__':
    generator(n_cores=20)
