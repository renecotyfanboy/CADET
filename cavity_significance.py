# basic libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatter
from scipy.odr import ODR, RealData, Model

# Astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord

# CIAO https://cxc.harvard.edu/ciao/
from ciao_contrib.runtool import *

q1, q3 = 0.15865525, 0.84134475


########################### SINGLE & DOUBLE BETA ###########################

def BETA_bkg(p, x):
	return p[0] * (1 + (x/p[1])**2)**(-3/2*p[2]) + p[3]

def DOUBLE_BETA_bkg(p, x):
	return p[0] * (1 + (x/p[1])**2)**(-3/2*p[2]) + p[3] * (1 + (x/p[4])**2)**(-3/2*p[5]) + p[6]


############################ MAKE FIT USING ODR ############################

def FIT_ODR(func, X, Y, sig_X, sig_Y, guess, errors=True, fixed=0):
	if errors: data = RealData(X, Y, sx=sig_X, sy=sig_Y)
	else: data = RealData(X, Y)
	model = Model(func)
	if not fixed: fixed = np.ones(len(guess))
	odr = ODR(data, model, beta0=guess, maxit=-1, ifixb=fixed)
	odr.set_job(fit_type=0)
	output = odr.run()
	return output.beta, output.sd_beta, output.cov_beta


############################## ANGLE TO SIDE ###############################

def angle_to_side(angle):
	if angle >= 337.5 or angle < 22.5: return "W"
	elif 22.5 <= angle < 67.5: return "NW"
	elif 67.5 <= angle < 112.5: return "N"
	elif 112.5 <= angle < 157.5: return "NE"
	elif 157.5 <= angle < 202.5: return "E"
	elif 202.5 <= angle < 247.5: return "SE"
	elif 247.5 <= angle < 292.5: return "S"
	elif 292.5 <= angle < 337.5: return "SW"


########################### CAVITIES into PIES ############################

def cavity_to_pie(galaxy, center, scale, cavities, N_radii=20, rmin=0.5, rmax=2.0):
    # GET RA/DEC COORDINATES of ELLIPSES and RA/DEC of CENTER
    coordinates = []
    for cavity in cavities:
        file = fits.open(f"{galaxy}/decomposed/{galaxy}_{scale}_{cavity}.fits")
        data = file[0].data
        wcs = WCS(file[0].header)
        x,y = np.nonzero(data)
        coordinates.append(wcs.wcs_pix2world(np.array([y,x]).T, 0).T)

    # PIES FOR AZIMUTHAL PROFILES
    sides, s1s, s2s, r1s, r2s = [], [], [], [], []
    for i, coords in zip(cavities, coordinates):
        angles, distances = [], []
        for c1, c2 in zip(*coords):
            cavity = SkyCoord(str(c1) + " degrees",str(c2) + " degrees", frame="fk5")

            angle = center.position_angle(cavity).deg
            angles.append((angle + 90))

            distances.append(center.separation(cavity).deg)

        angles = np.array(angles)
        if max(angles) - min(angles) > 180:
            angles = np.where(angles > 200, angles - 360, angles)
        if (angles > 360).all():
            angles = angles % 360

        s1, s2 = min(angles), max(angles)
    
        print(angles.std(), s2 - s1)

        N_angles = int(360 / (s2 - s1) * 3) + 1
        N_angles = max(N_angles, 8)

        width = 360 / N_angles
        side = np.mean(angles)
        s1s.append(s1);s2s.append(s2);sides.append(side)
        r1, r2 = min(distances) * 3600, max(distances) * 3600
        r1s.append(r1);r2s.append(r2)
        r1, r2 = r1 + (r2 - r1)*0.15, r2 - (r2 - r1)*0.1

        with open(f"{galaxy}/significance/azimuthal_{scale}_{i}.reg", "w") as file:
            for a in np.linspace(0, 360, N_angles + 1)[:-1]:
                a1, a2 = a + side - width / 2, a + side + width / 2
                ra = center.ra.to_string(u.hour).replace("h", ":").replace("m", ":").replace("s", "")
                dec = center.dec.to_string().replace("d", ":").replace("m", ":").replace("s", "")
                file.write(f"pie({ra}, {dec}, {r1:.1f}\", {r2:.1f}\", {a1}, {a2})\n")

    # PIES FOR RADIAL PROFILES
    s1, s2, s3, s4 = s1s[0], s2s[0], s1s[1], s2s[1]
    r1, r2, r3, r4 = r1s[0], r2s[0], r1s[1], r2s[1]

    # DECREASE CAVITY ANGLES by 10 PERCENT
    fac = 10 / 100
    s1_bkg, s2_bkg = s1, s2
    s3_bkg, s4_bkg = s3, s4
    s1, s2 = s1 + (s2-s1)*fac, s2 - (s2-s1)*fac
    s3, s4 = s3 + (s4-s3)*fac, s4 - (s4-s3)*fac

    rmax = max([r2, r4]) * rmax
    rmin = min([r1, r3]) * rmin

    # EQUIDISTANT RADII
    # rstep = (rmax - rmin) / N_radii
    # R = np.linspace(rmin, rmax - rstep, N_radii)

    # SQUARE ROOT RADII
    R = np.linspace(rmin**0.5, rmax**0.5, N_radii+1)**2

    for i,cavity in enumerate(cavities):
        radial = open(f"{galaxy}/significance/radial_{scale}_{cavity}.reg", "w")
        radial_bkg = open(f"{galaxy}/significance/radial_{scale}_{cavity}_bkg.reg", "w")

        for r1, r2 in zip(R[:-1], R[1:]):
            S1 = s1 if i == 0 else s3
            S2 = s2 if i == 0 else s4
            S1_bkg = s2_bkg if i == 0 else s4_bkg
            S2_bkg = s3_bkg if i == 0 else s1_bkg

            radial.write(f"pie({ra}, {dec}, {r1:.1f}\", {r2:.1f}\", {S1}, {S2})\n")
            radial_bkg.write(f"pie({ra}, {dec}, {r1:.1f}\", {r2:.1f}\", {S1_bkg}, {S2_bkg})\n")

        radial.close(); radial_bkg.close()

    return sides, s1s, s2s, r1s, r2s

########################## ANGULAR PLOT ###########################

def plot_angular(fname, side, s1, s2, r1, r2, ax1):
    # FIGURE SETTINGS
    fontsize, lw, labelpad = 16, 1.3, 10
    ax2 = ax1.twiny()
    ax1.tick_params(axis="both", which="major", length=8, width=lw, labelsize=fontsize)
    ax2.tick_params(axis="both", which="major", length=8, width=lw, labelsize=fontsize)

    # LOAD ANGULAR PROFILE FROM FITS
    data = fits.open(fname)[1].data
    a1, a2 = data["ROTANG"].T
    a, ae = (a1 + a2) / 2, abs((a2 - a1) / 2)
    a = a % 360
    y, ye = data["SUR_BRI"], data["SUR_BRI_ERR"]
    N = len(y)

    # SORT VALUES BY ANGLES & EXTEND ARRAYS
    i = np.argsort(a)
    a, ae, y, ye = a[i], ae[i], y[i], ye[i]
    A, AE = np.concatenate((a-360, a, a + 360)), np.concatenate((ae, ae, ae))
    Y, YE = np.concatenate((y,y,y)), np.concatenate((ye,ye,ye))

	# CAVITY DEPRESSION and SIGNIFICANCE
    c = 3 #4 #max(3, round(N / 4))

    # BACKGROUND LEVEL & ERROR ESTIMATION
    i1, i2 = A < s1, s2 < A
    bkgA1, bkgY1 = A[i1][-c:], Y[i1][-c:]
    bkgA2, bkgY2 = A[i2][:c], Y[i2][:c]

    Ss = np.concatenate((Y[i1][-c:], Y[i2][:c])).mean()
    Es = np.sqrt(np.concatenate((YE[i1][-c:]**2, YE[i2][:c]**2)).sum())

    # BACKGROUND POINTS IF CAVITY NEAR PLOT EDGE
    i1_2, i2_2 = (A - 360) < s1, s2 < (A - 360)
    bkgA1_2 = A[i1_2][-c:]
    bkgA2_2 = A[i2_2][:c]

    i1_3, i2_3 = (A + 360) < s1, s2 < (A + 360)
    bkgA1_3 = A[i1_3][-c:]
    bkgA2_3 = A[i2_3][:c]

    # CALCULATE SIGNIFICANCES
    indices = np.nonzero(np.logical_and(s1 < A, A < s2))[0]
    D, sign, A_sign, Y_sign = [], [], [], []
    sigmas, A_all = [], []
    for i in indices:     
        d = 1 - Y[i] / Ss
        sigma = d / ((1 - d) * np.sqrt((YE[i] / Y[i])**2 + (Es / Ss)**2))
        sigmas.append(sigma)
        A_all.append(A[i])
        if round(sigma, 1) >= 1:
            D.append(d * 100)
            sign.append(sigma)
            A_sign.append(A[i])
            Y_sign.append(Y[i])

    # PLOT CAVITY SIDE
    ax1.text(0.06, 0.94, angle_to_side(side)+" cavity", transform=ax1.transAxes, size=fontsize, ha="left", va="top")

    # PLOT DATA
    ax1.errorbar(A, Y, yerr=YE, marker="o", ms=5.5, elinewidth=lw, lw=lw, zorder=1)

    for i,n in enumerate(indices):
        if sigmas[i] < 1.0: color = "red"
        elif sigmas[i] < 3.0: color = "orange"
        elif sigmas[i] >= 3.0: color = "green"
        else: color = "black"
        ax1.plot(A[n], Y[n], "o", c=color, ms=6, zorder=2)

    # PLOT BACKGROUND DATAPOINTS
    color_bkg = "gray"
    ax1.plot(bkgA1, bkgY1, "o", c=color_bkg, ms=6, zorder=3)
    ax1.plot(bkgA2, bkgY2, "o", c=color_bkg, ms=6, zorder=3)

    try:
        ax1.plot(bkgA1_2, bkgY1, "o", c=color_bkg, ms=6, zorder=3)
        ax1.plot(bkgA2_2, bkgY2, "o", c=color_bkg, ms=6, zorder=3)
    except: pass

    try:
        ax1.plot(bkgA1_3, bkgY1, "o", c=color_bkg, ms=6, zorder=3)
        ax1.plot(bkgA2_3, bkgY2, "o", c=color_bkg, ms=6, zorder=3)
    except: pass

    # PLOT SIGNIFICANT RANGE
    try:
        if len(A_sign) == 1: sign1, sign2 = A_sign[0] - ae[0], A_sign[0] + ae[0]
        else: sign1, sign2 = min(A_sign)-ae[0], max(A_sign)+ae[0]
        # ax1.axvspan(sign1, sign2, alpha=0.5)
    except:
        sign1, sign2 = 359, 360

    # PLOT TEXT - DECREMENT & MAXIMAL SIGNIFICANCE
    for ai, y, s, d in zip(A_sign, Y_sign, sign, D):
        if s == max(sign):
            ax1.text(ai, y*0.8875, f"{s:.1f}$\sigma$", ha="center", va="center", size=fontsize)
            # ax1.text(ai, y*0.93, f"{d:.0f}$\,$%", ha="center", va="center", size=fontsize)

    # PLOT CAVITY BORDERS
    ax1.axvline(s1, ls="--", lw=lw, color="black")
    ax1.axvline(s2, ls="--", lw=lw, color="black")

    # AX1
    xticks = np.linspace(-360,720,25)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{i:.0f}°" for i in xticks], fontsize=fontsize)
    # ax1.set_yticklabels([f"{i:.2f}" for i in ax1.get_yticks()], fontsize=fontsize)
    
    xlim = np.array([0,360])
    if s2 > 360: xlim += 180
    ax1.set_xlim(xlim)
    ax1.set_ylim(ax1.get_ylim() * np.array([0.9, 1]))

    ax1.set_ylabel("surface brightness (counts s$^{-1}$ px$^{-2}$)", fontsize=fontsize, labelpad=labelpad)

    plt.setp(ax1.spines.values(), linewidth=1.2)
    plt.setp(ax2.spines.values(), linewidth=1.2)

    # AX2
    ax2.set_xticks(np.array([-360,-270,-180,-90,0,90,180,270,360,450,540,630,720]))
    ax2.set_xticklabels(["W", "N", "E", "S", "W", "N", "E", "S", "W", "N", "E", "S", "W"], fontsize=fontsize)
    ax2.set_xlim(ax1.get_xlim())

############################## RADIAL PLOT ##############################

def plot_prof(fnames, rmin1, rmin2, rmax1, rmax2, sides, axes, beta):
    fontsize, labelpad = 16, 8
    (ax1, ax2) = axes

    # LOAD BACKGROUND
    data = fits.open(fnames[1])[1].data
    r, re = data["R"].mean(1), data["R"].std(1)
    y1, y1e = data["SUR_BRI"], data["SUR_BRI_ERR"]

    data = fits.open(fnames[3])[1].data
    y2, y2e = data["SUR_BRI"], data["SUR_BRI_ERR"]

    # PLOT BACKGROUND DATA-POINTS
    y0, y0e = (y1 + y2) / 2, (y1e**2 + y2e**2)**0.5 / 2
    ax1.errorbar(r, y0, xerr=re, yerr=y0e, color="k",
                 lw=0, marker="o", elinewidth=1.5, label="azimuthal")

    # FIT BACKGROUND WITH SINGLE OR DOUBLE BETA MODEL
    if beta == "single": model = BETA_bkg
    elif beta == "double": model = DOUBLE_BETA_bkg

    beta_odr, beta_odr_e, beta_odr_cov = FIT_ODR(model, r, y0, re/3, y0e, [5, 20, 0.67, 0.1, 200, 0.7, 0.1])

    # GENERATE REALIZATIONS OF THE MODEL
    beta_pars = np.random.multivariate_normal(beta_odr, beta_odr_cov, 10000)
    beta_models = np.array([model(b, r) for b in beta_pars])
    
    beta_model = np.quantile(beta_models, 0.5, axis=0)
    ymin, ymax = np.quantile(beta_models, (q1, q3), axis=0)
    beta_model_e = (ymax - ymin) / 2

    # PLOT BETA MODEL + CONFIDENCE INTERVALS
    ax1.fill_between(r, ymin, ymax, color="gray", alpha=0.5)
    ax1.plot(r, beta_model, color="k", ls="-", lw=1.2, label="$\\beta$-model")

    # LOAD & PLOT CAVITY PROFILES
    sigmas = [[] for i in range(2)]
    for i, fname in enumerate(fnames[::2]):
        data = fits.open(fname)[1].data

        r, re = (data["R"][:,1]+data["R"][:,0]) / 2, (data["R"][:,1]-data["R"][:,0]) / 2
        y, ye = data["SUR_BRI"], data["SUR_BRI_ERR"]

        # CALCULATE SIGNIFICANCES
        indices = np.nonzero((rmin1 < r) & (r < rmax1))[0]
        for j in indices: #range(len(r)):
            Ss, Es = beta_model[j], beta_model_e[j]
            d = 1 - y[j] / beta_model[j]
            sigma = d / ((1 - d) * np.sqrt((ye[j] / y[j])**2 + (Es / Ss)**2))
            sigmas[i].append(sigma)

        name = f"{angle_to_side(sides[i])} cavity" + f" ({max(sigmas[i]):.1f}$\sigma$)"
        ax1.errorbar(r, y, xerr=re, yerr=ye, color=f"C{1-i}",
					 lw=0, marker="o", elinewidth=1.5, label=name)

    # PLOT CAVITY BORDERS
    ax1.axvline(rmin1/0.4928, color="C0", ls="--", lw=1.5)
    ax1.axvline(rmax1/0.4928, color="C0", ls="--", lw=1.5)
    ax1.axvline(rmin2/0.4928, color="C1", ls="--", lw=1.5)
    ax1.axvline(rmax2/0.4928, color="C1", ls="--", lw=1.5)

    h, l = ax1.get_legend_handles_labels()
    h = list(h[3:4]) + list(h[:3])[::-1]
    l = list(l[3:4]) + list(l[:3])[::-1]
    ax1.legend(h, l, frameon=False, 
               handlelength=1.5, loc="center left", fontsize=fontsize)

    plt.setp([ax1], xscale="log", yscale="log")
    xticks = np.array([1, 2, 5, 10, 20, 50, 100])
    xticks = xticks[(xticks >= r.min()) & (xticks <= r.max())]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])

    ax1.set_ylabel("surface brightness (counts s$^{-1}$ px$^{-2}$)", fontsize=fontsize, labelpad=labelpad)

    ax1.tick_params(axis="both", which="major", length=8, width=1.3, labelsize=fontsize)
    ax1.tick_params(axis="both", which="minor", length=4, width=1.1)

    plt.setp(ax1.spines.values(), linewidth=1.3)
    plt.setp(ax2.spines.values(), linewidth=1.3)

    # RESIDUAL PLOT
    for i, fname in enumerate(fnames[:2]):
        data = fits.open(fname)[1].data

        r, re = data["R"].mean(1), data["R"].std(1)
        y, ye = data["SUR_BRI"], data["SUR_BRI_ERR"]
        
        name = f"cav_{i+1}"
        ax2.errorbar(r, (y - y0) / ye, xerr=re, yerr=1, #ye / y0e,
					 lw=0, marker="o", elinewidth=1.5, label=name)

    ax2.axvline(rmin1/0.4928, color="C0", ls="--", lw=1.5)
    ax2.axvline(rmax1/0.4928, color="C0", ls="--", lw=1.5)
    ax2.axvline(rmin2/0.4928, color="C1", ls="--", lw=1.5)
    ax2.axvline(rmax2/0.4928, color="C1", ls="--", lw=1.5)

    ax2.set_ylabel("residual ($\sigma$)", fontsize=fontsize, labelpad=labelpad)
    ylim = max(abs(np.array(ax2.get_ylim()))) * 0.9
    ax2.set_ylim(-ylim, ylim)
    ax2.set_xscale("log")
    ax2.axhline(0, color="k", ls="--", lw=1.5)
    ax2.set_xlabel("radius (arcsec)", fontsize=fontsize, labelpad=labelpad)

    formatter = lambda x: [f"{i:.0f}" for i in x]
    xticks = np.array([1, 2, 5, 10, 20, 50, 100, 200])
    xticks = xticks[(xticks >= r.min()) & (xticks <= r.max())]
    ax2.set_xticks(xticks); ax2.set_xticklabels(xticks)
    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels(formatter(ax2.get_yticks()))

    ax2.tick_params(axis="both", which="major", length=8, width=1.3, labelsize=fontsize)
    ax2.tick_params(axis="both", which="minor", length=4, width=1.1)


########################## MAIN FUNCTION ##########################

def cavity_significance(galaxy, scale=1, cavities=[1,2], beta="single"):	
    # GET CENTRAL RA & DEC
    with fits.open(f"{galaxy}.fits") as file:
        data = file[0].data
        wcs = WCS(file[0].header)
        x0 = data.shape[0]//2
        RA, DEC = wcs.wcs_pix2world(np.array([[x0, x0]]),0)[0]
        center = SkyCoord(f"{RA} degrees", f"{DEC} degrees", frame="fk5")

    # CREATE SIGNIFICANCE FOLDERS
    os.system(f"mkdir -p {galaxy}/significance")
    path = f"{galaxy}/significance/"

    # TRANSFORM CAVITIES TO PIES
    sides, s1s, s2s, r1s, r2s = cavity_to_pie(galaxy, center, scale, cavities, N_radii=15, rmin=0.2, rmax=2.5)

    for cavity in cavities:
        region_file = f"{path}/azimuthal_{scale}_{cavity}.reg"
        fits_file = f"{path}/azimuthal_{scale}_{cavity}.fits"
        dmextract(f"{galaxy}.fits[bin sky=@{region_file}]",
                    fits_file, opt="generic", clobber=True)

        # RADIAL PROFILES
        for suf in ["", "_bkg"]:
            region_file = f"{path}/radial_{scale}_{cavity}{suf}.reg"
            fits_file = f"{path}/radial_{scale}_{cavity}{suf}.fits"
            dmextract(f"{galaxy}.fits[bin sky=@{region_file}]", 
                        fits_file, opt="generic", clobber=True)

    # FIGURE
    fig = plt.figure(figsize=(20,9))

    dx = 0.33
    axes1 = fig.add_axes([1.0-dx, 0.075, dx, 0.37])
    axes2 = fig.add_axes([1.0-dx, 0.575, dx, 0.37])

    axes3 = fig.add_axes([1.08, 0.068, 0.36, 0.21])
    axes4 = fig.add_axes([1.08, 0.31, 0.36, 0.63])

    axes = [axes3, axes4][::-1]

    # order = 1 if np.mean(cavities[0][1]) < np.mean(cavities[1][1]) else -1

    ax = [axes1, axes2]#[::order]
    axes = [axes3, axes4][::-1]

    fits_files = []
    for i,cavity in enumerate(cavities):
        fits_file = f"{path}/azimuthal_{scale}_{cavity}.fits"
        plot_angular(fits_file, sides[i], s1s[i], s2s[i], r1s[i], r2s[i], ax[i])

        fits_files.append(f"{path}/radial_{scale}_{cavity}.fits")
        fits_files.append(f"{path}/radial_{scale}_{cavity}_bkg.fits")

    plot_prof(fits_files, r1s[0], r1s[1], r2s[0], r2s[1], sides, axes, beta)

    fig.savefig(f"{galaxy}/{galaxy}_significance_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


if "__main__" == __name__:
    string = "\nError: Wrong number of arguments.\n"
    string += "Usage: python3 cavity_significance.py galaxy scale cavities [beta_model]\n"
    string += "Example: python3 CADET.py NGC4649 1 [1,2]\n"
    string += "Example: python3 CADET.py NGC4649 1 [1,2] single\n"
    string += "Example: python3 CADET.py NGC5813 2 [3,5] double\n"
    if len(sys.argv) < 4:
        print(string)

    elif len(sys.argv) == 4:
        cavity_significance(sys.argv[1], *[eval(arg) for arg in sys.argv[2:]])

    else:
        print(string)
