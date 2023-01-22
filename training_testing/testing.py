import glob, io, tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from astropy.io import fits
from scipy.interpolate import interp1d

from tensorflow import convert_to_tensor

# custom libraries
from functions import *
from beta_model import beta_model

q1, q3 = 0.25, 0.75


####################### TEST DATA ####################### 

suf = "_final" #_flatter"
images = glob.glob(f"mock_data{suf}/images/*.npy")
images_nocav = glob.glob(f"mock_data{suf}/images_nocav/*.npy")

images = sorted(images, key=lambda x: int(x.split("/")[-1].split("_")[0]))
images_nocav = sorted(images_nocav, key=lambda x: int(x.split("/")[-1].split("_")[0]))


def load_img(n, nocav=False):
    name = images[n].split("/")[-1]
    vol = float(name.split(".")[0].split("_")[1])
    
    if nocav: X = np.load(images_nocav[n]).reshape(128,128)
    else: X = np.load(images[n]).reshape(128,128)
    
    if nocav: y = np.zeros((128,128))
    else: y = np.load(f"mock_data{suf}/masks/" + name).reshape(128,128)
    return X, y, vol


def load_test_data(N, N0=0, nocav=False):
    X_test, y_test, v_test, c_test = [], [], [], []
    for i in range(N0, N0 + N):
        X, y, v = load_img(i, nocav=nocav)
        c_test.append(np.sum(X))
        X = np.log10(X+1) #/ np.max(np.log10(X+1))
        X_test.append(X)
        y_test.append(y)
        v_test.append(v)

    X = convert_to_tensor(np.array(X_test).reshape(N, 128, 128, 1))
    y = convert_to_tensor(np.array(y_test).reshape(N, 128, 128, 1))

    return X, y, np.array(v_test), np.array(c_test)


def get_areas_n_volumes(y_pred, y_test, v_test, threshold):
    cavs, _, _, _ = decompose(np.where(y_pred > threshold, 1, 0))
    # cavs = decompose_two(np.where(y_pred > threshold, 1, 0))

    img, cube, tp = np.zeros((128,128)), np.zeros((128,128,128)), 0
    for i1, cav in enumerate(cavs):
        if ((cav>0) & (y_test>0)).any():
        # if np.sum((cav>0) & (y_test>0)) >= np.sum(y_test) * 0.5 * 0.1:
            img += cav
            cube += make_cube(cav)
            tp += 0.5

    Ar = np.sum(y_test)
    Ap = np.sum(img)
    Vr = v_test
    Vp = np.sum(cube)

    return Ar, Ap, Vr, Vp, tp


def get_error(y_pred, y_test, v_test, c_test, bins, optimal):
    N = y_pred.shape[0]
    A, Ae, V, Ve, TP = np.empty((N, 2)), np.empty(N), np.empty((N, 2)), np.empty(N), np.empty(N)

    for i in tqdm.tqdm(range(N)):
        for j in range(len(optimal)):
            if bins[j] < c_test[i] < bins[j+1]: threshold = optimal[j]
            else: threshold = optimal[-1]

        Ar, Ap, Vr, Vp, tp = get_areas_n_volumes(y_pred[i], y_test[i], v_test[i], threshold)

        A[i] = [Ar, Ap]
        V[i] = [Vr, Vp]
        Ae[i] = (Ap / Ar) #* 100
        Ve[i] = (Vp / Vr) #* 100
        TP[i] = tp

    return A, Ae, V, Ve, TP


def get_threshold_error(y_pred, y_test, v_test, thresholds):
    N, Nth = y_pred.shape[0], len(thresholds)
    Ae, Ve, TP = np.empty((Nth, N)), np.empty((Nth, N)), np.empty((Nth, N))

    for j,th in enumerate(tqdm.tqdm(thresholds)):
        for i in range(len(y_test)):
            Ar, Ap, Vr, Vp, tp = get_areas_n_volumes(y_pred[i], y_test[i], v_test[i], th)

            Ae[j,i] = (Ap / Ar - 1) * 100
            Ve[j,i] = (Vp / Vr - 1) * 100
            TP[j,i] = tp

    return Ae, Ve, TP


def get_false_positives(y_pred, threshold=0.5):
    N = y_pred.shape[0]
    FP = np.empty(N)

    for i in range(N):
        pred = np.where(y_pred[i] > threshold, 1, 0)
        cavs, _, _, _ = decompose(pred)
        if pred.any() and (len(cavs) > 1):
           FP[i] = 1
        else:
           FP[i] = 0

    return FP


def get_optimal_false_positives(y_pred, c_test, bins, optimal):
    N = y_pred.shape[0]
    FP = np.empty(N)

    for i in tqdm.tqdm(range(N)):
        for j in range(len(optimal)):
            if bins[j] < c_test[i] < bins[j+1]: threshold = optimal[j]
        else: threshold = optimal[-1]
        pred = np.where(y_pred[i] > threshold, 1, 0)
        cavs, _, _, _ = decompose(pred)
        if pred.any() and (len(cavs) > 1):
           FP[i] = 1
        else:
           FP[i] = 0

    return FP


def plot_error_matrix(array, log=False, name="volume (pixels$^{\\text{3}}$)", bins=25):
    x, y = array[:,0], array[:,1]
    if log: x, y = x[(x > 0) & (y > 0)], y[(x > 0) & (y > 0)]

    MIN, MAX = np.min(x), np.max(x) * 0.7
    if "volume" in name: MAX = np.max(x) * 0.5
    if log: bins = np.logspace(np.log10(MIN), np.log10(MAX), bins)
    else: bins = np.linspace(0, MAX, bins)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.hist2d(x, y, bins=bins, norm=LogNorm());

    if log: ax.plot([MIN, MAX], [MIN, MAX], "--", color="k")
    else: ax.plot([0, MAX], [0, MAX], "--", color="k")

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(MIN, MAX)
        ax.set_ylim(MIN, MAX)
    else:
        ax.set_xlim(0, MAX)
        ax.set_ylim(0, MAX)

    ax.set_xlabel(f"true {name}")
    ax.set_ylabel(f"predicted {name}")

    if log: x_formatter = FuncFormatter(lambda y, _: "10$^{{\\text{{{:.2g}}}}}$".format(np.log10(y)))
    else: x_formatter = FuncFormatter(lambda y, _: "{:.0f}".format(y))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(x_formatter)

    ax.tick_params(axis="both", which="major", length=5, width=1.3)
    ax.tick_params(axis="both", which="minor", length=3, width=1.1)
    plt.setp(ax.spines.values(), linewidth=1.3)

    return fig


def plot_error_vs_counts(Ae, Ve, TP, TP_counts, FP, FP_counts, N_bins):
    alpha = 0.5

    counts = np.concatenate((TP_counts, FP_counts))
    MIN, MAX = np.min(counts), np.max(counts)
    bins = np.logspace(np.log10(MIN), np.log10(MAX), N_bins+1)
    # bins = np.quantile(counts, np.linspace(0, 1, N_bins+1))
    x = bins[1:] - bins[:-1]
    x[0], x[-1] = MIN, MAX

    # GRAPH
    fig, ax = plt.subplots(figsize=(7,5))
    ax2 = ax.twinx()

    p1 = []
    for n, errors in enumerate([Ve, Ae]):
        color = f"C{n}"
        label = "volume" if n == 0 else "area"

        # BIN BY COUNTS
        binned_errors = [[] for i in range(N_bins)]
        for e, tp, c in zip(errors, TP, TP_counts):
            for i in range(N_bins):
                if bins[i] < c < bins[i+1]:
                    if tp > 0:
                        binned_errors[i].append(e)

        # MEDIAN & ERRORBARS
        median, Q1, Q3 = np.empty(N_bins), np.empty(N_bins), np.empty(N_bins)
        for i,err in enumerate(binned_errors):
            median[i] = np.median(err)
            (q_1, q_3) = (q1, q3) if n == 0 else (0.25, 0.75)
            Q1[i] = np.quantile(err, q_1)
            Q3[i] = np.quantile(err, q_3)

        # ax.plot(x, median, color=color, zorder=n+1)
        p1.append(ax.fill_between(x, (Q1-1)*100, (Q3-1)*100, color=color, alpha=alpha, zorder=n+1))

    binned_TP, binned_TP_N = np.zeros(N_bins), np.zeros(N_bins)
    for c, tp in zip(TP_counts, TP):
        for i in range(N_bins):
            if bins[i] < c < bins[i+1]:
                binned_TP[i] += tp
                binned_TP_N[i] += 1

    binned_FP, binned_FP_N = np.zeros(N_bins), np.ones(N_bins)
    for c, fp in zip(FP_counts, FP):
        for i in range(N_bins):
            if bins[i] < c < bins[i+1]:
                binned_FP[i] += fp
                binned_FP_N[i] += 1

    p1.append(ax2.plot(x, binned_TP/binned_TP_N, color="C2", lw=1.5)[0])
    p1.append(ax2.plot(x, binned_FP/binned_FP_N, color="C3", lw=1.5)[0])

    ax.legend(p1, ["volume error", "area error", "TP rate", "FP rate"], 
              fontsize=13, handlelength=1.2, ncol=4, columnspacing=1.3, 
              loc=(0.03, 1.03)) #"upper center")
    ax.axhline(0, ls="--", color="black", lw=1.3)
    # ax.axhline(1, ls="--", color="black", lw=1.3)

    ax.set_xscale("log")
    # ax.invert_xaxis()
    # ax.set_ylim(0, 1.3)
    ax.set_ylim(-60, 60)
    ax2.set_ylim(0, 1)

    x_formatter = FuncFormatter(lambda y, _: "10$^{{\\text{{{:.2g}}}}}$".format(np.log10(y)))
    y_formatter = FuncFormatter(lambda y, _: "{:.0f}".format(y))
    y2_formatter = FuncFormatter(lambda y, _: "{:.1f}".format(y))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax2.yaxis.set_major_formatter(y2_formatter)

    ax.tick_params(axis="both", which="major", length=5, width=1.3)
    ax.tick_params(axis="both", which="minor", length=3, width=1.1)
    ax2.tick_params(axis="both", which="major", length=5, width=1.3)
    ax2.tick_params(axis="both", which="minor", length=3, width=1.1)
    plt.setp(ax.spines.values(), linewidth=1.3)

    ax.set_xlabel("image counts")
    ax.set_ylabel("relative error (\%)", labelpad=10)
    ax2.set_ylabel("rate", labelpad=10)

    return fig


def plot_discrimination_threshold_by_counts(df, TP, counts, thresholds, N_bins=2, name="volume"):
    alpha = 0.5
    
    xticks = np.linspace(0.1, 0.9, 9)
    bins = np.quantile(counts, np.linspace(0, 1, N_bins+1))

    df_N = [[] for i in range(N_bins)]
    TP_N = [[] for i in range(N_bins)]
    c_bins = []
    for i, c1, c2 in zip(range(N_bins), bins[:-1], bins[1:]):
        c_bins.append([c1, c2])
        for d,tp in zip(df, TP):
            df_N[i].append(d[(c1 < counts) & (counts <= c2)])
            TP_N[i].append(tp[(c1 < counts) & (counts <= c2)])

    # GRAPH
    fig, ax = plt.subplots(figsize=(7,4.5))

    for n,df,tp in zip(range(N_bins), df_N, TP_N):
        # MEDIAN & ERRORBARS
        median, Q1, Q3 = [], [], []
        for i in range(len(thresholds)):
            b = tp[i] == 1
            median.append(np.median(df[i][b]))
            Q1.append(np.quantile(df[i][b], q1))
            Q3.append(np.quantile(df[i][b], q3))

        # ax.plot(thresholds, median, color=f"C{n}", marker="")
        c1, c2 = int(round(c_bins[n][0], -3)), int(round(c_bins[n][1], -3))
        label = f"{c1}$-${c2} counts" if c1 < 50000 else f"$>$ {c1} counts"
        
        color = f"C{n}"
        if N_bins == 1: ax.plot(thresholds, median, color=color)
        ax.fill_between(thresholds, Q1, Q3, color=color, alpha=alpha, label=label)
    
    ax.axhline(0, ls="--", color="black", lw=1.3)
    if N_bins > 1: ax.legend(handlelength=1.4)

    lim = max(abs(np.array(ax.get_ylim())))
    lim = min(lim, 100)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0f}".format(y)))

    ax.tick_params(axis="both", which="major", length=5, width=1.3)
    ax.tick_params(axis="both", which="minor", length=3, width=1.1)
    plt.setp(ax.spines.values(), linewidth=1.3)

    ax.set_xlabel("discrimination threshold", labelpad=8)
    ylabel = lambda x: f"$\\textit{{{x}}}_{{\\text{{pred}}}} \, / \, \\textit{{{x}}}_{{\\text{{true}}}} - \\text{{1}} \; \\text{{(\%)}}$"
    ax.set_ylabel(ylabel("V" if name == "volume" else "A"), labelpad=8)

    return fig


def plot_discrimination_threshold_by_betas(df, TP, betas, thresholds, N_bins=2, name="volume"):
    alpha = 0.5

    xticks = np.linspace(0.1, 0.9, 9)
    bins = np.quantile(betas, np.linspace(0, 1, N_bins+1))

    df_N = [[] for i in range(N_bins)]
    TP_N = [[] for i in range(N_bins)]
    c_bins = []
    for i, c1, c2 in zip(range(N_bins), bins[:-1], bins[1:]):
        c_bins.append([c1, c2])
        for d,tp in zip(df, TP):
            df_N[i].append(d[(c1 < betas) & (betas <= c2)])
            TP_N[i].append(tp[(c1 < betas) & (betas <= c2)])

    # GRAPH
    fig, ax = plt.subplots(figsize=(7,4.5))

    for n,df,tp in zip(range(N_bins), df_N, TP_N):
        # MEDIAN & ERRORBARS
        median, Q1, Q3 = [], [], []
        for i in range(len(thresholds)):
            b = tp[i] == 1
            median.append(np.median(df[i][b]))
            Q1.append(np.quantile(df[i][b], q1))
            Q3.append(np.quantile(df[i][b], q3))

        c1, c2 = c_bins[n][0], c_bins[n][1]
        label = f"beta: {c1:.2f}-{c2:.2f}"

        color = f"C{n}"
        if N_bins == 1: ax.plot(thresholds, median, color=color)
        ax.fill_between(thresholds, Q1, Q3, color=color, alpha=alpha, 
                        label=label, zorder=N_bins-n)

    ax.axhline(0, ls="--", color="black", lw=1)
    ax.legend()

    lim = max(abs(np.array(ax.get_ylim())))
    lim = min(lim, 100)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0f}".format(y)))

    ax.set_xlabel("discrimination threshold", labelpad=8)
    ylabel = lambda x: f"$\\textit{{{x}}}_{{\\text{{pred}}}} \, / \, \\textit{{{x}}}_{{\\text{{true}}}} - \\text{{1}} \; \\text{{(\%)}}$"
    ax.set_ylabel(ylabel("V" if name == "volume" else "A"), labelpad=8)

    return fig


def plot_optimal_threshold_vs_counts(Ae, Ve, TP, FP, TP_counts, FP_counts, thresholds, N_bins=10, name="volume"):
    alpha = 0.5
    color = f"C0"

    log_counts = np.log10(np.concatenate([TP_counts, FP_counts]))
    TP_log_counts = np.log10(TP_counts)
    FP_log_counts = np.log10(FP_counts)

    bins = []
    Ae_N = [[] for i in range(N_bins)]
    Ve_N = [[] for i in range(N_bins)]
    TP_N = [[] for i in range(N_bins)]
    FP_N = [[] for i in range(N_bins)]
    for i in range(N_bins):
        c1, c2 = np.quantile(log_counts, ((i)/N_bins, (i+1)/N_bins))
        bins.append(c1)
        for A, V, tp, fp in zip(Ae, Ve, TP, FP):
            Ae_N[i].append(A[(c1 < TP_log_counts) & (TP_log_counts < c2)])
            Ve_N[i].append(V[(c1 < TP_log_counts) & (TP_log_counts < c2)])
            TP_N[i].append(tp[(c1 < TP_log_counts) & (TP_log_counts < c2)])
            FP_N[i].append(fp[(c1 < FP_log_counts) & (FP_log_counts < c2)])
    bins.append(max(log_counts))
    bins = np.array(bins)
    x_bins = (bins[1:] + bins[:-1]) / 2

    # GRAPH
    fig, ax = plt.subplots(figsize=(7,4.5))

    # print(df_N)

    A_optimal, V_optimal, FP_optimal, TP_optimal = [], [], [], []
    for n, A, V, tp, fp in zip(range(N_bins), Ae_N, Ve_N, TP_N, FP_N):
        # MEDIAN & ERRORBARS
        Amedian, AQ1, AQ3 = [], [], []
        Vmedian, VQ1, VQ3 = [], [], []
        FP, TP = [], []
        for i in range(len(thresholds)):
            b = tp[i] == 1
            Amedian.append(np.median(A[i][b]))
            AQ1.append(np.quantile(A[i][b], q1))
            AQ3.append(np.quantile(A[i][b], q3))

            Vmedian.append(np.median(V[i][b]))
            VQ1.append(np.quantile(V[i][b], q1))
            VQ3.append(np.quantile(V[i][b], q3))

            TP.append(sum(tp[i]) / len(tp[i]))
            FP.append(sum(fp[i]) / len(fp[i]))

        opt = float(interp1d(Amedian, thresholds, fill_value="extrapolate")(0))
        A_optimal.append(max(0, min(1, opt)))

        opt = float(interp1d(Vmedian, thresholds, fill_value="extrapolate")(0))
        V_optimal.append(max(0, min(1, opt)))

        FP_threshold = 0.05
        try: FP_optimal.append(float(interp1d(FP, thresholds)(FP_threshold)))
        except: FP_optimal.append(0.1)

        TP_threshold = 0.8
        try: TP_optimal.append(float(interp1d(TP, thresholds)(TP_threshold)))
        except: 
            if (TP > np.array(TP_threshold)).all(): TP_optimal.append(0.9)
            else: TP_optimal.append(0.1)

    x_bins  = 10**x_bins
    ax.plot(x_bins, A_optimal, color=f"C1", marker="o", lw=1.3, ls="-", label="area")
    ax.plot(x_bins, V_optimal, color=f"C0", marker="o", lw=1.3, ls="-", label="volume")
    ax.plot(x_bins, FP_optimal, color=f"C3", marker="o", lw=1.3, ls="-", label="FP (5\%)")
    ax.plot(x_bins, TP_optimal, color=f"C2", marker="o", lw=1.3, ls="-", label="TP (80\%)")

    ax.legend()

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: "10$^{{\\text{{{:.2g}}}}}$".format(np.log10(y))))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}".format(y)))

    ax.set_xlabel("image counts", labelpad=8)
    ax.set_ylabel("optimal threshold", labelpad=8)

    ax.tick_params(axis="both", which="major", length=5, width=1.3)
    ax.tick_params(axis="both", which="minor", length=3, width=1.1)
    plt.setp(ax.spines.values(), linewidth=1.3)

    optimal = V_optimal if name == "volume" else A_optimal

    return fig, 10**bins, optimal, FP_optimal


def plot_fprate_vs_discrimination_threshold_by_counts(FP, FP_counts, TP, TP_counts, thresholds, N_bins):
    alpha = 0.5
    color = f"C0"
    xticks = np.linspace(0.1, 0.9, 9)

    counts = np.concatenate((FP_counts, TP_counts))
    bins = np.quantile(counts, np.linspace(0, 1, N_bins+1))

    # GRAPH
    fig, ax = plt.subplots(figsize=(7,4.5))

    color = 0
    for c1,c2 in zip(bins[:-1], bins[1:]):
        FPs, TPs = [], []
        for i in range(len(thresholds)):

            fp = FP[i][(c1 < FP_counts) & (FP_counts < c2)]
            tp = TP[i][(c1 < TP_counts) & (TP_counts < c2)]

            FPs.append(sum(fp) / len(fp))
            TPs.append(sum(tp) / len(tp))

        c1, c2 = int(round(c1, -3)), int(round(c2, -3))
        label_TP = f"{c1}$-${c2} counts" if c2 < bins[-1] else f"$>$ {c1} counts"
        label_FP = f"{c1}$-${c2} counts" if c2 < bins[-1] else f"$>$ {c1} counts"

        ax.plot(thresholds, FPs, ls="--", color=f"C{color}", label=label_FP)
        ax.plot(thresholds, TPs, ls="-", color=f"C{color}", label=label_TP)
        color += 1

    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::2], l[::2])

    # legend1 = plt.legend(np.array([h[::2], h[1::2]]).T, ["FP", "TP"], loc=1)

    # h1, l1 = np.array([h[2*i:2*i+2] for i in range(len(h)//2)]), np.array(l[::2])
    # print(h1.shape, l1.shape)
    # plt.legend([h[2*i:2*i+2] for i in range(len(h)//2)], l[::2] , loc=4)
    # plt.gca().add_artist(legend1)

    ax.set_ylim(0, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}".format(y)))

    ax.tick_params(axis="both", which="major", length=5, width=1.3)
    ax.tick_params(axis="both", which="minor", length=3, width=1.1)
    plt.setp(ax.spines.values(), linewidth=1.3)

    ax.set_xlabel("discrimination threshold", labelpad=8)
    ax.set_ylabel("rate", labelpad=8)

    return fig


def plot_error_fprate_vs_discrimination_threshold(Ae, Ve, FP, TP, thresholds):
    alpha = 0.5
    color = f"C0"
    xticks = np.linspace(0.1, 0.9, 9)

    # GRAPH
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax2 = ax.twinx()

    Amedian, AQ1, AQ3 = [], [], []
    Vmedian, VQ1, VQ3 = [], [], []
    for i in range(len(thresholds)):
        b = TP[i] == 1
        Amedian.append(np.median(Ae[i][b]))
        AQ1.append(np.quantile(Ae[i][b], 0.25))
        AQ3.append(np.quantile(Ae[i][b], 0.75))
        Vmedian.append(np.median(Ve[i][b]))
        VQ1.append(np.quantile(Ve[i][b], q1))
        VQ3.append(np.quantile(Ve[i][b], q3))

    p = []
    p.append(ax.fill_between(thresholds, AQ1, AQ3, color=f"C1", alpha=alpha, zorder=2, label="area error"))
    p.append(ax.fill_between(thresholds, VQ1, VQ3, color=f"C0", alpha=alpha, zorder=1, label="volume error"))
    ax.axhline(0, ls="--", color="k", zorder=10)

    fp, tp = [], []
    for i in range(len(thresholds)):
        fp.append(sum(FP[i]) / len(FP[i]))
        tp.append(sum(TP[i]) / len(TP[i]))

    p.append(ax2.plot(thresholds, fp, ls="-", color=f"C3", label="FP rate", zorder=3)[0])
    p.append(ax2.plot(thresholds, tp, ls="-", color=f"C2", label="TP rate", zorder=4)[0])

    # ax.legend(loc="lower left", handlelength=1.4)
    # ax2.legend(loc="upper right", handlelength=1.35)

    ax.legend(p, ["area error", "volume error", "FP rate", "TP rate"],
              handlelength=1.2, ncol=4, columnspacing=1.3, fontsize=13,
              loc=(0.03, 1.03)) #"upper center")

    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-100, 100)
    ax2.set_ylim(0, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0f}".format(y)))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}".format(y)))

    ax.tick_params(axis="both", which="major", length=5, width=1.3)
    ax.tick_params(axis="both", which="minor", length=3, width=1.1)
    ax2.tick_params(axis="both", which="major", length=5, width=1.3)
    ax2.tick_params(axis="both", which="minor", length=3, width=1.1)
    plt.setp(ax.spines.values(), linewidth=1.3)

    ax.set_xlabel("discrimination threshold", labelpad=8)
    ax.set_ylabel("relative error (\%)", labelpad=8)
    ax2.set_ylabel("rate", labelpad=8)

    return fig


def plot_testgal_prediction(gal, img, y_pred, y_true, beta):
    fig = plt.figure(figsize=(10,5))

    plt.title(f"beta = {beta:.2f}")

    plt.subplot(1,3,1); plt.axis("off")
    plt.title(f"beta = {beta:.2f}")
    plt.imshow(img, origin="lower")

    plt.subplot(1,3,2); plt.axis("off")
    plt.imshow(y_pred, norm=Normalize(0,1), origin="lower")

    plt.subplot(1,3,3); plt.axis("off")
    plt.imshow(y_true, origin="lower")

    return fig 


####################### REAL DATA ####################### 

def read_realgals(real_gals):
    real_imgs = []
    for i, gal in enumerate(tqdm.tqdm(real_gals)):
        data = fits.getdata(gal)
        if gal.split("_")[1] in ["IC4296", "NGC507", "NGC1407", "NGC1399", "NGC4261"]:
            data = np.log10(data + 1)
        data = np.log10(data+1) #/ np.amax(np.log10(data+1))
        real_imgs.append(data)

    real_imgs = convert_to_tensor(real_imgs)

    return real_imgs


def plot_realgal_prediction(gal, img, y):
    fig = plt.figure(figsize=(10,5))
    plt.suptitle(gal, x=0.51, y=0.9, size=16)

    plt.subplot(1,2,1); plt.axis("off")
    plt.imshow(img, origin="lower")

    plt.subplot(1,2,2); plt.axis("off")
    plt.imshow(y, norm=Normalize(0,1), origin="lower")

    return fig 


###################### CUSTOM DATA ###################### 

def get_los_angle_error(N_angles=10, N=50, A=1, r0=10, beta=0.5, r=25, R=15):
    dphi = 90 // (N_angles-1)

    Xs = [[] for i in range(N_angles)]
    ys = [[] for i in range(N_angles)]
    As = [[] for i in range(N_angles)]
    Vs = [[] for i in range(N_angles)]
    counts = []

    for n in range(N_angles):
        for i in range(N):
            seed = np.random.randint(100000)
            varphi1 = np.random.uniform(0, 360)
            varphi2 = varphi1 + 180

            X, y, v = beta_model(seed, A=A, r0=r0, beta=beta, 
                                 theta1=n*dphi, theta2=n*dphi, 
                                 varphi1=varphi1, varphi2=varphi2,
                                 R1=R, R2=R, r1=r, r2=r)

            X = np.log10(X+1) #/ np.max(np.log10(X+1))

            Xs[n].append(X.reshape(128,128))
            ys[n].append(y.reshape(128,128))
            As[n].append(np.pi * (R**2 + R**2))
            Vs[n].append(v)

    return np.array(Xs), np.array(ys), np.array(As), np.array(Vs)


def plot_los_angle(N_angles, N, y_test, y_pred, As, Vs, bins, optimal):
    A = [[] for i in range(N_angles)]
    V = [[] for i in range(N_angles)]

    counts = np.sum(y_test[0])
    threshold = optimal[-1]
    for j in range(len(optimal)):
        if bins[j] < counts < bins[j+1]: threshold = optimal[j]

    for n in range(N_angles):
        for i in range(N):
            Ar, Ap, Vr, Vp, _ = get_areas_n_volumes(y_pred[n,i], y_test[n,i], Vs[n,i], threshold=0.5)
            A[n].append(Ap / As[n,i])
            V[n].append(Vp / Vs[n,i])
    A, V = np.array(A), np.array(V)

    x = np.linspace(0, 90, N_angles)

    fig, ax = plt.subplots(figsize=(7,5))

    medA, err = np.mean(A, axis=1), np.std(A, axis=1)
    # ax.plot(x, medA, marker="", color="C0", lw=1.3, zorder=4)
    ax.fill_between(x, medA-err, medA+err, color="C1", label="area", alpha=0.6, zorder=2)

    medV, err = np.mean(V, axis=1), np.std(V, axis=1)
    # ax.plot(x, medV, marker="", color="C1", lw=1.3, zorder=4)
    ax.fill_between(x, medV-err, medV+err, color="C0", label="volume", alpha=0.6, zorder=1)

    ax.axhline(1, ls="--", lw=1.3, color="k", zorder=5, alpha=0.9)
    ax.legend(handlelength=2, loc="center right") #loc=(0.6, 0.65))

    ax.set_xticks([0, 15, 30, 45, 60, 75, 90]);
    ax.set_xlim(0, 90)

    ax.set_xlabel("angle (degrees)", labelpad=6)
    ax.set_ylabel("predicted / true", labelpad=6)
    ax.tick_params(axis='both', which='major', length=5, width=1.3)
    ax.tick_params(axis='both', which='minor', length=0)

    formatter = FuncFormatter(lambda y, _: "{:.0f}".format(y))
    ax.xaxis.set_major_formatter(formatter)
    formatter = FuncFormatter(lambda y, _: "{:.2f}".format(y))
    ax.yaxis.set_major_formatter(formatter)

    return fig


def get_radii_error_per_los_angle(angles=[0, 20, 40], N_radii=10, N=50, A=1, r0=10, beta=0.5, r=10, R=8):
    A = np.ones(N) * A if type(A) in [int, float] else A
    r0 = np.ones(N) * r0 if type(r0) in [int, float] else r0
    beta = np.ones(N) * beta if type(beta) in [int, float] else beta

    ratio = R**2 / r
    dr = np.linspace(r, 50, N_radii)

    Xs = np.empty((len(angles), N_radii, N, 128, 128))
    ys = np.empty((len(angles), N_radii, N, 128, 128))
    As = np.empty((len(angles), N_radii, N))
    Vs = np.empty((len(angles), N_radii, N))
    counts = []

    for i,a in enumerate(angles):
        for j in range(N_radii):
            for k in range(N):
                seed = np.random.randint(100000)
                varphi1 = np.random.uniform(0, 360)
                varphi2 = varphi1 + 180 + np.random.normal(0, 20)

                R = np.sqrt(ratio * dr[j]) # constant A/r ratio
                X, y, v = beta_model(seed, A=A[k], r0=r0[k], beta=beta[k], 
                                     theta1=a, theta2=-a, 
                                     varphi1=varphi1, varphi2=varphi2,
                                     R1=R, R2=R, r1=dr[j], r2=dr[j])

                X = np.log10(X+1) #/ np.max(np.log10(X+1))

                Xs[i,j,k] = X.reshape(128,128)
                ys[i,j,k] = y.reshape(128,128)
                As[i,j,k] = np.pi * (R**2 + R**2)
                Vs[i,j,k] = v

    return Xs, ys, As, Vs, dr


def plot_radii_per_los_angle(angles, N_radii, N, y_test, y_pred, As, Vs, dr):
    fig, ax = plt.subplots(figsize=(7,5))

    for i in range(len(angles)):
        A = np.empty((N_radii, N))
        V = np.empty((N_radii, N))
        color = f"C{i}"

        for j in range(N_radii):
            for k in range(N):
                Ar, Ap, Vr, Vp, _ = get_areas_n_volumes(y_pred[i,j,k], y_test[i,j,k], Vs[i,j,k], threshold=0.5)
                A[j,k] = Ap / As[i,j,k]
                V[j,k] = Vp / Vs[i,j,k]

        medV, (Q1, Q3) = np.median(V, axis=1), np.quantile(V, (q1,q3), axis=1)
        # ax.plot(x, medV, marker="", color="C1", lw=1.3, zorder=4)
        ax.fill_between(dr, Q1, Q3, color=color, label=f"{angles[i]} degrees", alpha=0.6, zorder=1)

    ax.axhline(1, ls="--", lw=1.3, color="k", zorder=5, alpha=0.9)
    ax.legend(handlelength=2, loc="upper right") #loc=(0.6, 0.65))

    # ax.set_xticks([0, 15, 30, 45, 60, 75, 90]);
    ax.set_xlim(min(dr), max(dr))

    ax.set_xlabel("distance (pixels)", labelpad=6)
    ax.set_ylabel("estimated / true", labelpad=6)
    ax.tick_params(axis='both', which='major', length=5, width=1.3)
    ax.tick_params(axis='both', which='minor', length=0)

    formatter = FuncFormatter(lambda y, _: "{:.0f}".format(y))
    ax.xaxis.set_major_formatter(formatter)
    formatter = FuncFormatter(lambda y, _: "{:.2f}".format(y))
    ax.yaxis.set_major_formatter(formatter)

    return fig


