import os, glob, time, copy, datetime, sys, io, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
size = 12
plt.rc("mathtext", fontset="dejavuserif")
plt.rc('font', family='DejaVu Serif', size=size)

from astropy.io import fits
from astropy.nddata import CCDData

import scipy.cluster.hierarchy as hcluster
from scipy.ndimage import center_of_mass, rotate
from sklearn.cluster import KMeans, DBSCAN

import tensorflow as tf
from tensorflow.keras import callbacks, metrics, losses
from tensorflow.keras.applications import resnet50, DenseNet201, ResNet50V2
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.initializers import Constant, TruncatedNormal, RandomNormal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Dense, Dropout, Input, LeakyReLU, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, normalize

import warnings
warnings.filterwarnings('ignore') # :-)

# GPU initialization
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0],
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


# architecture of the CADET network
shape_image = (128, 128, 1)

stride = (1, 1)

# shapes_layers = [[32, (1,1)],
#                  [32, (3,3)],
#                  [10, (5,5)],
#                  [3, (7,7)],
#                  [1, (9,9)]]

shapes_layers = [[32, (1,1)],
                 [32, (3,3)],
                 [16, (5,5)],
                 [8, (7,7)],
                 [4, (9,9)],
                 [2, (11,11)],
                 [1, (13,13)]]

shapes_layers_final = [[8, (8,8)],
                       [4, (16,16)],
                       [2, (32,32)],
                       [1, (64,64)]]

shapes_blocks = [32, 64, 64, 32, 1]

init_kernel = TruncatedNormal(mean=0.0, stddev=0.03, seed=None)
init_bias = Constant(value=0.0)

def block(data, filters, shapes):
    layers = []
    for f, s in shapes:
        layers.append(Conv2D(filters = f, kernel_size = s, strides = stride,
                             kernel_initializer = init_kernel, bias_initializer = init_bias, #"constant",
                             padding = "same", activation = None)(data))
    layers.append(MaxPooling2D(pool_size=(1,1), strides=stride, padding="same")(data))
    layers = concatenate(layers, axis=-1)
    
    out_layer = Conv2D(filters = filters, kernel_size = (1,1), strides = stride, 
                       kernel_initializer = init_kernel, bias_initializer = init_bias, 
                       #padding = "same", activation = LeakyReLU(alpha=hp["relu"]))(layers)
                       padding = "same", activation = None)(layers)
    
    if filters > 1:
        out_layer = BatchNormalization(axis = -1, momentum = 0.9, epsilon = 0.001, scale = True)(out_layer)
        out_layer = Activation(LeakyReLU(alpha=hp["relu"]))(out_layer)
    else: out_layer = Activation("sigmoid")(out_layer)
    return out_layer

def network(shape_image):
    input_data = Input(shape=(shape_image))
    data = BatchNormalization(axis = -1, scale=True)(input_data)
    data = block(data, 32, shapes_layers)
    data = block(data, 64, shapes_layers)
    data = block(data, 64, shapes_layers)
    data = block(data, 32, shapes_layers)
    data = block(data, 1, shapes_layers_final)
    output = Model([input_data], data)
    return output

# image loading, normaliation, rebinning
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def read_data(nums, x, batch):
    X, y, weights, volume1, volume2, area1, area2 = [], [], [], [], [], [], []

    for i in range(batch):
        i = nums[x+i]
        img = np.load(path + "{0}_images/{1}_img.npy".format(gen,i))
        # c = shape_image[0] // 2
        # s = 2
        # img[c-s:c+s+1, c-s:c+s+1] = 0

        R1 = df["R1"].loc[i]
        R2 = df["R2"].loc[i]
        e1 = df["e1"].loc[i]
        e2 = df["e2"].loc[i]

        res = 2
        rx1 = R1 / res
        ry1 = R1 * (1 - e1) / res
        rz1 = max(rx1, ry1)
        S1 = np.pi * rx1 * ry1
        V1 = 4 * np.pi / 3 * rx1 * ry1 * rz1
        volume1.append(V1)
        area1.append(S1)

        rx2 = R2 / res
        ry2 = R2 * (1 - e2) / res
        rz2 = max(rx2, ry2)
        S2 = np.pi * rx2 * ry2
        V2 = 4 * np.pi / 3 * rx2 * ry2 * rz2
        volume2.append(V2)
        area2.append(S2)

        #img = np.log10(img+1) / np.amax(np.log10(img+1))
        #img = np.log10(img+1)
        X.append(np.log10(img+1) / np.amax(np.log10(img+1)))
        y.append(np.load(path + "{0}_binmasks/{1}_mask.npy".format(gen,i)))

    X = np.array(X).reshape(batch, *shape_image)
    y = np.array(y).reshape(batch, *shape_image)
    #X = (X - np.mean(X)) / np.std(X)
    #X = normalize(X, axis=0)

    return X, y#, volume1, volume2, area1, area2

# image generator
def img_generator(ids, batch = 32):
    x = 0
    while True:
        X, y = read_data(ids, x, batch)
        x += batch
        if x+batch >= len(ids):
            np.random.shuffle(ids)
            x = 0
        yield (X, y)

# 
path = "/home/tplsek/Artificial galaxies/"
gen = "gen2.4"
epochs = 5
N_val = 10000 // 10
N_test = 10000 // 10
N_train = 270000 // 100

df = pd.read_csv("{0}_params.csv".format(gen))
for c in df.columns:
    try: df[c] = df[c].astype(float)
    except: pass

ids = glob.glob(path+gen+"_images/*_img.npy")
ids = [int(i.split("_")[1].split("/")[1]) for i in ids]
print(len(ids))

hp = {"relu" : 0.0,
      "lr" : 0.0005,
      "batch" : 8,
      "beta1" : 0.9,
      "beta2" : 0.999,
      "adam_epsilon" : 1e-8,
      "decay" : 0.000}

# df = pd.read_csv("{0}_params.csv".format(gen))
# for c in df.columns:
#     try: df[c] = df[c].astype(float)
#     except: pass

# text data
#X_test, y_test, v1_test, v2_test, s1_test, s2_test = read_data(ids, N_train+N_val, N_test)
X_test, y_test = read_data(ids, N_train+N_val, N_test)
    
# validation data
#X_val, y_val, _, _, _, _ = read_data(ids, N_train, N_val)
X_val, y_val = read_data(ids, N_train, N_val)

# dirs and filenames
timestamp = "b{0}_lr{1}_d{2}_b{3}_final_topinka".format(hp["batch"], hp["lr"], hp["decay"], hp["beta1"])
log_dir = "tmp/{0}/{1}".format(gen, timestamp)
weight_dir = "weights/{0}/{1}".format(gen,timestamp)
models_dir = "models/{0}/{1}".format(gen,timestamp)

# Tensorflow callbacks
callbacks = [TensorBoard(log_dir=log_dir, update_freq="batch", write_graph=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                               min_lr=0.0, mode="min", verbose=1)]

# model compilation
def my_loss(y_true, y_pred):
    # y_true = tf.dtypes.cast(y_true, tf.float32)
    # sq_diff = tf.reduce_sum(tf.abs(y_true - y_pred), axis=(1,2))
    # sq_loss = sq_diff / tf.reduce_sum(y_true, axis=(1,2))
    y_true = tf.dtypes.cast(y_true, tf.float32)
    sq_diff = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    sq_loss = sq_diff / tf.reduce_sum(y_true)
    return tf.reduce_mean(sq_diff, axis=-1)

model = network(shape_image)
adam_opt = Adam(lr=hp["lr"], beta_1=hp["beta1"], beta_2=hp["beta2"],
                epsilon=hp["adam_epsilon"], decay=hp["decay"])
#model.compile(optimizer=adam_opt, loss=my_loss, metrics=["binary_accuracy"])
model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=["binary_accuracy"])

#ids = ids[:360000]
if 1:
    model.fit(img_generator(ids, batch = hp["batch"]),
              validation_data = (X_val, y_val),
              epochs=epochs,
              steps_per_epoch = N_train / hp["batch"],
              callbacks = callbacks, verbose=1)

    model.save_weights(weight_dir)
    model.save(models_dir+".keras")

# else:
# #    model.load_weights(weight_dir)
#     model = load_model("model_size.keras", custom_objects = {"LeakyReLU": LeakyReLU})
# #    model = load_model(models_dir+".keras", custom_objects = {"LeakyReLU": LeakyReLU})

# #    model.save("model_search.keras")

#     score = model.evaluate(X_test, y_test, batch_size = hp["batch"])
#     y_pred = model.predict(X_test)

#     def make_cube(image):
#         cen = center_of_mass(image)
#         x, y = [], []
#         for i in range(128):
#             for j in range(128):
#                 if image[i,j]:
#                     x.append(i-64)
#                     y.append(j-64)
#         r = np.sqrt((cen[0])**2+(cen[1])**2)
#         phi = np.pi-np.arccos(((cen[1])/r))+np.pi

#         image = rotate(image, -phi*180/np.pi, reshape=False, prefilter=False)
#         ranges = np.arange(0,128)
#         means, widths, indices = [], [], []
#         for i in range(128):
#             slices = image[:,i]
#             rang = np.where(slices > 0, ranges, 0)
#             if (rang > 0).any():
#                 b = rang > 0
#                 mean = np.mean(rang[b])
#     #			width = max(rang[b])-min(rang[b])+1
#                 width = sum(b)
#                 if mean: 
#                     indices.append(i)
#                     means.append(abs(mean))
#                     widths.append(width/2)

#         cube = np.zeros((128,128,128))
#         for m, w, i in zip(means, widths, indices):
#             x, y = np.indices((128, 128))
#             r = np.sqrt((x-m)**2 + (y-64)**2)
#             sliced = np.where(r < w-1.55, 1, 0)
#             cube[:,:,i] = sliced
#         return np.sum(cube)


#     ymins = np.linspace(0.05, 0.95, 10)
#     #ymins = [0.55]
#     acc_abs, acc_pm, acc_abs_med, acc_pm_med, acc_abs_std, acc_pm_std = [], [], [], [], [], []
#     s_acc_abs, s_acc_pm, s_acc_abs_med, s_acc_pm_med, s_acc_abs_std, s_acc_pm_std = [], [], [], [], [], []
#     for ymin in ymins:
#         v1_pred, v2_pred, s1_pred, s2_pred = [], [], [], []
#         for i, Y_pred in enumerate(y_pred):
#             Y_pred = np.where(Y_pred > ymin, Y_pred, 0)
#             x, y = Y_pred.reshape(128,128).nonzero()
#             data = np.array([x,y]).reshape(2, -1)
            
#             try:
#                 #clusters = KMeans(n_clusters=2).fit_transform(data.T)
#                 clusters = DBSCAN(eps=3, min_samples=2).fit(data.T)
#                 clusters = clusters.labels_
#                 #clusters = hcluster.fclusterdata(data.T, dmin, criterion="distance")#, method="complete")
#             except:
#                 clusters = np.zeros(len(x))

#             cavs = []
#             N = len(set(clusters))
#             if 0 < N < 10:
#                 for j in range(N):
#                     img = np.zeros((128,128))
#                     b = clusters == j
#                     xi, yi = x[b], y[b] 
#                     img[xi,yi] = 1
#                     #img[xi,yi] = Y_pred.reshape(128,128)[xi,yi]
#                     #img = np.where(img > ymin, 1, 0)
#                     #if np.sum(img) > 5: 
#                     cavs.append(img)
#             elif N == 0:
#                 cavs = [np.zeros((128,128)), np.zeros((128,128))]
#             else:
#                 img = np.zeros((128,128))
#                 img[x,y] = 1
#                 cavs.append(img)

#             vols, areas = [], []
#             for cav in cavs:
#                 areas.append(np.sum(cav))
#                 vols.append(make_cube(cav))
#             if len(vols) < 2:
#                 #print("tady")
#                 areas = [areas[0], areas[0]]
#                 vols = [vols[0], vols[0]]

#             v1_pred.append(sorted(vols)[-1])
#             v2_pred.append(sorted(vols)[-2])
#             s1_pred.append(sorted(areas)[-1])
#             s2_pred.append(sorted(areas)[-2])

#     #     test_areas = np.array([v1_test, v2_test])
#     #     s1_test, s2_test = test_areas.max(0), test_areas.min(0)
#     #     s1_pred, s2_pred = np.array(v1_pred), np.array(v2_pred)
#     #     test, pred = np.concatenate((s1_test, s2_test)), np.concatenate((s1_pred, s2_pred))
#     #     plt.figure(figsize=(6.2,5))
#     #     #plt.plot(test, pred, ".")
#     #     plt.hist2d(test/1e4, pred/1e4, (30, 30), ((0, max(test/1e4)), (0, max(test/1e4))), norm=LogNorm(vmin=1))
#     #     plt.plot([0, max(test/1e4)], [0, max(test/1e4)], "--k", alpha=0.6)
#     #     plt.colorbar(pad=0.02, shrink=1.0)
#     #     plt.xlabel("true volume ($10^4$ pixels$^3$)")
#     #     plt.ylabel("predicted volume ($10^4$ pixels$^3$)")
#     #     #plt.show()
#     #     plt.savefig("true_vs_pred.pdf", bbox_inches="tight")

#         test_vols = np.array([v1_test, v2_test])
#         v1_test, v2_test = test_vols.max(0), test_vols.min(0)
#         v1_pred, v2_pred = np.array(v1_pred), np.array(v2_pred)

#         test_areas = np.array([s1_test, s2_test])
#         s1_test, s2_test = test_areas.max(0), test_areas.min(0)
#         s1_pred, s2_pred = np.array(s1_pred), np.array(s2_pred)

#         #for i in range(len(v1_pred)):
#         #    print(v1_pred[i], v1_test[i])

#         res_abs = abs(v1_pred - v1_test) / v1_test
#         res = (v1_pred - v1_test) / v1_test

#         b = (-1.0 < res) & (res < 1.0)
#         gross = sum(np.logical_not(b)) / len(b)

#         acc_abs_med.append(np.median(res_abs[b]))
#         acc_pm_med.append(np.median(res[b]))

#         acc_abs.append(np.mean(res_abs[b]))
#         acc_pm.append(np.mean(res[b]))

#         acc_abs_std.append([acc_abs_med[-1]-np.quantile(res_abs[b],0.25), np.quantile(res_abs[b],0.75)-acc_abs_med[-1]])
#         acc_pm_std.append([acc_pm_med[-1]-np.quantile(res[b],0.25), np.quantile(res[b],0.75)-acc_pm_med[-1]])

#         print(ymin)
#         # if ymin.round(2) in [0.50, 0.55]:
#         #     print(ymin, acc_abs_med[-1], acc_abs_std[-1], acc_pm_med[-1], acc_pm_std[-1])

#         #print("volume gross error: ", gross*100, "%")

#         res_abs = abs(s1_pred - s1_test) / s1_test
#         res = (s1_pred - s1_test) / s1_test

#         b = (-1.0 < res) & (res < 1.0)
#         gross = sum(np.logical_not(b)) / len(b)

#         s_acc_abs_med.append(np.median(res_abs[b]))
#         s_acc_pm_med.append(np.median(res[b]))

#         s_acc_abs.append(np.mean(res_abs[b]))
#         s_acc_pm.append(np.mean(res[b]))

#         #s_acc_abs_std.append(np.std(res_abs[b]))
#         #s_acc_pm_std.append(np.std(res[b]))

#         s_acc_abs_std.append([s_acc_abs_med[-1]-np.quantile(res_abs[b],0.25), np.quantile(res_abs[b],0.75)-s_acc_abs_med[-1]])
#         s_acc_pm_std.append([s_acc_pm_med[-1]-np.quantile(res[b],0.25), np.quantile(res[b],0.75)-s_acc_pm_med[-1]])

#  #       if ymin.round(2) in [0.50, 0.55]:
#  #           print(ymin, s_acc_abs_med[-1], s_acc_abs_std[-1], s_acc_pm_med[-1], s_acc_pm_std[-1])
#         #print("area gross error: ", gross*100, "%")

#         #b = res < 1.0
#         #print(np.median(res[b]))
#         #plt.hist(res[b])
#         #plt.show()

#         #b = res_abs < 1.0
#         #print(np.median(res_abs[b]))
#         #plt.hist(res_abs[b])
#         #plt.show()

# #    plt.xkcd()
#     #plt.title("Volume")
#     plt.rc("lines", lw=1, marker="s", markersize=5)
#     fig, ax = plt.subplots()
#     #plt.errorbar(ymins, np.array(acc_abs)*100, yerr=np.array(acc_abs_std)*100, elinewidth=1, capsize=3, label="mean, absolute")
#     #plt.errorbar(ymins, np.array(acc_pm)*100, yerr=np.array(acc_pm_std)*100, elinewidth=1, capsize=3, label="mean")
#     ax.errorbar(ymins, np.array(acc_pm_med)*100, yerr=np.array(acc_pm_std).T*100, elinewidth=1, capsize=3, label="relative error",color="C0")
#     #plt.errorbar(ymins, np.array(acc_abs_med)*100, yerr=np.array(acc_abs_std).T*100, elinewidth=1, capsize=3, label="absolute relative error",color="C0")
#     ax.plot([min(ymins)-0.05, max(ymins)+0.05], [0, 0], "--k", lw=1)
#     ax.set_xlim(min(ymins)-0.05, max(ymins)+0.05)
#     ax.text(0.992, max(acc_pm_med+abs(np.array(acc_pm_std)).T.max(0)) * 6, "overestimated", va="center", size=10, weight="normal", ha="right")
#     ax.text(0.992, -max(acc_pm_med+abs(np.array(acc_pm_std)).T.max(0)) * 7, "underestimated", va="center", size=10, weight="normal", ha="right")
#     #ax.text(0.93, 0.9, "for threshold 0.55:", ha="right", size=11, transform=ax.transAxes)
#     #ax.text(0.93, 0.82, "abs. rel. error = ${0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}$".format(acc_pm_med[5]*100, acc_pm_std[5][0]*100, acc_pm_std[5][1]*100),\
#     #        ha="right", size=11, transform=ax.transAxes)
#     #ax.text(0.93, 0.74, "rel. error = ${0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}$".format(acc_abs_med[5]*100, acc_abs_std[5][0]*100, acc_abs_std[5][1]*100),\
#     #        ha="right", size=11, transform=ax.transAxes)
#     ax.set_xticks(ymins)
#     #plt.legend(loc="lower left", frameon=False)
#     #plt.grid()
#     ax.set_xlabel("discrimination threshold")
#     ax.set_ylabel("$\left(V_{\mathrm{pred}} - V_{\mathrm{ture}}\\right) \, / \, V_{\mathrm{true}}$ [%]", labelpad=0.02)
#     plt.savefig("volumes.pdf", bbox_inches="tight")
#     plt.show()

#     #plt.title("Area")
#     #plt.errorbar(ymins, np.array(s_acc_abs)*100, yerr=(s_acc_abs_std)*100, elinewidth=1, capsize=3, label="mean, absolute")
#     #plt.errorbar(ymins, np.array(s_acc_pm)*100, yerr=(s_acc_pm_std)*100, elinewidth=1, capsize=3, label="mean")
#     plt.errorbar(ymins, np.array(s_acc_pm_med)*100, yerr=np.array(s_acc_pm_std).T*100, elinewidth=1, capsize=3, label="relative error",color="C1")
#     plt.errorbar(ymins, np.array(s_acc_abs_med)*100, yerr=np.array(s_acc_abs_std).T*100, elinewidth=1, capsize=3, label="absolute relative error",color="C0")
#     plt.plot([min(ymins)-0.05, max(ymins)+0.05], [0, 0], "--k", lw=1)
#     plt.xlim(min(ymins)-0.05, max(ymins)+0.05)
#     plt.text(0.992, max(s_acc_pm_med+abs(np.array(s_acc_pm_std)).T.max(0)) * 5, "overestimated", va="center", size=10, weight="normal", ha="right")
#     plt.text(0.992, -max(s_acc_pm_med+abs(np.array(s_acc_pm_std)).T.max(0)) * 5, "underestimated", va="center", size=10, weight="normal", ha="right")
#     plt.xticks(ymins)
#     plt.legend(loc="lower left", frameon=False)
#     #plt.grid()
#     plt.xlabel("discrimination threshold")
#     plt.ylabel("$\left(S_{\mathrm{pred}} - S_{\mathrm{ture}}\\right) \, / \, S_{\mathrm{true}}$ [%]", labelpad=0.02)
#     plt.savefig("areas.pdf", bbox_inches="tight")
#     plt.show()

    # galaxies = {"IC4296" : [150, 150, 1],
    #             "NGC507" : [351, 351, 0.25],
    #             "NGC708" : [320, 320, 0.5],
    #             "NGC1316" : [518, 520, 1],
    #             "NGC1399" : [604, 604, 0.25],
    #             "NGC1407" : [250, 250, 0.5],
    #             "NGC1600" : [449, 449, 1],
    #             "NGC4261" : [200, 200, 0.5],
    #             "NGC4374" : [415, 415, 0.5],
    #             "NGC4472" : [401, 400, 1],
    #             "NGC4472_2" : [401, 400, 0.25],
    #             "NGC4486" : [650, 650, 0.5],
    #             "NGC4552" : [460, 461, 1],
    #             "NGC4636" : [373, 370, 1],
    #             "NGC4636_2" : [371, 370, 0.25],
    #             "NGC4649" : [300, 300, 1],
    #             "NGC4696" : [460, 450, 1],
    #             "NGC4778" : [350, 350, 0.5],
    #             "NGC5044" : [350, 350, 1],
    #             "NGC5044_2" : [350, 350, 0.5],
    #             "NGC5813" : [371, 369, 1],
    #             "NGC5813_2" : [371, 364, 0.5],
    #             "NGC5846" : [350, 350, 1],
    #             "NGC6166" : [223, 223, 0.5]}

    # def read_gal(name, xy):
    #     file = fits.open("real_data/{0}.fits".format(name))
    #     data = file[0].data
    #     size = int(64 / xy[2])
    #     crop = data[xy[1]-size:xy[1]+size, xy[0]-size:xy[0]+size]
    #     crop = rebin(crop, (128,128))
    #     # c = shape_image[0] // 2
    #     # s = 2
    #     # crop[c-s:c+s+1, c-s:c+s+1] = 0
    #     crop = np.log10(crop+1) / np.amax(np.log10(crop+1))
    #     return crop

    # def plot_to_image(figure):
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     plt.close(figure)
    #     buf.seek(0)
    #     image = tf.image.decode_png(buf.getvalue(), channels=4)
    #     image = tf.expand_dims(image, 0)
    #     return image

    # file_writer = tf.summary.create_file_writer(log_dir + "/validation")

    # def plot_testgal_prediction(img, y_pred, y_true):
    #     fig = plt.figure(figsize=(10,5))
    #     plt.subplot(1,3,1); plt.axis("off")
    #     plt.imshow(img.reshape(*shape_image[:-1]), origin="lower")
    #     plt.subplot(1,3,2); plt.axis("off")
    #     plt.imshow(y_pred.reshape(*shape_image[:-1]), origin="lower", norm=Normalize(vmin=0, vmax=1))
    #     plt.subplot(1,3,3); plt.axis("off")
    #     plt.imshow(y_true.reshape(*shape_image[:-1]), origin="lower")
    #     return fig 

    # def plot_realgal_prediction(gal, img, y):
    #     fig = plt.figure(figsize=(10,5))
    #     plt.suptitle(gal, x=0.51, y=0.9, size=16)
    #     plt.subplot(1,2,1); plt.axis("off")
    #     plt.imshow(img.reshape(*shape_image[:-1]), origin="lower")
    #     plt.subplot(1,2,2); plt.axis("off")
    #     plt.imshow(y.reshape(*shape_image[:-1]), origin="lower")
    #     return fig 

    # for i in range(20):
    #     img = X_val[i]
    #     y_pred = model.predict(img.reshape(1, *shape_image))
    #     y_true = y_val[i]

    #     with file_writer.as_default():
    #         tf.summary.image("Test data", plot_to_image(plot_testgal_prediction(img, y_pred, y_true)), step=i)

    # for i, gal in enumerate(galaxies.keys()):
    #     print(gal)
    #     X = read_gal(gal, galaxies[gal])

    #     img = np.log10(X+1)
    #     img = img / np.amax(img)
    #     y = model.predict(img.reshape(1, *shape_image))

    #     with file_writer.as_default():
    #         tf.summary.image("Real galaxies", plot_to_image(plot_realgal_prediction(gal, img, y)), step=i)