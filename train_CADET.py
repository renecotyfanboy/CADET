import glob
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.initializers import Constant, TruncatedNormal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Input, LeakyReLU, MaxPooling2D
from tensorflow.keras.optimizers import Adam

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

init_kernel = TruncatedNormal(mean=0.0, stddev=0.03, seed=None)
init_bias = Constant(value=0.0)

def block(data, filters, shapes):
    layers = []
    for f, s in shapes:
        layers.append(Conv2D(filters = f, kernel_size = s, strides = stride,
                             kernel_initializer = init_kernel, bias_initializer = init_bias,
                             padding = "same", activation = None)(data))
    layers.append(MaxPooling2D(pool_size=(1,1), strides=stride, padding="same")(data))
    layers = concatenate(layers, axis=-1)
    
    out_layer = Conv2D(filters = filters, kernel_size = (1,1), strides = stride, 
                       kernel_initializer = init_kernel, bias_initializer = init_bias, 
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


# data reading function
def read_data(nums, x, batch):
    X, y = [], []

    for i in range(batch):
        i = nums[x+i]
        img = np.load(path + "{0}_images/{1}_img.npy".format(gen,i))

        X.append(np.log10(img+1) / np.amax(np.log10(img+1)))
        y.append(np.load(path + "{0}_binmasks/{1}_mask.npy".format(gen,i)))

    X = np.array(X).reshape(batch, *shape_image)
    y = np.array(y).reshape(batch, *shape_image)

    return X, y

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

# basic params
path = "path_to_training_images"
gen = "generation_of_data"
epochs = 5
N_val = 10000
N_test = 10000
N_train = 300000

ids = glob.glob(path+gen+"_images/*_img.npy")
ids = [int(i.split("_")[1].split("/")[1]) for i in ids]
print(len(ids))

# hyperparameters
hp = {"relu" : 0.0,
      "lr" : 0.0005,
      "batch" : 8,
      "beta1" : 0.9,
      "beta2" : 0.999,
      "adam_epsilon" : 1e-8,
      "decay" : 0.000}

# text data
X_test, y_test = read_data(ids, N_train+N_val, N_test)
    
# validation data
X_val, y_val = read_data(ids, N_train, N_val)

# directories and filenames
timestamp = "b{0}_lr{1}_d{2}_b{3}_final".format(hp["batch"], hp["lr"], hp["decay"], hp["beta1"])
log_dir = "tmp/{0}/{1}".format(gen, timestamp)
models_dir = "models/{0}/{1}".format(gen,timestamp)

# Tensorflow callbacks
callbacks = [TensorBoard(log_dir=log_dir, update_freq="batch", write_graph=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                               min_lr=0.0, mode="min", verbose=1)]

# model compilation
def my_loss(y_true, y_pred):
    y_true = tf.dtypes.cast(y_true, tf.float32)
    sq_diff = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    sq_loss = sq_diff / tf.reduce_sum(y_true)
    return tf.reduce_mean(sq_loss, axis=-1)

model = network(shape_image)
adam_opt = Adam(lr=hp["lr"], beta_1=hp["beta1"], beta_2=hp["beta2"],
                epsilon=hp["adam_epsilon"], decay=hp["decay"])
#model.compile(optimizer=adam_opt, loss=my_loss, metrics=["binary_accuracy"])
model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=["binary_accuracy"])

# fitting the model
model.fit(img_generator(ids, batch = hp["batch"]),
            validation_data = (X_val, y_val),
            epochs=epochs,
            steps_per_epoch = N_train / hp["batch"],
            callbacks = callbacks, verbose=1)

model.save(models_dir+".keras")