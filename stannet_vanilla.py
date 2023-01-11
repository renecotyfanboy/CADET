from tensorflow.keras.initializers import Constant, TruncatedNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Input, MaxPooling2D
from tensorflow.nn import local_response_normalization

##################### STAN's MODEL #####################

stride = (1, 1)

shapes_layers = [[32, (1,1)],
                 [32, (3,3)],
                 [10, (5,5)],
                 [3, (7,7)],
                 [1, (9,9)]]

shapes_layers_final = [[8, (8,8)],
                       [4, (16,16)],
                       [2, (32,32)]]

init_kernel = TruncatedNormal(mean=0.0, stddev=0.03, seed=420)
init_bias = Constant(value=0.0)

def block(data, filters, shapes):
    # INCEPTION LAYER
    # data = BatchNormalization(axis = -1)(data)
    layers = []
    for f, s in shapes:
        layers.append(Conv2D(filters = f, kernel_size = s, strides = stride,
                           kernel_initializer = init_kernel, bias_initializer = init_bias,
                           padding = "same", activation = None)(data))

    # CONCATENATE INCEPTION FILTERS
    layers = concatenate(layers, axis=-1)
    
    # CONSEQUENT CONV LAYER
    out_layer = Conv2D(filters = filters, kernel_size = (1,1), strides = stride, 
                       kernel_initializer = init_kernel, bias_initializer = init_bias, 
                       padding = "same", activation = None)(layers)
    
    if filters > 1:
        out_layer = BatchNormalization(axis = -1)(out_layer)
        out_layer = Activation("relu")(out_layer)
    else: out_layer = Activation("sigmoid")(out_layer)
    return out_layer

def Stannet(shape_image):
    input_data = Input(shape=(shape_image))
    data = BatchNormalization(axis = -1)(input_data)
    data = block(data, 32, shapes_layers)
    data = block(data, 64, shapes_layers)
    data = block(data, 64, shapes_layers)
    data = block(data, 32, shapes_layers)
    data = block(data, 1, shapes_layers_final)
    output = Model([input_data], data)
    return output
