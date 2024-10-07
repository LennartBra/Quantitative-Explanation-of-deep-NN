#---------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------
import numpy as np
import os

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from tensorflow import squeeze
from tensorflow.keras.layers import Input, Conv1D, Permute, ReLU, BatchNormalization, Add, SpatialDropout1D, AveragePooling1D, MaxPooling1D, Flatten, Dense, GRU, concatenate, Dropout, LSTM
from keras.regularizers import l1, l2
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import tensorflow as tf

keras.backend.clear_session()
#---------------------------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------------------------
l2_subnet = 0.001
l2_resnet = 0.001
act_feat = "relu"
act_interp = "relu"


def resnet_block(x, n_conv, num_filter, strides=True):
    def conv_blocks(x, n_conv, num_filter):
        for i in range(n_conv):
            x = Conv1D(kernel_size=3, filters=num_filter, padding="same")(x)
            x = SpatialDropout1D(rate=0.2)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        return x

    def shortcut(x, num_filter):
        x = Conv1D(kernel_size=3, filters=num_filter, padding="same")(x)
        x = BatchNormalization()(x)
        return x

    outp_long = conv_blocks(x, n_conv=n_conv, num_filter=num_filter)
    outp_short = shortcut(x, num_filter=num_filter)
    outp = Add()([outp_long, outp_short])
    
    if strides==True:
    	outp = Conv1D(kernel_size=5, filters=num_filter, strides=2, padding="same")(outp)
    
    return outp

def signal_block(x, filters):
    x = resnet_block(x, 2, filters[0])
    x = resnet_block(x, 2,filters[1])
    x = GRU(128, return_sequences=True)(x)
    x = GRU(128, return_sequences=False)(x)
    x = Dense(256, activation=act_interp, kernel_regularizer=l2(l2_resnet))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
   
    return x

def subnet_x(x):
    x = Dense(256, activation=act_interp, kernel_regularizer=l2(l2_resnet))(x)
    x = Dense(256, activation=act_interp, kernel_regularizer=l2(l2_resnet))(x)
    
    x1 = Dense(128, activation=act_interp, kernel_regularizer=l1(l2_resnet))(x)
    x1 = Dense(64, activation=act_interp, kernel_regularizer=l1(l2_resnet))(x1)
    x1 = Dense(32, activation=act_interp, kernel_regularizer=l1(l2_resnet))(x1)
    
    x2 = Dense(128, activation=act_interp, kernel_regularizer=l1(l2_resnet))(x)
    x2 = Dense(64, activation=act_interp, kernel_regularizer=l1(l2_resnet))(x2)
    x2 = Dense(32, activation=act_interp, kernel_regularizer=l1(l2_resnet))(x2)
    
    outp_sbp = Dense(1, activation="linear", name="sbp_output")(x1)
    outp_dbp = Dense(1, activation="linear", name="dbp_output")(x2)
    outp = concatenate([outp_sbp, outp_dbp], axis=-1)

    return outp


def concat_block(x1, x2):
    x = concatenate([x1, x2], axis=-1)
    x = Dense(256, activation="relu")(x)
    outp = Dense(128, activation="relu")(x)

    return outp

def make_model():
    inp_time0 = Input(shape=(1000, 1), name="input_time0")
    inp_template0 = Input(shape=(1000, 1), name="input_template0")
    inp_time1 = Input(shape=(1000, 1), name="input_time1")
    inp_template1 = Input(shape=(1000, 1), name="input_template1")
    inp_time2 = Input(shape=(1000, 1), name="input_time2")
    inp_template2 = Input(shape=(1000, 1), name="input_template2")

    signal0 = signal_block(inp_time0, [32, 64])
    signal1 = signal_block(inp_time1, [32, 64])
    signal2 = signal_block(inp_time2, [32, 64])

    template0 = signal_block(inp_template0, [8, 16])
    template1 = signal_block(inp_template1, [8, 16])
    template2 = signal_block(inp_template2, [8, 16])

    concat0 = concat_block(signal0, template0)
    concat1 = concat_block(signal1, template1)
    concat2 = concat_block(signal2, template2)

    merged = concatenate([concat0, concat1, concat2])

    final_outp = subnet_x(merged)

    model = keras.Model(inputs=[inp_time0,
                                inp_time1,
                                inp_time2,
                                inp_template0,
                                inp_template1,
                                inp_template2],
                                outputs=final_outp,
                                name='Template_Model')

    return model
    