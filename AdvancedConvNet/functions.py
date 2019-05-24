from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler as LRS
from keras.preprocessing.image import ImageDataGenerator


# Initial convolution
def initial_conv(model, filters, activation, strides=(1,1)):

    # Set activation=False for denseNet, since this uses preactivation

    model = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   strides=strides, padding='same')(model)
    if activation:
        model = BN()(model)
        model = Activation('relu')(model)

    return model


## Network from RNA
def CBN(model, filters):

    model = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   padding='same')(model)
    model = BN()(model)
    model = Activation('relu')(model)

    model = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   padding='same')(model)
    model = BN()(model)
    model = Activation('relu')(model)

    model = MaxPooling2D(pool_size=2)(model)

    return model


def denseNet(model, filters, growth_rate, pool=False, bottleneck=False, compression=1):
    # Use pool=True at the end of a block.

    if bottleneck:
        conv = BN()(model)
        conv = Activation('relu')(conv)
        conv = Conv2D(growth_rate * 4, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(1, 1),
                      padding='same')(conv)
        conv = BN()(conv)

    else:
        conv = BN()(model)

    conv = Activation('relu')(conv)
    conv = Conv2D(growth_rate, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                  padding='same')(conv)

    filters += growth_rate

    model = concatenate([model, conv])

    if pool:
        model = BN()(model)
        model = Activation('relu')(model)
        model = Conv2D(int(filters * compression), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                       kernel_size=(1, 1), padding='same')(model)
        model = MaxPooling2D(pool_size=2)(model)

    return model, filters


def resNet(model, filters, firstStrides=(1, 1)):
    # Use firstStrides=(2,2) when #filters is increased to downsize.

    shortcut = model

    model = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   strides=firstStrides, padding='same')(model)
    model = BN()(model)
    model = Activation('relu')(model)

    model = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   padding='same')(model)
    model = BN()(model)

    if firstStrides != (1, 1):
        shortcut = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(1, 1),
                          strides=firstStrides, padding='same')(shortcut)
        shortcut = BN()(shortcut)

    model = keras.layers.add([shortcut, model])
    model = Activation('relu')(model)

    return model


def wideResNet(model, base, k, upscale=False, firstStrides=(1, 1), dropout=0.0):
    # Use firstStrides=(2,2) when increasing # filters.
    # Use upscale=True first time since wide makes base*k>initial filters.

    shortcut = model

    model = Conv2D(base * k, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   strides=firstStrides, padding='same')(model)

    if dropout > 0.0:
        model = Dropout(dropout)(model)

    model = BN()(model)
    model = Activation('relu')(model)

    model = Conv2D(base * k, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   padding='same')(model)
    model = BN()(model)

    if upscale or firstStrides != (1, 1):
        shortcut = Conv2D(base * k, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(1, 1),
                          strides=firstStrides, padding='same')(shortcut)
        shortcut = BN()(shortcut)

    model = keras.layers.add([shortcut, model])
    model = Activation('relu')(model)

    return model


def bottleneck(model, filters_low, filters_high, upscale=False, firstStrides=(1, 1)):
    # Use firstStrides=(2,2) when increasing # filters.
    # Use upscale=True first time.

    shortcut = model

    model = Conv2D(filters_low, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(1, 1),
                   strides=(1, 1), padding='same')(model)
    model = BN()(model)
    model = Activation('relu')(model)

    model = Conv2D(filters_low, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   strides=firstStrides, padding='same')(model)
    model = BN()(model)
    model = Activation('relu')(model)

    model = Conv2D(filters_high, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(1, 1),
                   strides=(1, 1), padding='same')(model)
    model = BN()(model)
    model = Activation('relu')(model)

    if upscale or firstStrides != (1, 1):
        shortcut = Conv2D(filters_high, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(1, 1),
                          strides=firstStrides, padding='same')(shortcut)
        shortcut = BN()(shortcut)

    model = keras.layers.add([shortcut, model])
    model = Activation('relu')(model)

    return model


def convNet(model, filters, firstStrides=(1, 1)):
    # Set firstStrides=(2,2) when number of filters increase

    model = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   strides=firstStrides, padding='same')(model)
    model = BN()(model)
    model = Activation('relu')(model)

    model = Conv2D(filters, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), kernel_size=(3, 3),
                   padding='same')(model)
    model = BN()(model)

    model = Activation('relu')(model)

    return model


# Runs the correct lines below depending on position and parameters
def runConv(model, filters, pos, typeConv, growth_rate, k, dropout, bottleneckDense, compression):
    if typeConv == "convNet":
        if pos == "First":
            filters *= 2
            model = convNet(model, filters=filters, firstStrides=(2, 2))
        else:
            model = convNet(model, filters=filters)

    elif typeConv == "resNet":
        if pos == "First":
            filters *= 2
            model = resNet(model, filters=filters, firstStrides=(2, 2))
        else:
            model = resNet(model, filters=filters)

    elif typeConv == "bottleneck":
        if pos == "First":
            filters *= 2
            model = bottleneck(model, filters, int(filters * 2), firstStrides=(2, 2))
        elif pos == "Start":
            model = bottleneck(model, filters, int(filters * 2), upscale=True)
        else:
            model = bottleneck(model, filters, int(filters * 2))

    elif typeConv == "wide":
        if pos == "Start":
            model = wideResNet(model, filters, k, upscale=True, dropout=dropout)
        elif pos == "First":
            filters *= 2
            model = wideResNet(model, filters, k, firstStrides=(2, 2), dropout=dropout)
        else:
            model = wideResNet(model, filters, k, dropout=dropout)

    elif typeConv == "denseNet":
        if pos == "Last":
            model, filters = denseNet(model, filters, growth_rate, pool=True, compression=compression,
                                      bottleneck=bottleneckDense)
        else:
            model, filters = denseNet(model, filters, growth_rate, compression=compression, bottleneck=bottleneckDense)

    return model, filters