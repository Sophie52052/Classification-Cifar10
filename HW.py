# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:10:01 2019

@author: Sophie
"""
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, AveragePooling2D, MaxPool2D, Flatten, Activation, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import BatchNormalization as BN
from keras.layers import Dropout

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #loda data

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

x_train = x_train.astype('float32') / 255 +0.5
x_test = x_test.astype('float32') / 255 +0.5
fit_y_train = np_utils.to_categorical(y_train)
pred_y_test = np_utils.to_categorical(y_test)

#MODEL
bn_input_layer = Input(shape=(32,32,3))
bn_layer_0 = BN()(bn_input_layer)
bn_conv1 = Conv2D(filters=64, kernel_size=3)(bn_layer_0)
bn_layer_1 = BN()(bn_conv1)
bn_conv1_active = Activation('relu')(bn_layer_1)
bn_cpool1 = MaxPooling2D(pool_size=2, strides=2, padding='same')(bn_conv1_active)

bn_conv2 = Conv2D(filters=128, kernel_size=3)(bn_cpool1)
bn_layer_2 = BN()(bn_conv2)
bn_conv2_active = Activation('relu')(bn_layer_2)
bn_cpool2 = MaxPooling2D(pool_size=2, strides=2, padding='same')(bn_conv2_active)

bn_conv3 = Conv2D(filters=256, kernel_size=3)(bn_cpool2)
bn_layer_3 = BN()(bn_conv3)
bn_conv3_active = Activation('relu')(bn_layer_3)
bn_cpool3 = MaxPooling2D(pool_size=2, strides=2, padding='same')(bn_conv3_active)

bn_flat_v = Flatten()(bn_cpool3)
#bn_dense1 = Dense(120, activation='relu')(bn_flat_v)
#bn_layer_3 = BN()(bn_dense1)
#bn_dense2 = Dense(82, activation='relu')(bn_dense1)
#bn_layer_4 = BN()(bn_dense2)
bn_output_layer = Dense(10, activation='softmax')(bn_flat_v)
bn_model = Model(bn_input_layer, bn_output_layer)
bn_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
bn_model.summary()


#train
bn_trianing_history = bn_model.fit(x=x_train, y=fit_y_train, validation_data=[x_test, pred_y_test], epochs=30, batch_size=64)