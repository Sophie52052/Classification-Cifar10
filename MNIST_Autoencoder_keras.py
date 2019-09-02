# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 23:35:18 2019

@author: Sophie
"""
#step - loss: 0.0171
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Activation, Dense, Input
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras import backend as K
from PIL import Image

np.random.seed(1337)  # for reproducibility
# download the mnist
(train_data, _), (test_data, _) = mnist.load_data()

# data pre-processing 2維
train_data = train_data.astype('float32') / 255       # minmax_normalized
test_data = test_data.astype('float32') / 255        # 有加比較好. + 0.5 
train_data = train_data.reshape((train_data.shape[0], -1))
test_data = test_data.reshape((test_data.shape[0], -1))
#print(train_data.shape)
#print(test_data.shape)

#4維
#image_size = train_data.shape[1] #EX.Y是個3*4的蹶鎮 則為4 0為3
#train_data = np.reshape(train_data, [-1, image_size, image_size, 1])
#test_data = np.reshape(test_data, [-1, image_size, image_size, 1])
#
#train_data = train_data.astype('float32') / 255
#test_data = test_data.astype('float32') / 255



# input placeholder
input_img = Input(shape=(784,))

# encoder layers
encoded = Dense(256, activation='tanh')(input_img) #sigmoid
encoded = Dense(128, activation='tanh')(encoded)
encoded = Dense(64, activation='tanh')(encoded)
encoder_output = Dense(10)(encoded)# 2D figure

# decoder layers
decoded = Dense(64, activation='tanh')(encoder_output)
decoded = Dense(128, activation='tanh')(decoded)
decoded = Dense(256, activation='tanh')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)
# compile
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(train_data, train_data,
                nb_epoch=20,
                batch_size=128,
                shuffle=True)
#step - loss: 0.0415
#step - loss: 0.0399
#0.2
#0.0419
#0.0398
#0.0199

#加入雜訊
# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5

#image_size = train_data.shape[1]
#
#noise = np.random.normal(loc=0.5, scale=0.5, size=train_data.shape)
#train_data_noisy = train_data + noise
#noise = np.random.normal(loc=0.5, scale=0.5, size=test_data.shape)
#test_data_noisy = test_data + noise
#
#train_data_noisy = np.clip(train_data_noisy, 0., 1.)
#test_data_noisy = np.clip(test_data_noisy, 0., 1.)
#
#x_decoded = autoencoder.predict(test_data_noisy)
#
#rows, cols = 10, 30
#num = rows * cols
#imgs = np.concatenate([test_data[:num], test_data_noisy[:num], x_decoded[:num]])
#imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
#imgs = np.vstack(np.split(imgs, rows, axis=1))
#imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
#imgs = np.vstack([np.hstack(i) for i in imgs])
#imgs = (imgs * 255).astype(np.uint8)
#plt.figure()
#plt.axis('off')
#plt.title('Original images: top rows, '
#          'Corrupted Input: middle rows, '
#          'Denoised Input:  third rows')
#plt.imshow(imgs, interpolation='none', cmap='gray')
#Image.fromarray(imgs).save('corrupted_and_denoised.png')
#plt.show()


