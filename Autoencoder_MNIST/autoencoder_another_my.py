#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:57:26 2021

@author: dentalcare999
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
import numpy as np
import pickle
from joblib import dump, load

from pyod.pyod.utils.stat_models import pairwise_distances_no_broadcast

with open("number_train_array.txt", "rb") as fp:   # Unpickling
 x_train = pickle.load(fp)
 
label = x_train[:,784]

x_train = x_train[:,0:784]

x_train = np.reshape(x_train,(len(x_train),28,28))
 
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
x_train = x_train/255.0
#x_test = x_test/255.0




LATENT_SIZE = 512

autoencoder = Sequential([
    #Encoder        
    Flatten(input_shape = (28, 28)),
    Dense(LATENT_SIZE),
    LR(),
    #Decoder
    Dense(784),
    Activation("relu"),
    Reshape((28, 28))
])


autoencoder.compile("adam", loss = "mean_squared_error")

EPOCHS = 10

for epoch in range(EPOCHS):
    autoencoder.fit(x_train, x_train)

autoencoder.save('autoencoder.h5')

predicted_output = autoencoder.predict(x_train)
single_predicted_output = predicted_output[4000]
single_predicted_output = np.reshape(single_predicted_output,(28,28))
single_predicted_output = single_predicted_output * 255.0
plt.imshow(single_predicted_output, cmap=plt.cm.binary)
plt.figure()

 
#autoencoder.summary()
 
orginal_input = x_train *255.0
single_original_input = orginal_input[4000]

plt.imshow(single_original_input, cmap=plt.cm.binary)
plt.figure()

#predicted_output = np.reshape(predicted_output,(len(predicted_output),len(predicted_output[0])*len(predicted_output[0][0])))
#x_train = np.reshape(x_train,(11693,784))
#
#errors = pairwise_distances_no_broadcast(x_train,predicted_output)
#average_error = np.sum(errors)/len(errors)
#print(average_error)