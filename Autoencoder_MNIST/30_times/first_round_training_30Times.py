#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:57:26 2021

@author: dentalcare999

"""
#--------------------------------------------------------------------------------------------------------------------------------------
repeat_times = 1
random_seed = []
average_error_0 = []

for i in range(0,repeat_times):
 random_seed.append(i)
 
LATENT_SIZE = [16,32,64,128,256,512]

for i in range(0,len(LATENT_SIZE)):
 average_error_0.append([])      
 for j in range(0,repeat_times):
  average_error_0[i].append(0)    

#--------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
import pickle

from pyod.pyod.utils.stat_models import pairwise_distances_no_broadcast
from tensorflow.keras import initializers


with open("dataset/number_train_array_0.txt", "rb") as fp:   # Unpickling
 x_train = pickle.load(fp)
 
label = x_train[:,784]

x_train = x_train[:,0:784]

x_train = np.reshape(x_train,(len(x_train),28,28))
 
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
x_train = x_train/255.0
#x_test = x_test/255.0

index_0 = []

for i in range(0,len(label)):
 if(label[i] == 0):
  index_0.append(i)     

raw_0 = []
for i,temp in enumerate(index_0):
  raw_0.append(x_train[temp])
  
raw_0 = np.reshape(raw_0,(len(raw_0),784))  
#--------------------------------------------------------------------------------------------------------------------------------------
for i in range(0,len(LATENT_SIZE)):
 for j in range(0,repeat_times):    
  autoencoder = Sequential([
      #Encoder        
      Flatten(input_shape = (28, 28)),
      Dense(LATENT_SIZE[i], kernel_initializer=initializers.RandomNormal(stddev=0.01, seed = random_seed[j]),
      bias_initializer=initializers.Zeros()),
      LR(),
      #Decoder
      Dense(784, kernel_initializer=initializers.RandomNormal(stddev=0.01, seed = random_seed[j]),
      bias_initializer=initializers.Zeros()),
      Activation("relu"),
      Reshape((28, 28))
  ])

  autoencoder.compile("adam", loss = "mean_squared_error")

  EPOCHS = 10

  for epoch in range(EPOCHS):
      autoencoder.fit(x_train, x_train, shuffle = False)
  my_string = 'autoencoder_LATENT_SIZE' + str(LATENT_SIZE[i]) + '_Seed' + str(random_seed[j]) + "Digit_0" + '.h5'
  autoencoder.save("dataset/Training_0/" + my_string)

  predicted_output = autoencoder.predict(x_train)


  predicted_0 = []


  for k,temp in enumerate(index_0):
   predicted_0.append(predicted_output[temp]) 

  predicted_0 = np.reshape(predicted_0,(len(predicted_0),784))    

  errors_0 = pairwise_distances_no_broadcast(raw_0,predicted_0)
  errors_0 = np.reshape(errors_0,(len(errors_0),1))
  print("i is",i,"j is ", j)
  average_error_0[i][j] = np.sum(errors_0)/len(errors_0)

with open("dataset/Training_0/average_error_0.txt", "wb") as fp:   #Pickling
  pickle.dump(average_error_0, fp)
