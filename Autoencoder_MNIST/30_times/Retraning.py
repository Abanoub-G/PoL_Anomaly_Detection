#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:57:26 2021

@author: dentalcare999

"""
import random 
import pickle
import numpy as np
import tensorflow

from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
import pickle

from pyod.pyod.utils.stat_models import pairwise_distances_no_broadcast
#from tensorflow.keras import initializer



#returning the predited samples and original samples, which have the same label  
def pre_error_cal(x_train,predicted_output,number):
 index = []
 raw = []
 output = []
 
 for z in range(0,len(x_train)):
  if(x_train[z][784] == number):
   index.append(z)  
   raw.append(x_train[z])   


 for z,temp in enumerate(index):
   output.append(predicted_output[temp])
 print("number is", number)     
 print("len(raw) is", len(raw))  
 print("len(output) is", len(output))  
  
 raw = np.reshape(raw,(len(raw),785))
 raw = raw[:,0:784]  
 output = np.reshape(output,(len(output),784))  
 
 return raw,output

#--------------------------------------------------------------------------------------------------------------------------------------
repeat_times = 1
random_seed = []
average_error_0 = []

LATENT_SIZE = [16,32,64,128,256,512]

dimesion = 785
EPOCHS = 10

for i in range(0,repeat_times):
 random_seed.append(i)




 #i is the retraining iteration 
for i in range(1,10):    
 for k,latent_space in enumerate(LATENT_SIZE):
  for w in range(0,repeat_times):   
   x_train = np.zeros((1,dimesion))
   average_error = []
   #Full_Retraining: Loading all the datasets   
   for j in range(0,i+1): 
    data_path_string = "dataset/number_train_array_" + str(j) + ".txt"
    
    average_error.append([])
    #   print("data_path_string is",data_path_string)
    with open(data_path_string, "rb") as fp:   # Unpickling   
     temp = pickle.load(fp) 
        
    x_train = np.vstack([x_train,temp])
   
   x_train = x_train[1:,:]
 #  shuffle Full Retraining Set  
  
   x_train = list(x_train) 
   random.shuffle(x_train)
   x_train = np.reshape(x_train,(len(x_train),dimesion))
    
   #Loading previous model
   # i doesn't mean just digit i itself. It means from digit 0 to i, including i itself
   # autoencoder_LATENT_SIZE16_Seed0Digit_0.h5  
   model_path_string = 'dataset/Training_'+str(i-1)+'/autoencoder_LATENT_SIZE'+str(latent_space) + '_Seed' + str(random_seed[w]) +'Digit_'+str(i-1) +'.h5'
 #  print("model_path_string is", model_path_string)    
   autoencoder = tensorflow.keras.models.load_model(model_path_string)
   x_train[:,0:784] = x_train[:,0:784]/255
   
   post_x_train = np.reshape(x_train[:,0:784],(len(x_train),28,28))
   for epoch in range(EPOCHS):
      autoencoder.fit(post_x_train, post_x_train, shuffle = False)
   my_string = 'dataset/Training_'+str(i)+'/autoencoder_LATENT_SIZE' + str(latent_space) + '_Seed'  + str(random_seed[w]) + 'Digit_' +str(i) + '.h5'
   autoencoder.save(my_string)

   predicted_output = autoencoder.predict(post_x_train)
   
   for y in range(0,len(average_error)):
    raw,out = pre_error_cal(x_train,predicted_output,y)
    average_error[y] = pairwise_distances_no_broadcast(raw,out)
    average_error[y] = np.reshape(average_error[y],(len(average_error[y]),1))
    average_error[y] = np.sum(average_error[y])/len(average_error[y])       
    error_path =  'dataset/Training_' + str(i) + '/average_error_for_'+ str(y) + '_autoencoder_LATENT_SIZE' + str(latent_space) + '_Seed'  + str(random_seed[w]) +'.txt'
    with open(error_path, "wb") as fp:   #Pickling
     pickle.dump(average_error[y], fp)   
   

   
print(x_train[:,784])

x_train = list(x_train)

 