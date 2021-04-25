#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 19:03:08 2021

@author: dentalcare999
"""

import pickle

latent_size = [16,32,64,128,256,512]
for i in range(1,10):
 for j in range(0,i+1):
  for k,latent in enumerate(latent_size):
   with open("dataset/Training_"+str(i)+"/average_error_for_"+str(j)+"_autoencoder_LATENT_SIZE"+str(latent)+"_Seed0.txt", "rb") as fp:   # Unpickling
    error = pickle.load(fp)
    my_string = "Training_"+str(i)+"/average_error_for_"+str(j)+"_autoencoder_LATENT_SIZE"+str(latent)+"_Seed0 is"
    print(my_string,error)