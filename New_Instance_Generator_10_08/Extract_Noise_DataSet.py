 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:57:36 2021
@author: dentalcare999
"""
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import torch
torch.manual_seed(0)

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


###############################################################################

def extract_dataset(path):
 import scipy.io
 mat = scipy.io.loadmat(path)

 x_train = mat["train_x"]
 t_train_temp = mat["train_y"]

 x_train = np.reshape(x_train,(len(x_train),1,28,28))


 t_train = []
 for i in range(0,len(t_train_temp)):
  for j in range(0,10):   
   if(t_train_temp[i][j] == 1):
     t_train.append(j)
 t_train = np.reshape(t_train,(len(t_train),))
 #x_train_1 = x_train_1.astype('float32')


 x_test = mat["train_x"]
 t_test_temp = mat["train_y"]

 x_test = np.reshape(x_test,(len(x_test),1,28,28))
 #x_test_1 = x_test_1.astype('float32')


 t_test = []
 for i in range(0,len(t_test_temp)):
  for j in range(0,10):   
   if(t_test_temp[i][j] == 1):
     t_test.append(j)
 t_test = np.reshape(t_test,(len(t_test),))
 
 return x_train,t_train,x_test,t_test



x_train_1, t_train_1, x_test_1, t_test_1 = extract_dataset('mnist-with-awgn.mat')
x_train_2, t_train_2, x_test_2, t_test_2 = extract_dataset('mnist-with-motion-blur.mat')
x_train_3, t_train_3, x_test_3, t_test_3 = extract_dataset('mnist-with-reduced-contrast-and-awgn.mat')

###############################################################################
def extract_each_digit(x_train, t_train, x_test, t_test):
    
  
  train_digit = {}
  for i in range(0,10):
   train_digit[i] = []

  test_digit = {}
  for i in range(0,10):
   test_digit[i] = []

  test_label = {}
  for i in range(0,10):
   test_label[i] = []
 
  train_label = {}
  for i in range(0,10):
   train_label[i] = []  
    
  for index,element in enumerate(t_train):
   for j in range(0,10):    
    if(element == j):
     train_digit[j].append(x_train[index])
     train_label[j].append(j)        

  for index,element in enumerate(t_test):
   for j in range(0,10):    
    if(element == j):
     test_digit[j].append(x_test[index]) 
     test_label[j].append(j)       
  
  for i in range(0,10):
   test_label[i] = np.reshape(test_label[i],(len(test_label[i]),1))
   train_label[i] = np.reshape(train_label[i],(len(train_label[i]),1))
   test_digit[i] = np.reshape(test_digit[i],(len(test_digit[i]),1,28,28))
   train_digit[i] = np.reshape(train_digit[i],(len(train_digit[i]),1,28,28))
  return  train_digit,train_label,test_digit,test_label




train_digit_1,train_label_1,test_digit_1,test_label_1 = extract_each_digit(x_train_1, t_train_1, x_test_1, t_test_1)
train_digit_2,train_label_2,test_digit_2,test_label_2 = extract_each_digit(x_train_2, t_train_2, x_test_2, t_test_2)
train_digit_3,train_label_3,test_digit_3,test_label_3 = extract_each_digit(x_train_3, t_train_3, x_test_3, t_test_3)

#t_1 = 0
#for i in range(0,len(t_train_1)):
# if(t_train_1[i] == 1):
#    t_1 += 1     
#    
#t_2 = 0
#for i in range(0,len(t_train_2)):
# if(t_train_2[i] == 1):
#    t_2 += 1 
#    
#t_3 = 0
#for i in range(0,len(t_train_3)):
# if(t_train_3[i] == 1):
#    t_3 += 1 


np.save('train_digit_awgn.npy', train_digit_1)
np.save('train_digit_motion.npy', train_digit_2)
np.save('train_digit_reduced.npy', train_digit_3)

np.save('train_label_awgn.npy', train_label_1)
np.save('train_label_motion.npy', train_label_2)
np.save('train_label_reduced.npy', train_label_3)

np.save('test_digit_awgn.npy', test_digit_1)
np.save('test_digit_motion.npy', test_digit_2)
np.save('test_digit_reduced.npy', test_digit_3)

np.save('test_label_awgn.npy', test_label_1)
np.save('test_label_motion.npy', test_label_2)
np.save('test_label_reduced.npy', test_label_3)

