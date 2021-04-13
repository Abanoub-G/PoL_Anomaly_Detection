#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:02:21 2021

@author: dentalcare999
"""

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import pickle
	
# generate random integer values
import random


# -----------------------------------------------------------------------------
# Load Data from local files
file = 'train-images.idx3-ubyte'
arr = idx2numpy.convert_from_file(file)

file1 = 't10k-images.idx3-ubyte'
arr1 = idx2numpy.convert_from_file(file1)

file2 = 't10k-labels.idx1-ubyte'
arr2 = idx2numpy.convert_from_file(file2)

file3 = 'train-labels.idx1-ubyte'
arr3 = idx2numpy.convert_from_file(file3)

#plt.imshow(arr[9], cmap=plt.cm.binary)
#print("arr2[0] is", arr3[9])

#plt.imshow(arr1[3], cmap=plt.cm.binary)
#print("arr2[0] is", arr2[3])

with open("all_training_digit.txt", "wb") as fp_2:
 pickle.dump(arr,fp_2)    

with open("all_training_label.txt", "wb") as fp_2:
 pickle.dump(arr3,fp_2)    

# -----------------------------------------------------------------------------
# For a certain digital, search for all data of that digital in order to train AE
number = 4
index_array_train = []
index_array_test = []


number_train_array = []
number_test_array = []

for i in range(0,len(arr3)):
 if(arr3[i] == number):
   index_array_train.append(i)

for i in range(0,len(arr2)):
 if(arr2[i] == number):
   index_array_test.append(i)

for i in range(0,len(index_array_train)):
 number_train_array.append(np.reshape(arr[index_array_train[i]],(1,784)))

for i in range(0,len(index_array_test)):
 number_test_array.append(np.reshape(arr1[index_array_test[i]],(1,784)))

lable = np.zeros((len(number_train_array),1))

for i in range(0,len(lable)):
 lable[i] = number
 
number_train_array = np.reshape(number_train_array,(len(number_train_array),len(number_train_array[0][0])))

number_train_array = np.hstack((number_train_array,lable)) 


# -----------------------------------------------------------------------------
# For another certain digital, search for all data of that digital in order to train AE
number = 8
index_array_train_another = []
index_array_test_another = []
number_train_array_another = []
number_test_array_another = []

for i in range(0,len(arr3)):
 if(arr3[i] == number):
   index_array_train_another.append(i)

for i in range(0,len(arr2)):
 if(arr2[i] == number):
   index_array_test_another.append(i)


for i in range(0,len(index_array_train_another)):    
 number_train_array_another.append(np.reshape(arr[index_array_train_another[i]],(1,784)))

for i in range(0,len(index_array_test_another)):
 number_test_array_another.append(np.reshape(arr1[index_array_test_another[i]],(1,784))) 
 
lable = np.zeros((len(index_array_train_another),1))

for i in range(0,len(lable)):
 lable[i] = number
 
number_train_array_another = np.reshape(number_train_array_another,(len(number_train_array_another),len(number_train_array_another[0][0])))

number_train_array_another = np.hstack((number_train_array_another,lable)) 

number_train_array = np.vstack((number_train_array,number_train_array_another))


# -----------------------------------------------------------------------------

random_order = []
random_order = random.sample(range(15000), len(number_train_array))
random_order = np.reshape(random_order,(len(random_order),1))
number_train_array = np.hstack((number_train_array,random_order))

number_train_array[number_train_array[:, len(number_train_array[0])-1].argsort()]

number_train_array = number_train_array[:,0:len(number_train_array[0])-1]
# -----------------------------------------------------------------------------
with open("number_train_array.txt", "wb") as fp:
 pickle.dump(number_train_array,fp)

with open("number_test_array.txt", "wb") as fp_1:
 pickle.dump(number_test_array,fp_1)

plt.imshow(arr[31], cmap=plt.cm.binary)