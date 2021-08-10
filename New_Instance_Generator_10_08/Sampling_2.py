#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:05:26 2021

@author: dentalcare999
"""

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import torch
torch.manual_seed(0)
###############################################################################

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from CNN import *
################################################################################
#from CNN import *
use_cuda = False

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
torch.manual_seed(1);

model = Net().to(device)
#model = Net().float
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


fisher_dict = {}
optpar_dict = {}
ewc_lambda = [1 for i in range(0,21)]
###############################################################################


import numpy as np

###############################################################################

train_digit_awgn = np.load('train_digit_awgn.npy',allow_pickle=True).item()
train_digit_motion = np.load('train_digit_motion.npy',allow_pickle=True).item()
train_digit_reduced = np.load('train_digit_reduced.npy',allow_pickle=True).item()

test_digit_awgn = np.load('test_digit_awgn.npy',allow_pickle=True).item()
test_digit_motion = np.load('test_digit_motion.npy',allow_pickle=True).item()
test_digit_reduced = np.load('test_digit_reduced.npy',allow_pickle=True).item()

train_label_awgn = np.load('train_label_awgn.npy',allow_pickle=True).item()
train_label_motion = np.load('train_label_motion.npy',allow_pickle=True).item()
train_label_reduced = np.load('train_label_reduced.npy',allow_pickle=True).item()

test_label_awgn = np.load('test_label_awgn.npy',allow_pickle=True).item()
test_label_motion = np.load('test_label_motion.npy',allow_pickle=True).item()
test_label_reduced = np.load('test_label_reduced.npy',allow_pickle=True).item()


###############################################################################
from continualai.colab.scripts import mnist
x_train, t_train, x_test, t_test = mnist.load()


# Initial training
EWC_ON = False
for epoch in range(0, 10):
   model = train_ewc(model, device, 0, x_train, t_train, optimizer, epoch, EWC_ON,0,fisher_dict,optpar_dict)
on_task_update(0, x_train, t_train,model,optimizer,device,fisher_dict,optpar_dict)
###############################################################################

#accuracy_threshold should be passed as integer such as 95, if you want filter out average accuracy on any class higher than 95%
def condition_selector(train_digit,train_label,accuracy_threshold):
    
 dataset = {}


 for i in range(0,10):  
  result = test(model, device, train_digit[i].astype('float32'), train_label[i].astype('uint8'),'initial_model')
  if(float(result) < accuracy_threshold):
   dataset[i] = train_digit[i]    
 
 return dataset  



selected_dataset_awgn = condition_selector(train_digit_awgn,train_label_awgn,95)
selected_dataset_motion = condition_selector(train_digit_motion,train_label_motion,95)
selected_dataset_reduced = condition_selector(train_digit_reduced,train_label_reduced,95)


###############################################################################

def cluster_misclassification(train_digit):
    
 dataset = {}
 dataset_false = {}
 dataset_correct = {}


 for i in range(0,10):
  if(i in train_digit.keys()):  
   dataset_false[i] = []
   dataset_correct[i] = []
 
  
   for j,item in enumerate(train_digit[i]):
    false_list = []
    correct_list = []      
    result = test_single_category(model, device, item, i)
    if(result != i):
      false_list.append(train_digit[i][j])
    else:
      correct_list.append(train_digit[i][j])  
     
    if(len(false_list)>0):     
     dataset_false[i].append(false_list) 
    if(len(correct_list)>0):     
     dataset_correct[i].append(correct_list) 
    
 dataset['correct'] =  dataset_correct
 dataset['false'] = dataset_false
 
 return dataset    
   
dataset_awgn = cluster_misclassification(selected_dataset_awgn)
dataset_motion = cluster_misclassification(selected_dataset_motion)
dataset_reduced = cluster_misclassification(selected_dataset_reduced)

dataset_awgn_false_dict = dataset_awgn['false']
dataset_motion_false_dict = dataset_motion['false']
dataset_reduced_false_dict = dataset_reduced['false']

##############################################################################



def data_augmentation(single_sample,duplicate_time):
 from keras.preprocessing.image import ImageDataGenerator
 augmented_x = []
 augmented_y = []
 
 for i in range(0,duplicate_time): 
    shift = i*0.01
    degree = 2*i 
   
    datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift,rotation_range=degree)
    # fit parameters from data
  
    temp_x = np.reshape(single_sample,(1, 1, 28, 28))
    temp_x = temp_x.astype('float32')
  
    datagen.fit(temp_x)
  
    temp_t = [i for k in range(0,len(temp_x))]



    for x_batch, y_batch in datagen.flow(temp_x, temp_t, batch_size=len(temp_x)):
     for i in range(0, len(temp_x)): 
        augmented_x.append(x_batch[i].reshape(28, 28))
        augmented_y.append (y_batch[i])    
     break
 
 augmented_x = np.reshape(augmented_x,(len(augmented_x),1,28,28))
 augmented_y = np.reshape(augmented_y,(len(augmented_y),))
 augmented_x = augmented_x.astype('float32')
 augmented_y = augmented_y.astype('uint8')
 
 return augmented_x,augmented_y

def shift_rotate_generator(dataset_false_dict,accuracy_threshold,duplicate_time):
 single_false_augmentation = {}   
 for i in list(dataset_false_dict.keys()):
  single_false_augmentation[i] = []   
  temp_x = np.reshape(dataset_false_dict[i],(len(dataset_false_dict[i]),1,28,28))
  for j in range(0,len(temp_x)):
    augmented_x,augmented_y = data_augmentation(temp_x[j],duplicate_time)     
    accuracy = test(model, device, augmented_x, augmented_y, "special_sampling")
    if(accuracy <= accuracy_threshold):
     single_false_augmentation[i].append(augmented_x)
     break
 return single_false_augmentation


single_false_augmentation_awgn = shift_rotate_generator(dataset_awgn_false_dict,0.7,10)
single_false_augmentation_motion = shift_rotate_generator(dataset_motion_false_dict,0.7,10)
single_false_augmentation_reduced = shift_rotate_generator(dataset_reduced_false_dict,0.7,10)


np.save('single_false_augmentation_awgn.npy', single_false_augmentation_awgn)

np.save('single_false_augmentation_motion.npy', single_false_augmentation_motion)

np.save('single_false_augmentation_reduced.npy', single_false_augmentation_reduced)








