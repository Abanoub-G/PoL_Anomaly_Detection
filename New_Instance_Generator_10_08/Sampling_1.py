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

#initial_model training for grouping each class of digits into the correct classification and misclassification
from continualai.colab.scripts import mnist
x_train, t_train, x_test, t_test = mnist.load()


# Initial training
EWC_ON = False
for epoch in range(0, 10):
   model = train_ewc(model, device, 0, x_train, t_train, optimizer, epoch, EWC_ON,0,fisher_dict,optpar_dict)
on_task_update(0, x_train, t_train,model,optimizer,device,fisher_dict,optpar_dict)
###############################################################################


###############################################################################



#accuracy_threshold should be passed as integer such as 95, if you want filter out average accuracy on any class higher than 95%
def condition_selector(train_digit,train_label,accuracy_threshold):
    
 dataset = {}


 for i in range(0,10):  
  result = test(model, device, train_digit[i].astype('float32'), train_label[i].astype('uint8'),'initial_model')
  if(float(result) < accuracy_threshold):
   dataset[i] = train_digit[i]    
 
 return dataset  



filtered_dataset_awgn = condition_selector(train_digit_awgn,train_label_awgn,95)
filtered_dataset_motion = condition_selector(train_digit_motion,train_label_motion,95)
filtered_dataset_reduced = condition_selector(train_digit_reduced,train_label_reduced,95)



#After condition_filter, cluster each class into the correct classified group and misclassification 
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
   
dataset_awgn  = cluster_misclassification(filtered_dataset_awgn )
dataset_motion = cluster_misclassification(filtered_dataset_motion)
dataset_reduced = cluster_misclassification(filtered_dataset_reduced)
np.save('sampling_with_selector_awgn.npy', dataset_awgn)
np.save('sampling_with_selector_motion.npy', dataset_motion)
np.save('sampling_with_selector_reduced.npy', dataset_reduced)