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

from continualai.colab.scripts import mnist
x_train, t_train, x_test, t_test = mnist.load()

task_0 = [(x_train, t_train), (x_test, t_test)]
tasks = [task_0]
fitness_tasks = [task_0]
X_new = np.load('test_assembly.npy',allow_pickle=True).item()
#Y_new = np.load('New_Instances_label.npy',allow_pickle=True).item()




###############################################################################
orinigal_train_digit = {}
for i in range(0,10):
 orinigal_train_digit[i] = []

orinigal_test_digit = {}
for i in range(0,10):
 orinigal_test_digit[i] = []

orinigal_test_label = {}
for i in range(0,10):
 orinigal_test_label[i] = []

orinigal_train_label = {}
for i in range(0,10):
 orinigal_train_label[i] = []


def extract_each_digit(x_train, t_train, x_test, t_test):
  for index,element in enumerate(t_train):
   for j in range(0,10):    
    if(element == j):
     orinigal_train_digit[j].append(x_train[index])
     orinigal_train_label[j].append(j)        

  for index,element in enumerate(t_test):
   for j in range(0,10):    
    if(element == j):
     orinigal_test_digit[j].append(x_test[index]) 
     orinigal_test_label[j].append(j)       
  
  for i in range(0,10):
   orinigal_test_label[i] = np.reshape(orinigal_test_label[i],(len(orinigal_test_label[i]),1))
   orinigal_test_digit[i] = np.reshape(orinigal_test_digit[i],(len(orinigal_test_digit[i]),1,28,28))
   orinigal_train_label[i] = np.reshape(orinigal_train_label[i],(len(orinigal_train_label[i]),1))
   orinigal_train_digit[i] = np.reshape(orinigal_train_digit[i],(len(orinigal_train_digit[i]),1,28,28))
   
extract_each_digit(x_train, t_train, x_test, t_test)

###############################################################################



####################################################################################################################
# Rehersal_Part
for i,key in enumerate(list(X_new.keys())):     
  X_temp = np.reshape(X_new[key],(len(X_new[key]),1,28,28))
  Y_temp = [key for k in range(0,len(X_new[key]))]   
  Y_temp = np.reshape(Y_temp,(len(Y_temp),))
  X_temp = X_temp.astype('float32') 
  Y_temp = Y_temp.astype('uint8') 
  task_temp = [(X_temp, Y_temp), (X_temp, Y_temp)]
  fitness_tasks.append(task_temp)
  
  #Accumulation
  for k in range(0,i):
   key_list = list(X_new.keys())   
   X_temp_k = np.reshape(X_new[key_list[k]],(len(X_new[key_list[k]]),1,28,28))
  
   Y_temp_k = [key_list[k] for j in range(0,len(X_new[key_list[k]]))]
   Y_temp_k = np.reshape(Y_temp_k,(len(Y_temp_k),))
  
   X_temp_k = X_temp_k.astype('float32') 
   Y_temp_k = Y_temp_k.astype('uint8') 
  
   X_temp = np.append(X_temp, X_temp_k , axis=0)
   Y_temp = np.append(Y_temp, Y_temp_k , axis=0)
 

 # Rehersal
  used_key_list = []
  for k in range(0,i):
   used_key_list.append(key_list[k])
  for j in range(10):
   if(j not in used_key_list):
    reherasl_array = orinigal_train_digit[j][0:100]
   else:
    reherasl_array = orinigal_train_digit[j][0:60]
      
   for temp_ in reherasl_array:
     temp_  = np.reshape(temp_,(1,1,28,28))
     temp_ = temp_.astype('float32')
     X_temp = np.append(X_temp, temp_ , axis=0)
    
    
     temp_ = [j]
     temp_  = np.reshape(temp_,(1,))
     temp_ = temp_.astype('uint8')
     Y_temp = np.append(Y_temp, temp_ , axis=0)
    
  indices_ = np.arange(X_temp.shape[0])
  np.random.shuffle(indices_)

  X_temp = X_temp[indices_]
  Y_temp = Y_temp[indices_]     
  task_temp = [(X_temp, Y_temp), (X_temp, Y_temp)]
  tasks.append(task_temp)           

######################################################################################################################
# Initial training
EWC_ON = True
for epoch in range(0, 10):
   model = train_ewc(model, device, 0, x_train, t_train, optimizer, epoch, EWC_ON,0,fisher_dict,optpar_dict)
on_task_update(0, x_train, t_train,model,optimizer,device,fisher_dict,optpar_dict) 

accuracy_on_each_task = []
for id_test, task in enumerate(fitness_tasks):
   print("Testing on train_task: ", id_test)
   _, (x_test, t_test) = task
   acc = test(model, device, x_test, t_test,"Initial_model")
   accuracy_on_each_task.append(acc)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)




test_digit_1 = np.load('train_digit_awgn.npy',allow_pickle=True).item()
test_digit_2 = np.load('train_digit_motion.npy',allow_pickle=True).item()
test_digit_3 = np.load('train_digit_reduced.npy',allow_pickle=True).item()

test_label_1 = np.load('train_label_awgn.npy',allow_pickle=True).item()
test_label_2 = np.load('train_label_motion.npy',allow_pickle=True).item()

test_label_3 = np.load('train_label_reduced.npy',allow_pickle=True).item()


task_0 = [(test_digit_1[4], test_label_1[4]), (test_digit_1[4].astype('float32'), test_label_1[4].astype('uint8'))]
new_instances_test_tasks = [task_0]
task_1 = [(test_digit_2[9], test_label_2[9]), (test_digit_2[9].astype('float32'), test_label_2[9].astype('uint8'))]
new_instances_test_tasks.append(task_1)
task_2 = [(test_digit_3[1], test_label_3[1]), (test_digit_3[1].astype('float32'), test_label_3[1].astype('uint8'))]
new_instances_test_tasks.append(task_2)
   
initai_model_accuracy_on_each_test_task = []
for id_test, task in enumerate(new_instances_test_tasks):
   print("Testing on test_task: ", id_test + 1)  
   _, (x_test, t_test) = task
   acc = test(model, device, x_test, t_test,"initai_model")
   initai_model_accuracy_on_each_test_task.append(acc)
#######################################################################################################################

accuracy_after_each_retrain = {}
retrained_model_accuracy_on_each_test_task = {}
for id, task in enumerate(tasks[1:]):
  print("Training on train_task: ", id+1)
  accuracy_after_each_retrain[id+1] = []
  retrained_model_accuracy_on_each_test_task[id+1] = []  
  
  (x_train, t_train), _ = task
  
  EWC_ON = False
  for epoch in range(0, 10):
   model = train_ewc(model, device, id+1, x_train, t_train, optimizer, epoch, EWC_ON,1,fisher_dict,optpar_dict)
  on_task_update(id+1, x_train, t_train,model,optimizer,device,fisher_dict,optpar_dict) 
    
  for id_test, task in enumerate(fitness_tasks):
    print("Testing on train_task: ", id_test)
    _, (x_test, t_test) = task
    acc = test(model, device, x_test, t_test,'after_retrain'+str(id+1))
    accuracy_after_each_retrain[id+1].append(acc)
   
  for id_test, task in enumerate(new_instances_test_tasks):
    print("Testing on task: ", id_test + 1)  
    _, (x_test, t_test) = task
    acc = test(model, device, x_test, t_test,'after_retrain'+str(id+1))
    retrained_model_accuracy_on_each_test_task[id+1].append(acc)  