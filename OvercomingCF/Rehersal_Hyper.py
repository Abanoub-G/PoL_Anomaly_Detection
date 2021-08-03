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
import math
import pickle
import statistics


from continualai.colab.scripts import mnist
from matplotlib import pyplot
#mnist.init()

x_train, t_train, x_test, t_test = mnist.load()


X_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
X_test = x_test.reshape((x_test.shape[0], 1, 28, 28))
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#shift = 0.2
## define data preparation
#from keras.preprocessing.image import ImageDataGenerator
##datagen = ImageDataGenerator(zca_whitening=True,featurewise_center=True, featurewise_std_normalization=True,rotation_range=90,width_shift_range=shift, height_shift_range=shift,horizontal_flip=True, vertical_flip=True)
#datagen = ImageDataGenerator(zca_whitening=True, featurewise_std_normalization=True,width_shift_range=shift, height_shift_range=shift)
## fit parameters from data
#datagen.fit(X_train)
#
#
#X_new = []
#Y_new = []
#
#for X_batch, y_batch in datagen.flow(X_train, t_train, batch_size=len(X_train)):
# for i in range(0, len(X_train)):   
#  X_new.append(X_batch[i].reshape(28, 28))
#  Y_new.append(y_batch[i])    
# break
#
#
#X_train = np.reshape(X_new, (len(X_new), 1, 28, 28))
#Y_new = np.reshape(Y_new, (len(Y_new), ))
#np.save('X_train.npy', X_train) # save
#np.save('Y_new.npy', Y_new) # save

X_train = np.load('X_train.npy') # load
Y_new = np.load('Y_new.npy') # load


task_0 = [(x_train, t_train), (x_test, t_test)]
tasks = [task_0]
fitness_tasks = [task_0]


#for i in range(20):
# task_temp = [(X_train[0+50*i:100+50*i], Y_new[0+50*i:100+50*i]), (X_train[0+50*i:100+50*i], Y_new[0+50*i:100+50*i])]
# tasks.append(task_temp)

# task list
# switch to False to use CPU
use_cuda = False

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
torch.manual_seed(1);



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def permute_mnist(mnist, seed):
    """ Given the training set, permute pixels of each img the same way. """

    np.random.seed(seed)
    print("starting permutation...")
    h = w = 28
    perm_inds = list(range(h*w))
    np.random.shuffle(perm_inds)
    # print(perm_inds)
    perm_mnist = []
    for set in mnist:
        num_img = set.shape[0]
        flat_set = set.reshape(num_img, w * h)
        perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, 1, w, h))
    print("done.")
    return perm_mnist













###############################################################################
train_digit_index = {}
for i in range(0,10):
 train_digit_index[i] = []

test_digit = {}
for i in range(0,10):
 test_digit[i] = []

test_label = {}
for i in range(0,10):
 test_label[i] = []


def extract_each_digit(x_train, t_train, x_test, t_test):
  for index,element in enumerate(t_train):
   for j in range(0,10):    
    if(element == j):
     train_digit_index[j].append(index)       

  for index,element in enumerate(t_test):
   for j in range(0,10):    
    if(element == j):
     test_digit[j].append(x_test[index]) 
     test_label[j].append(j)       
  
  for i in range(0,10):
   test_label[i] = np.reshape(test_label[i],(len(test_label[i]),1))
   test_digit[i] = np.reshape(test_digit[i],(len(test_digit[i]),1,28,28))
   
extract_each_digit(x_train, t_train, x_test, t_test)

###############################################################################

def extract_each_digit_variants(x, t):
  digit_index = {}  
  for i in range(0,10):
   digit_index[i] = []

  for index,element in enumerate(t):
   for j in range(0,10):    
    if(element == j):
     digit_index[j].append(index)       

  return digit_index
   


digit_index_dict = extract_each_digit_variants(X_train,Y_new)


def create_unseen_scenarios(x,t,digit_index, index):
 x_desired = []
 t_desired = []
 
 index_list = digit_index[index]
     
 for i in range(0,len(x)):
  if(i in index_list):
   x_desired.append(x[i])
      
 for i in range(0,len(t)):
  if(i in index_list):
   t_desired.append(t[i])   

 x_desired = np.reshape(x_desired,(len(x_desired),1,28,28))
 t_desired = np.reshape(t_desired,(len(t_desired),))     
 return x_desired,t_desired

Varaints_By_Class = []

for i in range(0,10):

 Varaints_By_Class.append(create_unseen_scenarios(X_train,Y_new,digit_index_dict,i))

####################################################################################################################
for i in range(10):
 X_temp,Y_temp = Varaints_By_Class[i]
 X_temp = X_temp[0:10]
 Y_temp = Y_temp[0:10]
 
 task_temp = [(X_temp, Y_temp), (X_temp, Y_temp)]
 fitness_tasks.append(task_temp)    

 
 for j in range(10):
  if(j != i):
   index_list_ = train_digit_index[j][0:15]
  else:
   index_list_ = train_digit_index[j][0:5]      
  for index_ in index_list_:
    temp_  = np.reshape(x_train[index_],(1,1,28,28))
    X_temp = np.append(X_temp, temp_ , axis=0)
    temp_  = np.reshape(t_train[index_],(1,))
    Y_temp = np.append(Y_temp, temp_ , axis=0)
    
 indices_ = np.arange(X_temp.shape[0])
 np.random.shuffle(indices_)

 X_temp = X_temp[indices_]
 Y_temp = Y_temp[indices_]     
 task_temp = [(X_temp, Y_temp), (X_temp, Y_temp)]
 tasks.append(task_temp)      

####################################################################################################################
#EWC


fisher_dict = {}
optpar_dict = {}
ewc_lambda = [1 for i in range(0,21)]



model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def on_task_update(task_id, x_mem, t_mem,model_input):

  model_input.train()
  optimizer.zero_grad()
  
#   accumulating gradients
  for start in range(0, len(t_mem)-1, 256):
      end = start + 256
      x, y = torch.from_numpy(x_mem[start:end]), torch.from_numpy(t_mem[start:end]).long()
      x, y = x.to(device), y.to(device)
      output = model_input(x)
      loss = F.cross_entropy(output, y)
      loss.backward()

  fisher_dict[task_id] = {}
  optpar_dict[task_id] = {}

  # gradients accumulated can be used to calculate fisher
  for name, param in model_input.named_parameters():
    
    optpar_dict[task_id][name] = param.data.clone()
    fisher_dict[task_id][name] = param.grad.data.clone().pow(2)




def train_ewc(model, device, task_id, x_train, t_train, optimizer, epoch, EWC_ON, ewc_lambda):
    model.train()
    
    #batch size is 100
    for start in range(0, len(t_train)-1, 64):
      end = start + 64
      x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
      x, y = x.to(device), y.to(device)
      
      optimizer.zero_grad()

      output = model(x)
      loss = F.cross_entropy(output, y)
      original_loss = F.cross_entropy(output, y)
      second_part = 0
      
      if EWC_ON:
          ## magic here! :-)
          for task in range(task_id):
            task_part = 0  
            for name, param in model.named_parameters():
              fisher = fisher_dict[task][name]
              
#              fisher = torch.negative(fisher)
#              fisher = torch.exp(fisher)
              
              optpar = optpar_dict[task_id -1][name]
              task_part +=  (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
            second_part = second_part + task_part* ewc_lambda
#            print('Train Epoch: {} \ttask_part: {:.6f}'.format(epoch, task_part))
    #          loss += ((optpar - param).pow(2)).sum() * ewc_lambda[task]
    #          loss += fisher.sum() * ewc_lambda[task]
      loss = loss + second_part
      loss.backward()
      optimizer.step()
      
      #print(loss.item())
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss))
    print('Train Epoch: {} \tOriginal_loss: {:.6f}'.format(epoch, original_loss.item()),'\n')
    return model
    
def test(model, device, x_test, t_test, test_description):
    model.eval()
    correct = 0
    for start in range(0, len(t_test)-1, 100):
      end = start + 100
      with torch.no_grad():
        x, y = torch.from_numpy(x_test[start:end]), torch.from_numpy(t_test[start:end]).long()
        x, y = x.to(device), y.to(device)
        output = model(x)

        pred = output.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(y.view_as(pred)).sum().item()

    print(test_description +' Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(t_test),
        100. * correct / len(t_test)))
    return 100. * correct / len(t_test)


ewc_accs = []
avg_acc = 0
  
(x_train, t_train), _ = tasks[0]

# Initial training
EWC_ON = False
for epoch in range(0, 10):
   model = train_ewc(model, device, 0, x_train, t_train, optimizer, epoch, EWC_ON,0)
on_task_update(0, x_train, t_train,model)

average = 0 
original_accuracy = {}
# Testing 
for i,dataset in enumerate(test_digit):
 print("Test on Digit", i)
 temp = test(model, device, test_digit[i], test_label[i],"Initial_model") 
 original_accuracy[i] = temp
 average +=  temp

average = average / 10
print("Test on Old Task")  
print("Initial Accuracy:", average) 

print("Test on New instances") 
original_accuracy[10] = test(model, device, X_train[0:200], Y_new[0:200],"Initial_model")

acc_list_after_each_training = []

intial_acc_list = []
for id_test, task in enumerate(fitness_tasks):
   print("Testing on task: ", id_test)
   _, (x_test, t_test) = task
   acc = test(model, device, x_test, t_test,"Initial_model")
   intial_acc_list.append(acc)
acc_list_after_each_training.append(intial_acc_list)





def fitness_function(arg):
 ewc_lambda = arg[0]
 learning_rate = arg[1]
 m = arg[2]
 e = arg[3]
 b = arg[4]
 
 (x_train, t_train), _ = fitness_tasks[id_]
  
# torch.save(model.state_dict(), 'model.pt')
# copyed_model = Net().to(device)
# copyed_model.load_state_dict(torch.load('model.pt'))
 
# copyed_model = copy.deepcopy(model)
 
 copyed_model = Net().to(device)
 copyed_model.load_state_dict(model.state_dict())
 optimizer = optim.SGD(copyed_model.parameters(), lr=learning_rate, momentum=m)
 
 for epoch in range(0, e):   
  copyed_model = retrain(copyed_model, device, id_, x_train, t_train, optimizer, epoch,False,ewc_lambda,b)
 on_task_update(id_, x_train, t_train,copyed_model)
 

 data = [] 
 avg_acc = 0
 new_tasks = fitness_tasks[0:id_+1]  
 for id_test, task in enumerate(new_tasks):
    _, (x_test, t_test) = task
    acc = test(copyed_model, device, x_test, t_test,"Temp_model")
    data.append(acc)
    avg_acc = avg_acc + acc

 variance = statistics.pvariance(data)
 avg_acc =  avg_acc /(id_+1)

# return variance/avg_acc
# return 0.00001*variance+(1/(avg_acc**2))
 return -avg_acc



from hyperopt import fmin, tpe,hp








def retrain (model, device, task_id, x_train, t_train, optimizer, epoch, EWC_ON, ewc_lambda,b):
    model.train()
    
    #batch size is 100
    for start in range(0, len(t_train)-1, b):
      end = start + b
      x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
      x, y = x.to(device), y.to(device)
      
      optimizer.zero_grad()

      output = model(x)
      loss = F.cross_entropy(output, y)
      original_loss = F.cross_entropy(output, y)
      second_part = 0
      
      if EWC_ON:
          ## magic here! :-)
          for task in range(task_id):
            task_part = 0  
            for name, param in model.named_parameters():
              fisher = fisher_dict[task][name]
              
#              fisher = torch.negative(fisher)
#              fisher = torch.exp(fisher)
              
              optpar = optpar_dict[task_id -1][name]
              task_part +=  (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
            second_part = second_part + task_part* ewc_lambda
#            print('Train Epoch: {} \ttask_part: {:.6f}'.format(epoch, task_part))
    #          loss += ((optpar - param).pow(2)).sum() * ewc_lambda[task]
    #          loss += fisher.sum() * ewc_lambda[task]
      loss = loss + second_part
      loss.backward()      
      
#      for i in range(0,task_id):
#       model.fc2.weight.grad[i] = 0.0
#       model.fc2.bias.grad[i] = 0.0          
#      for i in range(task_id+1,10):
#       model.fc2.weight.grad[i] = 0.0
#       model.fc2.bias.grad[i] = 0.0  
      
      optimizer.step()
      

      #print(loss.item())
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss))
    print('Train Epoch: {} \tOriginal_loss: {:.6f}'.format(epoch, original_loss.item()),'\n')
    return model










E_list = [5,10,15,20]
B_list = [8,16,32,64]


for i in range(0,10):
 (x_train, t_train), _ = tasks[i+1]
 id_ = i+1
 
 best = fmin(
    fitness_function,
    space=[hp.uniform('lambda', 0,10),hp.uniform('lr', 0.0001,0.002),hp.uniform('m', 0.1,0.9),hp.choice('E',[5,10,15,20]),hp.choice('B',[8,16,32,64])],
    algo=tpe.suggest,
    max_evals=100)
 
 
 
 EWC_ON = False
 optimizer = optim.SGD(model.parameters(), lr=best['lr'], momentum=best['m'])     
 for epoch in range(0, E_list[best['E']]):
   model = retrain(model, device, i+1, x_train, t_train, optimizer, epoch, EWC_ON,best['lambda'],B_list[best['B']])
 on_task_update(i+1, x_train, t_train,model)
 
 temp_list = []
 for id_test, task in enumerate(fitness_tasks):
   print("Testing on task: ", id_test)
   _, (x_test, t_test) = task
   acc = test(model, device, x_test, t_test,"Temp_model")
   temp_list.append(acc)
 acc_list_after_each_training.append(temp_list)   
   
final_acc_list = []
for id_test, task in enumerate(fitness_tasks):
   print("Testing on task: ", id_test)
   _, (x_test, t_test) = task
   acc = test(model, device, x_test, t_test,"Final_model")
   final_acc_list.append(acc)

final_acc_average = sum(final_acc_list) / len(final_acc_list)

var_ = statistics.pvariance(final_acc_list)


#for  name, param in model.fc2.named_parameters():
# print(name ,"is", param)
#
#print(model.fc2.weight.grad[0])
#
#model.fc2.weight.grad[0] = 0.0
#id_ = 1
#aha = fitness_function([1,0.001,0.9])

