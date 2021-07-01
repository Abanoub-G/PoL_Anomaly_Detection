 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:57:36 2021

@author: dentalcare999
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle


from continualai.colab.scripts import mnist
#mnist.init()

x_train, t_train, x_test, t_test = mnist.load()






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

def create_unseen_scenarios(x_train,t_train,train_digit_index, index):
 new_x_train = []
 new_t_train = []
 
 second_x_train = []
 second_t_train = []
 
 index_list = train_digit_index[index][6000:]
     
 for i in range(0,len(x_train)):
  if(i not in index_list):
   new_x_train.append(x_train[i])
  else:
   second_x_train.append(x_train[i])
      
 for i in range(0,len(t_train)):
  if(i not in index_list):
   new_t_train.append(t_train[i])
  else:
   second_t_train.append(t_train[i])      
   
 new_x_train = np.reshape(new_x_train,(len(new_x_train),1,28,28))
 new_t_train = np.reshape(new_t_train,(len(new_t_train),))

 second_x_train = np.reshape(second_x_train,(len(second_x_train),1,28,28))
 second_t_train = np.reshape(second_t_train,(len(second_t_train),))     
 return new_x_train,new_t_train,second_x_train,second_t_train      
    
x_train,t_train,second_x_train,second_t_train = create_unseen_scenarios(x_train,t_train,train_digit_index,7)   















       
# task 1
task_1 = [(x_train, t_train), (x_test, t_test)]

# task list
tasks = [task_1]



####################################################################################################################
#EWC


fisher_dict = {}
optpar_dict = {}
ewc_lambda = [5,1,1.0]



model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def on_task_update(task_id, x_mem, t_mem):

  model.train()
  optimizer.zero_grad()
  
#   accumulating gradients
  for start in range(0, len(t_mem)-1, 256):
      end = start + 256
      x, y = torch.from_numpy(x_mem[start:end]), torch.from_numpy(t_mem[start:end]).long()
      x, y = x.to(device), y.to(device)
      output = model(x)
      loss = F.cross_entropy(output, y)
      loss.backward()

  fisher_dict[task_id] = {}
  optpar_dict[task_id] = {}

  # gradients accumulated can be used to calculate fisher
  for name, param in model.named_parameters():
    
    optpar_dict[task_id][name] = param.data.clone()
    fisher_dict[task_id][name] = param.grad.data.clone().pow(2)
    fisher_flat = torch.reshape(fisher_dict[task_id][name], (-1,))
    for target in fisher_flat:
     if(math.isnan(target)):
       print("param is", target)
       print("name is", name,'/n')




def train_ewc(model, device, task_id, x_train, t_train, optimizer, epoch, EWC_ON, ewc_lambda):
    model.train()
    
    #batch size is 100
    for start in range(0, len(t_train)-1, 256):
      end = start + 256
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
              optpar = optpar_dict[task_id -1][name]
              task_part +=  ((fisher*(optpar - param).pow(2)).sum())
            second_part = second_part + task_part* ewc_lambda
            print('Train Epoch: {} \ttask_part: {:.6f}'.format(epoch, task_part))
    #          loss += ((optpar - param).pow(2)).sum() * ewc_lambda[task]
    #          loss += fisher.sum() * ewc_lambda[task]
      loss = loss + second_part
      loss.backward()
      optimizer.step()
      
      #print(loss.item())
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss))
    print('Train Epoch: {} \tOriginal_loss: {:.6f}'.format(epoch, original_loss.item()),'\n')
    
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
   train_ewc(model, device, 0, x_train, t_train, optimizer, epoch, EWC_ON,0)
on_task_update(0, x_train, t_train)
# Testing 
for i,dataset in enumerate(test_digit):
 print("Test on Digit", i)
 test(model, device, test_digit[i], test_label[i],"Initial_model")  

# Retraining with or without EWC
def fitness_function(ewc_lambda):   
 EWC_ON = True
 average = 0

 copyed_model = pickle.loads(pickle.dumps(model))   
 
 for epoch in range(0, 10):
   train_ewc(copyed_model, device, 1, second_x_train, second_t_train, optimizer, epoch, EWC_ON, ewc_lambda)
 # Testing
 for i,dataset in enumerate(test_digit):
  print("Test on Digit", i)
  average += test(copyed_model, device, test_digit[i], test_label[i],"Retrained_model")
 average = average / 10
 return -average    


#EWC_ON = True
#average = 0 
#for epoch in range(0, 10):
#  train_ewc(model, device, 1, second_x_train, second_t_train, optimizer, epoch, EWC_ON, 6.653807839860444)
## Testing
#for i,dataset in enumerate(test_digit):
# print("Test on Digit", i)
# average += test(model, device, test_digit[i], test_label[i],"Retrained_model")
#average = average / 10 











from hyperopt import fmin, tpe,hp

best = fmin(
    fn=lambda x: fitness_function(x),
    space=hp.uniform('x', 0,10),
    algo=tpe.suggest,
    max_evals=100)


   
#for id_test, task in enumerate(tasks):
#   print("Testing on task: ", id_test)
#   _, (x_test, t_test) = task
#   acc = test(model, device, x_test, t_test)
#   avg_acc = avg_acc + acc
#   
#print("Avg acc: ", avg_acc / 3)
#ewc_accs.append(avg_acc / 3)



