#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:50:45 2021

@author: dentalcare999
"""

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

def on_task_update(task_id, x_mem, t_mem,model_input,optimizer,device,fisher_dict,optpar_dict):

  model_input.train(mode=False)
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




def train_ewc(model, device, task_id, x_train, t_train, optimizer, epoch, EWC_ON, ewc_lambda,fisher_dict,optpar_dict):
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
    model.train(mode=False)
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

def test_single_probability(model, device, x_test,t_test):
    model.train(mode=False)
    
    x_test = np.reshape(x_test,(1,1,28,28))
    
    x_test = x_test.astype('float32')
    

    x = torch.from_numpy(x_test)
    
    
    x = x.to(device)
    output = model(x)

    output = output.detach().numpy()
    
    pred = output[0,t_test]
    
    return pred

def test_single_category(model, device, x_test,t_test):
    model.train(mode=False)
    
    x_test = np.reshape(x_test,(1,1,28,28))
    
    x_test = x_test.astype('float32')
    

    x = torch.from_numpy(x_test)
    x = x.to(device)
    with torch.no_grad():
     
     output = model(x)
    
    pred = output.max(1, keepdim=True)[1]
    pred = pred.numpy()
    pred = pred[0][0]
    
    return pred
