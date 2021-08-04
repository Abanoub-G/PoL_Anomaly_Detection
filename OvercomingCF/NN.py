
# =======================================================================================
# DESCRIPTION
# =======================================================================================
# This code can be used to investigated the effect of lambda and the dataset sizes used 
# (in both initial training and retraining) on forgetting whilst learning new tasks.
# =======================================================================================


import torch
torch.manual_seed(1);
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import statistics
import copy

# print("======================")
# print("Check GPU is info")
# print("======================")
# print("How many GPUs are there? Answer:",torch.cuda.device_count())
# print("The Current GPU:",torch.cuda.current_device())
# print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# # Is PyTorch using a GPU?
# print("Is Pytorch using GPU? Answer:",torch.cuda.is_available())
# print("======================")


from scripts.logs import logs
from scripts import mnist

import random
random.seed(0)

import numpy as np
np.random.seed(0)


# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");


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


# CNN sturcture
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



# Training for both EWC and without EWC
def train_ewc(model_input, device, task_id, x_train, t_train, optimizer, epoch, ewc_lambda, fisher_dict, optpar_dict):
	model_input.train()
	
	#epcho size is 256
	for start in range(0, len(t_train)-1, 256):
		end = start + 256
		x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
		x, y = x.to(device), y.to(device)
		
		optimizer.zero_grad()

		output = model_input(x)
		loss = F.cross_entropy(output, y)
		
		# EWC -- magic here! :-)
		for task in range(task_id):
			for name, param in model_input.named_parameters():
				fisher = fisher_dict[task][name]
				optpar = optpar_dict[task_id-1][name]
				loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
				# loss += ((optpar - param).pow(2)).sum() * ewc_lambda   
		loss.backward()
		optimizer.step()
		
		#print(loss.item())
	print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
	return model_input, fisher_dict, optpar_dict



def on_task_update(task_id, x_mem, t_mem, model_input, optimizer, fisher_dict, optpar_dict):

	model_input.train()
	optimizer.zero_grad()
	
	# accumulating gradients
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
		# print("model_input.name_parmaters",model_input.named_parameters())
		# print("name",name)
		# print("param",param)
		
		optpar_dict[task_id][name] = param.data.clone()
		fisher_dict[task_id][name] = param.grad.data.clone().pow(2)
		# print(param.grad.data.clone().pow(2))

	return fisher_dict, optpar_dict



def test(model_input, device, x_test, t_test):
	model_input.eval()
	test_loss = 0
	correct = 0
	for start in range(0, len(t_test)-1, 256):
		end = start + 256
		with torch.no_grad():
			x, y = torch.from_numpy(x_test[start:end]), torch.from_numpy(t_test[start:end]).long()
			x, y = x.to(device), y.to(device)
			output = model_input(x)
			test_loss += F.cross_entropy(output, y).item() # sum up batch loss
			pred = output.max(1, keepdim=True)[1] # get the index of the max logit
			correct += pred.eq(y.view_as(pred)).sum().item()

	test_loss /= len(t_test)
	# print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	# 		test_loss, correct, len(t_test),
	# 		100. * correct / len(t_test)))
	return 100. * correct / len(t_test)



def model_evaluation(model, tasks):
	acc  = [] 
	for i in range(len(tasks)):
		(x_train, y_train), (x_test1, y_test1), (x_test2, y_test2), (x_test3, y_test3) = tasks[i]
		acc.append(test(model, device, x_test1, y_test1))
		print("=========================================== ")
		print("Dataset no. = ",i)
		print("Accuracy on X_train =  ",test(model, device, x_train, y_train))
		print("Accuracy on X_test =  ",test(model, device, x_test1, y_test1))
		print("Accuracy on retraining_dataset =  ",test(model, device, x_test2, y_test2))
		print("Accuracy on only_new_instance_retraining_dataset =  ",test(model, device, x_test3, y_test3))

	return acc



