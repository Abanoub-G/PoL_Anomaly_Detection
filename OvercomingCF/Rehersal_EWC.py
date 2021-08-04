
# =======================================================================================
# DESCRIPTION
# =======================================================================================
# This code is used to ......
# =======================================================================================



import random
random.seed(0)

import numpy as np
np.random.seed(0)

import torch
torch.manual_seed(0);

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import statistics
import copy
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
from os import path
from skimage.transform import rotate, AffineTransform, warp

print("======================")
print("Check GPU is info")
print("======================")
print("How many GPUs are there? Answer:",torch.cuda.device_count())
print("The Current GPU:",torch.cuda.current_device())
print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
print("Is Pytorch using GPU? Answer:",torch.cuda.is_available())
print("======================")


from scripts.logs import logs
from scripts import mnist


# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");


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



# Training function for both EWC and without EWC
def train_ewc(model_input, device, task_id, x_train, t_train, optimizer, epoch, ewc_lambda):
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
	return model_input



def on_task_update(task_id, x_mem, t_mem, model_input):

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



def test(model_input, device, x_test, t_test):
	model_input.eval()
	test_loss = 0
	correct = 0
	for start in range(0, len(t_test)-1, 256):
		# print("start = ",start)
		end = start + 256
		with torch.no_grad():
			x, y = torch.from_numpy(x_test[start:end]), torch.from_numpy(t_test[start:end]).long()
			# print("x_test[start:end] = ",x_test[start:end])
			x, y = x.to(device), y.to(device)
			output = model_input(x)
			test_loss += F.cross_entropy(output, y).item() # sum up batch loss
			pred = output.max(1, keepdim=True)[1] # get the index of the max logit
			correct += pred.eq(y.view_as(pred)).sum().item()

	test_loss /= len(t_test)
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(t_test),
			100. * correct / len(t_test)))
	return 100. * correct / len(t_test)



def model_evaluation(model, datasets):
	acc = [] 
	for i in range(len(datasets)):
		_, (x_test, t_test) = datasets[i]
		acc.append(test(model, device, x_test, t_test))
		print("Dataset no. = ",i,", Accuracy =  ",acc)
	return acc



# ==========================================================
# Datasets prep
# ==========================================================

# == Get MNIST Dataset =====================================
results_logs = logs()
# for init_dataset_portion in np.arange(2,11,1)/10:

mnist.init()
x_train, t_train, x_test, t_test = mnist.load()

# === Choose retraining mode and samples sizes  ========================

# Select portion from original training dataset to use in initial training
init_dataset_portion = 1
portion_1 = int(len(x_train)*init_dataset_portion) #0.15

# Select portion from original training dataset to use for rehersal
portion_2 = int(len(x_train)/6)

# Select oversampling size of new instance
oversampling_size = 1000

# Choose to do initial training on new instance only instead of original training data
init_training_on_new_instance = False

# Choose to retrain or not
retraining_flag = False

# Choose to vary EWC when retraining or not and the ranges
vary_ewc_lambda = False
fixed_ewc_value = 0
ewc_values_list = list(np.arange(0,1,1))

# Choose to have rehersal mode on or not. 
rehersal = False


# == Create new instance ===========================================
new_instance_flag = 1
# New Instance 1
if new_instance_flag == 1:
	# Choose datapoint to create new instance out of it
	i = 1
	new_instance =  x_train[i]#[0][0]
	new_instance = 1-new_instance
	new_instance_label = t_train[i]

	# Add noise to it
	print("len(new_instance[0]) = ",len(new_instance[0]))
	print("len(new_instance[0][0]) = ",len(new_instance[0][0]))
	# print("news_instance = ",new_instance)
	for i_x in range(len(new_instance[0])):
		for i_y in range(len(new_instance[0][0])):
			# pass
			# new_instance[0][i_x][i_y] = new_instance[0][i_x][i_y] + random.uniform(0, 1) * 0.9
			new_instance[0][i_x][i_y] = min(new_instance[0][i_x][i_y], 1)

		plt.imshow(new_instance.reshape(28, 28), cmap=plt.get_cmap('gray'))
		# save the new instance picture
		plt.savefig('new_instance.png')

if new_instance_flag == 2:
	# Choose datapoint to create new instance out of it
	i = 1
	new_instance =  x_train[i]#[0][0]
	print(new_instance)
	print("new_instance.shape = ", new_instance.shape)
	print("new_instance.shape[0] = ", new_instance.shape[0])
	# new_instance = new_instance.reshape((new_instance.shape[0], 1, 28, 28))
	new_instance_label = t_train[i]
	datagen = ImageDataGenerator(zca_whitening=True)
	# datagen = ImageDataGenerator(
		# featurewise_center=True,
		# featurewise_std_normalization=True,
		# rotation_range=20,
		# width_shift_range=0.2,
		# height_shift_range=0.2,
		# horizontal_flip=True,
		# validation_split=0.2)
	print(new_instance[0])
	new_instance = datagen.fit(new_instance)
	print("Iam here")
	print(new_instance)
	# new_instance = new_instance.reshape((1, 28, 28))

	# print("new_instance.shape = ", new_instance.shape)

	plt.imshow(new_instance.reshape(28, 28), cmap=plt.get_cmap('gray'))
	# save the new instance picture
	plt.savefig('new_instance.png')

# # Setting oversampling size 
# oversampled_new_instance = []
# oversampled_new_instance_label = []
# for i in range(0,oversampling_size,1): 
# 	oversampled_new_instance.append(new_instance)
# 	oversampled_new_instance_label.append(new_instance_label)


# == Apply data augmentation to new instance ===========================================

# new_instance

retraining_dataset = [new_instance]
retraining_dataset_labels = [new_instance_label]
i = 0
# reshape to be [samples][width][height][channels]
new_instance = new_instance.reshape((new_instance.shape[0], 28, 28, 1))
# convert from int to float
new_instance = new_instance.astype('float32')
# define data preparation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit parameters from data
datagen.fit(new_instance)
new_instance[i].reshape(28, 28), cmap=pyplot.get_cmap('gray')
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# ===========================stopped here

for ang in range(-45,45,5): #tqdm(range(train_x.shape[0])):
	datagen = ImageDataGenerator(rotation_range=45)
	it = datagen.flow(samples, batch_size=1)
	batch = it.next()
    retraining_dataset.append(rotate(new_instance, angle=ang))
    retraining_dataset_labels.append(new_instance_label)
    plt.imshow(retraining_dataset[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    # save the new instance picture
    plt.savefig("{} .png".format(ang))
    i = i+1
    # final_train_data.append(np.fliplr(train_x[i]))
    # final_train_data.append(np.flipud(train_x[i]))
    # final_train_data.append(random_noise(train_x[i],var=0.2**2))
    # for j in range(5):
    #     final_target_train.append(train_y[i])




# == Task 1 prep ===========================================
if init_training_on_new_instance == True:
	# oversampling alone
	x_train1 = np.append(oversampled_new_instance, oversampled_new_instance,axis=0)
	x_train1 = x_train1[0:int(len(x_train1)/2)]

	t_train1 = np.append(oversampled_new_instance_label, oversampled_new_instance_label,axis=0)
	t_train1 = t_train1[0:int(len(t_train1)/2)]

else:
	x_train1 = x_train[0:portion_1]
	t_train1 = t_train[0:portion_1]

task_1 = [(x_train1, t_train1), (x_test, t_test)]

# == Task 2 prep ===========================================

# Sets retraining mode
if rehersal == True:
	# rehersal with oversampling of datapoint
	x_train2 = np.append(x_train[0:portion_2],oversampled_new_instance,axis=0)
	t_train2 = np.append(t_train[0:portion_2], oversampled_new_instance_label,axis=0)

else:
	# oversampling alone
	x_train2 = np.append(oversampled_new_instance, oversampled_new_instance,axis=0)
	x_train2 = x_train2[0:int(len(x_train2)/2)]

	t_train2 = np.append(oversampled_new_instance_label, oversampled_new_instance_label,axis=0)
	t_train2 = t_train2[0:int(len(t_train2)/2)]


task_2 = [(x_train2, t_train2), (x_test, t_test)]


print("len(x_train2) = ",len(x_train2))

print("len(t_train2) = ",len(t_train2))
print("t_train2 = ",t_train2)

# print("new_instance_label",new_instance_label)

# print(x_train)
# print(new_instance)

print("len(x_train) = ",len(x_train))

print("len(new_instance) = ",len(new_instance))


# task list
# tasks = [task_1, task_2, task_3]
tasks = [task_1, task_2]


# == Declaring dictionaries ========================================

fisher_dict = {}
optpar_dict = {}


# == Initial training ========================================

lr_value_init       = 0.01   # learning rate for initial learning
momentum_value_init = 0.9    # momentum value for initial learning
ewc_lambda          = 0      # setting ewc_lambda to 0, so no retraining factor
dataset_no          = 0      # dataset ID 
epoch_num_init      = 15  


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr_value_init, momentum=momentum_value_init)

print("Training on dataset no.", dataset_no, "(Initial training), with dataset size:",portion_1,"init_dataset_portion = ",init_dataset_portion)

(x_train, t_train), _ = tasks[dataset_no]
print("Training input data shape",x_train.shape)
print("Training input labels shape",t_train.shape)
for epoch in range(0, epoch_num_init):
	model = train_ewc(model, device, dataset_no, x_train, t_train, optimizer, epoch, ewc_lambda)
on_task_update(dataset_no, x_train, t_train, model)

# evaluate model
acc = model_evaluation(model, tasks)

# log
for i in range(len(tasks)):
	_, (x_test, t_test) = tasks[i]
	results_logs.append(current_training_dataset_no = dataset_no, \
						current_training_dataset_size = portion_1,\
						ewc_lambda = ewc_lambda, \
						lr_init = lr_value_init, \
						momentum_init = momentum_value_init, \
						lr_cont=0, \
						momentum_cont = 0, \
						evaluation_dataset_no = i, \
						evaluation_dataset_size = len(x_test), \
						acc = acc[i])



def test_instance(model_input, device, new_instance, new_instance_label):
	# turn model in to evaluation mode
	model_input.eval()

	# test_loss = 0
	correct = 0
	new_instance = np.array(new_instance)
	new_instance_label = np.array(new_instance_label)

	with torch.no_grad():
		x, y = torch.from_numpy(new_instance), torch.from_numpy(new_instance_label).long()
		x, y = x.to(device), y.to(device)
		output = model(x[None, ...].float())
		# print("output float = ",output)
		# test_loss += F.cross_entropy(output, y[None, ...].float()).item() # sum up batch loss

		pred = output.max(1, keepdim=True)[1] # get the index of the max logit
		correct += pred.eq(y.view_as(pred)).sum().item()
		print("correct = ",correct)
	# test_loss /= len(t_train)
	accuracy = 100. * correct #/ len(t_train))
	# print("Accuracy of new instance = ", accuracy)
	# print("correct = ",correct)
	return accuracy

print("Initial model accuracy on new instance = ",test_instance(model, device, new_instance, new_instance_label))

# == Retraining ==============================================

if retraining_flag == True:

	lr_value_cont = 1e-4                    # learning rate for continual retraining
	momentum_value_cont = 0.9               # momentum value for continual retraining
	epoch_num_cont = 100
	
	if vary_ewc_lambda == True:
		ewc_lambdas = ewc_values_list   # range of ewc_lambda used in retraining
	else:
		ewc_lambdas = [fixed_ewc_value]
	
	dataset_no          = 1                 # dataset ID 

	(x_train, t_train), _ = tasks[dataset_no]
	for ewc_lambda in ewc_lambdas:
		retrained_model = Net().to(device)
		retrained_model.load_state_dict(model.state_dict())
		optimizer = optim.SGD(retrained_model.parameters(), lr=lr_value_cont, momentum=momentum_value_cont)
		print("Retraining input data shape",x_train.shape)
		print("Retraining input labels shape",t_train.shape)
		for epoch in range(0, epoch_num_cont):
			retrained_model = train_ewc(retrained_model, device, dataset_no, x_train, t_train, optimizer, epoch, ewc_lambda)
		on_task_update(dataset_no, x_train, t_train,retrained_model)

		# evaluate model
		acc = model_evaluation(retrained_model, tasks)

		# log
		for i in range(len(tasks)):
			_, (x_test, t_test) = tasks[i]
			results_logs.append(current_training_dataset_no = dataset_no, \
								current_training_dataset_size = portion_2,\
								ewc_lambda = ewc_lambda, \
								lr_init = lr_value_init, \
								momentum_init = momentum_value_init, \
								lr_cont=lr_value_cont, \
								momentum_cont = momentum_value_cont, \
								evaluation_dataset_no = i, \
								evaluation_dataset_size = len(x_test), \
								acc = acc[i])



	print("Retrained model accuracy on new instance = ",test_instance(retrained_model, device, new_instance, new_instance_label))


# Save results
results_logs.write_file("results_22.csv")

