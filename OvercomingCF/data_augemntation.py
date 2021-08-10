
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
from sklearn.utils import shuffle

print("======================")
print("Check GPU is info")
print("======================")
print("How many GPUs are there? Answer:",torch.cuda.device_count())
print("The Current GPU:",torch.cuda.current_device())
print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
print("Is Pytorch using GPU? Answer:",torch.cuda.is_available())
print("======================")

# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");


from scripts.logs import logs
from scripts import mnist

from NN import Net, on_task_update, model_evaluation, permute_mnist, train_ewc

from keras.datasets import mnist as mnist_keras

# ==========================================================
# Datasets prep
# ==========================================================



def rotation_data_augmentation(instance,instance_label, num_of_new_instances, max_rotation):
	instance = instance.reshape((instance.shape[0], 28, 28, 1))
	instance = instance.astype('float32')
	instances = []
	instances_labels = []
	for i in range(num_of_new_instances): 
		datagen = ImageDataGenerator(rotation_range=max_rotation)
		# fit parameters from data
		datagen.fit(instance)
		for new_instance_augmented, new_instance_augmented_label in datagen.flow(instance, instance_label, batch_size=1):
			new_instance_augmented = new_instance_augmented.reshape((new_instance_augmented.shape[0], 28, 28))
			print("new_instance_augmented.dtype = ",new_instance_augmented.dtype)
			print("new_instance_augmented.shape = ",new_instance_augmented.shape)
			instances.append(new_instance_augmented)
			instances_labels = instances_labels + instance_label#instances_labels.append(new_instance_augmented_label)
			# plot
			# plt.imshow(new_instance_augmented.reshape(28, 28), cmap=plt.get_cmap('gray'))
			# plt.savefig('batch.png')
			# input("Press enter to continue")
			break # do not remove this break or you will get stuck in this for loop
	# print("instances.dtype = ",instances.dtype)
	# print("instances.shape = ",instances.shape)
	instances = np.array(instances)
	print("instances.dtype = ",instances.dtype)
	print("instances.shape = ",instances.shape)
	# input("press enter to continue")
	return instances, instances_labels

def shift_data_augmentation(instance,instance_label, num_of_new_instances, shift=0.2):
	instance = instance.reshape((instance.shape[0], 28, 28, 1))
	instance = instance.astype('float32')
	instances = []
	instances_labels = []
	for i in range(num_of_new_instances): 
		datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
		# fit parameters from data
		datagen.fit(instance)
		for new_instance_augmented, new_instance_augmented_label in datagen.flow(instance, instance_label, batch_size=1):
			new_instance_augmented = new_instance_augmented.reshape((new_instance_augmented.shape[0], 28, 28))
			instances.append(new_instance_augmented)
			instances_labels = instances_labels + instance_label#.append(new_instance_augmented_label)
			# print(len(instances))
			# print(len(instances_labels))
			# plot
			# plt.imshow(new_instance_augmented.reshape(28, 28), cmap=plt.get_cmap('gray'))
			# plt.savefig('batch.png')
			# input("Press enter to continue")
			break # do not remove this break or you will get stuck in this for loop
	instances = np.array(instances)
	print("instances.dtype = ",instances.dtype)
	print("instances.shape = ",instances.shape)
	# input("press enter to continue")
	return instances, instances_labels

def new_instances_generation(X_train, Y_train,\
								mnist_gn_train_X, mnist_gn_train_Y,\
								mnist_motion_train_X, mnist_motion_train_Y,\
								mnist_contrast_train_X, mnist_contrast_train_Y,\
								new_instance_flag):
	
	# Create the noisy instance and supply into the data augmentation code
	# == Create new instance ===========================================
	# new_instance_flag = 2
	# # New Instance 1
	# if new_instance_flag == 1:
	# 	# Choose datapoint to create new instance out of it
	# 	i = 1
	# 	new_instance =  X_train[i]#[0][0]
	# 	new_instance = 1-new_instance
	# 	new_instance_label = [Y_train[i]]

	# 	# Add noise to it
	# 	# print("len(new_instance[0]) = ",len(new_instance[0]))
	# 	# print("len(new_instance[0][0]) = ",len(new_instance[0][0]))
	# 	# print("news_instance = ",new_instance)
	# 	for i_x in range(len(new_instance[0])):
	# 		for i_y in range(len(new_instance[0][0])):
	# 			new_instance[0][i_x][i_y] = min(new_instance[0][i_x][i_y], 1)

	# 		# plt.imshow(new_instance.reshape(28, 28), cmap=plt.get_cmap('gray'))
	# 		# # save the new instance picture
	# 		# plt.savefig('new_instance.png')
	# 	print("instances.dtype = ",new_instance.dtype)
	# 	print("instances.shape = ",new_instance.shape)
	# 	plt.imshow(new_instance.reshape(28, 28), cmap=plt.get_cmap('gray'))
	# 	# save the new instance picture
	# 	plt.savefig('new_instance.png')
	# 	input("press enter to continue")


	# if new_instance_flag == 2:
		# Choose datapoint to create new instance out of it
	i = new_instance_flag
	new_instance =  mnist_contrast_train_X[i]#[0][0]
	new_instance_label = [mnist_contrast_train_Y[i]]
	# new_instance = new_instance.reshape(new_instance.shape[0], 1, 28, 28)
	# new_instance = new_instance.reshape(28, 28)

	new_instance = new_instance.reshape((new_instance.shape[0], 28, 28))
	new_instance = np.array(new_instance)
	print("instances.dtype = ",new_instance.dtype)
	print("instances.shape = ",new_instance.shape)
	plt.imshow(new_instance.reshape(28, 28), cmap=plt.get_cmap('gray'))
	# save the new instance picture
	plt.savefig('new_instance_HERE.png')
	print("new_instance_label = ", new_instance_label)
	# input("press enter to continue")

		# plt.imshow(new_instance.reshape(28, 28), cmap=plt.get_cmap('gray'))
		# # save the new instance picture
		# plt.savefig('new_instance.png')

	return new_instance, new_instance_label



def new_instance_oversampling(new_instance, new_instance_label):

	# == Data augmentation for new instance ===========================================
	# new_instance =  X_train[1]
	# new_instance_label = [Y_train[1]]

	# Add original new instance to retraining dataset
	instance = new_instance
	# print("instance.shape = ",instance.shape)

	# instance = instance.reshape((instance.shape[0], 28, 28, 1))
	print("instance.shape = ",instance.shape)
	print("instance.dtype = ",instance.dtype)
	# input("press enter to continue")
	# instance = instance.astype('float32')
	print("instances.dtype = ",instance.dtype)
	print("instances.shape = ",instance.shape)
	# input("press enter to continue 2")
	retraining_dataset        = [instance]
	retraining_dataset        = np.array(retraining_dataset)
	retraining_dataset_labels = new_instance_label

	# Create data augmented 
	num_of_new_instances = 1000 # per each data augemntation technique used
	max_rotation = 60
	max_shift    = 0.2
	rotation_oversampling, rotation_oversampling_labels = rotation_data_augmentation(new_instance,new_instance_label,num_of_new_instances,max_rotation)
	shift_oversampling, shift_oversampling_labels = shift_data_augmentation(new_instance,new_instance_label,num_of_new_instances,max_shift)

	print("instances.dtype = ",rotation_oversampling.dtype)
	print("instances.shape = ",rotation_oversampling.shape)
	# input("press enter to continue 1")
	# print("retraining_dataset = ", retraining_dataset)
	# print("retraining_dataset_labels = ", retraining_dataset_labels)
	# input("Press enter to continue")
	# Add dataaugementation instances to retraining dataset

	print("retraining_dataset.dtype = ",retraining_dataset.dtype)
	print("retraining_dataset.shape = ",retraining_dataset.shape)
	print("rotation_oversampling.dtype = ",rotation_oversampling.dtype)
	print("rotation_oversampling.shape = ",rotation_oversampling.shape)
	print("shift_oversampling.dtype = ",shift_oversampling.dtype)
	print("shift_oversampling.shape = ",shift_oversampling.shape)
	# input("press enter to continue 3")
	# retraining_dataset.extend(rotation_oversampling)
	# retraining_dataset.extend(shift_oversampling)#rotation_oversampling + shift_oversampling
	retraining_dataset = np.concatenate((retraining_dataset, rotation_oversampling, shift_oversampling), axis=0)
	retraining_dataset_labels = new_instance_label + rotation_oversampling_labels + shift_oversampling_labels
	print("retraining_dataset.dtype = ",retraining_dataset.dtype)
	print("retraining_dataset.shape = ",retraining_dataset.shape)
	# input("press enter to continue 4")
	# print("retraining_dataset = ", retraining_dataset)
	# print("retraining_dataset_labels = ", retraining_dataset_labels)
	# input("Press enter to continue")

	# Plot retraining dataset
	# for i in range(len(retraining_dataset_labels)):
		# plot
		# plt.imshow(retraining_dataset[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
		# plt.savefig('batch.png')
		# print("retraining_dataset = ", retraining_dataset)
		# print("retraining_dataset_labels = ", retraining_dataset_labels)
		# input("Press enter to continue")

	# print("rotation_oversampling = ", rotation_oversampling)
	# print("retraining_dataset_labels = ", retraining_dataset_labels)
	# print("shift_oversampling_labels = ", shift_oversampling_labels)
	print("len(retraining_dataset) = ", len(retraining_dataset))
	print("len(retraining_dataset_labels) = ", len(retraining_dataset_labels))

	only_new_instance_retraining_dataset = retraining_dataset
	only_new_instance_retraining_dataset_labels = retraining_dataset_labels




	# == Add an equal number of data points from the other classes ===========================================
	rehersal_flag = False
	if rehersal_flag == True:	
		# Extract each dataset for each class
		for digit in np.unique(Y_train):

			# get indiceis of all training point for this digit
			indicies = np.where(Y_train == digit)
			indicies = indicies[0]

			# Select a random chunks of samples for each digit of size equal to the oversampled new dataset size
			for i in range(num_of_new_instances):
				# Pick random index
				rand_index = random.choice(indicies)
				# print(rand_index)


				# Use random index to extract data point from training data
				# X_train[rand_index]
				instance = np.array([X_train[rand_index]])
				instance_label = [Y_train[rand_index]]
				print("HErrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
				print("instance.dtype = ",instance.dtype)
				print("instance.shape = ",instance.shape)
				print("retraining_dataset.dtype = ",retraining_dataset.dtype)
				print("retraining_dataset.shape = ",retraining_dataset.shape)
				# instance = instance.reshape((instance.shape[0], 28, 28, 1))
				# instance = instance.astype('float32')
				# plt.imshow(X_train[rand_index].reshape(28, 28), cmap=plt.get_cmap('gray'))
				# plt.savefig('batch.png')
				


				# Add this training data point to retraining dataset
				retraining_dataset = np.concatenate((retraining_dataset, instance), axis=0)
				# retraining_dataset = np.append(retraining_dataset, instance, axis=0)
				# retraining_dataset.append(instance)
				retraining_dataset_labels = retraining_dataset_labels + instance_label
				print("retraining_dataset.dtype = ",retraining_dataset.dtype)
				print("retraining_dataset.shape = ",retraining_dataset.shape)
				# input("Press Enter to cotinueee")

				# Remove the random index from the indices list so it does not get selected again in the next run 
				rm_index = np.where(indicies == rand_index)
				# print("len(indicies) before = ",len(indicies))
				indicies = np.delete(indicies, rm_index)
				# print("len(indicies) after = ",len(indicies))
				# print(rm_index)
				# print(np.where(indicies == rand_index))
				# print()


	# == Shuffle retraining dataset ===========================================
	only_new_instance_retraining_dataset, only_new_instance_retraining_dataset_labels = shuffle(only_new_instance_retraining_dataset, only_new_instance_retraining_dataset_labels, random_state=0)
	retraining_dataset, retraining_dataset_labels = shuffle(retraining_dataset, retraining_dataset_labels, random_state=0)
	print("=====================================================================")
	print("only_new_instance_retraining_dataset.dtype = ",only_new_instance_retraining_dataset.dtype)
	print("only_new_instance_retraining_dataset.shape = ",only_new_instance_retraining_dataset.shape)
	print("retraining_dataset.dtype = ",retraining_dataset.dtype)
	print("retraining_dataset.shape = ",retraining_dataset.shape)

	# print("retraining_dataset_labels = ", retraining_dataset_labels)
	# print("len(retraining_dataset) = ", len(retraining_dataset))
	# print("len(retraining_dataset_labels) = ", len(retraining_dataset_labels))

	# for i in range(len(only_new_instance_retraining_dataset_labels)):
	# 	# plot
	# 	plt.imshow(only_new_instance_retraining_dataset[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
	# 	plt.savefig('batch.png')
	# 	print(only_new_instance_retraining_dataset_labels[i])
	# 	# input("Press Enter to cotinue")

	# for i in range(len(retraining_dataset_labels)):
	# 	# plot
	# 	plt.imshow(retraining_dataset[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
	# 	plt.savefig('batch.png')
	# 	print(retraining_dataset_labels[i])
	# 	input("Press Enter to cotinue")
	return retraining_dataset, retraining_dataset_labels, only_new_instance_retraining_dataset, only_new_instance_retraining_dataset_labels
		




# == Datasets prep ===========================================
mnist.init()

X_train, Y_train, X_test, Y_test = mnist.load()

from scipy.io import loadmat
mnist_gn = loadmat("mnist-with-awgn.mat")
mnist_motion = loadmat("mnist-with-motion-blur.mat")
mnist_contrast = loadmat("mnist-with-reduced-contrast-and-awgn.mat")
print(mnist_motion)
mnist_gn_train_X = mnist_gn["train_x"]
mnist_gn_train_Y = mnist_gn["train_y"]
mnist_motion_train_X = mnist_motion["train_x"]
mnist_motion_train_Y = mnist_motion["train_y"]
mnist_contrast_train_X = mnist_contrast["train_x"]
mnist_contrast_train_Y = mnist_contrast["train_y"]
print("mnist_motion_train_X = ", mnist_motion_train_X)
print("mnist_motion_train_X.dtype = ",mnist_motion_train_X.dtype)
print("mnist_motion_train_X.shape = ",mnist_motion_train_X.shape)
print("mnist_motion_train_Y.dtype = ",mnist_motion_train_Y.dtype)
print("mnist_motion_train_Y.shape = ",mnist_motion_train_Y.shape)
mnist_gn_train_X = mnist_gn_train_X.astype('float32')
mnist_gn_train_X = mnist_gn_train_X.reshape((mnist_motion_train_X.shape[0], 1, 28, 28))
mnist_motion_train_X = mnist_motion_train_X.astype('float32')
mnist_motion_train_X = mnist_motion_train_X.reshape((mnist_motion_train_X.shape[0], 1, 28, 28))
mnist_contrast_train_X = mnist_contrast_train_X.astype('float32')
mnist_contrast_train_X = mnist_contrast_train_X.reshape((mnist_motion_train_X.shape[0], 1, 28, 28))
print("mnist_motion_train_X.dtype = ",mnist_motion_train_X.dtype)
print("mnist_motion_train_X.shape = ",mnist_motion_train_X.shape)
i = 0
plt.imshow(mnist_gn_train_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.savefig('mnist_gn_train_X.png')

plt.imshow(mnist_motion_train_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.savefig('mnist_motion_train_X.png')

plt.imshow(mnist_contrast_train_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.savefig('mnist_contrast_train_X.png')
mnist_gn_train_Y_temp = []
mnist_motion_train_Y_temp = []
mnist_contrast_train_Y_temp = []
for j in range(len(mnist_gn_train_Y)):
	mnist_gn_train_Y_temp.append(np.where(mnist_gn_train_Y[j] == 1)[0][0])
	mnist_motion_train_Y_temp.append(np.where(mnist_motion_train_Y[j] == 1)[0][0])
	mnist_contrast_train_Y_temp.append(np.where(mnist_contrast_train_Y[j] == 1)[0][0])
	# print(mnist_gn_train_Y_temp)

	# plt.imshow(mnist_gn_train_X[j].reshape(28, 28), cmap=plt.get_cmap('gray'))
	# plt.savefig('mnist_gn_train_X.png')
	# input("press eneter")
mnist_gn_train_Y = np.array(mnist_gn_train_Y_temp, dtype=np.uint8) 
mnist_motion_train_Y = np.array(mnist_motion_train_Y_temp, dtype=np.uint8) 
mnist_contrast_train_Y = np.array(mnist_contrast_train_Y_temp, dtype=np.uint8) 
print("mnist_gn_train_Y[i] = ", mnist_gn_train_Y[i])
print("mnist_motion_train_Y[i] = ", mnist_motion_train_Y[i])
print("mnist_contrast_train_Y[i] = ", mnist_contrast_train_Y[i])

print("mnist_gn_train_Y.dtype = ",mnist_gn_train_Y.dtype)
print("mnist_gn_train_Y.shape = ",mnist_gn_train_Y.shape)

print("Y_train.dtype = ",Y_train.dtype)
print("Y_train.shape = ",Y_train.shape)

print("X_train.dtype = ",X_train.dtype)
print("X_train.shape = ",X_train.shape)
print("Y_train[i] = ", Y_train[i])
plt.imshow(X_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.savefig('X_train.png')
# input("Press Enter to cotinueee")

initial_training = True
# indicies_of_bad_performing_datapoints = []
for new_instance_flag in range(len(X_train)):

	# new_instance_flag = 2
	new_instance, new_instance_label = new_instances_generation(X_train, \
																Y_train, \
																mnist_gn_train_X,\
																mnist_gn_train_Y,\
																mnist_motion_train_X,\
																mnist_motion_train_Y,\
																mnist_contrast_train_X,\
																mnist_contrast_train_Y,\
																new_instance_flag)

	retraining_dataset, \
	retraining_dataset_labels,\
	only_new_instance_retraining_dataset, \
	only_new_instance_retraining_dataset_labels = new_instance_oversampling(new_instance, new_instance_label)

	retraining_dataset_labels = np.array(retraining_dataset_labels, dtype=np.uint8)
	only_new_instance_retraining_dataset_labels = np.array(only_new_instance_retraining_dataset_labels, dtype=np.uint8)



	# print("len(X_train) = ",len(X_train))
	# print("X_train.dtype = ",X_train.dtype)
	# print("X_train.shape = ",X_train.shape)

	# print("len(Y_train) = ",len(Y_train))
	# print("Y_train.dtype = ",Y_train.dtype)
	# print("Y_train.shape = ",Y_train.shape)

	# print("len(X_test) = ",len(X_test))
	# print("X_test.dtype = ",X_test.dtype)
	# print("X_test.shape = ",X_test.shape)

	# print("len(Y_test) = ",len(Y_test))
	# print("Y_test.dtype = ",Y_test.dtype)
	# print("Y_test.shape = ",Y_test.shape)

	# print("len(retraining_dataset_labels) = ",len(retraining_dataset_labels))
	# print("retraining_dataset_labels.dtype = ", retraining_dataset_labels.dtype)
	# print("retraining_dataset_labels = ", retraining_dataset_labels)
	# print("retraining_dataset_labels.shape = ",retraining_dataset_labels.shape)

	# # print("retraining_dataset.shape = ",retraining_dataset.shape)
	# # retraining_dataset = np.array(retraining_dataset, dtype=np.float32)
	# print("len(retraining_dataset) = ",len(retraining_dataset))
	# print("retraining_dataset.dtype = ", retraining_dataset.dtype)
	# print("retraining_dataset.shape = ",retraining_dataset.shape)


	# # print("len(retraining_dataset_labels) = ",len(retraining_dataset_labels))
	# # print("retraining_dataset_labels = ", retraining_dataset_labels)
	# # print("retraining_dataset_labels.dtype = ",retraining_dataset_labels.dtype)

	# # print("len(retraining_dataset) = ",len(retraining_dataset))
	# # print("retraining_dataset.dtype = ",retraining_dataset.dtype)






	# task 1
	portion_1 = portion_2 = 1
	# portion_1 = int(len(x_train)*init_dataset_portion) #0.15
	# task_1 = [(X_train[0:portion_1], t_train[0:portion_1]), (x_test, t_test)]
	task_1 = [(X_train, Y_train), (X_test, Y_test),\
				(retraining_dataset, retraining_dataset_labels),\
				(only_new_instance_retraining_dataset, only_new_instance_retraining_dataset_labels)]

	# task 2
	task_2 = [(retraining_dataset, retraining_dataset_labels), (X_test, Y_test),\
				(retraining_dataset, retraining_dataset_labels),\
				(only_new_instance_retraining_dataset, only_new_instance_retraining_dataset_labels)]

	# task 3
	task_3 = [(only_new_instance_retraining_dataset, only_new_instance_retraining_dataset_labels), (X_test, Y_test),\
				(retraining_dataset, retraining_dataset_labels),\
				(only_new_instance_retraining_dataset, only_new_instance_retraining_dataset_labels)]

	# x_train2, x_test2 = permute_mnist([x_train[0:portion_2], x_test], 1)
	# task_2 = [(x_train2, t_train[0:portion_2]), (x_test2, t_test)]

	# task 3
	# x_train3, x_test3 = permute_mnist([x_train, x_test], 2)
	# task_3 = [(x_train3, t_train), (x_test3, t_test)]

	# task list
	# tasks = [task_1, task_2, task_3]

	# tasks = [task_1, task_2]
	tasks = [task_1, task_3]


	# == Declaring dictionaries ========================================

	fisher_dict = {}
	optpar_dict = {}

	results_logs = logs()

	if initial_training == True:
		initial_training = False
		# == Initial training ========================================

		lr_value_init       = 0.01   # learning rate for initial learning
		momentum_value_init = 0.9    # momentum value for initial learning
		ewc_lambda          = 0      # setting ewc_lambda to 0, so no retraining factor
		dataset_no          = 0      # dataset ID   


		model = Net().to(device)
		optimizer = optim.SGD(model.parameters(), lr=lr_value_init, momentum=momentum_value_init)

		# print("Training on dataset no.", dataset_no, "(Initial training), with dataset size:",portion_1,"init_dataset_portion = ",init_dataset_portion)

		# (x_train, t_train), _ = tasks[dataset_no]
		for epoch in range(0, 30):
			model, fisher_dict, optpar_dict = train_ewc(model, device, dataset_no, X_train, Y_train, optimizer, epoch, ewc_lambda, fisher_dict, optpar_dict)
		fisher_dict, optpar_dict = on_task_update(dataset_no, X_train, Y_train, model,optimizer, fisher_dict, optpar_dict)

	# evaluate model
	acc = model_evaluation(model, tasks)

	# log
	for i in range(len(tasks)):
		_, (temp_X_test, temp_Y_test),_,_ = tasks[i]
		# results_logs.append(current_training_dataset_no = dataset_no, \
		# 					current_training_dataset_size = portion_1,\
		# 					ewc_lambda = ewc_lambda, \
		# 					lr_init = lr_value_init, \
		# 					momentum_init = momentum_value_init, \
		# 					lr_cont=0, \
		# 					momentum_cont = 0, \
		# 					evaluation_dataset_no = i, \
		# 					evaluation_dataset_size = len(temp_X_test), \
		# 					acc = acc[i])

	if acc < 15.:
		print("new_instance_flag = ", new_instance_flag)
		print("new_instance_label = ", new_instance_label)
		input("Press enter")


	# == Retraining ==============================================

	retraining_flag = False
	vary_ewc_lambda = False


	if retraining_flag == True:

		lr_value_cont = 0.0001                    # learning rate for continual retraining
		momentum_value_cont = 0.9               # momentum value for continual retraining
		
		if vary_ewc_lambda == True:
			ewc_lambdas = list(np.arange(0,20,1))   # range of ewc_lambda used in retraining
		else:
			ewc_lambdas = [20]
		
		dataset_no          = 1                 # dataset ID 

		(x_train, t_train), _, _, _ = tasks[dataset_no]
		for ewc_lambda in ewc_lambdas:
			retrained_model = Net().to(device)
			retrained_model.load_state_dict(model.state_dict())
			optimizer = optim.SGD(retrained_model.parameters(), lr=lr_value_cont, momentum=momentum_value_cont)

			for epoch in range(0, 30):
				retrained_model, fisher_dict, optpar_dict = train_ewc(retrained_model, device, dataset_no, x_train, t_train, optimizer, epoch, ewc_lambda, fisher_dict, optpar_dict)
			fisher_dict, optpar_dict = on_task_update(dataset_no, x_train, t_train,retrained_model,optimizer, fisher_dict, optpar_dict)

			# evaluate model
			acc = model_evaluation(retrained_model, tasks)

			# log
			for i in range(len(tasks)):
				_, (temp_X_test, temp_Y_test), _, _  = tasks[i]
				results_logs.append(current_training_dataset_no = dataset_no, \
									current_training_dataset_size = portion_2,\
									ewc_lambda = ewc_lambda, \
									lr_init = lr_value_init, \
									momentum_init = momentum_value_init, \
									lr_cont=lr_value_cont, \
									momentum_cont = momentum_value_cont, \
									evaluation_dataset_no = i, \
									evaluation_dataset_size = len(temp_X_test), \
									acc = acc[i])




# Save results
results_logs.write_file("results_21.csv")