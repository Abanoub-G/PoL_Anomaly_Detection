
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

from scripts.logs import logs
from scripts import mnist, n_mnist, oversample

from NN import Net, on_task_update, model_evaluation, permute_mnist, train_ewc, test

from keras.datasets import mnist as mnist_keras

# ======================================================================
# == Check GPU is connected
# ======================================================================

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


# ======================================================================
# == Declarations
# ======================================================================
Misclassification_threshold         = 50 # Set the threshold for retraining
Misclassification_counter           = 0  # Counter to count number of mis-classifications
Misclassification_counter_threshold = 5 # Max number of misclassifications allowed to do retraining for.

Initial_training_flag = True

Retraining_flag       = False

new_instance_flag     = False

fisher_dict = {}
optpar_dict = {}

results_logs = logs()


# ======================================================================
# == Datasets prep
# ======================================================================

# Initial training dataset
mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

# Retraining dataset
n_mnist.init()
gn_dataset = [X_train_gn, Y_train_gn, X_test_gn, Y_test_gn] = n_mnist.load("gn") # load gaussian nosie dataset
blur_dataset = [X_train_blur, Y_train_blur, X_test_blur, Y_test_blur] = n_mnist.load("blur") # load blur dataset
contrast_dataset = [X_train_contrast, Y_train_contrast, X_test_contrast, Y_test_contrast] = n_mnist.load("contrast") # load low contrast dataset

# retraining sets representing the open world environment
retraining_sets = New_instances_suite_2(gn_dataset, blur_dataset, contrast_dataset)

# ======================================================================
# == Initial Training
# ======================================================================
if Initial_training_flag == True:
	Initial_training_flag = False

	lr_value_init       = 0.01   # learning rate for initial learning
	momentum_value_init = 0.9    # momentum value for initial learning
	epoch_init          = 30     # epoch value
	ewc_lambda          = 0      # setting ewc_lambda to 0, so no retraining factor
	dataset_no          = 0      # dataset ID   


	model = Net().to(device)
	optimizer = optim.SGD(model.parameters(), lr=lr_value_init, momentum=momentum_value_init)

	for epoch in range(0, epoch_init):
		model, fisher_dict, optpar_dict = train_ewc(model, device, dataset_no, X_train, Y_train, optimizer, epoch, ewc_lambda, fisher_dict, optpar_dict)
	fisher_dict, optpar_dict = on_task_update(dataset_no, X_train, Y_train, model,optimizer, fisher_dict, optpar_dict)


# ======================================================================
# == Evaluate Initial Training model
# ======================================================================

acc_train_init = test(model, device, X_train, Y_train) # Evaluate model on initial training data
print("Accuracy on initial training data (X_train, Y_train)", acc_train_init)

acc_test_init = test(model, device, X_test, Y_test)   # Evaluate model on initial testing data
print("Accuracy on initial testing data (X_test, Y_test)", acc_test_init)


# ======================================================================
# == Operational Environment
# ======================================================================

# Loop through all of the instances from my propriatory instances and the N-mnist datasets
for instance, instance_label in X_retraining, Y_retraining:
	
	# Do oversampling to the instance 
	instance_oversampled, instance_label_oversampled = oversample(instance, instance_label)

	# Evaluate oversampled instance on trained model
	acc_instance = test(model, device, instance_oversampled, instance_label_oversampled) # Evaluate model on input instance oversampled
	print("Accuracy on initial training data (instance_oversampled, instance_oversampled)", acc_instance)

	# If classification accuracy of instance is below threshold then consider it to be a new instance
	if acc_instance < Misclassification_threshold:
		print("Found a misclassified instance with accuracy = ", acc_instance)
		input("Press enter to continue")
		new_instance_flag              = True

		# ======================================================================
		# == Sequential Retraining 
		# ======================================================================
		if Retraining_flag == True:
			lr_value_retrain       = 0.01   # learning rate for retraining learning
			momentum_value_retrain = 0.9    # momentum value for retraining learning
			epoch_retrain          = 30     # epoch value 
			ewc_lambda             = 0      # setting ewc_lambda or the max lambda value
			vary_ewc_lambda        = False
			dataset_no             = 1      # dataset ID 

			if vary_ewc_lambda == True:
				ewc_lambdas = list(np.arange(0,ewc_lambda,1))   # range of ewc_lambda used in retraining
			else:
				ewc_lambdas = [ewc_lambda]


			model = Net().to(device)
			optimizer = optim.SGD(model.parameters(), lr=lr_value_retrain, momentum=momentum_value_retrain)

			for epoch in range(0, epoch_retrain):
				model, fisher_dict, optpar_dict = train_ewc(model, device, dataset_no, instance_oversampled, instance_label_oversampled, optimizer, epoch, ewc_lambda, fisher_dict, optpar_dict)
			fisher_dict, optpar_dict = on_task_update(dataset_no, X_train, Y_train, model,optimizer, fisher_dict, optpar_dict)
			
			# ======================================================================
			# == Evaluate Retrained model
			# ======================================================================
			acc_train_init = test(model, device, X_train, Y_train) # Evaluate model on initial training data
			print("Accuracy on initial training data (X_train, Y_train)", acc_train_init)

			acc_test_init = test(model, device, X_test, Y_test)   # Evaluate model on initial testing data
			print("Accuracy on initial testing data (X_test, Y_test)", acc_test_init)

			acc_instance = test(model, device, instance_oversampled, instance_label_oversampled) # Evaluate model on oversampled instance
			print("Accuracy on initial training data (X_train, Y_train)", acc_instance)

			# ======================================================================
			# == Sequential Accumilated Retraining 
			# ======================================================================

		else: 

			new_instance_flag = False






	# ADD evaluation on testing the noisy pattern to see whether retraining generalises or not. 

	# Retrain on new instance

	# First try retraining by just reducing the learning rate 
	# Evaluate retrained model on 1) init training, 2) init testing and 3)oversampled new instance datastes

	# Fourth try retraining by using EWC alone
	# Evaluate retrained model on 1) init training, 2) init testing and 3)oversampled new instance datastes

	# Fifth try retraining by just using rehersal alone 
	# Evaluate retrained model on 1) init training, 2) init testing and 3)oversampled new instance datastes

	# Second try retraining by just reducing learning rate and EWC
	# Evaluate retrained model on 1) init training, 2) init testing and 3)oversampled new instance datastes

	# Third try retraining by just reducing learning rate, rehersal
	# Evaluate retrained model on 1) init training, 2) init testing and 3)oversampled new instance datastes

	# Third try retraining by just by using EWC and rehersal
	# Evaluate retrained model on 1) init training, 2) init testing and 3)oversampled new instance datastes

	# Third try retraining by just reducing learning rate, rehersal and EWC
	# Evaluate retrained model on 1) init training, 2) init testing and 3)oversampled new instance datastes


	Rehersal flag  = True
	EWC_falg       = True

	# After retraining check performance on new instance and test dataset
