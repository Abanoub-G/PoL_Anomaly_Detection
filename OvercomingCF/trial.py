
import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import torch.optim as optim

import time

from NN import Net, on_task_update, train_ewc, test


from scripts import mnist, n_mnist, oversample
from scripts.instances_generator import New_instances_suite_1, New_instances_suite_2
from scripts.logs import logs




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
# == Initial training datasets prep
# ======================================================================
mnist.init()

X_train, Y_train, X_test, Y_test = mnist.load()

# ======================================================================
# == Retraining datasets prep
# ======================================================================
n_mnist.init()

gn_X_train, gn_Y_train ,gn_X_test, gn_Y_test, gn_dataset                              = n_mnist.load("gn") # load gaussian nosie dataset
blur_X_train, blur_Y_train ,blur_X_test, blur_Y_test, blur_dataset                    = n_mnist.load("blur") # load blur nosie dataset
contrast_X_train, contrast_Y_train ,contrast_X_test,contrast_Y_test, contrast_dataset = n_mnist.load("contrast") # load contrast noise dataset

# retraining sets representing the open world environment
n_retraining_sets  = 20  # Number of differnt retraining sets
num_of_oversamples = 1000 # number of oversampled instances to be generated
max_rotation = 30
max_shift    = 0.2

# TODO: 
# Add rehersal flag here 
# Modify new instances generator to incorporate rehersal 
# Allow the rehersal dataset to inputed like the noise datasets
# Add the number of instacnes to be used as rehersal instnaces 

retraining_sets = New_instances_suite_1(n_retraining_sets, num_of_oversamples, \
								max_rotation,max_shift, \
								gn_dataset, blur_dataset, contrast_dataset)

# retraining_sets = New_instances_suite_2(gn_dataset, blur_dataset, contrast_dataset)

print("len(retraining_sets) = ", len(retraining_sets))


# ======================================================================
# == Declarations
# ======================================================================
Misclassification_threshold         = 50 # Set the threshold for retraining
Misclassification_counter           = 0  # Counter to count number of mis-classifications
Misclassification_counter_threshold = 5 # Max number of misclassifications allowed to do retraining for.

Initial_training_flag = True

Retraining_flag       = True


fisher_dict = {}
optpar_dict = {}

results_logs = logs()

# ======================================================================
# == Initial Training
# ======================================================================

if Initial_training_flag == True:
	Initial_training_flag = False

	init_start_time = time.time()
	
	lr_value_init       = 0.01   # learning rate for initial learning
	momentum_value_init = 0.9    # momentum value for initial learning
	epoch_init          = 30     # epoch value
	ewc_lambda          = 0      # setting ewc_lambda to 0, so no retraining factor  


	model = Net().to(device)
	optimizer = optim.SGD(model.parameters(), lr=lr_value_init, momentum=momentum_value_init)

	for epoch in range(0, epoch_init):
		model, fisher_dict, optpar_dict = train_ewc(model, device, 0, X_train, Y_train, optimizer, epoch, ewc_lambda, fisher_dict, optpar_dict)
	fisher_dict, optpar_dict = on_task_update(0, X_train, Y_train, model, optimizer, fisher_dict, optpar_dict)

	init_end_time1 = time.time()

# ======================================================================
# == Evaluate Initial Training model
# ======================================================================

acc_train_init_orig = test(model, device, X_train, Y_train) # Evaluate model on initial training data
print("Accuracy on initial training data (X_train, Y_train)", acc_train_init_orig)

acc_test_init_orig = test(model, device, X_test, Y_test)   # Evaluate model on initial testing data
print("Accuracy on initial testing data (X_test, Y_test)", acc_test_init_orig)

init_end_time2 = time.time()
init_training_cost = init_end_time1 - init_start_time
print("Cost of initial training = ",init_training_cost)
init_training_evaluation_cost = init_end_time2 - init_start_time
print("Cost of initial training = ",init_training_evaluation_cost)

input("Press enter for next to start retraining")

# ======================================================================
# == Sequential Retraining 
# ======================================================================
lr_value_retrain       = 0.0001   # learning rate for retraining learning
momentum_value_retrain = 0.9    # momentum value for retraining learning
epoch_retrain          = 30     # epoch value 
ewc_lambdas            = []
trained_models       = [model] 
# ewc_lambda             = 0      # setting ewc_lambda or the max lambda value
# vary_ewc_lambda        = False
# dataset_no             = 1      # dataset ID 

# if vary_ewc_lambda == True:
# 	ewc_lambdas = list(np.arange(0,ewc_lambda,1))   # range of ewc_lambda used in retraining
# else:
# 	ewc_lambdas = [ewc_lambda]
retrained_sets = []
retrained_sets_counter = 0

if Retraining_flag == True :
	for retraining_set in retraining_sets:

		acc = test(trained_models[-1], device, retraining_set.train_X, retraining_set.train_Y) # Evaluate model on retraining training data
		print("acc on retraining set before retraining = ", acc)

		# if classification is less than threshold set then do retraining
		if acc < Misclassification_threshold:
			
			current_ID = retraining_set.ID = retrained_sets_counter # update the retraining set ID, important incase retraining set is above retraingin threshold
			retrained_sets_counter += 1  # Add one to the counter for the next retraining set 
			retrained_sets.append(retraining_set) # Add this retraining set to the retrained sets

			print("==============================================================")
			print("Retraining ID = ",current_ID)
			print("==============================================================")

			start_time = time.time()

			# make EWC lambdas exapnadable as the retraining datasets increase and all equal to 1. 
			ewc_lambdas.append(0)	

			retrained_model = Net().to(device)
			retrained_model.load_state_dict(trained_models[current_ID].state_dict())
			optimizer = optim.SGD(retrained_model.parameters(), lr=lr_value_retrain, momentum=momentum_value_retrain)


			# model = Net().to(device)
			# optimizer = optim.SGD(model.parameters(), lr=lr_value_retrain, momentum=momentum_value_retrain)

			for epoch in range(0, epoch_retrain):
				retrained_model, fisher_dict, optpar_dict = train_ewc(retrained_model, device, current_ID+1, retraining_set.train_X, retraining_set.train_Y, optimizer, epoch, ewc_lambdas, fisher_dict, optpar_dict)
			fisher_dict, optpar_dict = on_task_update(current_ID+1, retraining_set.train_X, retraining_set.train_Y, retrained_model, optimizer, fisher_dict, optpar_dict)
			
			# Append retrained model to trained models array
			trained_models.append(retrained_model)

			end_time1 = time.time()
			

			print("retraining_set.train_X.shape()", retraining_set.train_X.shape)
			print("retraining_set.train_X.type()", retraining_set.train_X.dtype)
		

			# ======================================================================
			# == Evaluate Retrained model
			# ======================================================================

			for i in range(retraining_set.ID+1):
				print("===")
			
				# test model on all of the previous retrainin sets tests datasets   
				print("Evaluation on retrained set ID = ", retrained_sets[i].ID)

				print("retraining set noise_type = ", retrained_sets[i].noise_type)
				
				acc_retraining_set_training_data = test(retrained_model, device, retrained_sets[i].train_X, retrained_sets[i].train_Y) # Evaluate model on retraining training data
				print("Accuracy on retraining set training data = ", acc_retraining_set_training_data)

				acc_retraining_set_testing_data  = test(retrained_model, device, retrained_sets[i].test_X, retrained_sets[i].test_Y)   # Evaluate model on retraining testing data
				print("Accuracy on retraining set testing data = ", acc_retraining_set_testing_data)

			print("===")

			acc_train_init          = test(retrained_model, device, X_train, Y_train)              # Evaluate model on all initial training data
			print("Accuracy on initial training data = ", acc_train_init)
			print("Dropped from initial training (train) = ",acc_train_init_orig - acc_train_init)

			acc_test_init           = test(retrained_model, device, X_test, Y_test)                # Evaluate model on all initial testing data
			print("Accuracy on initial testing data = ", acc_test_init)
			print("Dropped from initial training (test) = ",acc_test_init_orig - acc_test_init)

			print("===")

			acc_train_gn_noise      = test(retrained_model, device, gn_X_train, gn_Y_train)        # Evaluate model on all gn training data
			print("Accuracy on gn train dataset noise type 1 = ", acc_train_gn_noise)

			acc_test_gn_noise       = test(retrained_model, device, gn_X_test, gn_Y_test)          # Evaluate model on all gn testing data
			print("Accuracy on gn test dataset noise type 1 = ", acc_test_gn_noise)

			acc_train_blur_noise     = test(retrained_model, device, blur_X_train, blur_Y_train)   # Evaluate model on all blur training data
			print("Accuracy on blur train dataset noise type 2 = ", acc_train_blur_noise)

			acc_test_blur_noise      = test(retrained_model, device, blur_X_test, blur_Y_test)          # Evaluate model on all blur testing data
			print("Accuracy on blur test dataset noise type 2 = ", acc_test_blur_noise)

			acc_train_contrast_noise = test(retrained_model, device, contrast_X_train, contrast_Y_train)        # Evaluate model on all contrast training data
			print("Accuracy on contrast train dataset noise type 3 = ", acc_train_contrast_noise)

			acc_test_contrast_noise  = test(retrained_model, device, contrast_X_test, contrast_Y_test)  # Evaluate model on all contrast testing data
			print("Accuracy on contrast test dataset noise type 3 = ", acc_test_contrast_noise)

			end_time2 = time.time()
			training_cost = end_time1 - start_time
			print("Cost of this retraining = ",training_cost)
			training_evalation_cost = end_time2 - start_time
			print("Cost of this retraining = ",training_evalation_cost)

			input("Press enter for next retraining set")
		else:
			print("#########################################")
			print("Retraining set is above the threshold")
			print("#########################################")
			


else:
	print("Retraining not selected")