
import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import torch.optim as optim

import time

from matplotlib import pyplot as plt

from sklearn.utils import shuffle

from NN import Net, on_task_update, train_ewc, test


from scripts import mnist, n_mnist, oversample
from scripts.instances_generator import dset, New_instances_suite_1, New_instances_suite_2
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
num_of_oversamples = 50 # number of oversampled instances to be generated
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

Accumilation_flag     = True  


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

# input("Press enter for next to start retraining")

# ======================================================================
# == Sequential Retraining 
# ======================================================================
lr_value_retrain       = 0.01   # learning rate for retraining learning
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

			if Accumilation_flag == False:
				# make EWC lambdas exapnadable as the retraining datasets increase and all equal to 1. 
				ewc_lambdas = [0]	

				retrained_model = Net().to(device)
				retrained_model.load_state_dict(trained_models[current_ID].state_dict())
				optimizer = optim.SGD(retrained_model.parameters(), lr=lr_value_retrain, momentum=momentum_value_retrain)

				for epoch in range(0, epoch_retrain):
					retrained_model, fisher_dict, optpar_dict = train_ewc(retrained_model, device, current_ID+1, retraining_set.train_X, retraining_set.train_Y, optimizer, epoch, ewc_lambdas, fisher_dict, optpar_dict)
				fisher_dict, optpar_dict = on_task_update(current_ID+1, retraining_set.train_X, retraining_set.train_Y, retrained_model, optimizer, fisher_dict, optpar_dict)
			

			else:

				retrained_model = Net().to(device)
				retrained_model.load_state_dict(trained_models[0].state_dict()) # Use initial model to do the retraining instead of the last retrainied model
				optimizer = optim.SGD(retrained_model.parameters(), lr=lr_value_retrain, momentum=momentum_value_retrain)

				# Accumilated retraining sets
				# try to append all of the train X
				# for -- in range(retrained_sets_counter):

				
				# retraining_set.train_X
				# retraining_set.train_Y
				# print("retraining_set.train_X = ",retraining_set.train_X)
				# print("retraining_set.train_X = ",retraining_set.train_X.shape)
				# print("retraining_set.train_X = ",retraining_set.train_X.dtype)
				# print("current_ID = ", current_ID)
				noise_type_ID = 10   # mix of noise datasets
				dataset_ID = 1000000 # mix of noise datasets
				if current_ID == 0:
					train_X_temp = retraining_set.train_X
					train_Y_temp = retraining_set.train_Y
					test_X_temp = retraining_set.test_X
					test_Y_temp = retraining_set.test_Y
				else:
					train_X_temp = np.concatenate((train_X_temp, retraining_set.train_X), axis=0)
					train_Y_temp = np.concatenate((train_Y_temp, retraining_set.train_Y), axis=0)
					test_X_temp = np.concatenate((test_X_temp, retraining_set.test_X), axis=0)
					test_Y_temp = np.concatenate((test_Y_temp, retraining_set.test_Y), axis=0)

				# shuffle
				train_X_temp, train_Y_temp = shuffle(train_X_temp, train_Y_temp)
				# print(train_Y_temp)
				# input("Press enter to continue to retraining")

				# print("train_X_temp 0 = ",train_X_temp[0])
				# print("train_Y_temp 0 = ",train_Y_temp[0])
				# plt.imshow(train_X_temp[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
				# plt.savefig('batch.png')
				# input("Press enter to show next picture")

				# print("train_X_temp 1 = ",train_X_temp[1])
				# print("train_Y_temp 1 = ",train_Y_temp[1])
				# plt.imshow(train_X_temp[1].reshape(28, 28), cmap=plt.get_cmap('gray'))
				# plt.savefig('batch.png')
				# input("Press enter to show next picture")

				# print("train_X_temp 2 = ",train_X_temp[2])
				# print("train_Y_temp 2 = ",train_Y_temp[2])
				# plt.imshow(train_X_temp[2].reshape(28, 28), cmap=plt.get_cmap('gray'))
				# plt.savefig('batch.png')
				# input("Press enter to show next picture")

				# train_X_temp, train_Y_temp = shuffle(train_X_temp, train_Y_temp)

				# print("train_X_temp 0_shuffled = ",train_X_temp[0])
				# print("train_Y_temp 0_shuffled = ",train_Y_temp[0])
				# plt.imshow(train_X_temp[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
				# plt.savefig('batch.png')
				# input("Press enter to show next picture")

				# print("train_X_temp 1_shuffled = ",train_X_temp[1])
				# print("train_Y_temp 1_shuffled = ",train_Y_temp[1])
				# plt.imshow(train_X_temp[1].reshape(28, 28), cmap=plt.get_cmap('gray'))
				# plt.savefig('batch.png')
				# input("Press enter to show next picture")

				# print("train_X_temp 2_shuffled = ",train_X_temp[2])
				# print("train_Y_temp 2_shuffled = ",train_Y_temp[2])
				# plt.imshow(train_X_temp[2].reshape(28, 28), cmap=plt.get_cmap('gray'))
				# plt.savefig('batch.png')
				# input("Press enter to show next picture")

				# input("press enter to continue after shuffle")
				test_X_all_temp = None
				test_Y_all_temp = None
				accumilated_retraining_set = dset(dataset_ID, noise_type_ID,train_X_temp, train_Y_temp, test_X_temp, test_Y_temp, test_X_all_temp, test_Y_all_temp)
				# STOOPEED AT Plug in these into the retraining and retrun the overamples to 1000
				
				# print("current_ID = ", current_ID)
				# print("retraining_set.train_X = ",accumilated_retraining_set.train_X)
				# print("retraining_set.train_X.shape = ",accumilated_retraining_set.train_X.shape)
				# print("retraining_set.train_X.dtype = ",accumilated_retraining_set.train_X.dtype)
				# input("Press enter to continue retraining")

				# print("train_X_temp = ",train_X_temp)
				# print("train_X_temp.shape = ",train_X_temp.shape)
				# print("train_X_temp.dtype = ",train_X_temp.dtype)
				# input("Press enter to continue retraining")

				ewc_lambdas            = [0]
				for epoch in range(0, epoch_retrain):
					retrained_model, fisher_dict, optpar_dict = train_ewc(retrained_model, device, 1, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y, optimizer, epoch, ewc_lambdas, fisher_dict, optpar_dict)
				# fisher_dict, optpar_dict = on_task_update(0, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y, retrained_model, optimizer, fisher_dict, optpar_dict)
				# I commented out the above line as I am not doing EWC to remember the new instanes only the original trainining dataset.
			
			# Append retrained model to trained models array
			trained_models.append(retrained_model)

			end_time1 = time.time()
			

			# print("retraining_set.train_X.shape()", retraining_set.train_X.shape)
			# print("retraining_set.train_X.type()", retraining_set.train_X.dtype)
		

			# ======================================================================
			# == Evaluate Retrained model
			# ======================================================================

			# for i in range(retraining_set.ID+1):
			# 	print("===")
			
			# 	# test model on all of the previous retrainin sets tests datasets   
			# 	print("Evaluation on retrained set ID = ", retrained_sets[i].ID)

			# 	print("retraining set noise_type = ", retrained_sets[i].noise_type)
				
			# 	acc_retraining_set_training_data = test(retrained_model, device, retrained_sets[i].train_X, retrained_sets[i].train_Y) # Evaluate model on retraining training data
			# 	print("Accuracy on retraining set training data = ", acc_retraining_set_training_data)

			# 	acc_retraining_set_testing_data  = test(retrained_model, device, retrained_sets[i].test_X, retrained_sets[i].test_Y)   # Evaluate model on retraining testing data
			# 	print("Accuracy on retraining set testing data = ", acc_retraining_set_testing_data)
			print("=============================================================")
			print("=============================================================")
			
			# test model on all of the previous retrainin sets tests datasets   
			print("Evaluation on retrained set ID = ", retraining_set.ID)

			print("retraining set noise_type = ", retraining_set.noise_type)

			print("== Before retraining (original model)")

			i00=retraining_set.ID
			i01=retraining_set.noise_type
			i02=60000
			i03=num_of_oversamples
			i04=num_of_oversamples * (retraining_set.ID+1)
			i05=lr_value_retrain
			i06=ewc_lambdas[0]
			acc_train_init_orig = i1 =test(trained_models[0], device, X_train, Y_train) # Evaluate model on initial training data
			# print("1) Accuracy on initial training data (X_train, Y_train)", acc_train_init_orig)
			print("1)", acc_train_init_orig)

			acc_test_init_orig = i2 = test(trained_models[0], device, X_test, Y_test)   # Evaluate model on initial testing data
			# print("2) Accuracy on initial testing data (X_test, Y_test)", acc_test_init_orig)
			print("2)", acc_test_init_orig)

			acc_retraining_set_training_data = i3 = test(trained_models[0], device, retraining_set.train_X, retraining_set.train_Y) # Evaluate model on retraining training data
			# print("3) Accuracy on retraining set training data = ", acc_retraining_set_training_data)
			print("3) ", acc_retraining_set_training_data)
			

			acc_retraining_set_testing_data  = i4= test(trained_models[0], device, retraining_set.test_X, retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("4) Accuracy on retraining set testing data = ", acc_retraining_set_testing_data)
			print("4) ", acc_retraining_set_testing_data)

			acc_accumilated_retraining_set_training_data = i5 = test(trained_models[0], device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y) # Evaluate model on retraining training data
			# print("13) Accuracy on retraining set training data = ", acc_accumilated_retraining_set_training_data)
			print("5)  ", acc_accumilated_retraining_set_training_data)
			
			acc_accumilated_retraining_set_testing_data = i6 = test(trained_models[0], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("14) Accuracy on retraining set testing data = ", acc_accumilated_retraining_set_testing_data)
			print("6) ", acc_accumilated_retraining_set_testing_data)

			# acc_accumilated_retraining_set_training_data = test(trained_models[0], device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y) # Evaluate model on retraining training data
			# print("13) Accuracy on retraining set training data = ", acc_accumilated_retraining_set_training_data)
			
			# acc_accumilated_retraining_set_testing_data  = test(trained_models[0], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("14) Accuracy on retraining set testing data = ", acc_accumilated_retraining_set_testing_data)
			
			print("== Before retraining (last model)")

			acc_train_init_orig =i7= test(trained_models[-2], device, X_train, Y_train) # Evaluate model on initial training data
			# print("5) Accuracy on initial training data (X_train, Y_train)", acc_train_init_orig)
			print("7) ", acc_train_init_orig)

			acc_test_init_orig =i8= test(trained_models[-2], device, X_test, Y_test)   # Evaluate model on initial testing data
			# print("6) Accuracy on initial testing data (X_test, Y_test)", acc_test_init_orig)
			print("8)", acc_test_init_orig)


			acc_retraining_set_training_data= i9 = test(trained_models[-2], device, retraining_set.train_X, retraining_set.train_Y) # Evaluate model on retraining training data
			# print("7) Accuracy on retraining set training data = ", acc_retraining_set_training_data)
			print("9) ", acc_retraining_set_training_data)
			

			acc_retraining_set_testing_data= i10  = test(trained_models[-2], device, retraining_set.test_X, retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("8) Accuracy on retraining set testing data = ", acc_retraining_set_testing_data)
			print("10) ", acc_retraining_set_testing_data)

			acc_accumilated_retraining_set_training_data = i11 = test(trained_models[-2], device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y) # Evaluate model on retraining training data
			# print("13) Accuracy on retraining set training data = ", acc_accumilated_retraining_set_training_data)
			print("11)  ", acc_accumilated_retraining_set_training_data)
			
			acc_accumilated_retraining_set_testing_data = i12 = test(trained_models[-2], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("14) Accuracy on retraining set testing data = ", acc_accumilated_retraining_set_testing_data)
			print("12) ", acc_accumilated_retraining_set_testing_data)


			# acc_accumilated_retraining_set_training_data = test(trained_models[-2], device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y) # Evaluate model on retraining training data
			# print("13) Accuracy on retraining set training data = ", acc_accumilated_retraining_set_training_data)
			
			# acc_accumilated_retraining_set_testing_data  = test(trained_models[-2], device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("14) Accuracy on retraining set testing data = ", acc_accumilated_retraining_set_testing_data)

			
			print("== After retraining")

			acc_train_init_orig = i13 = test(retrained_model, device, X_train, Y_train) # Evaluate model on initial training data
			# print("9) Accuracy on initial training data (X_train, Y_train)", acc_train_init_orig)
			print("13)", acc_train_init_orig)


			acc_test_init_orig = i14 =test(retrained_model, device, X_test, Y_test)   # Evaluate model on initial testing data
			# print("10) Accuracy on initial testing data (X_test, Y_test)", acc_test_init_orig)
			print("14)", acc_test_init_orig)
			

			acc_retraining_set_training_data = i15 = test(retrained_model, device, retraining_set.train_X, retraining_set.train_Y) # Evaluate model on retraining training data
			# print("11) Accuracy on retraining set training data = ", acc_retraining_set_training_data)
			print("15)", acc_retraining_set_training_data)
			

			acc_retraining_set_testing_data = i16 = test(retrained_model, device, retraining_set.test_X, retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("12) Accuracy on retraining set testing data = ", acc_retraining_set_testing_data)
			print("16) ", acc_retraining_set_testing_data)

			acc_accumilated_retraining_set_training_data = i17 = test(retrained_model, device, accumilated_retraining_set.train_X, accumilated_retraining_set.train_Y) # Evaluate model on retraining training data
			# print("13) Accuracy on retraining set training data = ", acc_accumilated_retraining_set_training_data)
			print("17)  ", acc_accumilated_retraining_set_training_data)
			
			acc_accumilated_retraining_set_testing_data = i18 = test(retrained_model, device, accumilated_retraining_set.test_X, accumilated_retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("14) Accuracy on retraining set testing data = ", acc_accumilated_retraining_set_testing_data)
			print("118) ", acc_accumilated_retraining_set_testing_data)

			print(i00,",",i01,",",i02,",",i03,",",i04,",",i05,",",i06,",",i1,",",i2,",",i3,",",i4,",",i5,",",i6,",",i7,",",i8,",",i9,",",i10,",",i11,",",i12,",",i13,",",i14,",",i15,",",i16,",",i17,",",i18)

			with open("immediate_logs.txt", "a") as myfile:
				# myfile.write(str(i00)+","+str(i01)+","+str(i02)+","+str(i03)+","+str(i04)+","+str(i05)+","+str(i06)+","+str(i1)+","+str(i2)+","+str(i3)+","+str(i4)+","+str(i5)+","+str(i6)+","+str(i7)+","+str(i8)+","+str(i9)+","str(i10)+","+str(i11)+","+str(i12)+","+str(i13)+","+str(i14)+","+str(i15)","+str(i16)+","+str(i17)+","+str(i18))
				myfile.write('%d,%d,%d,%d,%d,%1.6f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f \n' %(i00, i01, i02, i03, i04, i05, i06, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18))
			# acc_retraining_set_training_data = test(retrained_model, device, retraining_set.train_X, retraining_set.train_Y) # Evaluate model on retraining training data
			# print("Accuracy on retraining set training data = ", acc_retraining_set_training_data)

			# acc_retraining_set_testing_data  = test(retrained_model, device, retraining_set.test_X, retraining_set.test_Y)   # Evaluate model on retraining testing data
			# print("Accuracy on retraining set testing data = ", acc_retraining_set_testing_data)

			# acc_train_init          = test(retrained_model, device, X_train, Y_train)              # Evaluate model on all initial training data
			# print("Accuracy on initial training data = ", acc_train_init)
			# print("Dropped from initial training (train) = ",acc_train_init_orig - acc_train_init)

			# acc_test_init           = test(retrained_model, device, X_test, Y_test)                # Evaluate model on all initial testing data
			# print("Accuracy on initial testing data = ", acc_test_init)
			# print("Dropped from initial training (test) = ",acc_test_init_orig - acc_test_init)

			print("===")

			# acc_train_gn_noise      = test(retrained_model, device, gn_X_train, gn_Y_train)        # Evaluate model on all gn training data
			# print("Accuracy on gn train dataset noise type 1 = ", acc_train_gn_noise)

			# acc_test_gn_noise       = test(retrained_model, device, gn_X_test, gn_Y_test)          # Evaluate model on all gn testing data
			# print("Accuracy on gn test dataset noise type 1 = ", acc_test_gn_noise)

			# acc_train_blur_noise     = test(retrained_model, device, blur_X_train, blur_Y_train)   # Evaluate model on all blur training data
			# print("Accuracy on blur train dataset noise type 2 = ", acc_train_blur_noise)

			# acc_test_blur_noise      = test(retrained_model, device, blur_X_test, blur_Y_test)          # Evaluate model on all blur testing data
			# print("Accuracy on blur test dataset noise type 2 = ", acc_test_blur_noise)

			# acc_train_contrast_noise = test(retrained_model, device, contrast_X_train, contrast_Y_train)        # Evaluate model on all contrast training data
			# print("Accuracy on contrast train dataset noise type 3 = ", acc_train_contrast_noise)

			# acc_test_contrast_noise  = test(retrained_model, device, contrast_X_test, contrast_Y_test)  # Evaluate model on all contrast testing data
			# print("Accuracy on contrast test dataset noise type 3 = ", acc_test_contrast_noise)

			end_time2 = time.time()
			training_cost = end_time1 - start_time
			print("Cost of this retraining = ",training_cost)
			training_evalation_cost = end_time2 - start_time
			print("Cost of this retraining = ",training_evalation_cost)

			# input("Press enter for next retraining set")
		else:
			print("#########################################")
			print("Retraining set is above the threshold")
			print("#########################################")
			


else:
	print("Retraining not selected")