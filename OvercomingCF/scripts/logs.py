import os

class logs():
	def __init__(self):
		self.current_training_dataset_no_array     = []
		self.current_training_dataset_size_array   = []
		self.ewc_lambda_array                      = []
		self.lr_init_array                         = []
		self.momentum_init_array                   = []
		self.lr_cont_array                         = []
		self.momentum_cont_array                   = []
		self.evaluation_dataset_no_array           = []
		self.evaluation_dataset_size_array         = []
		self.acc_array                             = []

	def append(self, current_training_dataset_no, current_training_dataset_size, ewc_lambda, lr_init, momentum_init, lr_cont, momentum_cont, evaluation_dataset_no, evaluation_dataset_size, acc):
		self.current_training_dataset_no_array.append(current_training_dataset_no)
		self.current_training_dataset_size_array.append(current_training_dataset_size)
		self.ewc_lambda_array.append(ewc_lambda)
		self.lr_init_array.append(lr_init)
		self.momentum_init_array.append(momentum_init)
		self.lr_cont_array.append(lr_cont)
		self.momentum_cont_array.append(momentum_cont)
		self.evaluation_dataset_no_array.append(evaluation_dataset_no)
		self.evaluation_dataset_size_array.append(evaluation_dataset_size)
		self.acc_array.append(acc)

	def write_file(self, file_name):
		# Folder "results" if not already there
		output_folder = "Results_logs"
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		file_path = os.path.join(output_folder, file_name)
		with open(file_path, 'w') as log_file: 
			log_file.write('current_training_dataset_no, current_training_dataset_size, ewc_lambda, lr_init, momentum_init, lr_cont, momentum_cont, evaluation_dataset_no, evaluation_dataset_size, acc\n')
			for i in range(len(self.current_training_dataset_no_array)):
				log_file.write('%d, %d, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %d, %d, %3.6f\n' %\
					(self.current_training_dataset_no_array[i], self.current_training_dataset_size_array[i],self.ewc_lambda_array[i], self.lr_init_array[i], self.momentum_init_array[i], self.lr_cont_array[i], self.momentum_cont_array[i], self.evaluation_dataset_no_array[i], self.evaluation_dataset_size_array[i], self.acc_array[i]))
		print('Log file SUCCESSFULLY generated!')



