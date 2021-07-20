import os

class logs():
	def __init__(self):
		self.retraining_dataset_no_array           = []
		self.ewc_lambda_array                      = []
		self.lr_init_array                         = []
		self.momentum_init_array                   = []
		self.lr_cont_array                         = []
		self.momentum_cont_array                   = []
		self.dataset_no_array                      = []
		self.dataset_size_array                    = []
		self.acc_array                             = []

	def append(self, retraining_dataset_no, ewc_lambda, lr_init, momentum_init, lr_cont, momentum_cont, dataset_no, dataset_size, acc):
		self.retraining_dataset_no_array.append(retraining_dataset_no)
		self.ewc_lambda_array.append(ewc_lambda)
		self.lr_init_array.append(lr_init)
		self.momentum_init_array.append(momentum_init)
		self.lr_cont_array.append(lr_cont)
		self.momentum_cont_array.append(momentum_cont)
		self.dataset_no_array.append(dataset_no)
		self.dataset_size_array.append(dataset_size)
		self.acc_array.append(acc)

	def write_file(self, file_name):
		# Folder "results" if not already there
		output_folder = "Results_logs"
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		file_path = os.path.join(output_folder, file_name)
		with open(file_path, 'w') as log_file: 
			log_file.write('retraining_dataset_no, ewc_lambda, lr_init, momentum_init, lr_cont, momentum_cont, dataset_no, dataset_size, acc\n')
			for i in range(len(self.retraining_dataset_no_array)):
				log_file.write('%d, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %d, %d, %3.6f\n' %\
					(self.retraining_dataset_no_array[i],self.ewc_lambda_array[i], self.lr_init_array[i], self.momentum_init_array[i], self.lr_cont_array[i], self.momentum_cont_array[i], self.dataset_no_array[i], self.dataset_size_array[i], self.acc_array[i]))
		print('Log file SUCCESSFULLY generated!')



