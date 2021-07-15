import os

class logs():
	def __init__(self):
		self.task_no_array                         = []
		self.ewc_lambda_array                      = []
		self.lr_array                              = []
		self.acc_old_tasks_array                   = []
		self.acc_new_task_array                    = []
		self.new_data_size_array                   = []

	def append(self, task_no, ewc_lambda, lr, acc_old_tasks, acc_new_task,new_data_size):
		self.task_no_array.append(task_no)
		self.ewc_lambda_array.append(ewc_lambda)
		self.lr_array.append(lr)
		self.acc_old_tasks_array.append(acc_old_tasks)
		self.acc_new_task_array.append(acc_new_task)
		self.new_data_size_array.append(new_data_size)

	def write_file(self, file_name):
		# Folder "results" if not already there
		output_folder = "Results_logs"
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		file_path = os.path.join(output_folder, file_name)
		with open(file_path, 'w') as log_file: 
			log_file.write('task_no ,ewc_lambda, lr, acc_old_tasks, acc_new_task, new_data_size\n')
			for i in range(len(self.task_no_array)):
				log_file.write('%d, %3.6f, %3.6f, %3.6f, %3.6f, %d\n' %\
					(self.task_no_array[i],self.ewc_lambda_array[i], self.lr_array[i], self.acc_old_tasks_array[i], self.acc_new_task_array[i], self.new_data_size_array[i]))
		print('Log file SUCCESSFULLY generated!')
