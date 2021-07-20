
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import statistics
import copy

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
#mnist.init()

x_train, t_train, x_test, t_test = mnist.load()



# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
torch.manual_seed(1);


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





# task 1
portion_1 = int(len(x_train)*0.15)
task_1 = [(x_train[0:portion_1], t_train[0:portion_1]), (x_test, t_test)]

# task 2
portion_2 = int(len(x_train)*0.5)
x_train2, x_test2 = permute_mnist([x_train[0:portion_2], x_test], 1)

task_2 = [(x_train2, t_train[0:portion_2]), (x_test2, t_test)]

# x_train2, x_test2 = permute_mnist([x_train, x_test], 1)

# task_2 = [(x_train2, t_train), (x_test2, t_test)]

# task 3
x_train3, x_test3 = permute_mnist([x_train, x_test], 2)
task_3 = [(x_train3, t_train), (x_test3, t_test)]

# task list
# tasks = [task_1, task_2, task_3]
tasks = [task_1, task_2]


####################################################################################################################
#EWC


# fisher_dict = {}
# optpar_dict = {}
# ewc_lambda = list(range(20))



# model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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




def train_ewc(model_input, device, task_id, x_train, t_train, optimizer, epoch,ewc_lambda):
		model_input.train()
		
		#epcho size is 256
		for start in range(0, len(t_train)-1, 256):
			end = start + 256
			x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
			x, y = x.to(device), y.to(device)
			
			optimizer.zero_grad()

			output = model_input(x)
			loss = F.cross_entropy(output, y)
			
			### magic here! :-)
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
		print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
				test_loss, correct, len(t_test),
				100. * correct / len(t_test)))
		return 100. * correct / len(t_test)



# ewc_accs = []
# avg_acc = 0
# print("Training on task: ", 0)
	
# (x_train, t_train), _ = tasks[0]
	
# for epoch in range(0, 15):
#   model = train_ewc(model, device, 0, x_train, t_train, optimizer, epoch,0)
# on_task_update(0, x_train, t_train,model)
		
# for id_test, task in enumerate(tasks):
#   print("Testing on task: ", id_test)
#   _, (x_test, t_test) = task
#   acc = test(model, device, x_test, t_test)
#   avg_acc = avg_acc + acc
	 
# print("Avg acc: ", avg_acc / 3)
# ewc_accs.append(avg_acc / 3)

# PATH = "entire_model.pt"
# torch.save(model, PATH)


def fitness_function(ewc_lambda,lr_value, momentum_value):
	(x_train, t_train), _ = tasks[1]
	
	torch.save(model.state_dict(), 'model.pt')
	copyed_model = Net().to(device)
	copyed_model.load_state_dict(torch.load('model.pt'))

	copyed_model = copy.deepcopy(model)

	copyed_model = Net().to(device)
	copyed_model.load_state_dict(model.state_dict())
	optimizer = optim.SGD(copyed_model.parameters(), lr=lr_value, momentum=momentum_value)

	for epoch in range(0, 15):
		copyed_model = train_ewc(copyed_model, device, 1, x_train, t_train, optimizer, epoch,ewc_lambda)
	
	on_task_update(1, x_train, t_train,copyed_model)

	acc = []

	_, (x_test, t_test) = tasks[0]
	acc.append(test(copyed_model, device, x_test, t_test))

	_, (x_test, t_test) = tasks[1]
	acc.append(test(copyed_model, device, x_test, t_test))

	return acc



# performance_on_current = []
# for item in ewc_lambda:
#   performance_on_current.append(fitness_function(item))



	
# Normal
def train(model, device, x_train, t_train, optimizer, epoch):
		model.train()
		
		for start in range(0, len(t_train)-1, 256):
			end = start + 256
			x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
			x, y = x.to(device), y.to(device)
			
			optimizer.zero_grad()

			output = model(x)
			loss = F.cross_entropy(output, y)
			loss.backward()
			optimizer.step()
			#print(loss.item())
		print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))





fisher_dict = {}
optpar_dict = {}
ewc_lambda = list(np.arange(0,20,1))

results_logs = logs()

lr_value_init = 0.01 # learning rate for initial learning
momentum_value_init = 0.9 # momentum value for initial learning

lr_value_cont = 0.01 # learning rate for continual retraining
momentum_value_cont = 0.9 # momentum value for continual retraining

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr_value_init, momentum=momentum_value_init)

ewc_accs = []
avg_acc = 0
print("Training on task: ", 0)
	
(x_train, t_train), _ = tasks[0]
	
for epoch in range(0, 15):
	model = train_ewc(model, device, 0, x_train, t_train, optimizer, epoch,0)
on_task_update(0, x_train, t_train,model)
		
for id_test, task in enumerate(tasks):
	print("Testing on task: ", id_test)
	_, (x_test, t_test) = task
	acc = test(model, device, x_test, t_test)
	avg_acc = avg_acc + acc
	 
print("Avg acc: ", avg_acc / 3)
ewc_accs.append(avg_acc / 3)

PATH = "entire_model.pt"
torch.save(model, PATH)


performance_on_current = []
for item in ewc_lambda:
	performance_on_current.append(fitness_function(item,lr_value_cont,momentum_value_cont),)

	print("acc_old",performance_on_current[-1][0])
	print("acc_new",performance_on_current[-1][1])
	dataset_no = 1
	retraining_dataset_no = 2
	results_logs.append(retraining_dataset_no, item, lr_value_init, momentum_value_init, lr_value_cont, momentum_value_cont, dataset_no, portion_1, performance_on_current[-1][0])
	dataset_no = 2
	retraining_dataset_no = 2
	results_logs.append(retraining_dataset_no, item, lr_value_init, momentum_value_init, lr_value_cont, momentum_value_cont, dataset_no, portion_2, performance_on_current[-1][1])




# Save results
results_logs.write_file("results_12.csv")













#baseline_accs = []
#for id, task in enumerate(tasks):
#  avg_acc = 0
#  print("Training on task: ", id)
#  
#  (x_train, t_train), _ = task
#  
#  for epoch in range(1, 20):
#    train(model, device, x_train, t_train, optimizer, epoch)
#  on_task_update(id, x_train, t_train)
#    
#  for id_test, task in enumerate(tasks):
#    print("Testing on task: ", id_test)
#    _, (x_test, t_test) = task
#    acc = test(model, device, x_test, t_test)
#    avg_acc = avg_acc + acc
#   
#  print("Avg acc: ", avg_acc / 3)
#  baseline_accs.append(avg_acc / 3)


