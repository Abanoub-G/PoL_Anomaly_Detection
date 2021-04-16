# baseline cnn model for mnist
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()

	# # filter digit 0's and 1's
	train_filter = np.where((trainY == 0 ) | (trainY == 1))
	test_filter = np.where((testY == 2) | (testY == 2))
	# # using the filters to get the subset of arrays by index.
	trainX, trainY = trainX[train_filter], trainY[train_filter]#
	testX, testY = testX[test_filter], testY[test_filter]
	# print("len(trainX) no filter = ",len(trainX))
	# print("len(trainY) no filter = ",len(trainY))
	# print("len(testX) filter = ",len(testX))
	# print("len(testY) filter = ",len(testY))

	# # plot raw pixel data
	# pyplot.imshow(trainX[0], cmap=pyplot.get_cmap('gray'))
	# # show the figure
	# pyplot.show()



	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))

	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	print(testX.shape)
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	print("trainY = ",trainY)
	print("testY = ",testY)

	return trainX, trainY, testX, testY
	
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(30, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(2, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=2):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories, model
	
# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], marker='o', color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], marker='o', color='orange', label='test')
		pyplot.legend()
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		# print(histories[i].history.keys())
		pyplot.plot(histories[i].history['acc'], marker='o', color='blue', label='train')
		pyplot.plot(histories[i].history['val_acc'], marker='o', color='orange', label='test')
		pyplot.legend()
		pyplot.xlabel("epochs")

	pyplot.show()
	
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

def save_model(model,file_name):
	# # serialize model to JSON
	# model_json = model.to_json()
	# with open(json_file_name, "w") as json_file:
	#     json_file.write(model_json)
	# # serialize weights to HDF5
	# model.save_weights(file_name)
	model.save(file_name)
	print("Saved model to disk")

# def load_model(file_name):
# 	# # serialize model to JSON
# 	# model_json = model.to_json()
# 	# with open(json_file_name, "w") as json_file:
# 	#     json_file.write(model_json)
# 	# # serialize weights to HDF5
# 	model = load_model(file_name)
# 	print("Model loaded")
# 	return model
	 

# def load_model(json_file_name, weights_file_name): 
# 	# load json and create model
# 	json_file = open(json_file_name, 'r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	loaded_model = model_from_json(loaded_model_json)
# 	# load weights into new model
# 	loaded_model.load_weights("model.h5")
# 	print("Loaded model from disk")
	 
# 	# evaluate loaded model on test data
# 	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 	score = loaded_model.evaluate(X, Y, verbose=0)
# 	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
# 	return


# run the test harness for evaluating a model
def run_test_harness():
	# # load dataset
	# trainX, trainY, testX, testY = load_dataset()
	# # prepare pixel data
	# trainX, testX = prep_pixels(trainX, testX)
	# # evaluate model
	# scores, histories, model = evaluate_model(trainX, trainY)
	# # learning curves
	# summarize_diagnostics(histories)
	# # summarize estimated performance
	# summarize_performance(scores)
	# # save model
	# file_name = "model_1.h5"
	# save_model(model,file_name)

	file_name = "model_1.h5"
	model = load_model(file_name)



	# save array of data point of interest
	# np.save("data_point_num2", testX[0])

	# load array of data point of interest and plot
	test_data_point = np.load("data_point_num1_noisy_0.9.npy")
	pyplot.imshow(test_data_point, cmap=pyplot.get_cmap('gray'))
	pyplot.show()
	test_data_point_reshaped = test_data_point.reshape((1, 28, 28, 1))

	a = model.predict(test_data_point_reshaped)
	print(a)



# entry point, run the test harness
run_test_harness()

