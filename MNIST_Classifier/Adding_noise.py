# baseline cnn model for mnist
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
import random

# load array of data point of interest and plot
image = np.load("data_point_num1.npy")
pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
pyplot.show()
# print(image.shape[0])
# print(image.shape[1])
for i_x in range(image.shape[0]):
	for i_y in range(image.shape[1]):
		# print(image[i_x][i_y])
		image[i_x][i_y] = image[i_x][i_y] + random.uniform(0, 1) * 0.9
		image[i_x][i_y] = min(image[i_x][i_y], 1)

		# random.uniform(0, 1) * 0.5
pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
pyplot.show()

# save array of data point of interest
np.save("data_point_num1_noisy_0.9", image)
print("Noisy image saved")


# image = 



# test_data_point_reshaped = test_data_point.reshape((1, 28, 28, 1))
