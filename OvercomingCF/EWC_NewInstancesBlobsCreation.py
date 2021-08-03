
# =======================================================================================
# DESCRIPTION
# =======================================================================================
# This code is used to ......
# =======================================================================================

# import torch
# torch.manual_seed(1);

# import torch.nn as nn
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import torch.optim as optim
# import torch.nn.functional as F

# import torch.utils.data
# import torch.utils.data.distributed

# from scripts.logs import logs
# from scripts import mnist


import os 
import torch  
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from PIL import Image

import torchvision

# %matplotlib inline

plt.figure()
img = mpimg.imread('car.jpg')
plt.imshow(img)
plt.show()


# img = Image.open('/content/2_city_car_.jpg')

# center_crop = torchvision.transforms.CenterCrop(size=(200,300))
# img = center_crop(img)
# plt.imshow(img)

################



# new_instances_blobs = []

# from keras.datasets import mnist
# from matplotlib import pyplot
# # load data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # create a grid of 3x3 images
# for i in range(0, 9):
# 	pyplot.subplot(330 + 1 + i)
# 	pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
# # show the plot
# pyplot.show()



