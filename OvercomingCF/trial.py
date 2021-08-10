

from scripts import mnist, n_mnist, oversample

# Retraining dataset
n_mnist.init()
X_train, Y_train, X_test, Y_test = n_mnist.load(noise_type = "gn")
print("gn X_train.shape = ",X_train.shape)
print("gn X_train.dtype = ",X_train.dtype)

print("gn Y_train.shape = ",Y_train.shape)
print("gn Y_train.dtype = ",Y_train.dtype)

print("gn X_test.shape = ",X_test.shape)
print("gn X_test.dtype = ",X_test.dtype)

print("gn Y_test.shape = ",Y_test.shape)
print("gn Y_test.dtype = ",Y_test.dtype)
print("=============================================")

mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()
print("gn X_train.shape = ",X_train.shape)
print("gn X_train.dtype = ",X_train.dtype)

print("gn Y_train.shape = ",Y_train.shape)
print("gn Y_train.dtype = ",Y_train.dtype)

print("gn X_test.shape = ",X_test.shape)
print("gn X_test.dtype = ",X_test.dtype)

print("gn Y_test.shape = ",Y_test.shape)
print("gn Y_test.dtype = ",Y_test.dtype)

