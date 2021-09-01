import numpy as np
from urllib import request
from os import path
from scipy.io import loadmat
import tarfile

filename = [
    ["n_mnist_awgn", "mnist-with-awgn.gz"],
    ["n_mnist_motion", "mnist-with-motion-blur.gz"],
    ["n_mnist_contrast", "mnist-with-reduced-contrast-and-awgn.gz"],
]


def download_file(url, filename):
    opener = request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0')
    opener.retrieve(url, filename)


def download_n_mnist():
    if path.exists("mnist-with-awgn.gz") and path.exists("mnist-with-motion-blur.gz") and path.exists("mnist-with-reduced-contrast-and-awgn.gz"):
        print('Original files from website already downloaded!')
    else:
        try:
            base_url = "http://csc.lsu.edu/~saikat/n-mnist/data/"

            for name in filename:
                print("Downloading " + name[1] + "...")
                print("Please wait for me... :)")
                download_file(base_url + name[1], name[1])
            print("Download complete.")

        except:
            print("Can find files in website please check manually this website to download files: http://csc.lsu.edu/~saikat/n-mnist/")

    if path.exists("mnist-with-awgn.mat") and path.exists("mnist-with-motion-blur.mat") and path.exists("mnist-with-reduced-contrast-and-awgn.mat"):
        print("N-MNIST dataset zip files are already unziped")

    elif path.exists("mnist-with-awgn.gz") and path.exists("mnist-with-motion-blur.gz") and path.exists("mnist-with-reduced-contrast-and-awgn.gz"):
        print("Unzipping N-MNIST dataset files")
        opened_tar = tarfile.open("mnist-with-awgn.gz")
        opened_tar.extractall()

        opened_tar = tarfile.open("mnist-with-motion-blur.gz")
        opened_tar.extractall()

        opened_tar = tarfile.open("mnist-with-reduced-contrast-and-awgn.gz")
        opened_tar.extractall()

        print("Done unzipping N-MNIST dataset files")
    else:
        print("Zipped N-MNIST dataset files do not exist, please download them.")


def load(noise_type):  
    if noise_type == "gn":   
        data = loadmat("mnist-with-awgn.mat") # gausian noise

    if noise_type == "blur": 
        data = loadmat("mnist-with-motion-blur.mat")

    if noise_type == "contrast":
        data = loadmat("mnist-with-reduced-contrast-and-awgn.mat")

    train_X = data["train_x"]
    train_Y = data["train_y"]
    test_X = data["test_x"]
    test_Y = data["test_y"]

    train_X = train_X.astype('float32')
    train_X = train_X.reshape((train_X.shape[0], 1, 28, 28)) 
    train_X = train_X / 255 # normalise data 
    test_X = test_X.astype('float32')
    test_X = test_X.reshape((test_X.shape[0], 1, 28, 28))
    test_X = test_X / 255 # normalise data 

    train_Y_temp = []
    test_Y_temp = []

    for j in range(len(train_Y)):
        train_Y_temp.append(np.where(train_Y[j] == 1)[0][0])

    for j in range(len(test_Y)):
        test_Y_temp.append(np.where(test_Y[j] == 1)[0][0])

    train_Y = np.array(train_Y_temp, dtype=np.uint8) 
    test_Y = np.array(test_Y_temp, dtype=np.uint8) 

    dataset = {}
    dataset["training_images"] = train_X
    dataset["training_labels"] = train_Y
    dataset["testing_images"]  = test_X
    dataset["testing_labels"]  = test_Y


    return dataset["training_images"], dataset["training_labels"], \
           dataset["testing_images"], dataset["testing_labels"],\
           dataset


def init():
    download_n_mnist()


if __name__ == '__main__':
    init()


