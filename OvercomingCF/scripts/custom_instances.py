import numpy as np
import mnist

filename = [
    ["n_mnist_awgn", "mnist-with-awgn.gz"],
    ["n_mnist_motion", "mnist-with-motion-blur.gz"],
    ["n_mnist_contrast", "mnist-with-reduced-contrast-and-awgn.gz"],
]


# def remove_not_working_mirrors():
#   if not hasattr(MNIST, 'mirrors'):
#     return
  
#   new_mirrors = [x for x in MNIST.mirrors if "yann.lecun.com" not in x]
#   if len(new_mirrors) == 0:
#     return

#   MNIST.mirrors = new_mirrors


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
        # with gzip.open('mnist-with-awgn.gz', 'rb') as f_in:
        #     with open('mnist-with-awgn.mat', 'wb') as f_out:
        #         shutil.copyfileobj(f_in, f_out)

        # with gzip.open('mnist-with-motion-blur.gz', 'rb') as f_in:
        #     with open('mnist-with-motion-blur.mat', 'wb') as f_out:
        #         shutil.copyfileobj(f_in, f_out)

        # with gzip.open('mnist-with-reduced-contrast-and-awgn.gz', 'rb') as f_in:
        #     with open('mnist-with-reduced-contrast-and-awgn.mat', 'wb') as f_out:
        #         shutil.copyfileobj(f_in, f_out)

        # with ZipFile('mnist-with-awgn.gz', 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            # zipObj.extractall()

        # with ZipFile('mnist-with-motion-blur.gz', 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            # zipObj.extractall()

        # with ZipFile('mnist-with-reduced-contrast-and-awgn.gz', 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            # zipObj.extractall()

        print("Done unzipping N-MNIST dataset files")
    else:
        print("Zipped N-MNIST dataset files do not exist, please download them.")

# STOPPED  AT i FINISHED THE CODE FOR DONWLOADING NOW I NEED TO DO THE POSTPROCESSEING.

def load():     
    mnist_gn = loadmat("mnist-with-awgn.mat")
    mnist_motion = loadmat("mnist-with-motion-blur.mat")
    mnist_contrast = loadmat("mnist-with-reduced-contrast-and-awgn.mat")

# def load():
#     with open("n_mnist.pkl", 'rb') as f:
#         n_mnist = pickle.load(f)
#     return n_mnist["images"], n_mnist["labels"]


# def save_mnist():
#     mnist = {}
#     for name in filename[:2]:
#         with gzip.open(name[1], 'rb') as f:
#             tmp = np.frombuffer(f.read(), np.uint8, offset=16)
#             mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255
#     for name in filename[-2:]:
#         with gzip.open(name[1], 'rb') as f:
#             mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
#     with open("mnist.pkl", 'wb') as f:
#         pickle.dump(mnist, f)
#     print("Save complete.")

    # remove_not_working_mirrors()
    # if not hasattr(N_MNIST, 'mirrors'):
    #   base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    # else:
    #   base_url = MNIST.mirrors[0]
    # for name in filename:
    #     print("Downloading " + name[1] + "...")
    #     download_file(base_url + name[1], name[1])
    # print("Download complete.")


def save_n_mnist():
    n_mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            n_mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            n_mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("n_mnist.pkl", 'wb') as f:
        pickle.dump(n_mnist, f)
    print("Save complete.")


def init():
    download_n_mnist()
    # # Check if already downloaded:
    # if path.exists("n_mnist.pkl"):
    #     print('Files already downloaded!')
    # else:  # Download Dataset
    #     download_n_mnist()
    #     save_n_mnist()

    # MNIST(path.join('data', 'mnist'), download=True)

# def load():
#     with open("n_mnist.pkl", 'rb') as f:
#         n_mnist = pickle.load(f)
#     return n_mnist["images"], n_mnist["labels"]

# def load():
#     with open("n_mnist.pkl", 'rb') as f:
#         n_mnist = pickle.load(f)
#     return n_mnist["training_images"], n_mnist["training_labels"], \
#            n_mnist["test_images"], n_mnist["test_labels"]


if __name__ == '__main__':
    init()




# from scipy.io import loadmat
# mnist_gn = loadmat("mnist-with-awgn.mat")
# mnist_motion = loadmat("mnist-with-motion-blur.mat")
# mnist_contrast = loadmat("mnist-with-reduced-contrast-and-awgn.mat")
# print(mnist_motion)
# mnist_gn_train_X = mnist_gn["train_x"]
# mnist_gn_train_Y = mnist_gn["train_y"]
# mnist_motion_train_X = mnist_motion["train_x"]
# mnist_motion_train_Y = mnist_motion["train_y"]
# mnist_contrast_train_X = mnist_contrast["train_x"]
# mnist_contrast_train_Y = mnist_contrast["train_y"]
# print("mnist_motion_train_X = ", mnist_motion_train_X)
# print("mnist_motion_train_X.dtype = ",mnist_motion_train_X.dtype)
# print("mnist_motion_train_X.shape = ",mnist_motion_train_X.shape)
# print("mnist_motion_train_Y.dtype = ",mnist_motion_train_Y.dtype)
# print("mnist_motion_train_Y.shape = ",mnist_motion_train_Y.shape)
# mnist_gn_train_X = mnist_gn_train_X.astype('float32')
# mnist_gn_train_X = mnist_gn_train_X.reshape((mnist_motion_train_X.shape[0], 1, 28, 28))
# mnist_motion_train_X = mnist_motion_train_X.astype('float32')
# mnist_motion_train_X = mnist_motion_train_X.reshape((mnist_motion_train_X.shape[0], 1, 28, 28))
# mnist_contrast_train_X = mnist_contrast_train_X.astype('float32')
# mnist_contrast_train_X = mnist_contrast_train_X.reshape((mnist_motion_train_X.shape[0], 1, 28, 28))
# print("mnist_motion_train_X.dtype = ",mnist_motion_train_X.dtype)
# print("mnist_motion_train_X.shape = ",mnist_motion_train_X.shape)
# i = 0
# plt.imshow(mnist_gn_train_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
# plt.savefig('mnist_gn_train_X.png')

# plt.imshow(mnist_motion_train_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
# plt.savefig('mnist_motion_train_X.png')

# plt.imshow(mnist_contrast_train_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
# plt.savefig('mnist_contrast_train_X.png')
# mnist_gn_train_Y_temp = []
# mnist_motion_train_Y_temp = []
# mnist_contrast_train_Y_temp = []
# for j in range(len(mnist_gn_train_Y)):
#     mnist_gn_train_Y_temp.append(np.where(mnist_gn_train_Y[j] == 1)[0][0])
#     mnist_motion_train_Y_temp.append(np.where(mnist_motion_train_Y[j] == 1)[0][0])
#     mnist_contrast_train_Y_temp.append(np.where(mnist_contrast_train_Y[j] == 1)[0][0])
#     # print(mnist_gn_train_Y_temp)

#     # plt.imshow(mnist_gn_train_X[j].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     # plt.savefig('mnist_gn_train_X.png')
#     # input("press eneter")
# mnist_gn_train_Y = np.array(mnist_gn_train_Y_temp, dtype=np.uint8) 
# mnist_motion_train_Y = np.array(mnist_motion_train_Y_temp, dtype=np.uint8) 
# mnist_contrast_train_Y = np.array(mnist_contrast_train_Y_temp, dtype=np.uint8) 
# print("mnist_gn_train_Y[i] = ", mnist_gn_train_Y[i])
# print("mnist_motion_train_Y[i] = ", mnist_motion_train_Y[i])
# print("mnist_contrast_train_Y[i] = ", mnist_contrast_train_Y[i])

# print("mnist_gn_train_Y.dtype = ",mnist_gn_train_Y.dtype)
# print("mnist_gn_train_Y.shape = ",mnist_gn_train_Y.shape)

# print("Y_train.dtype = ",Y_train.dtype)
# print("Y_train.shape = ",Y_train.shape)

# print("X_train.dtype = ",X_train.dtype)
# print("X_train.shape = ",X_train.shape)
# print("Y_train[i] = ", Y_train[i])
# plt.imshow(X_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
# plt.savefig('X_train.png')