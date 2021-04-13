# PoL_Anomaly_Detection

- Add details of how to setup opencv on a linux machine or just put a link on how to install that: https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/

- To work on the virtual environment execute the commande below: 
workon cv 


- Run video using this command: 
python3 main_video_detection.py --input videos/01.avi --output output/video01.avi --yolo yolo-coco

# Prerequisites
- Install flowiz:
sudo pip3 install flowiz -U

# Setup
- Get YOLO v3 wieghts folder from daknet's website: https://pjreddie.com/darknet/yolo/
OR 
Just execute this command in your terminal:
wget https://pjreddie.com/media/files/yolov3.weights

# Datasets:
Download datasets from here
- 
ShanghaiTech Campus dataset (Anomaly Detection): https://svip-lab.github.io/dataset/campus_dataset.html

- UCSD Anomaly Detection Dataset: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html 

- Avenue Dataset for Abnormal Event Detection: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

Then put them in folder ...

- MNIST dataset: http://yann.lecun.com/exdb/mnist/index.html

Download 1.training set images 2.training set labels 3.test set images 4.test set labels 

Exrtact these four files into the same directory as those two py file.



# Installing flownet2 

# Producing flownet image from two single images  [2]
- ./run-network.sh -n FlowNet2 -v data/0000000-imgL.png data/0000001-imgL.png flow.flo

# Repository hierarchy 


# How to setup opencv on a linux machine (refer to [1] for original tutorial copied here)
## Step #1: Install OpenCV dependencies on Ubuntu 18.04
- This guide is using Ubuntu 18.04 and Python 3.6
- We need to refresh/upgrade the pre-installed packages/libraries with the apt-get package manager: 
$ sudo apt-get update
$ sudo apt-get upgrade

- Install developer tools:
$ sudo apt-get install build-essential cmake unzip pkg-config

- Install OpenCV-specific prerequisites. OpenCV is an image processing/computer vision library and therefore it needs to be able to load standard image file formats such as JPEG, PNG, TIFF, etc. The following image I/O packages will allow OpenCV to work with image files:
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev

- Now try to install libjasper-dev:
$ sudo apt-get install libjasper-dev

If you receive an error about libjasper-dev being missing then follow the following instructions:
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install libjasper1 libjasper-dev

- Install video I/O packages as we often work with video. You’ll need the following packages so you can work with your camera stream and process video files:
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev

- OpenCV’s highgui module relies on the GTK library for GUI operations. The highgui module will allow you to create elementary GUIs which display images, handle kepresses/mouse clicks, and create sliders and trackbars. Advanced GUIs should be built with TK, Wx, or QT.  Install GTK:
$ sudo apt-get install libgtk-3-dev

- These two libraries which will optimize various OpenCV functions:
$ sudo apt-get install libatlas-base-dev gfortran

- Install Python 3 headers and libraries:
$ sudo apt-get install python3.6-dev

## Step #2: Download the official OpenCV source
- Download the official OpenCV release using wget:
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.4.zip

- Followed by the opencv_contrib module (Important: Both opencv and opencv_contrib versions must be identical):
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.4.zip

- Unzip the archives:
$ unzip opencv.zip
$ unzip opencv_contrib.zip

- Rename the directories:
$ mv opencv-3.4.4 opencv
$ mv opencv_contrib-3.4.4 opencv_contrib

## Step #3: Configure your Python 3 environment
- To install pip:
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py

- It is a best practice to use virtual environments. install  virtualenv and virtualenvwrapper:
$ sudo pip3 install virtualenv virtualenvwrapper
$ sudo rm -rf ~/get-pip.py ~/.cache/pip

- To finish the install update your .bashrc file.
$ echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
$ echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
$ echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
$ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc

- Source the .bashrc file:
$ source ~/.bashrc

- Creating a virtual environment to hold OpenCV and additional packages:
$ mkvirtualenv cv -p python3

- Verify that you are in the cv environment by using the workon command:
$ workon cv

- Install NumPy in your environment
$ pip install numpy


## Step #4: Configure and compile OpenCV for Ubuntu 18.04
- Ensure that we’re in the cv virtual environment:
$ workon cv

- Configure OpenCV with CMake
$ cd ~/opencv
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=ON ..

- Compiling OpenCV on Ubuntu 18.04
$ make -j4

- Installing and verifying OpenCV
$ sudo make install
$ sudo ldconfig

- To verify the install:
$ pkg-config --modversion opencv
You should get: 
3.4.4

## Step #5: Finish your Python+ OpenCV + Ubuntu 18.04 install

At this point, your Python 3 bindings for OpenCV should reside in the following folder:
$ ls /usr/local/python/cv2/python-3.6
cv2.cpython-36m-x86_64-linux-gnu.so

Rename them to simply cv2.so:
$ cd /usr/local/python/cv2/python-3.6
$ sudo mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.opencv3.4.4.so

- Last sub-step is to sym-link our OpenCV cv2.opencv3.4.4.so bindings into our cv virtual environment:
$ cd ~/.virtualenvs/cv/lib/python3.6/site-packages/
$ ln -s /usr/local/python/cv2/python-3.6/cv2.opencv3.4.4.so cv2.so


## Step #6: Testing your OpenCV 3 install on Ubuntu 18.04
$ cd ~
$ workon cv
$ pythons
Python 3.6.5 (default, Apr 1 2018, 05:46:30)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> cv2.__version__
'3.4.4'
>>> quit()


- You can safely delete the zips and directories in your home folder:

$ cd ~

$ rm opencv.zip opencv_contrib.zip

$ rm -rf opencv opencv_contrib



# How to setup Keras (Reference[6])
## Step #1: install python3
sudo apt-get install python3-pip python3-dev
## Step #2: install OpenBLAS
sudo apt-get install build-essential cmake git unzip pkg-config libopenblas-dev liblapack-dev
## Step #3: install NumPy, SciPy, matplotlib
sudo apt-get install python-numpy python-scipy python-matplotlib python-yaml
sudo pip3 install matplotlib
## Step #4: install HDF5
sudo apt-get install libhdf5-serial-dev python-h5py
## Step #5: install Graphviz, pydot-ng
sudo apt-get install graphviz
sudo pip3 install pydot-ng
## Step #6: install OpenCV
sudo apt-get install python-opencv
## Step #7: install TensorFlow (without GPU support)
sudo pip3 install tensorflow
## Step #8: install Keras (without GPU support)
sudo pip3 install keras

# How to update Keras and Tensorflow(Reference[7])
## Step #1: Upgrade Tensorboard
sudo pip3 install --user --upgrade tensorboard
## Step #2: Upgrade Tensorflow
sudo pip3 install --user --upgrade tensorflow-gpu
## Step #3: Downgrade Keras
sudo pip3 install keras==2.3.1
##S tep #4: Downgrade tensorflow-gpu
sudo pip3 install --user --upgrade tensorflow-gpu==1.14.0

# How to Install pydo(Reference[8])
## Step #1: Install right version of numbo
sudo pip3 install numba == 0.47
## Step #2: Install right version of llvmlite
sudo pip3 install llvmlite == 0.32.1
## Step #3: Install Dependency in [8]
   Python 2.7, 3.5, 3.6, or 3.7
   combo>=0.0.8
   joblib
   numpy>=1.13
   numba>=0.35
   pandas>=0.25
   scipy>=0.19.1
   scikit_learn>=0.19.1
   statsmodels
##S tep #4: Install pyod
sudo pip3 install pyod  




# References
[1] https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/)

[2] https://github.com/lmb-freiburg/flownet2-docker

# List of tutorials useful for autoencoders
[1] https://machinelearningmastery.com/autoencoder-for-classification/

[2] https://machinelearningmastery.com/lstm-autoencoders/

[3] https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

[4] https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/

[5] https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

[6] https://github.com/hsekia/learning-keras/wiki/How-to-install-Keras-to-Ubuntu-18.04

[7] https://stackoverflow.com/questions/62465620/error-keras-requires-tensorflow-2-2-or-higher

[8] https://github.com/yzhao062/pyod