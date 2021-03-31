# -*- coding: utf-8 -*-
"""Example of using AutoEncoder for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function


from pyod.pyod.models.auto_encoder import AutoEncoder
import numpy as np
from keras.models import load_model
from joblib import dump, load
import pandas as pd

X_train=pd.read_csv('X_normal.csv', sep=',',header=None)
X_train = np.array(X_train)
X_train = X_train[1:,1:]

X_test=pd.read_csv('X_outliers.csv', sep=',',header=None)
X_test = np.array(X_test)
X_test = X_test[1:,1:]

new_model = load('clf.joblib')

X_retrain = np.append(X_train[1:20000], X_test[1:3000] , axis=0)


#y_train_pred= new_model.predict(X_train)
#new_model.epochs = 30
#new_model.contamination = 0.1
#new_model.fit(X_retrain)

clf_name = 'AutoEncoder'
clf = AutoEncoder(epochs=30, contamination=0.1,random_state = 50,hidden_neurons = [32, 16, 16, 32])
clf.fit(X_retrain)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test) 

test_num_outlier_1 = 0
for lable in y_test_pred:
    if lable > 0:
        test_num_outlier_1 = test_num_outlier_1 + 1

train_num_outlier_1 = 0
for lable in y_train_pred:
    if lable > 0:
        train_num_outlier_1 = train_num_outlier_1 + 1

