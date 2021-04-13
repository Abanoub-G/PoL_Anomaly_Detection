# -*- coding: utf-8 -*-
"""Example of using AutoEncoder for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function



from pyod.models.auto_encoder import AutoEncoder
from joblib import dump, load
import numpy as np
import pandas as pd

contamination = 0.1  # percentage of outliers

#X_train = []
#with open('X_nomral') as my_file:
#    for line in my_file:
#        X_train.append(line)
#        


X_train=pd.read_csv('X_normal.csv', sep=',',header=None)
X_train = np.array(X_train)
X_train = X_train[1:,1:]



# train AutoEncoder detector
clf_name = 'AutoEncoder'
clf = AutoEncoder(epochs=30, contamination=contamination,random_state = 50,hidden_neurons = [32, 16, 16, 32])
clf.fit(X_train)


Y_train_label = clf.predict(X_train)

num_outlier = 0
for lable in Y_train_label:
    if lable > 0:
        num_outlier = num_outlier + 1

X_test=pd.read_csv('X_outliers.csv', sep=',',header=None)
X_test = np.array(X_test)
X_test = X_test[1:,1:]

Y_test_label = clf.predict(X_test)

test_num_outlier = 0
for lable in Y_test_label:
    if lable > 0:
        test_num_outlier = test_num_outlier + 1

dump(clf, 'clf.joblib')
