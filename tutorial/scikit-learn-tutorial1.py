#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:52:18 2017

@author: yinghuang
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

print (iris_X[:2, :])
print (iris_Y[:2])

# split and shuffle the data.
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)  # training
print (knn.predict(X_test))
print (Y_test)