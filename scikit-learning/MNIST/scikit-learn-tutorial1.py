# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 14:45:01 2017

@author: hyyh6
"""

from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()

classifier = svm.SVC(gamma=0.001, C=100.)

# Train
classifier.fit(digits.data[:-1], digits.target[:-1])

# Predict
a = classifier.predict(digits.data[-1])
print digits.data[-1]
print a