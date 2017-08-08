#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:02:17 2017

@author: yinghuang
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_Y = loaded_data.target
#print (data_X[:2, :])
#print (data_Y[:2])

model = LinearRegression()
model.fit(data_X, data_Y)

print(model.predict(data_X[:4, :]))
print(data_Y[:4])

print (model.coef_)
print (model.intercept_)
print (model.get_params())
print (model.score(data_X, data_Y))

X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)
plt.scatter(X, y)
plt.show()