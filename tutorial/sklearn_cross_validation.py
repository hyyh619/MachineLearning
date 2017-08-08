#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:17:35 2017

@author: yinghuang
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train, y_train)
#y_pred = knn.predict(X_test)
#print (knn.score(X_test, y_test))
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print (scores)
print (scores.mean())

k_range = range(1, 31)
k_scores = []
for k in k_range :
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error') # for regression
    # scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
    k_scores.append(loss.mean())
    
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()