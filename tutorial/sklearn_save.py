#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:15:37 2017

@author: yinghuang
"""

from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# method1: use pickle
import pickle
# Save
with open('clf.pickle', 'wb') as f :
    pickle.dump(clf, f)
    
# Restore    
with open('clf.pickle', 'rb') as f :
    clf2 = pickle.load(f)
    print (clf2.predict(X[0:2]))
    
# method2: use joblib
from sklearn.externals import joblib
# Save
joblib.dump(clf, 'clf.pkl')
# Restore
clf3 = joblib.load('clf.pkl')
print (clf3.predict(X[0:2]))