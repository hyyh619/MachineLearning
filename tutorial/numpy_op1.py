#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:28:29 2017

@author: hy
"""

import numpy as np

a = np.array([10, 20, 30, 40])
b = np.arange(4)

c = a * b
print c

d = np.tan(b)
print d
print(b<3)

a = np.array([[1, 1],
              [0, 1]])
b = np.arange(4).reshape((2, 2))
c = a * b
print c
c = np.dot(a, b)
print c

A = np.arange(14, 2, -1).reshape((3, 4))
print(A)
print np.argmin(A)
print np.argmax(A)
print np.mean(A)
print A.mean()
print np.average(A)
print np.median(A)
print np.cumsum(A)
print np.diff(A)
print np.nonzero(A)
print np.sort(A)
print np.transpose(A)
print A.T
print A.T.dot(A)
print np.clip(A, 5, 11)
print np.mean(A, axis=0) # column
print np.mean(A, axis=1) # row