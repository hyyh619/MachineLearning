#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:26:45 2017

@author: yinghuang
"""

import numpy as np

A = np.array([1, 1, 1])
B = np.array([2, 2, 2])
print A
print B

C = np.vstack((A, B)) # vertical stack
print C
print C.shape

D = np.hstack((A, B)) # horizontal stack
print D
print D.shape

E = np.array([[1], [1], [1]])
print E.T
print E
print A[np.newaxis, :]
print A[:, np.newaxis]

AT = A[:, np.newaxis]
BT = B[:, np.newaxis]
F = np.hstack((AT, BT))
print F

G = np.concatenate((AT, BT, BT, AT), axis=0)
print G
G = np.concatenate((AT, BT, BT, AT), axis=1)
print G