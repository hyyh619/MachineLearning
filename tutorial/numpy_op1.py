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