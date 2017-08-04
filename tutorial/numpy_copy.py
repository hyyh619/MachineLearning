#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:47:43 2017

@author: yinghuang
"""

import numpy as np

a = np.arange(4)
b = a
d = b
print a
print b

a[0] = -1
print b
print d[0]
print (b is a)

d[1:3] = [22, 33]
print a
b = a.copy()
b[0] = -2
print b
print a