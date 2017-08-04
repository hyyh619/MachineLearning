#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:15:58 2017

@author: yinghuang
"""

import numpy as np

A = np.arange(3, 15)
print A
print A[3]
print A[0]
print "***************"

A = A.reshape((3, 4))
print A
print A[2]    # row 2
print A[2, :] # row 2
print A[2][3]
print A[2, 3]
print A[:, 1] # column 1
print A[0]

# iterate all rows of matrix A.
for row in A:
    print row
    
# iterate all columns of matrix A.
for column in A.T:
    print column