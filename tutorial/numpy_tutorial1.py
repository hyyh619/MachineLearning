#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:01:26 2017

@author: hy
"""

import numpy as np

arr1 = np.array([[1, 2, 3],
                [4, 5, 6]],
                dtype=np.float)
print arr1
print("Size:", arr1.size)
print("Shape:", arr1.shape)
print("Dimension:", arr1.ndim)
print("Type:", arr1.dtype)

arr2 = np.zeros((3, 4))
print arr2

arr3 = np.arange(10, 20, 2)
print arr3

arr4 = np.arange(12).reshape((3, 4))
print arr4

arr5 = np.linspace(1, 11, 12).reshape((4, 3))
print arr5