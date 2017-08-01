#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:43:44 2017

@author: yinghuang
"""

import matplotlib.pyplot as plt
import numpy as np

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X) # for color value

plt.scatter(X, Y, s=75, c=T, alpha=0.5)
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))
# 取消x坐标plt.xticks(())

plt.show()