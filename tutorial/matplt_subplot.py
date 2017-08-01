#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:20:44 2017

@author: yinghuang
"""

import matplotlib.pyplot as plt

plt.figure()

# 将整个figure分成2行，2列
plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 2, 2)
plt.plot([0, 1], [0, 2])

plt.subplot(2, 2, 3)
plt.plot([0, 1], [0, 3])

plt.subplot(2, 2, 4)
plt.plot([0, 1], [0, 4])

plt.figure()

# 将整个figure分成2行
# 第一行只有一列，第二行有3列
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 3, 4)
plt.plot([0, 1], [0, 2])

plt.subplot(2, 3, 5)
plt.plot([0, 1], [0, 3])

plt.subplot(2, 3, 6)
plt.plot([0, 1], [0, 4])

plt.show()