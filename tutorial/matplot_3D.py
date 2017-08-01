#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:01:29 2017

@author: yinghuang
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

# X, Y value
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)

# Add 3D surface, rstride: row stride cstride: col stride
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

# Add contour, zdir: z direction
ax.contourf(X, Y, Z, zdir='x', offset=-4, cmap='rainbow')
ax.set_zlim(-2, 2)

#plt.xticks(())
#plt.yticks(())
plt.show()