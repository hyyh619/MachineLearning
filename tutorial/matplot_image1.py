#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:54:15 2017

@author: yinghuang
"""

import matplotlib.pyplot as plt
import numpy as np

a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

# 在该网站上面查看interpolation的取值范围
# http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')

# Add color bar
plt.colorbar(shrink=0.8)

plt.xticks(())
plt.yticks(())
plt.show()