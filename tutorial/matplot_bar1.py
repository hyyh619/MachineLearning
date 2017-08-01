#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:28:44 2017

@author: yinghuang
"""

import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y1, facecolor='#ff9999', edgecolor='white')
        
for x,y in zip(X, Y1) :
    # ha: horizontal alignment va: vertical alignment
    plt.text(x+0.1, y+0.01, '%.2f' % y, ha='center', va='bottom')
    
for x,y in zip(X, -Y2) :
    # ha: horizontal alignment va: vertical alignment
    plt.text(x+0.1, y-0.01, '%.2f' % y, ha='center', va='top')    

plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()