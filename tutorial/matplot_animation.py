#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:39:19 2017

@author: yinghuang
"""

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

fig, ax = plt.subplots()
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=100,
                              init_func=init,
                              interval=20,
                              blit=False)

plt.show()