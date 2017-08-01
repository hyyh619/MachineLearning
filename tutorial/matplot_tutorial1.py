#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:04:15 2017

@author: yinghuang
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = x**2 + 1
y2 = 2*x+1

figure1 = plt.figure()
plt.plot(x, y2, lw=5)
""" 添加标注 """
x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0, y0, s=50, color='k')
plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)

# method1: 添加annotation
plt.annotate(r"$2x+1=3$", xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.2'))
# method2:
plt.text(-3.7, 3, r"$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$",
         fontdict={'size':16, 'color':'r'})


# 调整坐标轴位置
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# 设置label
for label in ax.get_xticklabels() + ax.get_yticklabels() :
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='black', edgecolor='None', alpha=0.5))

figure2 = plt.figure(num=5, figsize=(8, 5))
""" 如果想传到handles中，必须在l1后面加, """
l1, = plt.plot(x, y1, color='red', linewidth=2.0, linestyle='--', label='up')
l2, = plt.plot(x, y2, label='down')
""" 打印图例 """
plt.legend(handles=[l1, l2],labels=['a', 'b'],loc='best')
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel("x")
plt.ylabel("y")

new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3],
           ['really bad', 'bad', 'normal', 'good', 'really good'])

# 调整坐标轴位置
# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

plt.show()