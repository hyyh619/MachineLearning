#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:22:55 2017

@author: yinghuang
"""

import pandas as pd
import numpy as np

dates = pd.date_range('20160101', periods=6)
df1 = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
df2 = pd.read_csv('pd.csv')
df3 = pd.read_csv('pd.csv')

# concatenating
res = pd.concat([df2, df3], axis=0)
print res
res = pd.concat([df2, df3], axis=0, ignore_index=True)
print res

# join, ['inner', 'outer']
res = pd.concat([df1, df2])
print res
res = pd.concat([df1, df2], join='outer')
print res
res = pd.concat([df1, df2], join='inner', ignore_index=True)
print res

# join_axes
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
print res

# append
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res = df1.append(s1, ignore_index=True)
print res