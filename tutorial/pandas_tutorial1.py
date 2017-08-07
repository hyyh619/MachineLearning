#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:11:59 2017

@author: yinghuang
"""

import pandas as pd
import numpy as np

s = pd.Series([1,3,6,np.nan,44,1]) # 1 dimensional matrix
print s

dates = pd.date_range('20160101', periods=6)
print dates

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print df
#df1 = pd.DataFrame(np.random.randn(6, 4))
#print df1
#print df1.dtypes
#print df1.index
#print df1.columns
#print df.index
#print df.columns
#print df.values
#print df.describe()
#print df.T
#print df.sort_index(axis=1, ascending=False)
#print df.sort_index(axis=0, ascending=False)
#print df.sort_values(by='a')

print (df['a'])
print (df.a)
print (df[0:3])
print (df['2016-01-01':'2016-01-04'])

# select by label: loc
print (df.loc['2016-01-02'])
print (df.loc[:,['b', 'c']])
print (df.loc['2016-01-02', ['b', 'c']])

print (df.iloc[1, 1])
print (df.iloc[[1, 3, 5], 1:3])

print (df[df.a > 0.0])