#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:20:41 2017

@author: yinghuang
"""

import pandas as pd
import numpy as np

dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])

print df
df.iloc[2, 2] = 101
print df
df.loc['2016-01-01', 'a'] = 99
print df

df.a[df.a<0] = -99
print df

#df['f'] = np.nan
#print df

df['e'] = pd.Series([1, 2, 3, 4, 5, 6], index = pd.date_range('20160101', periods=6))
print df

df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print df
print (df.dropna(axis=0, how='any'))
print (df.dropna(axis=1, how='any'))

df['f'] = np.nan
print (df.dropna(axis=0, how='all'))
print (df.dropna(axis=1, how='all'))

print (df.fillna(value=0))
print (df.isnull())
print (np.any(df.isnull()) == True)