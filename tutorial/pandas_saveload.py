#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:14:22 2017

@author: yinghuang
"""

import pandas as pd
import numpy as np

dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])

#df.to_csv('pd.csv')
df2 = pd.read_csv('pd.csv')
df2.to_pickle('pd.pickle')
print df2