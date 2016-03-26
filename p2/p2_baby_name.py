# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:32:27 2016

@author: 
"""

import os
import glob
import pandas as pd
import numpy as np


path = os.getcwd()
allfiles = glob.glob(os.path.join(path, "namesbystate", "*.TXT"))

frame = pd.DataFrame()
list_ = []
for file_ in allfiles:
    df = pd.read_csv(file_,index_col=None, header=None, 
                     names=['state', 'sex', 'year', 'name', 'occurence'])
    list_.append(df)
frame = pd.concat(list_)


