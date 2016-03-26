# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:32:27 2016

@author: 
"""

import os
import glob
import pandas as pd
import sqlite3 as db
from pandas.io import sql

path = os.getcwd()
allfiles = glob.glob(os.path.join(path, "namesbystate", "*.TXT"))

frame = pd.DataFrame()
list_ = []
for file_ in allfiles:
    df = pd.read_csv(file_,index_col=None, header=None, 
                     names=['state', 'sex', 'year', 'name', 'occurence'])
    list_.append(df)
frame = pd.concat(list_)


frame.describe()

# 2.  What is the most popular name of all time? (Of either gender.)
#    state  year      name  occurence
#sex                                 
#F      WY  2014    Zyriah       8184
#M      WY  2014  Zyshonne      10023
frame.groupby(["name"])


connection = db.connect('namesByState.db')
frame.to_sql("namesByState", connection)


