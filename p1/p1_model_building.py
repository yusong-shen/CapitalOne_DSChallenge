# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:31:14 2016

@author: yusong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random
from sklearn.linear_model import Ridge
from pandas.stats.api import ols

# reading data
train = pd.read_table("codetest_train.txt")
test = pd.read_table("codetest_test.txt")

# explore
train.isnull().sum().mean()
train.dtypes

# numerical variable
train_num = train.select_dtypes(include=[np.float])
test_num = test.select_dtypes(include=[np.float])

plt.hist(train_num["target"])
train_num.describe()
test_num.describe()

# convert the categorical variable to number
train_cat = train.select_dtypes(exclude=[np.float])
test_cat = test.select_dtypes(exclude=[np.float])

train_cat.describe()
test_cat.describe()

# fill all the missing value by its column mean for train_num, test_num
train_num = train_num.fillna(train_num.mean())
test_num = test_num.fillna(test_num.mean())


# fill all the missing category value each column by randomly sample one
train_cat = train_cat.apply( lambda x : x.fillna(random.choice(x.value_counts().index)) ) 
test_cat = test_cat.apply( lambda x : x.fillna(random.choice(x.value_counts().index)) ) 


# split the training data to training set and validation set
y = train["target"]
x_train_num = train_num.drop(["target"], axis=1)
x_train_cat = train_cat
x_train = pd.concat([x_train_num, x_train_cat], axis=1)


x_train_all, x_test, y_train_all, y_test = train_test_split(
    x_train, y, test_size = 0.2, random_state = 42)
    
    
    

# fit ordinary linear regression    
res = ols(y = y_train_all, x = x_train_all)    
res

# use only numerical data
x_train_all, x_test, y_train_all, y_test = train_test_split(
    x_train_num, y, test_size = 0.2, random_state = 42)
        
# fit regression model to data
ridge = Ridge(alpha = 1.0)
ridge.fit(x_train_all, y_train_all)

ridge.score(x_test, y_test)



