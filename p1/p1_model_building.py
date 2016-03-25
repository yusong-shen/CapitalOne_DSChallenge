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
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import Lasso

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

# categorival variable
train_cat = train.select_dtypes(exclude=[np.float])
test_cat = test.select_dtypes(exclude=[np.float])
for col in train_cat :
    train_cat[col] = train_cat[col].astype('category')
for col in test_cat :
    test_cat[col] = train_cat[col].astype('category')


train_cat.describe()
test_cat.describe()

# fill all the missing value by its column mean for train_num, test_num
train_num = train_num.fillna(train_num.mean())
test_num = test_num.fillna(test_num.mean())


# fill all the missing category value each column by randomly sample one
train_cat = train_cat.apply( lambda x : x.fillna(random.choice(x.value_counts().index)) ) 
test_cat = test_cat.apply( lambda x : x.fillna(random.choice(x.value_counts().index)) ) 

train_cat.describe()
test_cat.describe()

# convert the categorical variable to number

train_cat[train_cat.columns] = \
 train_cat[train_cat.columns].apply(lambda x: x.cat.codes)

test_cat[test_cat.columns] = \
 test_cat[test_cat.columns].apply(lambda x: x.cat.codes)

# split the training data to training set and validation set
y = train["target"]
x_train_num = train_num.drop(["target"], axis=1)
x_train_cat = train_cat
x_train = pd.concat([x_train_num, x_train_cat], axis=1)


x_train_all, x_test, y_train_all, y_test = train_test_split(
    x_train, y, test_size = 0.2, random_state = 42)
    
    
    
###############################################################################
# TODO : perform cross validation to select model
    
# fit ordinary linear regression    
# pandas version
ols_mod = ols(y = y_train_all, x = x_train_all)    

# sklearn version
ols_mod2 = linear_model.LinearRegression()
ols_mod2.fit(x_train_all, y_train_all)        

# training mse
y_ols_train = ols_mod2.predict(x_train_all)
mean_squared_error(y_train_all, y_ols_train)

# test mse
y_ols_pred = ols_mod2.predict(x_test)
mean_squared_error(y_test, y_ols_pred)

        
## fit regression model to data
# ridge regression
ridge = Ridge(alpha = 1.0)
ridge.fit(x_train_all, y_train_all)

# training mse
y_ridge_train = ridge.predict(x_train_all)
mean_squared_error(y_train_all, y_ridge_train)

# test mse
y_ridge_pred = ridge.predict(x_test)
mean_squared_error(y_test, y_ridge_pred)

# lasso regression
lasso = Lasso(alpha = 1.0)
lasso.fit(x_train_all, y_train_all)

# training mse
y_lasso_train = lasso.predict(x_train_all)
mean_squared_error(y_train_all, y_lasso_train)

# test mse
y_lasso_pred = lasso.predict(x_test)
mean_squared_error(y_test, y_lasso_pred)



