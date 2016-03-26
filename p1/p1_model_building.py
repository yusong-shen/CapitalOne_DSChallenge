# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:31:14 2016

@author: 
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
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV


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
    
# the real test set
x_final_test  =  pd.concat([test_num, test_cat], axis=1)
    
###############################################################################

### fit ordinary linear regression    
# pandas version
ols_mod = ols(y = y_train_all, x = x_train_all)    

# sklearn version
ols_mod2 = linear_model.LinearRegression()
ols_mod2.fit(x_train_all, y_train_all)        

# training mse
y_ols_train = ols_mod2.predict(x_train_all)
ols_train_mse = mean_squared_error(y_train_all, y_ols_train)

# test mse
y_ols_pred = ols_mod2.predict(x_test)
ols_test_mse = mean_squared_error(y_test, y_ols_pred)

        
### fit regression model to data
## ridge regression
ridge = Ridge(alpha = 1.0)
ridge.fit(x_train_all, y_train_all)

# training mse
y_ridge_train = ridge.predict(x_train_all)
ridge_train_mse = mean_squared_error(y_train_all, y_ridge_train)

# test mse
y_ridge_pred = ridge.predict(x_test)
ridge_test_mse = mean_squared_error(y_test, y_ridge_pred)

## lasso regression
lasso = Lasso(alpha = 1.0)
lasso.fit(x_train_all, y_train_all)

# training mse
y_lasso_train = lasso.predict(x_train_all)
lasso_train_mse = mean_squared_error(y_train_all, y_lasso_train)

# test mse
y_lasso_pred = lasso.predict(x_test)
lasso_test_mse = mean_squared_error(y_test, y_lasso_pred)

## elastic net
elastic = ElasticNet(alpha = 1.0, l1_ratio=0.5)
elastic.fit(x_train_all, y_train_all)

# training mse
y_elastic_train = elastic.predict(x_train_all)
elastic_train_mse = mean_squared_error(y_train_all, y_elastic_train)

# test mse
y_elastic_pred = elastic.predict(x_test)
elastic_test_mse = mean_squared_error(y_test, y_lasso_pred)

#### perform cross validation to select model
## ridge regression with CV
ridgeCV = RidgeCV(alphas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
ridgeCV.fit(x_train_all, y_train_all)

ridgeCV.alpha_
# training mse
y_ridgeCV_train = ridgeCV.predict(x_train_all)
ridgeCV_train_mse = mean_squared_error(y_train_all, y_ridgeCV_train)

# test mse
y_ridgeCV_pred = ridgeCV.predict(x_test)
ridgeCV_test_mse = mean_squared_error(y_test, y_ridgeCV_pred)

plt.scatter([i for i in range(y_test.size)], y_test - y_ridgeCV_pred)
    
###############################################################################
# compare result
methods = ["ols", "lasso", "elastic net", "ridge", "ridgeCV"]

train_mse = [ols_train_mse, lasso_train_mse, elastic_train_mse, 
             ridge_train_mse, ridgeCV_train_mse]    
test_mse = [ols_test_mse, lasso_test_mse, elastic_test_mse, 
             ridge_test_mse, ridgeCV_test_mse] 
             

plt.figure()
plt.plot(train_mse, label="train mse")  
plt.plot(test_mse, label="test mse")
plt.xticks(range(len(train_mse)), methods)
plt.title("mean square errors")
plt.show()           

###############################################################################
# output the result

ridgeCV_final_pred = ridgeCV.predict(x_final_test)
df_ridgeCV_final_pred = pd.DataFrame(ridgeCV_final_pred)
df_ridgeCV_final_pred.columns = [ "target"]

df_ridgeCV_final_pred.to_csv("ridgeCV_final_pred.csv")