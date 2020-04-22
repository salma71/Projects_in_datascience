#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:03:09 2020

@author: salmaelshahawy
"""


# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns


url = 'https://raw.githubusercontent.com/salma71/Projects_in_datascience/master/boston_house_prices/Boston.csv'

dataset = pd.read_csv(url)

print(dataset.shape)

print(dataset.dtypes)

set_option('precision', 1)
print(dataset.describe)

print(dataset.head(10))


set_option('precision', 2)
print(dataset.corr(method='pearson'))

# Unimodal visualization

# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1) 
plt.show()

# density
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
plt.show()

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
plt.show()


# Multimodal visualization

sns.pairplot(dataset)

# correlation matrix
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']

"""
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none') 
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
"""
sns.set(style = 'white')
corrd = dataset.corr()

# Generate a mask for the upper triangle
maskd = np.triu(np.ones_like(corrd, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrd, mask=maskd, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

## Split the dataset into train and test set
seed = 7
X = dataset.iloc[:, 1:14].values
y = dataset.iloc[:, 14].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=seed)

## we need to do feature scaling first before picking an algorithm
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

## Let's compare algorithms each one in turn 

## Create the List of Algorithms to Evaluate.models 
models = []

models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso())) 
models.append(('EN', ElasticNet())) 
models.append(('KNN', KNeighborsRegressor())) 
models.append(('CART', DecisionTreeRegressor())) 
models.append(('SVR', SVR()))

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=7, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)

"""
LR: -21.396935 (9.579436)
LASSO: -26.595761 (8.525059)
EN: -27.936655 (8.963998)
KNN: -19.513609 (9.135670)
CART: -24.932037 (10.215727)
SVR: -30.018945 (13.935143)

"""  
    
# KNN has both a tight distribution of error and has the lowest score.

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names)
plt.show()

## Tunning the algorithm 

"""
I used a grid search to try a set of different numbers of neighbors 
and see if I can improve the score. 
I tried odd k values from 1 to 21, an arbitrary range covering a known good value of 7. 
Each k value (n neighbors) is evaluated using 10-fold cross validation on 
the standardized copy of the training dataset.
"""

k_values = np.array([1, 3, 5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits = 10, random_state=seed)
grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = 'neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(X_train, y_train)

print('Best score: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

# print Output From Tuning the KNN Algorithm

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    mean_errors.append(mean)
    
"""
Best score: -18.109304 using {'n_neighbors': 3}
-20.169640 (14.986904) with: {'n_neighbors': 1}
-18.109304 (12.880861) with: {'n_neighbors': 3}
-20.063115 (12.138331) with: {'n_neighbors': 5}
-20.514297 (12.278136) with: {'n_neighbors': 7}
-20.319536 (11.554509) with: {'n_neighbors': 9}
-20.963145 (11.540907) with: {'n_neighbors': 11}
-21.099040 (11.870962) with: {'n_neighbors': 13}
-21.506843 (11.468311) with: {'n_neighbors': 15}
-22.739137 (11.499596) with: {'n_neighbors': 17}
-23.829011 (11.277558) with: {'n_neighbors': 19}
-24.320892 (11.849667) with: {'n_neighbors': 21}
"""
# best for k (n neighbors) is 3 providing a mean squared error of -18.109304 , the best so far

'''
## another method to choose the best k using elbow curve
from sklearn import neighbors
from math import sqrt
rmse_val = [] #to store rmse values for different k

for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
 '''
   
## Build the model using those results 

regressor = KNeighborsRegressor(n_neighbors = 3)
regressor.fit(X_train, y_train)

X_test = sc_X.fit_transform(X_test)
y_pred = regressor.predict(X_test)

print(mean_squared_error(y_test, y_pred))
# 33.755675381263615

## ensamble methods 

'''
Another way that we can improve the performance of algorithms 
on this problem is by using ensemble methods. 
In this section we will evaluate four different ensemble machine 
learning algorithms, two boosting and two bagging methods.

* Boosting Methods: 
    - AdaBoost (AB) and 
    - Gradient Boosting (GBM). 􏰀 
* Bagging Methods: 
    - Random Forests (RF) and 
    - Extra Trees (ET).
'''

# I will use same comparing method as last time

ensambles = []
ensambles.append(('AB', AdaBoostRegressor()))
ensambles.append(('GBM', GradientBoostingRegressor()))
ensambles.append(('RF', RandomForestRegressor()))
ensambles.append(('ET', ExtraTreesRegressor()))

results_ens = []
names_ens = []

for name, model in ensambles:
    kfold = KFold(n_splits = 7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold,scoring='neg_mean_squared_error')
    results_ens.append(cv_results)
    names_ens.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)

'''
AB: -14.075652 (5.874932)
GBM: -9.492096 (3.455731)
RF: -11.494699 (4.426762)
ET: -9.147649 (4.421966)
'''
'''
# We can see that we’re generally getting better scores than the linear and nonlinear algorithms 
# in previous results

ET seems to perform bettwe than GBM where it has the min MES. 
However, we can use the tuning to see if we can perform make the model perform better
'''

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison') 
ax = fig.add_subplot(111)
plt.boxplot(results_ens)
ax.set_xticklabels(names_ens)
plt.show()

## Parameter tuning 
param_et = dict(n_estimators= np.array([50,100,150,200,250,300,350,400]))
model_ET = ExtraTreesRegressor(random_state=seed)
kfold_ET = KFold(n_splits=10)
grid_et = GridSearchCV(estimator=model, param_grid=param_et)
grid_res_et = grid_et.fit(X_train, y_train)

print('Best params: %f using %s' % (grid_res_et.best_score_, grid_res_et.best_params_))
means_et = grid_res_et.cv_results_['mean_test_score']
stds_et = grid_res_et.cv_results_['std_test_score']
params_et = grid_res_et.cv_results_['params']

for mean, std, param in zip(means_et, stds_et, params_et):
    print('%f (%f) with: %r' % (mean, std, params))
    
'''
Best params: 0.894097 using {'n_estimators': 100}
'''
## Finalize the model

regressor_et = ExtraTreesRegressor(n_estimators=100, random_state=seed)
regressor_et.fit(X_train, y_train)

X_test = sc_X.fit_transform(X_test)

predection = regressor_et.predict(X_test)
print(mean_squared_error(y_test, predection)) 

#14.77384742156861

# as we can see the MSE is down by 50% ( from 33 to 14.7) when using the ensamble learning.
 























