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

models.append(('LR'), LinearRegression())
models.append(('LASSO', Lasso())) 
models.append(('EN', ElasticNet())) 
models.append(('KNN', KNeighborsRegressor())) 
models.append(('CART', DecisionTreeRegressor())) 
models.append(('SVR', SVR()))

results = []
names = []

for name, model








































