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

url = 'https://raw.githubusercontent.com/salma71/Projects_in_datascience/master/boston_house_prices/Boston.csv'

dataset = pd.read_csv(url)

print(dataset.shape)

print(dataset.dtypes)

set_option('precision', 1)
print(dataset.describe)

print(dataset.head(10))


set_option('precision', 2)
print(dataset.corr(method='pearson'))

























































