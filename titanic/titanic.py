#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:59:25 2020

@author: salmaelshahawy
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train = pd.read_csv('https://raw.githubusercontent.com/salma71/titanic_kaggle_compt/master/train.csv', sep=',')
test = pd.read_csv('https://raw.githubusercontent.com/salma71/titanic_kaggle_compt/master/test.csv', sep = ',')


# EDA part

print(train.info())
summary_stat = train.describe()


## from summary statistics we can see that we have some issues:
    ## 1. Survived is int not an object
    ## 2. Parch, Sibsp, Fare, and Age has a min of zero - means missing values.
    ## 3. Cabin and Embarked categorical variables, have missing values
# drop cabin column 

from pandas import set_option

set_option('display.width',  150)
correlations = train.corr(method= 'pearson')
print(correlations)

train.plot(kind = 'box', subplots = True, layout = (3,3), sharex=False, sharey = False)
plt.show()

# plot the correlation matrix 
names = ['Age', 'Fare', 'Parch', 'PassengerId', 'Pclass', 'SibSp', 'Survived']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1) 
fig.colorbar(cax)
ticks = np.arange(0,9,1) 
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names) 
ax.set_yticklabels(names)
plt.show()

import seaborn as sns
sns.set()
cols = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Survived']
sns.pairplot(train[cols], size = 2.5)

#from pandas.plotting import scatter_matrix
#scatter_matrix(dataset)
plt.show()

sns.countplot(train['Survived'])
plt.show()

sns.countplot(train['Pclass'])
plt.show()

sns.barplot(x='Pclass', y='Survived', data=train)
plt.show()


sns.distplot(train['Age'])
plt.title('Age distribution')
plt.show()

sns.boxplot(x='Survived', y='Age', data=train)
plt.show()

fig, axes = plt.subplots(2,2)
sns.boxenplot(data=train, x = 'Survived', y = 'Age', hue = 'Sex')

sns.distplot(train['Fare'])
plt.show()

sns.boxplot(x='Survived', y='Fare', data=train)
plt.show()


dataset = pd.concat([train, test])
dataset['Title'] = dataset['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
dataset['Title'].value_counts()


Common_Title = ['Mr', 'Miss', 'Mrs', 'Master']
dataset['Title'].replace(['Ms', 'Mlle', 'Mme'], 'Miss', inplace=True)
dataset['Title'].replace(['Lady'], 'Mrs', inplace=True)
dataset['Title'].replace(['Sir', 'Rev'], 'Mr', inplace=True)
dataset['Title'][~dataset['Title'].isin(Common_Title)] = 'Others'
# investigate howmany missing point

train_df = dataset[:len(train)]
test_df = dataset[len(train):]
sns.boxenplot(data=train_df, x = 'Title', y = 'Age', hue = 'Survived')


total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


## impute the age 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
dataset['Age'] = imputer.fit_transform(dataset[['Age']]).ravel()

dataset['Fare'] = imputer.fit_transform(dataset[['Fare']]).ravel()

dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode().iloc[0])


## split the dataset into inputs and putputs
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.319, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



