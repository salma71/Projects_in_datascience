#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:09:00 2020

@author: salmaelshahawy
"""

# Python Project Template
# 1. Prepare Problem 
    # a) Load libraries 
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

    # b) Load dataset
    #the dataset is avaliable on the UCI Machine learning repository
url = 'https://raw.githubusercontent.com/salma71/Projects_in_datascience/master/step_by_step_iris/iris/Iris.csv'
dataset = pd.read_csv(url)
# 2. Summarize Data
"""
It is the time to look into the data from different ways to understand it
    
    
    - Breakdown of the data by class variables
"""
    # a) Descriptive statistics 
#- Dimensions of the dataset
# shape 
print(dataset.shape) #(150, 6)
# we have 150 observations(instances) discribing 6 features(attributes) 

#- Peek into the data itself
print(dataset.head(20)) # we should  see the 1st 20 observations of the dataset
#- Statistical summary of all attributes
print(dataset.describe())
## we can see here that all the numerical values have the same scale (Cm)
## and similar range  from 0 to ~ 8 Cm

#- Breakdown of the data by class variables
# I  will  use the aggregation functions to investigate  the number of instances
# belonges to each Species variable

print(dataset.groupby('Species').size())
"""
Species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
"""

    # b) Data visualizations

# the next step is to visualize the data so you can better understand it
# I will make a univariate plots to understand each attribute
# Multivaliate plots to understand the relationships between  attributes

## Univariate plot 

# boxplot is good tool to understand data
sns.boxplot(data=dataset.iloc[:, 1:5])
# here I deselect column 0, 5 because it  is the id and species
# we see that the spal width has some outliers, let's inspect the distribution
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.distplot(dataset["SepalLengthCm"], color = "skyblue", ax = axes[0,0])
sns.distplot(dataset["SepalWidthCm"], color = "olive", ax = axes[0,1])
sns.distplot(dataset["PetalLengthCm"], color = "gold",ax = axes[1,0])
sns.distplot(dataset["PetalWidthCm"], color = "teal",ax = axes[1,1])

#as we can see the Sepal length and width having a Guassian distribution. 

# now we can perform some multivariate plots, we can perform that using the matrix of relationship 

sns.pairplot(dataset.iloc[:, 1:5])
# As we can see there is a relationship between length and width of each species. 
# However, there are an obvious pattern that Petal  width and length have a strong relation


# 3. Prepare Data
    # a) Data Cleaning
    # b) Feature Selection 
X = dataset.iloc[:, 1: 5].values
y = dataset.iloc[:, 5].values
    # c) Data Transforms
# 4. Evaluate Algorithms
    # a) Split-out validation dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # b) Test options and evaluation metric 
    # c) Spot Check Algorithms
# now it is time to evaluate some  of the appropriate algorithms that can fit our  problem.
models = []
models.append(('Log-Reg', LogisticRegression()))
models.append(('LDA', lda())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))

# evaluate each model in turns

    # d) Compare Algorithms
results = []

names  = []

for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % ( name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure() 
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names) 
plt.show()


# make the prediction 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# 5. Improve Accuracy 
    # a) Algorithm Tuning 
    # b) Ensembles
# 6. Finalize Model
    # a) Predictions on validation dataset
    # b) Create standalone model on entire training dataset 
    # c) Save model for later use

