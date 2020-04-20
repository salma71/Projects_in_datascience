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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

    # b) Load dataset


# 2. Summarize Data
    # a) Descriptive statistics 
    # b) Data visualizations
# 3. Prepare Data
    # a) Data Cleaning
    # b) Feature Selection 
    # c) Data Transforms
# 4. Evaluate Algorithms
    # a) Split-out validation dataset
    # b) Test options and evaluation metric 
    # c) Spot Check Algorithms
    # d) Compare Algorithms
# 5. Improve Accuracy 
    # a) Algorithm Tuning 
    # b) Ensembles
# 6. Finalize Model
    # a) Predictions on validation dataset
    # b) Create standalone model on entire training dataset 
    # c) Save model for later use

