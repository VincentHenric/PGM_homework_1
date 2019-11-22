# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:14:09 2019

@author: Vincent Henric & Antoine Lepeltier
"""
import numpy as np
import pandas as pd
import re

def import_data(filename):
    train = pd.read_csv("data/{}".format(filename), delim_whitespace=True, header = None)
    return np.array(train.iloc[:,:2]), np.array(train.iloc[:,2])


def add_intercept(X):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return X

def expit(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))

def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum()/len(y_true)

def misclassification_rate(y_pred, y_true):
    # in percentage
    return round(100 * (1 - accuracy(y_pred, y_true)), 2)

def sanitize(s):
    """
    remove all non alphanumeric from a string
    """
    return re.sub(r'\W+', '', s)
#' '.join(s.replace(';', ' ').split()).replace(' ','-')