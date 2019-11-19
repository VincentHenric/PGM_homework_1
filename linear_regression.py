# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:15:30 2019

@author: Vincent Henric
"""
import utils
import numpy as np
import matplotlib.pyplot as plt

def mle(X, y):
    w0, w1, intercept = np.linalg.inv(X.T @ X) @ X.T @ y
    return  w0, w1, intercept

def proba_func(x, w1, w2, b):
    return x @ np.array([w1, w2, b])

def predict(x, w1, w2, b):
    return proba_func(x, w1, w2, b) >= 0.5

def plot_boundary(X, y, coefs, title='', save = False):
    x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
    y_min, y_max = np.min(X[:,1]), np.max(X[:,1])
    offset = 1
    boundary_x = np.linspace(x_min-offset, x_max+offset, 1000)
    boundary_y = [(0.5 - coefs['b'] - coefs['w1']*x)/coefs['w2'] for x in boundary_x]

    plt.figure(figsize = (9, 9))
    plt.plot(boundary_x, boundary_y, c='g', label='Decision boundary')
    plt.scatter(X[y == 0,0], X[y == 0,1], label = 'class 0')
    plt.scatter(X[y == 1,0], X[y == 1,1], label = 'class 1')

    plt.xlim(x_min - offset, x_max + offset)
    plt.ylim(y_min - offset, y_max + offset)
    plt.title(title)
    plt.legend(loc="upper right", fontsize=16)
    if save:
        print("Save figure results folder.")
        plt.savefig('results/{}'.format(title))
    plt.show()