# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:48:22 2019

@author: Vincent Henric & Antoine Lepeltier
"""
import utils
import numpy as np
import matplotlib.pyplot as plt

def learn(X, y, max_iter = 1000, eps=pow(10,-6)):
    # IRLS method
    n_iter=0
    w = np.array([[0,0,0]])
    X_T = np.transpose(X)
    while (n_iter < max_iter):
        n_iter+=1
        grad = X_T@np.transpose(y-utils.expit(w@X_T))
        hess = -X_T@(X*np.transpose(utils.expit(-w@X_T)*utils.expit(w@X_T)))
        inv_hess = np.linalg.inv(hess)
        w = w -np.transpose(inv_hess@grad)
        lambda_criter=np.transpose(grad)@inv_hess@grad
        if (lambda_criter*lambda_criter/2 < eps):
            break
    w = w.flatten()
    print("Number of iterations IRLS : ",n_iter)
    return w[0], w[1], w[2]

def proba_func(x, w1, w2, b):
    return utils.expit(x @ np.array([w1, w2, b]))

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
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend(loc="upper right", fontsize=16)
    if save:
        print("Save figure results folder.")
        plt.savefig('results/{}'.format(utils.sanitize(title)))
    plt.show()