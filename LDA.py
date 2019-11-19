# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:10:52 2019

@author: Vincent Henric
"""
import utils
import numpy as np
import matplotlib.pyplot as plt

def mle(X, y):
    n = len(y)
    mu1 = (X[y==1].sum(axis=0))/(y.sum())
    mu0 = (X[y!=1].sum(axis=0))/(n - y.sum())
    sigma = ((X[y!=1]-mu0).T @ (X[y!=1]-mu0))/n + ((X[y==1]-mu1).T @ (X[y==1]-mu1))/n
    pi = y.sum()/n
    return mu0, mu1, sigma, pi

def linear_coef(mu0, mu1, sigma, pi):
    sigma_inv = np.linalg.inv(sigma)
    w1, w2 = sigma_inv @ (mu1-mu0)
    b = utils.logit(pi) - 1/2 * (mu1+mu0).T @ sigma_inv @ (mu1-mu0)
    return w1, w2, b

def proba_func(x, mu0, mu1, sigma, pi):
    sigma_inv = np.linalg.inv(sigma)
    exponent = -1/2 * (2 * (mu1 - mu0).T @ sigma_inv @ x + (mu1 + mu0).T @ sigma_inv @ (mu0 - mu1))
    proba_ratio = (1-pi)/pi
    return 1/(1+proba_ratio*np.exp(exponent))

def log_odds(x, mu0, mu1, sigma, pi):
    sigma_inv = np.linalg.inv(sigma)
    return utils.logit(pi) + (mu1 - mu0).T @ sigma_inv @ x -1/2 * (mu1 + mu0).T @ sigma_inv @ (mu1 - mu0)

def predict(x, mu0, mu1, sigma, pi):
    return log_odds(x, mu0, mu1, sigma, pi)>=0

def plot_boundary(X, y, coefs, title='', save = False):
    x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
    y_min, y_max = np.min(X[:,1]), np.max(X[:,1])
    offset = 1
    boundary_x = np.linspace(x_min-offset, x_max+offset, 1000)
    boundary_y = [(utils.logit(1/2)-coefs['b']-coefs['w1']*x)/coefs['w2'] for x in boundary_x]

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