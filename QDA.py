# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:16:15 2019

@author: Vincent Henric
"""

import utils
import numpy as np
import matplotlib.pyplot as plt

def mle(X, y):
    n = len(y)
    mu1 = (X[y==1].sum(axis=0))/(y.sum())
    mu0 = (X[y!=1].sum(axis=0))/(n - y.sum())
    sigma1 = ((X[y==1]-mu1).T @ (X[y==1]-mu1))/(y.sum())
    sigma0 = ((X[y!=1]-mu0).T @ (X[y!=1]-mu0))/(n - y.sum())
    pi = y.sum()/n
    return mu0, mu1, sigma0, sigma1, pi

def linear_coef(mu0, mu1, sigma0, sigma1, pi):
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    
    alpha = sigma0_inv - sigma1_inv
    beta = mu1.T @ sigma1_inv - mu0.T @ sigma0_inv
    
    a = 0.5 * alpha[0,0]
    b = 0.5 * alpha[0,1]
    c = 0.5 * alpha[1,1]
    d = beta[0]
    e = beta[1]
    f = utils.logit(pi) + 0.5 * np.log(np.linalg.det(sigma0) / np.linalg.det(sigma1)) - 0.5 * mu1.T @ sigma1_inv @ mu1 + 0.5 * mu0.T @ sigma0_inv @ mu0
    
    return a, b, c, d, e, f

def log_odds(x, mu0, mu1, sigma0, sigma1, pi):
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    return utils.logit(pi) + 0.5 * np.log(np.linalg.det(sigma0) / np.linalg.det(sigma1)) - 0.5 * (x-mu1).T@sigma1_inv@(x-mu1) + 0.5 * (x-mu0).T@sigma0_inv@(x-mu0)

def proba_func(x, mu0, mu1, sigma0, sigma1, pi):
    exponent = -1 * log_odds(x, mu0, mu1, sigma0, sigma1, pi)
    return 1/(1+np.exp(exponent))

def predict(x, mu0, mu1, sigma0, sigma1, pi):
    return log_odds(x, mu0, mu1, sigma0, sigma1, pi)>=0

def conics(x, y, coefs):
    return coefs['a']*x*x + 2*coefs['b']*x*y + coefs['c']*y*y + coefs['d']*x + coefs['e']*y + coefs['f']

def plot_boundary(X, y, coefs, title='', colormap = False, save = False):
    x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
    y_min, y_max = np.min(X[:,1]), np.max(X[:,1])
    offset = 1
    
    q = 500
    tx = np.linspace(x_min - offset, x_max + offset, num=q) 
    ty = np.linspace(y_min - offset, y_max + offset, num=q) 
    X_mesh, Y_mesh = np.meshgrid(tx, ty)
    Z = conics(X_mesh, Y_mesh, coefs)
    
    plt.figure(figsize = (9, 9))
    if colormap:
        plt.clf
        plt.imshow(Z, origin="lower", extent=[x_min - offset, x_max + offset, y_min - offset, y_max + offset], aspect = (x_max - x_min)/(y_max - y_min))
    contours = plt.contour(X_mesh, Y_mesh, Z, utils.logit(1/2), colors='g')
    contours.collections[0].set_label('Decision boundary')
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