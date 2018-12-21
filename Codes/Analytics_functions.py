import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from scipy import optimize
from sklearn import metrics

#curve_fitting portion
def func_guass(r,a,b,c,d):
    return a + b * np.exp(-c * (r-d)**2)
def func_exp(r,a,b):
    return a * np.exp(-b * r)
def curve_fit_guass(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_guass,xdata,ydata,(0,0,0,0))
    (a,b,c,d) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_guass(xdata,a,b,c,d) #generate function with shifted back range
    return a,b,c,d,pcovt
def func_lin(r,a,b):
    return a*r+b
def curve_fit_lin(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_lin,xdata,ydata,(0,0), sigma = np.sqrt(ydata))
    (a,b) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_lin(xdata,a,b) #generate function with shifted back range
    return a,b,pcovt
def curve_fit_exp(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_exp,xdata,ydata,(0,0))
    (a,b) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_exp(xdata,a,b) #generate function with shifted back range
    return a,b,pcovt
def chi_sq(data_1, data_true):
    x=[]
    bin = len(data_1)
    for i in range(len(data_1)):
        z = (data_1[i] - data_true[i])**2/data_true[i]
        x.append(z)
    s = sum(x)/(bin-1)
    if s > 1.:
        print('Bad fit: Chi_sq = %f' % (s,))
    if np.isclose(s,1.):
        print('Okay fit: Chi_sq = %f' % (s,))
    if s < 1.:
        print('Great fit: Chi_sq = %f' % (s,))
def sigma(data):
    x = []
    N = len(data)
    x_bar = sum(data)/len(data)
    for i in range(len(data)):
        d = (data[i] - x_bar)**2
        x.append(d)
    return np.sqrt(sum(x)/(N-1))
def func_quad(r,a,b,c):
    return a*r**2 + b*r + c
def curve_fit_quad(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_quad,xdata,ydata,(0,0,0), sigma = np.sqrt(ydata))
    (a,b,c) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_lin(xdata,a,b) #generate function with shifted back range
    return a,b,c,pcovt
def chi_sq_sig(data_1, data_true, sigma, bin):
    x=[]
    for i in range(len(data_1)):
        z = (data_1[i] - data_true[i])**2/sigma[i]
        x.append(z)
    chi2 = sum(x)/(bin-1)
    if chi2 > 1.0:
        print('Bad fit: %lf' % (chi2,))
    elif np.isclose(chi2,1.0,0.1):
        print('Okay fit %lf' % (chi2,))
    else :
        print('Great fit: %lf' % (chi2,))
        return chi2
