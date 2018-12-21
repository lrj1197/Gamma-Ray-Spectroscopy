#Import modules needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from scipy import optimize
from sklearn import metrics
from matplotlib import rc
rc('mathtext', default='regular')
%matplotlib inline

def dataset_func():
    dataset = []
    for f in os.listdir('/Users/lucas/Documents/SLab/Data/Lab_1_data/data_master'):
        data = np.genfromtxt('/Users/lucas/Documents/SLab/Data/Lab_1_data/data_master/%s' % (f,))
        data = data.astype(str)
        data = np.insert(data,0,f[:-4],axis=0)
        dataset.append(data)
    dataset = pd.DataFrame(dataset)
    dataset = dataset.T
    for index in dataset.columns:
        dataset[dataset[index][0]] = dataset[index]
        del dataset[index]
    dataset = dataset.drop(0)
    dataset = dataset.astype('float')
    return dataset
dataset = dataset_func()
