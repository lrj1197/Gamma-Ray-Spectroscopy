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
def func_lin(r,a,b):
    return a*r+b
def curve_fit_lin(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_lin,xdata,ydata,(0,0))
    (a,b) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_lin(xdata,a,b) #generate function with shifted back range
    return a,b,pcovt
def chi_sq(data_1, data_true):
    x=[]
    bin = len(data_1)
    for i in range(len(data_1)):
        z = (data_1[i] - data_true[i])**2/data_true[i]
        x.append(z)
    s = sum(x)/(bin-1)
    if s > 1.5:
        print('Bad fit: Chi_sq = %.2f' % (s,))
    if np.isclose(s,1.,1.):
        print('Okay fit: Chi_sq = %.2f' % (s,))
    if s < 1.:
        print('Great fit: Chi_sq = %.2f' % (s,))
def func_guass(r,a,b,c,d):
    return a + b * np.exp(-c * (r-d)**2)
def curve_fit_guass(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_guass,xdata,ydata,(0,0,0,0))
    (a,b,c,d) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_guass(xdata,a,b,c,d) #generate function with shifted back range
    return a,b,c,d,pcovt

dataset = dataset_func()
#Clean
min = 0
max = 1024
dataset['channel'] = pd.DataFrame(np.linspace(min,max,1024))
noise = dataset['BKG_1']
for index in dataset.columns:
    dataset[index + ' pure signal'] = dataset[index] - noise
    dataset[index + ' pure signal'] = dataset[index + ' pure signal'].replace(np.nan,0.0)
del dataset['BKG_1 pure signal']
del dataset['BKG_2 pure signal']
del dataset['channel pure signal']

#6.288024960309859, 64.43875342118824

scale = np.array([6.288024960309859, 64.43875342118824])
dataset['Energy'] = pd.DataFrame(scale[0]*dataset['channel'] + scale[1])



Ra = dataset['Ra_226 pure signal']
OA = dataset['Ore_A pure signal']
E = dataset['Energy']
C = dataset['channel']
plt.plot(C[29:37],Ra[29:37])
plt.plot(E[:100],Ra[:100])

OA_1 = OA[75:95]
OA_2 = OA[43:53]
OA_3 = OA[37:43]
OA_4 = OA[30:36]
OA_5 = OA[:25]
E[12]
Ra_5






Ra_1 = Ra[75:95]
Ra_2 = Ra[43:53]
Ra_3 = Ra[37:43]
Ra_4 = Ra[30:36]
Ra_5 = Ra[:25]

def OA():
    popt_oa5, pcovt_oa5 = optimize.curve_fit(func_guass,E[:25],OA_5[:],(200.,1200.,0.002,140.))
    err_oa5 = np.sqrt(np.diag(pcovt_oa5))
    sigma_oa5 = np.sqrt(1/(2*popt_oa5[2]))
    y_oa5 = func_guass(E, *popt_oa5)
    #plt.plot(E[:25],OA_5)
    #plt.plot(E[:36],y_oa5[:36])
    sigma_oa5 = sigma_oa5

    popt_oa4, pcovt_oa4 = optimize.curve_fit(func_guass,E[30:36],OA_4[:],(280.,200.,0.002,435.))
    err_oa4 = np.sqrt(np.diag(pcovt_oa4))
    sigma_oa4 = np.sqrt(1/(2*popt_oa4[2]))
    y_oa4 = func_guass(E, *popt_oa4)
    #plt.plot(E[30:36],OA_4[:])
    #plt.plot(E[30:36],y_oa4[30:36])
    sigma_oa4 = sigma_oa4

    popt_oa3, pcovt_oa3 = optimize.curve_fit(func_guass,E[37:43],OA_3[:],(320.,200.,0.002,350.))
    err_oa3 = np.sqrt(np.diag(pcovt_oa3))
    sigma_oa3 = np.sqrt(1/(2*popt_oa3[2]))
    y_oa3 = func_guass(E, *popt_oa3)
    #plt.plot(E[37:43],OA_3[:])
    #plt.plot(E[35:44],y_oa3[35:44])
    #err_oa3
    sigma_oa3 = sigma_oa3
    #popt_oa3


    popt_oa2, pcovt_oa2 = optimize.curve_fit(func_guass,E[43:53],OA_2[:],(200.,400.,0.002,400.))
    err_oa2 = np.sqrt(np.diag(pcovt_oa2))
    sigma_oa2 = np.sqrt(1/(2*popt_oa2[2]))
    y_oa2 = func_guass(E, *popt_oa2)
    #plt.plot(E[43:53],OA_2[:])
    #plt.plot(E[40:54],y_oa2[40:54])
    #err_oa2
    sigma_oa2 = sigma_oa2
    #popt_oa2

    popt_oa1, pcovt_oa1 = optimize.curve_fit(func_guass,E[75:95],OA_1[:],(50.,300.,0.002,620.))
    err_oa1 = np.sqrt(np.diag(pcovt_oa1))
    sigma_oa1 = np.sqrt(1/(2*popt_oa1[2]))
    y_oa1 = func_guass(E, *popt_oa1)
    #plt.plot(E[75:95],OA_1[:])
    #plt.plot(E[70:100],y_oa1[70:100])
    #err_oa1
    sigma_oa1 = sigma_oa1
    #popt_oa1
    sig = np.array([sigma_oa1,sigma_oa2,sigma_oa3,sigma_oa4,sigma_oa5])

    max_oa1 = np.max(y_oa1)
    max_oa2 = np.max(y_oa2)
    max_oa3 = np.max(y_oa3)
    max_oa4 = np.max(y_oa4)
    max_oa5 = np.max(y_oa5)
    oa1 = np.where(dataset['Ore_A pure signal'] == max_oa1)
    oa2 = np.where(dataset['Ore_A pure signal'] == max_oa2)
    oa3 = np.where(dataset['Ore_A pure signal'] == max_oa3)
    oa4 = np.where(dataset['Ore_A pure signal'] == max_oa4)
    oa5 = np.where(dataset['Ore_A pure signal'] == max_oa5)
    oa1
    max = np.array([oa1,oa2,oa3,oa4,oa5])

    return sig
def RA():
    popt_ra5, pcovt_ra5 = optimize.curve_fit(func_guass,E[:25],Ra_5[:],(200.,1200.,0.002,180.))
    err_ra5 = np.sqrt(np.diag(pcovt_ra5))
    sigma_ra5 = np.sqrt(1/(2*popt_ra5[2]))
    y_ra5 = func_guass(E, *popt_ra5)
    #plt.plot(E[:25],OA_5)
    #plt.plot(E[:36],y_oa5[:36])
    sigma_ra5 = sigma_ra5

    popt_ra4, pcovt_ra4 = optimize.curve_fit(func_guass,E[30:36],Ra_4[:],(280.,200.,0.002,305.))
    err_ra4 = np.sqrt(np.diag(pcovt_ra4))
    sigma_ra4 = np.sqrt(1/(2*popt_ra4[2]))
    y_ra4 = func_guass(E, *popt_ra4)
    #plt.plot(E[30:36],OA_4[:])
    #plt.plot(E[30:36],y_oa4[30:36])
    sigma_ra4 = sigma_ra4

    popt_ra3, pcovt_ra3 = optimize.curve_fit(func_guass,E[37:43],Ra_3[:],(350.,200.,0.002,345.))
    err_ra3 = np.sqrt(np.diag(pcovt_ra3))
    sigma_ra3 = np.sqrt(1/(2*popt_ra3[2]))
    y_ra3 = func_guass(E, *popt_ra3)
    #plt.plot(E[37:43],OA_3[:])
    #plt.plot(E[35:44],y_oa3[35:44])
    #err_oa3
    sigma_ra3 = sigma_ra3
    #popt_oa3


    popt_ra2, pcovt_ra2 = optimize.curve_fit(func_guass,E[43:53],Ra_2[:],(200.,400.,0.002,390.))
    err_ra2 = np.sqrt(np.diag(pcovt_ra2))
    sigma_ra2 = np.sqrt(1/(2*popt_ra2[2]))
    y_ra2 = func_guass(E, *popt_ra2)
    #plt.plot(E[43:53],OA_2[:])
    #plt.plot(E[40:54],y_oa2[40:54])
    #err_oa2
    sigma_ra2 = sigma_ra2
    #popt_oa2

    popt_ra1, pcovt_ra1 = optimize.curve_fit(func_guass,E[75:95],Ra_1[:],(50.,300.,0.002,610.))
    err_ra1 = np.sqrt(np.diag(pcovt_ra1))
    sigma_ra1 = np.sqrt(1/(2*popt_ra1[2]))
    y_ra1 = func_guass(E, *popt_ra1)
    #plt.plot(E[75:95],OA_1[:])
    #plt.plot(E[70:100],y_oa1[70:100])
    #err_oa1
    sigma_ra1  = sigma_ra1
    #popt_oa1

    return sigma_ra1,sigma_ra2,sigma_ra3,sigma_ra4,sigma_ra5

OA = OA()
sig_RA = RA()
sig_OA
sig_RA
OA
#get the maxes of all the photo peaks.

max_oa1 = np.max(oa1)
#max_ra226_2 = np.max(ra2262)
#max_ra226_3 = np.max(ra2263)
#max_ra226_4 = np.max(ra2264)
max_ra226_5 = np.max(ra2265)
max_bi207_1 = np.max(bi2071)
max_bi207_2 = np.max(bi2072)
max_bi207_3 = np.max(bi2073)
max_cs137_1 = np.max(cs1371)
max_cs137_2 = np.max(cs1372)
max_cs137_3 = np.max(cs1373)
max_co60_1 = np.max(co601)
max_co60_2 = np.max(co602)


#Find which channel those maximums occur at
cs1 = int(np.where(dataset['Cs_137 pure signal'] == max_cs137_1)[0])
cs2 = int(np.where(dataset['Cs_137 pure signal'] == max_cs137_2)[0])
cs3 = int(np.where(dataset['Cs_137 pure signal'] == max_cs137_3)[0])
co1 = int(np.where(dataset['Co_60 pure signal'] == max_co60_1)[0][0])
co2 = int(np.where(dataset['Co_60 pure signal'] == max_co60_2)[0][0])
bi1 = int(np.where(dataset['Bi_207 pure signal'] == max_bi207_1)[0])
bi2 = int(np.where(dataset['Bi_207 pure signal'] == max_bi207_2)[0])
bi3 = int(np.where(dataset['Bi_207 pure signal'] == max_bi207_3)[0])
ra1 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_1)[0])
#ra2 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_2)[0])
#ra3 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_3)[0])
#ra4 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_4)[0])
ra5 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_5)[0])
