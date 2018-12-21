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
def chi_sq(xdata, ydata, scale):
    s=0.0
    bin = len(ydata)
    for i in range(len(ydata)):
        d = (ydata[i] - scale[0]*xdata[i] + scale[1])**2
        n = ydata[i]
        s += d/n

    #s = sum(x)/(bin-1)
    if s > 4.:
        print('Bad fit: Chi_sq = %.2f' % (s,))
    if np.isclose(s,1.,1.):
        print('Okay fit: Chi_sq = %.2f' % (s,))
    if s < 1.:
        print('Great fit: Chi_sq = %.2f' % (s,))
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

#Slice the data and get the local maximums for eah signal
ra2265 = dataset['Ra_226 pure signal'][75:95]
#ra2264 = dataset['Ra_226 pure signal'][43:53]
#ra2263 = dataset['Ra_226 pure signal'][37:43]
#ra2262 = dataset['Ra_226 pure signal'][29:37]
ra2261 = dataset['Ra_226 pure signal'][8:16]
bi2073 = dataset['Bi_207 pure signal'][140:172]
bi2072 = dataset['Bi_207 pure signal'][70:90]
bi2071 = dataset['Bi_207 pure signal'][7:15]
cs1373 = dataset['Cs_137 pure signal'][80:108]
cs1372 = dataset['Cs_137 pure signal'][8:15]
cs1371 = dataset['Cs_137 pure signal'][:8]
co602 = dataset['Co_60 pure signal'][188:220]
co601 = dataset['Co_60 pure signal'][162:188]

#get the max for each peak
max_ra226_1 = np.max(ra2261)
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
#ra4,ra3,ra2
bin = np.array([ra5,bi2,ra1, ra4,ra3,ra2,cs3,co1,co2])+1
bin
ra1t = 609.31
ra2t = 351.92
ra3t = 295.21
ra4t = 241.98
ra5t = 186.211
cs1t = 661.657
bi1t = 569.702
co1t = 1173.237
co2t = 1332.501
sigma =np.array([3.25999329, 5.7862443 , 7.85374188, 4.56435465, 3.67060757,
       3.00886804, 4.14761529, 2.9373589 , 1.71719286])
#sigma.sort(axis=0)
#sigma
#len(sigma)
#ra4t,ra3t,ra2t,

Energy = np.array([ra5t,bi1t,ra1t,ra4t,ra3t,ra2t,cs1t,co1t,co2t])
Energy.sort(axis=0)
bin.sort(axis=0)
Energy = Energy.astype(float)
bins = np.linspace(0,1024,1024)

#error = np.array([1.0,2.987,13.421, 18.118])
#curvfit with lin function to find the parameters
scale = curve_fit_lin(bin,Energy)
#apply the scale to bin to find where they hit
E = scale[0]*bin + scale[1]
#get the uncertainty in the parameters
err = np.sqrt(np.diag(scale[2]))
t = np.linspace(0,230,1024)
y = scale[0] * t + scale[1]
err
print(scale)
plt.errorbar(bin,Energy,xerr=sigma ,linestyle = "None", label = 'Measured Values', c='r')
plt.scatter(bin, Energy,s = 20, c='k')
plt.plot(t,y,label = 'Fitted Values')
plt.xlabel('Channel')
plt.ylabel("Energy (keV)")
plt.savefig("CalibratednMCA.png")


dataset['Energy'] = pd.DataFrame(scale[0]*dataset['channel'] + scale[1])


measured_E = []
for index in bin:
    measured_E.append(dataset['Energy'][index])
np.array(measured_E).sort(axis=0)
dataset['channel'][1:][0]
chi = chi_sq(measured_E, Energy)
dataset['channel'][1]
chi_sq(dataset['channel'][1:],dataset['Energy'][1:],scale)

scale[1]



x = - np.log(0.5)/0.06

dxdI = -1./(Is*x)
dI = np.sqrt(Is)
dxdI0 = 1./(x*I0)
dI0 = np.sqrt(I0)
dxdu = np.log(Is/I0)/x**2
du = 0.005
dx = ((dxdI*dI)**2 + (dxdI0*dI0)**2 + (dxdu*du)**2)**(1/2)



u = mu(Is,I0,i)
u/11.




x
