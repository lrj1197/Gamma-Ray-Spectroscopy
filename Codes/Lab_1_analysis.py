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
    popt, pcovt = optimize.curve_fit(func_lin,xdata,ydata,(0,0))
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
    if s > 1.5:
        print('Bad fit: Chi_sq = %.2f' % (s,))
    if np.isclose(s,1.,1.):
        print('Okay fit: Chi_sq = %.2f' % (s,))
    if s < 1.:
        print('Great fit: Chi_sq = %.2f' % (s,))
def sig(data):
    x = []
    N = len(data)
    x_bar = sum(data)/len(data)
    for i in data.index:
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
    if chi2 >= 1.5:
        print('Bad fit: %lf' % (chi2,))
    elif np.isclose(chi2,1.0,1.):
        print('Okay fit %lf' % (chi2,))
    else :
        print('Great fit: %lf' % (chi2,))
        return chi2
def std_err(sigma, bin):
    return sigma/np.sqrt(bin)

dataset = dataset_func()
decay = pd.read_csv('~/Documents/SLab/Data/Lab_1_data/MASTER.csv', sep=',', header=None)
decay = pd.DataFrame(decay)
decay.columns
canidates = decay['Isotope'].unique()

cal_d = pd.read_csv('~/Documents/SLab/Data/CalibrationData.csv', sep=',', header=None)
cal_d[0] = cal_d[0].drop(0)
cal_d[0] = cal_d[0].drop(8)
cal_d[0] = cal_d[0].drop(15)
cal_d[0] = cal_d[0].drop(18)
cal_d[0] = cal_d[0].dropna()
cal_d[0] = cal_d[0].astype(float) #post del isotopes from column

cal_d
#
min = 0
max = 1024
dataset['channel'] = pd.DataFrame(np.linspace(min,max,1024))
noise = dataset['BKG_1']
for index in dataset.columns:
    dataset[index + ' pure signal'] = dataset[index] - noise
    dataset[index + ' pure signal'] = dataset[index + ' pure signal'].replace(0.0, np.nan)
del dataset['BKG_1 pure signal']
del dataset['BKG_2 pure signal']
del dataset['channel pure signal']


plt.figure()
plt.plot(dataset['Energy'][:100], dataset['Bi_207 pure signal'][:100], c = 'r',label= 'Bi-207')
plt.plot(dataset['Energy'][:120], dataset['Cs_137 pure signal'][:120], c = 'b',label= 'Cs-137')
plt.plot(dataset['Energy'][:120], dataset['Co_60 pure signal'][:120], c = 'g',label= 'Co-60')
plt.plot(dataset['Energy'][:120], dataset['Ra_226 pure signal'][:120], c = 'k',label= 'Ra-226')
plt.plot(dataset['Energy'][:120], dataset['Ore_A pure signal'][:120], c = 'c',label= 'Unknown Sample A')
plt.legend()
plt.ylim(0,5000)
plt.xlabel('Energy (keV)')
plt.ylabel('Counts per Energy')
#plt.title('Signals for Unknown sample Ore A plotted with known samples')
plt.savefig('/Users/lucas/Documents/SLab/SampleA.png')

plt.figure()
plt.plot(dataset['Energy'][100:220], dataset['Ra_226 pure signal'][100:220], c = 'k',label= 'Ra-226')
plt.plot(dataset['Energy'][:220], dataset['Ore_A pure signal'][:220], c = 'b',label= 'Unknown Sample A')
plt.legend()
plt.ylim(0,5000)
plt.xlabel('Energy (keV)')
plt.ylabel('Counts per Energy')
#plt.title('Signals for Unknown sample Ore A plotted with known samples')
plt.savefig('/Users/lucas/Documents/SLab/SampleA_Ra226.png')

#Ra226 6 k [8:16],[29:37],[37:43],[43:53],[75:95]
#Bi207 3 r [7:15],[70:90],[140:172]
#Cs137 2 b [:8],[8:15],[80:108]
#Co60 2 g [162:188],[188:220]
plt.plot(bins[:100],dataset['Ra_226 pure signal'][:100] , c = 'k',label= 'Ra-226')

#9/21/18
#to do
#get calibrations==done?, curvfit== done, get sigma== done, chi2 for lin fit, error,
#Attenuation
################################################################################
#Slice the data and get the local maximums for each signal
#Slice the data and get the local maximums for each signal
ra2265 = dataset['Ra_226 pure signal'][75:95]
ra2264 = dataset['Ra_226 pure signal'][41:53]
ra2263 = dataset['Ra_226 pure signal'][37:43]
ra2262 = dataset['Ra_226 pure signal'][29:37]
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
max_ra226_2 = np.max(ra2262)
max_ra226_3 = np.max(ra2263)
max_ra226_4 = np.max(ra2264)
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
ra2 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_2)[0])
ra3 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_3)[0])
ra4 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_4)[0])
ra5 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_5)[0])

bin = np.array([ra5,ra4,ra3,ra2,bi2,ra1,cs3,co1,co2])+1
ra1t = 609.31
ra2t = 351.92
ra3t = 295.21
ra4t = 241.98
ra5t = 186.211
cs1t = 661.657
bi1t = 569.702
co1t = 1173.237
co2t = 1332.501

Energy = np.array([ra5t,ra4t,ra3t,ra2t,bi1t,ra1t,cs1t,co1t,co2t])
Energy.sort(axis=0)
bin.sort(axis=0)
Energy = Energy.astype(float)
bins = np.linspace(0,1024,1024)
sigma_mean = sigma
sigma
#sigma = cs3,co1,co2,bi2,bi3,ra1,ra2,ra3,ra4
#error = np.array([1.0,2.987,13.421, 18.118])
#curvfit with lin function to find the parameters
scale = curve_fit_lin(bin,Energy)
#apply the scale to bin to find where they hit
E = scale[0]*bin + scale[1]
#get the uncertainty in the parameters
err = np.sqrt(np.diag(scale[2]))
t = np.linspace(0,230,1024)
y = scale[0] * t + scale[1]
lst = scale[0]*dataset['channel'] + scale[1]
dataset['Energy'] = pd.DataFrame(lst)
#chi_sq
vals = [19,28,37,46,79,86,95,176,202]
measured_E = []
for index in bin:
    measured_E.append(dataset['Energy'][index])
chi = chi_sq(measured_E, Energy)

scale
#plot
plt.errorbar(bin,Energy,xerr=sigma_cal ,linestyle = "None", label = 'Measured Values', c='r')
plt.scatter(bin, Energy)
plt.plot(t,y,label = 'Fitted Values')
plt.xlabel('Channel Number')
plt.ylabel('Energy (keV)')
plt.title('Calibration of MCA')
plt.savefig('Calibration of MCA.png')
plt.legend()
#plt.text(150,200, "$Chi^{2}$ = %f" % (chi2s,))
plt.savefig('/Users/lucas/Documents/SLab/Data/Lab_1_data/Calibration/CalibrationofMCA.png')
#get chi2 with sigma
#chi2s = chi_sq_sig(ypt, Energy, sigma_cal, len(Energy))
#err
#err =array([ 0.17565474, 19.57569468])
#build the Energy column in dataset

lst = scale[0]*bins + scale[1]
dataset['Energy'] = pd.DataFrame(lst)
#chi_sq
vals = [93,175,203,79,158,12,34,85]
measured_E = []
for index in vals:
    measured_E.append(dataset['Energy'][index])
chi = chi_sq(measured_E, Energy)
#x = dataset['Energy'][4] - dataset['Energy'][3]
#each channel covers 6.715586093213975  keV
#err
m = scale[0]
dm = err[0]
db = err[1]
x = dataset['Energy'][4] - dataset['Energy'][3]
dx = np.sqrt(1024.)
b = scale[1]
dE = np.sqrt((x*dm)**2 + (db)**2)
dE

#dE = 12.455915181662474
#chi sq = 1.2



#plot evereything to see it makes sense
plt.plot(dataset['Energy'][:300], dataset['BKG_1'][:300], c = 'r',label= 'Background')
plt.legend()
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.savefig("/Users/lucas/Documents/SLab/BKG.png")
################################################################################
################################################################################
#curvfit portion
#To do: finish curvfitting the elements=done, get all chi2


#To be fitted
ra = dataset['Ra_226 pure signal']
E  = dataset['Energy']


plt.plot(E[70:100],ra[70:100])

ra2261 = dataset['Ra_226 pure signal'][75:95]
ra2262 = dataset['Ra_226 pure signal'][43:53]
ra2263 = dataset['Ra_226 pure signal'][37:43]
ra2264 = dataset['Ra_226 pure signal'][29:37]
dataset['Energy'][34]
dataset['Ra_226 pure signal'][29:37]

cal_d
bi2071 = dataset['Bi_207 pure signal'][140:172]
bi2072 = dataset['Bi_207 pure signal'][70:90]
#bi2073 = dataset['Bi_207 pure signal'][7:15]
cs1371 = dataset['Cs_137 pure signal'][80:108]
#cs1372 = dataset['Cs_137 pure signal'][8:15]
#cs1373 = dataset['Cs_137 pure signal'][:8]
co60 = dataset['Co_60 pure signal']
co602 = dataset['Co_60 pure signal'][162:188]

bi2071am = dataset['Bi_207A_med pure signal']
bi2072am = dataset['Bi_207A_med pure signal']
bi2071as = dataset['Bi_207A_small pure signal']
bi2072as = dataset['Bi_207A_small pure signal']
cs1371am = dataset['Cs_137A_med pure signal']
cs1371as = dataset['Cs_137A_small pure signal']
co601am = dataset['Co_60A_med pure signal']
co602am = dataset['Co_60A_med pure signal']
co601as = dataset['Co_60A_small pure signal']
co602as = dataset['Co_60A_small pure signal']


#sigma = cs3,co1,co2,bi2,bi3,ra1,ra2,ra3,ra4
sigma = np.array([sigma_cs1,sigma_co1,sigma_co2,sigma_bi2,sigma_bi2,sigma_ra1,sigma_ra2,sigma_ra4])
#dataset['Energy'][29:37]
#Bi207 1
popt_bi1, pcovt_bi1 = optimize.curve_fit(func_guass,dataset['Energy'][140:172],bi2071[:],(10.,700.,0.002,1050.))
err_bi1 = np.sqrt(np.diag(pcovt_bi1))
y_bi1 = func_guass(dataset['Energy'],*popt_bi1)
sigma_bi1 = np.sqrt(1/(2*popt_bi1[2]))
plt.plot(dataset['Energy'][140:172],bi2071)
plt.plot(dataset['Energy'][130:180],y_bi1[130:180])
err_bi1
sigma_bi1
popt_bi1

#Bi207 2
popt_bi2, pcovt_bi2 = optimize.curve_fit(func_guass,dataset['Energy'][70:90],bi2072[:],(450.,4500.,0.002,590.))
err_bi2 = np.sqrt(np.diag(pcovt_bi2))
y_bi2 = func_guass(dataset['Energy'],*popt_bi2)
sigma_bi2 = np.sqrt(1/(2*popt_bi2[2]))
plt.plot(dataset['Energy'][70:90],bi2072)
plt.plot(dataset['Energy'][60:100],y_bi2[60:100])
err_bi2
sigma_bi2
popt_bi2

#Cs137 1
popt_cs1, pcovt_cs1 = optimize.curve_fit(func_guass,dataset['Energy'][80:108],cs1371,(600.,2000.,0.002,670))
err_cs1 = np.sqrt(np.diag(pcovt_cs1))
y_cs1 = func_guass(dataset['Energy'],*popt_cs1)
sigma_cs1 = np.sqrt(1/(2*popt_cs1[2]))
plt.plot(dataset['Energy'][80:108],cs1371)
plt.plot(dataset['Energy'],y_cs1)
err_cs1
sigma_cs1
popt_cs1

#Co60 1
popt_co1, pcovt_co1 = optimize.curve_fit(func_guass,dataset['Energy'][158:180],co60[158:180],(10,400,0.002,1325))
err_co1 = np.sqrt(np.diag(pcovt_co1))
y_co1 = func_guass(dataset['Energy'],*popt_co1)
sigma_co1 = np.sqrt(1/(2*popt_co1[2]))
plt.plot(dataset['Energy'][158:180],co60[158:180])
plt.plot(dataset['Energy'][150:220],y_co1[150:220])
err_co1
sigma_co1
popt_co1



#Co60 2
popt_co2, pcovt_co2 = optimize.curve_fit(func_guass,dataset['Energy'][162:188],co602,(100.,500.,0.002,1160))
err_co2 = np.sqrt(np.diag(pcovt_co2))
y_co2 = func_guass(dataset['Energy'],*popt_co2)
sigma_co2 = np.sqrt(1/(2*popt_co2[2]))
plt.plot(dataset['Energy'][162:188],co602)
plt.plot(dataset['Energy'][150:200],y_co2[150:200])
err_co2
sigma_co2
popt_co2


#Ra226 1
popt_ra1, pcovt_ra1 = optimize.curve_fit(func_guass,dataset['Energy'][75:95],ra2261[:],(200.,800.,0.002,620.))
err_ra1 = np.sqrt(np.diag(pcovt_ra1))
y_ra1 = func_guass(dataset['Energy'],*popt_ra1)
sigma_ra1 = np.sqrt(1/(2*popt_ra1[2]))
plt.plot(dataset['Energy'][75:95],ra2261)
plt.plot(dataset['Energy'][60:100],y_ra1[60:100])
err_ra1
sigma_ra1
popt_ra1
dataset['Energy'][75:95]


#Ra226 2
popt_ra2, pcovt_ra2 = optimize.curve_fit(func_guass,dataset['Energy'][43:53],ra2262[:],(700.,2000.,0.002,400.))
err_ra2 = np.sqrt(np.diag(pcovt_ra2))
sigma_ra2 = np.sqrt(1/(2*popt_ra2[2]))
y_ra2 = func_guass(dataset['Energy'],*popt_ra2)
plt.plot(dataset['Energy'][43:53],ra2262[:])
plt.plot(dataset['Energy'][40:60],y_ra2[40:60])
err_ra2
popt_ra2
sigma_ra2

#Ra226 3
popt_ra3, pcovt_ra3 = optimize.curve_fit(func_guass,dataset['Energy'][37:43],ra2263[:],(1000.,2000.,0.002,360.))
err_ra3 = np.sqrt(np.diag(pcovt_ra3))
y_ra3 = func_guass(dataset['Energy'],*popt_ra3)
sigma_ra3 = np.sqrt(1/(2*popt_ra3[2]))
plt.plot(dataset['Energy'][37:43],ra2263[:])
plt.plot(dataset['Energy'][37:43],y_ra3[37:43])
err_ra3
popt_ra3
sigma_ra3

#Ra226 4
#peak_ra4 = curve_fit_guass(dataset['Energy'][29:37], ra2264)
popt_ra4, pcovt_ra4 = optimize.curve_fit(func_guass,dataset['Energy'][29:37],ra2264,(1000.,1400.,0.002,320.))
err_ra4 = np.sqrt(np.diag(pcovt_ra4))
sigma_ra4 = np.sqrt(1/(2*popt_ra4[2]))
y_ra4 = func_guass(dataset['Energy'], *popt_ra4)
plt.plot(dataset['Energy'][29:37],ra2264[:])
plt.plot(dataset['Energy'][28:40],y_ra4[28:40])
err_ra4
sigma_ra4
popt_ra4

#Ra226 4
#peak_ra4 = curve_fit_guass(dataset['Energy'][29:37], ra2264)
popt_ra5, pcovt_ra5 = optimize.curve_fit(func_guass,dataset['Energy'][20:30],dataset['Ra_226 pure signal'][20:30],(1000.,500.,0.002,250.))
err_ra5 = np.sqrt(np.diag(pcovt_ra5))
sigma_ra5 = np.sqrt(1/(2*popt_ra5[2]))
y_ra5 = func_guass(dataset['Energy'], *popt_ra5)
plt.plot(dataset['Energy'][20:30],dataset['Ra_226 pure signal'][20:30])
plt.plot(dataset['Energy'][20:40],y_ra5[20:40])
err_ra5
sigma_ra5
popt_ra5

ra2265 = dataset['Ra_226 pure signal'][75:95]
ra2264 = dataset['Ra_226 pure signal'][41:53]
ra2263 = dataset['Ra_226 pure signal'][37:43]
ra2262 = dataset['Ra_226 pure signal'][29:37]
ra2261 = dataset['Ra_226 pure signal'][8:16]
bi2073 = dataset['Bi_207 pure signal'][140:172]
bi2072 = dataset['Bi_207 pure signal'][70:90]
bi2071 = dataset['Bi_207 pure signal'][7:15]
cs1373 = dataset['Cs_137 pure signal'][80:108]

co602 = dataset['Co_60 pure signal'][100:188]
co601 = dataset['Co_60 pure signal'][162:188]

plt.plot(dataset['Energy'][145:158],dataset['Co_60 pure signal'][145:158])
ra5t,bi1t,ra1t,cs1t,co1t,co2t


sigma = np.array([ sigma_ra5/np.sqrt(16-8), sigma_bi1/np.sqrt(90-70),sigma_ra1/np.sqrt(95-75),sigma_cs1/np.sqrt(108-80), sigma_co1/np.sqrt(158-145), sigma_co2/np.sqrt(170-158)])

sigma
################################################################################
#curvfit unknown sample
cal_d
OA = dataset['Ore_A pure signal']
E = dataset['Energy']
plt.plot(E[:100],OA[:100])
plt.plot(E[20:30],OA[20:30])

OA_1 = OA[75:95]
OA_2 = OA[43:53]
OA_3 = OA[37:43]
OA_4 = OA[30:36]
OA_5 = OA[20:30]

popt_oa5, pcovt_oa5 = optimize.curve_fit(func_guass,E[20:30],OA_5[:],(3250.,200.,0.002,245.))
err_oa5 = np.sqrt(np.diag(pcovt_oa5))
sigma_oa5 = np.sqrt(1/(2*popt_oa5[2]))
y_oa5 = func_guass(E, *popt_oa5)
plt.plot(E[20:30],OA_5[:])
plt.plot(E[10:36],y_oa5[10:36])
sigma_oa5

popt_oa4, pcovt_oa4 = optimize.curve_fit(func_guass,E[30:36],OA_4[:],(280.,400.,0.002,320.))
err_oa4 = np.sqrt(np.diag(pcovt_oa4))
sigma_oa4 = np.sqrt(1/(2*popt_oa4[2]))
y_oa4 = func_guass(E, *popt_oa4)
plt.plot(E[30:36],OA_4[:])
plt.plot(E[30:36],y_oa4[30:36])
err_oa4
sigma_oa4
popt_oa4

popt_oa3, pcovt_oa3 = optimize.curve_fit(func_guass,E[37:43],OA_3[:],(320.,200.,0.002,350.))
err_oa3 = np.sqrt(np.diag(pcovt_oa3))
sigma_oa3 = np.sqrt(1/(2*popt_oa3[2]))
y_oa3 = func_guass(E, *popt_oa3)
plt.plot(E[37:43],OA_3[:])
plt.plot(E[35:44],y_oa3[35:44])
err_oa3
sigma_oa3
popt_oa3


popt_oa2, pcovt_oa2 = optimize.curve_fit(func_guass,E[43:53],OA_2[:],(200.,400.,0.002,400.))
err_oa2 = np.sqrt(np.diag(pcovt_oa2))
sigma_oa2 = np.sqrt(1/(2*popt_oa2[2]))
y_oa2 = func_guass(E, *popt_oa2)
plt.plot(E[43:53],OA_2[:])
plt.plot(E[40:54],y_oa2[40:54])
err_oa2
sigma_oa2
popt_oa2

popt_oa1, pcovt_oa1 = optimize.curve_fit(func_guass,E[75:95],OA_1[:],(50.,300.,0.002,620.))
err_oa1 = np.sqrt(np.diag(pcovt_oa1))
sigma_oa1 = np.sqrt(1/(2*popt_oa1[2]))
y_oa1 = func_guass(E, *popt_oa1)
plt.plot(E[75:95],OA_1[:])
plt.plot(E[70:100],y_oa1[70:100])
err_oa1
sigma_oa1
popt_oa1

################################################################################
#plot OA and Ra together
Ra = dataset['Ra_226 pure signal']
E = dataset['Energy']
plt.plot(E[:100],3.3*OA[:100],label = 'Ore_A')
plt.plot(E[:100],Ra[:100],label='Ra226')
plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.legend()
plt.savefig("/Users/lucas/Documents/SLab/Ore_A_Ra226.png")

################################################################################
#plot the energy vs the counts vs the intensities to cross check.

intensities = np.array([3.59,7.43,19.3,37.6,46.1])*100.
Energies = np.array([186,241,295,351,609])

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(dataset['Energy'][:120],dataset['Ore_A pure signal'][:120],c='r', label = 'Unknown Source - Ore_A')
#ax2.scatter(80.574,i[86],c='k',s=5,label = 'Ho166')
#ax2.scatter(cal_d[0][2], cal_d[1][2],c='b',s=5,label = 'Ra266')
ax2.scatter(Energies, intensities, c='b', s=5)

'''
ax2.scatter(cal_d[0][21], cal_d[1][21],c='k',s=5,label = 'Co60')
ax2.scatter(cal_d[0][22], cal_d[1][22],c='k',s=5)
ax2.scatter(cal_d[0][10], cal_d[1][10],c='g',s=5,label = 'Bi207')
ax2.scatter(cal_d[0][12], cal_d[1][12],c='g',s=5)
ax2.scatter(cal_d[0][14],cal_d[1][14],c='g',s=5)
ax2.scatter(cal_d[0][17], cal_d[1][17],c='c',s=5,label = 'Cs137')
'''
ax.set_xlabel(r"Energy (keV)")
ax.set_ylabel(r"Counts")
ax2.set_ylabel(r"Number of Decays per 100 isotopes")
ax.set_xlim(0,750)
ax2.set_ylim(0, 7500)
ax.set_ylim(0,700)
ax.legend(loc=0)
ax2.legend(loc=2)
plt.show()
#Bi133?

#to do:
#subtract of local bkg from signal, replot
    #curvfit with lin local bkg from 100 to 600 keV
    #generate func from 0,1024
    #subtract bin from func

ra = dataset['Ra_226 pure signal']
plt.plot(E[5:100],ra[5:100])

#53
#44
#37
#30
#20
cal_d


signal = np.array([ra[20],ra[30],ra[37],ra[44],ra[53]])
En = np.array([E[20],E[30],E[37],E[44],E[53]])
plt.plot(En,signal)

scale = curve_fit_lin(En,signal)
#apply the scale to bin to find where they hit
E1 = scale[0]*bin + scale[1]
#get the uncertainty in the parameters
err = np.sqrt(np.diag(scale[2]))
t = np.linspace(250,425,len(ra[:]))
y = scale[0] * t + scale[1]

#plot
plt.scatter(En, signal)
plt.plot(t,y,label = 'Fitted Values')



ra = ra - y

ra = ra.apply(lambda x: np.abs(x))


################################################################################
#Attenuation
#call the columns
#get the I and I0s for each isotpe and size, use all info to get mu
bi1 = dataset['Bi_207 pure signal']
bi = dataset['Bi_207 pure signal']
ra = dataset['Ra_226 pure signal']
co1 = dataset['Co_60 pure signal']
E = dataset['Energy']

plt.plot(Elr[:200],bi[:200])
plt.plot(Elr[:200],ra[:200])
plt.plot(Elr[:200],cs[:200])


#cs1372 = dataset['Cs_137 pure signal'][8:15]
#cs1373 = dataset['Cs_137 pure signal'][:8]
co1 = dataset['Co_60 pure signal']
co2 = dataset['Co_60 pure signal']
bi1am = dataset['Bi_207A_med pure signal']
bi2am = dataset['Bi_207A_med pure signal']
bi1as = dataset['Bi_207A_small pure signal']
bi2as = dataset['Bi_207A_small pure signal']
cs1 = dataset['Cs_137 pure signal']
cs1as = dataset['Cs_137A_small pure signal']
co1am = dataset['Co_60A_med pure signal']
co2am = dataset['Co_60A_med pure signal']
co1as = dataset['Co_60A_small pure signal']
co2as = dataset['Co_60A_small pure signal']



bi1 = dataset['Bi_207 pure signal'][]
bi = dataset['Bi_207 pure signal']
ra = dataset['Ra_226 pure signal']
co1 = dataset['Co_60 pure signal']
E = dataset['Energy']

plt.plot(Elr[:200],bi[:200])
plt.plot(Elr[:200],ra[:200])
plt.plot(Elr[:200],cs[:200])


#cs1372 = dataset['Cs_137 pure signal'][8:15]
#cs1373 = dataset['Cs_137 pure signal'][:8]
co1 = dataset['Co_60 pure signal']
co2 = dataset['Co_60 pure signal']
bi1am = dataset['Bi_207A_med pure signal']
bi2am = dataset['Bi_207A_med pure signal']
bi1as = dataset['Bi_207A_small pure signal']
bi2as = dataset['Bi_207A_small pure signal']
cs1am = dataset['Cs_137A_med pure signal']
cs1as = dataset['Cs_137A_small pure signal']
co1am = dataset['Co_60A_med pure signal']
co2am = dataset['Co_60A_med pure signal']
co1as = dataset['Co_60A_small pure signal']
co2as = dataset['Co_60A_small pure signal']

E = dataset['Energy']
plt.plot(E[190:220],co1[190:220])
plt.plot(E[190:220],co1am[190:220])
plt.plot(E[158:180],co1as[158:180])

plt.plot(E[158:180],co2[158:180])
plt.plot(E[160:190],co2am[160:190])
plt.plot(E[140:158],co2as[140:158])



plt.plot(E[75:100],cs1as[75:100])

plt.plot(E[60:100],bi1[60:100])
plt.plot(E[60:100],bi1am[60:100])
plt.plot(E[60:100],bi1as[60:100])
#################################################################################
dataset = dataset.dropna()

#Bi207as 1
popt_bi1, pcovt_bi1 = optimize.curve_fit(func_guass,dataset['Energy'][60:100],bi2071as[60:100],(200.,2500.,0.002,580.))
y_bi1as = func_guass(dataset['Energy'],*popt_bi1)

#Cs137as 1
popt_cs1, pcovt_cs1 = optimize.curve_fit(func_guass,dataset['Energy'][75:100],cs1371as[75:100],(200.,1600.,0.002,660.))
y_cs1as = func_guass(dataset['Energy'],*popt_cs1)

#Co60as 1
popt_co1, pcovt_co1 = optimize.curve_fit(func_guass,dataset['Energy'][158:180],co601as[158:180],(10.,400.,0.002,1300))
y_co1as = func_guass(dataset['Energy'],*popt_co1)

#Co60as 2
popt_co2, pcovt_co2 = optimize.curve_fit(func_guass,dataset['Energy'][140:158],co602as[140:158],(50.,500.,0.002,1150))
y_co2as = func_guass(dataset['Energy'],*popt_co2)

plt.plot(E[:100], bi1[:100])
plt.ylim(0,3000)
plt.plot(E[:100],y_bi1as[:100])
#In[140]:

cs1

#79
En = np.array([292.866,E[79],E[93],1173.237,1332.501])
#Find max of gauss fit

max_Bi207_1 = np.max(bi1[60:100])
max_Cs137_1 = np.max(cs1[75:100])
max_Co60_1 = np.max(co1[158:180])
max_Co60_2 = np.max(co2[140:158])



max_Bi207_1as = np.max(y_bi1as)
max_Cs137_1as = np.max(y_cs1as)
max_Co60_1as = np.max(y_co1as)
max_Co60_2as = np.max(y_co1as)


max_Bi207_1
max_Bi207_1as


I0 = np.array([max_Bi207_1, max_Cs137_1, max_Co60_1, max_Co60_2])
#I = np.array([max_Bi207_1am, max_Bi207_2am, max_Cs137_1am, max_Co60_1am, max_Co60_2am])
Is = np.array([max_Bi207_1as, max_Cs137_1as, max_Co60_1as, max_Co60_2as])
x1 =(25,10)
def mu(x1,x2,d):
    return -np.log(x1/x2)/d
i = 10.0
'''
    else:
        dudI = -1./(Im*i)
        dI = np.sqrt(Im)
        dudI0 = 1./(i*I0)
        dI0 = np.sqrt(I0)
        dudx = np.log(Im/I0)/i**2
        dx = 0.05
        du = ((dudI*dI)**2 + (dudI0*dI0)**2 + (dudx*dx)**2)**(1/2)
        u = mu(Im,I0,i)
        print(u, du, i)
'''
dudI = -1./(Is*i)
dI = np.sqrt(Is)
dudI0 = 1./(i*I0)
dI0 = np.sqrt(I0)
dudx = np.log(Is/I0)/i**2
dx = 0.05
du = ((dudI*dI)**2 + (dudI0*dI0)**2 + (dudx*dx)**2)**(1/2)
u = mu(Is,I0,i)
u/11.
 # cm +- 0.5 mm
for i in x1:
    if i == x1[1]:
        dudI = -1./(Is*i)
        dI = np.sqrt(Is)
        dudI0 = 1./(i*I0)
        dI0 = np.sqrt(I0)
        dudx = np.log(Is/I0)/i**2
        dx = 0.05
        du = ((dudI*dI)**2 + (dudI0*dI0)**2 + (dudx*dx)**2)**(1/2)
        u = mu(Is,I0,i)
        print(u, du, i)


atten = np.genfromtxt('/Users/lucas/Documents/SLab/Data/Pb_atten.txt')
atten = pd.DataFrame(atten)
NIST_Energy = atten[0]
NIST_mu = atten[1]
us/1.1
plt.plot(NIST_Energy[42:50],NIST_mu[42:50])
plt.errorbar(En/1000, us/1.1, yerr = np.array([.03,.07,.04,.04])/1.1, linestyle = "None", c = 'r')
plt.scatter(En/1000, us/1.1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy (Mev)')
plt.ylabel(r'$\mu/\rho$ (cm$^{2}$/g)')
plt.savefig('/Users/lucas/Documents/SLab/MAC_zoomed.png')
#plt.title(r'NIST $\mu/\rho$ versus obtained data')
#a = plt.axes([.65, .6, .2, .2])
plt.plot(NIST_Energy[:],NIST_mu[:])
plt.errorbar(En/1000, us/1.1, yerr = np.array([0.02,.03,.04,.01,.01]), linestyle = "None", c = 'r')
plt.scatter(En/1000, us/1.1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy (Mev)')
plt.ylabel(r'$\mu/\rho$ (cm$^{2}$/g)')
plt.savefig('/Users/lucas/Documents/SLab/MAC.png')



# this is an inset axes over the main axes


us = np.array([0.215/1.1,0.11796455, 0.16772791, 0.06101919, 0.06101919])
dus = [0.0025,0.0031796 ,0.00745917, 0.0046732,  0.0046732 ]
um = np.array([0.02111443, 0.07474451, 0.08808224 ,0.0123268  ,0.0123268 ])
dum = [0.00264623, 0.00315199, 0.00422021, 0.00720678, 0.00720678]
################################################################################


################################################################################

x = [ 354.588, 646.387, 1055.096 ,716.784, 810.951 ,413.66, 519.326 ,323.637 ,779.143 ]
E = [569.70,1063.66,1770.24,1173.24,1332.50, 661.66,834.86,511.02,1274.54]


popt, pcovt = optimize.curve_fit(func_lin,x,E,(0,0))
popt
err = np.sqrt(np.diag(pcovt))
err

m = popt[0]
b = popt[1]
dm = err[0]
db = err[1]

dE = np.sqrt((x*dm)**2 + (db)**2)
dx=[0.039,0.035,0.403,0.129,0.096,0.036,0.042,0.073,0.068 ]
dE






################################################################################
'''
Background Noise Measured by Scintillator
Counts Vs. Bins
'''
dataset.columns

b = dataset['channel']
noise = dataset['BKG_1']


plt.plot(b[:200],noise[:200],label = 'Background Noise')
plt.xlabel('Channel Number')
plt.ylabel('Counts')
#plt.title('Calibration of MCA')
plt.savefig('Noise_channel.png')
plt.legend()
#plt.text(150,200, "$Chi^{2}$ = %f" % (chi2s,))

'''
Raw Calibration Data
Counts Vs Bins
Plot Co Cs and Bi all on same with noise included
'''

co = dataset['Co_60']
bi = dataset['Bi_207']
cs = dataset['Cs_137']
ra = dataset['Ra_226']
plt.plot(b[:250],noise[:250],label = 'Background Noise')
plt.plot(b[:250],co[:250],label = 'Co_60')
plt.plot(b[:250],cs[:250],label = 'Cs_137')
plt.plot(b[:250],bi[:250],label = 'Bi_207')
plt.plot(b[:250],ra[:250],label = 'Ra_226')
plt.xlabel('Channel Number')
plt.ylabel('Counts')
plt.ylim(0,5000)
plt.legend()
#plt.title('Calibration of MCA')
plt.savefig('RawCalibrationData.png')



'''
Pure Calibration Data
Counts Vs. Bins
Same plot as above just minus noise
'''
co = dataset['Co_60 pure signal']
bi = dataset['Bi_207 pure signal']
cs = dataset['Cs_137 pure signal']
ra = dataset['Ra_226 pure signal']
#plt.plot(b[:250],noise[:250],label = 'Background Noise')
plt.plot(b[:250],co[:250],label = 'Co_60')
plt.plot(b[:250],cs[:250],label = 'Cs_137')
plt.plot(b[:250],bi[:250],label = 'Bi_207')
plt.plot(b[:250],ra[:250],label = 'Ra_226')
plt.xlabel('Channel Number')
plt.ylabel('Counts')
plt.ylim(0,5000)
plt.legend()
#plt.title('Calibration of MCA')
plt.savefig('PureCalibrationData.png')




'''
Calibrated Environment
Counts Vs. keV
Same as above but calibrated x axis
'''
E = dataset['Energy']
co = dataset['Co_60 pure signal']
bi = dataset['Bi_207 pure signal']
cs = dataset['Cs_137 pure signal']
ra = dataset['Ra_226 pure signal']
#plt.plot(E[:250],noise[:250],label = 'Background Noise')
plt.plot(E[:250],co[:250],label = 'Co_60')
plt.plot(E[:250],cs[:250],label = 'Cs_137')
plt.plot(E[:250],bi[:250],label = 'Bi_207')
plt.plot(E[:250],ra[:250],label = 'Ra_226')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.ylim(0,5000)
plt.legend()
#plt.title('Calibration of MCA')
plt.savefig('
)

'''
Calibration of MCA
Bins Vs. keV (pretty sure?)
'''
'''
Ore A Photo Peaks
Counts Vs. keV
'''
OA = dataset['Ore_A pure signal']

plt.plot(E[:150],OA[:150],label = 'Ore_A')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')

plt.legend()
#plt.title('Calibration of MCA')
plt.savefig('OREAphotopeaks.png')


'''
Comparison of Ore A’s Photo Peaks and ^226 Ra
Counts Vs. keV



Im Not sure how I want to approach the attenuation data yet but I know ill need these! Let me know if any of these don’t make sense and you can either email them or put them on your git hub. Thanks again Luke!



Thank you,
Andrew Reid

'''
Ra = dataset['Ra_226 pure signal']
OA = dataset['Ore_A pure signal']
E = dataset['Energy']

plt.plot(E[:100],OA[:100])
plt.plot(E[:100],Ra[:100])

OA_1 = OA[75:95]
OA_2 = OA[43:53]
OA_3 = OA[37:43]
OA_4 = OA[30:36]
OA_5 = OA[20:30]

Ra_1 = Ra[75:95]
Ra_2 = Ra[43:53]
Ra_3 = Ra[37:43]
Ra_4 = Ra[30:36]
Ra_5 = Ra[20:30]


popt_oa5, pcovt_oa5 = optimize.curve_fit(func_guass,E[20:30],OA_5[:],(325.,200.,0.002,395.))
err_oa5 = np.sqrt(np.diag(pcovt_oa5))
sigma_oa5 = np.sqrt(1/(2*popt_oa5[2]))
y_oa5 = func_guass(E, *popt_oa5)
plt.plot(E[20:30],OA_5)
plt.plot(E[10:36],y_oa5[10:36])
sigma_oa5

popt_oa4, pcovt_oa4 = optimize.curve_fit(func_guass,E[30:36],OA_4[:],(280.,200.,0.002,435.))
err_oa4 = np.sqrt(np.diag(pcovt_oa4))
sigma_oa4 = np.sqrt(1/(2*popt_oa4[2]))
y_oa4 = func_guass(E, *popt_oa4)
plt.plot(E[30:36],OA_4[:])
plt.plot(E[30:36],y_oa4[30:36])
sigma_oa4

popt_oa3, pcovt_oa3 = optimize.curve_fit(func_guass,E[37:43],OA_3[:],(320.,200.,0.002,350.))
err_oa3 = np.sqrt(np.diag(pcovt_oa3))
sigma_oa3 = np.sqrt(1/(2*popt_oa3[2]))
y_oa3 = func_guass(E, *popt_oa3)
plt.plot(E[37:43],OA_3[:])
plt.plot(E[35:44],y_oa3[35:44])
err_oa3
sigma_oa3
popt_oa3


popt_oa2, pcovt_oa2 = optimize.curve_fit(func_guass,E[43:53],OA_2[:],(200.,400.,0.002,400.))
err_oa2 = np.sqrt(np.diag(pcovt_oa2))
sigma_oa2 = np.sqrt(1/(2*popt_oa2[2]))
y_oa2 = func_guass(E, *popt_oa2)
plt.plot(E[43:53],OA_2[:])
plt.plot(E[40:54],y_oa2[40:54])
err_oa2
sigma_oa2
popt_oa2

popt_oa1, pcovt_oa1 = optimize.curve_fit(func_guass,E[75:95],OA_1[:],(50.,300.,0.002,620.))
err_oa1 = np.sqrt(np.diag(pcovt_oa1))
sigma_oa1 = np.sqrt(1/(2*popt_oa1[2]))
y_oa1 = func_guass(E, *popt_oa1)
plt.plot(E[75:95],OA_1[:])
plt.plot(E[70:100],y_oa1[70:100])
err_oa1
sigma_oa1
popt_oa1
