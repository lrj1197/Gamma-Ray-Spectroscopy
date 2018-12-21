import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import statsmodels.api as sm
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model
#import statsmodels.formula.api as sm
#import datetime
#from dateutil.parser import parse

#read the raw dataset in
data_raw = pd.read_csv('<name of dataset>.csv', skipinitialspace = True)

#show the first couple rows to the raw dataself.
data_raw.head()

#plot a hist of the raw data to get a look at the distribution, look for any erroneous points/outliers
plt.figure()
plt.hist(data_raw)
plt.title("Raw Data")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.show()

#look at a scatter plot of the data, look for any erroneous points/outliers
plt.figure()
plt.scatter(<data_raw_1>, <data_raw_2>)
plt.title("Raw Data")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.show()

#set the bounds on the data set for cleaning and repeat for all columns
data_clean[column] = data_raw[column][<lower_bound> <= data_raw[column] <= <upper_bound>]
data_clean.head()
data_clean.to_pickle("path") #CAUTION!!! Only read pickles that YOU generate!!!!! No Exceptions!

#now get the avg, std dev, max, min,
mean = pd.mean(data_clean[column])
std = pd.std(data_clean[column])
min = pd.min(data_clean[column])
max = pd.max(data_clean[column])
print("Mean: %f, Stand Dev: %f, Minimum: %f, Maximum: %f").format(mean, std, min, max)

#now generate plots using clean data and save fig spec dir
plt.figure()
plt.<plot type>(<data_raw_1>, <data_raw_2>) #hist, scatter, plot, box
plt.title("Data") #describe what the plot is
plt.xlabel("X label")
plt.ylabel("Y label")
plt.savefig("path-to-directory.png") #pick a path that you know you'll find it
plt.show()
