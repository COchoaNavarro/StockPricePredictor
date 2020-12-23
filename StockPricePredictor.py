# This program uses Long Short Term Memory (LSTM) to predict the closing stock price of Tesla
# using the last 60 day stock price

#Importing Libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock quote from yahoo
dataf = web.DataReader('TSLA', data_source='yahoo', start='2015-01-01', end='2020-12-15')
#Show the data
dataf

#Collect the number of rows and columns from the data set
dataf.shape

#Display the closing price history specifically
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(dataf['Close'])
plt.xlabel('Date (Years)', fontsize=16)
plt.ylabel('Close Price $ (USD)',fontsize=16)
plt.show()

#Creatign a new dataf w/ only close
data = dataf.filter(['Close'])
#Convert dataf to numpy array
dataset = data.values
#Collect number of rows to train model on
training_data_len = math.ceil( len(dataset) * .8)

#Print
training_data_len

#Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scldata = scaler.fit_transform(dataset) #computes min and max values to be used for scaline and transforms off these 2 values
#Print
scldata

#Create a training data set
#1. Scaled training set data
train_data = scldata[0:training_data_len , :]

#Split into two data sets: x_train and y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
#Convert x_train and y_train to numPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#Build the LSTM model
mdl = Sequential()
mdl.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
mdl.add(LSTM(50, return_sequences=False))
mdl.add(Dense(25))
mdl.add(Dense(1))

#Compile model
mdl.compile(optimizer='adam', loss='mean_squared_error')

#Train model
mdl.fit(x_train, y_train, batch_size=1, epochs=1)

#Create testing data set
tdata = scldata[training_data_len - 60: , :]

#Create data sets xtest and ytest
xtest = []
ytest = dataset[training_data_len:, :]

for i in range(60, len(tdata)):
    xtest.append(tdata[i-60:i, 0])

#Convert to a numpy array
xtest = np.array(xtest)

#Reshape to 3D shape
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

#Get models predicted price value
prediction = mdl.predict(xtest)
prediction = scaler.inverse_transform(prediction)

#Collect the root mean squart error (RMSE)
rmse = np.sqrt(np.mean(prediction - ytest) **2)
#Show value
rmse

#Plot data
t = data[:training_data_len]
v = data[training_data_len:]
v['Prediction'] = prediction

#Show data
plt.figure(figsize=(16,8))
plt.title('TSLA Model')
plt.xlabel('Date (Years)', fontsize=16)
plt.ylabel('Close Price $ (USD)', fontsize=16)
plt.plot(t['Close'])
plt.plot(v[['Close', 'Prediction']])
plt.legend(['Train', 'Value', 'Prediction'], loc='lower right')
plt.show()

#Show acutal price vs predicted
v

#Predict next day closing stock price
#get quote
quote = web.DataReader('TSLA', data_source='yahoo', start='2015-01-01', end='2020-12-15')

#create new data frame
dataf2 = quote.filter(['Close'])

#Get last 60 day closing price values then convert dataframe to a array
last60d = dataf2[-60:].values
last60d_scaled = scaler.transform(last60d)

#Create empty list
Xtest = []
#append past 60 days
Xtest.append(last60d_scaled)

#convert xtest dataset to numpy array
Xtest = np.array(Xtest)
#reshape
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

#Get predicted scaled price
price_predicted = mdl.predict(Xtest)
#undo scaling
price_predicted = scaler.inverse_transform(price_predicted)

#show predicted price for next day
print(price_predicted)

#Show actual clsing price for enxt day
#get quote
actual = web.DataReader('TSLA', data_source='yahoo', start='2020-12-16', end='2020-12-16')
print(actual['Close'])
