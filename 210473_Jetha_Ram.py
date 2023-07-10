#Library to install
import numpy as np
import pandas as pd
!pip install tensorflow
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array
def evaluate():
    """
    """
#input of data 
df = pd.read_csv('https://raw.githubusercontent.com/jetharam171/Test-files/main/sample_input_incomplete.csv')
#fill up empty space
df1=df.fillna(method='pad')
df1=df1.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
#data scaling
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 7
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
### Create the Stacked LSTM model
model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(7,1)))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(25))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
## Train data RMSE
math.sqrt(mean_squared_error(y_train,train_predict))
print(f'Mean Square Error Train data: {math.sqrt(mean_squared_error(y_train,train_predict)):.6f}')
### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))
print(f'Mean Square Error Test data: {math.sqrt(mean_squared_error(ytest,test_predict)):.6f}')    
# shift train predictions for plotting
look_back=7
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
#these function for plot of future predictions and curve plot
#plot baseline and predictions
#plt.plot(scaler.inverse_transform(df1))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()  ## green test
# demonstrate prediction for next 2 days
x_input=test_data[11:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=7
i=0
while(i<2):
    
    if(len(temp_input)>7):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
       # print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
       # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
day_new=np.arange(1,8)
day_pred=np.arange(8,10)
df3=df1.tolist()
df3.extend(lst_output)
df3=scaler.inverse_transform(df3).tolist()
## directional accuracy function
def dir_accuracy(df1, df3):
  """
  Calculates the direction accuracy of the predictions.

  Args:
    prices: The historical prices of the stock.
    predictions: The predicted directions of the stock.

  Returns:
    The direction accuracy of the predictions.
  """

  correct = 0
  total = len(df1)

  for i in range(total - 1):
    if (df1[i + 1] - df1[i]) * df3[i] > 0:
      correct += 1

  return correct*100 / total
print(f'Directional Accuracy: {dir_accuracy(df1,df3)}')
print(f'Closing Price of day after last day: {df3[-2]}')
print(f'Closing Price of day two after last day: {df3[-1]}')
if __name__== "__main__":
    evaluate()