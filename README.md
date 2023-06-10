# Machine-Learning-using-ARIMA-model

This project is based on forecasting Temperature using past 7 days hourly temperature data set.
Data set is on real time data a person have to take data for past 7 days then apply this algorithm.
website for dataset :-https://www.timeanddate.com/weather/india/kolkata/historic

ARIMA MODEL :- This is a basic time series forecasting model which uses a auto regression technique to forecast future data. ARIMA(Auto Regression Integration Moving Average) in this project we will use pmdarima python package.

Code editor used :- PYCHARM Ide.

DESCRIPTION :-

1)Data files used :-

  weather_data.csv -> used for storing past 7days data 
  
  predicted.csv -> used for storing the predicted data used for finding RMSE(Root mean square error)
  
  forecasted.csv -> used for storing the forecasted data

2)Exploring dataset :-
  
  first 5 data in the data set ->
  
  ![image](https://github.com/rikuzavi/Machine-Learning-using-ARIMA-model/assets/96969805/f7d2f2ab-43ce-4ef3-94c9-cbdba968d4b1)
  
  last 5 data in the data set ->
  
  ![image](https://github.com/rikuzavi/Machine-Learning-using-ARIMA-model/assets/96969805/edb21847-6848-4257-9c9b-ae0f95d1a2f9)
  
  update temperature while using this.

3)Code Explaination :-

  o Imports ->
  
  import math
  
  import pmdarima as pm
  
  import pandas as pd
  
  import matplotlib.pyplot as plt
  
  from sklearn.metrics import mean_squared_error
    
  o Preparing Data set ->
  
  df = pd.read_csv('weather_data.csv', parse_dates=['datetime'], index_col='datetime')
  
  o Train and test set splitting for prediction using 6 days data ->
  
  train = df[:144]
  
  test = df[144:]
  
  y=train['temp']
  
  o Prepare model and get prediction ->
  
  model = pm.auto_arima(y)
  
  prediction = model.predict(n_periods=24)
  
  temperatures = prediction.values
  
  o Preparing the dataset and saving the prediction data in a csv file ->
  
  date_p = prediction.index
  
  data_frame_p = []
  
  for T,temp in zip(date_p,temperatures):
  
      data_frame_p.append(f"{T},{temp:.2f}")
      
  s_p='datetime,temp\n'+"\n".join(data_frame_p)
  
  with open('predicted.csv', 'w') as f:
  
      f.write(s_p)
      
  o Finding the rmse using the predicted data ->
  
  train_df = prediction.values
  
  test_df = test['temp']
  
  mse = round(math.sqrt(mean_squared_error(test_df, train_df)),2)
  
  print('Root mean square error : ', mse)
  
  OUTPUT ->
  
  ![image](https://github.com/rikuzavi/Machine-Learning-using-ARIMA-model/assets/96969805/b38025a1-c9eb-4da1-a87a-9f4b49c94bd4)
  
  o Using the same technique for finding the forecast but this time using the full 7 days data for forecast,full dataset used ->
  
  dt= df[:]
  
  y_f= dt['temp']
  
  model_f = pm.auto_arima(y_f)
  
  forecast = model_f.predict(n_periods=24)
  
  temperature_f = forecast.values
  
  date= forecast.index
  
  data_frame=[]
  
  for T,temp in zip(date,temperature_f):
  
      data_frame.append(f"{T},{temp:.2f}")
  
  s='datetime,temp\n'+"\n".join(data_frame)
  
  with open('forecasted.csv', 'w') as f:
      
      f.write(s)
      
  o Printing out the forecast ->
  
  df_pred = pd.read_csv('predicted.csv', parse_dates=['datetime'], index_col='datetime')
  
  df_fore = pd.read_csv('forecasted.csv', parse_dates=['datetime'], index_col='datetime')
  
  print('Next 24 hr FORECATED Temperature : \n',df_fore)
  
  OUTPUT ->
  
  ![image](https://github.com/rikuzavi/Machine-Learning-using-ARIMA-model/assets/96969805/355442d4-598c-4634-a060-a2c795bcd63b)
  
  o Plotting Graph ->
  
  fig, ax = plt.subplots(figsize=(14, 8))
  
  ax.plot(df, color='#1f76b4', label='Actual data')
  
  ax.plot(df_pred, color='orange', label='predicted Data')
  
  ax.plot(df_fore, color='orange', label='forecasted Data')
  
  ax.legend(loc='upper left')
  
  plt.title('Temperature RECENT VS FORECAST')
  
  plt.xlabel('Date and Time')
  
  plt.ylabel('Temperature (C)')
  
  plt.show()
  
  OUTPUT ->
  
  ![image](https://github.com/rikuzavi/Machine-Learning-using-ARIMA-model/assets/96969805/4e74e886-baad-4c71-8f7d-ea33eeb66373)

4)Future Work :- 
  
  This time I have used a very simple model ARIMA model later I am trying to use RNN and LSTM for forecasting and also trying to use 3 data for forecasting the     tempereature that will be temperature, humidity, pressure 
  
  
  
