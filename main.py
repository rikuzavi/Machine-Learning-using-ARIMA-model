

# All imports
import math
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def ARIMA_MODEL():
    # preparing the data set
    df = pd.read_csv('weather_data.csv', parse_dates=['datetime'], index_col='datetime')
    # train and test set split
    train = df[:144]
    test = df[144:]
    y=train['temp']
    #preparing the model and get prediction
    model = pm.auto_arima(y)
    prediction = model.predict(n_periods=24)
    temperatures = prediction.values
    #preparing data set and storing it in a csv file
    date_p = prediction.index
    data_frame_p = []
    for T,temp in zip(date_p,temperatures):
        data_frame_p.append(f"{T},{temp:.2f}")
    s_p='datetime,temp\n'+"\n".join(data_frame_p)
    with open('predicted.csv', 'w') as f:
        f.write(s_p)
    #finding out RMSE
    train_df = prediction.values
    test_df = test['temp']
    mse = round(math.sqrt(mean_squared_error(test_df, train_df)),2)
    print('Root mean square error : ', mse)

    #using same technique to forecast future temperature, this time using full data set to train
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
    df_pred = pd.read_csv('predicted.csv', parse_dates=['datetime'], index_col='datetime')
    df_fore = pd.read_csv('forecasted.csv', parse_dates=['datetime'], index_col='datetime')
    print('Next 24 hr FORECATED Temperature : \n',df_fore)

    #ploting the graph

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df, color='#1f76b4', label='Actual data')
    ax.plot(df_pred, color='orange', label='predicted Data')
    ax.plot(df_fore, color='orange', label='forecasted Data')
    ax.legend(loc='upper left')
    plt.title('Temperature RECENT VS FORECAST')
    plt.xlabel('Date and Time')
    plt.ylabel('Temperature (C)')
    plt.show()


ARIMA_MODEL()