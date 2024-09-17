import pandas as pd
import pytz
import yfinance as yf
from time import time
import datetime
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("hf://datasets/Ammok/apple_stock_price_from_1980-2021/AAPL.csv")

# print(df.shape)
# print(df.head())
# print(df.isnull().sum())
# print(df.isna().any())
# print(df.info())
# print(df.describe())

features = ['Open', 'High', 'Low']
target = 'Close'

x=df[features]
y=df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



# print(x_train.shape,x_test.shape)


df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

train = df.loc[df.index < '2010-01-01']
test = df.loc[df.index >= '2009-12-31']
train_ = train.loc[train.index > '2009-01-01']

# fig, ax = plt.subplots(figsize=(15, 5))
# train.plot(ax=ax, label='Training Set')
# test.plot(ax=ax, label='Test Set')
# ax.axvline(pd.to_datetime('2021-01-01'), color='black')
# ax.legend(['Train', 'Test'])
# plt.show()

regressor= SGDRegressor()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

#MODEL PERFORMANCE ON TRAIN DATA
    # predict = regressor.predict(x_train)
    # r2 = r2_score(y_train, predict)
    # mse = mean_squared_error(y_train, predict)
    # rmse = mean_squared_error(y_train, predict, squared=False)
    # mae = mean_absolute_error(y_train, predict)
    # print("R2 Score:", r2)
    # print("Mean Squared Error:", mse)
    # print("Root Mean Squared Error:", rmse)
    # print("Mean Absolute Error:", mae)

#MODEL PERFORMANCE ON TEST DATA
    # r2 = r2_score(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # mae = mean_absolute_error(y_test, y_pred)
    # print("R2 Score:", r2)
    # print("Mean Squared Error:", mse)
    # print("Root Mean Squared Error:", rmse)
    # print("Mean Absolute Error:", mae)

tickerSymbol = 'AAPL'
data = yf.Ticker(tickerSymbol)

end_date = datetime.date.today() - datetime.timedelta(days=1)
start_date = end_date - datetime.timedelta(days=20*365)
day = datetime.date.today()
stock = data.history(start=start_date, end=end_date)

def next_day_price(stock, day):
    x = stock[['Open', 'High', 'Low', 'Volume']]
    y = stock['Close']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
    model = SGDRegressor()
    model.fit(x_train, y_train)

    last_day_scaled = scaler.transform([stock.iloc[-1][['Open', 'High', 'Low', 'Volume']]])
    predicted_price = model.predict(last_day_scaled)
    print(f"Predicted price for {day}:", predicted_price[0])

next_day_price(stock, day)


