# Importing required libraries
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Loading the model
model_path = r'C:\Python\Stock\Stock Predictions Model.keras'
stock_model = load_model(model_path)

# Setting up the Streamlit UI
st.header('Stock Market Predictor')

# Input for stock symbol
symbol = st.text_input('Enter Stock Symbol', 'GOOG')

# Date range for stock data
date_range_start = '2012-01-01'
date_range_end = '2022-12-31'

# Downloading stock data
stock_data = yf.download(symbol, date_range_start, date_range_end)

# Displaying stock data
st.subheader('Stock Data')
st.write(stock_data)

# Splitting data into training and test sets
split_ratio = 0.80
split_index = int(len(stock_data) * split_ratio)
training_data = pd.DataFrame(stock_data['Close'][:split_index])
testing_data = pd.DataFrame(stock_data['Close'][split_index:])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
last_100_days = training_data.tail(100)
testing_data_combined = pd.concat([last_100_days, testing_data], ignore_index=True)
scaled_testing_data = scaler.fit_transform(testing_data_combined)

# Function to plot stock data with moving averages
def plot_stock_with_moving_averages(data, *ma_periods):
    fig = plt.figure(figsize=(8, 6))
    for ma_period in ma_periods:
        plt.plot(data.rolling(ma_period).mean(), label=f'MA{ma_period}')
    plt.plot(data, 'g', label='Closing Price')
    plt.legend()
    st.pyplot(fig)

# Plotting price with different moving averages
st.subheader('Price vs MA50')
plot_stock_with_moving_averages(stock_data['Close'], 50)

st.subheader('Price vs MA50 vs MA100')
plot_stock_with_moving_averages(stock_data['Close'], 50, 100)

st.subheader('Price vs MA100 vs MA200')
plot_stock_with_moving_averages(stock_data['Close'], 100, 200)

# Preparing test data for model prediction
x_test, y_test = [], []
for i in range(100, len(scaled_testing_data)):
    x_test.append(scaled_testing_data[i-100:i])
    y_test.append(scaled_testing_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
predictions = stock_model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
predictions_scaled = predictions * scale_factor
y_test_scaled = y_test * scale_factor

# Plotting original vs predicted prices
st.subheader('Original Price vs Predicted Price')
fig_predictions = plt.figure(figsize=(8, 6))
plt.plot(predictions_scaled, 'r', label='Predicted Price')
plt.plot(y_test_scaled, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig_predictions)
