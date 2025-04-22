import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Fetch data for a stock symbol (e.g., 'AAPL')
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2025-01-01'
data = yf.download(symbol, start=start_date, end=end_date)

# Feature engineering: Adding technical indicators

# Simple Moving Averages
data['SMA_5'] = data['Close'].rolling(window=5).mean()  # 5-day simple moving average
data['SMA_30'] = data['Close'].rolling(window=30).mean()  # 30-day simple moving average

# Relative Strength Index (RSI)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()  # 12-day Exponential Moving Average
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()  # 26-day Exponential Moving Average
data['MACD'] = data['EMA_12'] - data['EMA_26']  # MACD line
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()  # Signal line

# ATR (Average True Range)
data['H-L'] = data['High'] - data['Low']
data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
data['ATR'] = data['TR'].rolling(window=14).mean()

# Target variable: Weekly price change
data['Weekly_Close'] = data['Close'].shift(-5)  # Close price 5 days ahead
data['Weekly_Return'] = (data['Weekly_Close'] - data['Close']) / data['Close']  # Percentage change

# Labeling the target (binary classification: up or down)
data['Target'] = np.where(data['Weekly_Return'] > 0, 1, 0)  # 1 = price goes up, 0 = price goes down

# Drop rows with missing values (caused by technical indicators and shifting)
data.dropna(inplace=True)

# Select features and scale them
features = ['Close', 'SMA_5', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'ATR']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Save the dataset to a CSV file
data.to_csv('Training_Data/data3.csv')

# Display the first few rows of the dataset
print(data.head())
