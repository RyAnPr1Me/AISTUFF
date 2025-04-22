import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Fetch data for a stock symbol (e.g., 'AAPL')
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2025-01-01'
data = yf.download(symbol, start=start_date, end=end_date)

# Feature engineering: Adding technical indicators
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()
data['RSI'] = data['Close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (np.sum(x[-14:] > x[-15:]) / 14))))  # simple RSI approximation
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['ATR'] = data['High'].rolling(window=14).mean() - data['Low'].rolling(window=14).mean()

# Target variable: Weekly price change
data['Weekly_Close'] = data['Close'].shift(-5)  # Close price 5 days ahead

# Ensure no NaN values for 'Weekly_Close'
data.dropna(subset=['Weekly_Close'], inplace=True)

# Calculate Weekly Return: Percentage change from current close to the weekly close
data['Weekly_Return'] = (data['Weekly_Close'] - data['Close']) / data['Close']  # Percentage change

# Labeling the target (binary classification: up or down)
data['Target'] = np.where(data['Weekly_Return'] > 0, 1, 0)  # 1 = price goes up, 0 = price goes down

# Drop rows with missing values (caused by technical indicators and shifting)
data.dropna(inplace=True)

# Select features and scale them
features = ['Close', 'SMA_5', 'SMA_30', 'RSI', 'MACD', 'ATR']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Ensure the "Training_Data" folder exists
if not os.path.exists('Training_Data'):
    os.makedirs('Training_Data')

# Save the dataset to a CSV file in the "Training_Data" folder
file_name = f'Training_Data/data{len(os.listdir("Training_Data")) + 3}.csv'  # Start at data3.csv
data.to_csv(file_name)

# Display the first few rows of the dataset
print(data.head())
