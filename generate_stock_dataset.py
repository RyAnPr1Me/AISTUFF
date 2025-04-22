import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Stock info
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2025-01-01'

# Download stock data
print(f"[*] Downloading {symbol} data from {start_date} to {end_date}...")
data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

# Check if data is empty
if data.empty:
    raise ValueError("Stock data download failed!")

# Calculate Technical Indicators
data['SMA_5'] = data['Close'].rolling(window=5).mean()  # 5-day Simple Moving Average
data['SMA_30'] = data['Close'].rolling(window=30).mean()  # 30-day Simple Moving Average

# Exponential Moving Averages for MACD
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA
data['MACD'] = data['EMA_12'] - data['EMA_26']  # MACD = EMA12 - EMA26

# Relative Strength Index (RSI)
delta = data['Close'].diff()  # Difference between current and previous day's close
gain = np.where(delta > 0, delta, 0)  # Positive gains
loss = np.where(delta < 0, -delta, 0)  # Negative losses

# Convert gain and loss arrays to 1D
gain = gain.flatten()  # Ensure gain is 1D
loss = loss.flatten()  # Ensure loss is 1D

# Check if gain and loss arrays have the expected shape
if gain.shape[0] != len(data) or loss.shape[0] != len(data):
    raise ValueError("Shape of gain/loss arrays does not match the length of the data!")

# Calculate Average Gain and Loss (Rolling window of 14 days)
avg_gain = pd.Series(gain).rolling(window=14).mean()  # 14-day average gain
avg_loss = pd.Series(loss).rolling(window=14).mean()  # 14-day average loss
rs = avg_gain / avg_loss  # Relative strength
data['RSI'] = 100 - (100 / (1 + rs))  # RSI formula

# Target: Future weekly close
data['Future_Close'] = data['Close'].shift(-5)  # Shift 5 days to predict future close
data = data.dropna(subset=['Future_Close'])  # Drop rows where Future_Close is NaN

# Weekly return and target classification
data['Weekly_Return'] = (data['Future_Close'] - data['Close']) / data['Close']  # Weekly return as percentage change
data['Target'] = np.where(data['Weekly_Return'] > 0, 1, 0)  # 1 if price goes up, 0 if it goes down

# Feature selection
features = ['Close', 'SMA_5', 'SMA_30', 'MACD', 'RSI']
data.dropna(subset=features, inplace=True)  # Drop rows with NaN in any of the selected features

# Normalize features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])  # Scale the selected features

# Check if the feature scaling worked correctly
if data[features].isnull().any().any():
    raise ValueError("Feature scaling produced NaN values!")

# Create training data folder if it doesn't exist
output_dir = 'Training_Data'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if not exists

# Find the next available filename (starting from data3.csv)
existing = [int(f[4:-4]) for f in os.listdir(output_dir) if f.startswith("data") and f.endswith(".csv") and f[4:-4].isdigit()]
next_index = max(existing) + 1 if existing else 3  # Start at data3.csv
filename = f"data{next_index}.csv"  # Create new filename
filepath = os.path.join(output_dir, filename)

# Save the dataset to a CSV file
data.to_csv(filepath, index=False)  # Save without index
print(f"[+] Dataset saved to {filepath}")  # Confirmation message
