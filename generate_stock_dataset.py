import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Create the Training_Data directory if it doesn't exist
os.makedirs("Training_Data", exist_ok=True)

# Get the next dataset file number
existing_files = [f for f in os.listdir("Training_Data") if f.startswith("data") and f.endswith(".csv")]
existing_numbers = [int(f[4:-4]) for f in existing_files if f[4:-4].isdigit()]
next_num = max(existing_numbers, default=2) + 1  # Start at data3.csv

# Fetch stock data
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2025-01-01'
data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)

# Check for required columns
required_columns = ['Close', 'High', 'Low']
missing = [col for col in required_columns if col not in data.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Feature engineering (basic, without talib)
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()
data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean() / data['Close'].pct_change().rolling(14).std()))
data['ATR'] = (data['High'] - data['Low']).rolling(window=14).mean()

# MACD
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26

# Create the target: will price go up in the next week?
data['Weekly_Close'] = data['Close'].shift(-5)
data['Weekly_Return'] = (data['Weekly_Close'] - data['Close']) / data['Close']
data['Target'] = np.where(data['Weekly_Return'] > 0, 1, 0)

# Drop rows with NaNs
data.dropna(inplace=True)

# Scale features
features = ['Close', 'SMA_5', 'SMA_30', 'RSI', 'MACD', 'ATR']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Save the cleaned, feature-enriched dataset
output_file = f"Training_Data/data{next_num}.csv"
data.to_csv(output_file, index=False)

print(f"🔥 Dataset saved to {output_file}")
