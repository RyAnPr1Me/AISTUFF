import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Stock info
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2025-01-01'

print(f"[*] Downloading {symbol} data from {start_date} to {end_date}...")
data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

# Check if data downloaded
if data.empty:
    raise ValueError("Stock data download failed!")

# Technical indicators (DIY style)
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
delta = data['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Target: Future weekly close
data['Future_Close'] = data['Close'].shift(-5)
data = data.dropna(subset=['Future_Close'])

# Weekly return and target class
data['Weekly_Return'] = (data['Future_Close'] - data['Close']) / data['Close']
data['Target'] = np.where(data['Weekly_Return'] > 0, 1, 0)

# Feature selection
features = ['Close', 'SMA_5', 'SMA_30', 'MACD', 'RSI']
data.dropna(subset=features, inplace=True)

# Normalize features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Create training data folder if it doesn't exist
output_dir = 'Training_Data'
os.makedirs(output_dir, exist_ok=True)

# Find next filename (start at data3.csv)
existing = [int(f[4:-4]) for f in os.listdir(output_dir) if f.startswith("data") and f.endswith(".csv") and f[4:-4].isdigit()]
next_index = max(existing) + 1 if existing else 3
filename = f"data{next_index}.csv"
filepath = os.path.join(output_dir, filename)

# Save it
data.to_csv(filepath, index=False)
print(f"[+] Dataset saved to {filepath}")
