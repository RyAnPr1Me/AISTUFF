import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Config
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2025-01-01'
output_dir = 'Training_Data'
os.makedirs(output_dir, exist_ok=True)

# Download data
data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

# Technical indicators using just Pandas/NumPy
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
high_low = data['High'] - data['Low']
high_close = np.abs(data['High'] - data['Close'].shift())
low_close = np.abs(data['Low'] - data['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
data['ATR'] = tr.rolling(window=14).mean()

# Target: Weekly return
data['Weekly_Close'] = data['Close'].shift(-5)
data.dropna(subset=['Weekly_Close'], inplace=True)
data['Weekly_Return'] = (data['Weekly_Close'] - data['Close']) / data['Close']
data['Target'] = np.where(data['Weekly_Return'] > 0, 1, 0)

# Clean and scale
features = ['Close', 'SMA_5', 'SMA_30', 'RSI', 'MACD', 'ATR']
data.dropna(subset=features + ['Target'], inplace=True)
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Save to next available file (data3.csv onward)
existing_files = [f for f in os.listdir(output_dir) if f.startswith('data') and f.endswith('.csv')]
existing_nums = [int(f[4:-4]) for f in existing_files if f[4:-4].isdigit()]
next_num = max([2] + existing_nums) + 1
file_path = os.path.join(output_dir, f'data{next_num}.csv')
data.to_csv(file_path, index=False)

print(f"[+] Dataset saved to {file_path} — enjoy your AI stock overlord.")
