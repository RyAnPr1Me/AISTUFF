import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ========== CONFIG ==========
SYMBOL = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'
OUTPUT_DIR = 'Training_Data'
FEATURES = ['Close', 'SMA_5', 'SMA_30', 'RSI', 'MACD', 'ATR']
TARGET_SHIFT_DAYS = 5

# ========== SETUP ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== DOWNLOAD DATA ==========
print(f"[*] Downloading {SYMBOL} data from {START_DATE} to {END_DATE}...")
data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)

if data.empty or len(data) < 50:
    raise ValueError("Not enough data fetched. Try another symbol or date range.")

# ========== CALCULATE INDICATORS ==========
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()

delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-10)
data['RSI'] = 100 - (100 / (1 + rs))

ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26

high_low = data['High'] - data['Low']
high_close = np.abs(data['High'] - data['Close'].shift())
low_close = np.abs(data['Low'] - data['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
data['ATR'] = tr.rolling(window=14).mean()

# ========== TARGET RETURN ==========
data['Future_Close'] = data['Close'].shift(-TARGET_SHIFT_DAYS)
data['Weekly_Return'] = (data['Future_Close'] - data['Close']) / data['Close']
data['Target'] = (data['Weekly_Return'] > 0).astype(int)

# ========== CLEAN & SCALE ==========
required_cols = FEATURES + ['Weekly_Return', 'Target']
data.dropna(subset=required_cols, inplace=True)

if data.empty:
    raise ValueError("All rows dropped during cleaning. Adjust your window sizes or date range.")

scaler = StandardScaler()
data[FEATURES] = scaler.fit_transform(data[FEATURES])

# ========== EXPORT ==========
existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('data') and f.endswith('.csv')]
existing_nums = [int(f[4:-4]) for f in existing_files if f[4:-4].isdigit()]
next_num = max([2] + existing_nums) + 1
file_path = os.path.join(OUTPUT_DIR, f'data{next_num}.csv')

data_to_save = data[FEATURES + ['Target']].copy()
data_to_save.to_csv(file_path, index=False)
print(f"[+] Dataset saved to {file_path} — it’s cleaner than your search history.")
