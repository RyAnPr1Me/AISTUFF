import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import talib as ta

#========================================================================
# Script: generate_stock_dataset.py
# Purpose: Download historical stock data for multiple tickers
#          Generate advanced features and weekly movement labels
#          Save per-ticker CSVs named data3.csv, data4.csv, ... into Training_Data
# Usage: python generate_stock_dataset.py
# Dependencies:
#   pip install yfinance pandas numpy scikit-learn TA-Lib
#========================================================================

# Configuration
TICKERS = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'JNJ'
    # extend this list as needed
]
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'
OUTPUT_DIR = 'Training_Data'
START_INDEX = 3  # filenames will start at data3.csv

# Feature windows
LAG_DAYS = [1, 2, 3, 5, 7]
ROLLING_WINDOWS = [5, 10, 20]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterate tickers and process individually
for idx, ticker in enumerate(TICKERS, start=START_INDEX):
    print(f"Processing {ticker}, writing to data{idx}.csv...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    if df.empty:
        print(f"Warning: No data for {ticker}")
        continue

    # Price-based features
    df['Return'] = df['Close'].pct_change()
    for lag in LAG_DAYS:
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)

    # Rolling volatility
    for w in ROLLING_WINDOWS:
        df[f'Volatility_{w}'] = df['Return'].rolling(window=w).std()

    # Moving averages and EMA
    df['SMA_5'] = ta.SMA(df['Close'], timeperiod=5)
    df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
    df['EMA_5'] = ta.EMA(df['Close'], timeperiod=5)
    df['EMA_20'] = ta.EMA(df['Close'], timeperiod=20)

    # Bollinger Bands
    upper, mid, lower = ta.BBANDS(df['Close'], timeperiod=20)
    df['BB_upper'] = upper
    df['BB_mid'] = mid
    df['BB_lower'] = lower

    # Technical indicators
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    macd, macd_signal, _ = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])

    # Date features
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Week'] = df.index.isocalendar().week

    # Weekly target label
    df['Weekly_Close'] = df['Close'].shift(-5)
    df['Weekly_Return'] = (df['Weekly_Close'] - df['Close']) / df['Close']
    df['Target'] = np.where(df['Weekly_Return'] > 0, 1, 0)

    # Drop rows with NaNs
    df.dropna(inplace=True)

    # Scale numeric features
    feature_cols = [col for col in df.columns
                    if col.startswith(('Return_Lag', 'Volatility', 'SMA', 'EMA', 'BB_', 'RSI', 'MACD', 'ATR', 'OBV'))]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Add ticker column
    df['Ticker'] = ticker

    # Select columns to save
    save_cols = ['Ticker'] + feature_cols + ['DayOfWeek', 'Month', 'Week', 'Target']
    output_path = os.path.join(OUTPUT_DIR, f"data{idx}.csv")
    df[save_cols].to_csv(output_path, index=False)

print("All datasets generated in Training_Data folder.")
