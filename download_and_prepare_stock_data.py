# --- Ticker configuration section ---
TICKER = "AMZN"  # <--- Set your desired ticker here (overrides --symbol if not None)
# ------------------------------------

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

def compute_technical_indicators(df):
    # SMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    # Stochastic Oscillator
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-10)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
    # Lag features
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag5'] = df['Close'].shift(5)
    return df

def create_targets(df):
    # Future close price (5 days ahead)
    df['Future_Close'] = df['Close'].shift(-5)
    # Ensure both are Series, not DataFrames (fix for possible duplicate columns)
    if isinstance(df['Close'], pd.DataFrame):
        close = df['Close'].iloc[:, 0]
    else:
        close = df['Close']
    if isinstance(df['Future_Close'], pd.DataFrame):
        future_close = df['Future_Close'].iloc[:, 0]
    else:
        future_close = df['Future_Close']
    # Weekly return
    df['Weekly_Return'] = (future_close - close) / close
    # Target: 1 if up, 0 if down or unchanged
    df['label'] = (df['Weekly_Return'] > 0).astype(int)
    return df

def normalize_features(df, feature_cols):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled

def get_next_data_filename(folder):
    i = 3
    while True:
        fname = f"data{i}.csv"
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            return fpath
        i += 1

def add_text_column(df, symbol):
    """
    Add a 'text' column for compatibility with the data validator.
    Uses a template string with the date and symbol.
    """
    if 'text' not in df.columns:
        # Try to use 'Date' if present, else index
        if 'Date' in df.columns:
            date_col = df['Date'].astype(str)
        else:
            date_col = df.index.astype(str)
        # Ensure 'Close' and 'Weekly_Return' are Series, not DataFrames
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        weekly_return = df['Weekly_Return']
        if isinstance(weekly_return, pd.DataFrame):
            weekly_return = weekly_return.iloc[:, 0]
        # Compose a simple text headline for each row
        df['text'] = (
            symbol.upper() + " stock data for " + date_col +
            ". Close: " + close.round(2).astype(str) +
            ", Weekly return: " + weekly_return.round(4).astype(str)
        )
    return df

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Download and prepare stock data for training.")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='Training_Data', help='Output folder')
    args = parser.parse_args()

    # Use TICKER variable if set (not None or empty)
    symbol = TICKER if TICKER else args.symbol

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Downloading {symbol} data from {args.start} to {args.end}...")
    try:
        df = yf.download(symbol, start=args.start, end=args.end)
    except Exception as e:
        logging.error(f"Failed to download data: {e}")
        sys.exit(1)

    if df.empty:
        logging.error("No data downloaded. Check symbol and date range.")
        sys.exit(1)

    df = df.reset_index()

    # --- Fix for MultiIndex columns (happens with yfinance multi-ticker download) ---
    if isinstance(df.columns, pd.MultiIndex):
        ticker_level = df.columns.get_level_values(1)
        unique_tickers = list(set(ticker_level))
        if len(unique_tickers) == 1 and (unique_tickers[0] == '' or unique_tickers[0] == symbol):
            df.columns = [col[0] for col in df.columns]
        elif symbol in unique_tickers:
            df = df.xs(symbol, axis=1, level=1, drop_level=True)
            df = df.reset_index()
        else:
            logging.error(f"MultiIndex columns detected for multiple tickers, and '{symbol}' not found. Please use a single valid ticker.")
            sys.exit(1)
    # --- End fix ---

    # --- Fix: Ensure 'Date' column exists and is named correctly ---
    # After reset_index, the date column may be named 'index' or something else
    date_col_candidates = ['Date', 'date', 'Datetime', 'datetime', 'index']
    found_date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            found_date_col = c
            break
    if found_date_col and found_date_col != 'Date':
        df = df.rename(columns={found_date_col: 'Date'})
    elif not found_date_col:
        logging.error("No date column found after reset_index. Cannot proceed.")
        sys.exit(1)
    # --- End fix ---

    df = compute_technical_indicators(df)
    df = create_targets(df)

    # Select features and columns to keep
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_30', 'EMA_12', 'EMA_26', 'MACD',
        'RSI_14', 'BB_upper', 'BB_lower',
        'Stoch_%K', 'Stoch_%D',
        'Close_lag1', 'Close_lag5'
    ]
    keep_cols = ['Date'] + feature_cols + ['Future_Close', 'Weekly_Return', 'label']

    # Drop rows with missing values (from rolling calculations and lags)
    df = df[keep_cols].dropna().reset_index(drop=True)
    if df.empty:
        logging.error("No data left after dropping rows with missing values.")
        sys.exit(1)

    # Normalize features
    df_scaled = normalize_features(df, feature_cols)

    # Add a 'text' column for compatibility with the data validator
    df_scaled = add_text_column(df_scaled, symbol)

    # Reorder columns: text, features..., label
    ordered_cols = ['text'] + [c for c in df_scaled.columns if c not in ['text', 'label']] + ['label']
    df_scaled = df_scaled[ordered_cols]

    # Save to CSV
    out_path = get_next_data_filename(args.output_dir)
    try:
        df_scaled.to_csv(out_path, index=False)
        logging.info(f"Saved processed data to {out_path} ({len(df_scaled)} rows).")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")
        sys.exit(1)

    # --- New: Automatically format for ALBERT after saving ---
    try:
        from format_for_albert import format_dataset_for_albert
        albert_out = os.path.join(args.output_dir, f"albert_{os.path.basename(out_path)}")
        format_dataset_for_albert(out_path, albert_out)
        logging.info(f"Formatted for ALBERT: {albert_out}")
    except Exception as e:
        logging.warning(f"Could not format for ALBERT automatically: {e}")

if __name__ == "__main__":
    main()
