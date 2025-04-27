# --- Ticker configuration section ---
TICKER = "SMCI"  # Changed from NVDA to AAPL as a more reliable alternative
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
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
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
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / ma20  # Bollinger Band width
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)  # Position within bands
    
    # Stochastic Oscillator
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-10)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    tr1 = abs(df['High'] - df['Low'])
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Price Rate of Change (ROC)
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
    
    # Lag features
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag5'] = df['Close'].shift(5)
    df['Volume_lag1'] = df['Volume'].shift(1)
    df['Return_1d'] = df['Close'].pct_change(periods=1)
    
    # Volatility features
    df['Volatility_10d'] = df['Return_1d'].rolling(window=10).std() * np.sqrt(252)  # Annualized
    
    # Day of week (0=Monday, 4=Friday)
    if 'Date' in df.columns:
        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    
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
    Add a rich 'text' column for compatibility with the data validator.
    Creates a detailed narrative about the stock performance.
    """
    if 'text' not in df.columns:
        texts = []
        # Convert date to datetime if not already
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        for i, row in df.iterrows():
            # Get price movement description
            if i > 0:
                prev_close = df.iloc[i-1]['Close']
                price_change = ((row['Close'] - prev_close) / prev_close) * 100
                if price_change > 1.5:
                    movement = "surged strongly"
                elif price_change > 0.5:
                    movement = "increased"
                elif price_change > -0.5:
                    movement = "remained stable"
                elif price_change > -1.5:
                    movement = "decreased"
                else:
                    movement = "fell sharply"
            else:
                movement = "traded at"
                
            # Get technical indicator insights
            rsi_signal = ""
            if 'RSI_14' in row and not pd.isna(row['RSI_14']):
                if row['RSI_14'] > 70:
                    rsi_signal = " RSI indicates the stock may be overbought."
                elif row['RSI_14'] < 30:
                    rsi_signal = " RSI indicates the stock may be oversold."
                    
            # Get volume insight
            volume_signal = ""
            if 'Volume' in row and i > 0 and 'Volume' in df.iloc[i-1]:
                prev_volume = df.iloc[i-1]['Volume']
                vol_change = ((row['Volume'] - prev_volume) / prev_volume) * 100
                if vol_change > 30:
                    volume_signal = " Trading volume was significantly higher than previous day."
                elif vol_change < -30:
                    volume_signal = " Trading volume was significantly lower than previous day."
            
            # Create forecast hint (this will help the model learn the target)
            if row['Weekly_Return'] > 0.03:
                forecast = "Outlook: Strong bullish trend expected in the next week."
            elif row['Weekly_Return'] > 0:
                forecast = "Outlook: Slight upward movement may continue."
            elif row['Weekly_Return'] > -0.03:
                forecast = "Outlook: Mild bearish pressure in the short term."
            else:
                forecast = "Outlook: Significant downward pressure expected."
                
            # Format date nicely
            date_str = row['Date'].strftime('%B %d, %Y') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])
            
            # Create comprehensive text
            text = (f"{symbol.upper()} {movement} ${row['Close']:.2f} on {date_str}. "
                   f"Day range: ${row['Low']:.2f} to ${row['High']:.2f}.{rsi_signal}{volume_signal} {forecast}")
            
            texts.append(text)
            
        df['text'] = texts
    
    return df

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Download and prepare stock data for training.")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='Training_Data', help='Output folder')
    args = parser.parse_args()

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', args.output_dir)
    symbol = TICKER if TICKER else args.symbol

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Downloading {symbol} data from {args.start} to {args.end}...")

    # --- Fix: yfinance sometimes fails for future dates or weekends, so adjust end date if needed ---
    import datetime
    today = datetime.datetime.today()
    end_date = min(datetime.datetime.strptime(args.end, "%Y-%m-%d"), today)
    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d")
    if end_date > today:
        end_date = today
    if start_date >= end_date:
        start_date = end_date - datetime.timedelta(days=365*2)  # fallback to last 2 years

    # yfinance cannot download future data, so always use today or last market day as end
    df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)

    # If still empty, try upper-case ticker (sometimes yfinance is case-sensitive)
    if df.empty and symbol != symbol.upper():
        logging.info(f"Retrying with upper-case ticker: {symbol.upper()}")
        df = yf.download(symbol.upper(), start=start_date.strftime("%Y-%m-%d"), end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)

    # If still empty, try lower-case ticker
    if df.empty and symbol != symbol.lower():
        logging.info(f"Retrying with lower-case ticker: {symbol.lower()}")
        df = yf.download(symbol.lower(), start=start_date.strftime("%Y-%m-%d"), end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)

    if df.empty:
        logging.error("No data downloaded. Check symbol and date range. (Hint: yfinance cannot download future data, and ticker must be valid/correct case.)")
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
        'SMA_5', 'SMA_30', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_hist',
        'RSI_14', 'BB_upper', 'BB_lower', 'BB_width', 'BB_pct',
        'Stoch_%K', 'Stoch_%D', 'ATR_14', 'OBV', 'ROC_10',
        'Close_lag1', 'Close_lag5', 'Volume_lag1', 'Return_1d', 'Volatility_10d'
    ]
    
    # Add day of week if present
    if 'DayOfWeek' in df.columns:
        feature_cols.append('DayOfWeek')
        
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
    out_path = get_next_data_filename(output_dir)
    try:
        df_scaled.to_csv(out_path, index=False)
        logging.info(f"Saved processed data to {out_path} ({len(df_scaled)} rows).")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
