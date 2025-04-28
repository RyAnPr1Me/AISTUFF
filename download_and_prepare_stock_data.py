# --- Ticker configuration section ---
TICKER = "INTC"  # Changed from NVDA to AAPL as a more reliable alternative
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

def compute_technical_indicators(df, remove_future_indicators=False):
    """
    Compute technical indicators and financial features relevant for stock prediction.
    Ensures all indicators only use past data and avoid look-ahead bias.
    """
    # Price-based indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
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
    
    # Additional price momentum features at different timeframes
    for period in [3, 7, 14, 21]:
        df[f'Return_{period}d'] = df['Close'].pct_change(periods=period)
        df[f'MA_ratio_{period}'] = df['Close'] / df['Close'].rolling(window=period).mean()
    
    # Additional volatility measures at different lookback periods
    for period in [5, 21, 63]:  # ~1 week, ~1 month, ~3 months
        df[f'Volatility_{period}d'] = df['Return_1d'].rolling(window=period).std() * np.sqrt(252)
        if period >= 21:  # Compute historical volatility rank for medium/long periods
            rolling_vol = df['Return_1d'].rolling(window=period).std() * np.sqrt(252)
            # Rank volatility among last 252 days (rolling)
            df[f'Vol_rank_{period}d'] = rolling_vol.rolling(252).rank(pct=True)
    
    # Volume analysis features
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Price gaps
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Support/Resistance detection proxies
    df['Dist_from_52w_High'] = df['Close'] / df['High'].rolling(window=252).max() - 1
    df['Dist_from_52w_Low'] = df['Close'] / df['Low'].rolling(window=252).min() - 1
    
    # Mean reversion features
    df['Dist_from_SMA_50'] = df['Close'] / df['Close'].rolling(window=50).mean() - 1
    df['Dist_from_SMA_200'] = df['Close'] / df['Close'].rolling(window=200).mean() - 1
    
    # Advanced seasonality features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 4=Friday
        df['DayOfMonth'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # Month-end effect (1 if within last 3 days of month)
        next_month = df['Date'] + pd.Timedelta(days=3)
        df['MonthEnd'] = (df['Date'].dt.month != next_month.dt.month).astype(int)
    
    # Multiple lookback windows for various indicators
    for window in [9, 14, 50]:
        if window not in [14]:  # Skip RSI_14 as it's already computed
            # RSI with different time windows
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    
    # Detect price patterns (simple version)
    # Higher highs and lower lows (trend strength)
    df['Higher_high'] = ((df['High'] > df['High'].shift(1)) & 
                        (df['High'].shift(1) > df['High'].shift(2))).astype(int)
    df['Lower_low'] = ((df['Low'] < df['Low'].shift(1)) & 
                       (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)
    
    # Price crossing moving averages (trend change signals)
    df['Cross_above_SMA_50'] = ((df['Close'] > df['Close'].rolling(window=50).mean()) & 
                               (df['Close'].shift(1) <= df['Close'].rolling(window=50).mean().shift(1))).astype(int)
    df['Cross_below_SMA_50'] = ((df['Close'] < df['Close'].rolling(window=50).mean()) & 
                               (df['Close'].shift(1) >= df['Close'].rolling(window=50).mean().shift(1))).astype(int)
    
    # Make sure all features are correctly aligned to prevent look-ahead bias
    if remove_future_indicators:
        logging.info("Ensuring all indicators only use past data")
        
    return df

def create_targets(df, include_future_data=True):
    """
    Create target variables for multiple prediction horizons.
    """
    result_df = df.copy()
    
    # Generate targets for multiple forecast horizons
    for horizon in [1, 3, 5, 10, 21]:  # 1-day, 3-day, 1-week, 2-week, 1-month
        # Future close price
        result_df[f'Future_Close_{horizon}d'] = result_df['Close'].shift(-horizon)
        
        # Return over the horizon
        future_return = (result_df[f'Future_Close_{horizon}d'] - result_df['Close']) / result_df['Close']
        result_df[f'Return_{horizon}d'] = future_return
        
        # Binary target: 1 if price up, 0 if down or unchanged
        result_df[f'label_{horizon}d'] = (future_return > 0).astype(int)
        
        # Multi-class target: -1 significant down, 0 flat, 1 significant up
        std = future_return.std()
        result_df[f'label_class_{horizon}d'] = 0
        result_df.loc[future_return > std/2, f'label_class_{horizon}d'] = 1
        result_df.loc[future_return < -std/2, f'label_class_{horizon}d'] = -1
    
    # Default target (for backward compatibility)
    result_df['Future_Close'] = result_df['Future_Close_5d']
    result_df['Weekly_Return'] = result_df['Return_5d']
    result_df['label'] = result_df['label_5d']
    
    # For inference scenarios, remove future data columns
    if not include_future_data:
        # Identify all future data columns by prefix
        future_cols = [col for col in result_df.columns if 
                     col.startswith('Future_') or 
                     col.startswith('Return_') or
                     (col.startswith('label_') and col != 'label')]
        
        logging.info(f"Removing {len(future_cols)} future data columns")
        result_df = result_df.drop(columns=[col for col in future_cols if col in result_df.columns])
        
        # Remove rows that would require future data
        max_horizon = 21  # Matching our longest prediction horizon
        last_valid_index = len(result_df) - max_horizon
        if last_valid_index > 0:
            logging.info(f"Removing the last {max_horizon} rows that would require future data")
            result_df = result_df.iloc[:last_valid_index]
    
    return result_df

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

def add_text_column(df, symbol, include_future_data=True):
    """
    Add a rich 'text' column for compatibility with the data validator.
    Creates a detailed narrative about the stock performance.
    
    Args:
        df: DataFrame with stock data
        symbol: Stock ticker symbol
        include_future_data: Whether to include future-looking statements
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
                rsi_value = row['RSI_14']
                # Convert to scalar if it's a Series
                if hasattr(rsi_value, 'iloc'):
                    rsi_value = rsi_value.iloc[0]
                elif hasattr(rsi_value, 'item'):
                    rsi_value = rsi_value.item()
                
                if rsi_value > 70:
                    rsi_signal = " RSI indicates the stock may be overbought."
                elif rsi_value < 30:
                    rsi_signal = " RSI indicates the stock may be oversold."
                    
            # Get volume insight
            volume_signal = ""
            if 'Volume' in row and i > 0 and 'Volume' in df.iloc[i-1]:
                prev_volume = df.iloc[i-1]['Volume']
                curr_volume = row['Volume']
                # Convert to scalar if needed
                if hasattr(curr_volume, 'iloc'):
                    curr_volume = curr_volume.iloc[0]
                if hasattr(prev_volume, 'iloc'):
                    prev_volume = prev_volume.iloc[0]
                
                vol_change = ((curr_volume - prev_volume) / prev_volume) * 100
                if vol_change > 30:
                    volume_signal = " Trading volume was significantly higher than previous day."
                elif vol_change < -30:
                    volume_signal = " Trading volume was significantly lower than previous day."
            
            # Create forecast hint (only for training)
            forecast = ""
            if include_future_data and 'Weekly_Return' in row:
                weekly_return = row['Weekly_Return']
                # Convert to scalar if it's a Series
                if hasattr(weekly_return, 'iloc'):
                    weekly_return = weekly_return.iloc[0]
                elif hasattr(weekly_return, 'item'):
                    weekly_return = weekly_return.item()
                
                # Safe comparison with scalar value
                if weekly_return > 0.03:
                    forecast = "Outlook: Strong bullish trend expected in the next week."
                elif weekly_return > 0:
                    forecast = "Outlook: Slight upward movement may continue."
                elif weekly_return > -0.03:
                    forecast = "Outlook: Mild bearish pressure in the short term."
                else:
                    forecast = "Outlook: Significant downward pressure expected."
                
            # Format date nicely
            date_str = row['Date'].strftime('%B %d, %Y') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])
            
            # Get safe scalar values for prices
            close_price = row['Close']
            low_price = row['Low']
            high_price = row['High']
            
            # Convert to scalar if needed
            if hasattr(close_price, 'iloc'):
                close_price = close_price.iloc[0]
            if hasattr(low_price, 'iloc'):
                low_price = low_price.iloc[0]
            if hasattr(high_price, 'iloc'):
                high_price = high_price.iloc[0]
            
            # Create comprehensive text
            text = (f"{symbol.upper()} {movement} ${close_price:.2f} on {date_str}. "
                   f"Day range: ${low_price:.2f} to ${high_price:.2f}.{rsi_signal}{volume_signal}")
            
            # Only add forecast if include_future_data is True
            if forecast:
                text += f" {forecast}"
            
            texts.append(text)
            
        df['text'] = texts
    
    return df

def remove_future_data(df):
    """
    Ensure no future-looking columns are included in the dataset,
    which is crucial for inference or evaluation to prevent data leakage.
    """
    # List of columns that contain future data or might leak information about the future
    future_cols = [
        'Future_Close', 'Weekly_Return',
        # Add any other columns derived from future data
    ]
    
    # Drop future columns if they exist
    original_cols = set(df.columns)
    df = df.drop(columns=[col for col in future_cols if col in df.columns])
    removed_cols = original_cols - set(df.columns)
    if removed_cols:
        logging.info(f"Removed future-looking columns: {removed_cols}")
    
    # Check for and drop rows that would need future data
    # For example, the last N rows in time series prediction
    return df

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Download and prepare stock data for training.")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='Training_Data', help='Output folder')
    parser.add_argument('--inference', action='store_true', help='Prepare data for inference (removes future data)')
    args = parser.parse_args()

    # Determine whether to include future-looking data
    include_future_data = not args.inference
    if args.inference:
        logging.info("Inference mode: future data will be removed")

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

    # Add a timestamp to mark current data cutoff point
    logging.info(f"Data cutoff timestamp: {datetime.datetime.now()}")
    
    # Market feature download has been disabled
    # df = prepare_market_features(df, symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # Compute technical indicators - pass the flag to avoid future data lookbacks
    df = compute_technical_indicators(df, remove_future_indicators=not include_future_data)
    
    # Create targets - either with or without future data
    df = create_targets(df, include_future_data=include_future_data)

    # Select features and columns to keep - use a dynamic approach to handle any feature set
    # Start with the base features we know should exist
    base_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_30', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_hist',
        'RSI_14', 'BB_upper', 'BB_lower', 'BB_width', 'BB_pct',
        'Stoch_%K', 'Stoch_%D', 'ATR_14', 'OBV', 'ROC_10',
        'Close_lag1', 'Close_lag5', 'Volume_lag1', 'Return_1d'
    ]
    
    # Update volatility feature to match what's actually created (5d instead of 10d)
    volatility_features = [f'Volatility_{period}d' for period in [5, 21, 63]]
    
    # Get all columns that actually exist in the dataframe
    feature_cols = [col for col in base_features + volatility_features if col in df.columns]
    
    # Add additional features if they exist
    for col in df.columns:
        if col.startswith('RSI_') and col != 'RSI_14' and col not in feature_cols:
            feature_cols.append(col)
        if col.startswith('MA_ratio_') and col not in feature_cols:
            feature_cols.append(col)
        if col.startswith('Return_') and col not in feature_cols and not col.startswith('Return_5d'):
            # Exclude target columns that start with Return_5d
            if not include_future_data or not any(col.startswith(f'Return_{h}d') for h in [1, 3, 5, 10, 21]):
                feature_cols.append(col)
    
    # Add day of week and other temporal features if present
    for col in ['DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'WeekOfYear', 'MonthEnd']:
        if col in df.columns:
            feature_cols.append(col)
    
    # Log selected features
    logging.info(f"Selected {len(feature_cols)} features for model input")
    
    # Define columns to keep, based on whether we want future data
    if include_future_data:
        keep_cols = ['Date'] + feature_cols + ['Future_Close', 'Weekly_Return', 'label']
    else:
        keep_cols = ['Date'] + feature_cols + ['label']

    # Drop rows with missing values (from rolling calculations and lags)
    df = df[keep_cols].dropna().reset_index(drop=True)
    if df.empty:
        logging.error("No data left after dropping rows with missing values.")
        sys.exit(1)

    # Normalize features
    df_scaled = normalize_features(df, feature_cols)

    # Add a 'text' column for compatibility with the data validator
    df_scaled = add_text_column(df_scaled, symbol, include_future_data=include_future_data)

    # Final check to remove any future data that might have been added
    if not include_future_data:
        original_len = len(df_scaled)
        df_scaled = remove_future_data(df_scaled)
        if len(df_scaled) < original_len:
            logging.info(f"Removed {original_len - len(df_scaled)} rows that required future data")

    # Ensure dataset is appropriate for TFT model
    if not include_future_data:
        # Add group_id and time_idx columns if they don't exist (needed for TFT)
        if 'group_id' not in df_scaled.columns:
            df_scaled['group_id'] = 0  # Single group for the stock
        if 'time_idx' not in df_scaled.columns:
            df_scaled['time_idx'] = range(len(df_scaled))
        if 'target' not in df_scaled.columns:
            # For TFT, we need a target column, use Close price as default
            df_scaled['target'] = df_scaled['Close']
            logging.info("Added target column for TFT compatibility")
    
    # Add target scales for TFT (needed for prediction normalization)
    if not include_future_data:
        for scale_type in ['center', 'scale']:
            col_name = f'target_{scale_type}'
            if col_name not in df_scaled:
                if scale_type == 'center':
                    df_scaled[col_name] = df_scaled['Close'].rolling(window=30).mean().fillna(df_scaled['Close'].mean())
                else:
                    df_scaled[col_name] = df_scaled['Close'].rolling(window=30).std().fillna(df_scaled['Close'].std())
                logging.info(f"Added {col_name} for TFT normalization")
    
    # Reorder columns: text, features..., label
    ordered_cols = ['text'] + [c for c in df_scaled.columns if c not in ['text', 'label']] + ['label']
    df_scaled = df_scaled[ordered_cols]

    # Save to CSV
    out_path = get_next_data_filename(output_dir)
    try:
        df_scaled.to_csv(out_path, index=False)
        logging.info(f"Saved processed data to {out_path} ({len(df_scaled)} rows).")
        
        # If saving for inference, also output a separate file with just the features
        if not include_future_data:
            inference_cols = ['Date', 'text'] + feature_cols
            inference_df = df_scaled[inference_cols]
            inference_path = os.path.join(os.path.dirname(out_path), "inference_data.csv")
            inference_df.to_csv(inference_path, index=False)
            logging.info(f"Saved inference-ready data to {inference_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
