#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

#========================================================================
# Data Validation and Cleaning Script for Training Data
#========================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )


def try_format_dataframe(df, required_columns):
    """
    Attempt to coerce/format a DataFrame to the expected format:
    - Try to extract or combine columns for 'text' and 'label'
    - Try to infer label from other columns if not present
    - Try to create a 'text' column from multiple possible text fields
    - Lowercase all column names for easier matching
    - Remove columns with all NaN or empty values
    Returns a new DataFrame or None if not possible.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Remove all-NaN columns
    df = df.dropna(axis=1, how='all')
    # Try to create 'text'
    if 'text' not in df.columns:
        text_candidates = ['headline', 'news', 'sentence', 'content', 'body', 'title', 'summary']
        for c in text_candidates:
            if c in df.columns:
                df['text'] = df[c]
                break
        # Try to combine multiple text fields if possible
        if 'text' not in df.columns:
            combos = [c for c in text_candidates if c in df.columns]
            if combos:
                df['text'] = df[combos].astype(str).agg(' '.join, axis=1)
    # Try to create 'label'
    if 'label' not in df.columns:
        label_candidates = ['target', 'class', 'y', 'output']
        for c in label_candidates:
            if c in df.columns:
                df['label'] = df[c]
                break
        # Try to infer label from price/return columns
        if 'label' not in df.columns:
            # If there is a 'weekly_return' or similar, use it
            for c in df.columns:
                if 'return' in c:
                    df['label'] = (df[c] > 0).astype(int)
                    break
            # If there is a 'future_close' and 'close', use them
            if 'label' not in df.columns and 'future_close' in df.columns and 'close' in df.columns:
                df['label'] = (df['future_close'] > df['close']).astype(int)
    # Only keep required columns and any feature columns
    keep = []
    for col in required_columns:
        if col in df.columns:
            keep.append(col)
    # Add feature columns (those starting with 'feature_' or technical indicators)
    feature_cols = [c for c in df.columns if c.startswith('feature_') or c in [
        'open', 'high', 'low', 'close', 'volume',
        'sma_5', 'sma_30', 'ema_12', 'ema_26', 'macd',
        'rsi_14', 'bb_upper', 'bb_lower', 'stoch_%k', 'stoch_%d',
        'close_lag1', 'close_lag5'
    ]]
    for c in feature_cols:
        if c not in keep:
            keep.append(c)
    # Add 'date' if present
    if 'date' in df.columns and 'date' not in keep:
        keep.append('date')
    # Remove duplicates in keep
    keep = list(dict.fromkeys(keep))
    if not keep:
        return None
    df = df[keep]
    # Rename columns to standard names
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    return df


def validate_and_clean(df: pd.DataFrame,
                       required_columns: list,
                       tokenizer: AutoTokenizer) -> pd.DataFrame:
    """
    Enhanced, adaptable validation and cleaning:
     - Ensure required columns are present (case-insensitive, flexible mapping).
     - Attempt to infer/rename columns if possible.
     - Drop rows with NaNs in required columns.
     - Strip and drop empty text entries.
     - Convert labels to numeric and check for valid values.
     - Remove rows with outlier/invalid values in numeric columns.
     - Test tokenization on a sample of text if text exists.
     - Drop duplicate rows.
     - Log summary statistics.
    Returns cleaned DataFrame or None if validation fails.
    """
    # Flexible column mapping (case-insensitive, allow synonyms)
    col_map = {c.lower(): c for c in df.columns}
    required_map = {}
    synonyms = {
        'text': ['text', 'headline', 'news', 'sentence', 'content', 'body', 'title', 'summary'],
        'label': ['label', 'target', 'class', 'y', 'output'],
    }
    for req in required_columns:
        found = None
        # Try direct match
        if req in col_map:
            found = col_map[req]
        else:
            # Try synonyms
            for syn in synonyms.get(req, []):
                if syn in col_map:
                    found = col_map[syn]
                    break
        if found:
            required_map[req] = found
        else:
            # Try to format the DataFrame to create the required column
            logging.warning(f"Column '{req}' not found, attempting to format DataFrame...")
            df = try_format_dataframe(df, required_columns)
            if df is not None and req in df.columns:
                required_map[req] = req
            else:
                logging.error(f"Missing required column (or synonym): '{req}' after attempted formatting.")
                return None

    # Rename columns to standard names for downstream processing
    df = df.rename(columns={v: k for k, v in required_map.items()})

    # Drop rows with missing values in required columns
    before = len(df)
    df.dropna(subset=required_columns, inplace=True)
    after = len(df)
    if before != after:
        logging.info(f"Dropped {before - after} rows with missing values in required columns.")

    # Clean text column
    if 'text' in required_columns and 'text' in df.columns:
        df['text'] = df['text'].astype(str).str.strip()
        empty_text = df['text'] == ''
        if empty_text.any():
            count = empty_text.sum()
            logging.warning(f"Dropping {count} rows with empty text.")
            df = df[~empty_text]

    # Convert labels to numeric and check for valid values (0/1 or 0/1/2)
    if 'label' in required_columns and 'label' in df.columns:
        try:
            df['label'] = pd.to_numeric(df['label'])
        except Exception as e:
            logging.error(f"Label conversion failed: {e}")
            return None
        # Remove rows with invalid label values (allow any int for flexibility, but warn)
        valid_labels = {0, 1, 2}
        invalid = ~df['label'].isin(valid_labels)
        if invalid.any():
            count = invalid.sum()
            logging.warning(f"Dropping {count} rows with invalid label values (not in {valid_labels}).")
            df = df[~invalid]
        if df['label'].nunique() > 3:
            logging.warning(f"Label column has more than 3 unique values: {df['label'].unique()}")

    # Remove outliers in numeric columns (z-score > 5)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        zscores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std(ddof=0) + 1e-8))
        outlier_mask = (zscores > 5).any(axis=1)
        if outlier_mask.any():
            count = outlier_mask.sum()
            logging.warning(f"Dropping {count} rows with extreme outlier values (z-score > 5).")
            df = df[~outlier_mask]

    # Test tokenizer on a small sample if text exists
    if 'text' in required_columns and 'text' in df.columns:
        sample_texts = df['text'].iloc[:min(5, len(df))].tolist()
        try:
            tokenizer(sample_texts, padding=True, truncation=True, return_tensors='pt')
        except Exception as e:
            logging.error(f"Tokenizer error on sample texts: {e}")
            return None

    # Drop duplicates based on all required columns
    initial_len = len(df)
    df.drop_duplicates(subset=required_columns, inplace=True)
    dup_dropped = initial_len - len(df)
    if dup_dropped:
        logging.info(f"Dropped {dup_dropped} duplicate rows based on {required_columns}.")

    # Log summary statistics
    logging.info(f"Final row count: {len(df)}")
    if len(df) > 0:
        logging.info(f"Label distribution:\n{df['label'].value_counts().to_dict() if 'label' in df.columns else 'N/A'}")
        if 'text' in df.columns:
            logging.info(f"Sample texts: {df['text'].iloc[:2].tolist()}")

    return df if not df.empty else None


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Validate and clean CSV training data files"
    )
    parser.add_argument(
        '--data-dir', type=str, default='Training_Data',
        help='Directory containing CSV files to validate'
    )
    parser.add_argument(
        '--output', type=str, default='Training_Data/validated_data.csv',
        help='Path for the combined validated CSV output'
    )
    parser.add_argument(
        '--tokenizer', type=str, default='bert-large-uncased',
        help='Hugging Face tokenizer model name'
    )
    parser.add_argument(
        '--columns', nargs='+', default=['text', 'label'],
        help='List of required columns in each CSV'
    )
    # --- New: Option to download and process stock data for a given ticker ---
    parser.add_argument(
        '--download-ticker', type=str, default=None,
        help='If set, download stock data for this ticker and create a dataset (CSV) in data-dir'
    )
    parser.add_argument(
        '--start', type=str, default='2015-01-01',
        help='Start date for stock data download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str, default='2025-01-01',
        help='End date for stock data download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--albert-only', action='store_true', default=True,
        help='If set, only process files prefixed with albert_ (default: True)'
    )
    args = parser.parse_args()

    # --- New: Download and prepare stock data if requested ---
    if args.download_ticker:
        import yfinance as yf
        from sklearn.preprocessing import StandardScaler

        logging.info(f"Downloading {args.download_ticker} data from {args.start} to {args.end}...")
        df = yf.download(args.download_ticker, start=args.start, end=args.end)
        if df.empty:
            logging.error("No data downloaded. Check ticker and date range.")
            sys.exit(1)
        df = df.reset_index()

        # Compute technical indicators (same as in download_and_prepare_stock_data.py)
        # ...SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic Oscillator, lags...
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        ma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_upper'] = ma20 + 2 * std20
        df['BB_lower'] = ma20 - 2 * std20
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        df['Stoch_%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-10)
        df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag5'] = df['Close'].shift(5)
        # Targets
        df['Future_Close'] = df['Close'].shift(-5)
        close = df['Close']
        future_close = df['Future_Close']
        df['Weekly_Return'] = (future_close - close) / close
        df['label'] = (df['Weekly_Return'] > 0).astype(int)
        # Features/columns
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_30', 'EMA_12', 'EMA_26', 'MACD',
            'RSI_14', 'BB_upper', 'BB_lower',
            'Stoch_%K', 'Stoch_%D',
            'Close_lag1', 'Close_lag5'
        ]
        keep_cols = ['Date'] + feature_cols + ['Future_Close', 'Weekly_Return', 'label']
        df = df[keep_cols].dropna().reset_index(drop=True)
        if df.empty:
            logging.error("No data left after dropping rows with missing values.")
            sys.exit(1)
        # Normalize features
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        # Save to CSV in data-dir
        out_path = os.path.join(args.data_dir, f"{args.download_ticker}_data.csv")
        df.to_csv(out_path, index=False)
        logging.info(f"Saved downloaded and processed data to {out_path} ({len(df)} rows).")

    if not os.path.isdir(args.data_dir):
        logging.error(f"Directory not found: {args.data_dir}")
        sys.exit(1)

    csv_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.csv')]
    # Only process ALBERT-formatted files unless --albert-only is False
    if args.albert_only:
        csv_files = [f for f in csv_files if f.startswith('albert_')]
        if not csv_files:
            logging.error("No ALBERT-formatted CSV files found. Run format_for_albert.py first.")
            sys.exit(1)

    # Initialize tokenizer once
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{args.tokenizer}': {e}")
        sys.exit(1)

    validated_dfs = []
    for fname in csv_files:
        path = os.path.join(args.data_dir, fname)
        logging.info(f"Processing {fname}...")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logging.error(f"Failed to read {fname}: {e}")
            sys.exit(1)

        cleaned = validate_and_clean(df, args.columns, tokenizer)
        if cleaned is None:
            logging.error(f"Validation failed for {fname}. Skipping this file.")
            continue

        logging.info(f"{'Validated' if not cleaned.empty else 'No valid rows in'} {fname}: {len(cleaned)} rows.")
        validated_dfs.append(cleaned)

    if not validated_dfs:
        logging.error("No valid data after processing all files. Aborting.")
        sys.exit(1)

    # Concatenate all cleaned data
    combined = pd.concat(validated_dfs, ignore_index=True)
    if combined.empty:
        logging.error("No valid data after concatenation. Aborting.")
        sys.exit(1)

    # Shuffle the combined data
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final validated CSV
    try:
        combined.to_csv(args.output, index=False)
        logging.info(f"Saved {len(combined)} validated rows to {args.output}.")
    except Exception as e:
        logging.error(f"Failed to save validated data: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
