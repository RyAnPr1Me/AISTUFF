#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

#========================================================================
# Data Validation and Cleaning Script for Training Data with ALBERT Base v2
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
    df = df.dropna(axis=1, how='all')  # Remove all-NaN columns
    
    if 'text' not in df.columns:
        text_candidates = ['headline', 'news', 'sentence', 'content', 'body', 'title', 'summary']
        for c in text_candidates:
            if c in df.columns:
                df['text'] = df[c]
                break
        if 'text' not in df.columns:
            combos = [c for c in text_candidates if c in df.columns]
            if combos:
                df['text'] = df[combos].astype(str).agg(' '.join, axis=1)

    if 'label' not in df.columns:
        label_candidates = ['target', 'class', 'y', 'output']
        for c in label_candidates:
            if c in df.columns:
                df['label'] = df[c]
                break
        if 'label' not in df.columns:
            for c in df.columns:
                if 'return' in c:
                    df['label'] = (df[c] > 0).astype(int)
                    break
            if 'label' not in df.columns and 'future_close' in df.columns and 'close' in df.columns:
                df['label'] = (df['future_close'] > df['close']).astype(int)

    keep = [col for col in required_columns if col in df.columns]
    feature_cols = [c for c in df.columns if c.startswith('feature_') or c in ['open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_30', 'ema_12', 'ema_26', 'macd', 'rsi_14', 'bb_upper', 'bb_lower', 'stoch_%k', 'stoch_%d', 'close_lag1', 'close_lag5']]
    keep += feature_cols
    if 'date' in df.columns and 'date' not in keep:
        keep.append('date')
    
    keep = list(dict.fromkeys(keep))
    if not keep:
        return None
    df = df[keep]
    
    # Standardizing column names
    df = df.rename(columns={col: col.lower() for col in df.columns})
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
    col_map = {c.lower(): c for c in df.columns}
    required_map = {}
    synonyms = {'text': ['text', 'headline', 'news', 'sentence', 'content', 'body', 'title', 'summary'],
                'label': ['label', 'target', 'class', 'y', 'output']}

    for req in required_columns:
        found = None
        if req in col_map:
            found = col_map[req]
        else:
            for syn in synonyms.get(req, []):
                if syn in col_map:
                    found = col_map[syn]
                    break
        if found:
            required_map[req] = found
        else:
            logging.warning(f"Column '{req}' not found, attempting to format DataFrame...")
            df = try_format_dataframe(df, required_columns)
            if df is not None and req in df.columns:
                required_map[req] = req
            else:
                logging.error(f"Missing required column: '{req}' after formatting attempt.")
                return None

    df = df.rename(columns={v: k for k, v in required_map.items()})
    df.dropna(subset=required_columns, inplace=True)
    logging.info(f"Dropped rows with missing values in required columns: {required_columns}")

    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'] != '']

    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df.dropna(subset=['label'], inplace=True)
        df = df[df['label'].isin([0, 1, 2])]
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zscores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std(ddof=0) + 1e-8))
    outlier_mask = (zscores > 5).any(axis=1)
    df = df[~outlier_mask]

    if 'text' in df.columns:
        sample_texts = df['text'].iloc[:min(5, len(df))].tolist()
        try:
            tokenizer(sample_texts, padding=True, truncation=True, return_tensors='pt')
        except Exception as e:
            logging.error(f"Tokenizer error: {e}")
            return None

    df.drop_duplicates(subset=required_columns, inplace=True)
    logging.info(f"Dropped duplicates. Remaining rows: {len(df)}")

    return df


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

    parser = argparse.ArgumentParser(description="Validate and clean CSV training data files")
    parser.add_argument('--data-dir', type=str, default='Training_Data', help='Directory with CSV files to validate')
    parser.add_argument('--output', type=str, default='Training_Data/validated_data.csv', help='Output path for validated data')
    parser.add_argument('--tokenizer', type=str, default='albert-base-v2', help='Hugging Face tokenizer model name')
    parser.add_argument('--columns', nargs='+', default=['text', 'label'], help='List of required columns in each CSV')
    parser.add_argument('--download-ticker', type=str, default=None, help='If set, download stock data for this ticker and create a dataset (CSV) in data-dir')
    args = parser.parse_args()

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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    validated_dfs = []
    csv_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.csv')]
    
    for fname in csv_files:
        path = os.path.join(args.data_dir, fname)
        logging.info(f"Processing {fname}...")
        try:
            df = pd.read_csv(path)
            cleaned = validate_and_clean(df, args.columns, tokenizer)
            if cleaned is not None:
                validated_dfs.append(cleaned)
        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}")

    if validated_dfs:
        combined = pd.concat(validated_dfs, ignore_index=True)
        combined.to_csv(args.output, index=False)
        logging.info(f"Saved {len(combined)} rows to {args.output}")
    else:
        logging.error("No valid data after processing all files.")

if __name__ == '__main__':
    main()

