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


def validate_and_clean(df: pd.DataFrame,
                       required_columns: list,
                       tokenizer: AutoTokenizer) -> pd.DataFrame:
    """
    Validate and clean a DataFrame:
     - Ensure required columns are present.
     - Drop rows with NaNs in required columns.
     - Strip and drop empty text entries.
     - Convert labels to numeric.
     - Test tokenization on a sample of text.
     - Drop duplicate rows.

    Returns cleaned DataFrame or None if validation fails.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        logging.error(f"Missing required columns: {missing}")
        return None

    df = df.copy()
    # Drop rows with missing values in required columns
    df.dropna(subset=required_columns, inplace=True)

    # Clean text column
    df['text'] = df['text'].astype(str).str.strip()
    empty_text = df['text'] == ''
    if empty_text.any():
        count = empty_text.sum()
        logging.warning(f"Dropping {count} rows with empty text.")
        df = df[~empty_text]

    # Convert labels to numeric
    try:
        df['label'] = pd.to_numeric(df['label'])
    except Exception as e:
        logging.error(f"Label conversion failed: {e}")
        return None

    # Test tokenizer on a small sample
    sample_texts = df['text'].iloc[:min(5, len(df))].tolist()
    try:
        tokenizer(sample_texts, padding=True, truncation=True, return_tensors='pt')
    except Exception as e:
        logging.error(f"Tokenizer error on sample texts: {e}")
        return None

    # Drop duplicates based on text+label
    initial_len = len(df)
    df.drop_duplicates(subset=['text', 'label'], inplace=True)
    dup_dropped = initial_len - len(df)
    if dup_dropped:
        logging.info(f"Dropped {dup_dropped} duplicate rows.")

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
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logging.error(f"Directory not found: {args.data_dir}")
        sys.exit(1)

    csv_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        logging.error(f"No CSV files found in {args.data_dir}.")
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
            logging.error(f"Validation failed for {fname}. Aborting.")
            sys.exit(1)

        logging.info(f"{'Validated' if not cleaned.empty else 'No valid rows in'} {fname}: {len(cleaned)} rows.")
        validated_dfs.append(cleaned)

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
