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
        'text': ['text', 'headline', 'news', 'sentence', 'content', 'body'],
        'label': ['label', 'target', 'class', 'y'],
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
            logging.error(f"Missing required column (or synonym): '{req}'")
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
