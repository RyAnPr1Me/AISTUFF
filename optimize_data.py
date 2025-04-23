import argparse
import pandas as pd
import numpy as np
import logging
import os

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def optimize_data_for_ai(df, label_col='label', text_col='text', corr_thresh=0.98, min_var=1e-8):
    drop_cols = []

    # Remove constant columns
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        drop_cols.extend(const_cols)
        logging.info(f"Data optimizer: Dropping constant columns: {const_cols}")

    # Remove low variance columns (excluding label/text)
    var = df.var(numeric_only=True)
    low_var_cols = [col for col in var.index if var[col] < min_var and col not in [label_col, text_col]]
    if low_var_cols:
        drop_cols.extend(low_var_cols)
        logging.info(f"Data optimizer: Dropping low-variance columns: {low_var_cols}")

    # Remove highly correlated columns (keep only one from each group)
    feature_cols = [col for col in df.columns if col not in [label_col, text_col]]
    if len(feature_cols) > 1:
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        if to_drop:
            drop_cols.extend(to_drop)
            logging.info(f"Data optimizer: Dropping highly correlated columns: {to_drop}")

    drop_cols = list(set(drop_cols))
    df_optimized = df.drop(columns=drop_cols, errors='ignore')
    logging.info(f"Data optimizer: Final columns: {list(df_optimized.columns)}")
    return df_optimized

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Optimize tabular data for AI model training.")
    parser.add_argument('--input', type=str, required=True, help="Input CSV file to optimize")
    parser.add_argument('--output', type=str, required=True, help="Output CSV file for optimized data")
    parser.add_argument('--label-col', type=str, default='label', help="Label column name")
    parser.add_argument('--text-col', type=str, default='text', help="Text column name")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logging.error(f"Input file not found: {args.input}")
        exit(1)

    df = pd.read_csv(args.input)
    df_optimized = optimize_data_for_ai(df, label_col=args.label_col, text_col=args.text_col)
    df_optimized.to_csv(args.output, index=False)
    logging.info(f"Optimized data saved to {args.output}")

if __name__ == "__main__":
    main()
