import os
import sys
import pandas as pd
import argparse

def format_dataset_for_albert(input_path, output_path):
    """
    Format a CSV dataset for ALBERT (albert-large-v2) compatibility:
    - Ensures columns: 'text', feature columns, 'label'
    - Lowercases all column names
    - Ensures 'text' is string, 'label' is int
    - Removes rows with missing required fields
    - Reorders columns: text, features..., label
    """
    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Try to create 'text' if missing
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
    # Try to create 'label' if missing
    if 'label' not in df.columns:
        label_candidates = ['target', 'class', 'y', 'output']
        for c in label_candidates:
            if c in df.columns:
                df['label'] = df[c]
                break
        if 'label' not in df.columns:
            # Try to infer from returns
            for c in df.columns:
                if 'return' in c:
                    df['label'] = (df[c] > 0).astype(int)
                    break
            if 'label' not in df.columns and 'future_close' in df.columns and 'close' in df.columns:
                df['label'] = (df['future_close'] > df['close']).astype(int)
    # Remove rows with missing text or label
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    # Reorder columns: text, features..., label
    feature_cols = [c for c in df.columns if c not in ['text', 'label']]
    ordered_cols = ['text'] + feature_cols + ['label']
    df = df[ordered_cols]
    df.to_csv(output_path, index=False)
    print(f"Formatted and saved: {output_path}")

def format_all_datasets_in_folder(folder):
    for fname in os.listdir(folder):
        if fname.lower().endswith('.csv'):
            in_path = os.path.join(folder, fname)
            out_path = os.path.join(folder, f"albert_{fname}")
            format_dataset_for_albert(in_path, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format datasets for ALBERT (albert-large-v2) compatibility.")
    parser.add_argument('--input', type=str, help="Input CSV file (if formatting one file)")
    parser.add_argument('--output', type=str, help="Output CSV file (if formatting one file)")
    parser.add_argument('--folder', type=str, help="If set, format all CSVs in this folder")
    args = parser.parse_args()

    if args.folder:
        format_all_datasets_in_folder(args.folder)
    elif args.input and args.output:
        format_dataset_for_albert(args.input, args.output)
    else:
        print("Usage: python format_for_albert.py --input INPUT.csv --output OUTPUT.csv")
        print("   or: python format_for_albert.py --folder Training_Data")
