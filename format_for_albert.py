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
    - Logs warnings for missing/invalid columns and rows
    """
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[ERROR] Could not read {input_path}: {e}")
        return

    df.columns = [str(c).strip().lower() for c in df.columns]
    orig_len = len(df)

    # Try to create 'text' if missing
    if 'text' not in df.columns:
        text_candidates = ['headline', 'news', 'sentence', 'content', 'body', 'title', 'summary']
        found = False
        for c in text_candidates:
            if c in df.columns:
                df['text'] = df[c]
                found = True
                print(f"[INFO] Used '{c}' column as 'text'.")
                break
        if not found:
            combos = [c for c in text_candidates if c in df.columns]
            if combos:
                df['text'] = df[combos].astype(str).agg(' '.join, axis=1)
                print(f"[INFO] Combined columns {combos} to create 'text'.")
            else:
                print(f"[WARNING] No suitable text column found in {input_path}. All rows will be dropped.")
                df['text'] = None

    # Try to create 'label' if missing
    if 'label' not in df.columns:
        label_candidates = ['target', 'class', 'y', 'output']
        found = False
        for c in label_candidates:
            if c in df.columns:
                df['label'] = df[c]
                found = True
                print(f"[INFO] Used '{c}' column as 'label'.")
                break
        if not found:
            # Try to infer from returns
            for c in df.columns:
                if 'return' in c:
                    df['label'] = (df[c] > 0).astype(int)
                    found = True
                    print(f"[INFO] Inferred 'label' from '{c}' column (binary up/down).")
                    break
            if not found and 'future_close' in df.columns and 'close' in df.columns:
                df['label'] = (df['future_close'] > df['close']).astype(int)
                print(f"[INFO] Inferred 'label' from 'future_close' and 'close' columns (binary up/down).")
            elif not found:
                print(f"[WARNING] No suitable label column found in {input_path}. All rows will be dropped.")
                df['label'] = None

    # Remove rows with missing text or label
    before_drop = len(df)
    df = df.dropna(subset=['text', 'label'])
    after_drop = len(df)
    if after_drop < before_drop:
        print(f"[INFO] Dropped {before_drop - after_drop} rows with missing text or label.")

    # Clean and check types
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
    before_label_drop = len(df)
    df = df.dropna(subset=['label'])
    after_label_drop = len(df)
    if after_label_drop < before_label_drop:
        print(f"[INFO] Dropped {before_label_drop - after_label_drop} rows with invalid label values.")

    try:
        df['label'] = df['label'].astype(int)
    except Exception as e:
        print(f"[ERROR] Could not convert label column to int: {e}")
        return

    # Warn if label is not binary or multiclass
    unique_labels = df['label'].unique()
    if len(unique_labels) > 10:
        print(f"[WARNING] Label column has more than 10 unique values: {unique_labels[:10]}...")

    # Reorder columns: text, features..., label
    feature_cols = [c for c in df.columns if c not in ['text', 'label']]
    ordered_cols = ['text'] + feature_cols + ['label']
    missing_cols = [c for c in ordered_cols if c not in df.columns]
    if missing_cols:
        print(f"[WARNING] Missing columns in output: {missing_cols}")
        ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]

    try:
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Formatted and saved: {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    except Exception as e:
        print(f"[ERROR] Could not save {output_path}: {e}")

def format_all_datasets_in_folder(folder):
    for fname in os.listdir(folder):
        if fname.lower().endswith('.csv'):
            in_path = os.path.join(folder, fname)
            out_path = os.path.join(folder, f"albert_{fname}")
            print(f"[INFO] Formatting {in_path} -> {out_path}")
            format_dataset_for_albert(in_path, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format datasets for ALBERT (albert-large-v2) compatibility.")
    parser.add_argument('--input', type=str, help="Input CSV file (if formatting one file)")
    parser.add_argument('--output', type=str, help="Output CSV file (if formatting one file)")
    parser.add_argument('--folder', type=str, help="If set, format all CSVs in this folder")
    args = parser.parse_args()

    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"[ERROR] Folder not found: {args.folder}")
            sys.exit(1)
        format_all_datasets_in_folder(args.folder)
    elif args.input and args.output:
        if not os.path.isfile(args.input):
            print(f"[ERROR] Input file not found: {args.input}")
            sys.exit(1)
        format_dataset_for_albert(args.input, args.output)
    else:
        print("Usage: python format_for_albert.py --input INPUT.csv --output OUTPUT.csv")
        print("   or: python format_for_albert.py --folder Training_Data")
