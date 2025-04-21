import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer

EXPECTED_COLUMNS = ['headline', 'open', 'close', 'volume']
TEXT_COLUMN = 'headline'
TABULAR_COLUMNS = ['open', 'close', 'volume']
MODEL_NAME = 'bert-large-uncased'

def validate_csv(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Failed to read {filepath}: {e}")
        return False

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ {filepath} is missing columns: {missing_cols}")
        return False

    if df.isnull().any().any():
        print(f"❌ {filepath} contains null values")
        return False

    if df.empty:
        print(f"❌ {filepath} is empty")
        return False

    return True

def tokenize_and_save(df, tokenizer, out_path):
    print("✅ Tokenizing text data...")
    tokenized = tokenizer(df[TEXT_COLUMN].tolist(), padding=True, truncation=True, return_tensors="pt")
    tabular_data = torch.tensor(df[TABULAR_COLUMNS].values, dtype=torch.float32)

    torch.save({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'tabular_data': tabular_data
    }, out_path)

    print(f"✅ Saved tokenized and tabular data to {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help='Directory containing CSV files')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    any_failed = False

    for file in os.listdir(args.input_dir):
        if not file.endswith('.csv'):
            continue

        filepath = os.path.join(args.input_dir, file)
        if not validate_csv(filepath):
            any_failed = True
            continue

        df = pd.read_csv(filepath)
        output_path = os.path.join(args.input_dir, f"{os.path.splitext(file)[0]}_processed.pt")
        tokenize_and_save(df, tokenizer, output_path)

    if any_failed:
        print("❌ One or more files failed validation. Aborting training.")
        sys.exit(1)
    else:
        print("✅ All files validated and processed.")

if __name__ == '__main__':
    main()
