import os
import pandas as pd
from transformers import AutoTokenizer

DATA_DIR = "Training_Data"
VALIDATED_DATA_PATH = "Training_Data/validated_data.csv"

def is_valid_csv(df):
    """Ensure that the CSV has the correct columns."""
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        print(f"Missing required columns: {required_columns - set(df.columns)}")
        return False
    return True

def main():
    # Load all CSV files in the directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in Training_Data.")
        exit(1)

    df_list = []
    for f in csv_files:
        path = os.path.join(DATA_DIR, f)
        try:
            df = pd.read_csv(path)
            if not is_valid_csv(df):
                print(f"[ERROR] {f} is not a valid CSV.")
                exit(1)
            df_list.append(df)
        except Exception as e:
            print(f"[ERROR] Could not load {f}: {e}")
            exit(1)

    # Combine data from all CSVs into one DataFrame
    data = pd.concat(df_list)

    # Drop rows with missing values in 'text' or 'label' columns
    data = data.dropna(subset=["text", "label"])

    # Save the validated data
    data.to_csv(VALIDATED_DATA_PATH, index=False)
    print(f"Validated data saved to {VALIDATED_DATA_PATH}")

if __name__ == "__main__":
    main()
