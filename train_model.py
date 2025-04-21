import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer
from stock_ai import MultimodalStockPredictor
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = "Training_Data"
MODEL_SAVE_PATH = "trained_model/model_weights.pt"

# Parameters
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
TEXT_MODEL_NAME = "bert-large-uncased"
TABULAR_DIM = 64

# 1. Define custom dataset
class StockDataset(Dataset):
    def __init__(self, text_data, tabular_data, labels):
        self.text_data = text_data
        self.tabular_data = tabular_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.text_data[idx],
            "tabular": self.tabular_data[idx],
            "label": self.labels[idx]
        }

def is_valid_csv(df):
    # Check if required columns exist
    return {'text', 'label'}.issubset(df.columns)

def main():
    print("Loading data from:", DATA_DIR)
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in Training_Data.")
        sys.exit(1)

    df_list = []
    for f in csv_files:
        path = os.path.join(DATA_DIR, f)
        try:
            df = pd.read_csv(path)
            if not is_valid_csv(df):
                print(f"[ERROR] Missing required columns in {f}.")
                sys.exit(1)
            df_list.append(df)
        except Exception as e:
            print(f"[ERROR] Could not load {f}: {e}")
            sys.exit(1)

    data = pd.concat(df_list)
    data = data.dropna(subset=["text", "label"])

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_tokens = tokenizer(list(data["text"]), padding=True, truncation=True, return_tensors="pt")

    # Simulate tabular data if not provided
    if 'feature_0' not in data.columns:
        tabular_data = torch.randn(len(data), TABULAR_DIM)
    else:
        features = data[[col for col in data.columns if col.startswith("feature_")]]
        scaler = StandardScaler()
        tabular_data = torch.tensor(scaler.fit_transform(features.values), dtype=torch.float)

    labels = torch.tensor(data["label"].values, dtype=torch.long)

    # Train/val split
    X_train, X_val, t_train, t_val, y_train, y_val = train_test_split(
        text_tokens['input_ids'], tabular_data, labels, test_size=0.2)

    train_dataset = StockDataset(X_train, t_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultimodalStockPredictor()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                {"input_ids": batch["text"]},
                batch["tabular"]
            )
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} complete. Loss: {loss.item()}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
