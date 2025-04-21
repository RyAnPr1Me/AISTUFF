import os
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.models.stock_ai import MultimodalStockPredictor
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

VALIDATED_DATA_PATH = "Training_Data/validated_data.csv"
MODEL_SAVE_PATH = "trained_model/model_weights.pth"

# Parameters
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
TEXT_MODEL_NAME = "bert-large-uncased"
DEFAULT_TABULAR_DIM = 64  # Only used if no features present

class StockDataset(Dataset):
    def __init__(self, input_ids, attention_mask, tabular_data, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.tabular_data = tabular_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "tabular": self.tabular_data[idx],
            "label": self.labels[idx]
        }

def main():
    print(f"Loading validated data from {VALIDATED_DATA_PATH}")
    data = pd.read_csv(VALIDATED_DATA_PATH)

    # Tokenize text with attention mask
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    encoded = tokenizer(
        list(data["text"]),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Process tabular data
    if 'feature_0' not in data.columns:
        tabular_data = torch.randn(len(data), DEFAULT_TABULAR_DIM)
        actual_tabular_dim = DEFAULT_TABULAR_DIM
    else:
        features = data[[col for col in data.columns if col.startswith("feature_")]]
        scaler = StandardScaler()
        tabular_data = torch.tensor(scaler.fit_transform(features.values), dtype=torch.float)
        actual_tabular_dim = tabular_data.shape[1]

    labels = torch.tensor(data["label"].values, dtype=torch.long)

    # Train/validation split
    X_ids_train, X_ids_val, X_mask_train, X_mask_val, t_train, t_val, y_train, y_val = train_test_split(
        input_ids, attention_mask, tabular_data, labels, test_size=0.2
    )

    train_dataset = StockDataset(X_ids_train, X_mask_train, t_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model with correct tabular dimension
    model = MultimodalStockPredictor(tabular_dim=actual_tabular_dim)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]
                },
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
