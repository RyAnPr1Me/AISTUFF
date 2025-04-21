import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from src.models.stock_ai import MultimodalStockPredictor  # Corrected import
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
DEFAULT_TABULAR_DIM = 64  # default value, can be overridden

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

def main():
    # Load validated data
    print(f"Loading validated data from {VALIDATED_DATA_PATH}")
    data = pd.read_csv(VALIDATED_DATA_PATH)

    # Tokenize text using HuggingFace tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_tokens = tokenizer(list(data["text"]), padding=True, truncation=True, return_tensors="pt")

    # Handle tabular data
    if 'feature_0' not in data.columns:
        tabular_data = torch.randn(len(data), DEFAULT_TABULAR_DIM)
        actual_tabular_dim = DEFAULT_TABULAR_DIM
    else:
        features = data[[col for col in data.columns if col.startswith("feature_")]]
        scaler = StandardScaler()
        tabular_data = torch.tensor(scaler.fit_transform(features.values), dtype=torch.float)
        actual_tabular_dim = tabular_data.shape[1]  # <-- get actual number of features

    labels = torch.tensor(data["label"].values, dtype=torch.long)

    # Train/validation split
    X_train, X_val, t_train, t_val, y_train, y_val = train_test_split(
        text_tokens['input_ids'], tabular_data, labels, test_size=0.2)

    train_dataset = StockDataset(X_train, t_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MultimodalStockPredictor(tabular_dim=actual_tabular_dim)  # <-- fix: pass actual dim
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
