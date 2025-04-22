import os
import torch
import pandas as pd
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
TABULAR_DIM = 64  # Change this value if needed

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
        # If no tabular features, generate random data with the required number of features (64)
        tabular_data = torch.randn(len(data), TABULAR_DIM)
    else:
        # Extract the features
        features = data[[col for col in data.columns if col.startswith("feature_")]]
        scaler = StandardScaler()
        tabular_data = torch.tensor(scaler.fit_transform(features.values), dtype=torch.float)

        # Ensure that the data matches the required dimension (64 features)
        if tabular_data.shape[1] < TABULAR_DIM:
            # Padding tabular data with zeros if it has less than 64 features
            padded_features = torch.zeros(len(data), TABULAR_DIM)
            padded_features[:, :tabular_data.shape[1]] = tabular_data
            tabular_data = padded_features
        elif tabular_data.shape[1] > TABULAR_DIM:
            # Truncate if there are more than 64 features
            tabular_data = tabular_data[:, :TABULAR_DIM]

    labels = torch.tensor(data["label"].values, dtype=torch.long)

    # Train/validation split
    X_ids_train, X_ids_val, X_mask_train, X_mask_val, t_train, t_val, y_train, y_val = train_test_split(
        input_ids, attention_mask, tabular_data, labels, test_size=0.2
    )

    # Create Datasets
    train_dataset = StockDataset(X_ids_train, X_mask_train, t_train, y_train)
    val_dataset = StockDataset(X_ids_val, X_mask_val, t_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, optimizer
    model = MultimodalStockPredictor(tabular_dim=TABULAR_DIM)  # Ensure model expects 64 features
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0.0
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
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} complete. Training Loss: {train_loss / len(train_loader)}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:  # Use validation dataloader
                logits = model(
                    {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"]
                    },
                    batch["tabular"]
                )
                loss = loss_fn(logits, batch["label"])
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} Validation Loss: {val_loss / len(val_loader)}")
        model.train()

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()


