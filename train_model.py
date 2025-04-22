import os
import torch
import pandas as pd
import logging
import warnings
from transformers import AutoTokenizer
from src.models.stock_ai import MultimodalStockPredictor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

#========================================================================
# Train Model Script with Performance Optimizations
#========================================================================

# Paths (removed saving paths)
VALIDATED_DATA_PATH = "Training_Data/validated_data.csv"

# Training parameters
BATCH_SIZE = 8
EPOCHS = 5  # Reduced for faster testing
LR = 1e-4
TEXT_MODEL_NAME = "bert-base-uncased"
TABULAR_DIM = 64
EARLY_STOPPING_PATIENCE = 5
MAX_SEQ_LEN = 128  # Max sequence length for text

# Logging setup
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    warnings.filterwarnings('ignore', category=FutureWarning)

# Dataset class
class StockDataset(Dataset):
    def __init__(self, texts, tabular_data, labels, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tabular_data = tabular_data
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize the text on the fly
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "tabular": self.tabular_data[idx],
            "label": self.labels[idx]
        }

# Main training function
def main():
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load and preprocess data
    logging.info(f"Loading data from {VALIDATED_DATA_PATH}")
    data = pd.read_csv(VALIDATED_DATA_PATH)
    if data.isnull().values.any():
        logging.warning("Data contains NaN values; filling with zeros.")
        data = data.fillna(0)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Process tabular features
    if 'feature_0' not in data.columns:
        tabular_data = torch.randn(len(data), TABULAR_DIM)
    else:
        features = data[[col for col in data.columns if col.startswith("feature_")]]
        scaler = StandardScaler()
        tabular_data = torch.tensor(scaler.fit_transform(features.values), dtype=torch.float)
        if tabular_data.size(1) < TABULAR_DIM:
            pad = torch.zeros(len(data), TABULAR_DIM - tabular_data.size(1))
            tabular_data = torch.cat([tabular_data, pad], dim=1)
        else:
            tabular_data = tabular_data[:, :TABULAR_DIM]

    labels = torch.tensor(data["label"].values, dtype=torch.long)

    # Split data (combine splitting and dataset creation)
    X_text = list(data["text"])
    X_tab = tabular_data
    y = labels

    X_text_train, X_text_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
        X_text, X_tab, y, test_size=0.2, random_state=42
    )

    train_ds = StockDataset(X_text_train, X_tab_train, y_train, tokenizer)
    val_ds = StockDataset(X_text_val, X_tab_val, y_val, tokenizer)

    # DataLoader with parallelism and persistent workers
    num_workers = os.cpu_count() or 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True)

    # Initialize model, loss, optimizer, scheduler
    model = MultimodalStockPredictor(tabular_dim=TABULAR_DIM).to(device)
    model = torch.compile(model)  # PyTorch 2.x optimization
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs/stock_model')

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]", leave=False):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tab = batch['tabular'].to(device)
            lbl = batch['label'].to(device)

            logits = model({'input_ids': ids, 'attention_mask': mask}, tab)
            loss = loss_fn(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            train_preds.extend(preds)
            train_labels.extend(lbl.cpu().tolist())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]", leave=False):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tab = batch['tabular'].to(device)
            lbl = batch['label'].to(device)

            logits = model({'input_ids': ids, 'attention_mask': mask}, tab)
            loss = loss_fn(logits, lbl)
            val_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            val_preds.extend(preds)
            val_labels.extend(lbl.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('Prec/val', val_prec, epoch)
        writer.add_scalar('Rec/val', val_rec, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Val loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # LR scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered.")
                break

    writer.close()
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
