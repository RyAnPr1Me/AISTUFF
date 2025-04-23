import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import logging
from transformers import AutoTokenizer
from src.models.stock_ai import MultimodalStockPredictor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np

#========================================================================
# Train Model Script with Performance Optimizations and Informative Logging
#========================================================================


VALIDATED_DATA_PATH = "Training_Data/validated_data.csv"  # This should be ALBERT-formatted after pipeline

BATCH_SIZE = 32
EPOCHS = 5  # Reduced for faster testing
LR = 1e-4
TEXT_MODEL_NAME = "albert-large-v2"  # Use ALBERT for consistency with pipeline2

# Paths (removed saving paths)
VALIDATED_DATA_PATH = "Training_Data/validated_data.csv"

# Training parameters
BATCH_SIZE = 32
EPOCHS = 5  # Reduced for faster testing
LR = 1e-4
TEXT_MODEL_NAME = "albert-base-v2"

TABULAR_DIM = 64
EARLY_STOPPING_PATIENCE = 5
MAX_SEQ_LEN = 128  # Max sequence length for text

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    warnings.filterwarnings('ignore', category=FutureWarning)

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

def print_data_overview(data):
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"Columns: {list(data.columns)}")
    if 'label' in data.columns:
        logging.info(f"Label distribution: {data['label'].value_counts().to_dict()}")
    if 'text' in data.columns:
        logging.info(f"Sample text: {data['text'].iloc[0]}")
    logging.info(f"First row: {data.iloc[0].to_dict()}")

def print_metrics(epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_prec, val_rec, val_f1):
    logging.info(
        f"[Epoch {epoch+1}] "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}"
    )

def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"Validation Confusion Matrix:\n{cm}")

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

    print_data_overview(data)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Process tabular features
    feature_cols = [col for col in data.columns if col not in ['text', 'label']]
    if not feature_cols:
        logging.error("No tabular features found in data. Exiting.")
        return
    features = data[feature_cols]
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
        X_text, X_tab, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = StockDataset(X_text_train, X_tab_train, y_train, tokenizer)
    val_ds = StockDataset(X_text_val, X_tab_val, y_val, tokenizer)

    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True)

    # Initialize model, loss, optimizer, scheduler
    model = MultimodalStockPredictor(tabular_dim=TABULAR_DIM, text_model_name=TEXT_MODEL_NAME).to(device)
    try:
        model = torch.compile(model)
        logging.info("Model compiled with torch.compile for performance.")
    except Exception as e:
        logging.warning(f"torch.compile not available or failed: {e}")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    writer = SummaryWriter(log_dir='runs/stock_model')

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        logging.info(f"\n[Epoch {epoch+1}/{EPOCHS}] Training...")

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

        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Prec/train', precision_score(train_labels, train_preds, average='macro', zero_division=0), epoch)
        writer.add_scalar('Rec/train', recall_score(train_labels, train_preds, average='macro', zero_division=0), epoch)
        writer.add_scalar('F1/train', f1_score(train_labels, train_preds, average='macro', zero_division=0), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}")


        # Validation
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Validation...")
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():  # Disable gradients for validation to save memory
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]", leave=False):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                tab = batch['tabular'].to(device)
                lbl = batch['label'].to(device)


            with torch.no_grad():
                logits = model({'input_ids': ids, 'attention_mask': mask}, tab)
                loss = loss_fn(logits, lbl)
            val_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            val_preds.extend(preds)
            val_labels.extend(lbl.cpu().tolist())

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

        # Log metrics
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('Prec/val', val_prec, epoch)
        writer.add_scalar('Rec/val', val_rec, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)


        print_metrics(epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_prec, val_rec, val_f1)
        print_confusion(val_labels, val_preds)


        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        
        # LR scheduler step

        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            logging.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered.")
                break

    writer.close()
    logging.info("Training complete.")
    logging.info("Best validation loss: %.4f", best_val_loss)

if __name__ == "__main__":
    main()
