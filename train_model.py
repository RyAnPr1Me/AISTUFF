import os
import torch
import pandas as pd
import logging
from transformers import AutoTokenizer
from src.models.stock_ai import MultimodalStockPredictor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
VALIDATED_DATA_PATH = "Training_Data/validated_data.csv"
MODEL_SAVE_PATH = "trained_model/model_weights.pth"
CHECKPOINT_DIR = "trained_model/checkpoints"

# Training parameters
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
TEXT_MODEL_NAME = "bert-large-uncased"
TABULAR_DIM = 64
EARLY_STOPPING_PATIENCE = 5

# Logging setup
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Dataset class
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

# Training script
def main():
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading
    logging.info(f"Loading data from {VALIDATED_DATA_PATH}")
    data = pd.read_csv(VALIDATED_DATA_PATH)
    if data.isnull().values.any():
        logging.warning("Data contains NaN values; filling with zeros.")
        data = data.fillna(0)

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    encoded = tokenizer(
        list(data["text"]), padding=True, truncation=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Process tabular
    if 'feature_0' not in data.columns:
        tabular_data = torch.randn(len(data), TABULAR_DIM)
    else:
        features = data[[c for c in data.columns if c.startswith("feature_")]]
        scaler = StandardScaler()
        tabular_data = torch.tensor(scaler.fit_transform(features.values), dtype=torch.float)
        # pad or truncate
        if tabular_data.size(1) < TABULAR_DIM:
            pad = torch.zeros(len(data), TABULAR_DIM - tabular_data.size(1))
            tabular_data = torch.cat([tabular_data, pad], dim=1)
        else:
            tabular_data = tabular_data[:, :TABULAR_DIM]

    labels = torch.tensor(data["label"].values, dtype=torch.long)

    # Split
    (X_ids_train, X_ids_val,
     X_mask_train, X_mask_val,
     t_train, t_val,
     y_train, y_val) = train_test_split(
         input_ids, attention_mask, tabular_data, labels, test_size=0.2, random_state=42
    )

    train_ds = StockDataset(X_ids_train, X_mask_train, t_train, y_train)
    val_ds = StockDataset(X_ids_val, X_mask_val, t_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = MultimodalStockPredictor(tabular_dim=TABULAR_DIM).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # TensorBoard
    writer = SummaryWriter(log_dir='runs/stock_model')

    # Resume training
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            state = torch.load(MODEL_SAVE_PATH)
            model.load_state_dict(state)
            logging.info(f"Loaded model from {MODEL_SAVE_PATH}")
        except Exception as e:
            logging.warning(f"Could not load existing model: {e}")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for batch in train_loader:
            batch_ids = batch['input_ids'].to(device)
            batch_mask = batch['attention_mask'].to(device)
            batch_tab = batch['tabular'].to(device)
            batch_lbl = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model({'input_ids': batch_ids, 'attention_mask': batch_mask}, batch_tab)
            loss = loss_fn(logits, batch_lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            train_preds.extend(preds)
            train_labels.extend(batch_lbl.cpu().tolist())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_ids = batch['input_ids'].to(device)
                batch_mask = batch['attention_mask'].to(device)
                batch_tab = batch['tabular'].to(device)
                batch_lbl = batch['label'].to(device)

                logits = model({'input_ids': batch_ids, 'attention_mask': batch_mask}, batch_tab)
                loss = loss_fn(logits, batch_lbl)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_labels.extend(batch_lbl.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('Prec/val', val_prec, epoch)
        writer.add_scalar('Rec/val', val_rec, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Val loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Check early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Saved best model to {MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered.")
                break

        # Epoch checkpoint
        cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), cp_path)

    writer.close()
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
