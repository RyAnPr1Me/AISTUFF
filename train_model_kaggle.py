import os
import sys
import json
import warnings
import torch
import pandas as pd
import logging
import traceback
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import argparse
from tqdm import tqdm

# Kaggle/Notebook compatibility
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        pass
    return False

def setup_logging():
    if in_notebook():
        warnings.filterwarnings('ignore', category=FutureWarning)
        print("Notebook mode: using print for logging.")
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        warnings.filterwarnings('ignore', category=FutureWarning)
        handler = logging.StreamHandler(sys.stdout)
        logger = logging.getLogger()
        logger.addHandler(handler)

class MemoryEfficientStockDataset(Dataset):
    def __init__(self, texts, tabular_data, labels, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tabular_data = tabular_data
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
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
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    if 'label' in data.columns:
        print(f"Label distribution: {data['label'].value_counts().to_dict()}")
        print(f"Label unique values: {data['label'].unique()}")
    if 'text' in data.columns:
        print(f"Sample text: {data['text'].iloc[0]}")
        print(f"Text length stats: min={data['text'].str.len().min()}, max={data['text'].str.len().max()}, mean={data['text'].str.len().mean():.1f}")
    print(f"First row: {data.iloc[0].to_dict()}")
    print(f"Missing values per column: {data.isnull().sum().to_dict()}")

def print_metrics(epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_prec, val_rec, val_f1):
    msg = (
        f"[Epoch {epoch+1}] "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}"
    )
    print(msg)

def main():
    # Configurations
    DATA_PATH = "/kaggle/input/stocks/optimized_data (2).csv"
    MODEL_DIR = "/kaggle/working/model"
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 1e-4
    TABULAR_DIM = 32
    MAX_SEQ_LEN = 64
    EARLY_STOPPING_PATIENCE = 2
    NOISE_STD = 0.01
    USE_AMP = torch.cuda.is_available()
    TEXT_MODEL_NAME = "bert-base-uncased"

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    print(f"Using device: {device}")

    # Load data
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(data)} rows and {len(data.columns)} columns from {DATA_PATH}")
    if data.isnull().values.any():
        print("Data contains NaN values; filling with zeros.")
        data = data.fillna(0)
    print_data_overview(data)

    # --- Time Series Preparation for TFT ---
    # Assume data has columns: group_id, time_idx, target, and features
    required_cols = ["group_id", "time_idx", "target"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column '{col}' for TFT input.")

    # Identify feature columns (exclude group_id, time_idx, target)
    feature_cols = [col for col in data.columns if col not in ["group_id", "time_idx", "target"]]
    print(f"Using feature columns: {feature_cols}")

    # Split into train/val by time_idx (80% train, 20% val)
    max_time = data["time_idx"].max()
    min_time = data["time_idx"].min()
    split_time = min_time + int(0.8 * (max_time - min_time))
    train_df = data[data["time_idx"] <= split_time].reset_index(drop=True)
    val_df = data[data["time_idx"] > split_time].reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Dataset and DataLoader for TFT
    class TFTStockDataset(Dataset):
        def __init__(self, df, feature_cols):
            self.df = df.reset_index(drop=True)
            self.feature_cols = feature_cols
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            return {
                "group_id": int(row["group_id"]),
                "time_idx": int(row["time_idx"]),
                "target": float(row["target"]),
                "features": torch.tensor(row[self.feature_cols].values, dtype=torch.float)
            }

    train_ds = TFTStockDataset(train_df, feature_cols)
    val_ds = TFTStockDataset(val_df, feature_cols)

    def tft_collate_fn(batch):
        group_id = torch.tensor([item["group_id"] for item in batch], dtype=torch.long)
        time_idx = torch.tensor([item["time_idx"] for item in batch], dtype=torch.long)
        target = torch.tensor([item["target"] for item in batch], dtype=torch.float)
        features = torch.stack([item["features"] for item in batch])
        return {
            "group_id": group_id,
            "time_idx": time_idx,
            "target": target,
            "features": features
        }

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=tft_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tft_collate_fn)

    print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")

    # --- TemporalFusionTransformer Model ---
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    # Prepare TimeSeriesDataSet for TFT
    max_encoder_length = 30
    max_prediction_length = 1
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"] + feature_cols,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["target"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)

    # TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=1,
        loss=torch.nn.MSELoss(),
        log_interval=10,
        reduce_on_plateau_patience=EARLY_STOPPING_PATIENCE,
    )

    # Trainer
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    print("Setting up PyTorch Lightning Trainer...")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=EARLY_STOPPING_PATIENCE, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    progress_bar = TQDMProgressBar(refresh_rate=1)
    print(f"Trainer will save checkpoints to {MODEL_DIR}")
    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop_callback, checkpoint_callback, progress_bar],
        gradient_clip_val=0.1,
        logger=False,
    )

    print("Starting TFT model training...")
    # Train
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    print("Training complete.")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "tft_model.ckpt")
    trainer.save_checkpoint(model_path)
    print(f"Final TFT model checkpoint saved to {model_path}")

    # Log best checkpoint
    if checkpoint_callback.best_model_path:
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    else:
        print("No best model checkpoint found.")

    print("All done.")

if __name__ == "__main__":
    main()
