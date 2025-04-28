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
from pathlib import Path  # Ensure this import is present

# Add the src directory to the Python path
try:
    src_path = str(Path(__file__).resolve().parent / 'src')  # Use __file__ if available
except (NameError, AttributeError):
    src_path = str(Path.cwd() / 'src')  # Fallback for environments where __file__ is not defined
sys.path.append(src_path)

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
    # Check and create required columns if missing
    required_cols = ["group_id", "time_idx", "target"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"WARNING: Missing columns {missing_cols}. Attempting to create them for demo/training purposes.")
        if "group_id" in missing_cols:
            data["group_id"] = 0  # Assign all to one group
        if "time_idx" in missing_cols:
            # Try to infer time_idx from a date column, else use row index
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
            if date_cols:
                data = data.sort_values(by=date_cols[0]).reset_index(drop=True)
                data["time_idx"] = np.arange(len(data))
            else:
                data["time_idx"] = np.arange(len(data))
        if "target" in missing_cols:
            # Try to use 'label' or the last column as target
            if "label" in data.columns:
                data["target"] = data["label"]
            else:
                data["target"] = data.iloc[:, -1]
        print(f"After filling, columns are: {list(data.columns)}")

    # Identify feature columns (exclude group_id, time_idx, target)
    feature_cols = [col for col in data.columns if col not in ["group_id", "time_idx", "target", "text"]]
    
    # Clean and prepare the target column
    if "target" in data.columns:
        try:
            # Try to convert target to float if it's not already
            data["target"] = pd.to_numeric(data["target"], errors="coerce")
            if data["target"].isna().any():
                print(f"WARNING: {data['target'].isna().sum()} NaN values detected in target after conversion. Filling with 0.")
                data["target"] = data["target"].fillna(0)
        except Exception as e:
            print(f"Error converting target to numeric: {e}")
            print("Will use label column instead if available")
            if "label" in data.columns:
                data["target"] = pd.to_numeric(data["label"], errors="coerce").fillna(0)
    
    # Identify and handle categorical features
    categorical_cols = []
    continuous_cols = []
    
    for col in feature_cols:
        try:
            # Try to convert to numeric
            numeric_col = pd.to_numeric(data[col], errors="coerce")
            # If more than 20% of values are NaN after conversion, treat as categorical
            if numeric_col.isna().mean() > 0.2 or pd.api.types.is_object_dtype(data[col]):
                categorical_cols.append(col)
                # For categorical features, ensure they're strings
                data[col] = data[col].astype(str)
                print(f"Column {col} treated as categorical with {data[col].nunique()} unique values")
            else:
                # Replace NaNs with 0 in numeric columns
                data[col] = numeric_col.fillna(0)
                continuous_cols.append(col)
                print(f"Column {col} treated as continuous")
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            # Default to categorical if there's an error
            categorical_cols.append(col)
            data[col] = data[col].astype(str)
    
    print(f"Using {len(categorical_cols)} categorical features and {len(continuous_cols)} continuous features")
    print(f"Categorical features: {categorical_cols}")
    print(f"Continuous features: {continuous_cols}")

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
    from pytorch_forecasting.data import NaNLabelEncoder  # Import NaNLabelEncoder for categorical features
    
    max_encoder_length = 30
    max_prediction_length = 1
    
    # Create encoders for categorical variables
    categorical_encoders = {}
    for col in categorical_cols:
        try:
            encoder = NaNLabelEncoder(add_nan=True)
            encoder.fit(train_df[col].astype(str))  # Convert to string to ensure encoder works
            categorical_encoders[col] = encoder
        except Exception as e:
            print(f"Error creating encoder for column {col}: {e}")
            # Remove problematic columns from categorical_cols
            categorical_cols.remove(col)
    
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=categorical_cols,
        time_varying_known_reals=["time_idx"] + continuous_cols,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["target"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        categorical_encoders=categorical_encoders,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)

    # TFT model
    import torchmetrics
    import warnings

    # Configure hyperparameters for TFT model
    tft_hparams = {
        "learning_rate": LR,
        "hidden_size": 16,
        "attention_head_size": 1, 
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "output_size": 1,
        "log_interval": 10,
        "reduce_on_plateau_patience": EARLY_STOPPING_PATIENCE,
    }
    
    # Suppress specific warnings we expect during model creation
    warnings.filterwarnings("ignore", message=".*nn.Module.*is already saved during checkpointing.*")
    warnings.filterwarnings("ignore", message=".*`TorchScript` support for functional optimizers is deprecated.*")
    
    # Set environment variables to suppress CUDA errors
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"  # Help XLA find CUDA
    
    # Create the TFT model with simpler parameters to avoid compatibility issues
    from src.models.tft_model import TFTStockPredictor

    # Replace direct usage of TemporalFusionTransformer with TFTStockPredictor
    # Wrap the model creation and loading in TFTStockPredictor
    print("Creating TFT model wrapped in TFTStockPredictor...")
    tft = TFTStockPredictor.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=16,
        dropout=0.1,
        hidden_continuous_size=8,
        reduce_on_plateau_patience=EARLY_STOPPING_PATIENCE
    )
    
    print("Created TFT model - continuing with training")
    
    # Trainer with additional configurations to avoid CUDA warnings
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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

    # Make predictions with best model
    print("Making predictions with best model...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        # Predict on validation set
        try:
            # Try the standard prediction approach
            print("Getting predictions from model...")
            pred_result = best_tft.predict(val_dataloader, return_y=True)
            
            # Handle different return formats
            if isinstance(pred_result, tuple) and len(pred_result) >= 2:
                print("Predictions returned as tuple with multiple elements")
                predictions = pred_result[0].detach().cpu()
                actuals = pred_result[1].detach().cpu()
            else:
                print("Predictions returned in different format - attempting to extract")
                predictions = pred_result.detach().cpu() if hasattr(pred_result, 'detach') else pred_result
                # Try to get actuals from dataloader
                actuals = torch.cat([batch["target"] for batch in iter(val_dataloader)]).cpu()
            
            print(f"Predictions shape: {predictions.shape}, Actuals shape: {actuals.shape}")
            
            # Flatten arrays if needed
            predictions = predictions.reshape(-1)
            actuals = actuals.reshape(-1)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = np.sqrt(mean_squared_error(actuals.numpy(), predictions.numpy()))
            mae = mean_absolute_error(actuals.numpy(), predictions.numpy())
            
            print(f"Validation RMSE: {rmse:.4f}")
            print(f"Validation MAE: {mae:.4f}")
            
            # Save metrics
            metrics_path = os.path.join(MODEL_DIR, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump({
                    'rmse': float(rmse), 
                    'mae': float(mae)
                }, f)
            print(f"Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error making predictions: {e}")
            traceback.print_exc()
            
        # Try to visualize predictions
        try:
            import matplotlib.pyplot as plt
            # Plot predictions vs actuals
            plt.figure(figsize=(12, 6))
            plt.plot(actuals[:100], label='Actual')
            plt.plot(predictions[:100], label='Predicted')
            plt.legend()
            plt.title('TFT Predictions on Validation Data (first 100 points)')
            plt.tight_layout()
            pred_plot_path = os.path.join(MODEL_DIR, "predictions.png")
            plt.savefig(pred_plot_path)
            print(f"Prediction plot saved to {pred_plot_path}")
        except Exception as e:
            print(f"Could not create prediction plot: {e}")
    
    print("All done.")

if __name__ == "__main__":
    main()
