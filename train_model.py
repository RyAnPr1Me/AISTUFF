import os
import sys
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Enable GPU optimization with even more aggressive memory saving
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Limit CUDA memory splits

try:
    import warnings
    warnings.filterwarnings("ignore")

    import torch
    import pandas as pd
    import logging
    import traceback
    import gc  # For explicit garbage collection
    from transformers import AutoTokenizer, AutoConfig
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import time
    import argparse
    # Remove src.* imports for Kaggle/Colab/Notebook compatibility
    # from src.models.stock_ai import MultimodalStockPredictor
    # from src.models.dataloader import multimodal_collate_fn, tft_collate_fn
    from tqdm import tqdm
    
    # Print diagnostic information
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Working directory: {os.getcwd()}")
except Exception as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    import traceback as tb
    tb.print_exc()
    # Do not call sys.exit(1) in notebook environments
    # Do not raise, just print and continue in notebook
    if not 'get_ipython' in globals() or not hasattr(get_ipython(), 'config'):
        sys.exit(1)

# Define device - use GPU, TPU, or CPU when available
if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print("Using TPU device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")
print(f"Using device: {device}")

# Check for AMP (automatic mixed precision) support
AMP_AVAILABLE = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
if AMP_AVAILABLE:
    from torch.cuda.amp import autocast, GradScaler
    print("Automatic Mixed Precision (AMP) is available and will be used")

#========================================================================
# Low-memory SageMaker-compatible train model script
#========================================================================

# Paths and configurations - Kaggle settings
VALIDATED_DATA_PATH = "/kaggle/input/orion/pytorch/small/1/validated_data.csv"
OPTIMIZED_DATA_PATH = "/kaggle/input/orion/pytorch/small/1/optimized_data.csv"
DEFAULT_MODEL_DIR = "/kaggle/working/model"

# Training parameters - Low memory optimized
BATCH_SIZE = 16  # Smaller batch size for lower memory usage
GRADIENT_ACCUMULATION_STEPS = 4  # Use more gradient accumulation steps to compensate for smaller batch
EPOCHS = 5
LR = 1e-4
TEXT_MODEL_NAME = "TemporalFusionTransformer"  # SageMaker compatible model
TABULAR_DIM = 32  # Reduced dimension for tabular data
MAX_TRAIN_SECONDS = 3 * 60 * 60  # 3 hour time limit for SageMaker

# Performance optimizations - Low memory settings
USE_AMP = True  # Will be auto-disabled if not available
FREEZE_TEXT_ENCODER = True  # Completely freeze text encoder for lowest memory usage
TOKENIZER_BATCH_SIZE = 128  # Much smaller batch size for tokenization
MAX_SEQ_LEN = 64  # Reduced sequence length for lower memory
EARLY_STOPPING_PATIENCE = 2  # Set patience to 2 epochs for early stopping
NOISE_STD = 0.01  # Standard deviation for Gaussian noise injection

# Kaggle/Notebook compatibility: auto-detect notebook, adjust logging, and avoid sys.exit
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        pass
    return False

def setup_logging():
    # Use print for notebook, logging for scripts
    if in_notebook():
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        print("Notebook mode: using print for logging.")
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        warnings.filterwarnings('ignore', category=FutureWarning)
        # Ensure we're writing to stdout for SageMaker log collection
        handler = logging.StreamHandler(sys.stdout)
        logger = logging.getLogger()
        logger.addHandler(handler)

class MemoryEfficientStockDataset(Dataset):
    """Memory efficient dataset that tokenizes on-the-fly"""
    def __init__(self, texts, tabular_data, labels, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tabular_data = tabular_data
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # On-the-fly tokenization to save memory
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

class TFTStockDataset(Dataset):
    """
    Dataset for TFT: expects DataFrame with group_id, time_idx, target, and features.
    """
    def __init__(self, df, feature_cols):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            "group_id": int(row["group_id"]),
            "time_idx": int(row["time_idx"]),
            "target": float(row["target"]),
            "features": torch.tensor(row[self.feature_cols].values, dtype=torch.float)
        }
        return item

def print_data_overview(data):
    if in_notebook():
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
    else:
        logging.info(f"Data shape: {data.shape}")
        logging.info(f"Columns: {list(data.columns)}")
        if 'label' in data.columns:
            logging.info(f"Label distribution: {data['label'].value_counts().to_dict()}")
            logging.info(f"Label unique values: {data['label'].unique()}")
        if 'text' in data.columns:
            logging.info(f"Sample text: {data['text'].iloc[0]}")
            logging.info(f"Text length stats: min={data['text'].str.len().min()}, max={data['text'].str.len().max()}, mean={data['text'].str.len().mean():.1f}")
        logging.info(f"First row: {data.iloc[0].to_dict()}")
        logging.info(f"Missing values per column: {data.isnull().sum().to_dict()}")

def print_metrics(epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_prec, val_rec, val_f1):
    msg = (
        f"[Epoch {epoch+1}] "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}"
    )
    if in_notebook():
        print(msg)
    else:
        logging.info(msg)

def print_confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if in_notebook():
        print(f"Validation Confusion Matrix:\n{cm}")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Validation Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()
        except Exception:
            print("matplotlib/seaborn not available for confusion matrix plot.")
    else:
        logging.info(f"Validation Confusion Matrix:\n{cm}")

def optimize_data_for_ai(df, label_col='label', text_col='text', corr_thresh=0.98, min_var=1e-8):
    """
    Optimize tabular data for AI model:
    - Remove constant columns.
    - Remove highly correlated features.
    - Remove features with very low variance.
    - Log dropped columns.
    Returns optimized DataFrame.
    """
    drop_cols = []

    # Remove constant columns
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        drop_cols.extend(const_cols)
        logging.info(f"Data optimizer: Dropping constant columns: {const_cols}")

    # Remove low variance columns (excluding label/text)
    var = df.var(numeric_only=True)
    low_var_cols = [col for col in var.index if var[col] < min_var and col not in [label_col, text_col]]
    if low_var_cols:
        drop_cols.extend(low_var_cols)
        logging.info(f"Data optimizer: Dropping low-variance columns: {low_var_cols}")

    # Remove highly correlated columns (keep only one from each group)
    feature_cols = [col for col in df.columns if col not in [label_col, text_col]]
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    if to_drop:
        drop_cols.extend(to_drop)
        logging.info(f"Data optimizer: Dropping highly correlated columns: {to_drop}")

    drop_cols = list(set(drop_cols))
    df_optimized = df.drop(columns=drop_cols, errors='ignore')
    logging.info(f"Data optimizer: Final columns: {list(df_optimized.columns)}")
    return df_optimized

def main():
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--input-data', type=str, default=OPTIMIZED_DATA_PATH)
    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR)
    # Add the new hyperparameters
    parser.add_argument('--disable_mixed_precision', type=str, default='false')
    parser.add_argument('--disable_self_attention', type=str, default='false')
    parser.add_argument('--fusion_type', type=str, default='concat')
    parser.add_argument(
        '--tft', action='store_true',
        help='Train with Temporal Fusion Transformer'
    )
    parser.add_argument(
        '--tft-meta', type=str, default=None,
        help='Path to TFT metadata JSON file'
    )
    parser.add_argument(
        '--max-encoder-length', type=int, default=30,
        help='Max encoder length for TFT (lookback period)'
    )
    parser.add_argument(
        '--max-prediction-length', type=int, default=1,
        help='Max prediction length for TFT (forecast horizon)'
    )
    # Enhanced time series training arguments
    parser.add_argument('--ts-mode', action='store_true', help='Use time series specific optimizations')
    parser.add_argument('--backtest-windows', type=int, default=3, help='Number of windows for time series backtesting')
    parser.add_argument('--seasonal-period', type=int, default=5, help='Seasonal period for time features (e.g. 5 for weekly)')
    parser.add_argument('--fourier-terms', type=int, default=2, help='Number of Fourier terms for seasonality')
    parser.add_argument('--decompose-trend', action='store_true', help='Use trend decomposition')
    parser.add_argument('--differencing', action='store_true', help='Apply first differencing to make data stationary')
    parser.add_argument('--quantile-forecast', action='store_true', help='Use quantile forecasting for uncertainty estimation')
    parser.add_argument('--optimization-metric', type=str, default='SMAPE', 
                      choices=['RMSE', 'MAE', 'MAPE', 'SMAPE', 'MASE'], help='Metric to optimize')
    parser.add_argument('--hp-tuning-trials', type=int, default=0, help='Number of hyperparameter tuning trials (0 to disable)')
    
    args = parser.parse_args([] if in_notebook() else None)  # Use defaults in notebook

    # Start timing
    start_time = time.time()

    # Use Kaggle-friendly paths
    input_data_path = args.input_data
    model_dir = args.model_dir

    # Check if input_data_path is a directory and append the file name if necessary
    if os.path.isdir(input_data_path):
        input_data_path = os.path.join(input_data_path, 'optimized_data.csv')

    if not os.path.isfile(input_data_path):
        msg = f"Input data file not found: {input_data_path}"
        if in_notebook():
            print(msg)
            raise FileNotFoundError(msg)
        else:
            logging.error(msg)
            sys.exit(1)

    # Load the data
    data = pd.read_csv(input_data_path)
    logging.info(f"Loaded {len(data)} rows and {len(data.columns)} columns from {input_data_path}")
    if data.isnull().values.any():
        logging.warning("Data contains NaN values; filling with zeros.")
        data = data.fillna(0)

    print_data_overview(data)

    # --- Data optimizer step ---
    # (No need to call optimize_data_for_ai here, already optimized)

    # Tokenizer
    tokenizer = None
    if hasattr(args, 'tft') and not args.tft:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    # Process tabular features
    forbidden_features = {'future_close', 'weekly_return'}
    feature_cols = [col for col in data.columns if col not in ['text', 'label'] and col not in forbidden_features]
    if not feature_cols:
        msg = "No tabular features found in data. Exiting."
        if in_notebook():
            print(msg)
            return
        else:
            logging.error(msg)
            return
    
    # Ensure temporal split: sort by date if available
    date_cols = [col for col in data.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
    if date_cols:
        data = data.sort_values(by=date_cols[0]).reset_index(drop=True)
        logging.info(f"Sorted data by {date_cols[0]} for temporal split.")
    
    features = data[feature_cols].copy()
    labels = torch.tensor(data["label"].values, dtype=torch.long)
    X_text = list(data["text"])
    
    # Temporal split: 80% train, 20% val
    split_idx = int(0.8 * len(data))
    X_text_train, X_text_val = X_text[:split_idx], X_text[split_idx:]
    features_train, features_val = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    
    # Fit scaler only on training data
    scaler = StandardScaler()
    X_tab_train = torch.tensor(scaler.fit_transform(features_train.values), dtype=torch.float)
    X_tab_val = torch.tensor(scaler.transform(features_val.values), dtype=torch.float)
    
    # Pad/truncate to TABULAR_DIM
    def pad_tab(tab):
        if tab.size(1) < TABULAR_DIM:
            pad = torch.zeros(tab.size(0), TABULAR_DIM - tab.size(1))
            return torch.cat([tab, pad], dim=1)
        else:
            return tab[:, :TABULAR_DIM]
    X_tab_train = pad_tab(X_tab_train)
    X_tab_val = pad_tab(X_tab_val)

    if args.tft:
        try:
            # Import needed libraries
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
            from pytorch_lightning.loggers import TensorBoardLogger
            
            # Load the data with enhanced validation
            data = pd.read_csv(args.data_path)
            logging.info(f"Loaded data with shape {data.shape}")
            
            # Load metadata or use defaults
            if args.tft_meta:
                with open(args.tft_meta, 'r') as f:
                    meta = json.load(f)
            else:
                # Default metadata
                meta = {
                    "time_idx": "time_idx",
                    "target": "target",
                    "group_ids": ["group_id"],
                    "max_encoder_length": args.max_encoder_length if hasattr(args, 'max_encoder_length') else 30,
                    "max_prediction_length": args.max_prediction_length if hasattr(args, 'max_prediction_length') else 1,
                }
                logging.warning(f"No metadata file provided, using default: {meta}")
            
            # Data validation
            for required_col in [meta["time_idx"], meta["target"]] + meta["group_ids"]:
                if required_col not in data.columns:
                    raise ValueError(f"Missing required column '{required_col}' for TFT")
            
            # Set up data module with enhanced options
            from src.models.dataloader import TFTDataModule
            data_module = TFTDataModule(
                data,
                time_idx=meta["time_idx"],
                target=meta["target"],
                group_ids=meta["group_ids"],
                max_encoder_length=meta.get("max_encoder_length", 30),
                max_prediction_length=meta.get("max_prediction_length", 1),
                batch_size=args.batch_size,
                num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
                # Enhanced time series options
                add_encoder_length=True,
                add_relative_time=True,
                add_target_scales=True,
                add_lagged_features=args.ts_mode,
                decomposition=args.decompose_trend
            )
            
            # Prepare data
            data_module.prepare_data()
            data_module.setup()
            
            # Create TFT model with enhanced metrics
            from src.models.tft_model import TFTStockPredictor
            tft_model = TFTStockPredictor(
                hidden_size=args.hidden_dim if hasattr(args, 'hidden_dim') else 64,
                lstm_layers=args.lstm_layers if hasattr(args, 'lstm_layers') else 2,
                dropout=args.dropout,
                max_encoder_length=meta.get("max_encoder_length", 30),
                max_prediction_length=meta.get("max_prediction_length", 1),
                learning_rate=args.lr,
                # Enhanced time series options
                use_quantiles=args.quantile_forecast,
                seasonal_period=args.seasonal_period,
                optimization_metric=args.optimization_metric
            )
            
            # Create the model
            model = tft_model.create_model(data_module.train_dataset)
            
            # Set up callbacks
            early_stop_callback = EarlyStopping(
                monitor="val_loss", 
                min_delta=1e-4, 
                patience=10, 
                verbose=True, 
                mode="min"
            )
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_dir,
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            )
            
            # Set up logger
            logger = TensorBoardLogger(save_dir=model_dir, name="tft_logs")
            
            # Set up trainer
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                accelerator='auto',
                devices="auto",
                callbacks=[early_stop_callback, checkpoint_callback],
                gradient_clip_val=0.1,
                logger=logger,
            )
            
            # Train
            trainer.fit(
                model=tft_model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader()
            )
            
            # Save metadata along with the model
            model_dir_final = os.path.join(model_dir, "final_model")
            os.makedirs(model_dir_final, exist_ok=True)
            
            # Save the model
            trainer.save_checkpoint(os.path.join(model_dir_final, "tft_model.ckpt"))
            
            # Save the training metadata
            with open(os.path.join(model_dir_final, "tft_metadata.json"), "w") as f:
                json.dump({
                    **meta,
                    "model_params": tft_model.hparams,
                    "train_data_params": {
                        "batch_size": args.batch_size,
                        "max_encoder_length": meta.get("max_encoder_length", 30),
                        "max_prediction_length": meta.get("max_prediction_length", 1)
                    },
                    "data_cols": list(data.columns),
                    "time_varying_known_reals": data_module.time_varying_known_reals,
                    "time_varying_unknown_reals": data_module.time_varying_unknown_reals,
                }, f, indent=2, default=str)
            
            logging.info(f"Model saved to {model_dir_final}")
            
            # Hyperparameter tuning if requested
            if args.hp_tuning_trials > 0:
                logging.info(f"Running hyperparameter optimization with {args.hp_tuning_trials} trials...")
                from src.models.ts_utils import optimize_hyperparameters
                
                best_params = optimize_hyperparameters(
                    data_module,
                    num_trials=args.hp_tuning_trials,
                    max_epochs=min(args.epochs, 20),
                    optimization_metric=args.optimization_metric
                )
                
                # Update model with best parameters
                for param, value in best_params.items():
                    if hasattr(tft_model, param):
                        setattr(tft_model, param, value)
                
                # Recreate model with optimized parameters
                model = tft_model.create_model(data_module.train_dataset)
                logging.info(f"Using optimized hyperparameters: {best_params}")
            
            # Enhanced evaluation with backtesting
            if args.backtest_windows > 0 and args.ts_mode:
                logging.info(f"Performing time series backtesting with {args.backtest_windows} windows...")
                from src.models.ts_utils import run_backtesting
                
                backtest_results = run_backtesting(
                    model=tft_model,
                    df=data,
                    time_idx=meta["time_idx"],
                    target=meta["target"],
                    group_ids=meta["group_ids"],
                    max_encoder_length=meta.get("max_encoder_length", 30),
                    max_prediction_length=meta.get("max_prediction_length", 1),
                    num_windows=args.backtest_windows,
                    trainer=trainer
                )
                
                # Save backtest results
                backtest_file = os.path.join(model_dir_final, "backtest_results.json")
                with open(backtest_file, "w") as f:
                    json.dump(backtest_results, f, indent=2)
                
                logging.info(f"Backtest results:\n{json.dumps(backtest_results['summary'], indent=2)}")
            
        except ImportError as e:
            msg = f"Error importing TFT dependencies: {e}\nMake sure pytorch-forecasting and pytorch-lightning are installed"
            if in_notebook():
                print(msg)
                raise
            else:
                logging.error(msg)
                raise
    else:
        # Only use tokenizer in non-TFT mode
        if not args.tft:
            try:
                train_ds = MemoryEfficientStockDataset(X_text_train, X_tab_train, y_train, tokenizer)
                val_ds = MemoryEfficientStockDataset(X_text_val, X_tab_val, y_val, tokenizer)

                # DataLoader optimizations for Kaggle/TPU
                num_workers = 2
                pin_memory = (device.type == "cuda")
                persistent_workers = pin_memory

                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True,
                    pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers
                )
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=False,
                    pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers
                )

                print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")
                print(f"Batch size: {args.batch_size} | Num workers: {num_workers}")

                # Initialize model, loss, optimizer, scheduler
                try:
                    model = MultimodalStockPredictor(
                        tabular_dim=TABULAR_DIM, 
                        text_model_name=TEXT_MODEL_NAME,
                        use_mixed_precision=(AMP_AVAILABLE and USE_AMP),
                        use_self_attention=(args.disable_self_attention.lower() != 'true'),
                        fusion_type=args.fusion_type
                    ).to(device)
                    
                    # Freeze transformer layers for faster training if specified
                    if FREEZE_TEXT_ENCODER and hasattr(model, 'text_encoder'):
                        # Keep only the last 2 layers unfrozen for fine-tuning
                        if hasattr(model.text_encoder, 'encoder'):
                            # For ALBERT architecture
                            num_layers = len(model.text_encoder.encoder.albert_layer_groups)
                            for i, layer in enumerate(model.text_encoder.encoder.albert_layer_groups):
                                if i < num_layers - 2:  # Freeze all except last 2 layers
                                    for param in layer.parameters():
                                        param.requires_grad = False
                            logging.info(f"Froze {num_layers-2}/{num_layers} transformer encoder layers")
                        elif hasattr(model.text_encoder, 'layer'):
                            # For BERT architecture  
                            num_layers = len(model.text_encoder.layer)
                            for i in range(num_layers - 2):
                                for param in model.text_encoder.layer[i].parameters():
                                    param.requires_grad = False
                            logging.info(f"Froze {num_layers-2}/{num_layers} transformer encoder layers")
                    
                    logging.info("Successfully initialized MultimodalStockPredictor model")
                except ImportError as e:
                    if "torch.distributed.tensor" in str(e):
                        logging.error("ImportError with torch.distributed.tensor. Using simpler model configuration.")
                        # Fall back to a simpler model configuration
                        model = MultimodalStockPredictor(
                            tabular_dim=TABULAR_DIM, 
                            text_model_name=TEXT_MODEL_NAME,
                            use_mixed_precision=False,  # Disable mixed precision
                            use_self_attention=False,   # Disable self-attention
                            fusion_type='concat'        # Use simple concatenation fusion
                        ).to(device)
                    else:
                        # Re-raise if it's another import error
                        raise

                try:
                    # torch.compile may not be available, so skip if not
                    if hasattr(torch, "compile"):
                        # Check if we're using a PyTorch version that supports compile
                        compile_supported = True
                        # Safely check if torch.compiler.is_compiling exists to avoid the error
                        if hasattr(torch, "compiler") and not hasattr(torch.compiler, "is_compiling"):
                            compile_supported = False
                            logging.warning("torch.compiler.is_compiling not found. Disabling compilation.")
                        
                        if compile_supported:
                            try:
                                model = torch.compile(model)
                                logging.info("Model compiled with torch.compile for performance.")
                            except Exception as e:
                                logging.warning(f"torch.compile failed: {e}")
                    else:
                        logging.info("torch.compile not available in this PyTorch version.")
                except Exception as e:
                    logging.warning(f"Error when attempting to use torch.compile: {e}")
                
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)  # L2 regularization
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

                scaler = GradScaler() if AMP_AVAILABLE and USE_AMP else None
                best_val_loss = float('inf')
                epochs_no_improve = 0
                history = {
                    'train_loss': [], 'train_acc': [],
                    'val_loss': [], 'val_acc': [],
                    'val_f1': [], 'lr': []
                }

                for epoch in range(args.epochs):
                    epoch_start_time = time.time()
                    elapsed = time.time() - start_time
                    if elapsed > MAX_TRAIN_SECONDS:
                        logging.warning(f"Time limit of {MAX_TRAIN_SECONDS/3600:.2f} hours reached. Stopping training.")
                        break

                    print(f"Epoch {epoch+1}/{args.epochs} started.")

                    model.train()
                    train_loss, train_preds, train_labels = 0.0, [], []
                    optimizer.zero_grad()
                    progress_bar = tqdm(
                        train_loader, 
                        desc=f"Epoch {epoch+1}/{args.epochs}",
                        leave=False,
                        ncols=80,
                        disable=False
                    )
                    steps_since_update = 0
                    accumulated_loss = 0
                    for batch_idx, batch in enumerate(progress_bar):
                        ids = batch['input_ids'].to(device)
                        mask = batch['attention_mask'].to(device)
                        tab = batch['tabular'].to(device)
                        lbl = batch['label'].to(device)
                        # Inject small Gaussian noise into tabular features during training
                        if model.training:
                            tab = tab + torch.randn_like(tab) * NOISE_STD
                        text_inputs = {'input_ids': ids, 'attention_mask': mask}
                        
                        # Mixed precision forward pass
                        if AMP_AVAILABLE and USE_AMP:
                            with autocast():
                                logits = model(text_inputs, tab)
                                loss = loss_fn(logits, lbl)
                                # Scale the loss for gradient accumulation
                                loss = loss / GRADIENT_ACCUMULATION_STEPS
                            
                            # Backward pass with scaled gradients
                            scaler.scale(loss).backward()
                            accumulated_loss += loss.item()
                            
                            # Update weights after accumulating gradients
                            steps_since_update += 1
                            if steps_since_update == GRADIENT_ACCUMULATION_STEPS or (batch_idx + 1) == len(train_loader):
                                # Unscale gradients and clip
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                
                                # Update weights
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                                
                                # Log accumulated loss
                                steps_since_update = 0
                                train_loss += accumulated_loss
                                accumulated_loss = 0
                        else:
                            # Standard forward pass
                            logits = model(text_inputs, tab)
                            loss = loss_fn(logits, lbl)
                            loss = loss / GRADIENT_ACCUMULATION_STEPS
                            
                            # Backward pass
                            loss.backward()
                            accumulated_loss += loss.item()
                            
                            # Update weights after accumulating gradients
                            steps_since_update += 1
                            if steps_since_update == GRADIENT_ACCUMULATION_STEPS or (batch_idx + 1) == len(train_loader):
                                # Clip gradients
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                
                                # Update weights
                                optimizer.step()
                                optimizer.zero_grad()
                                
                                # Log accumulated loss
                                steps_since_update = 0
                                train_loss += accumulated_loss
                                accumulated_loss = 0
                        
                        # Track predictions for metrics (use detach to avoid memory leaks)
                        with torch.no_grad():
                            preds = logits.argmax(dim=1).cpu().tolist()
                            train_preds.extend(preds)
                            train_labels.extend(lbl.cpu().tolist())
                        
                        # Update progress bar
                        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                        
                        # Log less frequently to reduce overhead
                        if (batch_idx + 1) % (max(1, len(train_loader) // 5)) == 0 or (batch_idx + 1) == len(train_loader):
                            logging.info(
                                f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)}: "
                                f"Loss: {loss.item():.4f}"
                            )

                    avg_train_loss = train_loss * GRADIENT_ACCUMULATION_STEPS / len(train_loader)
                    train_acc = accuracy_score(train_labels, train_preds)
                    
                    # Store metrics
                    history['train_loss'].append(avg_train_loss)
                    history['train_acc'].append(train_acc)
                    history['lr'].append(optimizer.param_groups[0]['lr'])

                    # Log training metrics
                    logging.info(
                        f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f} "
                        f"(Processed {len(train_labels)} samples in {len(train_loader)} batches)"
                    )

                    # Validation
                    logging.info(f"Validating model...")
                    model.eval()
                    val_loss, val_preds, val_labels = 0.0, [], []
                    
                    # Less verbose validation
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_loader):
                            ids = batch['input_ids'].to(device)
                            mask = batch['attention_mask'].to(device)
                            tab = batch['tabular'].to(device)
                            lbl = batch['label'].to(device)
                            text_inputs = {'input_ids': ids, 'attention_mask': mask}
                            
                            if AMP_AVAILABLE and USE_AMP:
                                with autocast():
                                    logits = model(text_inputs, tab)
                                    loss = loss_fn(logits, lbl)
                            else:
                                logits = model(text_inputs, tab)
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
                    
                    # Store metrics
                    history['val_loss'].append(avg_val_loss)
                    history['val_acc'].append(val_acc)
                    history['val_f1'].append(val_f1)

                    # Log metrics
                    print_metrics(epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_prec, val_rec, val_f1)
                    
                    # LR scheduler step
                    scheduler.step(avg_val_loss)

                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        logging.info(f"New best validation loss: {best_val_loss:.4f}")
                        
                        # Save the best model
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, "best_model.pth")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_val_loss,
                            'accuracy': val_acc,
                            'f1': val_f1
                        }, model_path)
                        logging.info(f"Saved best model checkpoint to {model_path}")
                    else:
                        epochs_no_improve += 1
                        logging.info(f"No improvement for {epochs_no_improve} epoch(s).")
                        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                            logging.info("Early stopping triggered.")
                            break

                    # Log epoch time
                    epoch_time = time.time() - epoch_start_time
                    logging.info(f"Epoch completed in {epoch_time/60:.2f} minutes")
                    
                    elapsed = time.time() - start_time
                    if elapsed > MAX_TRAIN_SECONDS:
                        logging.warning(f"Time limit of {MAX_TRAIN_SECONDS/3600:.2f} hours reached. Stopping training.")
                        break

                print(f"Epoch {epoch+1}/{args.epochs} finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                print("Training complete.")
                print("Best validation loss: %.4f" % best_val_loss)

                # --- Save model weights to Kaggle working directory ---
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "model_weights.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Final model weights saved to {model_path}")

                # Try to save training history as a plot
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 8))
                    
                    plt.subplot(2, 2, 1)
                    plt.plot(history['train_loss'], label='Train Loss')
                    plt.plot(history['val_loss'], label='Val Loss')
                    plt.legend()
                    plt.title('Loss')
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(history['train_acc'], label='Train Acc')
                    plt.plot(history['val_acc'], label='Val Acc')
                    plt.legend()
                    plt.title('Accuracy')
                    
                    plt.subplot(2, 2, 3)
                    plt.plot(history['val_f1'], label='Val F1')
                    plt.legend()
                    plt.title('F1 Score')
                    
                    plt.subplot(2, 2, 4)
                    plt.plot(history['lr'], label='Learning Rate')
                    plt.legend()
                    plt.title('Learning Rate')
                    
                    plt.tight_layout()
                    history_path = os.path.join(model_dir, "training_history.png")
                    plt.savefig(history_path)
                    print(f"Saved training history plot to {history_path}")
                except Exception as e:
                    print(f"Could not save training history plot: {e}")

            # Catch-all for errors in the non-TFT branch
            except Exception as e:
                msg = f"An error occurred during training: {e}"
                if in_notebook():
                    print(msg)
                    import traceback as tb
                    tb.print_exc()
                    raise
                else:
                    logging.error(msg)
                    traceback.print_exc()
                    sys.exit(1)

if __name__ == "__main__":
    main()
