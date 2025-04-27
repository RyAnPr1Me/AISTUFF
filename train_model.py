import os
import sys

# Enable GPU optimization with SageMaker compatibility
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # Reduced logging verbosity

try:
    import warnings
    warnings.filterwarnings("ignore")

    import torch
    import pandas as pd
    import logging
    import traceback
    from transformers import AutoTokenizer, AutoConfig
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    import time
    import argparse
    from src.models.stock_ai import MultimodalStockPredictor
    from tqdm import tqdm
    
    # Print diagnostic information
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
except Exception as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Define device - use GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check for AMP (automatic mixed precision) support
AMP_AVAILABLE = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
if AMP_AVAILABLE:
    from torch.cuda.amp import autocast, GradScaler
    print("Automatic Mixed Precision (AMP) is available and will be used")

#========================================================================
# SageMaker-compatible train model script with performance optimizations
#========================================================================

# Paths and configurations - SageMaker friendly defaults
VALIDATED_DATA_PATH = "Training_Data/validated_data.csv"

# Training parameters - optimized defaults for SageMaker
BATCH_SIZE = 32  # More conservative for SageMaker
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced for SageMaker compatibility
EPOCHS = 5
LR = 1e-4
TEXT_MODEL_NAME = "albert-base-v2"  # SageMaker compatible model
TABULAR_DIM = 64

# Performance optimizations - SageMaker safe settings
USE_AMP = True  # Will be auto-disabled if not available
FREEZE_TRANSFORMER_LAYERS = True  # Speed up training in SageMaker
TOKENIZER_BATCH_SIZE = 512  # Reduced for SageMaker memory limits
MAX_SEQ_LEN = 80  # Balanced for accuracy and speed
EARLY_STOPPING_PATIENCE = 3
MAX_TRAIN_SECONDS = 3 * 60 * 60  # 3 hour time limit for SageMaker

# Input data path - SageMaker compatible
OPTIMIZED_DATA_PATH = "Training_Data/optimized_data.csv"

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Ensure we're writing to stdout for SageMaker log collection
    handler = logging.StreamHandler(sys.stdout)
    logger = logging.getLogger()
    logger.addHandler(handler)

class StockDataset(Dataset):
    def __init__(self, texts, tabular_data, labels, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tabular_data = tabular_data
        self.labels = labels
        self.max_len = max_len
        
        # Pre-tokenize all texts at initialization for better performance
        # Use batch processing to avoid memory issues on SageMaker
        self.encodings = self._batch_tokenize(texts)

    def _batch_tokenize(self, texts):
        """Batch tokenization optimized for SageMaker"""
        encodings = []
        for i in range(0, len(texts), TOKENIZER_BATCH_SIZE):
            batch_texts = texts[i:i+TOKENIZER_BATCH_SIZE]
            try:
                batch_encodings = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                
                # Split batch result into individual encodings
                for j in range(len(batch_texts)):
                    encodings.append({
                        'input_ids': batch_encodings['input_ids'][j],
                        'attention_mask': batch_encodings['attention_mask'][j]
                    })
            except Exception as e:
                print(f"Error tokenizing batch {i}: {e}")
                # Fallback to individual tokenization if batch fails
                for text in batch_texts:
                    try:
                        encoding = self.tokenizer(
                            text,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_len,
                            return_tensors='pt'
                        )
                        encodings.append({
                            'input_ids': encoding['input_ids'].squeeze(0),
                            'attention_mask': encoding['attention_mask'].squeeze(0)
                        })
                    except:
                        # If individual tokenization fails, create empty tensors
                        encodings.append({
                            'input_ids': torch.zeros(self.max_len, dtype=torch.long),
                            'attention_mask': torch.zeros(self.max_len, dtype=torch.long)
                        })
                
        return encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings[idx]['input_ids'],
            "attention_mask": self.encodings[idx]['attention_mask'],
            "tabular": self.tabular_data[idx],
            "label": self.labels[idx]
        }

def print_data_overview(data):
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
    logging.info(
        f"[Epoch {epoch+1}] "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}"
    )

def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"Validation Confusion Matrix:\n{cm}")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Validation Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    except Exception:
        logging.info("matplotlib/seaborn not available for confusion matrix plot.")

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
    try:
        setup_logging()
        logging.info("Starting training script with improved error handling")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Python: {sys.version}")

        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=EPOCHS)
        parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--input-data', type=str, default=OPTIMIZED_DATA_PATH)
        parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
        # Add the new hyperparameters
        parser.add_argument('--disable_mixed_precision', type=str, default='false')
        parser.add_argument('--disable_self_attention', type=str, default='false')
        parser.add_argument('--fusion_type', type=str, default='concat')
        args = parser.parse_args()

        # Start timing
        start_time = time.time()
        
        # Use SageMaker environment variables if present
        input_data_path = os.environ.get('SM_CHANNEL_TRAIN', args.input_data)
        model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)

        # Check if input_data_path is a directory and append the file name if necessary
        if os.path.isdir(input_data_path):
            input_data_path = os.path.join(input_data_path, 'optimized_data.csv')

        if not os.path.isfile(input_data_path):
            logging.error(f"Input data file not found: {input_data_path}")
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

        logging.info(f"Splitting data: {len(X_text)} total samples")
        logging.info(f"Train set: {len(X_text_train)} | Validation set: {len(X_text_val)}")
        logging.info(f"Tabular feature shape: {tabular_data.shape}")

        train_ds = StockDataset(X_text_train, X_tab_train, y_train, tokenizer)
        val_ds = StockDataset(X_text_val, X_tab_val, y_val, tokenizer)

        # DataLoader optimizations for Kaggle
        num_workers = min(2, os.cpu_count() or 2)
        pin_memory = torch.cuda.is_available()
        persistent_workers = pin_memory

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers
        )

        logging.info(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")
        logging.info(f"Batch size: {args.batch_size} | Num workers: {num_workers}")

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
            if FREEZE_TRANSFORMER_LAYERS and hasattr(model, 'text_encoder'):
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

        # Set up mixed precision training if available
        scaler = GradScaler() if AMP_AVAILABLE and USE_AMP else None
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Track training progress
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

            logging.info(
                f"\n[Epoch {epoch+1}/{args.epochs}] "
                f"Elapsed: {elapsed/60:.1f} min | "
                f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}"
            )

            # Training
            model.train()
            train_loss, train_preds, train_labels = 0.0, [], []
            
            # Reset gradients at the beginning of each epoch
            optimizer.zero_grad()
            
            # Use tqdm for progress bar but with minimal output to reduce overhead
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                ncols=80,
                disable=False
            )
            
            # Gradient accumulation variables
            steps_since_update = 0
            accumulated_loss = 0
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                tab = batch['tabular'].to(device)
                lbl = batch['label'].to(device)
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

        logging.info("Training complete.")
        logging.info("Best validation loss: %.4f", best_val_loss)

        # --- SageMaker: Save model weights ---
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model_weights.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Final model weights saved to {model_path}")
        
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
            logging.info(f"Saved training history plot to {history_path}")
        except Exception as e:
            logging.warning(f"Could not save training history plot: {e}")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
