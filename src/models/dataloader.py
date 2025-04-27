import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List
import numpy as np
import logging
import pytorch_lightning as pl

class MultiModalDataset(Dataset):
    """
    Dataset for multimodal inputs: text, tabular, (optional) vision, audio, time series.
    Each item is a dict with keys matching model inputs.
    """
    def __init__(self, data, tokenizer=None, max_text_len=128):
        """
        Args:
            data (list of dict): Each dict contains modalities and 'label'.
            tokenizer (callable, optional): Tokenizer for text data.
            max_text_len (int): Max length for text tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}

        # Text
        if 'text' in item and self.tokenizer is not None:
            sample['text_inputs'] = self.tokenizer(
                item['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            # Remove batch dimension
            sample['text_inputs'] = {k: v.squeeze(0) for k, v in sample['text_inputs'].items()}
        elif 'text_inputs' in item:
            sample['text_inputs'] = item['text_inputs']

        # Tabular
        if 'tabular' in item:
            sample['tabular_inputs'] = torch.tensor(item['tabular'], dtype=torch.float)

        # Vision
        if 'vision_inputs' in item:
            sample['vision_inputs'] = item['vision_inputs']

        # Audio
        if 'audio_inputs' in item:
            sample['audio_inputs'] = torch.tensor(item['audio_inputs'], dtype=torch.float)

        # Time series
        if 'time_series_inputs' in item:
            sample['time_series_inputs'] = torch.tensor(item['time_series_inputs'], dtype=torch.float)

        # Label
        if 'label' in item:
            sample['label'] = torch.tensor(item['label'], dtype=torch.long)

        return sample

def multimodal_collate_fn(batch):
    """
    Collate function to merge a list of samples to batch.
    """
    batch_out = {}
    keys = batch[0].keys()
    for key in keys:
        if key == 'text_inputs':
            # Merge dict of tensors
            batch_out[key] = {k: torch.stack([b[key][k] for b in batch]) for k in batch[0][key]}
        elif key == 'label':
            batch_out[key] = torch.stack([b[key] for b in batch])
        elif key == 'tabular_inputs':
            stacked = torch.stack([b[key] for b in batch])
            # Ensure shape is [batch, tabular_dim]
            if stacked.ndim > 2:
                # Flatten all but the batch dimension
                stacked = stacked.view(stacked.shape[0], -1)
            batch_out[key] = stacked
        else:
            batch_out[key] = torch.stack([b[key] for b in batch])
    return batch_out

def tft_collate_fn(batch):
    """
    Collate function for TFT: expects dicts with keys like group_id, time_idx, target, and features.
    """
    batch_out = {}
    keys = batch[0].keys()
    for key in keys:
        arr = [b[key] for b in batch]
        if isinstance(arr[0], (int, float, np.integer, np.floating)):
            batch_out[key] = torch.tensor(arr)
        else:
            batch_out[key] = torch.stack(arr)
    return batch_out

def get_dataloader(data, tokenizer=None, batch_size=32, shuffle=True, max_text_len=128, num_workers=0):
    """
    Returns a DataLoader for multimodal data.
    """
    dataset = MultiModalDataset(data, tokenizer=tokenizer, max_text_len=max_text_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=multimodal_collate_fn,
        num_workers=num_workers
    )

class TFTDataModule(pl.LightningDataModule):
    """Data module for Temporal Fusion Transformer"""
    def __init__(
        self, 
        df: pd.DataFrame,
        time_idx: str = "time_idx",
        target: str = "target",
        group_ids: List[str] = ["group_id"],
        max_encoder_length: int = 30,
        max_prediction_length: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        add_relative_time: bool = True,
        add_target_scales: bool = True,
        add_encoder_length: bool = True,
        # Enhanced time series features
        add_lagged_features: bool = False,
        decomposition: bool = False,
        variable_selection: bool = True,
        **kwargs
    ):
        super().__init__()
        self.df = df.copy()
        self.time_idx = time_idx
        self.target = target
        self.group_ids = group_ids
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_relative_time = add_relative_time
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.add_lagged_features = add_lagged_features
        self.decomposition = decomposition
        self.variable_selection = variable_selection
        self.kwargs = kwargs
        
        # Clean data types for TFT
        self._clean_data_types()
        
        # Add lagged features if needed
        if self.add_lagged_features:
            self._add_lagged_features()
        
        # Identify variable types
        self._identify_variable_types()
        
    def _clean_data_types(self):
        """Ensure data types are compatible with TFT"""
        # Convert group IDs to categorical
        for col in self.group_ids:
            self.df[col] = self.df[col].astype(str).astype('category')
            
        # Ensure target and numeric columns are float
        try:
            self.df[self.target] = self.df[self.target].astype(float)
        except Exception as e:
            logging.error(f"Error converting target column to float: {e}")
            
        # Ensure time_idx is integer
        try:
            self.df[self.time_idx] = self.df[self.time_idx].astype(int)
        except Exception as e:
            logging.error(f"Error converting time_idx column to int: {e}")
    
    def _add_lagged_features(self):
        """Add lagged features for key numeric columns"""
        # Find candidate numeric columns
        numeric_cols = self.df.select_dtypes(include=[float, int]).columns.tolist()
        
        # Don't create lags for time indices and target
        exclude_cols = [self.time_idx, self.target] + [f"{self.target}_lag{i}" for i in range(1, 10)]
        candidate_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Create lags for target
        for lag in [1, 2, 7]:  # Common lags for financial data
            self.df[f"{self.target}_lag{lag}"] = self.df.groupby(self.group_ids)[self.target].shift(lag)
        
        # Create lags for a few other important variables (first 5)
        for col in candidate_cols[:5]:  # Limit to avoid feature explosion
            for lag in [1, 2]:  # Smaller lags for other features
                self.df[f"{col}_lag{lag}"] = self.df.groupby(self.group_ids)[col].shift(lag)
        
        # Calculate rolling stats for target
        for window in [3, 7]:
            # Rolling mean
            self.df[f"{self.target}_roll_mean_{window}"] = (
                self.df.groupby(self.group_ids)[self.target]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Rolling std
            self.df[f"{self.target}_roll_std_{window}"] = (
                self.df.groupby(self.group_ids)[self.target]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
    
    def _identify_variable_types(self):
        """Identify variable types for TFT"""
        # Basic variable categorization
        self.time_varying_known_reals = []
        self.time_varying_unknown_reals = [self.target]
        self.static_categoricals = self.group_ids.copy()
        self.static_reals = []
        self.categorical_encoders = {}
        
        # Process each column to determine its type
        for col in self.df.columns:
            # Skip special columns
            if col in [self.time_idx, self.target] + self.group_ids:
                continue
                
            # Handle categorical columns
            if pd.api.types.is_categorical_dtype(self.df[col]):
                # Check if it varies over time within groups
                if self.df.groupby(self.group_ids + [col]).size().reset_index().groupby(self.group_ids).size().max() > 1:
                    # Value changes over time - time-varying categorical
                    self.categorical_encoders[col] = len(self.df[col].cat.categories)
                else:
                    # Constant within groups - static categorical
                    if col not in self.static_categoricals:
                        self.static_categoricals.append(col)
                continue
                
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if it's an unknown future variable (target-related)
                if any(lag_str in col for lag_str in ["_lag", "_roll_", "_diff"]) and self.target in col:
                    # Target-derived features are unknown in the future
                    self.time_varying_unknown_reals.append(col)
                elif self.decomposition and any(comp in col for comp in ["_trend", "_seasonal", "_residual"]):
                    if "_trend" in col or "_seasonal" in col:
                        # Trends and seasonality are known
                        self.time_varying_known_reals.append(col)
                    else:
                        # Residuals are unknown
                        self.time_varying_unknown_reals.append(col)
                else:
                    # Most other features are known
                    self.time_varying_known_reals.append(col)
                continue
                
        # Log the identified variables
        logging.info(f"TFT static categoricals: {self.static_categoricals}")
        logging.info(f"TFT static reals: {self.static_reals}")
        logging.info(f"TFT time varying known reals: {len(self.time_varying_known_reals)} features")
        logging.info(f"TFT time varying unknown reals: {self.time_varying_unknown_reals}")
        
        # If variable selection is enabled, identify multicollinearity and prune features
        if self.variable_selection and len(self.time_varying_known_reals) > 20:
            self._prune_multicollinear_features()
            
    def _prune_multicollinear_features(self, threshold=0.95):
        """Remove highly correlated features to reduce dimensionality"""
        # Only process known reals
        if len(self.time_varying_known_reals) <= 1:
            return
            
        # Calculate correlation matrix
        corr_matrix = self.df[self.time_varying_known_reals].corr().abs()
        
        # Create a mask to identify high correlations 
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            logging.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
            self.time_varying_known_reals = [col for col in self.time_varying_known_reals if col not in to_drop]
        
    def prepare_data(self):
        """Validate that data is suitable for TFT"""
        # Check for at least max_encoder_length consecutive time steps
        grouped = self.df.groupby(self.group_ids)
        valid_groups = []
        
        for group, data in grouped:
            if len(data) >= self.max_encoder_length:
                valid_groups.append(group)
                
        if len(valid_groups) == 0:
            raise ValueError(f"No groups have at least {self.max_encoder_length} time steps")
        
        if len(valid_groups) < len(grouped):
            logging.warning(f"Only {len(valid_groups)}/{len(grouped)} groups have enough time steps")
            
        # Fill missing values - important for time series
        self.df = self.df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
    def setup(self, stage=None):
        """Create datasets"""
        from pytorch_forecasting.data import TimeSeriesDataSet
        
        # Use max_prediction_length to set validation cutoff
        training_cutoff = self.df[self.time_idx].max() - self.max_prediction_length
        
        # Create training dataset
        self.train_dataset = TimeSeriesDataSet(
            self.df[lambda x: x[self.time_idx] <= training_cutoff],
            time_idx=self.time_idx,
            target=self.target,
            group_ids=self.group_ids,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            categorical_encoders=self.categorical_encoders,
            add_relative_time_idx=self.add_relative_time,
            add_target_scales=self.add_target_scales,
            add_encoder_length=self.add_encoder_length,
            **self.kwargs
        )
        
        # Create validation dataset
        self.val_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset, 
            self.df, 
            min_prediction_idx=training_cutoff + 1,
            stop_randomization=True
        )
        
    def train_dataloader(self):
        """Return the training dataloader with dynamic batch sizing based on series length"""
        effective_batch_size = max(1, self.batch_size // (self.max_encoder_length // 10))
        
        return self.train_dataset.to_dataloader(
            batch_size=effective_batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )
        
    def val_dataloader(self):
        """Return the validation dataloader"""
        return self.val_dataset.to_dataloader(
            batch_size=self.batch_size * 2,  # Larger batches for validation
            num_workers=self.num_workers, 
            shuffle=False
        )
