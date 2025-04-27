import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Union

try:
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss, SMAPE
except ImportError:
    logging.error("pytorch_forecasting not found. Install with: pip install pytorch-forecasting")
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    QuantileLoss = None

class TFTStockPredictor(pl.LightningModule):
    """
    Adapter for PyTorch Forecasting's TemporalFusionTransformer
    """
    def __init__(
        self,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        attention_head_size: int = 4,
        max_prediction_length: int = 1,
        max_encoder_length: int = 30,
        learning_rate: float = 0.001,
        reduce_on_plateau_patience: int = 3,
        log_interval: int = 10,
        # Enhanced time series parameters
        use_quantiles: bool = False,
        seasonal_period: int = 5,
        optimization_metric: str = 'SMAPE',
        hidden_continuous_size: Optional[int] = None,
        gradual_reintroduction: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.attention_head_size = attention_head_size
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.learning_rate = learning_rate
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        self.log_interval = log_interval
        self.use_quantiles = use_quantiles
        self.seasonal_period = seasonal_period
        self.optimization_metric = optimization_metric
        self.hidden_continuous_size = hidden_continuous_size or hidden_size // 2
        self.gradual_reintroduction = gradual_reintroduction
        self.model = None
        
    def create_model(self, training_dataset):
        """
        Initialize the TFT model with the training dataset
        """
        if TemporalFusionTransformer is None:
            raise ImportError("pytorch_forecasting not installed")
        
        # Select loss function based on optimization metric and quantile settings
        if self.use_quantiles:
            # Quantile loss for prediction intervals (10%, 50%, 90%)
            loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        else:
            # Single point forecasts with specified metric
            if self.optimization_metric == 'MAE':
                from pytorch_forecasting.metrics import MAE
                loss = MAE()
            elif self.optimization_metric == 'MAPE':
                from pytorch_forecasting.metrics import MAPE
                loss = MAPE()
            elif self.optimization_metric == 'RMSE':
                from pytorch_forecasting.metrics import RMSE
                loss = RMSE()
            elif self.optimization_metric == 'MASE':
                from pytorch_forecasting.metrics import MASE
                loss = MASE(seasonal_period=self.seasonal_period)
            else:
                from pytorch_forecasting.metrics import SMAPE
                loss = SMAPE()
        
        # Create the model with enhanced parameters
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            attention_head_size=self.attention_head_size,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=loss,
            log_interval=self.log_interval,
            log_val_interval=self.log_interval,
            # Enhanced TFT parameters
            learning_rate=self.learning_rate,
            reduce_on_plateau_patience=self.reduce_on_plateau_patience,
            logging_metrics=torch.nn.ModuleDict({
                "SMAPE": SMAPE(),
                "MAE": MAE(),
                "RMSE": RMSE()
            }),
            embedding_sizes={'group_id': (10, 5)} if 'group_id' in training_dataset.categorical_encoders else {},
        )
        
        return self.model
        
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        """Training step with enhanced logging"""
        output = self.model.training_step(batch, batch_idx)
        self.log("train_loss", output["loss"], prog_bar=True)
        
        # Gradual learning rate warm-up
        if self.gradual_reintroduction and self.trainer.current_epoch < 3:
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.learning_rate * (self.trainer.current_epoch + 1) / 3
                
        return output
        
    def validation_step(self, batch, batch_idx):
        """Validation step with enhanced metrics"""
        output = self.model.validation_step(batch, batch_idx)
        self.log("val_loss", output["loss"], prog_bar=True)
        
        # Log additional metrics if available
        for metric_name in output:
            if metric_name.startswith("val_"):
                self.log(metric_name, output[metric_name])
                
        return output
        
    def configure_optimizers(self):
        """Configure optimizers with customized scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-6  # Regularization for better generalization
        )
        
        # Reduce LR on plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.reduce_on_plateau_patience,
            verbose=True,
            threshold=1e-4,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
    
    def predict(self, dataloader, return_index=True, return_x=False, return_y=False, mode="prediction"):
        """Make predictions with the model"""
        if self.model is None:
            raise ValueError("Model not initialized. Call create_model first.")
            
        return self.model.predict(
            dataloader, 
            return_index=return_index,
            return_x=return_x,
            return_y=return_y,
            mode=mode,
        )
