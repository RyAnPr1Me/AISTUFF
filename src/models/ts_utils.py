import pandas as pd
import numpy as np
import logging
import torch
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import pytorch_lightning as pl
from scipy import stats
from statsmodels.tsa.seasonal import STL
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error

def add_time_features(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    seasonal_period: int = 5,
    fourier_terms: int = 2
) -> pd.DataFrame:
    """
    Add advanced time features to improve time series forecasting
    
    Args:
        df: DataFrame with time series data
        date_col: Date column name
        seasonal_period: Seasonal period for Fourier features
        fourier_terms: Number of Fourier terms to include
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Make sure date column is datetime
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            logging.warning(f"Could not convert {date_col} to datetime")
            return df
    else:
        return df
    
    # Basic time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_month'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Business day indicators
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_col].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_col].dt.is_year_end.astype(int)
    
    # Create continuous time index for Fourier features
    df['time_idx_continuous'] = (df[date_col] - df[date_col].min()).dt.total_seconds() / 86400.0
    
    # Add Fourier features for seasonality
    for period in [seasonal_period]:  # Can add multiple periods if needed
        for term in range(1, fourier_terms + 1):
            # Sin component
            df[f'sin_{period}_{term}'] = np.sin(
                2 * np.pi * term * df['time_idx_continuous'] / period
            )
            
            # Cos component
            df[f'cos_{period}_{term}'] = np.cos(
                2 * np.pi * term * df['time_idx_continuous'] / period
            )
    
    return df

def apply_differencing(
    df: pd.DataFrame,
    target_col: str = 'target',
    group_col: str = 'group_id',
    time_col: str = 'time_idx',
    order: int = 1
) -> pd.DataFrame:
    """
    Apply differencing to make time series stationary
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        group_col: Group column name for panel data
        time_col: Time index column
        order: Differencing order (1 = first difference, etc.)
        
    Returns:
        DataFrame with differenced target and original target
    """
    df = df.copy().sort_values([group_col, time_col])
    
    # Store original target
    original_target = f"{target_col}_original"
    df[original_target] = df[target_col].copy()
    
    # Apply differencing by group
    for o in range(1, order + 1):
        df[f"{target_col}_diff{o}"] = df.groupby(group_col)[target_col].diff(o)
    
    # Replace target with differenced value
    if order > 0:
        df[target_col] = df[f"{target_col}_diff{order}"]
    
    return df

def decompose_time_series(
    df: pd.DataFrame,
    target_col: str = 'target',
    group_col: str = 'group_id',
    time_col: str = 'time_idx',
    period: Optional[int] = None
) -> pd.DataFrame:
    """
    Decompose time series into trend, seasonal, and residual components
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        group_col: Group column name
        time_col: Time index column
        period: Seasonality period, if None will be estimated
        
    Returns:
        DataFrame with trend and seasonal components as features
    """
    df = df.copy().sort_values([group_col, time_col])
    
    # Store original target
    df[f"{target_col}_original"] = df[target_col].copy()
    
    # Process each group separately
    for group, group_data in df.groupby(group_col):
        if len(group_data) < 10:  # Skip if too short
            continue
            
        # Estimate period if not provided
        if period is None:
            # Simple heuristic - can be improved
            if len(group_data) >= 10:
                acf = pd.Series(group_data[target_col]).autocorr(lag=1)
                period = 7 if acf > 0.7 else 5  # Weekly vs business week
            else:
                period = 5  # Default to business week
        
        try:
            # Apply STL decomposition
            decomposition = STL(
                group_data[target_col], 
                period=period,
                robust=True
            ).fit()
            
            # Extract components
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Add components to DataFrame
            idx = df.index[df[group_col] == group]
            df.loc[idx, f"{target_col}_trend"] = trend
            df.loc[idx, f"{target_col}_seasonal"] = seasonal
            df.loc[idx, f"{target_col}_residual"] = residual
            
            # Replace target with residual for stationary modeling
            df.loc[idx, target_col] = residual
            
        except Exception as e:
            logging.warning(f"Decomposition failed for group {group}: {str(e)}")
    
    return df

def create_backtesting_datasets(
    df: pd.DataFrame,
    time_idx: str,
    target: str,
    group_ids: List[str],
    max_encoder_length: int,
    max_prediction_length: int,
    num_windows: int = 3
) -> List[Dict]:
    """
    Create multiple temporal validation windows for backtesting
    
    Args:
        df: DataFrame with time series data
        time_idx: Time index column
        target: Target column
        group_ids: Group ID columns
        max_encoder_length: Max encoder length
        max_prediction_length: Max prediction length
        num_windows: Number of validation windows
        
    Returns:
        List of dictionaries with train/val dataset info
    """
    df = df.copy().sort_values(group_ids + [time_idx])
    
    # Get unique time steps
    time_steps = df[time_idx].unique()
    time_steps.sort()
    
    # Determine window size based on number of windows
    total_steps = len(time_steps)
    available_steps = total_steps - max_encoder_length
    
    if available_steps <= num_windows * max_prediction_length:
        # Not enough data for requested windows, reduce number
        num_windows = max(1, available_steps // max_prediction_length)
        logging.warning(f"Reduced backtest windows to {num_windows} due to limited data")
    
    # Create validation windows
    windows = []
    step_size = available_steps // num_windows
    
    for i in range(num_windows):
        # Calculate cutoff points
        val_start_idx = total_steps - available_steps + i * step_size
        val_end_idx = min(val_start_idx + step_size, total_steps)
        
        val_start = time_steps[val_start_idx]
        val_end = time_steps[val_end_idx - 1]
        
        windows.append({
            "window": i + 1,
            "train_end": val_start - 1,
            "val_start": val_start,
            "val_end": val_end
        })
    
    return windows

def run_backtesting(
    model,
    df: pd.DataFrame,
    time_idx: str,
    target: str,
    group_ids: List[str],
    max_encoder_length: int,
    max_prediction_length: int,
    num_windows: int = 3,
    trainer: Optional[pl.Trainer] = None
) -> Dict[str, Any]:
    """
    Run backtesting on multiple temporal validation windows
    
    Args:
        model: TFT model
        df: DataFrame with time series data
        time_idx: Time index column
        target: Target column
        group_ids: Group ID columns
        max_encoder_length: Max encoder length
        max_prediction_length: Max prediction length
        num_windows: Number of validation windows
        trainer: Optional trainer instance
        
    Returns:
        Dictionary with backtesting results
    """
    from src.models.dataloader import TFTDataModule
    
    # Create validation windows
    windows = create_backtesting_datasets(
        df, time_idx, target, group_ids, 
        max_encoder_length, max_prediction_length, num_windows
    )
    
    results = {
        "windows": [],
        "summary": {}
    }
    
    metrics_by_window = []
    
    # Run each validation window
    for window_info in windows:
        window_num = window_info["window"]
        train_end = window_info["train_end"]
        val_start = window_info["val_start"]
        val_end = window_info["val_end"]
        
        logging.info(f"Running backtest window {window_num}: train_end={train_end}, val={val_start}-{val_end}")
        
        # Create train/val split
        train_df = df[df[time_idx] <= train_end].copy()
        val_df = df[(df[time_idx] >= val_start) & (df[time_idx] <= val_end)].copy()
        
        # Create data module for this window
        data_module = TFTDataModule(
            train_df,
            time_idx=time_idx,
            target=target,
            group_ids=group_ids,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            batch_size=64,
            num_workers=0
        )
        
        # Setup datasets
        data_module.prepare_data()
        data_module.setup()
        
        # Create model for this window
        window_model = model.create_model(data_module.train_dataset)
        
        # If trainer provided, fit model on this window
        if trainer is not None:
            trainer.fit(
                model=model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader()
            )
        
        # Create validation dataset for prediction
        val_dataset = data_module.val_dataset
        val_dataloader = data_module.val_dataloader()
        
        # Make predictions on validation set
        predictions = model.predict(
            val_dataloader,
            return_x=True,
            return_y=True,
            mode="prediction"
        )
        
        # Calculate metrics
        y_true = predictions.y.cpu().numpy()
        y_pred = predictions.output.cpu().numpy()
        
        mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        
        # Calculate MAPE with protection against zeros
        mask = y_true[:, 0] != 0
        mape = np.mean(np.abs((y_true[:, 0][mask] - y_pred[:, 0][mask]) / y_true[:, 0][mask])) * 100
        
        # Calculate SMAPE
        smape = np.mean(200.0 * np.abs(y_pred[:, 0] - y_true[:, 0]) / (np.abs(y_pred[:, 0]) + np.abs(y_true[:, 0]) + 1e-8))
        
        window_metrics = {
            "window": window_num,
            "train_end": int(train_end),
            "val_start": int(val_start),
            "val_end": int(val_end),
            "samples": len(y_true),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "smape": float(smape)
        }
        
        results["windows"].append(window_metrics)
        metrics_by_window.append(window_metrics)
    
    # Calculate summary metrics
    summary = {
        "num_windows": num_windows,
        "mae": np.mean([w["mae"] for w in metrics_by_window]),
        "rmse": np.mean([w["rmse"] for w in metrics_by_window]),
        "mape": np.mean([w["mape"] for w in metrics_by_window]),
        "smape": np.mean([w["smape"] for w in metrics_by_window]),
        "mae_std": np.std([w["mae"] for w in metrics_by_window]),
        "rmse_std": np.std([w["rmse"] for w in metrics_by_window]),
    }
    
    results["summary"] = summary
    
    return results

def optimize_hyperparameters(
    data_module,
    num_trials: int = 20,
    max_epochs: int = 20,
    optimization_metric: str = 'SMAPE'
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for TFT model
    
    Args:
        data_module: TFT data module
        num_trials: Number of optimization trials
        max_epochs: Max epochs per trial
        optimization_metric: Metric to optimize
        
    Returns:
        Dictionary with best hyperparameters
    """
    import optuna
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import SMAPE, RMSE, MAE, MAPE
    
    # Prepare datasets
    data_module.setup()
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    
    # Select metric to optimize
    if optimization_metric == 'RMSE':
        metric = RMSE()
    elif optimization_metric == 'MAE':
        metric = MAE()
    elif optimization_metric == 'MAPE':
        metric = MAPE()
    else:
        metric = SMAPE()  # Default
    
    # Define the objective function
    def objective(trial):
        # Sample hyperparameters
        hidden_size = trial.suggest_int("hidden_size", 16, 128)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        attention_head_size = trial.suggest_int("attention_head_size", 1, 4)
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 64)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        
        # Create model with sampled hyperparameters
        model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            attention_head_size=attention_head_size,
            hidden_continuous_size=hidden_continuous_size,
            dropout=dropout,
            learning_rate=learning_rate,
            loss=metric,
            log_interval=10,
            reduce_on_plateau_patience=3,
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=None,
            accelerator='auto',
        )
        
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_dataset.to_dataloader(batch_size=64, shuffle=True),
            val_dataloaders=val_dataset.to_dataloader(batch_size=64, shuffle=False),
        )
        
        # Validate model
        validation_dataloader = val_dataset.to_dataloader(batch_size=64, shuffle=False)
        validation_outputs = trainer.predict(model, dataloaders=validation_dataloader)
        
        predictions = torch.cat([output.output for output in validation_outputs])
        targets = torch.cat([output.y for output in validation_outputs])
        
        # Calculate loss
        val_loss = metric(predictions, targets).item()
        
        return val_loss
    
    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    
    # Get best parameters
    best_params = study.best_params
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best {optimization_metric}: {study.best_value}")
    
    return best_params

def plot_forecasts_for_groups(model, data_module, output_dir, max_samples=10):
    """
    Generate forecast plots for individual groups
    
    Args:
        model: TFT model
        data_module: TFT data module
        output_dir: Output directory for plots
        max_samples: Maximum number of groups to plot
        
    Returns:
        None (saves plots to disk)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get validation dataloader
    val_dataloader = data_module.val_dataloader()
    
    # Make predictions
    predictions = model.predict(
        val_dataloader,
        return_index=True,
        return_x=True,
        return_y=True,
        mode="prediction"
    )
    
    # Get group IDs
    group_ids = data_module.group_ids
    
    # Find unique groups
    all_indices = predictions.index
    unique_groups = {}
    
    for i, index in enumerate(all_indices[0]):
        # Get group ID values as a tuple
        if len(group_ids) == 1:
            group = index.item() if isinstance(index, torch.Tensor) else index
        else:
            group = tuple(idx.item() if isinstance(idx, torch.Tensor) else idx for idx in index)
            
        # Store first occurrence of each group
        if group not in unique_groups and len(unique_groups) < max_samples:
            unique_groups[group] = i
    
    # Plot each unique group
    for group, idx in unique_groups.items():
        plt.figure(figsize=(12, 6))
        
        # Format title
        if isinstance(group, tuple):
            group_title = ", ".join(f"{gid}={g}" for gid, g in zip(group_ids, group))
        else:
            group_title = f"{group_ids[0]}={group}"
        
        # Get predictions and actual values for this group
        y_true = predictions.y[idx].cpu().numpy()
        y_pred = predictions.output[idx].cpu().numpy()
        
        # Create time axis (this is a single prediction, so just use indices)
        time_points = np.arange(len(y_true))
        
        # Plot actual values
        plt.plot(time_points, y_true, 'o-', label='Actual', color='blue')
        
        # Plot predictions
        plt.plot(time_points, y_pred, 'x--', label='Prediction', color='red')
        
        # If we have quantiles, plot prediction intervals
        if hasattr(model, 'loss') and hasattr(model.loss, 'quantiles') and len(model.loss.quantiles) > 1:
            # Find the quantile indices (typically 0.1, 0.5, 0.9)
            quantiles = model.loss.quantiles
            lower_idx = 0  # Assuming the first quantile is the lower bound
            upper_idx = -1  # Assuming the last quantile is the upper bound
            
            # Plot prediction intervals
            plt.fill_between(
                time_points, 
                y_pred[:, lower_idx], 
                y_pred[:, upper_idx], 
                color='red', 
                alpha=0.2, 
                label=f'{quantiles[lower_idx]}-{quantiles[upper_idx]} PI'
            )
        
        plt.title(f'Forecast for {group_title}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        if isinstance(group, tuple):
            filename = f"forecast_{'_'.join(str(g) for g in group)}.png"
        else:
            filename = f"forecast_{group}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    logging.info(f"Saved {len(unique_groups)} forecast plots to {output_dir}")
