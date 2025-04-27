#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Union, Optional

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_model(model_path, metadata_path=None):
    """Load TFT model and metadata"""
    try:
        from pytorch_forecasting import TemporalFusionTransformer
    except ImportError:
        logging.error("Missing pytorch-forecasting. Install with: pip install pytorch-forecasting")
        sys.exit(1)
        
    # Infer metadata path if not provided
    if metadata_path is None:
        model_dir = Path(model_path).parent
        metadata_path = os.path.join(model_dir, "tft_metadata.json")
        if os.path.exists(metadata_path):
            logging.info(f"Using metadata from {metadata_path}")
        else:
            logging.warning(f"No metadata file found at {metadata_path}")
            metadata_path = None
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    
    # Load metadata if available
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def prepare_data_for_evaluation(df, metadata):
    """Prepare data for TFT evaluation"""
    from src.models.dataloader import TFTDataModule
    
    # Ensure required columns exist
    required_cols = [metadata["time_idx"], metadata["target"]] + metadata["group_ids"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean column types
    df = df.copy()
    df[metadata["time_idx"]] = df[metadata["time_idx"]].astype(int)
    for group_id in metadata["group_ids"]:
        df[group_id] = df[group_id].astype(str).astype('category')
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Create data module
    data_module = TFTDataModule(
        df,
        time_idx=metadata["time_idx"],
        target=metadata["target"],
        group_ids=metadata["group_ids"],
        max_encoder_length=metadata.get("max_encoder_length", 30),
        max_prediction_length=metadata.get("max_prediction_length", 1),
        batch_size=128,
        num_workers=0
    )
    
    # Setup data
    data_module.prepare_data()
    data_module.setup()
    
    return data_module

def evaluate_model(model, data_module, output_dir):
    """Evaluate model on the dataset and generate visualizations"""
    import pytorch_lightning as pl
    from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
    
    # Create trainer for evaluation
    trainer = pl.Trainer(
        accelerator='auto',
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )
    
    # Get predictions
    val_dataloader = data_module.val_dataloader()
    predictions = trainer.predict(model, dataloaders=val_dataloader)
    
    # Concatenate batch predictions
    y_true = torch.cat([p.y for p in predictions]).cpu().numpy()
    y_pred = torch.cat([p.output for p in predictions]).cpu().numpy()
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    
    # Calculate MAPE with protection against zeros
    mask = y_true[:, 0] != 0
    mape = np.mean(np.abs((y_true[:, 0][mask] - y_pred[:, 0][mask]) / y_true[:, 0][mask])) * 100
    
    # Calculate SMAPE
    smape = np.mean(200.0 * np.abs(y_pred[:, 0] - y_true[:, 0]) / (np.abs(y_pred[:, 0]) + np.abs(y_true[:, 0]) + 1e-8))
    
    # Create results
    results = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
        "n_samples": len(y_true)
    }
    
    # Get quantile predictions if available
    if hasattr(model, 'loss') and hasattr(model.loss, 'quantiles') and len(model.loss.quantiles) > 1:
        results["quantiles"] = [float(q) for q in model.loss.quantiles]
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    generate_evaluation_plots(y_true, y_pred, output_dir)
    
    # Generate feature importance plots if available
    if hasattr(model, 'interpret_output'):
        try:
            generate_feature_importance_plots(model, val_dataloader, output_dir)
        except Exception as e:
            logging.warning(f"Could not generate feature importance plots: {e}")
    
    return results

def generate_evaluation_plots(y_true, y_pred, output_dir):
    """Generate evaluation plots"""
    # Scatter plot of predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
    
    # Residual plot
    residuals = y_pred[:, 0] - y_true[:, 0]
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred[:, 0], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'))
    
    # Residual histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title('Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residual_histogram.png'))
    
    # QQ plot to check normality of residuals
    from scipy import stats
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, plot=plt)
    plt.title('QQ Plot of Residuals')
    plt.savefig(os.path.join(output_dir, 'qq_plot.png'))

def generate_feature_importance_plots(model, dataloader, output_dir):
    """Generate feature importance plots"""
    interpretation = model.interpret_output(
        dataloader,
        reduction="sum",
        attention_prediction_horizon=0  # attention for first prediction horizon
    )
    
    # Plot variable importance
    plt.figure(figsize=(10, 8))
    order = np.argsort(interpretation["variable_importance"].mean(axis=0))
    variables = interpretation["variable_importance"].mean(axis=0)[order][-30:]  # Top 30
    variables_names = [model.hparams.x_reals[idx] for idx in order][-30:]  # Top 30
    
    plt.barh(variables_names, variables)
    plt.title("Variable Importance")
    plt.savefig(os.path.join(output_dir, 'variable_importance.png'))
    
    # Save interpretation results
    if hasattr(interpretation, "to_dataframe"):
        interpretation_df = interpretation.to_dataframe()
        interpretation_df.to_csv(os.path.join(output_dir, 'interpretation.csv'))

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Evaluate trained TFT model on new data")
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to data CSV file'
    )
    parser.add_argument(
        '--metadata', type=str, default=None,
        help='Path to TFT metadata JSON file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='tft_evaluation',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--plot-predictions', action='store_true',
        help='Generate individual prediction plots for each time series'
    )
    
    args = parser.parse_args()
    
    # Load model and metadata
    model, metadata = load_model(args.model, args.metadata)
    
    # Load data
    logging.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logging.info(f"Loaded data with shape {df.shape}")
    
    # Prepare data
    data_module = prepare_data_for_evaluation(df, metadata)
    
    # Evaluate model
    results = evaluate_model(model, data_module, args.output_dir)
    
    # Print evaluation results
    logging.info("Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            logging.info(f"  {metric}: {value:.4f}")
        else:
            logging.info(f"  {metric}: {value}")
    
    # Plot individual predictions if requested
    if args.plot_predictions:
        logging.info("Generating individual prediction plots...")
        from src.models.ts_utils import plot_forecasts_for_groups
        plot_forecasts_for_groups(
            model=model,
            data_module=data_module,
            output_dir=os.path.join(args.output_dir, 'forecasts'),
            max_samples=10  # Limit to 10 samples to avoid too many plots
        )
    
    logging.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
