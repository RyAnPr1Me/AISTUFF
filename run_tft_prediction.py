#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pytorch_lightning as pl
from src.models.tft_model import TFTStockPredictor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_model(model_path, metadata_path=None):
    """Load TFT model and metadata"""
    try:
        from pytorch_forecasting.data import TimeSeriesDataSet
    except ImportError:
        logging.error("Missing pytorch-forecasting. Install with: pip install pytorch-forecasting")
        sys.exit(1)
        
    # Load model
    logging.info(f"Loading model from {model_path}")
    model = TFTStockPredictor.load_from_checkpoint(model_path)
    
    # Load metadata if provided
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def prepare_inference_data(df, metadata):
    """Prepare data for TFT inference"""
    # Ensure required columns exist
    required_cols = [
        metadata['time_idx'], 
        metadata['target']
    ] + metadata['group_ids']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in data")
    
    # Convert column types
    df[metadata['time_idx']] = df[metadata['time_idx']].astype(int)
    for group_id in metadata['group_ids']:
        df[group_id] = df[group_id].astype(str).astype('category')
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Ensure all columns are float except group_ids and time_idx
    for col in df.columns:
        if col not in metadata['group_ids'] + [metadata['time_idx']]:
            try:
                df[col] = df[col].astype(float)
            except:
                pass
    
    return df

def plot_predictions(predictions, df, metadata, output_dir):
    """Plot prediction results"""
    os.makedirs(output_dir, exist_ok=True)
    
    target = metadata['target']
    time_idx = metadata['time_idx']
    group_ids = metadata['group_ids']
    
    # Extract prediction components
    y_hat = predictions.output.cpu().numpy()
    index = predictions.index
    
    # Create dataframe with predictions
    pred_df = pd.DataFrame()
    for i in range(len(index)):
        group_vals = {group_ids[j]: index[0][i][j] for j in range(len(group_ids))}
        time_val = index[1][i]
        
        row = {
            **group_vals,
            time_idx: time_val,
            'prediction': y_hat[i][0]
        }
        pred_df = pd.concat([pred_df, pd.DataFrame([row])], ignore_index=True)
    
    # Merge with original data
    result = pd.merge(
        df,
        pred_df,
        on=[*group_ids, time_idx],
        how="right"
    )
    
    # Save merged results
    result.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    # Create plots for each group
    groups = result.groupby(group_ids)
    for i, (group_name, group_data) in enumerate(groups):
        if i >= 10:  # Limit to first 10 groups
            break
            
        plt.figure(figsize=(12, 6))
        
        # Format the group name for display
        if len(group_ids) == 1:
            group_title = f"{group_ids[0]}: {group_name}"
        else:
            group_title = ", ".join([f"{gid}: {gval}" for gid, gval in zip(group_ids, group_name)])
        
        # Plot actual values
        plt.plot(
            group_data[time_idx], 
            group_data[target], 
            'o-', 
            label='Actual', 
            color='blue'
        )
        
        # Plot predictions
        plt.plot(
            group_data[time_idx], 
            group_data['prediction'], 
            'x-', 
            label='Prediction', 
            color='red'
        )
        
        plt.title(f"Predictions for {group_title}")
        plt.xlabel(time_idx)
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        filename = f"group_{'_'.join(str(g) for g in group_name if str(g) != '')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    logging.info(f"Saved predictions and plots to {output_dir}")
    return result

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Make predictions using trained TFT model")
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
        '--output-dir', type=str, default='tft_predictions',
        help='Directory to save predictions and plots'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Maximum number of samples to predict'
    )
    
    args = parser.parse_args()
    
    # Infer metadata path if not provided
    if args.metadata is None:
        model_dir = Path(args.model).parent
        metadata_path = os.path.join(model_dir, "tft_metadata.json")
        if os.path.exists(metadata_path):
            args.metadata = metadata_path
            logging.info(f"Using metadata from {metadata_path}")
    
    # Load model and metadata
    model, metadata = load_model(args.model, args.metadata)
    
    if metadata is None:
        logging.error("No metadata found. Please provide --metadata or ensure tft_metadata.json exists in model directory")
        sys.exit(1)
    
    # Load and prepare data
    logging.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    if args.max_samples:
        df = df.head(args.max_samples)
        logging.info(f"Limited to {args.max_samples} samples")
    
    # Prepare data for inference
    df = prepare_inference_data(df, metadata)
    
    # Create TimeSeriesDataSet for inference
    from pytorch_forecasting.data import TimeSeriesDataSet
    
    # Get max_prediction_length from model
    max_prediction_length = metadata.get('max_prediction_length', 1)
    max_encoder_length = metadata.get('max_encoder_length', 30)
    
    # Create dataset
    dataset = TimeSeriesDataSet(
        df,
        time_idx=metadata['time_idx'],
        target=metadata['target'],
        group_ids=metadata['group_ids'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=metadata.get('time_varying_known_reals', []),
        time_varying_unknown_reals=metadata.get('time_varying_unknown_reals', [metadata['target']]),
        static_categoricals=[],
        static_reals=[],
        predict=True
    )
    
    # Create dataloader
    dataloader = dataset.to_dataloader(batch_size=128, num_workers=0, shuffle=False)
    
    # Make predictions
    logging.info("Making predictions...")
    predictions = model.predict(
        dataloader,
        mode="prediction",
        return_index=True,
        return_y=False
    )
    
    # Plot and save predictions
    results = plot_predictions(predictions, df, metadata, args.output_dir)
    
    logging.info(f"Predictions completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
