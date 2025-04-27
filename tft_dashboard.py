#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st
import json
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main(args=None):
    """
    Run Streamlit dashboard for TFT forecasting visualization
    
    Args:
        args: Command line arguments
    """
    if args is None:
        parser = argparse.ArgumentParser(description="TFT Forecasting Dashboard")
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
        args = parser.parse_args()

    # Streamlit app
    st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")
    st.title("Time Series Forecasting Dashboard")
    
    # Load model and data
    model, metadata, df = load_model_and_data(args.model, args.data, args.metadata)
    
    # Sidebar for inputs
    with st.sidebar:
        st.subheader("Configuration")
        
        # Get unique groups
        group_ids = metadata["group_ids"]
        
        # If we have multiple group IDs, let user select one at a time
        selected_groups = {}
        for group_id in group_ids:
            unique_values = sorted(df[group_id].unique().tolist())
            selected_groups[group_id] = st.selectbox(f"Select {group_id}", unique_values)
        
        # Get prediction parameters
        forecast_horizon = st.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=30,
            value=metadata.get("max_prediction_length", 1)
        )
        
        confidence_interval = st.slider(
            "Confidence Interval (%)",
            min_value=50,
            max_value=99,
            value=90,
            step=5
        )
        
        st.subheader("Analysis Options")
        show_components = st.checkbox("Show Time Series Components", value=False)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)
        
        # Button to generate forecast
        generate_button = st.button("Generate Forecast")
    
    # Main content area
    if generate_button:
        st.subheader("Time Series Forecast")
        
        # Filter data for selected group
        filtered_df = df.copy()
        for group_id, value in selected_groups.items():
            filtered_df = filtered_df[filtered_df[group_id] == value]
        
        if len(filtered_df) == 0:
            st.error("No data available for the selected criteria.")
            return
        
        # Generate forecast
        with st.spinner("Generating forecast..."):
            forecast_df, metrics = generate_forecast(
                model, metadata, filtered_df, forecast_horizon, confidence_interval
            )
        
        # Display metrics if requested
        if show_metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{metrics['rmse']:.4f}")
            col2.metric("MAE", f"{metrics['mae']:.4f}")
            col3.metric("MAPE", f"{metrics['mape']:.2f}%")
            col4.metric("SMAPE", f"{metrics['smape']:.2f}%")
        
        # Plot forecast
        fig = plot_forecast(forecast_df, metadata["target"], confidence_interval)
        st.pyplot(fig)
        
        # Plot time series components if requested
        if show_components:
            st.subheader("Time Series Components")
            components_fig = plot_time_series_components(filtered_df, metadata["target"])
            st.pyplot(components_fig)
        
        # Allow downloading the forecast as CSV
        st.download_button(
            label="Download Forecast CSV",
            data=forecast_df.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv",
        )
    else:
        st.info("Select parameters and click 'Generate Forecast' to create a forecast.")
        
        # Display model information
        st.subheader("Model Information")
        if "model_info" in metadata:
            model_info = metadata["model_info"]
            for key, value in model_info.items():
                st.text(f"{key}: {value}")
        else:
            st.text("No detailed model information available.")

def load_model_and_data(model_path, data_path, metadata_path=None):
    """Load TFT model, metadata, and data"""
    try:
        from pytorch_forecasting import TemporalFusionTransformer
    except ImportError:
        st.error("Missing pytorch-forecasting. Install with: pip install pytorch-forecasting")
        st.stop()
        
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
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Load metadata if available
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        # Extract minimal metadata from model
        metadata = {
            "time_idx": "time_idx",
            "target": "target",
            "group_ids": ["group_id"],
            "max_encoder_length": model.hparams.max_encoder_length,
            "max_prediction_length": model.hparams.max_prediction_length,
        }
    
    # Load data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    return model, metadata, df

def generate_forecast(model, metadata, df, forecast_horizon, confidence_interval):
    """Generate forecast for the selected data"""
    from src.models.dataloader import TFTDataModule
    
    # Create data module
    data_module = TFTDataModule(
        df,
        time_idx=metadata["time_idx"],
        target=metadata["target"],
        group_ids=metadata["group_ids"],
        max_encoder_length=metadata.get("max_encoder_length", 30),
        max_prediction_length=forecast_horizon,
        batch_size=128,
        num_workers=0
    )
    
    # Prepare data
    data_module.prepare_data()
    data_module.setup()
    
    # Get prediction dataset
    prediction_dataset = data_module.get_prediction_data()
    prediction_dataloader = prediction_dataset.to_dataloader(batch_size=128, num_workers=0, shuffle=False)
    
    # Make predictions
    predictions = model.predict(
        prediction_dataloader,
        mode="prediction",
        return_x=True,
        return_y=True,
        return_index=True,
    )
    
    # Extract predictions, actuals, and index
    y_hat = predictions.output.cpu().numpy()
    y_true = predictions.y.cpu().numpy()
    index = predictions.index
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Only evaluate where we have actual values
    valid_mask = ~np.isnan(y_true[:, 0])
    
    if valid_mask.sum() > 0:
        mae = mean_absolute_error(y_true[valid_mask, 0], y_hat[valid_mask, 0])
        rmse = np.sqrt(mean_squared_error(y_true[valid_mask, 0], y_hat[valid_mask, 0]))
        
        # Calculate MAPE with protection against zeros
        mape_mask = (valid_mask) & (y_true[:, 0] != 0)
        mape = np.mean(np.abs((y_true[mape_mask, 0] - y_hat[mape_mask, 0]) / y_true[mape_mask, 0])) * 100
        
        # Calculate SMAPE
        smape = np.mean(200.0 * np.abs(y_hat[valid_mask, 0] - y_true[valid_mask, 0]) / (np.abs(y_hat[valid_mask, 0]) + np.abs(y_true[valid_mask, 0]) + 1e-8))
    else:
        mae, rmse, mape, smape = 0, 0, 0, 0
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape
    }
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame()
    
    # Process each prediction
    time_idx_col = metadata["time_idx"]
    target_col = metadata["target"]
    
    # Convert index to more usable form
    df_index = []
    for group_idx, time_idx in zip(index[0], index[1]):
        # Handle both single and multiple group IDs
        if isinstance(group_idx, (list, tuple)):
            row = {metadata["group_ids"][i]: group_idx[i] for i in range(len(metadata["group_ids"]))}
        else:
            row = {metadata["group_ids"][0]: group_idx}
        
        row[time_idx_col] = time_idx
        df_index.append(row)
    
    index_df = pd.DataFrame(df_index)
    
    # Add predictions and actuals to DataFrame
    forecast_df = index_df.copy()
    forecast_df[f"{target_col}_actual"] = y_true[:, 0]
    forecast_df[f"{target_col}_forecast"] = y_hat[:, 0]
    
    # Add prediction intervals if quantiles are available
    if hasattr(model.loss, 'quantiles') and len(model.loss.quantiles) > 1:
        quantiles = model.loss.quantiles
        for i, q in enumerate(quantiles):
            forecast_df[f"{target_col}_q{int(q*100)}"] = y_hat[:, i]
    else:
        # Calculate prediction intervals based on empirical error distribution
        error = y_true[valid_mask, 0] - y_hat[valid_mask, 0]
        lower_percentile = (100 - confidence_interval) / 2
        upper_percentile = 100 - lower_percentile
        
        lower_bound = np.percentile(error, lower_percentile)
        upper_bound = np.percentile(error, upper_percentile)
        
        forecast_df[f"{target_col}_lower"] = y_hat[:, 0] + lower_bound
        forecast_df[f"{target_col}_upper"] = y_hat[:, 0] + upper_bound
    
    return forecast_df, metrics

def plot_forecast(forecast_df, target_col, confidence_interval):
    """Create forecast plot with prediction intervals"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual values
    ax.scatter(
        forecast_df["time_idx"], 
        forecast_df[f"{target_col}_actual"], 
        color='blue', 
        alpha=0.6, 
        label='Actual'
    )
    
    # Plot forecast
    ax.plot(
        forecast_df["time_idx"], 
        forecast_df[f"{target_col}_forecast"], 
        'r-', 
        label='Forecast', 
        linewidth=2
    )
    
    # Plot prediction intervals
    if f"{target_col}_q10" in forecast_df.columns and f"{target_col}_q90" in forecast_df.columns:
        # Use quantiles directly from model
        ax.fill_between(
            forecast_df["time_idx"],
            forecast_df[f"{target_col}_q10"],
            forecast_df[f"{target_col}_q90"],
            color='red',
            alpha=0.2,
            label='80% Prediction Interval'
        )
    elif f"{target_col}_lower" in forecast_df.columns:
        # Use empirical prediction intervals
        ax.fill_between(
            forecast_df["time_idx"],
            forecast_df[f"{target_col}_lower"],
            forecast_df[f"{target_col}_upper"],
            color='red',
            alpha=0.2,
            label=f'{confidence_interval}% Prediction Interval'
        )
    
    ax.set_title(f'Time Series Forecast for {target_col}')
    ax.set_xlabel('Time')
    ax.set_ylabel(target_col)
    ax.grid(True)
    ax.legend()
    
    return fig

def plot_time_series_components(df, target_col):
    """
    Plot time series components (trend, seasonality, residuals)
    using STL decomposition
    """
    from statsmodels.tsa.seasonal import STL
    import matplotlib.pyplot as plt
    
    # Ensure time series is sorted
    df = df.sort_values('time_idx')
    
    # Get the target column
    target_values = df[target_col].values
    
    # Check if we have enough data points
    if len(target_values) < 10:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Not enough data points for decomposition", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Determine seasonality period (simple heuristic)
    acf = pd.Series(target_values).autocorr(lag=1)
    period = 7 if acf > 0.7 else 5  # Weekly vs business week
    period = min(period, len(target_values) // 2)  # Ensure period isn't too large
    
    try:
        # STL Decomposition
        stl = STL(target_values, period=period, robust=True)
        result = stl.fit()
        
        # Create plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Original time series
        axes[0].plot(target_values, 'k-', label='Original')
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'Time Series Decomposition for {target_col}')
        axes[0].legend()
        axes[0].grid(True)
        
        # Trend component
        axes[1].plot(result.trend, 'b-', label='Trend')
        axes[1].set_ylabel('Trend')
        axes[1].legend()
        axes[1].grid(True)
        
        # Seasonal component
        axes[2].plot(result.seasonal, 'g-', label='Seasonal')
        axes[2].set_ylabel('Seasonal')
        axes[2].legend()
        axes[2].grid(True)
        
        # Residual component
        axes[3].plot(result.resid, 'r-', label='Residual')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        # Fallback if decomposition fails
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"STL decomposition failed: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        return fig

if __name__ == "__main__":
    setup_logging()
    main()
