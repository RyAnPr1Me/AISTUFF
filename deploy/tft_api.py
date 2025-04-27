#!/usr/bin/env python3
import os
import argparse
import logging
import pandas as pd
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import uvicorn
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    forecast_horizon: Optional[int] = 1
    confidence_interval: Optional[int] = 90

class ForecastResponse(BaseModel):
    forecasts: List[Dict[str, Any]]
    metrics: Optional[Dict[str, float]]

app = FastAPI(title="Time Series Forecasting API", 
              description="API for making predictions with Temporal Fusion Transformer models")

@app.on_event("startup")
async def startup_event():
    """Load model and metadata on startup"""
    global model, metadata
    
    # Load model and metadata
    from pytorch_forecasting import TemporalFusionTransformer
    
    if not os.path.exists(args.model):
        raise ValueError(f"Model file not found: {args.model}")
        
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(args.model)
        logging.info(f"Loaded model from {args.model}")
        
        # Try to load metadata from same directory
        model_dir = Path(args.model).parent
        metadata_path = os.path.join(model_dir, "tft_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Loaded metadata from {metadata_path}")
        else:
            # Create basic metadata from model
            metadata = {
                "time_idx": "time_idx",
                "target": "target",
                "group_ids": ["group_id"],
                "max_encoder_length": model.hparams.max_encoder_length,
                "max_prediction_length": model.hparams.max_prediction_length,
            }
            logging.info("Created default metadata from model")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "TFT Forecasting API", 
        "model_info": f"Model loaded: {args.model}",
        "docs_url": "/docs"
    }

@app.get("/metadata")
async def get_metadata():
    """Get model metadata"""
    return metadata

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Generate forecasts from input data"""
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Prepare data for prediction
        df = prepare_data_for_prediction(df, metadata)
        
        # Create dataset and dataloader
        from pytorch_forecasting.data import TimeSeriesDataSet
        
        dataset = TimeSeriesDataSet(
            df,
            time_idx=metadata["time_idx"],
            target=metadata["target"],
            group_ids=metadata["group_ids"],
            max_encoder_length=metadata.get("max_encoder_length", 30),
            max_prediction_length=request.forecast_horizon,
            time_varying_known_reals=metadata.get("time_varying_known_reals", []),
            time_varying_unknown_reals=metadata.get("time_varying_unknown_reals", [metadata["target"]]),
            static_categoricals=[],
            static_reals=[],
            predict=True
        )
        
        dataloader = dataset.to_dataloader(batch_size=128, num_workers=0, shuffle=False)
        
        # Make predictions
        with torch.no_grad():
            predictions = model.predict(
                dataloader,
                mode="prediction",
                return_x=True,
                return_y=True,
                return_index=True
            )
        
        # Process predictions
        forecast_results = process_predictions(predictions, df, metadata, request.confidence_interval)
        
        # Calculate metrics if actuals are available
        metrics = calculate_metrics(predictions) if hasattr(predictions, 'y') else None
        
        return {
            "forecasts": forecast_results,
            "metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/forecast/csv")
async def forecast_from_csv(
    file: UploadFile = File(...), 
    forecast_horizon: int = 1,
    confidence_interval: int = 90
):
    """Generate forecasts from CSV file"""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Prepare request
        data = df.to_dict(orient="records")
        request = ForecastRequest(
            data=data,
            forecast_horizon=forecast_horizon,
            confidence_interval=confidence_interval
        )
        
        # Use the existing forecast endpoint
        return await forecast(request)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

def prepare_data_for_prediction(df, metadata):
    """Prepare data for TFT prediction"""
    # Ensure required columns exist
    required_cols = [metadata["time_idx"], metadata["target"]] + metadata["group_ids"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data types
    df = df.copy()
    df[metadata["time_idx"]] = df[metadata["time_idx"]].astype(int)
    for group_id in metadata["group_ids"]:
        df[group_id] = df[group_id].astype(str).astype('category')
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

def process_predictions(predictions, df, metadata, confidence_interval):
    """Process TFT predictions into response format"""
    # Extract components
    y_hat = predictions.output.cpu().numpy()
    index = predictions.index
    
    # Create result list
    results = []
    
    # Get column names for clarity
    time_idx_col = metadata["time_idx"]
    target_col = metadata["target"]
    
    # Process each prediction
    for i in range(len(index[0])):
        # Get index values
        if len(metadata["group_ids"]) == 1:
            group_id = {metadata["group_ids"][0]: index[0][i].item()}
        else:
            group_id = {metadata["group_ids"][j]: index[0][i][j].item() for j in range(len(metadata["group_ids"]))}
        
        time_idx = index[1][i].item()
        
        # Create forecast entry
        forecast_entry = {
            **group_id,
            time_idx_col: time_idx,
            f"{target_col}_forecast": float(y_hat[i, 0])
        }
        
        # Add prediction intervals if quantile predictions are available
        if hasattr(predictions, 'y') and hasattr(model.loss, 'quantiles') and len(model.loss.quantiles) > 1:
            for j, q in enumerate(model.loss.quantiles):
                forecast_entry[f"{target_col}_q{int(q*100)}"] = float(y_hat[i, j])
        else:
            # Add simple confidence intervals
            if hasattr(predictions, 'y'):
                y_true = predictions.y.cpu().numpy()
                errors = y_true[~np.isnan(y_true)] - y_hat[~np.isnan(y_true)]
                lower_percentile = (100 - confidence_interval) / 2
                upper_percentile = 100 - lower_percentile
                
                lower_bound = np.percentile(errors, lower_percentile) if len(errors) > 0 else -0.1
                upper_bound = np.percentile(errors, upper_percentile) if len(errors) > 0 else 0.1
                
                forecast_entry[f"{target_col}_lower"] = float(y_hat[i, 0] + lower_bound)
                forecast_entry[f"{target_col}_upper"] = float(y_hat[i, 0] + upper_bound)
        
        results.append(forecast_entry)
    
    return results

def calculate_metrics(predictions):
    """Calculate forecast accuracy metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    y_true = predictions.y.cpu().numpy()
    y_pred = predictions.output.cpu().numpy()
    
    # Only evaluate where we have actual values (not NaN)
    valid_mask = ~np.isnan(y_true[:, 0])
    
    if valid_mask.sum() > 0:
        mae = mean_absolute_error(y_true[valid_mask, 0], y_pred[valid_mask, 0])
        rmse = np.sqrt(mean_squared_error(y_true[valid_mask, 0], y_pred[valid_mask, 0]))
        
        # Calculate MAPE with protection against zeros
        mape_mask = (valid_mask) & (y_true[:, 0] != 0)
        mape = np.mean(np.abs((y_true[mape_mask, 0] - y_pred[mape_mask, 0]) / y_true[mape_mask, 0])) * 100 if mape_mask.sum() > 0 else None
        
        # Calculate SMAPE
        smape = np.mean(200.0 * np.abs(y_pred[valid_mask, 0] - y_true[valid_mask, 0]) / (np.abs(y_pred[valid_mask, 0]) + np.abs(y_true[valid_mask, 0]) + 1e-8))
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape) if mape is not None else None,
            "smape": float(smape)
        }
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TFT model as a REST API")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    
    args = parser.parse_args()
    
    # Run the FastAPI app
    uvicorn.run(app, host=args.host, port=args.port)
