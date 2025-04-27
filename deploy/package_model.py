#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import shutil
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def package_model(model_path, output_path, include_examples=True):
    """
    Package a TFT model for easy deployment and distribution.
    This creates a directory with:
    1. The model file (saved in a portable format)
    2. Metadata JSON
    3. Example inference code
    4. README with usage instructions
    """
    from pytorch_forecasting import TemporalFusionTransformer
    
    # Create output directory structure
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and metadata
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logging.info(f"Loading model from {model_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    
    # Look for metadata in the model directory
    model_dir = model_file.parent
    metadata_path = model_dir / "tft_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Loaded metadata from {metadata_path}")
    else:
        # Create basic metadata
        metadata = {
            "time_idx": "time_idx",
            "target": "target",
            "group_ids": ["group_id"],
            "max_encoder_length": model.hparams.max_encoder_length,
            "max_prediction_length": model.hparams.max_prediction_length,
        }
        logging.info("Created default metadata")
    
    # Save model and metadata to output directory
    model_output_path = output_dir / "model.pt"
    metadata_output_path = output_dir / "metadata.json"
    
    # Save the model with TorchScript for portability
    try:
        script_model = model.to_torchscript()
        torch.jit.save(script_model, model_output_path)
        logging.info(f"Saved TorchScript model to {model_output_path}")
    except Exception as e:
        logging.warning(f"Failed to save TorchScript model: {e}")
        logging.info("Falling back to regular model checkpoint")
        shutil.copy2(model_path, model_output_path)
    
    # Save metadata
    with open(metadata_output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {metadata_output_path}")
    
    # Create example inference code if requested
    if include_examples:
        write_example_code(output_dir, model, metadata)
    
    # Create README
    write_readme(output_dir)
    
    logging.info(f"Model successfully packaged to {output_dir}")
    return output_dir

def write_example_code(output_dir, model, metadata):
    """Write example code for model inference"""
    example_path = output_dir / "example_inference.py"
    
    code = f"""#!/usr/bin/env python3
import pandas as pd
import torch
import json
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

def load_model_and_metadata():
    # Load the packaged model
    model = torch.jit.load("model.pt")
    
    # Load metadata
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    return model, metadata

def prepare_data(df, metadata):
    # Ensure required columns exist
    required_cols = [metadata["time_idx"], metadata["target"]] + metadata["group_ids"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {{missing_cols}}")
    
    # Clean data types
    df = df.copy()
    df[metadata["time_idx"]] = df[metadata["time_idx"]].astype(int)
    for group_id in metadata["group_ids"]:
        df[group_id] = df[group_id].astype(str).astype('category')
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

def make_predictions(model, df, metadata, forecast_horizon=1):
    # Prepare the dataset
    dataset = TimeSeriesDataSet(
        df,
        time_idx=metadata["time_idx"],
        target=metadata["target"],
        group_ids=metadata["group_ids"],
        max_encoder_length=metadata.get("max_encoder_length", 30),
        max_prediction_length=forecast_horizon,
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
    y_hat = predictions.output.cpu().numpy()
    y_true = predictions.y.cpu().numpy() if hasattr(predictions, 'y') else None
    index = predictions.index
    
    # Create prediction DataFrame
    result_df = pd.DataFrame()
    
    # Process each prediction
    for i in range(len(index[0])):
        # Get group info
        if len(metadata["group_ids"]) == 1:
            group_id = {{metadata["group_ids"][0]: index[0][i].item()}}
        else:
            group_id = {{metadata["group_ids"][j]: index[0][i][j].item() for j in range(len(metadata["group_ids"]))}}
        
        time_idx = index[1][i].item()
        
        # Create row
        row = {{
            **group_id,
            metadata["time_idx"]: time_idx,
            f"{{metadata['target']}}_forecast": float(y_hat[i, 0])
        }}
        
        # Add actual value if available
        if y_true is not None and not np.isnan(y_true[i, 0]):
            row[metadata["target"]] = float(y_true[i, 0])
        
        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
    
    return result_df

def example_usage():
    # Load model and metadata
    model, metadata = load_model_and_metadata()
    
    # Create sample data
    data = {{
        "group_id": ["A", "A", "A", "B", "B", "B"],
        "time_idx": [1, 2, 3, 1, 2, 3],
        "target": [100, 120, 130, 200, 210, 220],
        "feature1": [10, 11, 12, 20, 21, 22],
        "feature2": [5, 6, 7, 15, 16, 17]
    }}
    df = pd.DataFrame(data)
    
    # Prepare data
    df = prepare_data(df, metadata)
    
    # Make predictions for next 2 steps
    predictions = make_predictions(model, df, metadata, forecast_horizon=2)
    
    print("Predictions:")
    print(predictions)
    
    # Save predictions
    predictions.to_csv("predictions.csv", index=False)
    print("Saved predictions to predictions.csv")

if __name__ == "__main__":
    example_usage()
"""
    
    with open(example_path, 'w') as f:
        f.write(code)
    
    logging.info(f"Created example code at {example_path}")

def write_readme(output_dir):
    """Write README.md file with usage instructions"""
    readme_path = output_dir / "README.md"
    
    content = """# Packaged Time Series Forecasting Model

This package contains a trained Temporal Fusion Transformer (TFT) model for time series forecasting.

## Contents

- `model.pt`: The trained model file
- `metadata.json`: Metadata needed for model usage
- `example_inference.py`: Example code for making predictions
- `README.md`: This file

## Requirements

```bash
pip install torch pandas pytorch-forecasting
```

## Usage

### Option 1: Using the example script

```bash
python example_inference.py
```

### Option 2: Custom integration

```python
import pandas as pd
import torch
import json

# Load model and metadata
model = torch.jit.load("model.pt")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# Prepare your data
# ...

# See example_inference.py for complete usage example
```

## Model Information

This model is a Temporal Fusion Transformer, designed for multi-horizon time series forecasting with interpretable attention mechanisms. See the metadata.json file for details on required inputs and model configuration.
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    logging.info(f"Created README at {readme_path}")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Package a TFT model for deployment")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--output", type=str, default="packaged_model", help="Output directory path")
    parser.add_argument("--no-examples", action="store_false", dest="include_examples", help="Skip including example code")
    
    args = parser.parse_args()
    
    package_model(args.model, args.output, args.include_examples)

if __name__ == "__main__":
    main()
