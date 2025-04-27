# Training Guide

This guide covers the process of training models using the MultimodalStockPredictor framework, with a particular focus on Temporal Fusion Transformer (TFT) models for time series forecasting.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Training Process Overview](#training-process-overview)
- [Training a TFT Model](#training-a-tft-model)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Handling Large Datasets](#handling-large-datasets)
- [Saving Models](#saving-models)
- [Advanced Options](#advanced-options)

## Prerequisites

Before training, ensure:
1. Your data is properly prepared (see [Data Preparation Guide](data_preparation.md))
2. Dependencies are installed: `pip install -r requirements.txt`
3. You have sufficient compute resources (GPU recommended for large models)

## Training Process Overview

The training process follows these steps:
1. Load and validate the dataset
2. Prepare data structures for the model
3. Initialize the model with specified hyperparameters
4. Train the model using mini-batch optimization
5. Validate performance on a holdout set
6. Save the trained model and metadata

## Training a TFT Model

To train a TFT model, use the `train_model.py` script with the `--tft` flag:

```bash
python train_model.py \
  --data-path Training_Data/tft_data.csv \
  --tft \
  --tft-meta Training_Data/tft_data_tft_meta.json \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --output-dir models/tft_model \
  --ts-mode
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data-path` | Path to input CSV file | required |
| `--tft-meta` | Path to TFT metadata JSON file | required |
| `--epochs` | Number of training epochs | 10 |
| `--batch-size` | Batch size for training | 32 |
| `--lr` | Learning rate | 0.001 |
| `--hidden-dim` | Hidden dimension size | 64 |
| `--lstm-layers` | Number of LSTM layers | 2 |
| `--dropout` | Dropout rate | 0.1 |

### Time Series Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ts-mode` | Enable time series specific optimizations | False |
| `--backtest-windows` | Number of windows for backtesting | 3 |
| `--seasonal-period` | Period for seasonal features | 5 |
| `--fourier-terms` | Number of Fourier terms | 2 |
| `--decompose-trend` | Use trend decomposition | False |
| `--differencing` | Apply first differencing | False |
| `--quantile-forecast` | Enable probabilistic forecasting | False |

## Hyperparameter Tuning

To automatically tune hyperparameters:

```bash
python train_model.py \
  --data-path Training_Data/tft_data.csv \
  --tft \
  --tft-meta Training_Data/tft_data_tft_meta.json \
  --output-dir models/tft_tuned \
  --ts-mode \
  --hp-tuning-trials 20 \
  --optimization-metric SMAPE
```

This uses Optuna to search for optimal hyperparameters, testing 20 different configurations.

## Handling Large Datasets

For large datasets, use these memory optimization techniques:

```bash
python train_model.py \
  --data-path large_dataset.csv \
  --tft \
  --tft-meta metadata.json \
  --batch-size 16 \
  --output-dir models/tft_large \
  --num-workers 4 \
  --accumulate-grad-batches 4
```

## Saving Models

Models are automatically saved to the specified `--output-dir`. The saved artifacts include:

- Model checkpoint (`.ckpt` file)
- Training metadata (`.json` file)
- Training history plot (`.png` file)

## Advanced Options

### Logging Options
- `--log-level INFO`: Set logging level
- `--tensorboard`: Enable TensorBoard logging

### Training Control
- `--early-stopping-patience 10`: Set patience for early stopping
- `--gradient-clip-val 0.1`: Set gradient clipping value
- `--accumulate-grad-batches 2`: Accumulate gradients over multiple batches

### Model Architecture
- `--attention-head-size 4`: Number of attention heads
- `--hidden-continuous-size 32`: Hidden size for continuous variables

See `python train_model.py --help` for a complete list of options.
