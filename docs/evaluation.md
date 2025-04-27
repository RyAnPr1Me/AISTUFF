# Evaluation Guide

This guide covers how to evaluate trained models, particularly focused on Temporal Fusion Transformer (TFT) models for time series forecasting.

## Table of Contents
- [Evaluation Overview](#evaluation-overview)
- [Basic Model Evaluation](#basic-model-evaluation)
- [Backtesting](#backtesting)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizing Results](#visualizing-results)
- [Feature Importance Analysis](#feature-importance-analysis)
- [Error Analysis](#error-analysis)

## Evaluation Overview

Model evaluation is a critical step to ensure your forecasting model performs reliably. The MultimodalStockPredictor provides several tools for comprehensive evaluation:

1. Basic evaluation on a test dataset
2. Time-based backtesting with multiple validation windows
3. Feature importance analysis
4. Visualization of forecasts and errors

## Basic Model Evaluation

To evaluate a trained TFT model on new data:

```bash
python evaluate_tft_model.py \
  --model models/tft_model/final_model/tft_model.ckpt \
  --data test_data.csv \
  --output-dir evaluation_results \
  --plot-predictions
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--model` | Path to trained model checkpoint |
| `--data` | Path to evaluation data CSV |
| `--metadata` | Optional path to metadata JSON |
| `--output-dir` | Directory to save evaluation results |
| `--plot-predictions` | Generate individual prediction plots |

### Output

The evaluation script produces:
- Evaluation metrics (RMSE, MAE, MAPE, SMAPE)
- Scatter plot of predictions vs actual values
- Residual distribution plot
- QQ plot for residual normality check
- Feature importance plot (if available)

## Backtesting

Backtesting evaluates your model across multiple time windows to test how it would have performed historically:

```bash
python run_tft_backtesting.py \
  --model models/tft_model/final_model/tft_model.ckpt \
  --data historical_data.csv \
  --output-dir backtest_results \
  --n-windows 5
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--model` | Path to trained model checkpoint |
| `--data` | Path to historical data CSV |
| `--n-windows` | Number of validation windows |
| `--output-dir` | Directory to save backtest results |

### Understanding Backtest Results

The backtest results include:
- Summary metrics across all windows
- Individual window performance
- Visualizations of forecasts in each window
- Performance trends over time

## Evaluation Metrics

Key metrics for time series evaluation:

| Metric | Description | Formula |
|--------|-------------|---------|
| RMSE | Root Mean Squared Error | √(Σ(yᵢ - ŷᵢ)² / n) |
| MAE | Mean Absolute Error | Σ\|yᵢ - ŷᵢ\| / n |
| MAPE | Mean Absolute Percentage Error | 100 × Σ(\|yᵢ - ŷᵢ\| / \|yᵢ\|) / n |
| SMAPE | Symmetric Mean Absolute Percentage Error | 200 × Σ(\|yᵢ - ŷᵢ\| / (\|yᵢ\| + \|ŷᵢ\|)) / n |

Lower values indicate better performance for all metrics.

## Visualizing Results

For an interactive visualization of model performance:

```bash
streamlit run tft_dashboard.py -- \
  --model models/tft_model/final_model/tft_model.ckpt \
  --data test_data.csv
```

The dashboard provides:
- Interactive forecast visualization
- Confidence interval adjustments
- Time series decomposition analysis
- Performance metrics
- Downloadable forecasts as CSV

## Feature Importance Analysis

TFT models provide built-in feature importance analysis. To examine this:

1. View the automatically generated feature importance plots in your evaluation output directory
2. The plots show:
   - Variable importance across all time steps
   - Attention patterns across the time series
   - Relative importance of static vs. time-varying features

## Error Analysis

To perform a detailed error analysis:

1. Examine the residual plots from evaluation
2. Check for patterns in errors (e.g., heteroscedasticity, autocorrelation)
3. Analyze errors by group or time period:

```bash
python error_analysis.py \
  --predictions evaluation_results/predictions.csv \
  --output-dir error_analysis
```

(Note: Create a custom error_analysis.py script tailored to your specific needs)
