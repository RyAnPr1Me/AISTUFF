# MultimodalStockPredictor

A PyTorch model for stock movement prediction using text, tabular, time series, and optional vision data.

## Capabilities

- Leverages transformer-based text encoders (e.g., BERT) for financial news, reports, or sentiment.
- Encodes tabular numerical features (e.g., technical indicators, fundamentals).
- Supports Temporal Fusion Transformer (TFT) for advanced time series forecasting.
- Optionally integrates vision models for chart or image data.
- Flexible fusion head with configurable depth, activation, dropout, and normalization.
- Suitable for classification tasks (e.g., up/down/neutral movement) and regression (time series forecasting).

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AISTUFF.git
cd AISTUFF

# Install dependencies
pip install -r requirements.txt
```

## Getting Started: Training Example

### 1. Prepare your data

For classification tasks with text and tabular data:
```bash
# Download and prepare stock data
python download_and_prepare_stock_data.py --symbol AAPL --start 2018-01-01 --end 2023-01-01

# Format data for ALBERT text encoder
python format_for_albert.py --folder Training_Data

# Validate and prepare data
python validate_and_prepare.py --data-dir Training_Data --output Training_Data/validated_data.csv
```

For time series forecasting with TFT:
```bash
# Prepare data with TFT format
python validate_and_prepare.py --data-dir Training_Data --output Training_Data/tft_data.csv --tft --tft-enhanced
```

### 2. Train models

For classification:
```bash
python train_model.py --data-path Training_Data/validated_data.csv --epochs 10 --batch-size 32 --output-dir models/stock_classifier
```

For time series forecasting with TFT:
```bash
python train_model.py --data-path Training_Data/tft_data.csv --tft --tft-meta Training_Data/tft_data_tft_meta.json --epochs 50 --batch-size 64 --output-dir models/tft_forecaster --ts-mode --backtest-windows 3
```

## Time Series Forecasting with TFT

### Complete TFT Workflow

1. **Data Preparation**
   ```bash
   python validate_and_prepare.py --data-dir your_data_dir --output output.csv --tft --tft-enhanced --tft-horizon 5 --tft-lookback 30
   ```

2. **Training with Advanced Options**
   ```bash
   python train_model.py --data-path output.csv --tft --tft-meta output_tft_meta.json --epochs 100 --batch-size 64 --output-dir models/tft_model --ts-mode --backtest-windows 3 --hp-tuning-trials 20 --seasonal-period 7 --fourier-terms 3 --quantile-forecast
   ```

3. **Evaluation and Backtesting**
   ```bash
   python evaluate_tft_model.py --model models/tft_model/final_model/tft_model.ckpt --data test_data.csv --output-dir evaluation_results --plot-predictions
   
   python run_tft_backtesting.py --model models/tft_model/final_model/tft_model.ckpt --data test_data.csv --n-windows 5 --output-dir backtest_results
   ```

4. **Making Predictions**
   ```bash
   python run_tft_prediction.py --model models/tft_model/final_model/tft_model.ckpt --data new_data.csv --output-dir predictions
   ```

5. **Interactive Dashboard**
   ```bash
   streamlit run tft_dashboard.py -- --model models/tft_model/final_model/tft_model.ckpt --data new_data.csv
   ```

### Key TFT Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tft-horizon` | Forecast horizon (prediction length) | 1 |
| `--tft-lookback` | Historical context length (encoder length) | 30 |
| `--seasonal-period` | Period for seasonal features (5=weekly) | 5 |
| `--fourier-terms` | Number of Fourier terms for seasonality | 2 |
| `--hp-tuning-trials` | Number of hyperparameter tuning trials | 0 |
| `--quantile-forecast` | Enable probabilistic forecasting | False |

## Model Deployment

### Option 1: REST API with FastAPI

Create a REST API endpoint using the provided script:

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn python-multipart

# Run the API server
python deploy/tft_api.py --model models/tft_model/final_model/tft_model.ckpt --port 8000
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs.

### Option 2: Batch Prediction

For batch prediction on large datasets:

```bash
python run_tft_prediction.py --model models/tft_model/final_model/tft_model.ckpt --data large_dataset.csv --output-dir batch_predictions
```

### Option 3: Interactive Dashboard

Deploy the interactive dashboard for business users:

```bash
# Install Streamlit
pip install streamlit

# Run the dashboard
streamlit run tft_dashboard.py -- --model models/tft_model/final_model/tft_model.ckpt --data new_data.csv
```

## Customization

- Change `fusion_layers`, `activation`, `tabular_dropout`, etc. when instantiating the model for different architectures.
- For TFT models, adjust hyperparameters like hidden size, LSTM layers, and attention heads.
- See the [detailed documentation](docs/README.md) for more customization options.
