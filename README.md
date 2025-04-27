# MultimodalStockPredictor

A PyTorch model for stock movement prediction using text, tabular, and optional vision data.

## Capabilities

- Leverages transformer-based text encoders (e.g., BERT) for financial news, reports, or sentiment.
- Encodes tabular numerical features (e.g., technical indicators, fundamentals).
- Optionally integrates vision models for chart or image data.
- Flexible fusion head with configurable depth, activation, dropout, and normalization.
- Suitable for classification tasks (e.g., up/down/neutral movement).

## Getting Started: Training Example

### 1. Prepare your data

- Tokenize text using a HuggingFace tokenizer matching `text_model_name`.
- Prepare tabular features as torch tensors (shape: `[batch_size, tabular_dim]`).
- (Optional) Prepare vision inputs as required by the vision transformer.

### 2. Instantiate the model

```python
from stock_ai import MultimodalStockPredictor
import torch
from transformers import AutoTokenizer

model = MultimodalStockPredictor()
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
```

### 3. Prepare a batch

```python
text_inputs = tokenizer(["AAPL earnings beat expectations"], return_tensors="pt", padding=True, truncation=True)
tabular_inputs = torch.randn(1, 64)  # Example tabular data
labels = torch.tensor([1])  # Example label
```

### 4. Forward pass and loss

```python
logits = model(text_inputs, tabular_inputs)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

### 5. Training loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = model(text_inputs, tabular_inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
```

## Customization

- Change `fusion_layers`, `activation`, `tabular_dropout`, etc. when instantiating the model for different architectures.
- See the source code for more details and options.

## Temporal Fusion Transformer Support

This project now includes support for Temporal Fusion Transformer (TFT) models for time series forecasting. TFT combines high-performance multi-horizon forecasting with interpretability features.

### TFT Data Preparation

To prepare data for TFT:

```bash
python validate_and_prepare.py --data-dir your_data_dir --output output.csv --tft
```

This creates:
- A CSV file with data formatted for TFT
- A metadata JSON file describing variable types

### TFT Training

Train a TFT model:

```bash
python train_model.py --data-path output.csv --tft --tft-meta output_tft_meta.json --epochs 50
```

### TFT Prediction

Make predictions with a trained model:

```bash
python run_tft_prediction.py --model path/to/model.ckpt --data new_data.csv --meta output_tft_meta.json
```

### TFT Parameters

- `max-encoder-length`: History length used for prediction (default: 30)
- `max-prediction-length`: Forecast horizon (default: 1)
