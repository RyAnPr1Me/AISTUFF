# Multimodal Model

A flexible PyTorch-based framework for building, training, and interpreting multimodal models with support for image, text, audio, and time series data.

---

## Features

- **Multiple Modalities**: Image, text, audio, and time series support out-of-the-box.
- **Flexible Fusion**: Choose from concatenation, gated fusion, or cross-modal attention.
- **Custom Encoders**: Plug in your own encoders or use provided defaults.
- **Configurable Hyperparameters**: Easily adjust model and training settings.
- **Training Utilities**: Early stopping and learning rate schedulers included.
- **Interpretability**: Feature importance and attention visualization utilities.

---

## Quick Start

### 1. Installation

Make sure you have `torch` and `matplotlib` installed:

```bash
pip install torch matplotlib
```

### 2. Prepare Your Data

Organize your data as dictionaries mapping modality names to tensors. For example:

```python
batch_inputs = {
    'image': torch.randn(batch_size, 3, 224, 224),        # Example image batch
    'text': torch.randint(0, 10000, (batch_size, 50)),    # Example text batch (token ids)
    'audio': torch.randn(batch_size, 1, 16000),           # Example audio batch
    'time_series': torch.randn(batch_size, 100, 8),       # Example time series batch
}
batch_targets = torch.randint(0, num_classes, (batch_size,))
```

### 3. Define Model Configuration

Specify the modalities, encoders, fusion strategy, and other hyperparameters:

```python
config = {
    'image': {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 128},
    'text': {'vocab_size': 10000, 'embed_dim': 128, 'output_dim': 128},
    'audio': {'input_dim': 1, 'hidden_dim': 32, 'output_dim': 64},
    'time_series': {'input_dim': 8, 'hidden_dim': 32, 'output_dim': 64},
    'fusion': 'gated',  # Options: 'concat', 'gated', 'cross_attention'
    'fusion_output_dim': 128,
    # Optionally, you can provide custom encoder modules:
    # 'audio_encoder': MyCustomAudioEncoder(...),
    # 'time_series_encoder': MyCustomTimeSeriesEncoder(...),
    # ...other hyperparameters...
}
```

### 4. Initialize the Model

```python
from models.multimodal_model import MultiModalModel

model = MultiModalModel(config)
```

### 5. Training Loop Example

```python
import torch.optim as optim
from models.training_utils import EarlyStopping, get_scheduler

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = get_scheduler(optimizer, scheduler_type='plateau', patience=2)
early_stopper = EarlyStopping(patience=5)

for epoch in range(num_epochs):
    model.train()
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss += loss_fn(val_outputs, val_targets).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    early_stopper(val_loss)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break
```

### 6. Inference

```python
model.eval()
with torch.no_grad():
    predictions = model(batch_inputs)
    predicted_classes = predictions.argmax(dim=1)
```

---

## Interpretability

### Feature Importance

Compute feature importance using input gradients:

```python
from models.interpretability import compute_feature_importance
import torch.nn.functional as F

importances = compute_feature_importance(model, batch_inputs, batch_targets, F.cross_entropy)
print(importances)  # Dictionary of modality -> importance scores
```

### Attention Visualization

If using attention-based fusion, visualize attention weights:

```python
from models.interpretability import visualize_attention

# Suppose attn_weights is a tensor or numpy array of attention weights
visualize_attention(attn_weights)
```

---

## Advanced Usage

- **Custom Encoders**: Pass your own encoder modules in the config (e.g., `'audio_encoder': MyAudioEncoder(...)`).
- **Hyperparameter Tuning**: Adjust encoder/fusion parameters in the config dictionary.
- **Fusion Strategies**: Switch between `'concat'`, `'gated'`, and `'cross_attention'` via the `'fusion'` key.

---

## Example Configuration

```python
config = {
    'image': {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 128},
    'text': {'vocab_size': 10000, 'embed_dim': 128, 'output_dim': 128},
    'audio': {'input_dim': 1, 'hidden_dim': 32, 'output_dim': 64},
    'fusion': 'cross_attention',
    'fusion_output_dim': 128,
}
```

---

## References

- See `src/models/` for encoder and fusion module implementations.
- See `src/models/training_utils.py` for training helpers.
- See `src/models/interpretability.py` for interpretability utilities.

---
