# Multimodal Model

## Features

- Supports image, text, audio, and time series modalities
- Flexible fusion strategies: concat, gated fusion, cross-modal attention
- Custom encoders and hyperparameters via config
- Training utilities: early stopping, learning rate schedulers
- Interpretability: feature importance, attention visualization

## Example Usage

```python
from models.multimodal_model import MultiModalModel
from models.training_utils import EarlyStopping, get_scheduler

config = {
    'image': {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 128},
    'text': {'vocab_size': 10000, 'embed_dim': 128, 'output_dim': 128},
    'audio': {'input_dim': 1, 'hidden_dim': 32, 'output_dim': 64},
    'fusion': 'gated',
    'fusion_output_dim': 128,
    # ...other hyperparameters...
}
model = MultiModalModel(config)
# ...training loop with EarlyStopping and scheduler...
```

## Interpretability

```python
from models.interpretability import compute_feature_importance, visualize_attention
import torch.nn.functional as F

# Compute feature importance
importances = compute_feature_importance(model, batch_inputs, batch_targets, F.cross_entropy)

# Visualize attention (if using attention-based fusion)
# visualize_attention(attn_weights)
```
