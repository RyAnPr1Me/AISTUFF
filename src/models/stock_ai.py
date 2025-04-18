"""
MultimodalStockPredictor: A PyTorch model for stock movement prediction using text, tabular, and optional vision data.

Capabilities:
- Leverages transformer-based text encoders (e.g., BERT) for financial news, reports, or sentiment.
- Encodes tabular numerical features (e.g., technical indicators, fundamentals).
- Optionally integrates vision models for chart or image data.
- Flexible fusion head with configurable depth, activation, dropout, and normalization.
- Suitable for classification tasks (e.g., up/down/neutral movement).

Getting Started: Training Example
-------------------------------
1. Prepare your data:
   - Tokenize text using a HuggingFace tokenizer matching `text_model_name`.
   - Prepare tabular features as torch tensors (shape: [batch_size, tabular_dim]).
   - (Optional) Prepare vision inputs as required by the vision transformer.

2. Instantiate the model:
   ```python
   from stock_ai import MultimodalStockPredictor
   import torch
   from transformers import AutoTokenizer

   model = MultimodalStockPredictor()
   tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
   ```

3. Prepare a batch:
   ```python
   text_inputs = tokenizer(["AAPL earnings beat expectations"], return_tensors="pt", padding=True, truncation=True)
   tabular_inputs = torch.randn(1, 64)  # Example tabular data
   labels = torch.tensor([1])  # Example label
   ```

4. Forward pass and loss:
   ```python
   logits = model(text_inputs, tabular_inputs)
   loss_fn = torch.nn.CrossEntropyLoss()
   loss = loss_fn(logits, labels)
   ```

5. Training loop:
   ```python
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
   for epoch in range(num_epochs):
       optimizer.zero_grad()
       logits = model(text_inputs, tabular_inputs)
       loss = loss_fn(logits, labels)
       loss.backward()
       optimizer.step()
   ```

See the class docstring and code for more customization options.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultimodalStockPredictor(nn.Module):
    """
    MultimodalStockPredictor combines text, tabular, and optional vision data for stock movement prediction.
    Supports configurable fusion, optional attention-based fusion, and feature extraction.
    """
    def __init__(self, 
                 text_model_name="bert-large-uncased",
                 vision_model_name=None,
                 tabular_dim=64,
                 hidden_dim=1024,
                 num_labels=3,
                 fusion_layers=2,
                 activation=nn.ReLU,
                 tabular_dropout=0.1,
                 fusion_dropout=0.2,
                 fusion_layernorm=True,
                 use_attention_fusion=False,
                 use_residual_fusion=False):
        """
        Args:
            text_model_name (str): HuggingFace model name for text encoder.
            vision_model_name (str or None): HuggingFace model name for vision encoder.
            tabular_dim (int): Input dimension for tabular data.
            hidden_dim (int): Hidden dimension for encoders and fusion.
            num_labels (int): Number of output classes.
            fusion_layers (int): Number of layers in the fusion head.
            activation (nn.Module): Activation function.
            tabular_dropout (float): Dropout for tabular encoder.
            fusion_dropout (float): Dropout for fusion head.
            fusion_layernorm (bool): Use LayerNorm in fusion head.
            use_attention_fusion (bool): If True, use attention-based fusion.
            use_residual_fusion (bool): If True, add residual connection in fusion head.
        """
        super().__init__()
        # Text encoder (large transformer)
        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, config=self.text_config)
        
        # Optional: Vision encoder (for chart images, etc.)
        if vision_model_name:
            self.vision_config = AutoConfig.from_pretrained(vision_model_name)
            self.vision_encoder = AutoModel.from_pretrained(vision_model_name, config=self.vision_config)
            vision_out_dim = self.vision_config.hidden_size
        else:
            self.vision_encoder = None
            vision_out_dim = 0

        # Tabular (numerical) data encoder with dropout
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            activation(),
            nn.Dropout(tabular_dropout),
            nn.LayerNorm(hidden_dim)
        )

        self.use_attention_fusion = use_attention_fusion
        self.use_residual_fusion = use_residual_fusion

        # Attention-based fusion (optional)
        if self.use_attention_fusion:
            fusion_input_dim = self.text_config.hidden_size + hidden_dim
            if vision_model_name:
                fusion_input_dim += self.vision_config.hidden_size
            self.attn_fusion = nn.MultiheadAttention(embed_dim=fusion_input_dim, num_heads=4, batch_first=True)
        else:
            self.attn_fusion = None

        # Fusion and prediction head (deeper, configurable)
        fusion_dim = self.text_config.hidden_size + vision_out_dim + hidden_dim
        fusion_head_layers = []
        in_dim = fusion_dim
        for i in range(fusion_layers - 1):
            fusion_head_layers.append(nn.Linear(in_dim, hidden_dim))
            fusion_head_layers.append(activation())
            fusion_head_layers.append(nn.Dropout(fusion_dropout))
            if fusion_layernorm:
                fusion_head_layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        fusion_head_layers.append(nn.Linear(in_dim, num_labels))
        self.fusion_head = nn.Sequential(*fusion_head_layers)

    def forward(self, text_inputs, tabular_inputs, vision_inputs=None):
        """
        Forward pass for the model.
        Args:
            text_inputs (dict): Tokenized text inputs for transformer.
            tabular_inputs (Tensor): Tabular features.
            vision_inputs (dict or None): Vision transformer inputs.
        Returns:
            logits (Tensor): Output logits.
        """
        # Text encoding
        text_outputs = self.text_encoder(**text_inputs)
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Tabular encoding
        tabular_feat = self.tabular_encoder(tabular_inputs)

        # Vision encoding (optional)
        if self.vision_encoder and vision_inputs is not None:
            vision_outputs = self.vision_encoder(**vision_inputs)
            vision_feat = vision_outputs.last_hidden_state[:, 0, :]
            features = torch.cat([text_feat, tabular_feat, vision_feat], dim=1)
        else:
            features = torch.cat([text_feat, tabular_feat], dim=1)

        # Attention-based fusion (optional)
        if self.attn_fusion is not None:
            # Add sequence dimension for attention: (batch, seq, feature)
            features_seq = features.unsqueeze(1)
            attn_out, _ = self.attn_fusion(features_seq, features_seq, features_seq)
            features = attn_out.squeeze(1)

        # Residual connection (optional)
        if self.use_residual_fusion:
            fusion_input = features
            logits = self.fusion_head(features)
            logits += fusion_input[:, :logits.shape[1]]  # Residual on matching dims
        else:
            logits = self.fusion_head(features)
        return logits

    def extract_features(self, text_inputs, tabular_inputs, vision_inputs=None):
        """
        Extract intermediate features before the final prediction head.
        Returns:
            features (Tensor): Concatenated feature vector.
        """
        text_outputs = self.text_encoder(**text_inputs)
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        tabular_feat = self.tabular_encoder(tabular_inputs)
        if self.vision_encoder and vision_inputs is not None:
            vision_outputs = self.vision_encoder(**vision_inputs)
            vision_feat = vision_outputs.last_hidden_state[:, 0, :]
            features = torch.cat([text_feat, tabular_feat, vision_feat], dim=1)
        else:
            features = torch.cat([text_feat, tabular_feat], dim=1)
        return features

# Example usage:
# model = MultimodalStockPredictor(fusion_layers=3, activation=nn.GELU)
# text_inputs = tokenizer("AAPL earnings beat expectations", return_tensors="pt")
# tabular_inputs = torch.randn(1, 64)
# output = model(text_inputs, tabular_inputs)