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
# Add imports for optional encoders
from .audio_encoder import AudioEncoder
from .time_series_encoder import TimeSeriesEncoder

class MultimodalStockPredictor(nn.Module):
    """
    MultimodalStockPredictor combines text, tabular, vision, audio, and time series data for stock movement prediction.
    Supports configurable fusion, optional attention-based fusion, and feature extraction.
    """
    def __init__(self, 
                 text_model_name="bert-large-uncased",
                 vision_model_name=None,
                 tabular_dim=64,
                 audio_dim=None,
                 time_series_dim=None,
                 hidden_dim=1024,
                 num_labels=3,
                 fusion_layers=2,
                 activation=nn.ReLU,
                 tabular_dropout=0.1,
                 fusion_dropout=0.2,
                 fusion_layernorm=True,
                 use_attention_fusion=False,
                 use_residual_fusion=False,
                 use_audio=False,
                 use_time_series=False,
                 audio_hidden_dim=128,
                 audio_out_dim=128,
                 time_series_hidden_dim=128,
                 time_series_out_dim=128,
                 audio_encoder=None,
                 time_series_encoder=None):
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
            use_audio (bool): If True, use audio encoder.
            use_time_series (bool): If True, use time series encoder.
            audio_dim (int or None): Input dimension for audio data.
            time_series_dim (int or None): Input dimension for time series data.
            audio_hidden_dim (int): Hidden dimension for audio encoder.
            audio_out_dim (int): Output dimension for audio encoder.
            time_series_hidden_dim (int): Hidden dimension for time series encoder.
            time_series_out_dim (int): Output dimension for time series encoder.
            audio_encoder (nn.Module or None): Custom audio encoder.
            time_series_encoder (nn.Module or None): Custom time series encoder.
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

        # Optional: Audio encoder
        self.use_audio = use_audio
        if use_audio and audio_dim is not None:
            self.audio_encoder = audio_encoder or AudioEncoder(audio_dim, audio_hidden_dim, audio_out_dim)
            audio_out_dim_ = audio_out_dim
        else:
            self.audio_encoder = None
            audio_out_dim_ = 0

        # Optional: Time series encoder
        self.use_time_series = use_time_series
        if use_time_series and time_series_dim is not None:
            self.time_series_encoder = time_series_encoder or TimeSeriesEncoder(time_series_dim, time_series_hidden_dim, time_series_out_dim)
            time_series_out_dim_ = time_series_out_dim
        else:
            self.time_series_encoder = None
            time_series_out_dim_ = 0

        # Attention-based fusion (optional)
        if self.use_attention_fusion:
            fusion_input_dim = self.text_config.hidden_size + hidden_dim + vision_out_dim + audio_out_dim_ + time_series_out_dim_
            self.attn_fusion = nn.MultiheadAttention(embed_dim=fusion_input_dim, num_heads=4, batch_first=True)
        else:
            self.attn_fusion = None

        # Fusion and prediction head (deeper, configurable)
        fusion_dim = self.text_config.hidden_size + vision_out_dim + hidden_dim + audio_out_dim_ + time_series_out_dim_
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

    def forward(self, text_inputs, tabular_inputs, vision_inputs=None, audio_inputs=None, time_series_inputs=None):
        """
        Forward pass for the model.
        Args:
            text_inputs (dict): Tokenized text inputs for transformer.
            tabular_inputs (Tensor): Tabular features.
            vision_inputs (dict or None): Vision transformer inputs.
            audio_inputs (Tensor or None): Audio features.
            time_series_inputs (Tensor or None): Time series features.
        Returns:
            logits (Tensor): Output logits.
        """
        # Text encoding
        text_outputs = self.text_encoder(**text_inputs)
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Tabular encoding
        tabular_feat = self.tabular_encoder(tabular_inputs)

        features_list = [text_feat, tabular_feat]

        # Vision encoding (optional)
        if self.vision_encoder and vision_inputs is not None:
            vision_outputs = self.vision_encoder(**vision_inputs)
            vision_feat = vision_outputs.last_hidden_state[:, 0, :]
            features_list.append(vision_feat)

        # Audio encoding (optional)
        if self.audio_encoder and audio_inputs is not None:
            audio_feat = self.audio_encoder(audio_inputs)
            features_list.append(audio_feat)

        # Time series encoding (optional)
        if self.time_series_encoder and time_series_inputs is not None:
            ts_feat = self.time_series_encoder(time_series_inputs)
            features_list.append(ts_feat)

        features = torch.cat(features_list, dim=1)

        # Attention-based fusion (optional)
        if self.attn_fusion is not None:
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

    def extract_features(self, text_inputs, tabular_inputs, vision_inputs=None, audio_inputs=None, time_series_inputs=None):
        """
        Extract intermediate features before the final prediction head.
        Returns:
            features (Tensor): Concatenated feature vector.
        """
        text_outputs = self.text_encoder(**text_inputs)
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        tabular_feat = self.tabular_encoder(tabular_inputs)
        features_list = [text_feat, tabular_feat]
        if self.vision_encoder and vision_inputs is not None:
            vision_outputs = self.vision_encoder(**vision_inputs)
            vision_feat = vision_outputs.last_hidden_state[:, 0, :]
            features_list.append(vision_feat)
        if self.audio_encoder and audio_inputs is not None:
            audio_feat = self.audio_encoder(audio_inputs)
            features_list.append(audio_feat)
        if self.time_series_encoder and time_series_inputs is not None:
            ts_feat = self.time_series_encoder(time_series_inputs)
            features_list.append(ts_feat)
        features = torch.cat(features_list, dim=1)
        return features

# Example usage:
# model = MultimodalStockPredictor(fusion_layers=3, activation=nn.GELU)
# text_inputs = tokenizer("AAPL earnings beat expectations", return_tensors="pt")
# tabular_inputs = torch.randn(1, 64)
# output = model(text_inputs, tabular_inputs)