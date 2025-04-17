import torch
import torch.nn as nn
from transformers import AutoConfig

class StockTransformer(nn.Module):
    def __init__(
        self,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        ff_dim=4096,
        max_sequence_length=1000,
        dropout=0.1
    ):
        super().__init__()
        
        # This configuration will give us approximately 500M parameters
        self.config = AutoConfig.from_pretrained(
            "bert-large-uncased",
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            intermediate_size=ff_dim,
            max_position_embeddings=max_sequence_length,
        )
        
        # Input embedding layers
        self.price_embeddings = nn.Linear(5, d_model)  # OHLCV data
        self.technical_embeddings = nn.Linear(32, d_model)  # Technical indicators
        self.positional_embeddings = nn.Embedding(max_sequence_length, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 3)  # Predict: Buy, Sell, Hold
        )
        
    def forward(self, price_data, technical_data, timestamps):
        batch_size, seq_len, _ = price_data.shape
        
        # Create embeddings
        price_embed = self.price_embeddings(price_data)
        tech_embed = self.technical_embeddings(technical_data)
        pos_embed = self.positional_embeddings(timestamps)
        
        # Combine all embeddings
        x = price_embed + tech_embed + pos_embed
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Get predictions for the last timestamp
        predictions = self.prediction_head(x[:, -1, :])
        
        return predictions

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x