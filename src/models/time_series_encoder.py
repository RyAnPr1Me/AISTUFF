import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    """
    Time series encoder that can be used standalone or with TFT
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM for time series
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.out_features = output_dim  # For compatibility with fusion module
        
    def forward(self, x):
        """
        Args:
            x: Input time series [batch_size, seq_length, features] or [batch_size, features]
        """
        # Ensure input is 3D: [batch, seq, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take final time step or apply attention
        final_hidden = lstm_out[:, -1, :]
        
        # Project to output dimension
        output = self.fc(self.dropout(final_hidden))
        
        return output
