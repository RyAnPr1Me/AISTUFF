import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    """
    Simple time series encoder using LSTM.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
