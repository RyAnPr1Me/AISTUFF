import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    """
    Simple audio encoder using 1D CNN.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.cnn(x)
        return out.squeeze(-1)
