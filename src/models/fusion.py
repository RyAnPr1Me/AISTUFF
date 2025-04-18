import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Gated fusion for two or more modalities.
    """
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.gates = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])
        self.fcs = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])

    def forward(self, features):
        # features: list of tensors [batch, dim]
        gated = []
        for i, feat in enumerate(features):
            gate = torch.sigmoid(self.gates[i](feat))
            proj = self.fcs[i](feat)
            gated.append(gate * proj)
        return sum(gated)

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for two modalities.
    """
    def __init__(self, dim_q, dim_kv, dim_out):
        super().__init__()
        self.query = nn.Linear(dim_q, dim_out)
        self.key = nn.Linear(dim_kv, dim_out)
        self.value = nn.Linear(dim_kv, dim_out)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv):
        # q: (batch, dim_q), kv: (batch, dim_kv)
        Q = self.query(q).unsqueeze(1)  # (batch, 1, dim_out)
        K = self.key(kv).unsqueeze(1)   # (batch, 1, dim_out)
        V = self.value(kv).unsqueeze(1) # (batch, 1, dim_out)
        attn = self.softmax(torch.bmm(Q, K.transpose(1,2)))
        out = torch.bmm(attn, V).squeeze(1)
        return out
