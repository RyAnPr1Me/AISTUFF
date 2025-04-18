import torch
import torch.nn as nn
try:
    from .image_encoder import ImageEncoder
except ImportError:
    ImageEncoder = None

try:
    from .text_encoder import TextEncoder
except ImportError:
    TextEncoder = None

try:
    from .audio_encoder import AudioEncoder
except ImportError:
    AudioEncoder = None

try:
    from .time_series_encoder import TimeSeriesEncoder
except ImportError:
    TimeSeriesEncoder = None

from .fusion import GatedFusion, CrossModalAttention

class MultiModalModel(nn.Module):
    """
    Flexible multimodal model supporting image, text, audio, and time series.
    Supports custom encoders and fusion strategies.
    """
    def __init__(self, config):
        super().__init__()
        # config: dict with keys for each modality and fusion
        self.encoders = nn.ModuleDict()
        if 'image' in config:
            self.encoders['image'] = config.get('image_encoder') or ImageEncoder(**config['image'])
        if 'text' in config:
            self.encoders['text'] = config.get('text_encoder') or TextEncoder(**config['text'])
        if 'audio' in config:
            self.encoders['audio'] = config.get('audio_encoder') or AudioEncoder(**config['audio'])
        if 'time_series' in config:
            self.encoders['time_series'] = config.get('time_series_encoder') or TimeSeriesEncoder(**config['time_series'])
        self.fusion_type = config.get('fusion', 'concat')
        if self.fusion_type == 'gated':
            input_dims = [enc.out_features for enc in self.encoders.values()]
            self.fusion = GatedFusion(input_dims, config['fusion_output_dim'])
        elif self.fusion_type == 'cross_attention':
            # Assume two modalities for simplicity
            dims = list(self.encoders.values())
            self.fusion = CrossModalAttention(dims[0].out_features, dims[1].out_features, config['fusion_output_dim'])
        else:
            self.fusion = None  # fallback to concat

        self.classifier = nn.Linear(sum(enc.out_features for enc in self.encoders.values()), config['num_classes'])

    def forward(self, inputs):
        # inputs: dict of modality_name -> tensor
        feats = []
        for name, encoder in self.encoders.items():
            feats.append(encoder(inputs[name]))
        if self.fusion_type == 'gated':
            fused = self.fusion(feats)
        elif self.fusion_type == 'cross_attention':
            fused = self.fusion(feats[0], feats[1])
        else:
            fused = torch.cat(feats, dim=-1)
        return self.classifier(fused)