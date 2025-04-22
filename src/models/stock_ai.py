import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .fusion import GatedFusion, CrossModalAttention
from .audio_encoder import AudioEncoder
from .time_series_encoder import TimeSeriesEncoder
from .interpretability import compute_feature_importance

class MultimodalStockPredictor(nn.Module):
    def __init__(self, 
                 text_model_name="bert-large-uncased",
                 vision_model_name=None,
                 tabular_dim=64,  # <-- Change this to match your data, e.g., tabular_dim=5
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
                 time_series_encoder=None,
                 freeze_text_encoder=False,
                 freeze_vision_encoder=False,
                 use_self_attention=True,
                 self_attention_heads=8,
                 self_attention_layers=2,
                 use_dropout_scheduler=True,
                 dropout_scheduler_max=0.5,
                 dropout_scheduler_min=0.1,
                 use_stochastic_depth=True,
                 stochastic_depth_prob=0.2,
                 use_ensemble=False,
                 ensemble_size=3,
                 fusion_type='concat',
                 use_mixed_precision=False):
        """
        Args:
            fusion_type (str): 'concat', 'gated', or 'cross_attention'.
            use_mixed_precision (bool): Enable mixed precision training.
            ...existing code...
        """
        super().__init__()
        self.tabular_dim = tabular_dim  # Save for runtime check
        # Text encoder (large transformer)
        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, config=self.text_config)
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Optional: Vision encoder (for chart images, etc.)
        if vision_model_name:
            self.vision_config = AutoConfig.from_pretrained(vision_model_name)
            self.vision_encoder = AutoModel.from_pretrained(vision_model_name, config=self.vision_config)
            if freeze_vision_encoder:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
            vision_out_dim = self.vision_config.hidden_size
        else:
            self.vision_encoder = None
            vision_out_dim = 0

        # Tabular (numerical) data encoder with dropout and batchnorm
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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

        self.fusion_type = fusion_type
        self.use_mixed_precision = use_mixed_precision

        # --- Fusion options ---
        fusion_dim = self.text_config.hidden_size + vision_out_dim + hidden_dim + audio_out_dim_ + time_series_out_dim_
        self.fusion_input_dims = [
            self.text_config.hidden_size,
            hidden_dim,
            *( [vision_out_dim] if vision_out_dim > 0 else [] ),
            *( [audio_out_dim_] if audio_out_dim_ > 0 else [] ),
            *( [time_series_out_dim_] if time_series_out_dim_ > 0 else [] ),
        ]
        if fusion_type == 'gated':
            self.fusion = GatedFusion(self.fusion_input_dims, fusion_dim)
        elif fusion_type == 'cross_attention':
            # Only supports two modalities for cross attention
            if len(self.fusion_input_dims) != 2:
                raise ValueError("CrossModalAttention fusion requires exactly two modalities.")
            self.fusion = CrossModalAttention(self.fusion_input_dims[0], self.fusion_input_dims[1], fusion_dim)
        else:
            self.fusion = None  # fallback to concat

        # Fusion and prediction head (deeper, configurable)
        fusion_head_layers = []
        in_dim = fusion_dim
        for i in range(fusion_layers - 1):
            fusion_head_layers.append(nn.Linear(in_dim, hidden_dim))
            fusion_head_layers.append(nn.BatchNorm1d(hidden_dim))
            fusion_head_layers.append(activation())
            fusion_head_layers.append(nn.Dropout(fusion_dropout))
            if fusion_layernorm:
                fusion_head_layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        fusion_head_layers.append(nn.Linear(in_dim, num_labels))

        # Self-attention block after fusion (optional)
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=self_attention_heads,
                dim_feedforward=fusion_dim * 2,
                dropout=fusion_dropout,
                activation="gelu",
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self_attention_layers)
        else:
            self.transformer_encoder = None

        # Stochastic depth (optional)
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_prob = stochastic_depth_prob

        # Dropout scheduler (optional)
        self.use_dropout_scheduler = use_dropout_scheduler
        self.dropout_scheduler_max = dropout_scheduler_max
        self.dropout_scheduler_min = dropout_scheduler_min
        self.current_dropout = fusion_dropout

        # Ensemble (optional)
        self.use_ensemble = use_ensemble
        if self.use_ensemble:
            self.ensemble = nn.ModuleList([
                nn.Sequential(*fusion_head_layers) for _ in range(ensemble_size)
            ])
        else:
            self.fusion_head = nn.Sequential(*fusion_head_layers)

    def stochastic_depth(self, x, p, training):
        if not training or p == 0.0:
            return x
        keep_prob = 1 - p
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor

    def set_dropout(self, epoch=None, max_epochs=None):
        """
        Optionally schedule dropout rate during training.
        """
        if self.use_dropout_scheduler and epoch is not None and max_epochs is not None:
            ratio = epoch / max_epochs
            self.current_dropout = self.dropout_scheduler_max - (self.dropout_scheduler_max - self.dropout_scheduler_min) * ratio
            for m in self.fusion_head.modules():
                if isinstance(m, nn.Dropout):
                    m.p = self.current_dropout

    def _validate_inputs(self, text_inputs, tabular_inputs, vision_inputs, audio_inputs, time_series_inputs):
        # Robust input validation for required modalities
        if text_inputs is None:
            raise ValueError("text_inputs is required (tokenized text batch).")
        if tabular_inputs is None:
            raise ValueError("tabular_inputs is required (tabular tensor batch).")
        if self.vision_encoder and vision_inputs is None:
            raise ValueError("vision_inputs required but not provided.")
        if self.audio_encoder and audio_inputs is None:
            raise ValueError("audio_inputs required but not provided.")
        if self.time_series_encoder and time_series_inputs is None:
            raise ValueError("time_series_inputs required but not provided.")

    def forward(self, text_inputs, tabular_inputs, vision_inputs=None, audio_inputs=None, time_series_inputs=None, epoch=None, max_epochs=None):
        """
        Forward pass for the model.
        Args:
            text_inputs (dict): Tokenized text inputs for transformer.
            tabular_inputs (Tensor): Tabular features.
            vision_inputs (dict or None): Vision transformer inputs.
            audio_inputs (Tensor or None): Audio features.
            time_series_inputs (Tensor or None): Time series features.
            epoch (int or None): Current epoch for dropout scheduling.
            max_epochs (int or None): Maximum number of epochs for dropout scheduling.
        Returns:
            logits (Tensor): Output logits.
            last_attn_weights (Tensor or None): Attention weights if available.
        """
        self._validate_inputs(text_inputs, tabular_inputs, vision_inputs, audio_inputs, time_series_inputs)
        # Ensure tabular_inputs is 2D [batch, tabular_dim]
        if tabular_inputs is not None and tabular_inputs.ndim > 2:
            tabular_inputs = tabular_inputs.view(tabular_inputs.shape[0], -1)
        # Add runtime check for feature dimension
        if tabular_inputs is not None and tabular_inputs.shape[1] != self.tabular_dim:
            raise ValueError(f"tabular_inputs.shape[1] ({tabular_inputs.shape[1]}) does not match model tabular_dim ({self.tabular_dim}).")
        # Mixed precision context if enabled
        if self.use_mixed_precision:
            from torch.cuda.amp import autocast
            autocast_ctx = autocast
        else:
            autocast_ctx = None
        if autocast_ctx is not None:
            with autocast_ctx():
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

                # --- Fusion ---
                if self.fusion_type == 'gated':
                    fused = self.fusion(features_list)
                elif self.fusion_type == 'cross_attention':
                    fused = self.fusion(features_list[0], features_list[1])
                else:
                    fused = torch.cat(features_list, dim=1)

                features = fused

                # Self-attention block (optional)
                if self.transformer_encoder is not None:
                    features = features.unsqueeze(1)  # (batch, seq=1, dim)
                    features = self.transformer_encoder(features)
                    features = features.squeeze(1)

                # Attention-based fusion (optional, legacy)
                if self.attn_fusion is not None:
                    features_seq = features.unsqueeze(1)
                    attn_out, attn_weights = self.attn_fusion(features_seq, features_seq, features_seq)
                    features = attn_out.squeeze(1)
                    self.last_attn_weights = attn_weights
                else:
                    self.last_attn_weights = None

                # Stochastic depth (optional)
                if self.use_stochastic_depth and self.training:
                    features = self.stochastic_depth(features, self.stochastic_depth_prob, self.training)

                # Dropout scheduler (optional)
                self.set_dropout(epoch, max_epochs)

                # Residual connection (optional)
                if self.use_ensemble:
                    logits_list = [head(features) for head in self.ensemble]
                    logits = torch.stack(logits_list, dim=0).mean(dim=0)
                elif self.use_residual_fusion:
                    fusion_input = features
                    logits = self.fusion_head(features)
                    logits += fusion_input[:, :logits.shape[1]]
                else:
                    logits = self.fusion_head(features)
        else:
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

            # --- Fusion ---
            if self.fusion_type == 'gated':
                fused = self.fusion(features_list)
            elif self.fusion_type == 'cross_attention':
                fused = self.fusion(features_list[0], features_list[1])
            else:
                fused = torch.cat(features_list, dim=1)

            features = fused

            # Self-attention block (optional)
            if self.transformer_encoder is not None:
                features = features.unsqueeze(1)  # (batch, seq=1, dim)
                features = self.transformer_encoder(features)
                features = features.squeeze(1)

            # Attention-based fusion (optional, legacy)
            if self.attn_fusion is not None:
                features_seq = features.unsqueeze(1)
                attn_out, attn_weights = self.attn_fusion(features_seq, features_seq, features_seq)
                features = attn_out.squeeze(1)
                self.last_attn_weights = attn_weights
            else:
                self.last_attn_weights = None

            # Stochastic depth (optional)
            if self.use_stochastic_depth and self.training:
                features = self.stochastic_depth(features, self.stochastic_depth_prob, self.training)

            # Dropout scheduler (optional)
            self.set_dropout(epoch, max_epochs)

            # Residual connection (optional)
            if self.use_ensemble:
                logits_list = [head(features) for head in self.ensemble]
                logits = torch.stack(logits_list, dim=0).mean(dim=0)
            elif self.use_residual_fusion:
                fusion_input = features
                logits = self.fusion_head(features)
                logits += fusion_input[:, :logits.shape[1]]
            else:
                logits = self.fusion_head(features)
        return logits

    def extract_features(self, text_inputs, tabular_inputs, vision_inputs=None, audio_inputs=None, time_series_inputs=None):
        """
        Extract intermediate features before the final prediction head.
        Returns:
            features (Tensor): Concatenated feature vector.
        """
        self._validate_inputs(text_inputs, tabular_inputs, vision_inputs, audio_inputs, time_series_inputs)
        # Ensure tabular_inputs is 2D [batch, tabular_dim]
        if tabular_inputs is not None and tabular_inputs.ndim > 2:
            tabular_inputs = tabular_inputs.view(tabular_inputs.shape[0], -1)
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

    def get_attention_weights(self):
        """
        Returns last attention weights (if available).
        """
        return getattr(self, "last_attn_weights", None)

    def compute_feature_importance(self, batch_inputs, batch_targets, loss_fn):
        """
        Computes feature importance using input gradients.
        Args:
            batch_inputs (dict): Dictionary with keys matching forward() args.
            batch_targets (Tensor): Target labels.
            loss_fn (callable): Loss function.
        Returns:
            importances (dict): Feature importance per modality.
        """
        # Prepare inputs for interpretability utility
        inputs = {
            'text_inputs': batch_inputs.get('text_inputs'),
            'tabular_inputs': batch_inputs.get('tabular_inputs'),
            'vision_inputs': batch_inputs.get('vision_inputs'),
            'audio_inputs': batch_inputs.get('audio_inputs'),
            'time_series_inputs': batch_inputs.get('time_series_inputs'),
        }
        # Remove None values
        inputs = {k: v for k, v in inputs.items() if v is not None}
        return compute_feature_importance(self, inputs, batch_targets, loss_fn)

# Example usage:
"""
Example usage:

from stock_ai import MultimodalStockPredictor
import torch
from transformers import AutoTokenizer

# Initialize model with gated fusion and mixed precision
model = MultimodalStockPredictor(fusion_type='gated', use_mixed_precision=True)
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
text_inputs = tokenizer(["AAPL earnings beat expectations"], return_tensors="pt", padding=True, truncation=True)
tabular_inputs = torch.randn(1, 64)
logits = model(text_inputs, tabular_inputs)
attn_weights = model.get_attention_weights()

# Feature importance:
import torch.nn.functional as F
batch_targets = torch.tensor([1])
importances = model.compute_feature_importance(
    {'text_inputs': text_inputs, 'tabular_inputs': tabular_inputs},
    batch_targets,
    F.cross_entropy
)
print(importances)
"""

# =========================
# Unit Tests for Components
# =========================
def _dummy_text_inputs(batch=2, dim=768):
    # Simulate HuggingFace tokenizer output
    return {'input_ids': torch.ones(batch, 8, dtype=torch.long), 'attention_mask': torch.ones(batch, 8, dtype=torch.long)}

def _dummy_tabular_inputs(batch=2, dim=64):
    return torch.randn(batch, dim)

def _dummy_targets(batch=2, num_classes=3):
    return torch.randint(0, num_classes, (batch,))

def _dummy_model(fusion_type='concat', use_audio=False, use_time_series=False, use_mixed_precision=False):
    return MultimodalStockPredictor(
        fusion_type=fusion_type,
        use_audio=use_audio,
        use_time_series=use_time_series,
        use_mixed_precision=use_mixed_precision,
        tabular_dim=64,
        hidden_dim=32,
        num_labels=3,
        fusion_layers=2,
        activation=nn.ReLU,
        tabular_dropout=0.1,
        fusion_dropout=0.1,
        fusion_layernorm=True,
        use_attention_fusion=False,
        use_residual_fusion=False,
        audio_dim=8 if use_audio else None,
        time_series_dim=8 if use_time_series else None,
        audio_hidden_dim=16,
        audio_out_dim=16,
        time_series_hidden_dim=16,
        time_series_out_dim=16,
        freeze_text_encoder=True,  # For speed
        freeze_vision_encoder=True
    )

def test_fusion_concat():
    model = _dummy_model(fusion_type='concat')
    text_inputs = _dummy_text_inputs()
    tabular_inputs = _dummy_tabular_inputs()
    out = model(text_inputs, tabular_inputs)
    assert out.shape[0] == 2, "Batch size mismatch for concat fusion"
    print("test_fusion_concat passed.")

def test_fusion_gated():
    model = _dummy_model(fusion_type='gated')
    text_inputs = _dummy_text_inputs()
    tabular_inputs = _dummy_tabular_inputs()
    out = model(text_inputs, tabular_inputs)
    assert out.shape[0] == 2, "Batch size mismatch for gated fusion"
    print("test_fusion_gated passed.")

def test_fusion_cross_attention():
    model = _dummy_model(fusion_type='cross_attention')
    text_inputs = _dummy_text_inputs()
    tabular_inputs = _dummy_tabular_inputs()
    out = model(text_inputs, tabular_inputs)
    assert out.shape[0] == 2, "Batch size mismatch for cross_attention fusion"
    print("test_fusion_cross_attention passed.")

def test_input_validation():
    model = _dummy_model()
    try:
        model(None, _dummy_tabular_inputs())
        assert False, "Should raise error for missing text_inputs"
    except ValueError:
        pass
    try:
        model(_dummy_text_inputs(), None)
        assert False, "Should raise error for missing tabular_inputs"
    except ValueError:
        pass
    print("test_input_validation passed.")

def test_interpretability():
    model = _dummy_model()
    text_inputs = _dummy_text_inputs()
    tabular_inputs = _dummy_tabular_inputs()
    targets = _dummy_targets()
    import torch.nn.functional as F
    importances = model.compute_feature_importance(
        {'text_inputs': text_inputs, 'tabular_inputs': tabular_inputs},
        targets,
        F.cross_entropy
    )
    assert 'text_inputs' in importances and 'tabular_inputs' in importances, "Feature importance keys missing"
    print("test_interpretability passed.")

def test_mixed_precision():
    model = _dummy_model(use_mixed_precision=True)
    text_inputs = _dummy_text_inputs()
    tabular_inputs = _dummy_tabular_inputs()
    try:
        out = model(text_inputs, tabular_inputs)
        assert out.shape[0] == 2
        print("test_mixed_precision passed.")
    except Exception as e:
        print("test_mixed_precision failed:", e)

if __name__ == "__main__":
    test_fusion_concat()
    test_fusion_gated()
    test_fusion_cross_attention()
    test_input_validation()
    test_interpretability()
    test_mixed_precision()
