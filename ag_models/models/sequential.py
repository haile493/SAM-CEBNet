"""
Standardized Sequential, Tabular, and Bioinformatics Models (`mirnomes`).

Includes:
1. `HighwayBiLSTM` (`highway_bilstm`): Dual-stream Sequential BiLSTM + Attention + Tabular Highway Network
   for pan-cancer and tissue-of-origin (TOO) classification.
2. `DANNModel` (`dann`): Domain-Adversarial Neural Network with `GradientReversal` for cohort batch alignment.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ag_models.core.base import BaseModel, ModelConfig, ModelOutput
from ag_models.core.registry import MODEL_REGISTRY
from ag_models.modules.adapters import GradientReversal


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Sequential:
    """Standardized Multi-Layer Perceptron builder."""
    layers = []
    curr_dim = in_dim
    for i in range(max(1, num_layers - 1)):
        layers.extend([
            nn.Linear(curr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        curr_dim = hidden_dim
    layers.append(nn.Linear(curr_dim, out_dim))
    return nn.Sequential(*layers)


@MODEL_REGISTRY.register("highway_bilstm", aliases=["bilstm_highway"])
class HighwayBiLSTM(BaseModel):
    """
    Dual-Branch Highway-BiLSTM with Temporal Token Attention (`mirnomes/unified_hierarchical_bilstm.py`).
    Supports both:
    - Dual inputs `(x_seq, x_flat)` or
    - Single flat input `x_flat` (automatically reshaped into `(B, flat_dim // step_dim, step_dim)`).
    """

    def __init__(
        self,
        step_dim: int = 5,
        flat_dim: int = 150,
        hidden_dim: int = 128,
        num_classes: int = 13,
        dropout: float = 0.25,
        return_tuple: bool = True,
    ):
        super().__init__(
            ModelConfig(
                arch="highway_bilstm",
                dims=1,
                in_channels=flat_dim,
                num_cls_classes=num_classes,
                dropout=dropout,
                extra={"step_dim": step_dim, "flat_dim": flat_dim, "hidden_dim": hidden_dim},
            )
        )
        self.step_dim = step_dim
        self.flat_dim = flat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.return_tuple = return_tuple

        self.proj = nn.Sequential(
            nn.Linear(step_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.highway = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_flat: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], ModelOutput]:
        if x_flat is None:
            # Automatically derive x_seq and x_flat if a single 2D tensor is passed
            if x_seq.dim() == 2:
                x_flat = x_seq
                seq_len = x_flat.shape[1] // self.step_dim
                x_seq = x_flat[:, : seq_len * self.step_dim].view(x_flat.shape[0], seq_len, self.step_dim)
            else:
                x_flat = x_seq.flatten(1)

        h_seq = self.proj(x_seq)
        out, _ = self.lstm(h_seq)
        attn_w = torch.softmax(self.attn(out), dim=1)
        c_lstm = torch.sum(attn_w * out, dim=1)
        c_hw = self.highway(x_flat)
        fused = torch.cat([c_lstm, c_hw], dim=1)
        logits = self.head(fused)
        attn_weights = attn_w.squeeze(-1)

        if self.return_tuple:
            return logits, attn_weights
        return ModelOutput(cls_logits=logits, attn_weights=attn_weights, latent_feat=fused)


@MODEL_REGISTRY.register("dann", aliases=["domain_adversarial_mlp"])
class DANNModel(BaseModel):
    """
    Standardized Domain-Adversarial Neural Network (DANN) with Dual-Domain Encoders
    and `GradientReversal` (`mirnomes/head_model/preprocess/app/util/dann.py`).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        discriminator_loss_weight: float = 1.0,
        scale: float = 0.1,
    ):
        super().__init__(
            ModelConfig(
                arch="dann",
                dims=1,
                in_channels=in_dim,
                num_cls_classes=num_classes,
                extra={
                    "hidden_dim": hidden_dim,
                    "latent_dim": latent_dim,
                    "num_layers": num_layers,
                    "discriminator_loss_weight": discriminator_loss_weight,
                    "scale": scale,
                },
            )
        )
        self.encoder1 = build_mlp(in_dim, hidden_dim, latent_dim, num_layers=num_layers)
        self.encoder2 = build_mlp(in_dim, hidden_dim, latent_dim, num_layers=num_layers)
        self.predictor = build_mlp(latent_dim, hidden_dim, num_classes, num_layers=num_layers)
        self.discriminator = build_mlp(latent_dim, hidden_dim, 1, num_layers=num_layers)
        self.g_reversal = GradientReversal(scale=scale)
        self.discriminator_loss_weight = discriminator_loss_weight

    def encode(self, x: torch.Tensor, domain_z: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(x)
        z_col = domain_z.float().view(-1, 1)
        return z_col * e1 + (1.0 - z_col) * e2

    def forward(
        self,
        x: torch.Tensor,
        domain_z: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        if domain_z is None:
            domain_z = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        e = self.encode(x, domain_z)
        cls_logits = self.predictor(e)
        dom_logits = self.discriminator(self.g_reversal(e)).squeeze(-1)

        loss = None
        if y is not None:
            loss_sup = nn.CrossEntropyLoss()(cls_logits, y)
            loss_unsup = nn.BCEWithLogitsLoss()(dom_logits, domain_z.float())
            loss = loss_sup + self.discriminator_loss_weight * loss_unsup

        return ModelOutput(
            cls_logits=cls_logits,
            domain_logits=dom_logits,
            latent_feat=e,
            loss=loss,
        )
