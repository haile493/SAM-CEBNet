"""
Standardized Bi-Temporal and Spatiotemporal Sequence Architectures (`cangio`).

Includes:
1. `SiameseUNet` (`siamese_unet`): Weight-shared bi-temporal ResNet encoder + multi-scale `SiameseFeatureFusionBlock`.
2. `EarlyFusionBiTemporalUNet` (`early_fusion_unet`): Channel-concatenated bi-temporal U-Net.
3. `TemporalConvLSTMUNet` (`temporal_convlstm_unet`): Multi-scale `ConvLSTMCell` spatiotemporal U-Net.
4. `TemporalAttentionUNet` (`temporal_attention_unet`): Multi-scale pixel-wise Temporal Self-Attention U-Net.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ag_models.core.base import BaseModel, ModelConfig
from ag_models.core.registry import BLOCK_REGISTRY, MODEL_REGISTRY
from ag_models.modules.adapters import AutoPadWrapper
from ag_models.modules.blocks import ConvLSTMCell, SiameseFeatureFusionBlock


@BLOCK_REGISTRY.register("temporal_self_attention")
class TemporalSelfAttentionModule(nn.Module):
    """Pixel-wise Temporal Self-Attention across time steps `T` (`(B, T, C, H, W) -> (B, C, H, W)`)."""

    def __init__(self, channels: int):
        super().__init__()
        k_dim = max(16, channels // 4)
        self.q = nn.Conv2d(channels, k_dim, kernel_size=1)
        self.k = nn.Conv2d(channels, k_dim, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_seq: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = feat_seq.shape
        x_flat = feat_seq.view(b * t, c, h, w)
        q = self.q(x_flat).view(b, t, -1, h * w).permute(0, 3, 1, 2)
        k = self.k(x_flat).view(b, t, -1, h * w).permute(0, 3, 2, 1)
        v = self.v(x_flat).view(b, t, c, h * w).permute(0, 3, 1, 2)

        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.softmax(torch.matmul(q, k) * scale, dim=-1)
        attended = torch.matmul(attn, v)

        aggregated = attended.mean(dim=2).permute(0, 2, 1).view(b, c, h, w)
        return feat_seq[:, -1] + self.gamma * aggregated


@MODEL_REGISTRY.register("siamese_unet")
class SiameseUNet(BaseModel):
    """
    Siamese 2D U-Net for Bi-Temporal Change Detection (`(x1, x2) -> logits`).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__(
            ModelConfig(
                arch="siamese_unet",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=classes,
                pretrained=(encoder_weights is not None),
                extra={"encoder_name": encoder_name},
            )
        )
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.fusions = nn.ModuleList([SiameseFeatureFusionBlock(ch, dims=2) for ch in enc_channels])
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_channels, decoder_channels=decoder_channels, n_blocks=5
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        feats1 = self.encoder(x1)
        feats2 = self.encoder(x2)
        fused = [fusion(f1, f2) for f1, f2, fusion in zip(feats1, feats2, self.fusions)]
        return self.segmentation_head(self.decoder(fused))


@MODEL_REGISTRY.register("early_fusion_unet")
class EarlyFusionBiTemporalUNet(BaseModel):
    """Early-Fusion Bi-Temporal U-Net concatenating pre/post along channel axis."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels_per_image: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
    ):
        super().__init__(
            ModelConfig(
                arch="early_fusion_unet",
                dims=2,
                in_channels=in_channels_per_image * 2,
                num_seg_classes=classes,
                pretrained=(encoder_weights is not None),
                extra={"encoder_name": encoder_name},
            )
        )
        import segmentation_models_pytorch as smp

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels_per_image * 2,
            classes=classes,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x1, x2], dim=1))


@MODEL_REGISTRY.register("temporal_convlstm_unet", aliases=["convlstm_unet"])
class TemporalConvLSTMUNet(BaseModel):
    """Multi-Scale Spatiotemporal ConvLSTM U-Net (`(B, T, C, H, W) -> (B, classes, H, W)`)."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__(
            ModelConfig(
                arch="temporal_convlstm_unet",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=classes,
                pretrained=(encoder_weights is not None),
                extra={"encoder_name": encoder_name},
            )
        )
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.temporal_cells = nn.ModuleList([ConvLSTMCell(ch, ch, kernel_size=3, dims=2) for ch in enc_channels])
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_channels, decoder_channels=decoder_channels, n_blocks=5
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_seq.shape
        all_feats = self.encoder(x_seq.view(b * t, c, h, w))
        fused_feats = []
        for feat, cell in zip(all_feats, self.temporal_cells):
            _, ch, hi, wi = feat.shape
            feat_seq = feat.view(b, t, ch, hi, wi)
            state = None
            for step in range(t):
                h_t, state = cell(feat_seq[:, step], state)
            fused_feats.append(h_t)
        return self.segmentation_head(self.decoder(fused_feats))


TemporalConvLSTM_UNet = TemporalConvLSTMUNet


@MODEL_REGISTRY.register("temporal_attention_unet")
class TemporalAttentionUNet(BaseModel):
    """Multi-Scale Pixel-Wise Temporal Self-Attention U-Net (`(B, T, C, H, W) -> (B, classes, H, W)`)."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__(
            ModelConfig(
                arch="temporal_attention_unet",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=classes,
                pretrained=(encoder_weights is not None),
                extra={"encoder_name": encoder_name},
            )
        )
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.temporal_attns = nn.ModuleList([TemporalSelfAttentionModule(ch) for ch in enc_channels])
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_channels, decoder_channels=decoder_channels, n_blocks=5
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_seq.shape
        all_feats = self.encoder(x_seq.view(b * t, c, h, w))
        fused_feats = [
            attn(feat.view(b, t, feat.shape[1], feat.shape[2], feat.shape[3]))
            for feat, attn in zip(all_feats, self.temporal_attns)
        ]
        return self.segmentation_head(self.decoder(fused_feats))


def build_bitemporal_model(
    model_type: str = "siamese",
    encoder_name: str = "resnet34",
    in_channels: int = 9,
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    if model_type == "siamese":
        base = SiameseUNet(encoder_name=encoder_name, in_channels=in_channels, encoder_weights=encoder_weights)
    elif model_type in ("early_fusion", "concat"):
        base = EarlyFusionBiTemporalUNet(encoder_name=encoder_name, in_channels_per_image=in_channels, encoder_weights=encoder_weights)
    else:
        raise ValueError(f"Unknown bi-temporal model_type: {model_type}")
    return AutoPadWrapper(base, multiple=32, dims=2)


def build_sequential_model(
    model_type: str = "convlstm",
    encoder_name: str = "resnet34",
    in_channels: int = 9,
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    if model_type == "convlstm":
        base = TemporalConvLSTMUNet(encoder_name=encoder_name, in_channels=in_channels, encoder_weights=encoder_weights)
    elif model_type == "attention":
        base = TemporalAttentionUNet(encoder_name=encoder_name, in_channels=in_channels, encoder_weights=encoder_weights)
    else:
        raise ValueError(f"Unknown sequential model_type: {model_type}")
    return AutoPadWrapper(base, multiple=32, dims=2)
