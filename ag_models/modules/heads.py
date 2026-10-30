"""
Standardized Task Prediction Heads for Segmentation, Multi-Label Classification,
Masked Feature Pooling (MFP), and Quantitative Regression.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ag_models.core.layers import (
    get_activation,
    get_adaptive_avg_pool,
    get_adaptive_max_pool,
    get_conv,
    interpolate_nd,
)
from ag_models.core.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("masked_feature_pooling_head", aliases=["mfp_head"])
class MaskedFeaturePoolingHead(nn.Module):
    """
    Segmentation-Guided Masked Feature Pooling (MFP) Classification Head (from `WaveCo_MTLNet`).
    Combines Global Average Pooling, Global Max Pooling, and Lesion/Foreground-Masked Spatial Pooling
    to classify subtle focal pathologies without background dilution.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 5,
        dims: int = 3,
        dropout: float = 0.3,
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.dims = dims
        self.global_pool_avg = get_adaptive_avg_pool(dims, 1)
        self.global_pool_max = get_adaptive_max_pool(dims, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.BatchNorm1d(in_channels),
            get_activation(act_type),
            nn.Dropout(dropout),
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            get_activation(act_type),
            nn.Dropout(max(0.1, dropout - 0.1)),
            nn.Linear(in_channels // 2, num_classes),
        )

    def forward(self, bottleneck_feat: torch.Tensor, fg_prob: torch.Tensor) -> torch.Tensor:
        if fg_prob.shape[2:] != bottleneck_feat.shape[2:]:
            fg_prob_low = interpolate_nd(fg_prob, size=bottleneck_feat.shape[2:], dims=self.dims)
        else:
            fg_prob_low = fg_prob

        f_avg = self.global_pool_avg(bottleneck_feat).flatten(1)
        f_max = self.global_pool_max(bottleneck_feat).flatten(1)

        spatial_dims = tuple(range(2, bottleneck_feat.ndim))
        mask_weights = fg_prob_low / (fg_prob_low.sum(dim=spatial_dims, keepdim=True) + 1e-5)
        f_masked = (bottleneck_feat * mask_weights).sum(dim=spatial_dims)

        f_in = torch.cat([f_avg, f_max, f_masked], dim=1)
        return self.mlp(f_in)


@HEAD_REGISTRY.register("multitask_global_head")
class MultiTaskGlobalHead(nn.Module):
    """
    Global Multi-Task Classification & Quantitative Burden Regression Head.
    """

    def __init__(
        self,
        in_channels: int,
        num_cls_classes: int = 5,
        hidden_dim: int = 128,
        enable_burden: bool = True,
        dims: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.enable_burden = enable_burden
        self.pool = get_adaptive_avg_pool(dims, 1)
        self.cls_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cls_classes),
        )
        if enable_burden:
            self.burden_head = nn.Sequential(
                nn.Linear(in_channels, hidden_dim // 2),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.burden_head = None

    def forward(self, z: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        z_pool = self.pool(z).flatten(1) if z.ndim > 2 else z
        cls_logits = self.cls_head(z_pool)
        burden = self.burden_head(z_pool) if self.burden_head is not None else None
        return {"cls_logits": cls_logits, "tumor_burden": burden}


@HEAD_REGISTRY.register("segmentation_head")
class SegmentationHead(nn.Module):
    """
    Standardized N-D Segmentation Output Head with optional intermediate refinement conv
    and spatial upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dims: int = 3,
        upsample_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.upsample_scale = upsample_scale
        conv_cls = get_conv(dims)
        if mid_channels is not None:
            self.block = nn.Sequential(
                conv_cls(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01, inplace=True),
                conv_cls(mid_channels, out_channels, kernel_size=1),
            )
        else:
            self.block = conv_cls(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.upsample_scale > 1:
            out = interpolate_nd(out, scale_factor=float(self.upsample_scale), dims=self.dims)
        return out
