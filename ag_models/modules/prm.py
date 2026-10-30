"""
Unified Multi-Scale Progressive Refinement Module (PRM & WaveAttn-PRM) for 2D and 3D.

Standardizes and unifies the implementations from:
- `liverseg/models/sopseg_liver.py` (`ProgressiveRefinementModule3D`)
- `liverseg/models/waveattn_sopseg_liver.py` (`WaveAttnProgressiveRefinementModule3D`)
- `cangio/src/sopseg_enhanced.py` (`EnhancedProgressiveRefinementModule`)
"""

from typing import Dict, List

import torch
import torch.nn as nn

from ag_models.core.layers import get_activation, get_conv, get_conv_transpose, get_norm, interpolate_nd
from ag_models.core.registry import BLOCK_REGISTRY
from ag_models.modules.attention import GLCAMContextAttention
from ag_models.modules.blocks import ResidualRefinementBlock
from ag_models.modules.wavelet import WaveFusionBlock


@BLOCK_REGISTRY.register("progressive_refinement_module", aliases=["prm", "waveattn_prm"])
class ProgressiveRefinementModule(nn.Module):
    """
    Unified 2D / 3D Multi-Scale Progressive Refinement Module (PRM).
    Progressively sharpens coarse mask & edge predictions `[M, E]` across 3 spatial scales
    (1/4 -> 1/2 -> 1/1), fusing deep bottleneck features, skip connections, image guidance,
    and intermediate predictions.

    Configurable with:
    - `use_wavelet=True`: Uses `WaveFusionBlock` (DWT/IDWT edge-semantic separation).
    - `use_wavelet=False`: Uses standard `ResidualRefinementBlock` (RRB).
    - `use_glcam=True`: Applies `GLCAMContextAttention` on deep and skip features.
    """

    def __init__(
        self,
        deep_dim: int = 192,
        skip_dims: List[int] = [96, 48, 24],
        feat_dim: int = 48,
        out_channels: int = 2,
        guidance_channels: int = 1,
        use_wavelet: bool = True,
        use_glcam: bool = True,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_channels = out_channels
        self.use_wavelet = use_wavelet
        self.use_glcam = use_glcam
        self.dims = dims

        conv_cls = get_conv(dims)
        conv_t = get_conv_transpose(dims)

        if use_glcam:
            self.glcam_deep = GLCAMContextAttention(deep_dim, dims=dims, act_type=act_type)
            self.glcam_s3 = GLCAMContextAttention(skip_dims[0], dims=dims, act_type=act_type)
            self.glcam_s2 = GLCAMContextAttention(skip_dims[1], dims=dims, act_type=act_type)
            self.glcam_s1 = GLCAMContextAttention(skip_dims[2], dims=dims, act_type=act_type)
        else:
            self.glcam_deep = nn.Identity()
            self.glcam_s3 = nn.Identity()
            self.glcam_s2 = nn.Identity()
            self.glcam_s1 = nn.Identity()

        # Stage 1 (Scale 1/4)
        self.up_deep = nn.Sequential(
            conv_t(deep_dim, feat_dim, kernel_size=2, stride=2),
            get_norm(norm_type, feat_dim, dims=dims),
            get_activation(act_type),
        )
        in_ch1 = feat_dim + skip_dims[0] + guidance_channels + out_channels
        if use_wavelet:
            self.fuse1 = nn.Sequential(
                conv_cls(in_ch1, feat_dim, kernel_size=1, bias=False),
                get_norm(norm_type, feat_dim, dims=dims),
                get_activation(act_type),
            )
            self.refine1 = WaveFusionBlock(feat_dim, feat_dim, dims=dims, norm_type=norm_type, act_type=act_type)
        else:
            self.fuse1 = nn.Identity()
            self.refine1 = ResidualRefinementBlock(in_ch1, feat_dim, dims=dims, norm_type=norm_type, act_type=act_type)
        self.phi1 = conv_cls(feat_dim, out_channels, kernel_size=1)

        # Stage 2 (Scale 1/2)
        self.up1 = nn.Sequential(
            conv_t(feat_dim, feat_dim, kernel_size=2, stride=2),
            get_norm(norm_type, feat_dim, dims=dims),
            get_activation(act_type),
        )
        in_ch2 = feat_dim + skip_dims[1] + guidance_channels + out_channels
        if use_wavelet:
            self.fuse2 = nn.Sequential(
                conv_cls(in_ch2, feat_dim, kernel_size=1, bias=False),
                get_norm(norm_type, feat_dim, dims=dims),
                get_activation(act_type),
            )
            self.refine2 = WaveFusionBlock(feat_dim, feat_dim, dims=dims, norm_type=norm_type, act_type=act_type)
        else:
            self.fuse2 = nn.Identity()
            self.refine2 = ResidualRefinementBlock(in_ch2, feat_dim, dims=dims, norm_type=norm_type, act_type=act_type)
        self.phi2 = conv_cls(feat_dim, out_channels, kernel_size=1)

        # Stage 3 (Scale 1/1)
        self.up2 = nn.Sequential(
            conv_t(feat_dim, feat_dim // 2, kernel_size=2, stride=2),
            get_norm(norm_type, feat_dim // 2, dims=dims),
            get_activation(act_type),
        )
        in_ch3 = (feat_dim // 2) + skip_dims[2] + guidance_channels + out_channels
        if use_wavelet:
            self.fuse3 = nn.Sequential(
                conv_cls(in_ch3, feat_dim // 2, kernel_size=1, bias=False),
                get_norm(norm_type, feat_dim // 2, dims=dims),
                get_activation(act_type),
            )
            self.refine3 = WaveFusionBlock(feat_dim // 2, feat_dim // 2, dims=dims, norm_type=norm_type, act_type=act_type)
        else:
            self.fuse3 = nn.Identity()
            self.refine3 = ResidualRefinementBlock(in_ch3, feat_dim // 2, dims=dims, norm_type=norm_type, act_type=act_type)
        self.phi3 = conv_cls(feat_dim // 2, out_channels, kernel_size=1)

    def forward(
        self,
        deep_feats: torch.Tensor,
        skips: List[torch.Tensor],
        image_vol: torch.Tensor,
        init_preds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        s3, s2, s1 = skips

        deep_att = self.glcam_deep(deep_feats)
        s3_att = self.glcam_s3(s3)
        s2_att = self.glcam_s2(s2)
        s1_att = self.glcam_s1(s1)

        # Stage 1 (Scale 1/4, e.g., 16^3)
        x1_in = self.up_deep(deep_att)
        size_s3 = s3_att.shape[2:]
        i_s3 = interpolate_nd(image_vol, size=size_s3, dims=self.dims)
        p_s3 = interpolate_nd(init_preds, size=size_s3, dims=self.dims)
        cat1 = torch.cat([x1_in, s3_att, i_s3, p_s3], dim=1)
        x1 = self.refine1(self.fuse1(cat1))
        p1 = self.phi1(x1)

        # Stage 2 (Scale 1/2, e.g., 32^3)
        x2_in = self.up1(x1)
        size_s2 = s2_att.shape[2:]
        i_s2 = interpolate_nd(image_vol, size=size_s2, dims=self.dims)
        p_s2 = interpolate_nd(p1, size=size_s2, dims=self.dims)
        cat2 = torch.cat([x2_in, s2_att, i_s2, p_s2], dim=1)
        x2 = self.refine2(self.fuse2(cat2))
        p2 = self.phi2(x2)

        # Stage 3 (Scale 1/1, e.g., 64^3)
        x3_in = self.up2(x2)
        size_s1 = s1_att.shape[2:]
        i_s1 = interpolate_nd(image_vol, size=size_s1, dims=self.dims) if image_vol.shape[2:] != size_s1 else image_vol
        p_s1 = interpolate_nd(p2, size=size_s1, dims=self.dims)
        cat3 = torch.cat([x3_in, s1_att, i_s1, p_s1], dim=1)
        x3 = self.refine3(self.fuse3(cat3))
        p3 = self.phi3(x3)

        return {
            "p16": p1,
            "p32": p2,
            "p64": p3,
            "p_quarter": p1,
            "p_half": p2,
            "p_full": p3,
            "last_feat": x3,
        }


# Backward-compatible 3D aliases
class WaveAttnProgressiveRefinementModule3D(ProgressiveRefinementModule):
    def __init__(
        self,
        deep_dim: int = 192,
        skip_dims: List[int] = [96, 48, 24],
        feat_dim: int = 48,
        out_channels: int = 2,
    ):
        super().__init__(
            deep_dim=deep_dim,
            skip_dims=skip_dims,
            feat_dim=feat_dim,
            out_channels=out_channels,
            use_wavelet=True,
            use_glcam=True,
            dims=3,
        )


class ProgressiveRefinementModule3D(ProgressiveRefinementModule):
    def __init__(
        self,
        deep_dim: int = 192,
        skip_dims: List[int] = [96, 48, 24],
        feat_dim: int = 48,
        out_channels: int = 2,
    ):
        super().__init__(
            deep_dim=deep_dim,
            skip_dims=skip_dims,
            feat_dim=feat_dim,
            out_channels=out_channels,
            use_wavelet=False,
            use_glcam=False,
            dims=3,
        )
