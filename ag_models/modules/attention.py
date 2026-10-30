"""
Unified 2D & 3D Attention, Context Gating, Second-Order Covariance Pooling, and Feature Fusion Modules.

Standardizes and unifies the implementations from:
- `liverseg/models/weedsense_liver.py` & `liverseg/weedsense_impl/models/blocks.py` (`SqueezeExcite`)
- `cangio/src/sopseg_enhanced.py` & `liverseg/models/waveattn_sopseg_liver.py` (`GLCAMContextAttention`)
- `liverseg/models/multitask_msff_net.py` (`ECModule`, `ECAddBlock`, `DFEAttention`)
- `liverseg/models/waveco_mtl_net.py` & `weakmedsam_coseg_waveattn.py` (`SOPSegTextureBlock`)
- `liverseg/models/resvnet_fusion.py` (`InterNetworkChannelFeatureFusion`)
"""

import math
import torch
import torch.nn as nn

from ag_models.core.layers import (
    get_activation,
    get_adaptive_avg_pool,
    get_adaptive_max_pool,
    get_conv,
    get_norm,
    interpolate_nd,
    make_divisible,
)
from ag_models.core.registry import ATTENTION_REGISTRY, BLOCK_REGISTRY


@ATTENTION_REGISTRY.register("squeeze_excite", aliases=["se", "se_block"])
class SqueezeExcite(nn.Module):
    """
    Dimension-agnostic (1D, 2D, or 3D) Squeeze-and-Excitation Channel Attention Block.
    """

    def __init__(
        self,
        in_channels: int,
        rd_ratio: float = 0.25,
        dims: int = 3,
        act_type: str = "leakyrelu",
        divisor: int = 4,
    ):
        super().__init__()
        self.dims = dims
        rd_channels = make_divisible(in_channels * rd_ratio, divisor=divisor)
        conv_cls = get_conv(dims)
        self.pool = get_adaptive_avg_pool(dims, 1)
        self.conv_reduce = conv_cls(in_channels, rd_channels, kernel_size=1, bias=True)
        self.act = get_activation(act_type)
        self.conv_expand = conv_cls(rd_channels, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.conv_reduce(w))
        w = self.gate(self.conv_expand(w))
        return x * w


@ATTENTION_REGISTRY.register("glcam", aliases=["glcam_context_attention"])
class GLCAMContextAttention(nn.Module):
    """
    Unified 2D / 3D Global-Local Contextual Attention Module (GLCAM).
    Combines dual-pooled channel attention (AvgPool + MaxPool + shared MLP)
    with spatial attention gating and a residual shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
        dims: int = 3,
        spatial_kernel: int = 3,
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.dims = dims
        red_dim = max(1, in_channels // reduction)
        conv_cls = get_conv(dims)

        self.avg_pool = get_adaptive_avg_pool(dims, 1)
        self.max_pool = get_adaptive_max_pool(dims, 1)

        self.channel_mlp = nn.Sequential(
            conv_cls(in_channels, red_dim, kernel_size=1, bias=False),
            get_activation(act_type),
            conv_cls(red_dim, in_channels, kernel_size=1, bias=False),
        )

        padding = spatial_kernel // 2
        self.spatial_conv = nn.Sequential(
            conv_cls(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Channel attention
        ca = torch.sigmoid(self.channel_mlp(self.avg_pool(x)) + self.channel_mlp(self.max_pool(x)))
        x_ca = x * ca

        # 2. Spatial attention
        avg_sp = torch.mean(x_ca, dim=1, keepdim=True)
        max_sp, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = self.spatial_conv(torch.cat([avg_sp, max_sp], dim=1))
        return x_ca * sa + x


@ATTENTION_REGISTRY.register("ec_module", aliases=["eca"])
class ECModule(nn.Module):
    """
    Efficient Channel (EC) Attention Module (from EDNet / MultiTaskLiverNet_MSFF).
    Computes adaptive 1D channel convolution weights without dimensionality reduction.
    """

    def __init__(self, channels: int, r: int = 2, b: int = 1, dims: int = 3):
        super().__init__()
        self.dims = dims
        t = int(abs((math.log2(max(channels, 2)) + b) / r))
        k = t if t % 2 == 1 else t + 1
        k = max(k, 3)

        self.avg_pool = get_adaptive_avg_pool(dims, 1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        z = self.avg_pool(x).view(b, 1, c)
        w = self.sigmoid(self.conv1d(z))
        view_shape = (b, c) + (1,) * self.dims
        return x * w.view(*view_shape)


@BLOCK_REGISTRY.register("ec_add_block")
class ECAddBlock(nn.Module):
    """
    Residual Block with Automated Feature Fusion (`ECAdd`) across shallow and deep convs.
    Supports both 2D and 3D inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        conv_cls = get_conv(dims)
        self.conv1 = nn.Sequential(
            conv_cls(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
            get_activation(act_type),
        )
        self.conv2 = nn.Sequential(
            conv_cls(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
            get_activation(act_type),
        )
        self.ec1 = ECModule(out_channels, dims=dims)
        self.ec2 = ECModule(out_channels, dims=dims)

        self.shortcut = (
            nn.Sequential(
                conv_cls(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm(norm_type, out_channels, dims=dims),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = get_activation(act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x_fused = self.ec1(x1) + self.ec2(x2)
        return self.act(x_fused + res)


@ATTENTION_REGISTRY.register("dfe_attention")
class DFEAttention(nn.Module):
    """
    Deep Feature Enhancement (DFE) Attention Module for Multi-Scale Skip Connections.
    Aligns deep feature map to shallow spatial resolution, fuses via 1x1 conv,
    and gates with spatial-channel attention.
    """

    def __init__(
        self,
        in_channels_shallow: int,
        in_channels_deep: int,
        out_channels: int,
        dims: int = 3,
        norm_type: str = "instancenorm",
    ):
        super().__init__()
        self.dims = dims
        conv_cls = get_conv(dims)
        total_ch = in_channels_shallow + in_channels_deep

        self.conv_fuse = nn.Sequential(
            conv_cls(total_ch, out_channels, kernel_size=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
            nn.ReLU(inplace=True),
        )
        self.attn_conv = nn.Sequential(
            conv_cls(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
            nn.ReLU(inplace=True),
            conv_cls(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_shallow: torch.Tensor, x_deep: torch.Tensor) -> torch.Tensor:
        if x_deep.shape[2:] != x_shallow.shape[2:]:
            x_deep = interpolate_nd(x_deep, size=x_shallow.shape[2:], dims=self.dims)
        f = self.conv_fuse(torch.cat([x_shallow, x_deep], dim=1))
        return f * self.attn_conv(f)


@ATTENTION_REGISTRY.register("sopseg_texture", aliases=["second_order_pooling"])
class SOPSegTextureBlock(nn.Module):
    """
    Unified 2D / 3D Second-Order Bilinear Covariance Pooling (SOPSeg Texture Block).
    Captures non-linear feature channel covariance to discriminate fine-grained
    lesion / canopy textures from background parenchyma.
    """

    def __init__(
        self,
        in_c: int,
        reduced_c: int = 48,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.dims = dims
        conv_cls = get_conv(dims)
        self.conv_reduce = nn.Sequential(
            conv_cls(in_c, reduced_c, kernel_size=1, bias=False),
            get_norm(norm_type, reduced_c, dims=dims),
            get_activation(act_type),
        )
        self.fc = nn.Sequential(
            nn.Linear(reduced_c, in_c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        feat = self.conv_reduce(x)
        r = feat.shape[1]
        flat = feat.view(b, r, -1)
        num_elements = max(1, flat.shape[-1])
        cov = torch.bmm(flat, flat.transpose(1, 2)) / float(num_elements)
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        view_shape = (b, c) + (1,) * self.dims
        weights = self.fc(diag).view(*view_shape)
        return x * (1.0 + weights)


@ATTENTION_REGISTRY.register("icff", aliases=["inter_network_channel_fusion"])
class InterNetworkChannelFeatureFusion(nn.Module):
    """
    Inter-Network Channel Feature Fusion (ICFF) between Segmentation and Classification streams.
    Applies SE recalibration on segmentation features, aligns spatial dimensions, and fuses.
    """

    def __init__(
        self,
        seg_channels: int,
        cls_channels: int,
        out_channels: int,
        dims: int = 3,
        norm_type: str = "instancenorm",
    ):
        super().__init__()
        self.dims = dims
        conv_cls = get_conv(dims)
        self.se = SqueezeExcite(seg_channels, rd_ratio=1.0 / 16.0, dims=dims, act_type="relu")
        self.fuse_conv = nn.Sequential(
            conv_cls(seg_channels + cls_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_seg: torch.Tensor, f_cls: torch.Tensor) -> torch.Tensor:
        f_seg_se = self.se(f_seg)
        if f_seg_se.shape[2:] != f_cls.shape[2:]:
            f_seg_se = interpolate_nd(f_seg_se, size=f_cls.shape[2:], dims=self.dims)
        return self.fuse_conv(torch.cat([f_seg_se, f_cls], dim=1))


# ---------------------------------------------------------------------------
# Explicit 2D & 3D Convenience Subclasses (Backward-Compatible Drop-ins)
# ---------------------------------------------------------------------------

class SqueezeExcite2D(SqueezeExcite):
    def __init__(self, in_channels: int, rd_ratio: float = 0.25):
        super().__init__(in_channels, rd_ratio=rd_ratio, dims=2, act_type="relu", divisor=8)


class SqueezeExcite3D(SqueezeExcite):
    def __init__(self, in_channels: int, rd_ratio: float = 0.25):
        super().__init__(in_channels, rd_ratio=rd_ratio, dims=3, act_type="leakyrelu", divisor=4)


SEBlock3D = SqueezeExcite3D


class GLCAMContextAttention2D(GLCAMContextAttention):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__(in_channels, reduction=reduction, dims=2, spatial_kernel=7, act_type="relu")


class GLCAMContextAttention3D(GLCAMContextAttention):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__(in_channels, reduction=reduction, dims=3, spatial_kernel=3, act_type="leakyrelu")


class SOPSegTextureBlock2D(SOPSegTextureBlock):
    def __init__(self, in_c: int, reduced_c: int = 64):
        super().__init__(in_c, reduced_c=reduced_c, dims=2, norm_type="batchnorm", act_type="relu")


class SOPSegTextureBlock3D(SOPSegTextureBlock):
    def __init__(self, in_c: int, reduced_c: int = 48):
        super().__init__(in_c, reduced_c=reduced_c, dims=3, norm_type="instancenorm", act_type="leakyrelu")
