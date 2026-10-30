"""
Unified 2D & 3D Convolutional, Residual, Inverted Bottleneck (UIB), Bilateral Aggregation (BGA),
and Spatiotemporal Building Blocks.

Standardizes and unifies the implementations from:
- `liverseg/models/coseg_liver.py` & `waveco_mtl_net.py` (`ConvBlock3D`, `ResBlock3D`)
- `liverseg/models/weedsense_liver.py` & `weedsense_impl/models/blocks.py` (`UIBBlock`, `LayerScale`, `BGALayer`, `StemBlock`)
- `liverseg/models/sopseg_liver.py` (`ResidualRefinementBlock`)
- `cangio/src/temporal_models.py` (`ConvLSTMCell`, `SiameseFeatureFusionBlock`)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ag_models.core.layers import (
    get_activation,
    get_adaptive_avg_pool,
    get_avg_pool,
    get_conv,
    get_max_pool,
    get_norm,
    interpolate_nd,
    make_divisible,
)
from ag_models.core.registry import BLOCK_REGISTRY
from ag_models.modules.attention import SqueezeExcite


@BLOCK_REGISTRY.register("conv_norm_act", aliases=["conv_module"])
class ConvNormAct(nn.Sequential):
    """
    Standardized N-D Convolution -> Normalization -> Activation block.
    Replaces `ConvModule` (2D) and `ConvModule3D` (3D).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
        act: bool = True,
    ):
        if padding is None:
            padding = (kernel_size // 2) * dilation
        conv_cls = get_conv(dims)
        layers = [
            conv_cls(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=(norm_type in ("none", "identity", "")),
            ),
            get_norm(norm_type, out_channels, dims=dims),
        ]
        if act and act_type not in ("none", "identity", ""):
            layers.append(get_activation(act_type))
        super().__init__(*layers)


@BLOCK_REGISTRY.register("res_block", aliases=["conv_block"])
class ResBlock(nn.Module):
    """
    Unified 2D / 3D Residual Convolutional Block (`ResBlock2D`, `ResBlock3D`, `ConvBlock3D`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        conv_cls = get_conv(dims)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_norm(norm_type, out_channels, dims=dims)
        self.act1 = get_activation(act_type)

        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_channels, dims=dims)
        self.act2 = get_activation(act_type)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv_cls(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm(norm_type, out_channels, dims=dims),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


@BLOCK_REGISTRY.register("double_conv")
class DoubleConv(nn.Module):
    """Standard U-Net Double Convolution Block (2D or 3D)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dims: int = 2,
        norm_type: str = "batchnorm",
        act_type: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type),
            ConvNormAct(out_channels, out_channels, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@BLOCK_REGISTRY.register("layer_scale")
class LayerScale(nn.Module):
    """Dimension-agnostic LayerScale for stabilizing deep inverted bottlenecks."""

    def __init__(self, dim: int, init_values: float = 1e-5, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        view_shape = (1, -1) + (1,) * self.dims
        return x * self.gamma.view(*view_shape)


@BLOCK_REGISTRY.register("uib_block")
class UIBBlock(nn.Module):
    """
    Unified 2D / 3D Universal Inverted Bottleneck (UIB) Block (S0-M3-E0 configuration)
    with Squeeze-and-Excitation and LayerScale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        exp_ratio: float = 4.0,
        se_ratio: float = 0.25,
        layer_scale_init: float = 1e-5,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.has_skip = stride == 1 and in_channels == out_channels
        divisor = 4 if dims == 3 else 8
        mid_channels = make_divisible(in_channels * exp_ratio, divisor=divisor)
        conv_cls = get_conv(dims)

        self.pw_exp = ConvNormAct(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
            dims=dims, norm_type=norm_type, act_type=act_type,
        )
        self.dw_mid = ConvNormAct(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
            groups=mid_channels, dims=dims, norm_type=norm_type, act_type=act_type,
        )
        self.se = SqueezeExcite(mid_channels, rd_ratio=se_ratio, dims=dims, act_type=act_type, divisor=divisor)
        self.pw_proj = nn.Sequential(
            conv_cls(mid_channels, out_channels, kernel_size=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
        )
        self.layer_scale = LayerScale(out_channels, init_values=layer_scale_init, dims=dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = x + shortcut
        return x


@BLOCK_REGISTRY.register("bga_layer")
class BGALayer(nn.Module):
    """
    Unified 2D / 3D Bilateral Guided Aggregation (BGA) Layer.
    Fuses high-resolution detail features (1/8) with low-resolution semantic features (1/32)
    via bidirectional cross-attention gating.
    """

    def __init__(
        self,
        channels: int = 64,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.dims = dims
        conv_cls = get_conv(dims)

        self.left1 = nn.Sequential(
            conv_cls(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            get_norm(norm_type, channels, dims=dims),
            conv_cls(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            conv_cls(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm(norm_type, channels, dims=dims),
            get_avg_pool(dims=dims, kernel_size=2, stride=2, padding=0),
        )
        self.right1 = nn.Sequential(
            conv_cls(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            get_norm(norm_type, channels, dims=dims),
        )
        self.right2 = nn.Sequential(
            conv_cls(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            get_norm(norm_type, channels, dims=dims),
            conv_cls(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conv = ConvNormAct(channels, channels, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        l1 = self.left1(x_d)
        l2 = self.left2(x_d)

        r1 = torch.sigmoid(interpolate_nd(self.right1(x_s), size=x_d.shape[2:], dims=self.dims))
        r2 = torch.sigmoid(self.right2(x_s))

        if l2.shape[2:] != r2.shape[2:]:
            l2 = interpolate_nd(l2, size=r2.shape[2:], dims=self.dims)

        fuse_d = l1 * r1
        fuse_s = interpolate_nd(l2 * r2, size=x_d.shape[2:], dims=self.dims)
        return self.conv(fuse_d + fuse_s)


@BLOCK_REGISTRY.register("residual_refinement_block", aliases=["rrb"])
class ResidualRefinementBlock(nn.Module):
    """
    Unified 2D / 3D Residual Refinement Block (RRB) from SOPSeg.
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
        self.conv1 = ConvNormAct(
            in_channels, out_channels, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type
        )
        self.conv2 = nn.Sequential(
            conv_cls(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
        )
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
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


@BLOCK_REGISTRY.register("conv_lstm_cell")
class ConvLSTMCell(nn.Module):
    """
    Unified 2D / 3D Convolutional LSTM Cell for spatiotemporal feature dynamics.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dims: int = 2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dims = dims
        padding = kernel_size // 2
        conv_cls = get_conv(dims)
        self.conv = conv_cls(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            b = x.shape[0]
            spatial = x.shape[2:]
            h_cur = torch.zeros(b, self.hidden_channels, *spatial, device=x.device, dtype=x.dtype)
            c_cur = torch.zeros(b, self.hidden_channels, *spatial, device=x.device, dtype=x.dtype)
        else:
            h_cur, c_cur = state

        combined = torch.cat([x, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(self.conv(combined), 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


@BLOCK_REGISTRY.register("siamese_fusion_block")
class SiameseFeatureFusionBlock(nn.Module):
    """
    Multi-scale Bi-Temporal Feature Difference & State Fusion Block (`Concat([|F1 - F2|, F2])`).
    """

    def __init__(
        self,
        channels: int,
        dims: int = 2,
        norm_type: str = "batchnorm",
        act_type: str = "relu",
    ):
        super().__init__()
        self.conv = ConvNormAct(
            channels * 2, channels, kernel_size=1, stride=1, padding=0,
            dims=dims, norm_type=norm_type, act_type=act_type,
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        return self.conv(torch.cat([diff, f2], dim=1))


@BLOCK_REGISTRY.register("transformer_block")
class TransformerBlock(nn.Module):
    """
    Standard Transformer Self-Attention + Feed-Forward Block (used in WeedSense TGD & Co-Seg).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


# ---------------------------------------------------------------------------
# Explicit 2D & 3D Convenience Subclasses (Backward-Compatible Drop-ins)
# ---------------------------------------------------------------------------

class ConvBlock3D(ResBlock):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(in_channels, out_channels, stride=stride, dims=3, norm_type="instancenorm", act_type="leakyrelu")


ResBlock3D = ConvBlock3D


class ResBlock2D(ResBlock):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(in_channels, out_channels, stride=stride, dims=2, norm_type="batchnorm", act_type="relu")


class UIBBlock2D(UIBBlock):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, exp_ratio: float = 6.0, se_ratio: float = 0.25):
        super().__init__(in_channels, out_channels, stride=stride, exp_ratio=exp_ratio, se_ratio=se_ratio, dims=2, norm_type="batchnorm", act_type="relu")


class UIBBlock3D(UIBBlock):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, exp_ratio: float = 4.0, se_ratio: float = 0.25):
        super().__init__(in_channels, out_channels, stride=stride, exp_ratio=exp_ratio, se_ratio=se_ratio, dims=3, norm_type="instancenorm", act_type="leakyrelu")


class BGALayer3D(BGALayer):
    def __init__(self, channels: int = 64):
        super().__init__(channels=channels, dims=3, norm_type="instancenorm", act_type="leakyrelu")


class ResidualRefinementBlock3D(ResidualRefinementBlock):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, dims=3, norm_type="instancenorm", act_type="leakyrelu")
