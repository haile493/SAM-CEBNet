"""
Unified 2D & 3D Wavelet Transform, WaveFusion Refinement, and Wavelet-Guided Attention Modules.

Standardizes and unifies the implementations from:
- `cangio/src/sopseg_enhanced.py` (2D Haar DWT/IDWT & WaveFusion)
- `liverseg/models/waveattn_sopseg_liver.py` (3D Haar DWT/IDWT & WaveFusion)
- `liverseg/models/waveco_mtl_net.py` (3D Wavelet Attention Gate)
- `liverseg/models/weakmedsam_coseg_waveattn.py` (2D Wavelet Attention Gate)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ag_models.core.layers import get_activation, get_conv, get_norm, interpolate_nd
from ag_models.core.registry import ATTENTION_REGISTRY, BLOCK_REGISTRY


def _build_haar_filters(dims: int = 3) -> torch.Tensor:
    """
    Constructs orthonormal 2D (4 subbands) or 3D (8 subbands) Haar wavelet convolution kernels.
    """
    low = torch.tensor([1.0, 1.0], dtype=torch.float32) / (2.0 ** 0.5)
    high = torch.tensor([-1.0, 1.0], dtype=torch.float32) / (2.0 ** 0.5)

    if dims == 2:
        # Order: LL, LH, HL, HH -> (4, 1, 2, 2)
        ll = torch.outer(low, low)
        lh = torch.outer(high, low)
        hl = torch.outer(low, high)
        hh = torch.outer(high, high)
        return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
    elif dims == 3:
        # Order: LLL + 7 high-frequency subbands -> (8, 1, 2, 2, 2)
        filters = []
        for i in (low, high):
            for j in (low, high):
                for k in (low, high):
                    filters.append(torch.einsum("i,j,k->ijk", i, j, k))
        return torch.stack(filters, dim=0).unsqueeze(1)
    else:
        raise ValueError(f"Haar wavelet transform supports dims=2 or dims=3, got dims={dims}.")


@BLOCK_REGISTRY.register("haar_dwt")
class HaarDWT(nn.Module):
    """
    Dimension-agnostic (2D or 3D) Haar Discrete Wavelet Transform using grouped convolutions.

    - For `dims=2`: Decomposes `(B, C, H, W)` into:
      `low`: `(B, C, H/2, W/2)` and `high`: `(B, 3*C, H/2, W/2)`.
    - For `dims=3`: Decomposes `(B, C, D, H, W)` into:
      `low`: `(B, C, D/2, H/2, W/2)` and `high`: `(B, 7*C, D/2, H/2, W/2)`.
    """

    def __init__(self, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.num_subbands = 4 if dims == 2 else 8
        self.num_high = self.num_subbands - 1
        self.register_buffer("filters", _build_haar_filters(dims))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        # Automatically pad odd spatial dimensions by 1 if needed
        pad = []
        needs_pad = False
        for s in reversed(spatial):
            p = s % 2
            pad.extend([0, p])
            if p > 0:
                needs_pad = True
        if needs_pad:
            x = F.pad(x, pad, mode="reflect")

        if self.dims == 2:
            w_filt = self.filters.repeat(c, 1, 1, 1)
            dwt_out = F.conv2d(x, w_filt, stride=2, groups=c)
            h2, w2 = dwt_out.shape[-2:]
            dwt_out = dwt_out.view(b, c, 4, h2, w2)
            low = dwt_out[:, :, 0]
            high = dwt_out[:, :, 1:].reshape(b, c * 3, h2, w2)
            return low, high
        else:
            w_filt = self.filters.repeat(c, 1, 1, 1, 1)
            dwt_out = F.conv3d(x, w_filt, stride=2, groups=c)
            d2, h2, w2 = dwt_out.shape[-3:]
            dwt_out = dwt_out.view(b, c, 8, d2, h2, w2)
            low = dwt_out[:, :, 0]
            high = dwt_out[:, :, 1:].reshape(b, c * 7, d2, h2, w2)
            return low, high


@BLOCK_REGISTRY.register("haar_idwt")
class HaarIDWT(nn.Module):
    """
    Dimension-agnostic (2D or 3D) Inverse Haar Discrete Wavelet Transform
    using transposed grouped convolutions. Exact inverse of `HaarDWT`.
    """

    def __init__(self, dims: int = 3):
        super().__init__()
        self.dims = dims
        self.num_subbands = 4 if dims == 2 else 8
        self.num_high = self.num_subbands - 1
        self.register_buffer("filters", _build_haar_filters(dims))

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        b, c = low.shape[:2]
        if self.dims == 2:
            h2, w2 = low.shape[-2:]
            high_reshaped = high.view(b, c, 3, h2, w2)
            x_dwt = torch.cat([low.unsqueeze(2), high_reshaped], dim=2).view(b, c * 4, h2, w2)
            w_filt = self.filters.repeat(c, 1, 1, 1)
            return F.conv_transpose2d(x_dwt, w_filt, stride=2, groups=c)
        else:
            d2, h2, w2 = low.shape[-3:]
            high_reshaped = high.view(b, c, 7, d2, h2, w2)
            x_dwt = torch.cat([low.unsqueeze(2), high_reshaped], dim=2).view(b, c * 8, d2, h2, w2)
            w_filt = self.filters.repeat(c, 1, 1, 1, 1)
            return F.conv_transpose3d(x_dwt, w_filt, stride=2, groups=c)


@BLOCK_REGISTRY.register("wave_fusion_block", aliases=["wavefusion"])
class WaveFusionBlock(nn.Module):
    """
    Unified 2D / 3D WaveFusion Refinement Block.
    Decomposes features into low-frequency semantics and high-frequency edge subbands,
    processes each with dedicated residual convolutional streams, and reconstructs via IDWT.
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = dims
        num_high = 3 if dims == 2 else 7
        conv_cls = get_conv(dims)

        self.dwt = HaarDWT(dims=dims)
        self.idwt = HaarIDWT(dims=dims)

        # High-frequency edge processing stream
        self.conv_high1 = conv_cls(in_channels * num_high, in_channels, kernel_size=1, bias=False)
        self.res_high = nn.Sequential(
            conv_cls(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, in_channels, dims=dims),
            get_activation(act_type),
            conv_cls(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, in_channels, dims=dims),
        )
        self.conv_high2 = conv_cls(in_channels, in_channels * num_high, kernel_size=1, bias=False)

        # Low-frequency semantic processing stream
        self.res_low = nn.Sequential(
            conv_cls(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, in_channels, dims=dims),
            get_activation(act_type),
            conv_cls(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, in_channels, dims=dims),
        )
        self.act = get_activation(act_type)

        # Reconstruction projection & residual shortcut
        self.out_conv = nn.Sequential(
            conv_cls(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels, dims=dims),
            get_activation(act_type),
        )
        self.shortcut = (
            nn.Sequential(
                conv_cls(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm(norm_type, out_channels, dims=dims),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]
        low, high = self.dwt(x)

        high_feat = self.conv_high1(high)
        high_res = self.res_high(high_feat)
        high_out = self.conv_high2(self.act(high_feat + high_res))

        low_res = self.res_low(low)
        low_out = self.act(low + low_res)

        rec = self.idwt(low_out, high_out)
        if rec.shape[2:] != orig_size:
            rec = interpolate_nd(rec, size=orig_size, dims=self.dims)

        return self.out_conv(rec) + self.shortcut(x)


@ATTENTION_REGISTRY.register("wavelet_attention_gate", aliases=["wavelet_gate"])
class WaveletAttentionGate(nn.Module):
    """
    Unified 2D / 3D Wavelet-Guided Attention Gate for Decoder Skip Connections.
    Supports optional high-frequency DWT edge injection (`use_dwt_high=True`).
    """

    def __init__(
        self,
        f_g: int,
        f_l: int,
        f_int: int,
        dims: int = 3,
        use_dwt_high: bool = False,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.dims = dims
        self.use_dwt_high = use_dwt_high
        conv_cls = get_conv(dims)
        num_high = 3 if dims == 2 else 7

        self.w_g = nn.Sequential(
            conv_cls(f_g, f_int, kernel_size=1, bias=False),
            get_norm(norm_type, f_int, dims=dims),
        )
        self.w_x = nn.Sequential(
            conv_cls(f_l, f_int, kernel_size=1, bias=False),
            get_norm(norm_type, f_int, dims=dims),
        )

        if use_dwt_high:
            self.dwt = HaarDWT(dims=dims)
            self.conv_high = nn.Sequential(
                conv_cls(f_l * num_high, f_int, kernel_size=1, bias=False),
                get_norm(norm_type, f_int, dims=dims),
                get_activation(act_type),
            )

        self.psi = nn.Sequential(
            conv_cls(f_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = get_activation(act_type)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = interpolate_nd(g1, size=x1.shape[2:], dims=self.dims)

        combined = g1 + x1
        if self.use_dwt_high:
            _, high = self.dwt(x)
            high_proj = self.conv_high(high)
            high_up = interpolate_nd(high_proj, size=x1.shape[2:], dims=self.dims)
            combined = combined + high_up

        psi = self.psi(self.act(combined))
        return x * psi


# ---------------------------------------------------------------------------
# Explicit 2D & 3D Convenience Subclasses (Backward-Compatible Drop-ins)
# ---------------------------------------------------------------------------

class HaarDWT2D(HaarDWT):
    def __init__(self):
        super().__init__(dims=2)


class HaarIDWT2D(HaarIDWT):
    def __init__(self):
        super().__init__(dims=2)


class HaarDWT3D(HaarDWT):
    def __init__(self):
        super().__init__(dims=3)


class HaarIDWT3D(HaarIDWT):
    def __init__(self):
        super().__init__(dims=3)


class WaveFusionRefinementBlock2D(WaveFusionBlock):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, dims=2, norm_type="batchnorm", act_type="gelu")


class WaveFusionRefinementBlock3D(WaveFusionBlock):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, dims=3, norm_type="instancenorm", act_type="leakyrelu")


class WaveletAttentionGate2D(WaveletAttentionGate):
    def __init__(self, f_g: int, f_l: int, f_int: int, use_dwt_high: bool = True):
        super().__init__(f_g, f_l, f_int, dims=2, use_dwt_high=use_dwt_high, norm_type="batchnorm", act_type="relu")


class WaveletAttentionGate3D(WaveletAttentionGate):
    def __init__(self, f_g: int, f_l: int, f_int: int, use_dwt_high: bool = False):
        super().__init__(f_g, f_l, f_int, dims=3, use_dwt_high=use_dwt_high, norm_type="instancenorm", act_type="leakyrelu")
