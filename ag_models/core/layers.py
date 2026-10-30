"""
Dimension-Agnostic Layer Factories (1D, 2D, and 3D).

Allows architectural modules (Convolutions, Normalization, Pooling, Attention,
Wavelets, Refinement Blocks) to be written once and instantiated seamlessly
for 1D sequences, 2D remote sensing / slice images, or 3D volumetric medical CT/MRI.
"""

from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Ensures channel counts are divisible by `divisor` (hardware-friendly widths)."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_conv(dims: int = 3) -> Type[nn.Module]:
    """Returns `nn.Conv1d`, `nn.Conv2d`, or `nn.Conv3d` based on spatial `dims`."""
    if dims == 1:
        return nn.Conv1d
    elif dims == 2:
        return nn.Conv2d
    elif dims == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported spatial dims={dims}. Expected 1, 2, or 3.")


def get_conv_transpose(dims: int = 3) -> Type[nn.Module]:
    """Returns `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, or `nn.ConvTranspose3d`."""
    if dims == 1:
        return nn.ConvTranspose1d
    elif dims == 2:
        return nn.ConvTranspose2d
    elif dims == 3:
        return nn.ConvTranspose3d
    raise ValueError(f"Unsupported spatial dims={dims}. Expected 1, 2, or 3.")


class _SafeInstanceNorm(nn.Module):
    """
    Durable wrapper around `nn.InstanceNorm{1d,2d,3d}` that safely handles `1x1(x1)`
    bottleneck tensors during training without raising PyTorch's spatial size ValueError.
    """

    def __init__(self, norm_module: nn.Module):
        super().__init__()
        self.norm = norm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() > 0 and prod_spatial(x.shape[2:]) <= 1:
            if getattr(self.norm, "affine", False) and self.norm.weight is not None:
                view_shape = (1, -1) + (1,) * (x.ndim - 2)
                return x * self.norm.weight.view(*view_shape) + self.norm.bias.view(*view_shape)
            return x
        return self.norm(x)


def prod_spatial(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= s
    return p


def get_norm(
    norm_type: str,
    num_channels: int,
    dims: int = 3,
    num_groups: int = 8,
    affine: bool = True,
) -> nn.Module:
    """
    Instantiates a normalization layer across 1D, 2D, or 3D tensors.

    Supported `norm_type`:
    - `'instancenorm'` / `'in'`
    - `'batchnorm'` / `'bn'`
    - `'groupnorm'` / `'gn'`
    - `'layernorm'` / `'ln'`
    - `'none'` / `'identity'`
    """
    nt = (norm_type or "none").lower().replace("_", "")
    if nt in ("none", "identity", ""):
        return nn.Identity()

    if nt in ("instancenorm", "in"):
        if dims == 1:
            return _SafeInstanceNorm(nn.InstanceNorm1d(num_channels, affine=affine))
        elif dims == 2:
            return _SafeInstanceNorm(nn.InstanceNorm2d(num_channels, affine=affine))
        elif dims == 3:
            return _SafeInstanceNorm(nn.InstanceNorm3d(num_channels, affine=affine))

    if nt in ("batchnorm", "bn"):
        if dims == 1:
            return nn.BatchNorm1d(num_channels, affine=affine)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels, affine=affine)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels, affine=affine)

    if nt in ("groupnorm", "gn"):
        g = min(num_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels, affine=affine)

    if nt in ("layernorm", "ln"):
        return nn.GroupNorm(1, num_channels, affine=affine)

    raise ValueError(f"Unsupported norm_type='{norm_type}' for dims={dims}.")


def get_activation(act_type: str = "leakyrelu", inplace: bool = True) -> nn.Module:
    """
    Instantiates an activation layer by name.

    Supported `act_type`: `'relu'`, `'leakyrelu'`, `'gelu'`, `'silu'`/`'swish'`,
    `'elu'`, `'sigmoid'`, `'tanh'`, `'none'`.
    """
    at = (act_type or "none").lower().replace("_", "")
    if at in ("none", "identity", ""):
        return nn.Identity()
    if at == "relu":
        return nn.ReLU(inplace=inplace)
    if at in ("leakyrelu", "lrelu"):
        return nn.LeakyReLU(0.01, inplace=inplace)
    if at == "gelu":
        return nn.GELU()
    if at in ("silu", "swish"):
        return nn.SiLU(inplace=inplace)
    if at == "elu":
        return nn.ELU(inplace=inplace)
    if at == "sigmoid":
        return nn.Sigmoid()
    if at == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported act_type='{act_type}'.")


def get_adaptive_avg_pool(dims: int = 3, output_size: int = 1) -> nn.Module:
    """Returns `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, or `AdaptiveAvgPool3d`."""
    if dims == 1:
        return nn.AdaptiveAvgPool1d(output_size)
    elif dims == 2:
        return nn.AdaptiveAvgPool2d(output_size)
    elif dims == 3:
        return nn.AdaptiveAvgPool3d(output_size)
    raise ValueError(f"Unsupported dims={dims}.")


def get_adaptive_max_pool(dims: int = 3, output_size: int = 1) -> nn.Module:
    """Returns `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, or `AdaptiveMaxPool3d`."""
    if dims == 1:
        return nn.AdaptiveMaxPool1d(output_size)
    elif dims == 2:
        return nn.AdaptiveMaxPool2d(output_size)
    elif dims == 3:
        return nn.AdaptiveMaxPool3d(output_size)
    raise ValueError(f"Unsupported dims={dims}.")


def get_max_pool(dims: int = 3, kernel_size: int = 2, stride: int = 2, padding: int = 0) -> nn.Module:
    """Returns `MaxPool1d`, `MaxPool2d`, or `MaxPool3d`."""
    if dims == 1:
        return nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
    elif dims == 2:
        return nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    elif dims == 3:
        return nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
    raise ValueError(f"Unsupported dims={dims}.")


def get_avg_pool(dims: int = 3, kernel_size: int = 2, stride: int = 2, padding: int = 0) -> nn.Module:
    """Returns `AvgPool1d`, `AvgPool2d`, or `AvgPool3d`."""
    if dims == 1:
        return nn.AvgPool1d(kernel_size, stride=stride, padding=padding, count_include_pad=True)
    elif dims == 2:
        return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=True)
    elif dims == 3:
        return nn.AvgPool3d(kernel_size, stride=stride, padding=padding, count_include_pad=True)
    raise ValueError(f"Unsupported dims={dims}.")


def get_interp_mode(dims: int = 3) -> str:
    """Returns default linear interpolation mode (`'linear'`, `'bilinear'`, or `'trilinear'`)."""
    if dims == 1:
        return "linear"
    elif dims == 2:
        return "bilinear"
    elif dims == 3:
        return "trilinear"
    raise ValueError(f"Unsupported dims={dims}.")


def interpolate_nd(
    x: torch.Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[float] = None,
    dims: Optional[int] = None,
    align_corners: Optional[bool] = False,
    mode: Optional[str] = None,
) -> torch.Tensor:
    """Dimension-aware spatial interpolation helper."""
    inferred_dims = dims or (x.ndim - 2)
    interp_mode = mode or get_interp_mode(inferred_dims)
    if interp_mode in ("nearest", "area"):
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=interp_mode)
    return F.interpolate(
        x,
        size=size,
        scale_factor=scale_factor,
        mode=interp_mode,
        align_corners=align_corners,
    )
