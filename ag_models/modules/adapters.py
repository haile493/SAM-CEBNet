"""
Model Adapters, Wrappers, Parameter-Efficient Fine-Tuning (LoRA), and Domain Adaptation Utilities.

Standardizes and unifies the implementations from:
- `cangio/src/sam_satellite.py` (`LoRALinear`, `apply_lora_to_sam`)
- `cangio/src/models.py` & `cangio/src/temporal_models.py` (`AutoPadWrapper`, `AutoPadModelWrapper`)
- `mirnomes/head_model/preprocess/app/util/gradient_reversal.py` (`GradientReversal`)
- `cangio/src/sopseg.py` (`RegionAdaptiveMagnification`)
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ag_models.core.registry import BLOCK_REGISTRY


@BLOCK_REGISTRY.register("lora_linear")
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for any `nn.Linear` layer.
    Computes: `y = W_0(x) + (x @ A @ B) * (alpha / rank)`.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / float(rank)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_f = linear.in_features
        out_f = linear.out_features
        device = linear.weight.device
        dtype = linear.weight.dtype

        self.lora_A = nn.Parameter(
            torch.randn(in_f, rank, device=device, dtype=dtype) * (1.0 / math.sqrt(in_f))
        )
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(
    model: nn.Module,
    target_names: Iterable[str] = ("qkv", "q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    freeze_base: bool = True,
) -> List[LoRALinear]:
    """
    Generic utility to inject `LoRALinear` adapters into matching `nn.Linear` submodules
    of any PyTorch model (ViT, SAM, Swin, LLM, etc.).
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    targets = set(target_names)
    injected: List[LoRALinear] = []

    for _, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
                injected.append(lora_layer)

    return injected


@BLOCK_REGISTRY.register("autopad_wrapper")
class AutoPadWrapper(nn.Module):
    """
    Dynamically pads spatial dimensions (2D `(B, C, H, W)`, 3D `(B, C, D, H, W)`,
    or 5D temporal sequences `(B, T, C, H, W)`) to multiples of `multiple` (e.g., 32)
    before calling `base_model`, and crops outputs back to original spatial dimensions.
    """

    def __init__(self, base_model: nn.Module, multiple: int = 32, dims: int = 2):
        super().__init__()
        self.base_model = base_model
        self.multiple = multiple
        self.dims = dims

    def _pad_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if self.dims == 3 and x.dim() == 5:
            # 3D volumetric: (B, C, D, H, W)
            d, h, w = x.shape[-3:]
            pd = (self.multiple - d % self.multiple) % self.multiple
            ph = (self.multiple - h % self.multiple) % self.multiple
            pw = (self.multiple - w % self.multiple) % self.multiple
            if pd == 0 and ph == 0 and pw == 0:
                return x, (d, h, w)
            return F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect"), (d, h, w)

        # 2D image (B, C, H, W) or 5D temporal sequence (B, T, C, H, W)
        h, w = x.shape[-2:]
        ph = (self.multiple - h % self.multiple) % self.multiple
        pw = (self.multiple - w % self.multiple) % self.multiple
        if ph == 0 and pw == 0:
            return x, (h, w)

        if x.dim() == 4:
            return F.pad(x, (0, pw, 0, ph), mode="reflect"), (h, w)
        elif x.dim() == 5:
            b, t, c, _, _ = x.shape
            x_flat = x.view(b * t, c, h, w)
            x_pad = F.pad(x_flat, (0, pw, 0, ph), mode="reflect").view(b, t, c, h + ph, w + pw)
            return x_pad, (h, w)
        return x, (h, w)

    def _unpad_output(self, out: Any, orig_spatial: Tuple[int, ...]) -> Any:
        if isinstance(out, torch.Tensor):
            if len(orig_spatial) == 3 and out.dim() == 5:
                d, h, w = orig_spatial
                return out[..., :d, :h, :w]
            elif len(orig_spatial) == 2 and out.dim() >= 4:
                h, w = orig_spatial
                return out[..., :h, :w]
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            x_pad, orig_spatial = self._pad_tensor(args[0])
            out = self.base_model(x_pad, **kwargs)
            return self._unpad_output(out, orig_spatial)
        elif len(args) == 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            x1_pad, orig_spatial = self._pad_tensor(args[0])
            x2_pad, _ = self._pad_tensor(args[1])
            out = self.base_model(x1_pad, x2_pad, **kwargs)
            return self._unpad_output(out, orig_spatial)
        return self.base_model(*args, **kwargs)


AutoPadModelWrapper = AutoPadWrapper


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


@BLOCK_REGISTRY.register("gradient_reversal")
class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer (GRL) for Domain-Adversarial Neural Networks (DANN).
    Identity in forward pass; multiplies gradient by `-scale` in backward pass.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.scale)


class RegionAdaptiveMagnification:
    """
    Region-Adaptive Magnification (RAM) Strategy from SOPSeg (`cangio/src/sopseg.py`).
    Computes adaptive context crop sizes `S` around small objects based on `d = max(w, h)`.
    """

    def __init__(
        self,
        m: int = 32,
        k0: float = 2.0,
        s_max: int = 1024,
        s_min: int = 16,
        s_in: int = 256,
    ):
        self.m = m
        self.k0 = k0
        self.s_max = s_max
        self.s_min = s_min
        self.s_in = s_in
        self.k = (s_max - k0 * m) / float(s_max - m)

    def compute_region_size(self, d: float) -> int:
        if d < self.m:
            s = self.k0 * d
        else:
            s = self.k * d + (self.k0 - self.k) * self.m
        return max(self.s_min, min(self.s_max, int(round(s))))
