"""
Reusable change-detection modules from three 2026 studies:

1. Ma et al. (2026), Pattern Recognition 114925 - SSCP attention for water body CD
   (official code: https://github.com/QingMa1/SSCP):
   - `SSCPAttention` (MSA + SRGA + CSA)
2. Chen et al. (2026), IEEE GRSL 23, 6003805 - FAPMNet (flow-aligned prototype memory):
   - `FlowAlignmentModule` (FAM), `PrototypeMemoryModule` (PMM), `CrossScaleAttentionFusion` (CSAF)
3. Wang et al. (2026), IEEE TGRS 64, 4401714 - FarmCD (region-guided gated CD):
   - `FarmCDFFAM`, `FarmCDDFM`, `FarmCDFSGM`, `FarmCDMFRM`

Notes on faithful vs. adapted parts are given in each docstring.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ag_models.core.registry import ATTENTION_REGISTRY, BLOCK_REGISTRY


def _cbr(in_c: int, out_c: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# =============================================================================
# 1. SSCP attention (Ma et al., 2026)
# =============================================================================


@ATTENTION_REGISTRY.register("sscp_attention", aliases=["sscp"])
class SSCPAttention(nn.Module):
    """
    Spatial Semantics and Continuity Perception (SSCP) attention.

    Follows the official `SSCP.py`:
      x -> MSA (multi-scale directional 1-D depthwise convs on H- and W-pooled descriptors,
               4 channel groups with kernels 3/5/7/9, gates on rows and columns)
        -> SRGA (spatial relation-aware global attention: pairwise position relations between all
               positions, embedded and fused with a channel-mean descriptor into one spatial gate), residual
        -> CSA (window-pooled channel self-attention over the head dimension, global-pooled channel gate).

    Adaptation: the official SRGA hard-codes the number of positions (`in_spatial = H*W`). To accept any
    input size, the relation matrix is computed on a fixed `grid x grid` adaptive-average-pooled map and the
    resulting spatial gate is bilinearly resized to (H, W).
    """

    def __init__(
        self,
        dim: int,
        head_num: int = 4,
        window_size: int = 7,
        group_kernel_sizes: Tuple[int, int, int, int] = (3, 5, 7, 9),
        grid: int = 8,
        cha_ratio: int = 8,
        spa_ratio: int = 8,
        down_ratio: int = 8,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % 4 == 0 and dim % head_num == 0, "dim must be divisible by 4 and head_num"
        self.dim, self.head_num, self.head_dim = dim, head_num, dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size
        self.grid = grid
        gc = dim // 4
        self.group_chans = gc

        self.local_dwc = nn.Conv1d(gc, gc, group_kernel_sizes[0], padding=group_kernel_sizes[0] // 2, groups=gc)
        self.global_dwc_s = nn.Conv1d(gc, gc, group_kernel_sizes[1], padding=group_kernel_sizes[1] // 2, groups=gc)
        self.global_dwc_m = nn.Conv1d(gc, gc, group_kernel_sizes[2], padding=group_kernel_sizes[2] // 2, groups=gc)
        self.global_dwc_l = nn.Conv1d(gc, gc, group_kernel_sizes[3], padding=group_kernel_sizes[3] // 2, groups=gc)
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        inter_c = max(dim // cha_ratio, 1)
        n_pos = grid * grid
        inter_s = max(n_pos // spa_ratio, 1)
        self.phi = _cbr(dim, inter_c, 1)
        self.eta = _cbr(dim, inter_c, 1)
        self.gx = _cbr(dim, inter_c, 1)
        self.gg = _cbr(n_pos * 2, inter_s, 1)
        ch_s = 1 + inter_s
        self.w_spatial = nn.Sequential(
            nn.Conv2d(ch_s, max(ch_s // down_ratio, 1), 1, bias=False),
            nn.BatchNorm2d(max(ch_s // down_ratio, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch_s // down_ratio, 1), 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )

        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, 1, bias=False, groups=dim)
        self.k = nn.Conv2d(dim, dim, 1, bias=False, groups=dim)
        self.v = nn.Conv2d(dim, dim, 1, bias=False, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def _msa(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gc = self.group_chans
        x_h = x.mean(dim=3)
        x_w = x.mean(dim=2)
        lh, sh, mh, gh = torch.split(x_h, gc, dim=1)
        lw, sw, mw, gw = torch.split(x_w, gc, dim=1)
        ah = torch.sigmoid(
            self.norm_h(torch.cat([self.local_dwc(lh), self.global_dwc_s(sh), self.global_dwc_m(mh), self.global_dwc_l(gh)], 1))
        ).view(b, c, h, 1)
        aw = torch.sigmoid(
            self.norm_w(torch.cat([self.local_dwc(lw), self.global_dwc_s(sw), self.global_dwc_m(mw), self.global_dwc_l(gw)], 1))
        ).view(b, c, 1, w)
        return x * ah * aw

    def _srga(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.grid
        xp = F.adaptive_avg_pool2d(x, (g, g))
        phi = self.phi(xp).flatten(2).transpose(1, 2)  # B, N, C'
        eta = self.eta(xp).flatten(2)  # B, C', N
        gs = torch.bmm(phi, eta)  # B, N, N
        gs_in = gs.transpose(1, 2).reshape(b, g * g, g, g)
        gs_out = gs.reshape(b, g * g, g, g)
        gs_joint = self.gg(torch.cat([gs_in, gs_out], dim=1))
        gx = self.gx(xp).mean(dim=1, keepdim=True)
        w_ys = self.w_spatial(torch.cat([gx, gs_joint], dim=1))
        gate = torch.sigmoid(F.interpolate(w_ys, size=(h, w), mode="bilinear", align_corners=False))
        return gate * x

    def _csa(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        if h >= ws and w >= ws:
            y = F.avg_pool2d(x, ws, stride=ws)
        else:
            y = F.adaptive_avg_pool2d(x, 1)
        hh, ww = y.shape[-2:]
        y = self.norm(y)
        q, k, v = self.q(y), self.k(y), self.v(y)

        def split(t):
            return t.reshape(b, self.head_num, self.head_dim, hh * ww)

        q, k, v = split(q), split(k), split(v)
        attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scaler).softmax(dim=-1))
        out = (attn @ v).reshape(b, c, hh, ww).mean(dim=(2, 3), keepdim=True)
        return torch.sigmoid(out) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._msa(x)
        x = x + self._srga(x)
        return self._csa(x)


# =============================================================================
# 2. FAPMNet modules (Chen et al., 2026)
# =============================================================================


def flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Differentiable bilinear warp of `x` (B,C,H,W) with a pixel-unit displacement field `flow` (B,2,H,W)."""
    b, _, h, w = x.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    gx = (xs.unsqueeze(0) + flow[:, 0]) * 2.0 / max(w - 1, 1) - 1.0
    gy = (ys.unsqueeze(0) + flow[:, 1]) * 2.0 / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


@BLOCK_REGISTRY.register("fapmnet_fam", aliases=["flow_alignment_module"])
class FlowAlignmentModule(nn.Module):
    """
    Flow Alignment Module (FAM, Eqs. 2-4): a light conv head on concat(pre, post) regresses a 2-channel
    displacement field, which warps one temporal feature onto the other by bilinear sampling. It is applied
    symmetrically (post->pre and pre->post); each aligned feature is merged back through a sigmoid gate.

    Adaptation: Eq. (4) of the letter is typeset ambiguously; the gated residual fusion
    `x + sigmoid(Conv([x, aligned])) * aligned` is used. The flow head's last layer is zero-initialised so
    training starts from the identity warp.
    """

    def __init__(self, channels: int):
        super().__init__()

        def head():
            h = nn.Sequential(_cbr(channels * 2, channels, 3), nn.Conv2d(channels, 2, 3, padding=1))
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)
            return h

        self.flow_post = head()
        self.flow_pre = head()
        self.gate_pre = nn.Conv2d(channels * 2, channels, 1)
        self.gate_post = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        f_post = self.flow_post(torch.cat([pre, post], dim=1))
        f_pre = self.flow_pre(torch.cat([post, pre], dim=1))
        post_al = flow_warp(post, f_post)
        pre_al = flow_warp(pre, f_pre)
        pre_out = pre + torch.sigmoid(self.gate_pre(torch.cat([pre, post_al], 1))) * post_al
        post_out = post + torch.sigmoid(self.gate_post(torch.cat([post, pre_al], 1))) * pre_al
        return pre_out, post_out, f_post


@BLOCK_REGISTRY.register("fapmnet_pmm", aliases=["prototype_memory_module"])
class PrototypeMemoryModule(nn.Module):
    """
    Prototype Memory Module (PMM, Eqs. 5-6): K=16 learnable prototypes; cosine similarity between each
    pixel feature and each prototype is softmax-normalised to reconstruct a class-guided feature, which is
    fused back with a residual and conv refinement. Returns (refined feature, similarity map).
    """

    def __init__(self, channels: int, num_prototypes: int = 16, temperature: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, channels) * 0.1)
        self.temperature = temperature
        self.refine = _cbr(channels * 2, channels, 3)

    def forward(self, x: torch.Tensor):
        xn = F.normalize(x, dim=1)
        pn = F.normalize(self.prototypes, dim=1)
        sim = torch.einsum("bchw,kc->bkhw", xn, pn)
        w = F.softmax(sim / self.temperature, dim=1)
        recon = torch.einsum("bkhw,kc->bchw", w, self.prototypes)
        return x + self.refine(torch.cat([x, recon], dim=1)), sim


@ATTENTION_REGISTRY.register("fapmnet_csaf", aliases=["cross_scale_attention_fusion"])
class CrossScaleAttentionFusion(nn.Module):
    """Cross-scale attention and fusion (CSAF): spatial (conv on mean/max maps) and channel (SE) attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, channels, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa * self.se(x) + x


# =============================================================================
# 3. FarmCD modules (Wang et al., 2026)
# =============================================================================


def _channel_attn_mlp(c: int, r: int = 8) -> nn.Sequential:
    m = max(c // r, 4)
    return nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(m, c, 1, bias=False))


@BLOCK_REGISTRY.register("farmcd_ffam", aliases=["farmland_feature_aware_module"])
class FarmCDFFAM(nn.Module):
    """
    Farmland (region) Feature Aware Module, Eqs. (1)-(3):
      F'_H = ReLU(BN(W_H * Up(F_H)));  A = sigmoid(MLP(AVG(F_L+F'_H)) + MLP(MAX(F_L+F'_H)));
      F_out = A * F_L + (1-A) * F'_H.
    """

    def __init__(self, low_channels: int, high_channels: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(high_channels, low_channels, 1, bias=False), nn.BatchNorm2d(low_channels), nn.ReLU(inplace=True))
        self.mlp = _channel_attn_mlp(low_channels)

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(f_high, size=f_low.shape[-2:], mode="bilinear", align_corners=False)
        fh = self.proj(up)
        s = f_low + fh
        a = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(s, 1)) + self.mlp(F.adaptive_max_pool2d(s, 1)))
        return a * f_low + (1.0 - a) * fh


class _CoordAtt(nn.Module):
    """Coordinate attention producing directional gates (A_h, A_w)."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        m = max(c // r, 8)
        self.conv1 = nn.Sequential(nn.Conv2d(c, m, 1), nn.BatchNorm2d(m), nn.ReLU(inplace=True))
        self.conv_h = nn.Conv2d(m, c, 1)
        self.conv_w = nn.Conv2d(m, c, 1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        xh = x.mean(dim=3, keepdim=True)  # B,C,H,1
        xw = x.mean(dim=2, keepdim=True).transpose(2, 3)  # B,C,W,1
        y = self.conv1(torch.cat([xh, xw], dim=2))
        yh, yw = torch.split(y, [h, w], dim=2)
        return torch.sigmoid(self.conv_h(yh)), torch.sigmoid(self.conv_w(yw.transpose(2, 3)))


@BLOCK_REGISTRY.register("farmcd_dfm", aliases=["differential_fusion_module"])
class FarmCDDFM(nn.Module):
    """
    Differential Fusion Module, Eqs. (4)-(8):
      F_d = |X1 - X2|;  F_c = [X1, X2];  Z_c = ReLU(BN(W1 * ReLU(BN(W3 * F_c))));
      Z_d = F_d * A_w * A_h (coordinate attention from F_d);  F_out = Z_c + Z_d.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = _cbr(channels * 2, channels, 3)
        self.conv1 = _cbr(channels, channels, 1)
        self.coord = _CoordAtt(channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        fd = torch.abs(x1 - x2)
        zc = self.conv1(self.conv3(torch.cat([x1, x2], dim=1)))
        ah, aw = self.coord(fd)
        return zc + fd * aw * ah


@BLOCK_REGISTRY.register("farmcd_fsgm", aliases=["soft_hard_gate_module"])
class FarmCDFSGM(nn.Module):
    """
    Soft-Hard Gate Module, Eqs. (9)-(13): soft adaptive gate A_s = sigmoid(conv1x1-ReLU-conv1x1(P_s)) from the
    region probability map, hard gate P_b (binary mask, here `P_s > threshold`), F_g = F * A_s * P_b, optional residual.

    Adaptation: the paper's hybrid extractor additionally filters connected components of the thresholded *change*
    map by overlap with the region masks (Eq. 14); this inference-time post-processing is not included.
    """

    def __init__(self, channels: int, mid: int = 8, threshold: float = 0.3, residual: bool = True):
        super().__init__()
        self.threshold = threshold
        self.residual = residual
        self.soft = nn.Sequential(nn.Conv2d(1, mid, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, 1, 1))

    def forward(self, feat: torch.Tensor, p_soft: torch.Tensor) -> torch.Tensor:
        ps = F.interpolate(p_soft, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        a_s = torch.sigmoid(self.soft(ps))
        p_b = (ps > self.threshold).to(feat.dtype)
        fg = feat * a_s * p_b
        return fg + feat if self.residual else fg


@BLOCK_REGISTRY.register("farmcd_mfrm", aliases=["multiscale_refinement_module"])
class FarmCDMFRM(nn.Module):
    """
    Multiscale refinement module, Eqs. (15)-(16): channel recalibration F' = sigmoid(MLP(AVG(F)+MAX(F))) * F, then
    sigmoid -> 2x bilinear upsampling -> 1x1 projection -> BN (the typeset Eq. 16 is ambiguous; this is a
    straightforward reading).
    """

    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True):
        super().__init__()
        self.mlp = _channel_attn_mlp(in_channels)
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1)))
        y = torch.sigmoid(a * x)
        if self.upsample:
            y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(y)
