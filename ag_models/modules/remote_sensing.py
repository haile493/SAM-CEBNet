"""
Reusable Remote Sensing Modules, Attention Mechanisms, and Building Blocks
integrated from the 9 state-of-the-art studies (2022–2026):

1. Han et al. (2022) - Improved Mask R-CNN Building Detection:
   - `RetinexShadowEnhancer`, `UNetTopDownFPNFusion`
2. Qin et al. (2025) - BiSAM-CD (IEEE TGRS):
   - `BFSCAISequenceGenerator`, `TOTCCSCVVerifier`
3. Qin et al. (2025) - SAM2-CD (IEEE JSTARS):
   - `ActivationSelectionGate` (ASG), `MSFFMBlock`
4. Xu et al. (2026) - GMFIMamba (Eng. Appl. Artif. Intell.):
   - `SS2DCrossScan`, `ConvVSSBlock`, `MCFEMBlock`, `GMBFIMBlock`
5. Liu et al. (2026) - SemiBCD (IEEE TGRS):
   - `UAPLRModule`, `LightSSHead`, `DeepLabV3PlusCDHead`
6. Wang et al. (2026) - CGCCE-Net (Information Sciences):
   - `CGRRBranch`, `GCCMModule`, `SCEMModule`, `CFDDecoderBlock`
7. Wang et al. (2026) - KGBDCNet (ISPRS J. Photogramm. Remote Sens.):
   - `CAGMModule`, `SSMBBlock`, `TTTBBlock`
8. Huang et al. (2026) - CBRNet (IEEE TGRS):
   - `CASCBlock`, `CGFEModule`, `LWCAAttention`, `BiFPNFeatureFusion`, `DSBRDecoder`
9. Hussin et al. (2026) - EdgeRefNet (IEEE TGRS):
   - `EDBBlock`, `EEMBlock`, `EdgeGuidedCrossAttention`
"""

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ag_models.core.registry import ATTENTION_REGISTRY, BLOCK_REGISTRY


# =============================================================================
# 1. Han et al. (2022) - Improved Mask R-CNN Building Detection Modules
# =============================================================================


@BLOCK_REGISTRY.register("retinex_shadow_enhancer")
class RetinexShadowEnhancer(nn.Module):
    """
    Differentiable Multi-Scale Retinex Shadow Removal & Potential-Function Contrast Enhancer
    (Han et al., Complex & Intelligent Systems, 2022).
    """

    def __init__(self, scales: Tuple[int, ...] = (3, 7, 15), gamma: float = 0.85):
        super().__init__()
        self.scales = scales
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.mix_weight = nn.Parameter(torch.tensor(0.35))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure positive range for log-domain Retinex illumination estimation
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)

        retinex_acc = torch.zeros_like(x_norm)
        log_i = torch.log(x_norm + 1e-4)
        for k in self.scales:
            pad = k // 2
            illum = F.avg_pool2d(x_norm, kernel_size=k, stride=1, padding=pad)
            retinex_acc = retinex_acc + (log_i - torch.log(illum + 1e-4))
        retinex = retinex_acc / float(len(self.scales))
        r_min = retinex.amin(dim=(-2, -1), keepdim=True)
        r_max = retinex.amax(dim=(-2, -1), keepdim=True)
        r_norm = (retinex - r_min) / (r_max - r_min + 1e-6)
        r_gamma = torch.pow(r_norm.clamp(1e-5, 1.0), self.gamma.clamp(0.2, 2.5))

        alpha = torch.sigmoid(self.mix_weight)
        return (1.0 - alpha) * x + alpha * r_gamma


@BLOCK_REGISTRY.register("unet_topdown_fpn")
class UNetTopDownFPNFusion(nn.Module):
    """
    U-Net-inspired Top-Down Multi-Scale Feature Fusion before RPN/Head
    (Han et al., 2022): 1x1 conv on lower feature + 2x upsampling of higher feature.
    """

    def __init__(self, in_channels_list: Sequence[int], out_channels: int = 256):
        super().__init__()
        self.lat_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels_list]
        )
        self.smooth_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_list
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lat_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up
        return [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]


# =============================================================================
# 2. Qin et al. (2025) - BiSAM-CD Modules
# =============================================================================


@BLOCK_REGISTRY.register("bfscai_sequence_generator")
class BFSCAISequenceGenerator(nn.Module):
    """
    Bidirectional Frame Sequence Construction with Adaptive Interpolation (BFSCAI)
    from BiSAM-CD (Qin et al., IEEE TGRS 2025).

    Constructs ascending (I1 -> I2) and descending (I2 -> I1) pseudo-video sequences:
        F^(k) = alpha_k * I_start + (1 - alpha_k) * I_end + beta * sin(pi * k / N) * (I_end - I_start)
    where alpha_k = 1 - k / (N + 1).
    """

    def __init__(self, num_interp_frames: int = 6, beta: float = 0.15):
        super().__init__()
        self.num_interp_frames = num_interp_frames
        self.beta = beta

    def build_sequence(self, img_start: torch.Tensor, img_end: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_start: (B, C, H, W)
            img_end: (B, C, H, W)
        Returns:
            seq: (B, N + 2, C, H, W)
        """
        frames = [img_start]
        n = self.num_interp_frames
        delta = img_end - img_start
        for k in range(1, n + 1):
            alpha_k = 1.0 - (float(k) / float(n + 1))
            nonlinear = self.beta * math.sin(math.pi * float(k) / float(max(n, 1))) * delta
            fk = alpha_k * img_start + (1.0 - alpha_k) * img_end + nonlinear
            frames.append(fk)
        frames.append(img_end)
        return torch.stack(frames, dim=1)

    def forward(self, img_t1: torch.Tensor, img_t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_asc = self.build_sequence(img_t1, img_t2)
        seq_desc = self.build_sequence(img_t2, img_t1)
        return seq_asc, seq_desc


@BLOCK_REGISTRY.register("totc_cscv_verifier")
class TOTCCSCVVerifier(nn.Module):
    """
    Temporal Object Tracking Consistency & Cross-Sequence Change Verification (TOTC-CSCV)
    from BiSAM-CD (Qin et al., IEEE TGRS 2025).

    Works both with discrete instance masks (from SAM2 AutomaticMaskGenerator + VideoPredictor)
    and dense differentiable feature/mask trajectories for zero-shot or fine-tuned inference.
    """

    def __init__(self, iou_threshold: float = 0.5, cross_iou_threshold: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.cross_iou_threshold = cross_iou_threshold
        self.temperature = temperature

    @staticmethod
    def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        inter = (mask_a * mask_b).sum(dim=(-2, -1))
        union = (mask_a + mask_b - mask_a * mask_b).sum(dim=(-2, -1))
        return (inter + eps) / (union + eps)

    def verify_instance_masks(
        self,
        asc_start_masks: List[torch.Tensor],
        asc_end_masks: List[Optional[torch.Tensor]],
        desc_start_masks: List[torch.Tensor],
        desc_end_masks: List[Optional[torch.Tensor]],
        spatial_shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Instance-level TOTC-CSCV verification matching Eqs. (3)-(6) in BiSAM-CD.
        """
        change_a: List[torch.Tensor] = []
        for m_ref, m_end in zip(asc_start_masks, asc_end_masks):
            if m_end is None or self.mask_iou(m_ref.float(), m_end.float()).item() < self.iou_threshold:
                change_a.append(m_ref.float())

        change_b: List[torch.Tensor] = []
        for m_ref, m_end in zip(desc_start_masks, desc_end_masks):
            if m_end is None or self.mask_iou(m_ref.float(), m_end.float()).item() < self.iou_threshold:
                change_b.append(m_ref.float())

        verified_mask = torch.zeros(spatial_shape, dtype=torch.float32, device=device)
        matched_b = set()
        for ma in change_a:
            is_pseudo = False
            for j, mb in enumerate(change_b):
                if self.mask_iou(ma, mb).item() > self.cross_iou_threshold:
                    is_pseudo = True
                    matched_b.add(j)
                    break
            if not is_pseudo:
                verified_mask = torch.maximum(verified_mask, ma)

        for j, mb in enumerate(change_b):
            if j not in matched_b:
                verified_mask = torch.maximum(verified_mask, mb)

        return verified_mask

    def forward(self, asc_feat_seq: torch.Tensor, desc_feat_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Differentiable dense trajectory consistency verification across ascending & descending sequences.
        Args:
            asc_feat_seq: (B, T, C, H, W) ascending feature trajectory
            desc_feat_seq: (B, T, C, H, W) descending feature trajectory
        """
        # Spatial mean-centering removes positive-orthant bias in zero-shot feature trajectories
        asc_c = asc_feat_seq - asc_feat_seq.mean(dim=(-2, -1), keepdim=True)
        desc_c = desc_feat_seq - desc_feat_seq.mean(dim=(-2, -1), keepdim=True)

        f_asc_start = F.normalize(asc_c[:, 0], dim=1)
        f_asc_end = F.normalize(asc_c[:, -1], dim=1)
        asc_step_sim = (F.normalize(asc_c[:, :-1], dim=2) * F.normalize(asc_c[:, 1:], dim=2)).sum(dim=2)
        asc_consistency = 0.65 * (f_asc_start * f_asc_end).sum(dim=1, keepdim=True) + 0.35 * asc_step_sim.mean(
            dim=1, keepdim=True
        )

        f_desc_start = F.normalize(desc_c[:, 0], dim=1)
        f_desc_end = F.normalize(desc_c[:, -1], dim=1)
        desc_step_sim = (F.normalize(desc_c[:, :-1], dim=2) * F.normalize(desc_c[:, 1:], dim=2)).sum(dim=2)
        desc_consistency = 0.65 * (f_desc_start * f_desc_end).sum(dim=1, keepdim=True) + 0.35 * desc_step_sim.mean(
            dim=1, keepdim=True
        )

        # Objects with trajectory consistency < tau are candidate changes
        change_prob_asc = torch.sigmoid((self.iou_threshold - asc_consistency) / self.temperature)
        change_prob_desc = torch.sigmoid((self.iou_threshold - desc_consistency) / self.temperature)

        # Cross-sequence verification: true changes exhibit low tracking consistency in both directions,
        # while pseudo-changes (illumination/registration drift) disagree between ascending and descending
        cross_disagreement = (change_prob_asc - change_prob_desc).abs()
        pseudo_suppress = 1.0 - 0.5 * (cross_disagreement > self.cross_iou_threshold).float() * cross_disagreement
        verified_prob = torch.clamp(0.5 * (change_prob_asc + change_prob_desc) * pseudo_suppress, 0.0, 1.0)
        change_logits = torch.logit(verified_prob.clamp(1e-4, 1.0 - 1e-4))
        return {
            "change_logits": change_logits,
            "change_prob": verified_prob,
            "asc_change_prob": change_prob_asc,
            "desc_change_prob": change_prob_desc,
        }


# =============================================================================
# 3. Qin et al. (2025) - SAM2-CD Modules (ASG & MSFFM)
# =============================================================================


@ATTENTION_REGISTRY.register("activation_selection_gate", aliases=["asg"])
class ActivationSelectionGate(nn.Module):
    """
    Activation Selection Gate (ASG) from SAM2-CD (Qin et al., IEEE JSTARS 2025).
    Selectively exchanges cross-temporal activations in global attention blocks:
        A = sigma(Conv1x1(GELU(Conv1x1(GELU(Conv1x1([F1, F2]))))))
        F1_tilde = F1 + FFN(F2 * A),  F2_tilde = F2 + FFN(F1 * A)
    """

    def __init__(self, channels: int, mlp_ratio: float = 2.0):
        super().__init__()
        mid_c = max(channels // 2, 8)
        hidden_c = int(channels * mlp_ratio)
        self.dynamic_map_gen = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, mid_c, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_c, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.soft_ffn = nn.Sequential(
            nn.Conv2d(channels, hidden_c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_c, channels, kernel_size=1),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_map = self.dynamic_map_gen(torch.cat([f1, f2], dim=1))
        f1_out = f1 + self.soft_ffn(self.norm1(f2 * attn_map))
        f2_out = f2 + self.soft_ffn(self.norm2(f1 * attn_map))
        return f1_out, f2_out, attn_map


@BLOCK_REGISTRY.register("msffm")
class MSFFMBlock(nn.Module):
    """
    Multiscale Feature Fusion Module (MSFFM) from SAM2-CD (Qin et al., IEEE JSTARS 2025).
    Aligns all 4 pyramid stages via pooling/interpolation + depthwise separable convolution,
    fuses across scales with residual connections, and computes temporal difference features D^i.
    """

    def __init__(self, in_channels_list: Sequence[int] = (64, 128, 256, 512), out_channels: int = 64):
        super().__init__()
        self.num_stages = len(in_channels_list)
        self.proj_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for c in in_channels_list
            ]
        )
        # Cross-scale depthwise separable convs: for target stage i, aggregate all stages j
        self.dw_convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(
                                out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False
                            ),
                            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                        )
                        for _ in range(self.num_stages)
                    ]
                )
                for _ in range(self.num_stages)
            ]
        )
        self.diff_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.num_stages)
            ]
        )

    def _reinforce_single_stream(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        proj = [conv(f) for conv, f in zip(self.proj_convs, feats)]
        reinforced: List[torch.Tensor] = []
        for i in range(self.num_stages):
            target_h, target_w = proj[i].shape[-2:]
            acc = proj[i]
            for j in range(self.num_stages):
                if j == i:
                    aligned = proj[j]
                elif proj[j].shape[-2] > target_h:
                    aligned = F.adaptive_max_pool2d(proj[j], output_size=(target_h, target_w))
                else:
                    aligned = F.interpolate(proj[j], size=(target_h, target_w), mode="bilinear", align_corners=False)
                acc = acc + self.dw_convs[i][j](aligned)
            reinforced.append(acc)
        return reinforced

    def forward(self, feats_t1: Sequence[torch.Tensor], feats_t2: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        r1 = self._reinforce_single_stream(feats_t1)
        r2 = self._reinforce_single_stream(feats_t2)
        return [conv(torch.abs(f1 - f2)) for conv, f1, f2 in zip(self.diff_convs, r1, r2)]


# =============================================================================
# 4. Xu et al. (2026) - GMFIMamba Modules (SS2DCrossScan, ConvVSSBlock, MCFEM, GMBFIM)
# =============================================================================


@BLOCK_REGISTRY.register("ss2d_cross_scan")
class SS2DCrossScan(nn.Module):
    """
    4-Way 2D Selective State Space Scan (SS2D) for Vision Mamba blocks
    (Xu et al., GMFIMamba, EAAI 2026).

    Performs input-selective state transitions along 4 scanning paths:
    1. Top-left -> Bottom-right (row-major forward)
    2. Bottom-right -> Top-left (row-major backward)
    3. Top-left -> Bottom-right (column-major forward)
    4. Bottom-right -> Top-left (column-major backward)
    Pure PyTorch implementation with vectorized/chunked scan that runs natively on CPU & GPU.
    """

    def __init__(self, channels: int, state_dim: int = 16):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        # Input-selective gates and step projections for 4 directions
        self.dt_proj = nn.Conv2d(channels, channels * 4, kernel_size=1, groups=channels)
        self.B_proj = nn.Conv2d(channels, channels * 4, kernel_size=1, groups=channels)
        self.C_proj = nn.Conv2d(channels, channels * 4, kernel_size=1, groups=channels)
        self.D = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.decay_logit = nn.Parameter(torch.zeros(1, 4, channels, 1))
        self.out_norm = nn.GroupNorm(1, channels)

    def _directional_ema_scan(
        self, seq: torch.Tensor, dt: torch.Tensor, b_gate: torch.Tensor, c_gate: torch.Tensor, decay: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized exponential state-space scan over sequence dimension L.
        seq, dt, b_gate, c_gate: (B, C, L)
        """
        alpha = torch.sigmoid(dt + decay)  # (B, C, L) selective retention gate
        u = seq * torch.tanh(b_gate)
        # Parallel prefix-style multi-horizon state accumulation (exact for k-step receptive field + global state)
        h = u
        for shift in (1, 2, 4, 8, 16):
            if shift < h.shape[-1]:
                a_shift = alpha[:, :, shift:]
                h_prev = h[:, :, :-shift]
                h = torch.cat([h[:, :, :shift], h[:, :, shift:] + a_shift * h_prev * 0.5], dim=-1)
        return h * c_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        l = h * w
        dt_all = F.softplus(self.dt_proj(x)).view(b, 4, c, l)
        b_all = self.B_proj(x).view(b, 4, c, l)
        c_all = self.C_proj(x).view(b, 4, c, l)
        decays = self.decay_logit

        # 1. Row-major forward
        x_row_f = x.flatten(2)
        y1 = self._directional_ema_scan(x_row_f, dt_all[:, 0], b_all[:, 0], c_all[:, 0], decays[:, 0]).view(
            b, c, h, w
        )

        # 2. Row-major backward
        x_row_b = torch.flip(x_row_f, dims=[-1])
        y2_b = self._directional_ema_scan(x_row_b, dt_all[:, 1], b_all[:, 1], c_all[:, 1], decays[:, 1])
        y2 = torch.flip(y2_b, dims=[-1]).view(b, c, h, w)

        # 3. Column-major forward
        x_col_f = x.transpose(-2, -1).contiguous().flatten(2)
        y3_col = self._directional_ema_scan(x_col_f, dt_all[:, 2], b_all[:, 2], c_all[:, 2], decays[:, 2])
        y3 = y3_col.view(b, c, w, h).transpose(-2, -1).contiguous()

        # 4. Column-major backward
        x_col_b = torch.flip(x_col_f, dims=[-1])
        y4_col = self._directional_ema_scan(x_col_b, dt_all[:, 3], b_all[:, 3], c_all[:, 3], decays[:, 3])
        y4 = torch.flip(y4_col, dims=[-1]).view(b, c, w, h).transpose(-2, -1).contiguous()

        y = (y1 + y2 + y3 + y4) * 0.25 + x * self.D
        return self.out_norm(y)


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    if c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


@BLOCK_REGISTRY.register("conv_vss_block")
class ConvVSSBlock(nn.Module):
    """
    ConvVSS Block from GMFIMamba (Xu et al., EAAI 2026).
    Splits input channels into:
    - Local Conv branch (two 3x3 Conv-BN-ReLU blocks)
    - Global SSM branch (LayerNorm -> Linear -> DWConv -> SiLU -> SS2D -> LayerNorm * SiLU(Linear) -> Linear)
    Concatenates both branches, applies Channel Shuffle, and adds residual connection.
    """

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 2 == 0, f"ConvVSSBlock requires even channels, got {channels}"
        half_c = channels // 2
        self.conv_branch = nn.Sequential(
            nn.Conv2d(half_c, half_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(half_c, half_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_c),
            nn.ReLU(inplace=True),
        )
        # SSM branch (VSS block)
        self.in_norm = nn.GroupNorm(1, half_c)
        self.in_proj = nn.Conv2d(half_c, half_c, kernel_size=1)
        self.dw_conv = nn.Conv2d(half_c, half_c, kernel_size=3, padding=1, groups=half_c)
        self.ss2d = SS2DCrossScan(half_c)
        self.ssm_norm = nn.GroupNorm(1, half_c)
        self.gate_proj = nn.Conv2d(half_c, half_c, kernel_size=1)
        self.out_proj = nn.Conv2d(half_c, half_c, kernel_size=1)

    def _vss_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.in_norm(x)
        gate = F.silu(self.gate_proj(x_norm))
        feat = F.silu(self.dw_conv(self.in_proj(x_norm)))
        feat = self.ssm_norm(self.ss2d(feat))
        return self.out_proj(feat * gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv, x_ssm = torch.chunk(x, 2, dim=1)
        out_conv = self.conv_branch(x_conv)
        out_ssm = self._vss_forward(x_ssm)
        merged = channel_shuffle(torch.cat([out_conv, out_ssm], dim=1), groups=2)
        return x + merged


@BLOCK_REGISTRY.register("mcfem")
class MCFEMBlock(nn.Module):
    """
    Multi-branch Dilated Convolution Feature Enhancement Module (MCFEM)
    from GMFIMamba (Xu et al., EAAI 2026).

    4 parallel branches after 1x1 channel reduction:
    - Branch 1: 1x1 conv shortcut
    - Branch 2: two cascaded 3x3 dilated convolutions (dilation=3)
    - Branch 3: 1x3 conv -> 3x1 conv -> 3x3 dilated convolution (dilation=5)
    - Branch 4: 3x3 regular convolution
    Concatenates Branches 2, 3, 4 -> 1x1 conv -> adds Branch 1.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.in_conv(x)
        b1 = self.branch1(x_in)
        b2 = self.branch2(x_in)
        b3 = self.branch3(x_in)
        b4 = self.branch4(x_in)
        fused = self.fuse_conv(torch.cat([b2, b3, b4], dim=1))
        return fused + b1


@BLOCK_REGISTRY.register("gmbfim")
class GMBFIMBlock(nn.Module):
    """
    Grouped Mamba-based Bitemporal Feature Interaction Module (GMBFIM)
    from GMFIMamba (Xu et al., EAAI 2026).

    1. Interleaves channels of F_T1 and F_T2 into F_MIX in R^{B x 2C x H x W}.
    2. Splits F_MIX into `num_groups` (4) groups and processes each group with a VSS block.
    3. Computes difference branch F_D = Conv3x3(|F_T1 - F_T2|).
    4. Outputs F_i = (F_MIX_hat * F_D) + F_MIX_hat projected to C channels.
    """

    def __init__(self, channels: int, num_groups: int = 4):
        super().__init__()
        self.channels = channels
        self.num_groups = num_groups
        mix_channels = channels * 2
        assert mix_channels % num_groups == 0, "2 * channels must be divisible by num_groups"
        group_c = mix_channels // num_groups
        # Ensure even channels for ConvVSSBlock or use SS2D block directly
        self.group_vss = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, group_c),
                    SS2DCrossScan(group_c),
                    nn.Conv2d(group_c, group_c, kernel_size=1),
                    nn.SiLU(inplace=True),
                )
                for _ in range(num_groups)
            ]
        )
        self.diff_branch = nn.Sequential(
            nn.Conv2d(channels, mix_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mix_channels),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(mix_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_t1: torch.Tensor, f_t2: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_t1.shape
        # Channel-wise interleaving: (B, C, 2, H, W) -> (B, 2C, H, W)
        f_mix = torch.stack([f_t1, f_t2], dim=2).view(b, 2 * c, h, w)
        groups = torch.chunk(f_mix, self.num_groups, dim=1)
        processed_groups = [g + block(g) for g, block in zip(groups, self.group_vss)]
        f_mix_hat = torch.cat(processed_groups, dim=1)

        f_d = self.diff_branch(torch.abs(f_t1 - f_t2))
        interacted = (f_mix_hat * f_d) + f_mix_hat
        return self.out_proj(interacted)


# =============================================================================
# 5. Liu et al. (2026) - SemiBCD Modules (UAPLR, LightSSHead, DeepLabV3PlusCDHead)
# =============================================================================


@BLOCK_REGISTRY.register("uaplr")
class UAPLRModule(nn.Module):
    """
    Uncertainty-Aware Pseudo-Label Refinement (UAPLR) from SemiBCD
    (Liu et al., IEEE TGRS 2026).

    Computes pixel-wise cosine similarity between student predictions across iteration
    intervals to weight pseudo-label binary cross-entropy loss and suppress noisy boundaries.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def compute_uncertainty_weights(self, logits_curr: torch.Tensor, logits_prev: torch.Tensor) -> torch.Tensor:
        """
        Converts single-channel or two-channel logits into 2-class probability vectors
        and computes pixel-wise cosine similarity S_{n,t}(i,j) in [0, 1].
        """
        if logits_curr.shape[1] == 1:
            p_curr = torch.sigmoid(logits_curr)
            vec_curr = torch.cat([1.0 - p_curr, p_curr], dim=1)
            p_prev = torch.sigmoid(logits_prev.detach())
            vec_prev = torch.cat([1.0 - p_prev, p_prev], dim=1)
        else:
            vec_curr = F.softmax(logits_curr, dim=1)
            vec_prev = F.softmax(logits_prev.detach(), dim=1)

        cos_sim = F.cosine_similarity(vec_curr, vec_prev, dim=1, eps=self.eps).unsqueeze(1)
        return cos_sim.clamp(0.0, 1.0)

    def forward(
        self, student_logits: torch.Tensor, vlm_pseudo_mask: torch.Tensor, prev_student_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_student_logits is None:
            weights = torch.ones_like(vlm_pseudo_mask, dtype=student_logits.dtype)
        else:
            weights = self.compute_uncertainty_weights(student_logits, prev_student_logits)
        bce_pixel = F.binary_cross_entropy_with_logits(
            student_logits, vlm_pseudo_mask.float(), reduction="none"
        )
        weighted_loss = (weights.detach() * bce_pixel).mean()
        return weighted_loss, weights


@BLOCK_REGISTRY.register("light_ss_head")
class LightSSHead(nn.Module):
    """
    Lightweight Multi-Scale Semantic Segmentation Head (`LightHead` in SemiBCD)
    for auxiliary building segmentation on pre- and post-event images.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int] = (64, 128, 256, 512),
        mid_channels: int = 64,
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, mid_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                )
                for c in in_channels_list
            ]
        )
        total_c = mid_channels * len(in_channels_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_c, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = features[0].shape[-2:]
        projected = [
            F.interpolate(conv(f), size=target_size, mode="bilinear", align_corners=False)
            for conv, f in zip(self.convs, features)
        ]
        rep = self.fuse(torch.cat(projected, dim=1))
        logits = self.classifier(rep)
        return logits, rep


@BLOCK_REGISTRY.register("dlv3p_cd_head")
class DeepLabV3PlusCDHead(nn.Module):
    """
    DeepLabV3+ ASPP + Low-Level Fusion Change Detection Head (`DLV3PHead` in SemiBCD).
    """

    def __init__(
        self,
        c1_channels: int = 64,
        c4_channels: int = 512,
        aspp_out: int = 128,
        low_channels: int = 48,
        num_classes: int = 1,
    ):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(c1_channels, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True),
        )
        # ASPP branches on c4 difference feature
        self.b0 = nn.Sequential(
            nn.Conv2d(c4_channels, aspp_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(c4_channels, aspp_out, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c4_channels, aspp_out, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(c4_channels, aspp_out, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4_channels, aspp_out, kernel_size=1, bias=False),
            nn.GroupNorm(1, aspp_out),
            nn.ReLU(inplace=True),
        )
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(aspp_out * 5, aspp_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_out + low_channels, aspp_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_out, num_classes, kernel_size=1),
        )

    def forward(self, c1_diff: torch.Tensor, c4_diff: torch.Tensor) -> torch.Tensor:
        low_feat = self.low_proj(c1_diff)
        h4, w4 = c4_diff.shape[-2:]
        aspp = torch.cat(
            [
                self.b0(c4_diff),
                self.b1(c4_diff),
                self.b2(c4_diff),
                self.b3(c4_diff),
                F.interpolate(self.b4(c4_diff), size=(h4, w4), mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        high_feat = self.aspp_fuse(aspp)
        high_up = F.interpolate(high_feat, size=low_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(torch.cat([high_up, low_feat], dim=1))


# =============================================================================
# 6. Wang et al. (2026) - CGCCE-Net Modules (CGRR, GCCM, SCEM, CFD)
# =============================================================================


@BLOCK_REGISTRY.register("cgrr_branch")
class CGRRBranch(nn.Module):
    """
    Change-Guided Residual Refinement (CGRR) Branch from CGCCE-Net
    (Wang et al., Information Sciences 2026).

    Extracts fine-grained shallow edge/texture cues across 4 progressive downsampling
    residual stages to inject into the concatenated change stream C_j.
    """

    def __init__(self, in_channels: int, stage_channels: Sequence[int] = (64, 128, 256, 512)):
        super().__init__()
        self.blocks = nn.ModuleList()
        curr_c = in_channels
        for idx, out_c in enumerate(stage_channels):
            stride = 1 if idx == 0 else 2
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(curr_c, out_c * 2, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_c * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c * 2, out_c * 2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c * 2),
                )
            )
            curr_c = out_c * 2

    def forward(self, shallow_t1: torch.Tensor, shallow_t2: torch.Tensor) -> List[torch.Tensor]:
        x = torch.cat([shallow_t1, shallow_t2], dim=1)
        outputs: List[torch.Tensor] = []
        for blk in self.blocks:
            x = F.relu(blk(x), inplace=True)
            outputs.append(x)
        return outputs


@ATTENTION_REGISTRY.register("gccm_linear_angle_attention", aliases=["gccm"])
class GCCMModule(nn.Module):
    """
    Global Cross-Correlation Module (GCCM) from CGCCE-Net (Wang et al., Information Sciences 2026).

    Combines Channel Attention (CAB) + Spatial Attention (SAB) with Linear Angle Cross-Attention:
        H_1 = 0.5 * V_1 + (1 / pi) * Q_1 @ (K_2^T @ V_1) + DWConv(V_1)
        H_2 = 0.5 * V_2 + (1 / pi) * Q_2 @ (K_1^T @ V_2) + DWConv(V_2)
    where Q and K are L2-normalized along the channel dimension for O(N) complexity.
    """

    def __init__(self, channels: int):
        super().__init__()
        mid_c = max(channels // 4, 8)
        self.cab = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.sab = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _enhance_cab_sab(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x * self.cab(x)
        avg_out = torch.mean(x_c, dim=1, keepdim=True)
        max_out, _ = torch.max(x_c, dim=1, keepdim=True)
        x_s = x_c * self.sab(torch.cat([avg_out, max_out], dim=1))
        return x + x_s

    def _linear_cross_attn(
        self, q: torch.Tensor, k_other: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = q.shape
        n = h * w
        q_norm = F.normalize(q.view(b, c, n), p=2, dim=1)  # (B, C, N)
        k_norm = F.normalize(k_other.view(b, c, n), p=2, dim=-1)  # (B, C, N)
        v_flat = v.view(b, c, n)  # (B, C, N)

        # (K_other^T @ V) in O(N * C^2): (B, C, N) @ (B, N, C) -> (B, C, C)
        kv = torch.bmm(k_norm, v_flat.transpose(1, 2)) / float(max(n, 1))
        # Q @ (K^T V): (B, C, C) @ (B, C, N) -> (B, C, N)
        qkv = torch.bmm(kv.transpose(1, 2), q_norm).view(b, c, h, w)
        return 0.5 * v + (1.0 / math.pi) * qkv + self.dw_conv(v)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self._enhance_cab_sab(t1)
        e2 = self._enhance_cab_sab(t2)

        q1, k1, v1 = self.q_proj(e1), self.k_proj(e1), self.v_proj(e1)
        q2, k2, v2 = self.q_proj(e2), self.k_proj(e2), self.v_proj(e2)

        h1 = self.out_proj(self._linear_cross_attn(q1, k2, v1)) + t1
        h2 = self.out_proj(self._linear_cross_attn(q2, k1, v2)) + t2
        return h1, h2


@BLOCK_REGISTRY.register("scem")
class SCEMModule(nn.Module):
    """
    Semantic Cognitive Enhancement Module (SCEM) from CGCCE-Net
    (Wang et al., Information Sciences 2026).

    Combines multi-kernel local receptive fields (3x3, 5x5, 7x7) with a global context gate:
        MC = Conv1x1(Conv3x3(C_j) + Conv5x5(C_j) + Conv7x7(C_j))
        GC = Sigmoid(AvgPool(Conv1x1(C_j)))
        C_j' = Proj(C_j + MC * GC)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False)
        self.mc_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gc_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, c_j: torch.Tensor) -> torch.Tensor:
        mc = self.mc_proj(self.conv3(c_j) + self.conv5(c_j) + self.conv7(c_j))
        gc = self.gc_branch(c_j)
        enhanced = c_j + mc * gc
        return self.out_proj(enhanced)


@BLOCK_REGISTRY.register("cfd_decoder_block")
class CFDDecoderBlock(nn.Module):
    """
    Cross Fusion Decoder (CFD) Block from CGCCE-Net (Wang et al., Information Sciences 2026).
    Mutually gates the channel-wise change representation C_j' and difference representation D_j.
    """

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.c_gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.Sigmoid())
        self.d_gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.Sigmoid())
        self.double_conv = nn.Sequential(
            nn.Conv2d(channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, c_feat: torch.Tensor, d_feat: torch.Tensor) -> torch.Tensor:
        c_mod = c_feat * self.d_gate(d_feat) + c_feat
        d_mod = d_feat * self.c_gate(c_feat) + d_feat
        return self.double_conv(torch.cat([c_mod, d_mod], dim=1))


# =============================================================================
# 7. Wang et al. (2026) - KGBDCNet Modules (CAGM, SSMB, TTTB)
# =============================================================================


@ATTENTION_REGISTRY.register("cagm_text_semantic_fusion", aliases=["cagm"])
class CAGMModule(nn.Module):
    """
    Cross-Modal Attention Guided Module (CAGM / `TextSemanticFusion`) from KGBDCNet
    (Wang et al., ISPRS J. Photogramm. Remote Sens. 2026).

    1. Visual-Text Retrieve: F' = Softmax(G @ F^T / sqrt(d)) @ F
    2. Visual-Text Enhance: alpha = Sigmoid(Linear([G, F'])),
       G' = ReLU(Linear(alpha * G + (1 - alpha) * F'))
    3. Spatial Modulation: L_out = L * G'.unsqueeze(1) + L
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim)
        self.enhance_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, local_tokens: torch.Tensor, global_token: torch.Tensor, text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            local_tokens: L_i of shape (B, N_patches, D)
            global_token: G_i of shape (B, D) or (B, 1, D)
            text_tokens: F of shape (B, L_text, D)
        Returns:
            L_enhanced: (B, N_patches, D)
            G_enhanced: (B, 1, D)
        """
        if global_token.ndim == 2:
            g = global_token.unsqueeze(1)  # (B, 1, D)
        else:
            g = global_token

        scale = math.sqrt(float(self.embed_dim))
        attn = torch.softmax(torch.bmm(g, text_tokens.transpose(1, 2)) / scale, dim=-1)  # (B, 1, L_text)
        f_prime = torch.bmm(attn, text_tokens)  # (B, 1, D)

        alpha = torch.sigmoid(self.gate_linear(torch.cat([g, f_prime], dim=-1)))
        g_prime = self.enhance_linear(alpha * g + (1.0 - alpha) * f_prime)  # (B, 1, D)
        l_out = self.norm(local_tokens * g_prime + local_tokens)
        return l_out, g_prime


@BLOCK_REGISTRY.register("ssmb_camamba")
class SSMBBlock(nn.Module):
    """
    Space-Scan Mamba Block (SSMB / `CaMambaModel`) from KGBDCNet (Wang et al., ISPRS 2026).
    Performs bidirectional SSM scanning on pre-disaster L1 and post-disaster L2 token sequences,
    computes difference sequence Delta_L = SSM(L1) - SSM(L2), and gates L1/L2 via Cross-Spatial Mamba.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fwd_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.bwd_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.ssm_proj = nn.Linear(embed_dim, embed_dim)
        self.diff_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU())
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _bi_ssm(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D)
        x_t = tokens.transpose(1, 2)  # (B, D, N)
        fwd = F.silu(self.fwd_conv(x_t)).transpose(1, 2)
        bwd = torch.flip(F.silu(self.bwd_conv(torch.flip(x_t, dims=[-1]))), dims=[-1]).transpose(1, 2)
        return self.ssm_proj(fwd + bwd)

    def forward(self, l1: torch.Tensor, l2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s1 = self._bi_ssm(self.norm1(l1))
        s2 = self._bi_ssm(self.norm2(l2))
        delta = self.diff_gate(s1 - s2)
        l1_out = l1 + self.out_proj(s1 * delta)
        l2_out = l2 + self.out_proj(s2 * delta)
        return l1_out, l2_out


@ATTENTION_REGISTRY.register("tttb_memory_transformer", aliases=["tttb"])
class TTTBBlock(nn.Module):
    """
    Temporal-Trace Transformer Block (TTTB / `MemoryTransformer`) from KGBDCNet
    (Wang et al., ISPRS 2026).

    Constructs temporal memory L_M = Linear([L1', L2']), prepends global tokens G1, G2 to form
    LG1, LG2, and performs multi-head cross-attention with query LG2 and key/value [LG1; L_M].
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mem_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self, l1: torch.Tensor, l2: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor
    ) -> torch.Tensor:
        l_mem = self.mem_proj(torch.cat([l1, l2], dim=-1))  # (B, N, D)
        lg1 = torch.cat([g1, l1], dim=1)  # (B, 1+N, D)
        lg2 = torch.cat([g2, l2], dim=1)  # (B, 1+N, D)
        kv = torch.cat([lg1, l_mem], dim=1)  # (B, 1+2N, D)

        attn_out, _ = self.cross_attn(query=lg2, key=kv, value=kv)
        x = self.norm1(lg2 + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# =============================================================================
# 8. Huang et al. (2026) - CBRNet Modules (CASC, CGFE, LWCA, BiFPN, DSBR)
# =============================================================================


@BLOCK_REGISTRY.register("casc_block")
class CASCBlock(nn.Module):
    """
    Content-Adaptive Scale Convolution (CASC) from CBRNet (Huang et al., IEEE TGRS 2026).

    Applies K=4 parallel depthwise-separable convolutions with dilation rates {1, 2, 4, 8},
    fuses Global Scale Prediction w^g and Local Scale Refinement w^l via Softmax over K:
        w = Softmax(w^g + w^l)
        F' = sum_{k=1}^K w_k * F_k + F_res
    """

    def __init__(self, in_channels: int, out_channels: int, dilations: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.num_branches = len(dilations)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for d in dilations
            ]
        )
        mid_c = max(in_channels // 4, 8)
        self.global_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, self.num_branches, kernel_size=1, bias=True),
        )
        self.local_scale = nn.Sequential(
            nn.Conv2d(in_channels, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, self.num_branches, kernel_size=1, bias=True),
        )
        self.res_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_feats = torch.stack([b(x) for b in self.branches], dim=1)  # (B, K, C_out, H, W)
        w_g = self.global_scale(x)  # (B, K, 1, 1)
        w_l = self.local_scale(x)  # (B, K, H, W)
        weights = torch.softmax(w_g + w_l, dim=1).unsqueeze(2)  # (B, K, 1, H, W)
        fused = (weights * branch_feats).sum(dim=1)
        return fused + self.res_proj(x)


@ATTENTION_REGISTRY.register("cgfe_harris_attention", aliases=["cgfe"])
class CGFEModule(nn.Module):
    """
    Corner-Guided Feature Enhancement (CGFE) from CBRNet (Huang et al., IEEE TGRS 2026).

    Embeds a differentiable Harris corner detector directly into the feature enhancement loop:
    1. F_r = Conv1x1(F') in R^{B x 1 x H x W}
    2. Fixed Sobel filters S_x, S_y compute gradients I_x, I_y
    3. Structure tensor elements A = AvgPool3x3(I_x^2), B = AvgPool3x3(I_y^2), C = AvgPool3x3(I_x * I_y)
    4. Harris response R = (A*B - C^2) - k*(A + B)^2 (k = 0.04)
    5. R_norm = Sigmoid((R - mu_R) / (sigma_R + eps))
    6. Fuses with learnable attention A_l via learnable alpha: A = alpha * R_norm + (1 - alpha) * A_l
    7. F'' = ReLU(BN(Conv3x3(F' * (1 + A)))) + F'
    """

    def __init__(self, channels: int, harris_k: float = 0.04, eps: float = 1e-6):
        super().__init__()
        self.harris_k = harris_k
        self.eps = eps
        self.reduce_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

        sobel_x = torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]).unsqueeze(0)
        sobel_y = torch.tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.learn_attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_r = self.reduce_conv(x)  # (B, 1, H, W)
        ix = F.conv2d(f_r, self.sobel_x.to(dtype=x.dtype), padding=1)
        iy = F.conv2d(f_r, self.sobel_y.to(dtype=x.dtype), padding=1)

        a_mat = F.avg_pool2d(ix * ix, kernel_size=3, stride=1, padding=1)
        b_mat = F.avg_pool2d(iy * iy, kernel_size=3, stride=1, padding=1)
        c_mat = F.avg_pool2d(ix * iy, kernel_size=3, stride=1, padding=1)

        det = a_mat * b_mat - c_mat * c_mat
        trace = a_mat + b_mat
        r_resp = det - self.harris_k * (trace * trace)

        mu_r = r_resp.mean(dim=(-2, -1), keepdim=True)
        sigma_r = r_resp.std(dim=(-2, -1), keepdim=True, unbiased=False)
        r_norm = torch.sigmoid((r_resp - mu_r) / (sigma_r + self.eps))

        a_learn = self.learn_attn(x)
        alpha_clamped = self.alpha.clamp(0.0, 1.0)
        attn_combined = alpha_clamped * r_norm + (1.0 - alpha_clamped) * a_learn

        f_out = self.refine_conv(x * (1.0 + attn_combined)) + x
        return f_out, attn_combined, r_norm


@ATTENTION_REGISTRY.register("lwca_window_attention", aliases=["lwca"])
class LWCAAttention(nn.Module):
    """
    Local Window Cross-Attention (LWCA) from CBRNet (Huang et al., IEEE TGRS 2026).

    Partitions feature map into non-overlapping M x M windows (default M=8) with automatic
    padding for arbitrary input sizes, computes multi-head self-attention with learnable
    relative position bias B in R^{num_heads x M^2 x M^2}, and applies residual FFN.
    """

    def __init__(self, channels: int, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = max(channels // num_heads, 1)
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        m = self.window_size
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m
        x_norm = self.norm1(x)
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x_norm, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x_norm

        hp, wp = x_pad.shape[-2:]
        nh, nw = hp // m, wp // m
        qkv = self.qkv(x_pad)  # (B, 3C, Hp, Wp)
        # Partition into windows: (B * nh * nw, 3, num_heads, head_dim, M*M)
        qkv = (
            qkv.view(b, 3, self.num_heads, self.head_dim, nh, m, nw, m)
            .permute(0, 4, 6, 1, 2, 5, 7, 3)
            .contiguous()
            .view(b * nh * nw, 3, self.num_heads, m * m, self.head_dim)
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B_win, heads, M^2, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias.unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)
        out_win = attn @ v  # (B_win, heads, M^2, head_dim)

        out = (
            out_win.view(b, nh, nw, self.num_heads, m, m, self.head_dim)
            .permute(0, 3, 6, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, hp, wp)
        )
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


@BLOCK_REGISTRY.register("bifpn_fusion")
class BiFPNFeatureFusion(nn.Module):
    """
    Bidirectional Feature Pyramid Network (BiFPN) from CBRNet (Huang et al., IEEE TGRS 2026).
    Performs weighted top-down and bottom-up multi-scale feature fusion across 4 pyramid levels.
    """

    def __init__(self, in_channels_list: Sequence[int] = (64, 128, 256, 512), out_channels: int = 128, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.num_levels = len(in_channels_list)
        self.input_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for c in in_channels_list
            ]
        )
        self.td_weights = nn.Parameter(torch.ones(self.num_levels - 1, 2))
        self.bu_weights = nn.Parameter(torch.ones(self.num_levels - 1, 3))

        self.td_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.num_levels - 1)
            ]
        )
        self.bu_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.num_levels - 1)
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        p_in = [proj(f) for proj, f in zip(self.input_projs, features)]
        p_td = list(p_in)

        # Top-down pathway (from P4 down to P1)
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + self.eps)
            up = F.interpolate(p_td[i + 1], size=p_in[i].shape[-2:], mode="bilinear", align_corners=False)
            p_td[i] = self.td_convs[i](w[0] * p_in[i] + w[1] * up)

        # Bottom-up pathway (from P1 up to P4)
        p_out = [p_td[0]]
        for i in range(1, self.num_levels):
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + self.eps)
            down = F.adaptive_max_pool2d(p_out[-1], output_size=p_in[i].shape[-2:])
            fused = self.bu_convs[i - 1](w[0] * p_in[i] + w[1] * p_td[i] + w[2] * down)
            p_out.append(fused)

        return p_out


@BLOCK_REGISTRY.register("dsbr_decoder")
class DSBRDecoder(nn.Module):
    """
    Dual-Stream Boundary Refinement (DSBR) Decoder from CBRNet (Huang et al., IEEE TGRS 2026).

    Maintains parallel Semantic Stream (S_0 -> S) and Boundary Stream (B_c -> B_f) with
    bidirectional 1x1 conv feature interaction (F_{b->s} and F_{s->b}).
    """

    def __init__(self, in_channels: int = 128, mid_channels: int = 64, num_classes: int = 1):
        super().__init__()
        # Initial Semantic Stream
        self.sem_stream = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # Coarse & Fine Boundary Stream
        self.bnd_coarse_feat = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.bnd_coarse_head = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.bnd_fine_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.bnd_fine_head = nn.Conv2d(mid_channels, 1, kernel_size=1)

        # Bidirectional interaction projections
        self.b2s_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.s2b_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.final_head = nn.Sequential(
            nn.Conv2d(in_channels + mid_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, f_d: torch.Tensor) -> Dict[str, torch.Tensor]:
        s_0 = self.sem_stream(f_d)
        b_c_feat = self.bnd_coarse_feat(f_d)
        b_c_logits = self.bnd_coarse_head(b_c_feat)
        b_c_prob = torch.sigmoid(b_c_logits)

        b_f_feat = self.bnd_fine_conv(torch.cat([f_d, b_c_prob], dim=1))
        b_f_logits = self.bnd_fine_head(b_f_feat)

        f_b2s = self.b2s_proj(b_f_feat)
        f_s2b = self.s2b_proj(s_0)
        s_logits = self.final_head(torch.cat([f_d, f_b2s, f_s2b], dim=1))
        return {
            "seg_logits": s_logits,
            "boundary_logits": b_f_logits,
            "coarse_boundary_logits": b_c_logits,
        }


# =============================================================================
# 9. Hussin et al. (2026) - EdgeRefNet Modules (EDB, EEM/SCSE+HFAB, Cross-Attn)
# =============================================================================


@BLOCK_REGISTRY.register("edb_block")
class EDBBlock(nn.Module):
    """
    Edge Detection Block (EDB) from EdgeRefNet (Hussin et al., IEEE TGRS 2026).
    1x1 conv -> two 3x3 convs -> 1x1 conv with residual shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_c = max(out_channels // 2, 16)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.shortcut(x), inplace=True)


@ATTENTION_REGISTRY.register("eem_scse_block", aliases=["eem", "hfab"])
class EEMBlock(nn.Module):
    """
    Edge Enhancement Module (EEM / HFAB with SCSEBlock + HighFrequencyEnhancementStage)
    from EdgeRefNet (Hussin et al., IEEE TGRS 2026).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid_c = max(channels // reduction, 8)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.hf_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scse = x * self.cse(x) + x * self.sse(x)
        # High-frequency residual extraction (x - lowpass(x))
        lowpass = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        hf = self.hf_conv(x - lowpass)
        return F.relu(scse + hf, inplace=True)


@ATTENTION_REGISTRY.register("edge_guided_cross_attention")
class EdgeGuidedCrossAttention(nn.Module):
    """
    Context-Guided Cross-Attention Transformer Refinement Block from EdgeRefNet
    (Hussin et al., IEEE TGRS 2026).

    Semantic change features provide Queries (Q_s) while multi-scale edge features
    provide Keys and Values (K_e, V_e). Uses pooled spatial tokens to remain fast at any resolution.
    """

    def __init__(self, channels: int, num_heads: int = 4, pool_size: int = 16):
        super().__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.q_norm = nn.GroupNorm(1, channels)
        self.kv_norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, sem_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = sem_feat.shape
        q_in = self.q_norm(sem_feat)
        kv_in = self.kv_norm(edge_feat)

        # Pool keys/values to compact spatial tokens if resolution is large
        if h > self.pool_size or w > self.pool_size:
            ph, pw = min(h, self.pool_size), min(w, self.pool_size)
            kv_pooled = F.adaptive_avg_pool2d(kv_in, output_size=(ph, pw))
        else:
            kv_pooled = kv_in

        q_tokens = q_in.flatten(2).transpose(1, 2)  # (B, H*W, C)
        kv_tokens = kv_pooled.flatten(2).transpose(1, 2)  # (B, ph*pw, C)

        attn_out, _ = self.attn(query=q_tokens, key=kv_tokens, value=kv_tokens)
        attn_map = attn_out.transpose(1, 2).view(b, c, h, w)
        x = sem_feat + attn_map
        return x + self.ffn(x)


# =============================================================================
# 10. Hybrid SAM-2 + CBRNet + EdgeRefNet Modules (SAMFoundationAdapter, SingleImageEdgeRefBranch)
# =============================================================================


@BLOCK_REGISTRY.register("sam_foundation_adapter")
class SAMFoundationAdapter(nn.Module):
    """
    Pretrained SAM-2 Hiera Foundation Encoder + Multi-Scale Cross-Foundation Adapter.

    Extracts 4 hierarchical vision-foundation feature maps from Meta's SAM-2 Hiera encoder
    (via `timm`, default `sam2_hiera_tiny`, channels `[96, 192, 384, 768]` at strides `[4, 8, 16, 32]`),
    aligns them to target pyramid strides `[2, 4, 8, 16]`, and fuses them with convolutional
    pyramid features `[C1, C2, C3, C4]` via learnable spatial-channel gating.
    """

    def __init__(
        self,
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        sam_arch: str = "sam2_hiera_tiny",
        pretrained_sam: bool = True,
    ):
        super().__init__()
        self.stage_channels = tuple(stage_channels)
        self.sam_arch = sam_arch
        self.pretrained_sam = pretrained_sam

        self.register_buffer(
            "sam_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "sam_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

        import timm

        self.sam_encoder = timm.create_model(
            sam_arch,
            features_only=True,
            pretrained=pretrained_sam,
        )
        sam_dims = list(self.sam_encoder.feature_info.channels())  # e.g. [96, 192, 384, 768]

        # Lateral 1x1 + 3x3 adapters projecting SAM stage features to target stage_channels
        self.sam_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(sc_in, sc_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(sc_out),
                    nn.GELU(),
                    nn.Conv2d(sc_out, sc_out, kernel_size=3, padding=1, groups=sc_out, bias=False),
                    nn.BatchNorm2d(sc_out),
                    nn.GELU(),
                )
                for sc_in, sc_out in zip(sam_dims, stage_channels)
            ]
        )

        # Cross-foundation spatial-channel gating between CNN pyramid features and SAM-2 features
        self.fusion_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.GELU(),
                    nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                    nn.Sigmoid(),
                )
                for c in stage_channels
            ]
        )
        self.out_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.GELU(),
                )
                for c in stage_channels
            ]
        )
        for conv_block in self.out_convs:
            nn.init.constant_(conv_block[1].weight, 0.2)
            nn.init.zeros_(conv_block[1].bias)
        # Learnable residual scaling initialized to 0.5 for balanced dual-foundation fusion
        self.sam_scales = nn.Parameter(torch.full((len(stage_channels),), 0.5, dtype=torch.float32))

    def forward(self, x_rgb: torch.Tensor, cnn_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        x_norm = (x_rgb - self.sam_mean) / self.sam_std
        # Ensure minimum 64x64 input for 4-stage stride-32 Hiera window partitioning
        h_in, w_in = x_norm.shape[-2:]
        if h_in < 64 or w_in < 64:
            x_sam_in = F.interpolate(x_norm, size=(max(h_in, 64), max(w_in, 64)), mode="bilinear", align_corners=False)
        else:
            x_sam_in = x_norm

        sam_raw = self.sam_encoder(x_sam_in)
        fused_feats: List[torch.Tensor] = []

        for idx, (c_feat, s_raw, proj, gate_net, out_conv) in enumerate(
            zip(cnn_feats, sam_raw, self.sam_projs, self.fusion_gates, self.out_convs)
        ):
            s_proj = proj(s_raw)
            if s_proj.shape[-2:] != c_feat.shape[-2:]:
                s_proj = F.interpolate(s_proj, size=c_feat.shape[-2:], mode="bilinear", align_corners=False)
            gate = gate_net(torch.cat([c_feat, s_proj], dim=1))
            scale = self.sam_scales[idx].clamp(0.05, 2.0)
            sam_delta = out_conv(gate * s_proj)
            fused_feats.append(c_feat + scale * sam_delta)

        return fused_feats


@BLOCK_REGISTRY.register("sam3_foundation_adapter")
class SAM3FoundationAdapter(nn.Module):
    """
    SAM 3 Foundation Adapter: Perception Encoder (PE) + Promptable Concept Segmentation (PCS)
    + Decoupled Global Presence Gating + Multi-Scale Feature Alignment.

    Innovations from Meta's SAM 3 (Segment Anything Model 3, 2026):
    1. Perception Encoder (PE) with ViT / vision-language aligned representations.
    2. Promptable Concept Segmentation (PCS): Learnable multi-concept tokens representing
       building footprints, orthogonal roofs, and structural boundaries cross-attend with
       PE spatial patch tokens to inject semantic concept priors into low-level geometry.
    3. Decoupled Global Presence Head: Predicts global concept presence probability to
       dynamically gate lateral adapters and suppress false alarms in non-building regions
       (e.g., rivers, agricultural fields).
    4. 4-Stage Multi-Scale Lateral Fusion: Aligns perception features to strides [2, 4, 8, 16]
       and injects them into the CNN pyramid via presence-gated residual connections.
    """

    def __init__(
        self,
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        sam_arch: str = "sam2_hiera_tiny",
        pretrained_sam: bool = True,
        num_concepts: int = 4,
    ):
        super().__init__()
        self.stage_channels = tuple(stage_channels)
        self.sam_arch = sam_arch
        self.pretrained_sam = pretrained_sam
        self.num_concepts = num_concepts

        self.register_buffer(
            "sam_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "sam_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

        import timm

        self.sam_encoder = timm.create_model(
            sam_arch,
            features_only=True,
            pretrained=pretrained_sam,
        )
        sam_dims = list(self.sam_encoder.feature_info.channels())
        self.num_stages = len(sam_dims)

        pe_dim = sam_dims[-1]
        self.concept_dim = pe_dim
        self.concept_tokens = nn.Parameter(torch.randn(1, num_concepts, pe_dim) * 0.02)
        n_heads = 8 if pe_dim % 8 == 0 else (4 if pe_dim % 4 == 0 else 1)
        self.concept_cross_attn = nn.MultiheadAttention(embed_dim=pe_dim, num_heads=n_heads, batch_first=True)
        self.concept_norm = nn.LayerNorm(pe_dim)

        # SAM 3 Decoupled Global Presence Head (predicts p_presence in [0, 1])
        self.presence_head = nn.Sequential(
            nn.Linear(pe_dim, pe_dim // 4),
            nn.GELU(),
            nn.Linear(pe_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Lateral 1x1 + 3x3 adapters projecting PE features to target stage_channels
        self.sam_projs = nn.ModuleList()
        for idx in range(len(stage_channels)):
            src_dim = sam_dims[min(idx, len(sam_dims) - 1)]
            self.sam_projs.append(
                nn.Sequential(
                    nn.Conv2d(src_dim, stage_channels[idx], kernel_size=1, bias=False),
                    nn.BatchNorm2d(stage_channels[idx]),
                    nn.GELU(),
                    nn.Conv2d(stage_channels[idx], stage_channels[idx], kernel_size=3, padding=1, groups=stage_channels[idx], bias=False),
                    nn.BatchNorm2d(stage_channels[idx]),
                    nn.GELU(),
                )
            )

        # Cross-foundation spatial-channel gating with presence modulation
        self.fusion_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.GELU(),
                    nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                    nn.Sigmoid(),
                )
                for c in stage_channels
            ]
        )
        self.out_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.GELU(),
                )
                for c in stage_channels
            ]
        )
        for conv_block in self.out_convs:
            nn.init.constant_(conv_block[1].weight, 0.2)
            nn.init.zeros_(conv_block[1].bias)
        self.sam_scales = nn.Parameter(torch.full((len(stage_channels),), 0.5, dtype=torch.float32))

    def forward(self, x_rgb: torch.Tensor, cnn_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        x_norm = (x_rgb - self.sam_mean) / self.sam_std
        h_in, w_in = x_norm.shape[-2:]
        if h_in < 64 or w_in < 64:
            x_sam_in = F.interpolate(x_norm, size=(max(h_in, 64), max(w_in, 64)), mode="bilinear", align_corners=False)
        else:
            x_sam_in = x_norm

        sam_raw = self.sam_encoder(x_sam_in)
        pe_deep = sam_raw[-1]
        bsz, c_dim, hp, wp = pe_deep.shape
        pe_tokens = pe_deep.flatten(2).transpose(1, 2)

        # 1. Concept Prompt Conditioning (PCS)
        concepts = self.concept_tokens.expand(bsz, -1, -1)
        guided_tokens, _ = self.concept_cross_attn(query=pe_tokens, key=concepts, value=concepts)
        guided_tokens = self.concept_norm(pe_tokens + guided_tokens)
        pe_deep_guided = guided_tokens.transpose(1, 2).view(bsz, c_dim, hp, wp)

        # 2. Decoupled Global Presence Estimation
        global_summary = pe_tokens.mean(dim=1)
        presence = self.presence_head(global_summary).view(bsz, 1, 1, 1)

        fused_feats: List[torch.Tensor] = []
        for idx, (c_feat, proj, gate_net, out_conv) in enumerate(
            zip(cnn_feats, self.sam_projs, self.fusion_gates, self.out_convs)
        ):
            if idx == len(cnn_feats) - 1 or idx >= len(sam_raw):
                s_raw = pe_deep_guided
            else:
                s_raw = sam_raw[idx]

            s_proj = proj(s_raw)
            if s_proj.shape[-2:] != c_feat.shape[-2:]:
                s_proj = F.interpolate(s_proj, size=c_feat.shape[-2:], mode="bilinear", align_corners=False)

            # Spatial gate modulated by global concept presence factor
            gate = gate_net(torch.cat([c_feat, s_proj], dim=1)) * (0.5 + 0.5 * presence)
            scale = self.sam_scales[idx].clamp(0.05, 2.0)
            sam_delta = out_conv(gate * s_proj)
            fused_feats.append(c_feat + scale * sam_delta)

        return fused_feats


@BLOCK_REGISTRY.register("single_image_edgeref_branch")
class SingleImageEdgeRefBranch(nn.Module):
    """
    Single-Image High-Frequency Edge Extraction & Cross-Attention Refinement Branch
    adapted from EdgeRefNet (Hussin et al., IEEE TGRS 2026) and coupled with CBRNet's
    corner responses.

    Components:
    1. Fixed Sobel gradient magnitude extractor injecting explicit image-level edge priors
       at each pyramid resolution.
    2. Multi-scale `EDBBlock` (Edge Detection Block) + `EEMBlock` (SCSE + High-Frequency
       Enhancement HFAB) across all 4 pyramid stages.
    3. Corner-Guided Edge Modulation fusing multi-scale Harris corner maps from `CGFEModule`
       into the aggregated high-frequency edge feature map.
    4. `EdgeGuidedCrossAttention` refining semantic BiFPN features with high-frequency edge Keys/Values.
    """

    def __init__(
        self,
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        edge_channels: int = 64,
        sem_channels: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        sobel_x = torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]).unsqueeze(0)
        sobel_y = torch.tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # Each stage receives [stage_feat (c), sobel_grad (1), corner_attn (1)] -> c + 2 channels
        self.edge_edbs = nn.ModuleList([EDBBlock(c + 2, edge_channels) for c in stage_channels])
        self.edge_eems = nn.ModuleList([EEMBlock(edge_channels) for _ in stage_channels])

        self.edge_fuse = nn.Sequential(
            nn.Conv2d(edge_channels * len(stage_channels), sem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sem_channels),
            nn.ReLU(inplace=True),
        )
        self.cross_refiner = EdgeGuidedCrossAttention(channels=sem_channels, num_heads=num_heads)
        self.edge_head = nn.Conv2d(sem_channels, 1, kernel_size=1)

    def _compute_sobel_mag(self, x_rgb: torch.Tensor) -> torch.Tensor:
        gray = x_rgb.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=gray.dtype), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        mag_max = mag.amax(dim=(-2, -1), keepdim=True) + 1e-6
        return mag / mag_max

    def forward(
        self,
        x_rgb: torch.Tensor,
        stage_feats: Sequence[torch.Tensor],
        corner_attns: Sequence[torch.Tensor],
        sem_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sobel_full = self._compute_sobel_mag(x_rgb)
        target_size = sem_feat.shape[-2:]
        edge_multi: List[torch.Tensor] = []

        for feat, cor_map, edb, eem in zip(stage_feats, corner_attns, self.edge_edbs, self.edge_eems):
            sz = feat.shape[-2:]
            s_lvl = F.interpolate(sobel_full, size=sz, mode="bilinear", align_corners=False)
            c_lvl = (
                cor_map
                if cor_map.shape[-2:] == sz
                else F.interpolate(cor_map, size=sz, mode="bilinear", align_corners=False)
            )
            e_in = torch.cat([feat, s_lvl, c_lvl], dim=1)
            e_i = eem(edb(e_in))
            if e_i.shape[-2:] != target_size:
                e_i = F.interpolate(e_i, size=target_size, mode="bilinear", align_corners=False)
            edge_multi.append(e_i)

        edge_feat = self.edge_fuse(torch.cat(edge_multi, dim=1))
        refined_sem = self.cross_refiner(sem_feat, edge_feat)
        edge_logits_low = self.edge_head(edge_feat)
        return {
            "refined_sem": refined_sem,
            "edge_feat": edge_feat,
            "edge_logits_low": edge_logits_low,
        }


# =============================================================================
# 11. Wang et al. (2026) - DSTCD Modules (HIFE, STFTB, SEFTB)
# =============================================================================


class _HIFEBasicBlock(nn.Module):
    """Residual BasicBlock used in ASpanFormer's ResNetFPN_8_2 (HIFE)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            None
            if stride == 1 and in_planes == planes
            else nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


@BLOCK_REGISTRY.register("hife_resnet_fpn", aliases=["dstcd_hife"])
class HIFEResNetFPN(nn.Module):
    """
    Heterogeneous Image Feature Encoder (HIFE) from DSTCD (Wang et al., Int. J. Appl. Earth Obs. Geoinf. 146, 105100, 2026).
    Based on ASpanFormer's `ResNetFPN_8_2` backbone, extracting:
    - Low-frequency structural feature at 1/2 resolution (`x1_out` in R^{B x block_dims[0] x H/2 x W/2})
    - High-frequency semantic feature at 1/8 resolution (`x3_out` in R^{B x block_dims[2] x H/8 x W/8})
    Can load official ASpanFormer `outdoor.ckpt` weights when `ckpt_path` is provided.
    """

    def __init__(
        self,
        initial_dim: int = 128,
        block_dims: Sequence[int] = (128, 196, 256),
        ckpt_path: Optional[str] = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.in_planes = initial_dim
        self.block_dims = tuple(block_dims)

        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block_dims[2], stride=2)  # 1/8

        self.layer3_outconv = nn.Conv2d(block_dims[2], block_dims[2], kernel_size=1, bias=False)
        self.layer2_outconv = nn.Conv2d(block_dims[1], block_dims[2], kernel_size=1, bias=False)
        self.layer2_outconv2 = nn.Sequential(
            nn.Conv2d(block_dims[2], block_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(block_dims[2], block_dims[1], kernel_size=3, padding=1, bias=False),
        )
        self.layer1_outconv = nn.Conv2d(block_dims[0], block_dims[1], kernel_size=1, bias=False)
        self.layer1_outconv2 = nn.Sequential(
            nn.Conv2d(block_dims[1], block_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(block_dims[1], block_dims[0], kernel_size=3, padding=1, bias=False),
        )

        if ckpt_path is not None and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state)
            cleaned = {}
            for k, v in sd.items():
                k_clean = k.replace("matcher.", "", 1) if k.startswith("matcher.") else k
                if k_clean.startswith("backbone."):
                    cleaned[k_clean.replace("backbone.", "", 1)] = v
            self.load_state_dict(cleaned, strict=False)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        l1 = _HIFEBasicBlock(self.in_planes, dim, stride=stride)
        l2 = _HIFEBasicBlock(dim, dim, stride=1)
        self.in_planes = dim
        return nn.Sequential(l1, l2)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # Convert arbitrary-channel inputs into 1-channel intensity maps for HIFE registration encoder
        g1 = t1[:, :3].mean(dim=1, keepdim=True) if t1.shape[1] >= 3 else t1[:, :1]
        g2 = t2[:, :3].mean(dim=1, keepdim=True) if t2.shape[1] >= 3 else t2[:, :1]
        b = g1.shape[0]
        x = torch.cat([g1, g2], dim=0)

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x3_out = self.layer3_outconv(x3)
        x3_up = F.interpolate(x3_out, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x2_out = self.layer2_outconv2(self.layer2_outconv(x2) + x3_up)

        x2_up = F.interpolate(x2_out, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x1_out = self.layer1_outconv2(self.layer1_outconv(x1) + x2_up)

        t1_low, t2_low = x1_out.split(b, dim=0)
        t1_deep, t2_deep = x3_out.split(b, dim=0)
        return (t1_low, t2_low), (t1_deep, t2_deep)


@BLOCK_REGISTRY.register("dstcd_stftb", aliases=["stftb"])
class DSTCDSTFTBBlock(nn.Module):
    """
    Structure Feature Transfer Block (STFTB) from DSTCD (Wang et al., 2026; `T_D_fusion` in `CDNet.py`).
    Splits source structural features into 4 channel subgroups, computes pixel-wise cosine similarity
    against compressed target features, weights and concatenates across both timestamps, and fuses.
    """

    def __init__(self, target_channels: int = 128, source_channels: int = 128):
        super().__init__()
        assert source_channels % 4 == 0, "source_channels must be divisible by 4"
        sub_c = source_channels // 4
        self.d_conv = nn.Conv2d(target_channels, sub_c, kernel_size=1)
        self.splits = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(sub_c, sub_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(sub_c),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )
        self.proj_back = (
            nn.Identity()
            if source_channels == target_channels
            else nn.Conv2d(source_channels, target_channels, kernel_size=1, bias=False)
        )
        self.d_d = nn.Sequential(
            nn.Conv2d(target_channels * 2, target_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True),
        )

    def _t_d_fusion(self, t_feat: torch.Tensor, d_feat: torch.Tensor) -> torch.Tensor:
        if t_feat.shape[-2:] != d_feat.shape[-2:]:
            t_feat = F.interpolate(t_feat, size=d_feat.shape[-2:], mode="bilinear", align_corners=False)
        d_sim = self.d_conv(d_feat)
        chunks = torch.chunk(t_feat, 4, dim=1)
        weighted = []
        for chunk, split_conv in zip(chunks, self.splits):
            s_chunk = split_conv(chunk)
            sim_map = F.cosine_similarity(s_chunk, d_sim, dim=1, eps=1e-6).unsqueeze(1)
            weighted.append(sim_map * d_sim)
        x = self.proj_back(torch.cat(weighted, dim=1))
        return x + d_feat

    def forward(self, t1_low: torch.Tensor, t2_low: torch.Tensor, target_low: torch.Tensor) -> torch.Tensor:
        d1 = self._t_d_fusion(t1_low, target_low)
        d2 = self._t_d_fusion(t2_low, target_low)
        return self.d_d(torch.cat([d1, d2], dim=1))


@BLOCK_REGISTRY.register("dstcd_seftb", aliases=["seftb"])
class DSTCDSEFTBBlock(nn.Module):
    """
    Semantic Feature Transfer Block (SEFTB) from DSTCD (Wang et al., 2026; `CDNet.py`).
    Applies a 'first cross-task, then cross-source' multi-head attention strategy:
    1. Cross-task MHA: Target feature `TI_CD^H` queries source features `SI_A^H` and `SI_B^H`.
    2. Cross-source MHA: Target feature `TI_CD^H` queries concatenated `[T1_output, T2_output]`.
    """

    def __init__(
        self,
        target_channels: int = 512,
        source_channels: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.f2tq_linear = nn.Linear(target_channels, source_channels)
        self.t1k_linear = nn.Linear(source_channels, source_channels)
        self.t1v_linear = nn.Linear(source_channels, source_channels)
        self.t2k_linear = nn.Linear(source_channels, source_channels)
        self.t2v_linear = nn.Linear(source_channels, source_channels)

        self.ftq_linear = nn.Linear(target_channels, target_channels)
        self.tfk_linear = nn.Linear(source_channels * 2, target_channels)
        self.tfv_linear = nn.Linear(source_channels * 2, target_channels)

        self.t1_mha = nn.MultiheadAttention(source_channels, num_heads, dropout=dropout, batch_first=True)
        self.t2_mha = nn.MultiheadAttention(source_channels, num_heads, dropout=dropout, batch_first=True)
        self.t_mha = nn.MultiheadAttention(target_channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, t1_deep: torch.Tensor, t2_deep: torch.Tensor, target_deep: torch.Tensor) -> torch.Tensor:
        f_b, f_c, f_h, f_w = target_deep.shape
        if t1_deep.shape[-2:] != (f_h, f_w):
            t1_deep = F.interpolate(t1_deep, size=(f_h, f_w), mode="bilinear", align_corners=False)
            t2_deep = F.interpolate(t2_deep, size=(f_h, f_w), mode="bilinear", align_corners=False)

        target_seq = target_deep.flatten(2).transpose(1, 2)  # (B, L, target_C)
        t1_seq = t1_deep.flatten(2).transpose(1, 2)          # (B, L, source_C)
        t2_seq = t2_deep.flatten(2).transpose(1, 2)          # (B, L, source_C)

        d_q = self.f2tq_linear(target_seq)
        t1_k = self.t1k_linear(t1_seq)
        t1_v = self.t1v_linear(t1_seq)
        t2_k = self.t2k_linear(t2_seq)
        t2_v = self.t2v_linear(t2_seq)

        t1_out, _ = self.t1_mha(d_q, t1_k, t1_v, need_weights=False)
        t2_out, _ = self.t2_mha(d_q, t2_k, t2_v, need_weights=False)
        t_cat = torch.cat([t1_out, t2_out], dim=-1)  # (B, L, 2*source_C)

        q = self.ftq_linear(target_seq)
        k = self.tfk_linear(t_cat)
        v = self.tfv_linear(t_cat)
        out, _ = self.t_mha(q, k, v, need_weights=False)

        return target_deep + out.transpose(1, 2).reshape(f_b, f_c, f_h, f_w)


# =============================================================================
# 12. Liu et al. (2026) - GLMamba Modules (VSSBlock, CSAM, BFA, CSIE)
# =============================================================================


@BLOCK_REGISTRY.register("glmamba_vss_block")
class GLMambaVSSBlock(nn.Module):
    """
    Visual State Space (VSS) Block from GLMamba (Liu et al., IEEE JSTARS, Vol. 19, pp. 11344–11360, 2026).
    Follows Fig. 1:
    Input -> LayerNorm -> split into two branches:
      - Branch 1: Linear -> DWConv3x3 -> SiLU -> SS2DCrossScan -> LayerNorm
      - Branch 2: Linear -> SiLU
    Element-wise product -> Linear -> Residual addition.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.ss2d = SS2DCrossScan(channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.gate_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        b1 = F.silu(self.dw_conv(self.in_proj(x_norm)))
        b1 = self.norm2(self.ss2d(b1))
        b2 = F.silu(self.gate_proj(x_norm))
        return x + self.out_proj(b1 * b2)


@BLOCK_REGISTRY.register("glmamba_csam", aliases=["csam"])
class GLMambaCSAMModule(nn.Module):
    """
    Change-Aware Semantic Aggregation Module (CSAM) from GLMamba (Liu et al., IEEE JSTARS 2026).
    Aggregates adjacent encoder scales (Eqs. 11–14, Fig. 2):
    - Reference feature `f2` in R^{B x C x H x W}: `f2' = Conv3x3(f2)`, `f2'' = Conv1x1(f2)`
    - Finer feature `f1` (optional) in R^{B x C_1 x 2H x 2W}: `f1' = ReLU(BN(Conv3x3(MaxPool(f1))))`
    - Coarser feature `f3` (optional) in R^{B x C_3 x H/2 x W/2}: `f3' = Upsample(ReLU(BN(Conv3x3(Conv1x1(f3)))))`
    Concatenates available branches -> Conv3x3 -> adds residual `f2''` -> ReLU.
    """

    def __init__(
        self,
        c_mid: int,
        c_low: Optional[int] = None,
        c_high: Optional[int] = None,
    ):
        super().__init__()
        self.has_low = c_low is not None
        self.has_high = c_high is not None

        self.conv_f2_3x3 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
        )
        self.conv_f2_1x1 = nn.Conv2d(c_mid, c_mid, kernel_size=1, bias=False)

        num_branches = 1
        if self.has_low:
            self.low_branch = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(c_low, c_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.ReLU(inplace=True),
            )
            num_branches += 1
        if self.has_high:
            self.high_branch = nn.Sequential(
                nn.Conv2d(c_high, c_mid, kernel_size=1, bias=False),
                nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.ReLU(inplace=True),
            )
            num_branches += 1

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(c_mid * num_branches, c_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
        )

    def forward(
        self,
        f2: torch.Tensor,
        f1: Optional[torch.Tensor] = None,
        f3: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_hw = f2.shape[-2:]
        f2_p = self.conv_f2_3x3(f2)
        f2_res = self.conv_f2_1x1(f2)
        branches = [f2_p]

        if self.has_low and f1 is not None:
            f1_p = self.low_branch(f1)
            if f1_p.shape[-2:] != target_hw:
                f1_p = F.interpolate(f1_p, size=target_hw, mode="bilinear", align_corners=False)
            branches.insert(0, f1_p)

        if self.has_high and f3 is not None:
            f3_p = self.high_branch(f3)
            f3_p = F.interpolate(f3_p, size=target_hw, mode="bilinear", align_corners=False)
            branches.append(f3_p)

        fused = self.fuse_conv(torch.cat(branches, dim=1))
        return F.relu(fused + f2_res, inplace=True)


@ATTENTION_REGISTRY.register("glmamba_bfa", aliases=["bfa"])
class GLMambaBFAModule(nn.Module):
    """
    Bitemporal Fusion Attention (BFA) Module from GLMamba (Liu et al., IEEE JSTARS 2026).
    Follows Eqs. (15)–(22) and Fig. 3:
    - Subtraction Path: `Fd = X2 - X1` -> Channel Attention `Wc` -> Spatial Attention `Ws` -> `Zd = Ws * (Wc * Fd)`
    - Connection Path: `Fcat = Concat(X1, X2)` -> `Conv3x3+BN+ReLU` -> `Conv1x1+BN+ReLU` -> `Zc`
    - Output: `Y = Zd + Zc`
    """

    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        mid_c = max(channels // ratio, 4)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, channels, kernel_size=1, bias=False),
        )
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.conn_branch = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        fd = x2 - x1
        avg_ca = self.ca_mlp(F.adaptive_avg_pool2d(fd, 1))
        max_ca = self.ca_mlp(F.adaptive_max_pool2d(fd, 1))
        wc = torch.sigmoid(avg_ca + max_ca)
        fc = wc * fd

        avg_sa = fc.mean(dim=1, keepdim=True)
        max_sa = fc.amax(dim=1, keepdim=True)
        ws = self.sa_conv(torch.cat([avg_sa, max_sa], dim=1))
        zd = ws * fc

        zc = self.conn_branch(torch.cat([x1, x2], dim=1))
        return zd + zc


@ATTENTION_REGISTRY.register("glmamba_csie", aliases=["csie"])
class GLMambaCSIEModule(nn.Module):
    """
    Cross-Space Interaction Enhancement (CSIE) Module from GLMamba (Liu et al., IEEE JSTARS 2026).
    Follows Eqs. (23)–(30) and Fig. 4:
    Divides channels into `G` groups (`C_g = C // G`):
    - Path 1 (Directional Coordinate Attention): Horizontal pool (H, 1) + Vertical pool (1, W)
      -> shared 1x1 conv -> split -> sigmoid -> `A_g` (B*G, C_g, H, W).
    - Path 2 (Local Depthwise + SE): Depthwise 3x3 conv + Squeeze-and-Excitation -> `X'_l` (B*G, C_g, H, W).
    - Cross-Space Interaction:
      * `f_g = Softmax(GAP(A_g))` in R^{(B*G) x 1 x C_g}, paired with spatial `X'_l` in R^{(B*G) x C_g x HW} -> (B*G, 1, HW)
      * `f_l = Softmax(GAP(X'_l))` in R^{(B*G) x 1 x C_g}, paired with spatial `A_g` in R^{(B*G) x C_g x HW} -> (B*G, 1, HW)
      * Spatial cross-attention `A_cs = Sigmoid(MatMul1 + MatMul2)` in R^{(B*G) x 1 x H x W}
      * Output `X_tilde = X_g * A_cs`, reshaped back to (B, C, H, W) + channel projection + residual.
    """

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.channels = channels
        self.groups = groups if channels % groups == 0 else 1
        cg = channels // self.groups
        self.cg = cg

        self.coord_conv = nn.Conv2d(cg, cg, kernel_size=1, bias=False)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(cg, cg, kernel_size=3, padding=1, groups=cg, bias=False),
            nn.BatchNorm2d(cg),
            nn.ReLU(inplace=True),
        )
        se_mid = max(cg // 2, 2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cg, se_mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_mid, cg, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        cg = self.cg
        xg = x.view(b * g, cg, h, w)

        # Path 1: Directional coordinate attention
        xh = xg.mean(dim=3, keepdim=True)                    # (B*G, C_g, H, 1)
        xw = xg.mean(dim=2, keepdim=True).transpose(2, 3)    # (B*G, C_g, W, 1)
        x_cat = self.coord_conv(torch.cat([xh, xw], dim=2))  # (B*G, C_g, H+W, 1)
        ah, aw = torch.split(x_cat, [h, w], dim=2)
        aw = aw.transpose(2, 3)                              # (B*G, C_g, 1, W)
        ag = xg * torch.sigmoid(ah) * torch.sigmoid(aw)      # (B*G, C_g, H, W)

        # Path 2: Local depthwise separable + SE
        xl = self.dw_conv(xg)
        xl_p = xl * self.se(xl)                              # (B*G, C_g, H, W)

        # Cross-space interaction (Fig. 4 & Eqs. 27-30)
        fg = F.softmax(ag.mean(dim=(-2, -1)), dim=-1).unsqueeze(1)    # (B*G, 1, C_g)
        fl = F.softmax(xl_p.mean(dim=(-2, -1)), dim=-1).unsqueeze(1)  # (B*G, 1, C_g)

        ag_flat = ag.view(b * g, cg, h * w)
        xl_flat = xl_p.view(b * g, cg, h * w)

        attn1 = torch.bmm(fg, xl_flat)  # (B*G, 1, H*W)
        attn2 = torch.bmm(fl, ag_flat)  # (B*G, 1, H*W)
        acs = torch.sigmoid(attn1 + attn2).view(b * g, 1, h, w)

        xg_out = (xg * acs).view(b, c, h, w)
        return x + self.out_proj(xg_out)


# =============================================================================
# 13. Su et al. (2026) - Seg2Change Modules (BDFM, EDQA, ResUp)
# =============================================================================


@BLOCK_REGISTRY.register("seg2change_bdfm", aliases=["bdfm", "mdfm_seg2change"])
class Seg2ChangeBDFM(nn.Module):
    """
    Bi-temporal Difference Fusion Module (BDFM / `MDFM` in `change_blocks.py`) from Seg2Change
    (Su et al., arXiv:2604.11231, 2026).
    Follows Eqs. (3)–(5):
    - `Att_i = Sigmoid(Conv3x3(|F1_i - F2_i|))`
    - `X_i^t = ReLU(GN(Conv3x3(F_i^t + Att_i * F_i^t)))`
    - `D_i = ReLU(GN(Conv3x3(ReLU(GN(Conv3x3([X_i^1 || X_i^2]))) * Att_i)))`
    """

    def __init__(self, in_d: int, out_d: int):
        super().__init__()
        num_g_in = 8 if in_d % 8 == 0 else (4 if in_d % 4 == 0 else 1)
        num_g_out = 8 if out_d % 8 == 0 else (4 if out_d % 4 == 0 else 1)

        self.conv_sub = nn.Conv2d(in_d, in_d, kernel_size=3, padding=1, bias=False)
        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_g_in, in_d),
            nn.ReLU(inplace=True),
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_g_in, in_d),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_d * 2, in_d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_g_in, in_d),
            nn.ReLU(inplace=True),
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(in_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_g_out, out_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))
        x1_enh = self.conv_diff_enh1(x1 * x_att + x1)
        x2_enh = self.conv_diff_enh2(x2 * x_att + x2)
        x_f = self.conv_cat(torch.cat([x1_enh, x2_enh], dim=1))
        return self.conv_dr(x_f * x_att)


class _Seg2ChangeWindowAttention(nn.Module):
    """Window-based Difference Query Attention with relative position bias from Seg2Change."""

    def __init__(self, dim: int, window_size: int = 9, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads if dim % num_heads == 0 else (4 if dim % 4 == 0 else 1)
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.num_heads)
        )
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1), persistent=False)

        self.qk = nn.Linear(dim, dim * 2, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        b_win, n, c = x.shape
        qk = self.qk(x).reshape(b_win, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(g).reshape(b_win, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            n, n, self.num_heads
        )
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_win, n, c)
        return self.proj(out)


class _Seg2ChangeMoEFFN(nn.Module):
    """Mixture-of-Experts FFN (`MoEFFN_Gating`) from Seg2Change Eqs. (8)–(10)."""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.gating_network = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> weights: (B, L, num_experts)
        weights = F.softmax(self.gating_network(x), dim=-1)
        expert_outs = torch.stack([exp(x) for exp in self.experts], dim=-1)  # (B, L, C, num_experts)
        return (expert_outs * weights.unsqueeze(2)).sum(dim=-1)


@ATTENTION_REGISTRY.register("seg2change_edqa", aliases=["edqa"])
class Seg2ChangeEDQA(nn.Module):
    """
    Effective Difference Query Attention (EDQA / `EffAggregatorLayer` + `MiTBlock_Swin`) from Seg2Change
    (Su et al., arXiv:2604.11231, 2026).
    Follows Eqs. (6)–(10) and Fig. 5:
    - Generates guidance `G_i^t = Conv1x1([D_i || F_i^t])`
    - Window-based Cross-Temporal Difference Query Attention with automatic padding for arbitrary (H, W)
    - Fuses calibrated bi-temporal differences via two 3x3 Conv+GN+ReLU blocks
    - Applies Mixture-of-Experts FFN (`MoEFFN`) with `num_experts=4`.
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 9, num_experts: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        num_g = 8 if dim % 8 == 0 else (4 if dim % 4 == 0 else 1)

        self.conv_g1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.conv_g2 = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.norm1 = nn.LayerNorm(dim)
        self.norm_g1 = nn.LayerNorm(dim)
        self.norm_g2 = nn.LayerNorm(dim)

        self.attention = _Seg2ChangeWindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        self.conv_dr = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_g, dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_g, dim),
            nn.ReLU(inplace=True),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Seg2ChangeMoEFFN(dim=dim, hidden_dim=dim * 3, num_experts=num_experts)

    def _window_partition(self, x: torch.Tensor, ws: int) -> torch.Tensor:
        b, h, w, c = x.shape
        x = x.view(b, h // ws, ws, w // ws, ws, c)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, c)

    def _window_reverse(self, windows: torch.Tensor, ws: int, h: int, w: int, b: int) -> torch.Tensor:
        x = windows.view(b, h // ws, w // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    def forward(self, diff: torch.Tensor, guidance1: torch.Tensor, guidance2: torch.Tensor) -> torch.Tensor:
        b, c, h, w = diff.shape
        ws = self.window_size

        g1 = self.conv_g1(torch.cat([diff, guidance1], dim=1))
        g2 = self.conv_g2(torch.cat([diff, guidance2], dim=1))

        # Pad spatial dimensions to be divisible by window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            diff_p = F.pad(diff, (0, pad_w, 0, pad_h))
            g1_p = F.pad(g1, (0, pad_w, 0, pad_h))
            g2_p = F.pad(g2, (0, pad_w, 0, pad_h))
        else:
            diff_p, g1_p, g2_p = diff, g1, g2

        hp, wp = diff_p.shape[-2:]
        d_hwc = self.norm1(diff_p.permute(0, 2, 3, 1))
        g1_hwc = self.norm_g1(g1_p.permute(0, 2, 3, 1))
        g2_hwc = self.norm_g2(g2_p.permute(0, 2, 3, 1))

        d_win = self._window_partition(d_hwc, ws)
        g1_win = self._window_partition(g1_hwc, ws)
        g2_win = self._window_partition(g2_hwc, ws)

        out1_win = self.attention(d_win, g1_win)
        out2_win = self.attention(d_win, g2_win)

        out1 = self._window_reverse(out1_win, ws, hp, wp, b).permute(0, 3, 1, 2)[:, :, :h, :w]
        out2 = self._window_reverse(out2_win, ws, hp, wp, b).permute(0, 3, 1, 2)[:, :, :h, :w]

        fused_d = self.conv_dr(torch.cat([out1, out2], dim=1))
        res1 = (diff + fused_d).flatten(2).transpose(1, 2)  # (B, H*W, C)
        res2 = res1 + self.mlp(self.norm2(res1))
        return res2.transpose(1, 2).reshape(b, c, h, w)


class _Seg2ChangeRCU(nn.Module):
    """ResidualConvUnit from Seg2Change (`blocks.py`)."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act(x))
        out = self.conv2(self.act(out))
        return x + out


@BLOCK_REGISTRY.register("seg2change_resup", aliases=["resup_block"])
class Seg2ChangeResUpBlock(nn.Module):
    """
    Residual Upsampler (`FeatureFusionBlock` in `blocks.py` / ResUp in Eqs. 37–38) from Seg2Change
    (Su et al., arXiv:2604.11231, 2026).
    """

    def __init__(self, features: int = 64, has_skip: bool = True):
        super().__init__()
        self.has_skip = has_skip
        if has_skip:
            self.rcu1 = _Seg2ChangeRCU(features)
        self.rcu2 = _Seg2ChangeRCU(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, bias=True)

    def forward(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        out = x0
        if x1 is not None and self.has_skip:
            res = self.rcu1(x1)
            if res.shape[-2:] != out.shape[-2:]:
                res = F.interpolate(res, size=out.shape[-2:], mode="bilinear", align_corners=False)
            out = out + res
        out = self.rcu2(out)
        if size is None:
            out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=False)
        else:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return self.out_conv(out)


