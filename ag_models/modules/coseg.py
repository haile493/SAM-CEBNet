"""
Collaborative Co-Segmentation Modules (RP-Encoder, MP-Decoder, Mutual Cross-Guidance Gates).

Standardizes and unifies the implementations from:
- `liverseg/models/coseg_liver.py` (`RPEncoder3D`, `MPDecoderHead3D`)
- `liverseg/models/waveattn_sopseg_liver.py` (Mutual Query Cross-Guidance)
- `liverseg/models/waveco_mtl_net.py` & `weakmedsam_coseg_waveattn.py` (Lightweight Mutual Prompting Gates)
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from ag_models.core.layers import get_activation, get_conv, get_conv_transpose, get_norm
from ag_models.core.registry import BLOCK_REGISTRY
from ag_models.modules.blocks import ResBlock


@BLOCK_REGISTRY.register("rp_encoder")
class RPEncoder(nn.Module):
    """
    Unified 2D / 3D Region-Aware Prompt Encoder (RP-Encoder).
    Encodes coarse mask predictions via 8x downsampling convolutions, self-attention,
    and cross-attention with shared visual embeddings `z_embed` to produce dense region prompts `g`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        dims: int = 3,
        norm_type: str = "instancenorm",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dims = dims
        conv_cls = get_conv(dims)

        self.downsample = nn.Sequential(
            conv_cls(in_channels, embed_dim // 4, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm(norm_type, embed_dim // 4, dims=dims),
            nn.GELU(),
            conv_cls(embed_dim // 4, embed_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm(norm_type, embed_dim // 2, dims=dims),
            nn.GELU(),
            conv_cls(embed_dim // 2, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm(norm_type, embed_dim, dims=dims),
            nn.GELU(),
            conv_cls(embed_dim, embed_dim, kernel_size=1, bias=True),
        )

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_ca = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, mask_probs: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        b, c = z_embed.shape[:2]
        spatial = z_embed.shape[2:]

        m = self.downsample(mask_probs)
        m_tokens = m.flatten(2).permute(0, 2, 1)
        z_tokens = z_embed.flatten(2).permute(0, 2, 1)

        sa_out, _ = self.self_attn(m_tokens, m_tokens, m_tokens)
        m_tokens = self.norm_sa(m_tokens + sa_out)

        ca_out, _ = self.cross_attn(m_tokens, z_tokens, z_tokens)
        prompt_tokens = self.norm_ca(m_tokens + ca_out)
        prompt_tokens = self.norm_ffn(prompt_tokens + self.ffn(prompt_tokens))

        return prompt_tokens.permute(0, 2, 1).view(b, c, *spatial)


@BLOCK_REGISTRY.register("coseg_query_interaction")
class CoSegQueryInteraction(nn.Module):
    """
    Task-specific Learnable Query Interaction Layer for Co-Seg.
    Refines prompt-conditioned bottleneck features (`z + g_other`) using learnable queries
    via Self-Attention -> Cross-Attention -> Reverse Cross-Attention.
    """

    def __init__(self, embed_dim: int = 192, num_queries: int = 16, num_heads: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.attn_sa = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_ca = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.attn_rev = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_feat = nn.LayerNorm(embed_dim)

    def forward(self, z_guided: torch.Tensor) -> torch.Tensor:
        b = z_guided.shape[0]
        z_seq = z_guided.flatten(2).transpose(1, 2)

        q = self.queries.repeat(b, 1, 1)
        q_sa, _ = self.attn_sa(q, q, q)
        q = self.norm_q(q + q_sa)
        q_ca, _ = self.attn_ca(q, z_seq, z_seq)
        q = self.norm_q(q + q_ca)

        feat_seq, _ = self.attn_rev(z_seq, q, q)
        return self.norm_feat(z_seq + feat_seq).transpose(1, 2).view_as(z_guided)


@BLOCK_REGISTRY.register("mp_decoder_head")
class MPDecoderHead(nn.Module):
    """
    Unified 2D / 3D Mutual Prompt Mask Decoder Head (MP-Decoder).
    Conditioned on mutual region prompt `g_prompt_other` and multi-scale encoder skips.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_queries: int = 16,
        num_heads: int = 4,
        out_channels: int = 1,
        skip_channels: List[int] = [96, 48, 24],
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        self.query_interact = CoSegQueryInteraction(embed_dim, num_queries=num_queries, num_heads=num_heads)
        conv_t = get_conv_transpose(dims)
        conv_cls = get_conv(dims)

        self.up1 = conv_t(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.fuse1 = ResBlock(embed_dim // 2 + skip_channels[0], embed_dim // 2, dims=dims, norm_type=norm_type, act_type=act_type)

        self.up2 = conv_t(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2)
        self.fuse2 = ResBlock(embed_dim // 4 + skip_channels[1], embed_dim // 4, dims=dims, norm_type=norm_type, act_type=act_type)

        self.up3 = conv_t(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2)
        self.fuse3 = ResBlock(embed_dim // 8 + skip_channels[2], embed_dim // 8, dims=dims, norm_type=norm_type, act_type=act_type)

        self.head = conv_cls(embed_dim // 8, out_channels, kernel_size=1)

    def forward(
        self,
        z_embed: torch.Tensor,
        g_prompt_other: torch.Tensor,
        skips: List[torch.Tensor],
    ) -> torch.Tensor:
        h_vol = self.query_interact(z_embed + g_prompt_other)
        s3, s2, s1 = skips

        d1 = self.fuse1(torch.cat([self.up1(h_vol), s3], dim=1))
        d2 = self.fuse2(torch.cat([self.up2(d1), s2], dim=1))
        d3 = self.fuse3(torch.cat([self.up3(d2), s1], dim=1))
        return self.head(d3)


@BLOCK_REGISTRY.register("mutual_guidance_gate")
class MutualGuidanceGate(nn.Module):
    """
    Lightweight Bidirectional Mutual Prompting Gate between Macro (e.g., Organ/Liver)
    and Micro (e.g., Lesion/Tumor) feature streams.
    """

    def __init__(
        self,
        channels: int,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__()
        conv_cls = get_conv(dims)
        self.macro_stream = nn.Sequential(
            conv_cls(channels, channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, channels, dims=dims),
            get_activation(act_type),
        )
        self.micro_stream = nn.Sequential(
            conv_cls(channels, channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, channels, dims=dims),
            get_activation(act_type),
        )
        self.gate_macro_to_micro = nn.Sequential(conv_cls(channels, channels, kernel_size=1), nn.Sigmoid())
        self.gate_micro_to_macro = nn.Sequential(conv_cls(channels, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, shared_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        macro_f = self.macro_stream(shared_feat)
        micro_f = self.micro_stream(shared_feat)

        g_m2u = self.gate_macro_to_micro(macro_f)
        g_u2m = self.gate_micro_to_macro(micro_f)

        macro_guided = macro_f * (1.0 + g_u2m)
        micro_guided = micro_f * (1.0 + g_m2u)
        return macro_guided, micro_guided


# Backward-compatible 3D aliases
class RPEncoder3D(RPEncoder):
    def __init__(self, in_channels: int = 1, embed_dim: int = 128, num_heads: int = 4):
        super().__init__(in_channels=in_channels, embed_dim=embed_dim, num_heads=num_heads, dims=3)


class MPDecoderHead3D(MPDecoderHead):
    def __init__(
        self,
        embed_dim: int = 128,
        num_queries: int = 16,
        num_heads: int = 4,
        out_channels: int = 1,
        skip_channels: List[int] = [96, 48, 24],
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            out_channels=out_channels,
            skip_channels=skip_channels,
            dims=3,
        )
