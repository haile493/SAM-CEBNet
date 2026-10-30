"""
Standardized Volumetric (3D) & Slice (2D) Multi-Task Medical Imaging Architectures.

Constructed entirely from standardized `ag_models.modules` and `BaseModel`:
1. `WaveCoMTLNet` (`waveco_mtl`): Wavelet-Attentive Collaborative Multi-Task Learning Network.
2. `WaveAttnSOPSegNet` (`waveattn_sopseg`): Standalone Multi-Scale Wavelet-Attentive PRM Network.
3. `CoSegWaveAttnSOPSegNet` (`coseg_waveattn_sopseg`): Synergistic Co-Seg + WaveAttn-SOPSeg Network.
4. `WeedSenseNet` (`weedsense_3d` / `weedsense`): Dual-Path Detail + UIB Semantic + BGA + Transformer Global Decoder.
5. `MultiTaskMSFFNet` (`msff_mtl`): ECAdd + Deep Feature Enhancement (DFE) Attention Network.
6. `ResVNetFusion` (`resvnet_fusion`): V-Net + Inter-Network Channel Feature Fusion (ICFF) + ResNet.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ag_models.core.base import BaseModel, ModelConfig, ModelOutput
from ag_models.core.layers import (
    get_activation,
    get_adaptive_avg_pool,
    get_adaptive_max_pool,
    get_conv,
    get_conv_transpose,
    get_max_pool,
    get_norm,
    interpolate_nd,
)
from ag_models.core.registry import MODEL_REGISTRY
from ag_models.modules.attention import (
    DFEAttention,
    ECAddBlock,
    InterNetworkChannelFeatureFusion,
    SOPSegTextureBlock,
)
from ag_models.modules.blocks import BGALayer, ConvNormAct, ResBlock, TransformerBlock, UIBBlock
from ag_models.modules.coseg import CoSegQueryInteraction, MutualGuidanceGate, RPEncoder
from ag_models.modules.heads import MaskedFeaturePoolingHead, MultiTaskGlobalHead, SegmentationHead
from ag_models.modules.prm import ProgressiveRefinementModule
from ag_models.modules.wavelet import HaarDWT, WaveletAttentionGate


# ---------------------------------------------------------------------------
# 1. WaveCo-MTL Network
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("waveco_mtl", aliases=["waveco_mtl_net", "waveco_mtl_3d"])
class WaveCoMTLNet(BaseModel):
    """
    WaveCo-MTL: Wavelet-Attentive Collaborative Multi-Task Learning Network.
    Supports both 3D (`dims=3`) and 2D (`dims=2`) tensors via dimension-agnostic modules.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 2,
        num_cls_classes: int = 5,
        base_channels: int = 24,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__(
            ModelConfig(
                arch="waveco_mtl",
                dims=dims,
                in_channels=in_channels,
                num_seg_classes=num_seg_classes,
                num_cls_classes=num_cls_classes,
                base_channels=base_channels,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        c = base_channels
        self.dims = dims
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        num_high = 3 if dims == 2 else 7
        conv_cls = get_conv(dims)
        conv_t = get_conv_transpose(dims)

        self.stem = ConvNormAct(in_channels, c, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage1 = ResBlock(c, c, stride=1, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage2 = ResBlock(c, c * 2, stride=2, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage3 = ResBlock(c * 2, c * 4, stride=2, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage4 = ResBlock(c * 4, c * 8, stride=2, dims=dims, norm_type=norm_type, act_type=act_type)

        self.dwt = HaarDWT(dims=dims)
        self.dwt_conv = ConvNormAct(
            c * 4 * num_high, c * 4, kernel_size=1, stride=1, padding=0,
            dims=dims, norm_type=norm_type, act_type=act_type,
        )
        self.sopseg = SOPSegTextureBlock(in_c=c * 8, reduced_c=48, dims=dims, norm_type=norm_type, act_type=act_type)
        self.bottleneck_fusion = ConvNormAct(
            c * 8 + c * 4, c * 8, kernel_size=1, stride=1, padding=0,
            dims=dims, norm_type=norm_type, act_type=act_type,
        )

        self.up4 = conv_t(c * 8, c * 4, kernel_size=2, stride=2)
        self.att4 = WaveletAttentionGate(c * 4, c * 4, c * 2, dims=dims, norm_type=norm_type, act_type=act_type)
        self.dec4 = ResBlock(c * 8, c * 4, dims=dims, norm_type=norm_type, act_type=act_type)

        self.up3 = conv_t(c * 4, c * 2, kernel_size=2, stride=2)
        self.att3 = WaveletAttentionGate(c * 2, c * 2, c, dims=dims, norm_type=norm_type, act_type=act_type)
        self.dec3 = ResBlock(c * 4, c * 2, dims=dims, norm_type=norm_type, act_type=act_type)

        self.up2 = conv_t(c * 2, c, kernel_size=2, stride=2)
        self.att2 = WaveletAttentionGate(c, c, max(1, c // 2), dims=dims, norm_type=norm_type, act_type=act_type)
        self.dec2 = ResBlock(c * 2, c, dims=dims, norm_type=norm_type, act_type=act_type)

        self.mutual_gate = MutualGuidanceGate(c, dims=dims, norm_type=norm_type, act_type=act_type)
        self.head_liver = SegmentationHead(c, 1, mid_channels=max(1, c // 2), dims=dims)
        self.head_tumor = SegmentationHead(c, max(1, num_seg_classes - 1), mid_channels=max(1, c // 2), dims=dims)

        self.cls_head = MaskedFeaturePoolingHead(c * 8, num_classes=num_cls_classes, dims=dims, act_type=act_type)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x0 = self.stem(x)
        e1 = self.stage1(x0)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)

        e4_sop = self.sopseg(e4)
        _, e3_high = self.dwt(e3)
        e3_high_proj = self.dwt_conv(e3_high)
        if e3_high_proj.shape[2:] != e4_sop.shape[2:]:
            e3_high_proj = interpolate_nd(e3_high_proj, size=e4_sop.shape[2:], dims=self.dims)

        b_feat = self.bottleneck_fusion(torch.cat([e4_sop, e3_high_proj], dim=1))

        d4 = self.up4(b_feat)
        d4 = self.dec4(torch.cat([self.att4(d4, e3), d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self.att3(d3, e2), d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(d2, e1), d2], dim=1))

        liv_guided, tum_guided = self.mutual_gate(d2)
        logit_liver = self.head_liver(liv_guided)
        logit_tumor = self.head_tumor(tum_guided)
        seg_logits = torch.cat([logit_liver, logit_tumor], dim=1)

        prob_tumor = torch.sigmoid(logit_tumor[:, 0:1])
        cls_logits = self.cls_head(b_feat, prob_tumor)

        return ModelOutput(
            seg_logits=seg_logits,
            cls_logits=cls_logits,
            latent_feat=b_feat,
            tumor_prob=prob_tumor,
        )


WaveCo_MTLNet = WaveCoMTLNet


# ---------------------------------------------------------------------------
# 2. WaveAttn-SOPSeg & Co-Seg Collaborative Network
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("waveattn_sopseg", aliases=["waveattn_sopseg_liver", "sopseg_liver"])
class WaveAttnSOPSegNet(BaseModel):
    """
    Unified 2D / 3D WaveAttn-SOPSeg Multi-Task Network (with optional Co-Seg Mutual Query Guidance).
    When `use_coseg=True`, behaves identically to `CoSeg_WaveAttn_SOPSegLiverNet`.
    When `use_wavelet=False` and `use_glcam=False`, behaves as standard `SOPSegLiverNet`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 6,
        num_cls_classes: int = 5,
        base_channels: int = 24,
        use_coseg: bool = False,
        use_wavelet: bool = True,
        use_glcam: bool = True,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__(
            ModelConfig(
                arch="coseg_waveattn_sopseg" if use_coseg else "waveattn_sopseg",
                dims=dims,
                in_channels=in_channels,
                num_seg_classes=num_seg_classes,
                num_cls_classes=num_cls_classes,
                base_channels=base_channels,
                norm_type=norm_type,
                act_type=act_type,
                extra={"use_coseg": use_coseg, "use_wavelet": use_wavelet, "use_glcam": use_glcam},
            )
        )
        self.dims = dims
        self.use_coseg = use_coseg
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        conv_cls = get_conv(dims)

        self.stem = ConvNormAct(in_channels, c1, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage1 = ResBlock(c1, c1, stride=1, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage2 = ResBlock(c1, c2, stride=2, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage3 = ResBlock(c2, c3, stride=2, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stage4 = ResBlock(c3, c4, stride=2, dims=dims, norm_type=norm_type, act_type=act_type)

        self.coarse_liver_head = conv_cls(c4, 2, kernel_size=1)
        self.coarse_tumor_head = conv_cls(c4, 2, kernel_size=1)

        if use_coseg:
            self.rp_encoder_liver = RPEncoder(in_channels=1, embed_dim=c4, num_heads=4, dims=dims, norm_type=norm_type)
            self.rp_encoder_tumor = RPEncoder(in_channels=1, embed_dim=c4, num_heads=4, dims=dims, norm_type=norm_type)
            self.query_interact_liver = CoSegQueryInteraction(embed_dim=c4, num_queries=16, num_heads=4)
            self.query_interact_tumor = CoSegQueryInteraction(embed_dim=c4, num_queries=16, num_heads=4)

        self.prm_liver = ProgressiveRefinementModule(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2,
            guidance_channels=in_channels,
            use_wavelet=use_wavelet,
            use_glcam=use_glcam,
            dims=dims,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.prm_tumor = ProgressiveRefinementModule(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2,
            guidance_channels=in_channels,
            use_wavelet=use_wavelet,
            use_glcam=use_glcam,
            dims=dims,
            norm_type=norm_type,
            act_type=act_type,
        )

        num_subtypes = max(1, num_seg_classes - 2)
        self.subtype_head = conv_cls(24, num_subtypes, kernel_size=1)
        self.global_head = MultiTaskGlobalHead(c4, num_cls_classes=num_cls_classes, dims=dims)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        full_spatial = x.shape[2:]
        x0 = self.stem(x)
        s1 = self.stage1(x0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        z = self.stage4(s3)

        init_liver = self.coarse_liver_head(z)
        init_tumor = self.coarse_tumor_head(z)

        m0_liv_full = interpolate_nd(init_liver[:, 0:1], size=full_spatial, dims=self.dims)
        m0_tum_full = interpolate_nd(init_tumor[:, 0:1], size=full_spatial, dims=self.dims)

        if self.use_coseg:
            g_liver = self.rp_encoder_liver(torch.sigmoid(m0_liv_full), z)
            g_tumor = self.rp_encoder_tumor(torch.sigmoid(m0_tum_full), z)
            z_liv = self.query_interact_liver(z + g_tumor)
            z_tum = self.query_interact_tumor(z + g_liver)
        else:
            z_liv, z_tum = z, z

        ref_liver = self.prm_liver(z_liv, [s3, s2, s1], x, init_liver)
        ref_tumor = self.prm_tumor(z_tum, [s3, s2, s1], x, init_tumor)

        liv_p64 = ref_liver["p64"]
        tum_p64 = ref_tumor["p64"]

        pred_liver_mask = liv_p64[:, 0:1]
        pred_liver_edge = liv_p64[:, 1:2]
        pred_tumor_mask = tum_p64[:, 0:1]
        pred_tumor_edge = tum_p64[:, 1:2]

        subtypes = self.subtype_head(ref_tumor["last_feat"])
        bg_logit = -pred_liver_mask
        liver_parenchyma = pred_liver_mask - pred_tumor_mask
        lesion_logits = subtypes + pred_tumor_mask
        full_seg = torch.cat([bg_logit, liver_parenchyma, lesion_logits], dim=1)

        global_out = self.global_head(z)

        return ModelOutput(
            seg_logits=full_seg,
            cls_logits=global_out["cls_logits"],
            tumor_burden=global_out["tumor_burden"],
            refined_liver_mask=pred_liver_mask,
            refined_liver_edge=pred_liver_edge,
            refined_tumor_mask=pred_tumor_mask,
            refined_tumor_edge=pred_tumor_edge,
            pred_liver_mask=pred_liver_mask,
            pred_liver_edge=pred_liver_edge,
            pred_tumor_mask=pred_tumor_mask,
            pred_tumor_edge=pred_tumor_edge,
            coarse_liver_logits=m0_liv_full,
            coarse_tumor_logits=m0_tum_full,
            p16_liver=ref_liver["p16"],
            p32_liver=ref_liver["p32"],
            p16_tumor=ref_tumor["p16"],
            p32_tumor=ref_tumor["p32"],
            latent_feat=z,
        )


@MODEL_REGISTRY.register("coseg_waveattn_sopseg", aliases=["coseg_waveattn_sopseg_liver"])
class CoSegWaveAttnSOPSegNet(WaveAttnSOPSegNet):
    """Combined Co-Seg + WaveAttn-SOPSeg Collaborative Network."""

    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 6,
        num_cls_classes: int = 5,
        base_channels: int = 24,
        dims: int = 3,
    ):
        super().__init__(
            in_channels=in_channels,
            num_seg_classes=num_seg_classes,
            num_cls_classes=num_cls_classes,
            base_channels=base_channels,
            use_coseg=True,
            use_wavelet=True,
            use_glcam=True,
            dims=dims,
        )


WaveAttn_SOPSegLiverNet = WaveAttnSOPSegNet
CoSeg_WaveAttn_SOPSegLiverNet = CoSegWaveAttnSOPSegNet


# ---------------------------------------------------------------------------
# 3. WeedSense Dual-Path + UIB + BGA + Transformer Global Decoder Network
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("weedsense", aliases=["weedsense_3d", "weedsense_liver"])
class WeedSenseNet(BaseModel):
    """
    Unified 2D / 3D WeedSense Architecture:
    - Detail Branch (Shallow-Wide)
    - Semantic Branch (Deep-Narrow with UIB blocks + Context Embedding)
    - Bilateral Guided Aggregation (BGA)
    - Transformer Global Multi-Task Decoder (TGD)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 6,
        num_cls_classes: int = 5,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__(
            ModelConfig(
                arch="weedsense",
                dims=dims,
                in_channels=in_channels,
                num_seg_classes=num_seg_classes,
                num_cls_classes=num_cls_classes,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        self.dims = dims
        conv_cls = get_conv(dims)

        # Detail Branch (1/8 resolution, 64 channels)
        self.detail = nn.Sequential(
            ConvNormAct(in_channels, 32, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type),
            ConvNormAct(32, 32, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type),
            ConvNormAct(32, 32, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type),
            ConvNormAct(32, 32, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type),
            ConvNormAct(32, 64, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type),
            ConvNormAct(64, 64, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type),
        )

        # Semantic Branch Stem (1/4 resolution, 16 channels)
        self.stem_conv1 = ConvNormAct(in_channels, 16, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type)
        self.stem_left = nn.Sequential(
            ConvNormAct(16, 8, 1, 1, 0, dims=dims, norm_type=norm_type, act_type=act_type),
            ConvNormAct(8, 16, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type),
        )
        self.stem_right = get_max_pool(dims=dims, kernel_size=3, stride=2, padding=1)
        self.stem_fuse = ConvNormAct(32, 16, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type)

        # Semantic UIB Stages
        self.s3 = nn.Sequential(
            UIBBlock(16, 32, stride=2, exp_ratio=4, dims=dims, norm_type=norm_type, act_type=act_type),
            UIBBlock(32, 32, stride=1, exp_ratio=4, dims=dims, norm_type=norm_type, act_type=act_type),
        )
        self.s4 = nn.Sequential(
            UIBBlock(32, 64, stride=2, exp_ratio=4, dims=dims, norm_type=norm_type, act_type=act_type),
            UIBBlock(64, 64, stride=1, exp_ratio=4, dims=dims, norm_type=norm_type, act_type=act_type),
        )
        self.s5_1 = nn.Sequential(
            UIBBlock(64, 64, stride=2, exp_ratio=4, dims=dims, norm_type=norm_type, act_type=act_type),
            UIBBlock(64, 64, stride=1, exp_ratio=4, dims=dims, norm_type=norm_type, act_type=act_type),
        )
        self.context_pool = get_adaptive_avg_pool(dims, 1)
        self.context_conv1 = nn.Sequential(conv_cls(64, 64, 1, bias=True), get_activation(act_type))
        self.context_conv2 = ConvNormAct(64, 64, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type)

        # Bilateral Guided Aggregation
        self.bga = BGALayer(channels=64, dims=dims, norm_type=norm_type, act_type=act_type)

        # Segmentation Head
        self.seg_head = nn.Sequential(
            ConvNormAct(64, 64, 3, 1, 1, dims=dims, norm_type=norm_type, act_type=act_type),
            conv_cls(64, num_seg_classes, kernel_size=1),
        )
        self.aux_head_s4 = conv_cls(64, num_seg_classes, kernel_size=1)

        # Transformer Global Decoder (TGD)
        self.tgd_attn = TransformerBlock(embed_dim=64, num_heads=4, dim_feedforward=128, dropout=0.1)
        self.global_head = MultiTaskGlobalHead(64, num_cls_classes=num_cls_classes, hidden_dim=64, dims=dims)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        orig_size = x.shape[2:]
        x_d = self.detail(x)

        s0 = self.stem_conv1(x)
        x2 = self.stem_fuse(torch.cat([self.stem_left(s0), self.stem_right(s0)], dim=1))
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        x5_1 = self.s5_1(x4)
        x5_2 = self.context_conv2(x5_1 + self.context_conv1(self.context_pool(x5_1)))

        bga_feat = self.bga(x_d, x5_2)
        seg_logits = interpolate_nd(self.seg_head(bga_feat), size=orig_size, dims=self.dims)
        aux_logits = interpolate_nd(self.aux_head_s4(x4), size=orig_size, dims=self.dims)

        tokens = x5_2.flatten(2).transpose(1, 2)
        tgd_tokens = self.tgd_attn(tokens)
        pooled_tgd = tgd_tokens.mean(dim=1)
        global_out = self.global_head(pooled_tgd)

        return ModelOutput(
            seg_logits=seg_logits,
            aux_seg_logits=aux_logits,
            cls_logits=global_out["cls_logits"],
            tumor_burden=global_out["tumor_burden"],
            latent_feat=x5_2,
        )


WeedSenseLiverNet = WeedSenseNet


# ---------------------------------------------------------------------------
# 4. MultiTaskMSFFNet (ECAdd + DFE Attention)
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("msff_mtl", aliases=["multitask_msff_net", "ednet_mtl"])
class MultiTaskMSFFNet(BaseModel):
    """
    Unified 2D / 3D Multi-Scale Feature Fusion Network with ECAdd Blocks and DFE Skip Attention.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 6,
        num_cls_classes: int = 5,
        base_channels: int = 16,
        dims: int = 3,
        norm_type: str = "instancenorm",
        act_type: str = "leakyrelu",
    ):
        super().__init__(
            ModelConfig(
                arch="msff_mtl",
                dims=dims,
                in_channels=in_channels,
                num_seg_classes=num_seg_classes,
                num_cls_classes=num_cls_classes,
                base_channels=base_channels,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        c = base_channels
        self.dims = dims
        conv_cls = get_conv(dims)
        conv_t = get_conv_transpose(dims)

        self.enc1 = ECAddBlock(in_channels, c, dims=dims, norm_type=norm_type, act_type=act_type)
        self.down1 = ConvNormAct(c, c, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type)

        self.enc2 = ECAddBlock(c, c * 2, dims=dims, norm_type=norm_type, act_type=act_type)
        self.down2 = ConvNormAct(c * 2, c * 2, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type)

        self.enc3 = ECAddBlock(c * 2, c * 4, dims=dims, norm_type=norm_type, act_type=act_type)
        self.down3 = ConvNormAct(c * 4, c * 4, 3, 2, 1, dims=dims, norm_type=norm_type, act_type=act_type)

        self.bottleneck = ECAddBlock(c * 4, c * 8, dims=dims, norm_type=norm_type, act_type=act_type)

        self.dfe3 = DFEAttention(c * 4, c * 8, c * 4, dims=dims, norm_type=norm_type)
        self.up3 = conv_t(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ECAddBlock(c * 8, c * 4, dims=dims, norm_type=norm_type, act_type=act_type)

        self.dfe2 = DFEAttention(c * 2, c * 4, c * 2, dims=dims, norm_type=norm_type)
        self.up2 = conv_t(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ECAddBlock(c * 4, c * 2, dims=dims, norm_type=norm_type, act_type=act_type)

        self.dfe1 = DFEAttention(c, c * 2, c, dims=dims, norm_type=norm_type)
        self.up1 = conv_t(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ECAddBlock(c * 2, c, dims=dims, norm_type=norm_type, act_type=act_type)

        self.seg_head = conv_cls(c, num_seg_classes, kernel_size=1)
        self.aux_seg_head = conv_cls(c * 2, num_seg_classes, kernel_size=1)

        self.cls_pool_avg = get_adaptive_avg_pool(dims, 1)
        self.cls_pool_max = get_adaptive_max_pool(dims, 1)
        self.cls_head = nn.Sequential(
            nn.Linear((c * 8 + c * 4) * 2, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(c * 4, num_cls_classes),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), self.dfe3(e3, b)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.dfe2(e2, d3)], dim=1))
        aux_logits = interpolate_nd(self.aux_seg_head(d2), size=x.shape[2:], dims=self.dims)
        d1 = self.dec1(torch.cat([self.up1(d2), self.dfe1(e1, d2)], dim=1))
        seg_logits = self.seg_head(d1)

        cls_feat = torch.cat(
            [
                self.cls_pool_avg(b).flatten(1),
                self.cls_pool_max(b).flatten(1),
                self.cls_pool_avg(d3).flatten(1),
                self.cls_pool_max(d3).flatten(1),
            ],
            dim=1,
        )
        cls_logits = self.cls_head(cls_feat)

        return ModelOutput(
            seg_logits=seg_logits,
            aux_seg_logits=aux_logits,
            cls_logits=cls_logits,
            latent_feat=b,
        )


MultiTaskLiverNet_MSFF = MultiTaskMSFFNet


# ---------------------------------------------------------------------------
# 5. ResVNetFusion (V-Net + ICFF + ResNet)
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("resvnet_fusion")
class ResVNetFusion(BaseModel):
    """
    Unified 2D / 3D V-Net Segmentation + Inter-Network Channel Feature Fusion (ICFF) + ResNet Classifier.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 2,
        num_cls_classes: int = 5,
        base_channels: int = 16,
        dims: int = 3,
        norm_type: str = "instancenorm",
    ):
        super().__init__(
            ModelConfig(
                arch="resvnet_fusion",
                dims=dims,
                in_channels=in_channels,
                num_seg_classes=num_seg_classes,
                num_cls_classes=num_cls_classes,
                base_channels=base_channels,
                norm_type=norm_type,
            )
        )
        c = base_channels
        self.dims = dims
        self.num_seg_classes = num_seg_classes
        conv_cls = get_conv(dims)
        conv_t = get_conv_transpose(dims)

        self.seg_enc1 = ResBlock(in_channels, c, dims=dims, norm_type=norm_type, act_type="relu")
        self.seg_down1 = conv_cls(c, c * 2, kernel_size=2, stride=2)
        self.seg_enc2 = ResBlock(c * 2, c * 2, dims=dims, norm_type=norm_type, act_type="relu")
        self.seg_down2 = conv_cls(c * 2, c * 4, kernel_size=2, stride=2)
        self.seg_enc3 = ResBlock(c * 4, c * 4, dims=dims, norm_type=norm_type, act_type="relu")
        self.seg_down3 = conv_cls(c * 4, c * 8, kernel_size=2, stride=2)
        self.seg_bottleneck = ResBlock(c * 8, c * 8, dims=dims, norm_type=norm_type, act_type="relu")

        self.seg_up3 = conv_t(c * 8, c * 4, kernel_size=2, stride=2)
        self.seg_dec3 = ResBlock(c * 8, c * 4, dims=dims, norm_type=norm_type, act_type="relu")
        self.seg_up2 = conv_t(c * 4, c * 2, kernel_size=2, stride=2)
        self.seg_dec2 = ResBlock(c * 4, c * 2, dims=dims, norm_type=norm_type, act_type="relu")
        self.seg_up1 = conv_t(c * 2, c, kernel_size=2, stride=2)
        self.seg_dec1 = ResBlock(c * 2, c, dims=dims, norm_type=norm_type, act_type="relu")
        self.seg_out = conv_cls(c, num_seg_classes, kernel_size=1)

        self.icff1 = InterNetworkChannelFeatureFusion(c * 2, c * 2, c * 2, dims=dims, norm_type=norm_type)
        self.icff2 = InterNetworkChannelFeatureFusion(c * 4, c * 4, c * 4, dims=dims, norm_type=norm_type)
        self.icff3 = InterNetworkChannelFeatureFusion(c * 8, c * 8, c * 8, dims=dims, norm_type=norm_type)

        self.cls_init = ConvNormAct(in_channels, c, 3, 1, 1, dims=dims, norm_type=norm_type, act_type="relu")
        self.cls_stage1 = ResBlock(c, c * 2, stride=2, dims=dims, norm_type=norm_type, act_type="relu")
        self.cls_stage2 = ResBlock(c * 2, c * 4, stride=2, dims=dims, norm_type=norm_type, act_type="relu")
        self.cls_stage3 = ResBlock(c * 4, c * 8, stride=2, dims=dims, norm_type=norm_type, act_type="relu")

        self.cls_head = nn.Sequential(
            get_adaptive_avg_pool(dims, 1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c * 4, num_cls_classes),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        s1 = self.seg_enc1(x)
        s2 = self.seg_enc2(self.seg_down1(s1))
        s3 = self.seg_enc3(self.seg_down2(s2))
        sb = self.seg_bottleneck(self.seg_down3(s3))

        d3 = self.seg_dec3(torch.cat([self.seg_up3(sb), s3], dim=1))
        d2 = self.seg_dec2(torch.cat([self.seg_up2(d3), s2], dim=1))
        d1 = self.seg_dec1(torch.cat([self.seg_up1(d2), s1], dim=1))
        seg_logits = self.seg_out(d1)

        p_tumor = torch.sigmoid(seg_logits[:, 1:2]) if self.num_seg_classes >= 2 else torch.sigmoid(seg_logits)
        x_masked = x * p_tumor

        c0 = self.cls_init(x_masked)
        c1 = self.icff1(d2, self.cls_stage1(c0))
        c2 = self.icff2(d3, self.cls_stage2(c1))
        c3 = self.icff3(sb, self.cls_stage3(c2))
        cls_logits = self.cls_head(c3)

        return ModelOutput(
            seg_logits=seg_logits,
            cls_logits=cls_logits,
            p_tumor=p_tumor,
            latent_feat=sb,
        )
