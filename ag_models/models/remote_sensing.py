"""
Standardized Remote Sensing Object/Building Detection, Bi-Temporal Change Detection,
Building Damage Assessment, and Change Captioning Model Architectures (2022–2026).

All 9 models inherit from `BaseModel`, return `ModelOutput` (supporting both attribute
and dictionary access), and are registered in `MODEL_REGISTRY`:
1. `ImprovedMaskRCNNBuilding` (Han et al., Complex & Intelligent Systems, 2022)
2. `BiSAMCD` (Qin et al., IEEE TGRS, 2025)
3. `SAM2CD` (Qin et al., IEEE JSTARS, 2025)
4. `GMFIMamba` (Xu et al., Engineering Applications of Artificial Intelligence, 2026)
5. `SemiBCD` (Liu et al., IEEE TGRS, 2026)
6. `CGCCENet` (Wang et al., Information Sciences, 2026)
7. `KGBDCNet` (Wang et al., ISPRS J. Photogramm. Remote Sens., 2026)
8. `CBRNet` (Huang et al., IEEE TGRS, 2026)
9. `EdgeRefNet` (Hussin et al., IEEE TGRS, 2026)
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ag_models.core.base import BaseModel, ModelConfig, ModelOutput
from ag_models.core.registry import MODEL_REGISTRY
from ag_models.modules.adapters import inject_lora
from ag_models.modules.attention import GLCAMContextAttention2D
from ag_models.modules.remote_sensing import (
    ActivationSelectionGate,
    BFSCAISequenceGenerator,
    BiFPNFeatureFusion,
    CAGMModule,
    CASCBlock,
    CFDDecoderBlock,
    CGFEModule,
    CGRRBranch,
    ConvVSSBlock,
    DeepLabV3PlusCDHead,
    DSBRDecoder,
    EDBBlock,
    EdgeGuidedCrossAttention,
    EEMBlock,
    GCCMModule,
    GMBFIMBlock,
    LightSSHead,
    LWCAAttention,
    MCFEMBlock,
    MSFFMBlock,
    RetinexShadowEnhancer,
    SAMFoundationAdapter,
    SAM3FoundationAdapter,
    SCEMModule,
    SingleImageEdgeRefBranch,
    SSMBBlock,
    TOTCCSCVVerifier,
    TTTBBlock,
    UAPLRModule,
    UNetTopDownFPNFusion,
    HIFEResNetFPN,
    DSTCDSTFTBBlock,
    DSTCDSEFTBBlock,
    GLMambaVSSBlock,
    GLMambaCSAMModule,
    GLMambaBFAModule,
    GLMambaCSIEModule,
    Seg2ChangeBDFM,
    Seg2ChangeEDQA,
    Seg2ChangeResUpBlock,
)
from ag_models.modules.wavelet import WaveFusionBlock


# =============================================================================
# Shared Multi-Scale Pyramid Encoder Helper
# =============================================================================


class ResNetBottleneckStage(nn.Module):
    """3-conv residual block stage matching ResNet-50/101 & Han et al. (2022)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, num_blocks: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        for idx in range(num_blocks):
            s = stride if idx == 0 else 1
            c_in = in_channels if idx == 0 else out_channels
            mid_c = max(out_channels // 2, 16)
            layers.append(_ThreeConvResBlock(c_in, mid_c, out_channels, stride=s))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _ThreeConvResBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class FourStagePyramidBackbone(nn.Module):
    """
    Configurable 4-stage hierarchical pyramid encoder producing [C1, C2, C3, C4] at strides [2, 4, 8, 16].
    Supports both lightweight from-scratch residual stages (`pretrained=False`) and ImageNet-pretrained
    ResNet-34 / ResNet-50 backbones (`pretrained=True`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
        pretrained: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        super().__init__()
        self.pretrained = bool(pretrained and in_channels == 3)
        self.pretrained_arch = pretrained_arch
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

        if self.pretrained:
            import torchvision.models as tvm

            if pretrained_arch == "resnet50":
                net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
                enc_c = (256, 512, 1024, 2048)
            else:
                net = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
                enc_c = (64, 128, 256, 512)

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
            # Preserve stride-2 spatial resolution at stage1 for sharp remote-sensing building boundaries
            self.stage1 = net.layer1
            self.stage2 = net.layer2
            self.stage3 = net.layer3
            self.stage4 = net.layer4
            self.lateral_adapters = nn.ModuleList(
                [
                    nn.Identity()
                    if ec == sc
                    else nn.Sequential(
                        nn.Conv2d(ec, sc, kernel_size=1, bias=False),
                        nn.BatchNorm2d(sc),
                        nn.ReLU(inplace=True),
                    )
                    for ec, sc in zip(enc_c, stage_channels)
                ]
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stage_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stage_channels[0]),
                nn.ReLU(inplace=True),
            )
            self.stage1 = ResNetBottleneckStage(
                stage_channels[0], stage_channels[0], stride=1, num_blocks=stage_blocks[0]
            )
            self.stage2 = ResNetBottleneckStage(
                stage_channels[0], stage_channels[1], stride=2, num_blocks=stage_blocks[1]
            )
            self.stage3 = ResNetBottleneckStage(
                stage_channels[1], stage_channels[2], stride=2, num_blocks=stage_blocks[2]
            )
            self.stage4 = ResNetBottleneckStage(
                stage_channels[2], stage_channels[3], stride=2, num_blocks=stage_blocks[3]
            )
            self.lateral_adapters = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.pretrained:
            x = (x - self.imagenet_mean) / self.imagenet_std
        c0 = self.stem(x)
        raw1 = self.stage1(c0)
        raw2 = self.stage2(raw1)
        raw3 = self.stage3(raw2)
        raw4 = self.stage4(raw3)
        return [
            self.lateral_adapters[0](raw1),
            self.lateral_adapters[1](raw2),
            self.lateral_adapters[2](raw3),
            self.lateral_adapters[3](raw4),
        ]


# =============================================================================
# 1. Han et al. (2022) - Improved Mask R-CNN Building Detection
# =============================================================================


@MODEL_REGISTRY.register("improved_maskrcnn_building", aliases=["maskrcnn_building"])
class ImprovedMaskRCNNBuilding(BaseModel):
    """
    Improved Mask R-CNN for Remote Sensing Building Detection
    (Han, Yin, Zheng, & Chen, Complex & Intelligent Systems, Vol. 8, pp. 1847–1855, 2022).

    Key innovations implemented:
    1. Differentiable HSV/Multi-Scale Retinex shadow removal & contrast screening (`RetinexShadowEnhancer`).
    2. Modified 3-conv residual block backbone (`FourStagePyramidBackbone`).
    3. U-Net-inspired Top-Down Feature Fusion (`UNetTopDownFPNFusion`: 1x1 conv + 2x upsample + add).
    4. Fully Convolutional (FCN) 1024-channel RPN + Box Regression + Classification + Building Mask Heads
       replacing parameter-heavy fully connected layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        fcn_channels: int = 256,
        use_retinex: bool = True,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        config = ModelConfig(
            arch="improved_maskrcnn_building",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            extra={
                "fcn_channels": fcn_channels,
                "use_retinex": use_retinex,
                "pretrained_backbone": pretrained_backbone,
                "pretrained_arch": pretrained_arch,
            },
        )
        super().__init__(config=config)
        self.use_retinex = use_retinex
        self.retinex = RetinexShadowEnhancer() if use_retinex else nn.Identity()

        stage_c = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.backbone = FourStagePyramidBackbone(
            in_channels=in_channels,
            stage_channels=stage_c,
            pretrained=pretrained_backbone,
            pretrained_arch=pretrained_arch,
        )
        self.fpn = UNetTopDownFPNFusion(in_channels_list=stage_c, out_channels=base_channels * 2)

        fpn_c = base_channels * 2
        # Fully Convolutional (FCN) Mask & Proposal Heads (replacing FC layers per Section 3.3)
        self.rpn_objectness_head = nn.Conv2d(fpn_c, 1, kernel_size=1)
        self.rpn_bbox_head = nn.Conv2d(fpn_c, 4, kernel_size=1)

        self.fcn_mask_head = nn.Sequential(
            nn.Conv2d(fpn_c, fcn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fcn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fcn_channels, fcn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fcn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fcn_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h, w = x.shape[-2:]
        x_enh = self.retinex(x)
        pyramid_feats = self.backbone(x_enh)
        fused_feats = self.fpn(pyramid_feats)

        # Top-resolution fused feature map (stride 2)
        p1 = fused_feats[0]
        mask_logits_s2 = self.fcn_mask_head(p1)
        seg_logits = F.interpolate(mask_logits_s2, size=(h, w), mode="bilinear", align_corners=False)

        rpn_obj = self.rpn_objectness_head(p1)
        rpn_bbox = torch.sigmoid(self.rpn_bbox_head(p1))

        return ModelOutput(
            seg_logits=seg_logits,
            rpn_logits=F.interpolate(rpn_obj, size=(h, w), mode="bilinear", align_corners=False),
            bbox_pred=rpn_bbox,
            enhanced_input=x_enh,
        )


# =============================================================================
# 2. Qin et al. (2025) - BiSAM-CD (Zero-Shot Bidirectional Temporal Memory CD)
# =============================================================================


@MODEL_REGISTRY.register("bisam_cd")
class BiSAMCD(BaseModel):
    """
    BiSAM-CD: Zero-Shot Remote Sensing Change Detection via Bidirectional Temporal Memory in SAM2
    (Qin, Chen, Wang, & Pan, IEEE TGRS, Vol. 63, Art. no. 4417812, 2025).

    Supports two execution modes:
    1. Pluggable External SAM2 VideoPredictor + AutomaticMaskGenerator (`external_sam2_runner`),
       executing exact instance mask tracking and cross-sequence IoU verification.
    2. Built-in Self-Contained Bidirectional Temporal Memory Feature Tracker (`memory_encoder` +
       `BFSCAISequenceGenerator` + `TOTCCSCVVerifier`), which runs out-of-the-box without requiring
       a 2 GB external SAM2 checkpoint download and also supports optional fine-tuning.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_interp_frames: int = 6,
        beta: float = 0.15,
        iou_threshold: float = 0.82,
        cross_iou_threshold: float = 0.5,
        external_sam2_runner: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        config = ModelConfig(
            arch="bisam_cd",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=1,
            base_channels=embed_dim,
            extra={
                "num_interp_frames": num_interp_frames,
                "beta": beta,
                "iou_threshold": iou_threshold,
                "cross_iou_threshold": cross_iou_threshold,
            },
        )
        super().__init__(config=config)
        self.external_sam2_runner = external_sam2_runner
        self.bfscai = BFSCAISequenceGenerator(num_interp_frames=num_interp_frames, beta=beta)
        self.verifier = TOTCCSCVVerifier(
            iou_threshold=iou_threshold, cross_iou_threshold=cross_iou_threshold
        )

        # Self-contained spatial-temporal memory encoder & refinement head
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
        )
        self.memory_gru = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.memory_gru.weight)
        nn.init.constant_(self.memory_gru.bias, 1.5)
        self.refine_head = nn.Sequential(
            nn.Conv2d(embed_dim * 2 + 3, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, kernel_size=1),
        )
        # Initialize refinement residual weight small so zero-shot TOTC-CSCV dominates prior to any fine-tuning
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)

    def _track_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Propagates temporal memory across frame sequence seq: (B, T, C, H, W) -> (B, T, D, H, W).
        """
        b, t, c, h, w = seq.shape
        feats = self.frame_encoder(seq.view(b * t, c, h, w)).view(b, t, -1, h, w)
        tracked: List[torch.Tensor] = []
        mem = feats[:, 0]
        tracked.append(mem)
        for step in range(1, t):
            cur = feats[:, step]
            gate = torch.sigmoid(self.memory_gru(torch.cat([cur, mem], dim=1)))
            mem = gate * cur + (1.0 - gate) * mem
            tracked.append(mem)
        return torch.stack(tracked, dim=1)

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            # Support stacked (B, 2, C, H, W) or concatenated (B, 2C, H, W) inputs
            if img_t1.ndim == 5:
                img_t1, img_t2 = img_t1[:, 0], img_t1[:, 1]
            else:
                img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        if self.external_sam2_runner is not None:
            ext_mask = self.external_sam2_runner(img_t1, img_t2)
            ext_logits = torch.logit(ext_mask.clamp(1e-4, 1.0 - 1e-4))
            return ModelOutput(seg_logits=ext_logits, change_prob=ext_mask)

        seq_asc, seq_desc = self.bfscai(img_t1, img_t2)
        asc_traj = self._track_sequence(seq_asc)
        desc_traj = self._track_sequence(seq_desc)

        ver_out = self.verifier(asc_traj, desc_traj)
        zs_logits = ver_out["change_logits"]

        # Combine zero-shot tracking consistency logits with optional learnable boundary refinement
        diff_feat = torch.abs(asc_traj[:, 0] - desc_traj[:, 0])
        concat_cue = torch.cat(
            [
                diff_feat,
                asc_traj[:, -1] - desc_traj[:, -1],
                ver_out["change_prob"],
                ver_out["asc_change_prob"],
                ver_out["desc_change_prob"],
            ],
            dim=1,
        )
        seg_logits = zs_logits + self.refine_head(concat_cue)

        return ModelOutput(
            seg_logits=seg_logits,
            zero_shot_logits=zs_logits,
            change_prob=torch.sigmoid(seg_logits),
            asc_change_prob=ver_out["asc_change_prob"],
            desc_change_prob=ver_out["desc_change_prob"],
        )


# =============================================================================
# 3. Qin et al. (2025) - SAM2-CD (Remote Sensing Image Change Detection With SAM2)
# =============================================================================


@MODEL_REGISTRY.register("sam2_cd")
class SAM2CD(BaseModel):
    """
    SAM2-CD: Remote Sensing Image Change Detection With SAM2
    (Qin, Wang, Fan, & Pan, IEEE JSTARS, Vol. 18, pp. 24575–24587, 2025).

    Architecture components:
    1. Shared 4-stage hierarchical encoder with Activation Selection Gate (`ActivationSelectionGate` / ASG)
       in global stages to exchange cross-temporal information.
    2. Multiscale Feature Fusion Module (`MSFFMBlock`) aligning all 4 scales via pooling/interpolation +
       depthwise separable convolutions and extracting temporal difference maps D^1..D^4.
    3. Wavelet Feature Fusion (`WaveFusionBlock`) using 2D Haar DWT/IDWT on deep features D^3, D^4.
    4. Progressive Top-Down Decoder equipped with Global-Local Context Attention (`GLCAMContextAttention2D`)
       and multi-scale deep supervision outputs (`seg_logits`, `aux_seg_logits`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        config = ModelConfig(
            arch="sam2_cd",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            extra={"pretrained_backbone": pretrained_backbone, "pretrained_arch": pretrained_arch},
        )
        super().__init__(config=config)
        stage_c = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.encoder = FourStagePyramidBackbone(
            in_channels=in_channels,
            stage_channels=stage_c,
            pretrained=pretrained_backbone,
            pretrained_arch=pretrained_arch,
        )
        self.backbone = self.encoder

        # ASG gates at global stages (stage 2, 3, 4)
        self.asg2 = ActivationSelectionGate(stage_c[1])
        self.asg3 = ActivationSelectionGate(stage_c[2])
        self.asg4 = ActivationSelectionGate(stage_c[3])

        dec_c = base_channels
        self.msffm = MSFFMBlock(in_channels_list=stage_c, out_channels=dec_c)

        # WaveFusion on deep features D^3 and D^4
        self.wave_fusion = WaveFusionBlock(in_channels=dec_c * 2, out_channels=dec_c, dims=2)

        # Top-down Decoder with GLCAM context attention
        self.glcam4 = GLCAMContextAttention2D(dec_c)
        self.glcam3 = GLCAMContextAttention2D(dec_c)
        self.glcam2 = GLCAMContextAttention2D(dec_c)
        self.glcam1 = GLCAMContextAttention2D(dec_c)

        self.dec_fuse3 = nn.Sequential(
            nn.Conv2d(dec_c * 2, dec_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_c),
            nn.ReLU(inplace=True),
        )
        self.dec_fuse2 = nn.Sequential(
            nn.Conv2d(dec_c * 2, dec_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_c),
            nn.ReLU(inplace=True),
        )
        self.dec_fuse1 = nn.Sequential(
            nn.Conv2d(dec_c * 2, dec_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_c),
            nn.ReLU(inplace=True),
        )

        self.head_p1 = nn.Conv2d(dec_c, num_classes, kernel_size=1)
        self.head_p2 = nn.Conv2d(dec_c, num_classes, kernel_size=1)
        self.head_p3 = nn.Conv2d(dec_c, num_classes, kernel_size=1)

    def _encode_pair(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.encoder.pretrained:
            x1 = (x1 - self.encoder.imagenet_mean) / self.encoder.imagenet_std
            x2 = (x2 - self.encoder.imagenet_mean) / self.encoder.imagenet_std

        f1_0 = self.encoder.stem(x1)
        f2_0 = self.encoder.stem(x2)

        raw1_1 = self.encoder.stage1(f1_0)
        raw2_1 = self.encoder.stage1(f2_0)
        c1_1 = self.encoder.lateral_adapters[0](raw1_1)
        c2_1 = self.encoder.lateral_adapters[0](raw2_1)

        raw1_2 = self.encoder.stage2(raw1_1)
        raw2_2 = self.encoder.stage2(raw2_1)
        c1_2 = self.encoder.lateral_adapters[1](raw1_2)
        c2_2 = self.encoder.lateral_adapters[1](raw2_2)
        c1_2, c2_2, _ = self.asg2(c1_2, c2_2)

        raw1_3 = self.encoder.stage3(raw1_2)
        raw2_3 = self.encoder.stage3(raw2_2)
        c1_3 = self.encoder.lateral_adapters[2](raw1_3)
        c2_3 = self.encoder.lateral_adapters[2](raw2_3)
        c1_3, c2_3, _ = self.asg3(c1_3, c2_3)

        raw1_4 = self.encoder.stage4(raw1_3)
        raw2_4 = self.encoder.stage4(raw2_3)
        c1_4 = self.encoder.lateral_adapters[3](raw1_4)
        c2_4 = self.encoder.lateral_adapters[3](raw2_4)
        c1_4, c2_4, _ = self.asg4(c1_4, c2_4)

        return [c1_1, c1_2, c1_3, c1_4], [c2_1, c2_2, c2_3, c2_4]

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            if img_t1.ndim == 5:
                img_t1, img_t2 = img_t1[:, 0], img_t1[:, 1]
            else:
                img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        h, w = img_t1.shape[-2:]
        feats_t1, feats_t2 = self._encode_pair(img_t1, img_t2)
        d1, d2, d3, d4 = self.msffm(feats_t1, feats_t2)

        # Wavelet fusion on deep stages D^3 and upsampled D^4
        d4_up_d3 = F.interpolate(d4, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        d_global = self.wave_fusion(torch.cat([d3, d4_up_d3], dim=1))

        p4 = self.glcam4(d4)
        p3_in = F.interpolate(p4, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.glcam3(self.dec_fuse3(torch.cat([d_global, p3_in], dim=1)))

        p2_in = F.interpolate(p3, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.glcam2(self.dec_fuse2(torch.cat([d2, p2_in], dim=1)))

        p1_in = F.interpolate(p2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.glcam1(self.dec_fuse1(torch.cat([d1, p1_in], dim=1)))

        seg_logits = F.interpolate(self.head_p1(p1), size=(h, w), mode="bilinear", align_corners=False)
        aux_p2 = F.interpolate(self.head_p2(p2), size=(h, w), mode="bilinear", align_corners=False)
        aux_p3 = F.interpolate(self.head_p3(p3), size=(h, w), mode="bilinear", align_corners=False)

        return ModelOutput(
            seg_logits=seg_logits,
            aux_seg_logits=aux_p2,
            mask_p2=aux_p2,
            mask_p3=aux_p3,
        )


# =============================================================================
# 4. Xu et al. (2026) - GMFIMamba (Group Mamba Feature Interaction for BCD & BDA)
# =============================================================================


@MODEL_REGISTRY.register("gmfimamba")
class GMFIMamba(BaseModel):
    """
    GMFIMamba: Remote sensing change detection based on group Mamba feature interaction
    (Xu, Chen, Bi, Wang, & Shu, Engineering Applications of Artificial Intelligence, Vol. 163, 2026).

    Supports both:
    - Binary Building Change Detection (`task_mode="bcd"` -> `seg_logits`)
    - Two-Stage Building Localization & Multi-Class Damage Assessment (`task_mode="bda"` ->
      `seg_logits` for 4 damage levels + `loc_logits` for pre-disaster building footprint).

    Architecture components:
    1. ConvMamba Encoder with 4 stages of `ConvVSSBlock` (local Conv + global SS2D + Channel Shuffle).
    2. `MCFEMBlock` (Multi-branch Dilated Convolution Feature Enhancement Module) at each stage.
    3. `GMBFIMBlock` (Grouped Mamba-based Bitemporal Feature Interaction Module) interleaving
       pre/post channels into 4 groups with parallel VSS blocks and difference gating.
    4. Multi-level Mamba Decoder with VSS blocks and residual fusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        stage_depths: Sequence[int] = (1, 1, 2, 1),
        task_mode: str = "bcd",
        num_damage_classes: int = 5,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        config = ModelConfig(
            arch="gmfimamba",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_damage_classes if task_mode == "bda" else num_classes,
            base_channels=base_channels,
            extra={
                "task_mode": task_mode,
                "stage_depths": list(stage_depths),
                "pretrained_backbone": pretrained_backbone,
                "pretrained_arch": pretrained_arch,
            },
        )
        super().__init__(config=config)
        self.task_mode = task_mode.lower()
        self.pretrained_backbone = bool(pretrained_backbone)
        dims = [base_channels * (2 ** i) for i in range(4)]

        if self.pretrained_backbone:
            self.backbone = FourStagePyramidBackbone(
                in_channels=in_channels,
                stage_channels=dims,
                pretrained=True,
                pretrained_arch=pretrained_arch,
            )
        else:
            # Stem + 4-stage ConvMamba Encoder
            self.downsample_layers = nn.ModuleList()
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[0]),
                    nn.ReLU(inplace=True),
                )
            )
            for i in range(3):
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(dims[i + 1]),
                        nn.ReLU(inplace=True),
                    )
                )

            self.encoder_stages = nn.ModuleList(
                [
                    nn.Sequential(*[ConvVSSBlock(dims[i]) for _ in range(stage_depths[i])])
                    for i in range(4)
                ]
            )
        self.mcfem_blocks = nn.ModuleList([MCFEMBlock(dims[i]) for i in range(4)])
        self.gmbfim_blocks = nn.ModuleList([GMBFIMBlock(dims[i], num_groups=4) for i in range(4)])

        # Multi-level Mamba Decoder
        self.dec_vss4 = ConvVSSBlock(dims[3])
        self.dec_up4 = nn.Conv2d(dims[3], dims[2], kernel_size=1, bias=False)

        self.dec_vss3 = ConvVSSBlock(dims[2])
        self.dec_up3 = nn.Conv2d(dims[2], dims[1], kernel_size=1, bias=False)

        self.dec_vss2 = ConvVSSBlock(dims[1])
        self.dec_up2 = nn.Conv2d(dims[1], dims[0], kernel_size=1, bias=False)

        self.dec_vss1 = ConvVSSBlock(dims[0])

        out_c = num_damage_classes if self.task_mode == "bda" else num_classes
        self.change_head = nn.Conv2d(dims[0], out_c, kernel_size=1)

        if self.task_mode == "bda":
            self.loc_head = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[0], 1, kernel_size=1),
            )

    def _encode_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        if getattr(self, "pretrained_backbone", False):
            feats = self.backbone(x)
            return [mcfem(f) for mcfem, f in zip(self.mcfem_blocks, feats)]
        feats: List[torch.Tensor] = []
        cur = x
        for down, stage, mcfem in zip(self.downsample_layers, self.encoder_stages, self.mcfem_blocks):
            cur = down(cur)
            cur = stage(cur)
            feats.append(mcfem(cur))
        return feats

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            if img_t1.ndim == 5:
                img_t1, img_t2 = img_t1[:, 0], img_t1[:, 1]
            else:
                img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        h, w = img_t1.shape[-2:]
        feats_t1 = self._encode_single(img_t1)
        feats_t2 = self._encode_single(img_t2)

        f1, f2, f3, f4 = [
            gmbfim(t1, t2) for gmbfim, t1, t2 in zip(self.gmbfim_blocks, feats_t1, feats_t2)
        ]

        d4 = self.dec_vss4(f4)
        d3_in = F.interpolate(self.dec_up4(d4), size=f3.shape[-2:], mode="bilinear", align_corners=False) + f3
        d3 = self.dec_vss3(d3_in)

        d2_in = F.interpolate(self.dec_up3(d3), size=f2.shape[-2:], mode="bilinear", align_corners=False) + f2
        d2 = self.dec_vss2(d2_in)

        d1_in = F.interpolate(self.dec_up2(d2), size=f1.shape[-2:], mode="bilinear", align_corners=False) + f1
        d1 = self.dec_vss1(d1_in)

        seg_logits = F.interpolate(self.change_head(d1), size=(h, w), mode="bilinear", align_corners=False)

        loc_logits = None
        if self.task_mode == "bda":
            loc_logits = F.interpolate(self.loc_head(feats_t1[0]), size=(h, w), mode="bilinear", align_corners=False)

        return ModelOutput(
            seg_logits=seg_logits,
            loc_logits=loc_logits,
        )


# =============================================================================
# 5. Liu et al. (2026) - SemiBCD (Semi-Supervised Building Change Detection)
# =============================================================================


@MODEL_REGISTRY.register("semibcd")
class SemiBCD(BaseModel):
    """
    SemiBCD: Semi-Supervised Building Change Detection From Bitemporal Remote Sensing Images
    Leveraging Visual–Language Models and Consistency Learning
    (Liu et al., IEEE TGRS, Vol. 64, Art. no. 5610015, 2026).

    Architecture & Consistency Components:
    1. Shared 4-stage ResNet pyramid encoder.
    2. Auxiliary Building Semantic Segmentation Head (`LightSSHead`) producing `ss_logits_t1`,
       `ss_logits_t2` and L2-normalizable building representations `ss_rep_t1`, `ss_rep_t2`
       for Task-Homogeneous Alignment (`L_tha = L_fc + L_sc`).
    3. DeepLabV3+ Change Detection Head (`DeepLabV3PlusCDHead`) with feature-level dropout
       perturbation (`need_fp=True`) for Multi-View Consistency Regularization (`L_mvc`).
    4. Uncertainty-Aware Pseudo-Label Refinement (`UAPLRModule`) module integrated directly on the model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        fp_dropout: float = 0.5,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        config = ModelConfig(
            arch="semibcd",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            dropout=fp_dropout,
            extra={"pretrained_backbone": pretrained_backbone, "pretrained_arch": pretrained_arch},
        )
        super().__init__(config=config)
        stage_c = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.backbone = FourStagePyramidBackbone(
            in_channels=in_channels,
            stage_channels=stage_c,
            pretrained=pretrained_backbone,
            pretrained_arch=pretrained_arch,
        )
        self.ss_head = LightSSHead(in_channels_list=stage_c, mid_channels=base_channels, num_classes=num_classes)
        self.cd_head = DeepLabV3PlusCDHead(
            c1_channels=stage_c[0],
            c4_channels=stage_c[3],
            aspp_out=base_channels * 2,
            low_channels=base_channels // 2,
            num_classes=num_classes,
        )
        self.fp_drop = nn.Dropout2d(fp_dropout)
        self.uaplr = UAPLRModule()

    def forward(
        self,
        img_t1: torch.Tensor,
        img_t2: Optional[torch.Tensor] = None,
        need_fp: bool = False,
        img_t1_strong: Optional[torch.Tensor] = None,
        img_t2_strong: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        if img_t2 is None:
            if img_t1.ndim == 5:
                img_t1, img_t2 = img_t1[:, 0], img_t1[:, 1]
            else:
                img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        h, w = img_t1.shape[-2:]
        feats_t1 = self.backbone(img_t1)
        feats_t2 = self.backbone(img_t2)

        ss_logits_t1, ss_rep_t1 = self.ss_head(feats_t1)
        ss_logits_t2, ss_rep_t2 = self.ss_head(feats_t2)

        c1_diff = torch.abs(feats_t1[0] - feats_t2[0])
        c4_diff = torch.abs(feats_t1[3] - feats_t2[3])

        cd_logits_low = self.cd_head(c1_diff, c4_diff)
        seg_logits = F.interpolate(cd_logits_low, size=(h, w), mode="bilinear", align_corners=False)

        fp_logits = None
        if need_fp:
            fp_low = self.cd_head(self.fp_drop(c1_diff), self.fp_drop(c4_diff))
            fp_logits = F.interpolate(fp_low, size=(h, w), mode="bilinear", align_corners=False)

        strong_logits = None
        if img_t1_strong is not None and img_t2_strong is not None:
            s_t1 = self.backbone(img_t1_strong)
            s_t2 = self.backbone(img_t2_strong)
            s_low = self.cd_head(torch.abs(s_t1[0] - s_t2[0]), torch.abs(s_t1[3] - s_t2[3]))
            strong_logits = F.interpolate(s_low, size=(h, w), mode="bilinear", align_corners=False)

        return ModelOutput(
            seg_logits=seg_logits,
            fp_logits=fp_logits,
            strong_logits=strong_logits,
            ss_logits_t1=F.interpolate(ss_logits_t1, size=(h, w), mode="bilinear", align_corners=False),
            ss_logits_t2=F.interpolate(ss_logits_t2, size=(h, w), mode="bilinear", align_corners=False),
            ss_rep_t1=ss_rep_t1,
            ss_rep_t2=ss_rep_t2,
        )


# =============================================================================
# 6. Wang et al. (2026) - CGCCE-Net (Change-Guided Cross Correlation Enhancement)
# =============================================================================


@MODEL_REGISTRY.register("cgcce_net", aliases=["cgccenet"])
class CGCCENet(BaseModel):
    """
    CGCCE-Net: Change-guided cross correlation enhancement network for remote sensing
    building change detection (Wang, Gao, & Li, Information Sciences, Vol. 742, Art. no. 123320, 2026).

    Architecture components:
    1. Siamese 4-stage hierarchical encoder extracting shallow features (T_{i1}, T_{i2}) and
       deep semantic features (T_{i3}, T_{i4}).
    2. Global Cross-Correlation Module (`GCCMModule`) with Linear Angle Cross-Attention on deep stages.
    3. Dual Change Feature Representation:
       - Pixel-wise subtraction D_j = |T_{1j} - T_{2j}|
       - Channel-wise concatenation C_j = Cat(T_{1j}, T_{2j}) + CGRR shallow texture residual.
    4. Semantic Cognitive Enhancement Module (`SCEMModule`) with multi-kernel (3x3, 5x5, 7x7) and
       global context gating on C_j.
    5. Cross Fusion Decoder (`CFDDecoderBlock`) mutually gating C_j' and D_j across all 4 stages.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        config = ModelConfig(
            arch="cgcce_net",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            extra={"pretrained_backbone": pretrained_backbone, "pretrained_arch": pretrained_arch},
        )
        super().__init__(config=config)
        stage_c = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.encoder = FourStagePyramidBackbone(
            in_channels=in_channels,
            stage_channels=stage_c,
            pretrained=pretrained_backbone,
            pretrained_arch=pretrained_arch,
        )
        self.backbone = self.encoder

        # CGRR Branch operating on concatenated shallow stage-1 features (2 * stage_c[0])
        self.cgrr = CGRRBranch(in_channels=stage_c[0] * 2, stage_channels=stage_c)

        # GCCM on deep stages 3 and 4
        self.gccm3 = GCCMModule(stage_c[2])
        self.gccm4 = GCCMModule(stage_c[3])

        # SCEM on concatenated change stream C_j (2 * stage_c[j] -> stage_c[j])
        self.scem_blocks = nn.ModuleList(
            [SCEMModule(in_channels=stage_c[j] * 2, out_channels=stage_c[j]) for j in range(4)]
        )

        # CFD Decoder blocks from stage 4 down to stage 1
        self.cfd4 = CFDDecoderBlock(stage_c[3], stage_c[2])
        self.cfd3 = CFDDecoderBlock(stage_c[2], stage_c[1])
        self.cfd2 = CFDDecoderBlock(stage_c[1], stage_c[0])
        self.cfd1 = CFDDecoderBlock(stage_c[0], stage_c[0])

        self.head = nn.Conv2d(stage_c[0], num_classes, kernel_size=1)

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            if img_t1.ndim == 5:
                img_t1, img_t2 = img_t1[:, 0], img_t1[:, 1]
            else:
                img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        h, w = img_t1.shape[-2:]
        t1_feats = self.encoder(img_t1)
        t2_feats = self.encoder(img_t2)

        # Apply GCCM to deep stages (stage 3 and stage 4)
        t1_feats[2], t2_feats[2] = self.gccm3(t1_feats[2], t2_feats[2])
        t1_feats[3], t2_feats[3] = self.gccm4(t1_feats[3], t2_feats[3])

        # CGRR shallow texture residuals across 4 stages
        cgrr_res = self.cgrr(t1_feats[0], t2_feats[0])

        # Dual change streams: D_j (difference) and C_j' (SCEM-enhanced concatenation + CGRR)
        d_feats: List[torch.Tensor] = []
        c_feats: List[torch.Tensor] = []
        for j in range(4):
            d_j = torch.abs(t1_feats[j] - t2_feats[j])
            c_j = torch.cat([t1_feats[j], t2_feats[j]], dim=1) + cgrr_res[j]
            c_j_prime = self.scem_blocks[j](c_j)
            d_feats.append(d_j)
            c_feats.append(c_j_prime)

        # Cross Fusion Decoder (CFD) top-down refinement
        u4 = self.cfd4(c_feats[3], d_feats[3])
        u4_up = F.interpolate(u4, size=c_feats[2].shape[-2:], mode="bilinear", align_corners=False)

        u3 = self.cfd3(c_feats[2] + u4_up, d_feats[2] + u4_up)
        u3_up = F.interpolate(u3, size=c_feats[1].shape[-2:], mode="bilinear", align_corners=False)

        u2 = self.cfd2(c_feats[1] + u3_up, d_feats[1] + u3_up)
        u2_up = F.interpolate(u2, size=c_feats[0].shape[-2:], mode="bilinear", align_corners=False)

        u1 = self.cfd1(c_feats[0] + u2_up, d_feats[0] + u2_up)
        seg_logits = F.interpolate(self.head(u1), size=(h, w), mode="bilinear", align_corners=False)

        return ModelOutput(seg_logits=seg_logits)


# =============================================================================
# 7. Wang et al. (2026) - KGBDCNet (Keyword-Guided Building Damage Captioning)
# =============================================================================


@MODEL_REGISTRY.register("kgbdcnet")
class KGBDCNet(BaseModel):
    """
    KGBDCNet: Keyword-guided building damage captioning network for bi-temporal remote sensing images
    (Wang et al., ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 234, pp. 369–385, 2026).

    Architecture components:
    1. Multi-Modal Feature Encoder with LoRA-adapted projections (`inject_lora`), extracting
       spatial patch tokens L_1, L_2, global tokens G_1, G_2, and hybrid soft prompt + hard
       damage keyword embeddings F.
    2. Cross-Modal Attention Guided Module (`CAGMModule`) performing Visual-Text Retrieval and
       Gate-Controlled Visual-Text Enhancement.
    3. Spatial–Temporal Memory-Enhanced Module (STMEM) combining stacked Space-Scan Mamba Blocks
       (`SSMBBlock`) and Temporal-Trace Transformer Block (`TTTBBlock`).
    4. Autoregressive / Teacher-Forced Transformer Caption Decoder generating building damage
       descriptions (`caption_logits`, `generated_tokens`) plus an auxiliary Cross-Modal Spatial
       Change Head (`seg_logits`) fusing STMEM tokens with high-resolution visual stem features.
    """

    def __init__(
        self,
        in_channels: int = 3,
        vocab_size: int = 1000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_ssmb_layers: int = 2,
        num_decoder_layers: int = 2,
        num_soft_prompts: int = 8,
        num_keywords: int = 16,
        max_seq_len: int = 64,
        lora_rank: int = 8,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        config = ModelConfig(
            arch="kgbdcnet",
            dims=2,
            in_channels=in_channels,
            base_channels=embed_dim,
            extra={
                "vocab_size": vocab_size,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "max_seq_len": max_seq_len,
                "lora_rank": lora_rank,
                "pretrained_backbone": pretrained_backbone,
                "pretrained_arch": pretrained_arch,
            },
        )
        super().__init__(config=config)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pretrained_backbone = bool(pretrained_backbone)

        if self.pretrained_backbone:
            self.backbone = FourStagePyramidBackbone(
                in_channels=in_channels,
                stage_channels=[embed_dim // 2, embed_dim, embed_dim * 2, embed_dim * 4],
                pretrained=True,
                pretrained_arch=pretrained_arch,
            )
        else:
            # Vision patch encoder producing stride-4 spatial map and 8x8=64 spatial tokens + 1 global token
            self.vision_stem_s2 = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
            )
            self.vision_stem_s4 = nn.Sequential(
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
        self.vision_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.vision_qkv = nn.Linear(embed_dim, embed_dim)
        self.vision_proj = nn.Linear(embed_dim, embed_dim)
        # Inject LoRA into visual linear projections (r=8) per Section 3.1
        inject_lora(self, target_names=("vision_qkv", "vision_proj"), rank=lora_rank, alpha=float(lora_rank))

        # Hybrid Prompt Text Encoder: soft learnable prompt tokens + hard damage keyword embeddings
        self.soft_prompts = nn.Parameter(torch.randn(1, num_soft_prompts, embed_dim) * 0.02)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.default_keywords = nn.Parameter(torch.randn(1, num_keywords, embed_dim) * 0.02)
        self.prompt_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )

        # CAGM
        self.cagm = CAGMModule(embed_dim=embed_dim)

        # STMEM: stacked SSMB + TTTB
        self.ssmb_layers = nn.ModuleList([SSMBBlock(embed_dim=embed_dim) for _ in range(num_ssmb_layers)])
        self.tttb = TTTBBlock(embed_dim=embed_dim, num_heads=num_heads)

        # Auxiliary Cross-Modal Spatial Change Segmentation Head (fuses STMEM 8x8 tokens + s4 + s2 features)
        self.spatial_s4_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )
        self.spatial_s2_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
        )

        # Caption Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.caption_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.vocab_head = nn.Linear(embed_dim, vocab_size)

    def _encode_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(self, "pretrained_backbone", False):
            feats = self.backbone(img)
            s2 = feats[0]
            s4 = feats[1]
        else:
            s2 = self.vision_stem_s2(img)  # (B, D//2, H/2, W/2)
            s4 = self.vision_stem_s4(s2)  # (B, D, H/4, W/4)
        feat_map = self.vision_pool(s4)  # (B, D, 8, 8)
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, 64, D)
        tokens = self.vision_proj(F.gelu(self.vision_qkv(tokens)))
        global_token = tokens.mean(dim=1, keepdim=True)  # (B, 1, D)
        return tokens, global_token, s2, s4

    def _encode_hybrid_prompts(
        self, batch_size: int, keyword_ids: Optional[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        soft = self.soft_prompts.expand(batch_size, -1, -1)
        if keyword_ids is not None:
            hard = self.token_embedding(keyword_ids.to(device))
        else:
            hard = self.default_keywords.expand(batch_size, -1, -1)
        prompt_seq = torch.cat([soft, hard], dim=1)
        return self.prompt_transformer(prompt_seq)

    def forward(
        self,
        img_t1: torch.Tensor,
        img_t2: Optional[torch.Tensor] = None,
        caption_targets: Optional[torch.Tensor] = None,
        keyword_ids: Optional[torch.Tensor] = None,
        max_gen_len: int = 20,
        change_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        if img_t2 is None:
            if img_t1.ndim == 5:
                img_t1, img_t2 = img_t1[:, 0], img_t1[:, 1]
            else:
                img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        b, _, h, w = img_t1.shape
        device = img_t1.device

        l1, g1, s2_1, s4_1 = self._encode_image(img_t1)
        l2, g2, s2_2, s4_2 = self._encode_image(img_t2)
        text_feat = self._encode_hybrid_prompts(b, keyword_ids, device)

        # CAGM cross-modal modulation
        l1_cagm, g1_cagm = self.cagm(l1, g1, text_feat)
        l2_cagm, g2_cagm = self.cagm(l2, g2, text_feat)

        # STMEM: SSMB + TTTB
        cur_l1, cur_l2 = l1_cagm, l2_cagm
        for ssmb in self.ssmb_layers:
            cur_l1, cur_l2 = ssmb(cur_l1, cur_l2)
        memory = self.tttb(cur_l1, cur_l2, g1_cagm, g2_cagm)  # (B, 1+N, D)

        # Auxiliary spatial change segmentation map from cross-modal STMEM tokens + visual stem
        stmem_diff_8x8 = torch.abs(cur_l1 - cur_l2).transpose(1, 2).contiguous().view(b, self.embed_dim, 8, 8)
        stmem_up_s4 = F.interpolate(stmem_diff_8x8, size=s4_1.shape[-2:], mode="bilinear", align_corners=False)
        s4_diff = torch.abs(s4_1 - s4_2)
        s4_fused = self.spatial_s4_fuse(torch.cat([s4_diff, stmem_up_s4], dim=1))
        s4_up_s2 = F.interpolate(s4_fused, size=s2_1.shape[-2:], mode="bilinear", align_corners=False)
        s2_diff = torch.abs(s2_1 - s2_2)
        seg_low = self.spatial_s2_head(torch.cat([s2_diff, s4_up_s2], dim=1))
        seg_logits = F.interpolate(seg_low, size=(h, w), mode="bilinear", align_corners=False)

        # Teacher-forced training or greedy decoding
        loss = None
        if caption_targets is not None:
            seq_len = caption_targets.shape[1]
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)
            tgt_emb = self.token_embedding(caption_targets) + self.pos_embedding(pos)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            dec_out = self.caption_decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            caption_logits = self.vocab_head(dec_out)  # (B, L, vocab_size)
            if seq_len > 1:
                loss = F.cross_entropy(
                    caption_logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                    caption_targets[:, 1:].contiguous().view(-1),
                )
            generated_tokens = caption_logits.argmax(dim=-1)
        else:
            # Greedy autoregressive generation starting from BOS token (1)
            cur_tokens = torch.ones((b, 1), dtype=torch.long, device=device)
            logits_steps: List[torch.Tensor] = []
            for step in range(max_gen_len):
                seq_len = cur_tokens.shape[1]
                pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)
                tgt_emb = self.token_embedding(cur_tokens) + self.pos_embedding(pos)
                dec_out = self.caption_decoder(tgt=tgt_emb, memory=memory)
                step_logits = self.vocab_head(dec_out[:, -1:, :])
                logits_steps.append(step_logits)
                next_tok = step_logits.argmax(dim=-1)
                cur_tokens = torch.cat([cur_tokens, next_tok], dim=1)
            caption_logits = torch.cat(logits_steps, dim=1)
            generated_tokens = cur_tokens[:, 1:]

        if change_mask is not None:
            tgt_mask = change_mask.to(device).float()
            probs = torch.sigmoid(seg_logits)
            inter = (probs * tgt_mask).sum(dim=(1, 2, 3))
            union = probs.sum(dim=(1, 2, 3)) + tgt_mask.sum(dim=(1, 2, 3))
            dice_l = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()
            seg_l = F.binary_cross_entropy_with_logits(seg_logits, tgt_mask) + dice_l
            loss = seg_l if loss is None else (loss + seg_l)

        return ModelOutput(
            seg_logits=seg_logits,
            cls_logits=caption_logits,
            caption_logits=caption_logits,
            generated_tokens=generated_tokens,
            latent_feat=memory,
            loss=loss,
        )


# =============================================================================
# 8. Huang et al. (2026) - CBRNet (Corner-Guided Boundary Refinement Network)
# =============================================================================


@MODEL_REGISTRY.register("cbrnet")
class CBRNet(BaseModel):
    """
    CBRNet: Corner-Guided Boundary Refinement Network for High-Precision Building Extraction
    From Remote Sensing Imagery (Huang et al., IEEE TGRS, Vol. 64, Art. no. 4406216, 2026).

    Architecture components:
    1. 4-stage ResNet backbone extracting C_1, C_2, C_3, C_4.
    2. `CASCBlock` (Content-Adaptive Scale Convolution) on C_3 and C_4.
    3. `CGFEModule` (Corner-Guided Feature Enhancement with differentiable Harris response) on all 4 stages.
    4. `LWCAAttention` (Local Window Cross-Attention) on all 4 stages.
    5. `BiFPNFeatureFusion` (Bidirectional Feature Pyramid Network) fusing P_1..P_4.
    6. `DSBRDecoder` (Dual-Stream Boundary Refinement Decoder) outputting building masks,
       fine boundaries, coarse boundaries, and multi-scale corner responses.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        fpn_channels: int = 128,
        window_size: int = 8,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet50",
    ):
        config = ModelConfig(
            arch="cbrnet",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            extra={
                "fpn_channels": fpn_channels,
                "window_size": window_size,
                "pretrained_backbone": pretrained_backbone,
                "pretrained_arch": pretrained_arch,
            },
        )
        super().__init__(config=config)
        stage_c = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.backbone = FourStagePyramidBackbone(
            in_channels=in_channels,
            stage_channels=stage_c,
            pretrained=pretrained_backbone,
            pretrained_arch=pretrained_arch,
        )

        # CASC applied to high-level features C_3 and C_4 per Section III-B
        self.casc3 = CASCBlock(stage_c[2], stage_c[2])
        self.casc4 = CASCBlock(stage_c[3], stage_c[3])

        # CGFE + LWCA across all 4 stages
        self.cgfe_blocks = nn.ModuleList([CGFEModule(c) for c in stage_c])
        self.lwca_blocks = nn.ModuleList(
            [LWCAAttention(c, window_size=window_size, num_heads=4) for c in stage_c]
        )

        # BiFPN & DSBR Decoder
        self.bifpn = BiFPNFeatureFusion(in_channels_list=stage_c, out_channels=fpn_channels)
        self.dsbr = DSBRDecoder(in_channels=fpn_channels, mid_channels=base_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h, w = x.shape[-2:]
        c1, c2, c3, c4 = self.backbone(x)
        c3 = self.casc3(c3)
        c4 = self.casc4(c4)

        raw_feats = [c1, c2, c3, c4]
        enhanced_feats: List[torch.Tensor] = []
        corner_attns: List[torch.Tensor] = []
        corner_resps: List[torch.Tensor] = []

        for feat, cgfe, lwca in zip(raw_feats, self.cgfe_blocks, self.lwca_blocks):
            f_cgfe, attn_map, r_norm = cgfe(feat)
            f_lwca = lwca(f_cgfe)
            enhanced_feats.append(f_lwca)
            corner_attns.append(attn_map)
            corner_resps.append(r_norm)

        p_outs = self.bifpn(enhanced_feats)
        # Aggregate multi-scale BiFPN outputs at P_1 resolution
        p1_size = p_outs[0].shape[-2:]
        f_d = p_outs[0]
        for p_lvl in p_outs[1:]:
            f_d = f_d + F.interpolate(p_lvl, size=p1_size, mode="bilinear", align_corners=False)

        dsbr_out = self.dsbr(f_d)
        seg_logits = F.interpolate(dsbr_out["seg_logits"], size=(h, w), mode="bilinear", align_corners=False)
        bnd_logits = F.interpolate(dsbr_out["boundary_logits"], size=(h, w), mode="bilinear", align_corners=False)
        coarse_bnd_logits = F.interpolate(
            dsbr_out["coarse_boundary_logits"], size=(h, w), mode="bilinear", align_corners=False
        )

        return ModelOutput(
            seg_logits=seg_logits,
            edge_logits=bnd_logits,
            boundary_logits=bnd_logits,
            coarse_boundary_logits=coarse_bnd_logits,
            corner_attentions=corner_attns,
            corner_responses=corner_resps,
        )


# =============================================================================
# 9. Hussin et al. (2026) - EdgeRefNet (Edge-Guided Refinement Network for BCD)
# =============================================================================


@MODEL_REGISTRY.register("edgerefnet")
class EdgeRefNet(BaseModel):
    """
    EdgeRefNet: An Edge-Guided Refinement Network for Building Change Detection in Remote Sensing Images
    (Hussin, Lu, Ashraf, Mao, & Peng, IEEE TGRS, Vol. 64, Art. no. 4409610, 2026).

    Architecture components:
    1. Weight-shared Siamese ResNet backbone extracting 4-stage bi-temporal features.
    2. Dual-Path Feature Extraction:
       - Semantic Path: Multi-scale feature difference & concatenation across stages.
       - Edge Path: Dedicated `EDBBlock` (Edge Detection Block) + `EEMBlock` (SCSE + High-Frequency
         Enhancement / HFAB) across stages producing multi-scale edge features and auxiliary `edge_logits`.
    3. Context-Guided Cross-Attention Transformer Refinement (`EdgeGuidedCrossAttention`), where
       semantic features supply Queries (Q_s) and edge features supply Keys and Values (K_e, V_e).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet34",
    ):
        config = ModelConfig(
            arch="edgerefnet",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            extra={"pretrained_backbone": pretrained_backbone, "pretrained_arch": pretrained_arch},
        )
        super().__init__(config=config)
        stage_c = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.backbone = FourStagePyramidBackbone(
            in_channels=in_channels,
            stage_channels=stage_c,
            pretrained=pretrained_backbone,
            pretrained_arch=pretrained_arch,
        )

        dec_c = base_channels
        # Semantic Path projections
        self.sem_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c * 2, dec_c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(dec_c),
                    nn.ReLU(inplace=True),
                )
                for c in stage_c
            ]
        )
        # Edge Path: EDB + EEM (SCSE + HFAB)
        self.edge_edbs = nn.ModuleList([EDBBlock(c * 2, dec_c) for c in stage_c])
        self.edge_eems = nn.ModuleList([EEMBlock(dec_c) for _ in stage_c])

        self.sem_fuse = nn.Sequential(
            nn.Conv2d(dec_c * 4, dec_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_c),
            nn.ReLU(inplace=True),
        )
        self.edge_fuse = nn.Sequential(
            nn.Conv2d(dec_c * 4, dec_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_c),
            nn.ReLU(inplace=True),
        )

        # Context-Guided Cross-Attention Transformer Refinement
        self.cross_refiner = EdgeGuidedCrossAttention(channels=dec_c, num_heads=4)

        self.edge_head = nn.Conv2d(dec_c, 1, kernel_size=1)
        self.change_head = nn.Sequential(
            nn.Conv2d(dec_c * 2, dec_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_c, num_classes, kernel_size=1),
        )

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            if img_t1.ndim == 5:
                img_t1, img_t2 = img_t1[:, 0], img_t1[:, 1]
            else:
                img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        h, w = img_t1.shape[-2:]
        feats_t1 = self.backbone(img_t1)
        feats_t2 = self.backbone(img_t2)

        target_size = feats_t1[0].shape[-2:]
        sem_multi: List[torch.Tensor] = []
        edge_multi: List[torch.Tensor] = []

        for idx in range(4):
            pair_cat = torch.cat([feats_t1[idx], feats_t2[idx]], dim=1)
            s_i = self.sem_convs[idx](pair_cat)
            e_i = self.edge_eems[idx](self.edge_edbs[idx](pair_cat))
            if s_i.shape[-2:] != target_size:
                s_i = F.interpolate(s_i, size=target_size, mode="bilinear", align_corners=False)
                e_i = F.interpolate(e_i, size=target_size, mode="bilinear", align_corners=False)
            sem_multi.append(s_i)
            edge_multi.append(e_i)

        sem_feat = self.sem_fuse(torch.cat(sem_multi, dim=1))
        edge_feat = self.edge_fuse(torch.cat(edge_multi, dim=1))

        refined_sem = self.cross_refiner(sem_feat, edge_feat)
        change_low = self.change_head(torch.cat([refined_sem, edge_feat], dim=1))
        edge_low = self.edge_head(edge_feat)

        seg_logits = F.interpolate(change_low, size=(h, w), mode="bilinear", align_corners=False)
        edge_logits = F.interpolate(edge_low, size=(h, w), mode="bilinear", align_corners=False)

        return ModelOutput(
            seg_logits=seg_logits,
            edge_logits=edge_logits,
        )


# =============================================================================
# 10. Proposed Architecture (2026) - SAM-CEBNet (SAM-Guided Corner and Edge Building Network)
# =============================================================================


@MODEL_REGISTRY.register("sam_cebnet", aliases=["sam-cebnet", "sam_cbr_edgenet", "sam_cbr_edgerefnet", "cbr_edgeref_sam"])
class SAMCEBNet(BaseModel):
    """
    SAM-CEBNet: SAM-Guided Corner and Edge Building Network for High-Precision Single-Image
    Building Extraction:
    1. Pretrained SAM-2 Hiera Foundation Encoder (`SAMFoundationAdapter` via `timm` `sam2_hiera_tiny`)
       fused with a 4-stage high-resolution convolutional pyramid (`FourStagePyramidBackbone`,
       optionally ResNet-34 pretrained) preserving stride-2 (H/2 x W/2) boundary fidelity.
    2. Corner & Scale Enhancement Path:
       - `CASCBlock` (Content-Adaptive Scale Convolution) on deep stages C_3, C_4.
       - `CGFEModule` (Differentiable Harris Corner Detector + Learnable Attention) on all 4 stages.
       - `LWCAAttention` (Local Window Cross-Attention) on all 4 stages.
       - `BiFPNFeatureFusion` (Bidirectional Feature Pyramid Network).
    3. High-Frequency Edge Path & Cross-Attention Refinement:
       - `SingleImageEdgeRefBranch` integrating explicit Sobel gradient priors, multi-scale
         `EDBBlock` + `EEMBlock` (SCSE + High-Frequency Residual HFAB), corner-edge modulation,
         and `EdgeGuidedCrossAttention` (Semantic Queries Q_s, High-Frequency Edge Keys/Values K_e, V_e).
    4. Dual-Stream Boundary Refinement (`DSBRDecoder`) + Edge-Guided Residual Mask Refiner (`edge_mask_refiner`)
       jointly predicting building footprints (`seg_logits`), fine/coarse boundaries (`boundary_logits`,
       `coarse_boundary_logits`), high-frequency edge logits (`edge_ref_logits`), and multi-scale corner responses.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        fpn_channels: int = 128,
        window_size: int = 8,
        pretrained_backbone: bool = False,
        pretrained_arch: str = "resnet50",
        pretrained_sam: Optional[bool] = None,
        sam_arch: str = "sam2_hiera_tiny",
        sam_version: str = "sam2",
        num_concepts: int = 4,
    ):
        if pretrained_sam is None:
            pretrained_sam = bool(pretrained_backbone)
        self.sam_version = sam_version.lower()
        config = ModelConfig(
            arch="sam3_cebnet" if self.sam_version == "sam3" else "sam_cebnet",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            extra={
                "fpn_channels": fpn_channels,
                "window_size": window_size,
                "pretrained_backbone": pretrained_backbone,
                "pretrained_arch": pretrained_arch,
                "pretrained_sam": pretrained_sam,
                "sam_arch": sam_arch,
                "sam_version": self.sam_version,
                "num_concepts": num_concepts,
            },
        )
        super().__init__(config=config)
        stage_c = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)

        # 1. Dual Foundation Backbone: High-Res CNN Pyramid + Foundation Adapter (SAM 2 or SAM 3)
        self.cnn_backbone = FourStagePyramidBackbone(
            in_channels=in_channels,
            stage_channels=stage_c,
            pretrained=pretrained_backbone,
            pretrained_arch=pretrained_arch,
        )
        if self.sam_version == "sam3" or "sam3" in sam_arch.lower():
            self.sam_adapter = SAM3FoundationAdapter(
                stage_channels=stage_c,
                sam_arch=sam_arch,
                pretrained_sam=pretrained_sam,
                num_concepts=num_concepts,
            )
        else:
            self.sam_adapter = SAMFoundationAdapter(
                stage_channels=stage_c,
                sam_arch=sam_arch,
                pretrained_sam=pretrained_sam,
            )
        # Group pretrained foundation encoders under `self.backbone` for differential LR fine-tuning
        self.backbone = nn.ModuleList([self.cnn_backbone, self.sam_adapter.sam_encoder])

        # 2. CBRNet Scale & Corner Enhancement (CASC + CGFE + LWCA + BiFPN)
        self.casc3 = CASCBlock(stage_c[2], stage_c[2])
        self.casc4 = CASCBlock(stage_c[3], stage_c[3])
        self.cgfe_blocks = nn.ModuleList([CGFEModule(c) for c in stage_c])
        self.lwca_blocks = nn.ModuleList(
            [LWCAAttention(c, window_size=window_size, num_heads=4) for c in stage_c]
        )
        self.bifpn = BiFPNFeatureFusion(in_channels_list=stage_c, out_channels=fpn_channels)

        # 3. EdgeRefNet Single-Image High-Frequency Edge Path & Cross-Attention Refinement
        self.edge_ref_branch = SingleImageEdgeRefBranch(
            stage_channels=stage_c,
            edge_channels=base_channels,
            sem_channels=fpn_channels,
            num_heads=4,
        )
        self.sem_edge_fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * 2, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
        )
        nn.init.constant_(self.sem_edge_fusion[1].weight, 0.2)
        nn.init.zeros_(self.sem_edge_fusion[1].bias)

        # 4. CBRNet Dual-Stream Boundary Refinement (DSBR) Decoder + EdgeRefNet Residual Mask Refiner
        self.dsbr = DSBRDecoder(in_channels=fpn_channels, mid_channels=base_channels, num_classes=num_classes)
        self.edge_mask_refiner = nn.Sequential(
            nn.Conv2d(fpn_channels + 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
        )
        nn.init.zeros_(self.edge_mask_refiner[-1].weight)
        nn.init.zeros_(self.edge_mask_refiner[-1].bias)
        self.bnd_logit_alpha = nn.Parameter(torch.tensor(0.6))

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h, w = x.shape[-2:]
        cnn_feats = self.cnn_backbone(x)
        sam_feats = self.sam_adapter(x, cnn_feats)

        c1, c2, c3, c4 = sam_feats
        c3 = self.casc3(c3)
        c4 = self.casc4(c4)

        raw_feats = [c1, c2, c3, c4]
        enhanced_feats: List[torch.Tensor] = []
        corner_attns: List[torch.Tensor] = []
        corner_resps: List[torch.Tensor] = []

        for feat, cgfe, lwca in zip(raw_feats, self.cgfe_blocks, self.lwca_blocks):
            f_cgfe, attn_map, r_norm = cgfe(feat)
            f_lwca = lwca(f_cgfe)
            enhanced_feats.append(f_lwca)
            corner_attns.append(attn_map)
            corner_resps.append(r_norm)

        p_outs = self.bifpn(enhanced_feats)
        p1_size = p_outs[0].shape[-2:]
        f_sem = p_outs[0]
        for p_lvl in p_outs[1:]:
            f_sem = f_sem + F.interpolate(p_lvl, size=p1_size, mode="bilinear", align_corners=False)

        # EdgeRefNet high-frequency edge extraction & cross-attention refinement
        edge_out = self.edge_ref_branch(x, enhanced_feats, corner_attns, f_sem)
        edge_delta = self.sem_edge_fusion(torch.cat([edge_out["refined_sem"] - f_sem, edge_out["edge_feat"]], dim=1))
        f_joint = f_sem + edge_delta

        dsbr_out = self.dsbr(f_joint)
        base_seg_s2 = dsbr_out["seg_logits"]
        edge_prob_s2 = torch.sigmoid(edge_out["edge_logits_low"])
        seg_prob_s2 = torch.sigmoid(base_seg_s2)
        refined_seg_s2 = base_seg_s2 + self.edge_mask_refiner(
            torch.cat([edge_out["edge_feat"], seg_prob_s2, edge_prob_s2], dim=1)
        )

        seg_logits = F.interpolate(refined_seg_s2, size=(h, w), mode="bilinear", align_corners=False)
        dsbr_bnd_logits = F.interpolate(dsbr_out["boundary_logits"], size=(h, w), mode="bilinear", align_corners=False)
        coarse_bnd_logits = F.interpolate(
            dsbr_out["coarse_boundary_logits"], size=(h, w), mode="bilinear", align_corners=False
        )
        edge_ref_logits = F.interpolate(edge_out["edge_logits_low"], size=(h, w), mode="bilinear", align_corners=False)

        alpha = self.bnd_logit_alpha.clamp(0.1, 0.9)
        bnd_logits = alpha * dsbr_bnd_logits + (1.0 - alpha) * edge_ref_logits

        return ModelOutput(
            seg_logits=seg_logits,
            edge_logits=bnd_logits,
            boundary_logits=bnd_logits,
            coarse_boundary_logits=coarse_bnd_logits,
            edge_ref_logits=edge_ref_logits,
            corner_attentions=corner_attns,
            corner_responses=corner_resps,
        )


# Backward-compatible class alias
SAMCBREdgeNet = SAMCEBNet


@MODEL_REGISTRY.register("sam3_cebnet", aliases=["sam3-cebnet", "sam3_cbr_edgenet", "sam3_ceb_net"])
class SAM3CEBNet(SAMCEBNet):
    """
    SAM3-CEBNet: SAM-3 Powered Corner and Edge Building Network.
    Integrates Meta's SAM 3 Perception Encoder (PE) with learnable Promptable Concept
    Segmentation (PCS) tokens and a Decoupled Global Presence Head with 4-stage presence-gated
    lateral feature fusion to eliminate false alarms and refine building footprints.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("sam_version", "sam3")
        super().__init__(*args, **kwargs)





# ==============================================================================
# Bi-Temporal Wetland Surface Water & Change Detection Architectures (2025-2026)
# ==============================================================================

class SpatialAttentionPool(nn.Module):
    """Pools 2D feature map into L semantic tokens using learned spatial attention."""
    def __init__(self, in_channels: int, num_tokens: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_tokens, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        attn = self.conv(x).view(B, -1, H * W)
        attn = F.softmax(attn, dim=-1)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        tokens = torch.bmm(attn, x_flat)
        return tokens

class BITTransformerDecoderLayer(nn.Module):
    """Decoder layer with cross-attention between pixel queries and temporal tokens."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dim_feedforward: int = 256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = tgt
        k = v = memory
        attn_out, _ = self.cross_attn(q, k, v)
        x = self.norm1(tgt + attn_out)
        ffn_out = self.linear2(self.act(self.linear1(x)))
        return self.norm2(x + ffn_out)

@MODEL_REGISTRY.register("bit_cd")
class BITChangeDetector(BaseModel):
    """
    Bitemporal Image Transformer (BIT) for Remote Sensing Change Detection
    Reference: Chen et al., "Remote Sensing Image Change Detection with Transformers", IEEE TGRS 2021.
    """
    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 1,
        embed_dim: int = 64,
        num_tokens: int = 16,
        num_heads: int = 4,
        enc_depth: int = 2,
        dec_depth: int = 2,
    ):
        super().__init__(
            ModelConfig(
                arch="bit_cd",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                extra={"embed_dim": embed_dim, "num_tokens": num_tokens}
            )
        )
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=1)
        self.token_pool = SpatialAttentionPool(embed_dim, num_tokens=num_tokens)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        
        self.dec_t1 = nn.ModuleList([BITTransformerDecoderLayer(embed_dim, num_heads) for _ in range(dec_depth)])
        self.dec_t2 = nn.ModuleList([BITTransformerDecoderLayer(embed_dim, num_heads) for _ in range(dec_depth)])
        
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)
        )

    def _extract_features(self, x: torch.Tensor):
        f = self.init_conv(x)
        f = self.layer1(f)
        f = self.layer2(f)
        feat_embed = self.proj(f)
        tokens = self.token_pool(feat_embed)
        return feat_embed, tokens

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x1.shape
        f1, tok1 = self._extract_features(x1)
        f2, tok2 = self._extract_features(x2)
        
        tok_both = torch.cat([tok1, tok2], dim=1)
        tok_both = self.transformer_encoder(tok_both)
        
        L = tok1.shape[1]
        tok1_enc, tok2_enc = tok_both[:, :L, :], tok_both[:, L:, :]
        
        B, D, H8, W8 = f1.shape
        f1_flat = f1.view(B, D, H8 * W8).permute(0, 2, 1)
        f2_flat = f2.view(B, D, H8 * W8).permute(0, 2, 1)
        
        for layer in self.dec_t1:
            f1_flat = layer(f1_flat, tok1_enc)
        for layer in self.dec_t2:
            f2_flat = layer(f2_flat, tok2_enc)
            
        f1_rec = f1_flat.permute(0, 2, 1).view(B, D, H8, W8)
        f2_rec = f2_flat.permute(0, 2, 1).view(B, D, H8, W8)
        
        diff = torch.abs(f1_rec - f2_rec)
        logits = self.head(diff)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

class SNUNetConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SNUNetECAM(nn.Module):
    """Ensemble Channel Attention Module from SNUNet-CD."""
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(16, in_channels // ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, in_channels // ratio), in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.avg_pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w

@MODEL_REGISTRY.register("snunet_cd")
class SNUNetCD(BaseModel):
    """
    Siam-NestedUNet with Ensemble Channel Attention (SNUNet-CD).
    Reference: Fang et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection", IEEE GRSL 2021.
    """
    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 1,
        base_channels: int = 32,
    ):
        super().__init__(
            ModelConfig(
                arch="snunet_cd",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                extra={"base_channels": base_channels}
            )
        )
        c = [base_channels * (2**i) for i in range(4)]
        self.conv0_0 = SNUNetConvBlock(in_channels, c[0])
        self.conv1_0 = SNUNetConvBlock(c[0], c[1])
        self.conv2_0 = SNUNetConvBlock(c[1], c[2])
        self.conv3_0 = SNUNetConvBlock(c[2], c[3])
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        self.conv0_1 = SNUNetConvBlock(c[0] * 2 + c[1], c[0])
        self.conv1_1 = SNUNetConvBlock(c[1] * 2 + c[2], c[1])
        self.conv2_1 = SNUNetConvBlock(c[2] * 2 + c[3], c[2])
        
        self.conv0_2 = SNUNetConvBlock(c[0] * 3 + c[1], c[0])
        self.conv1_2 = SNUNetConvBlock(c[1] * 3 + c[2], c[1])
        
        self.conv0_3 = SNUNetConvBlock(c[0] * 4 + c[1], c[0])
        
        total_diff_ch = c[0] * 4
        self.ecam = SNUNetECAM(total_diff_ch)
        self.head = nn.Sequential(
            nn.Conv2d(total_diff_ch, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[0], num_classes, kernel_size=1)
        )

    def _encoder_forward(self, x: torch.Tensor):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        return x0_0, x1_0, x2_0, x3_0

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x0_0_1, x1_0_1, x2_0_1, x3_0_1 = self._encoder_forward(x1)
        x0_0_2, x1_0_2, x2_0_2, x3_0_2 = self._encoder_forward(x2)
        
        x0_1_1 = self.conv0_1(torch.cat([x0_0_1, x0_0_1, self.up(x1_0_1)], dim=1))
        x0_1_2 = self.conv0_1(torch.cat([x0_0_2, x0_0_2, self.up(x1_0_2)], dim=1))
        x1_1_1 = self.conv1_1(torch.cat([x1_0_1, x1_0_1, self.up(x2_0_1)], dim=1))
        x1_1_2 = self.conv1_1(torch.cat([x1_0_2, x1_0_2, self.up(x2_0_2)], dim=1))
        x2_1_1 = self.conv2_1(torch.cat([x2_0_1, x2_0_1, self.up(x3_0_1)], dim=1))
        x2_1_2 = self.conv2_1(torch.cat([x2_0_2, x2_0_2, self.up(x3_0_2)], dim=1))
        
        x0_2_1 = self.conv0_2(torch.cat([x0_0_1, x0_1_1, x0_1_1, self.up(x1_1_1)], dim=1))
        x0_2_2 = self.conv0_2(torch.cat([x0_0_2, x0_1_2, x0_1_2, self.up(x1_1_2)], dim=1))
        x1_2_1 = self.conv1_2(torch.cat([x1_0_1, x1_1_1, x1_1_1, self.up(x2_1_1)], dim=1))
        x1_2_2 = self.conv1_2(torch.cat([x1_0_2, x1_1_2, x1_1_2, self.up(x2_1_2)], dim=1))
        
        x0_3_1 = self.conv0_3(torch.cat([x0_0_1, x0_1_1, x0_2_1, x0_2_1, self.up(x1_2_1)], dim=1))
        x0_3_2 = self.conv0_3(torch.cat([x0_0_2, x0_1_2, x0_2_2, x0_2_2, self.up(x1_2_2)], dim=1))
        
        d1 = torch.abs(x0_0_1 - x0_0_2)
        d2 = torch.abs(x0_1_1 - x0_1_2)
        d3 = torch.abs(x0_2_1 - x0_2_2)
        d4 = torch.abs(x0_3_1 - x0_3_2)
        
        diff_stack = torch.cat([d1, d2, d3, d4], dim=1)
        ecam_diff = self.ecam(diff_stack)
        return self.head(ecam_diff)

@MODEL_REGISTRY.register("hydro_early_fusion_net")
class HydroEarlyFusionNet(BaseModel):
    """
    HydroEarlyFusionNet: ResNet34 Early-Fusion Architecture with
    Spectral Water Difference Gating (SWDG) and scSE Decoder.
    Explicitly modulates encoder multi-scale skip connections using
    physical differential signatures (|Delta MNDWI|, |Delta NDWI|, |Delta LWI|).
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels_per_image: int = 10,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__(
            ModelConfig(
                arch="hydro_early_fusion_net",
                dims=2,
                in_channels=in_channels_per_image * 2,
                num_seg_classes=classes,
                pretrained=(encoder_weights is not None),
                extra={"encoder_name": encoder_name, "attention": "swdg+scse"}
            )
        )
        import segmentation_models_pytorch as smp
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels_per_image * 2,
            depth=5,
            weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        
        self.water_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(4, max(4, ch // 4), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(max(4, ch // 4)),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(4, ch // 4), ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.Sigmoid()
            ) for ch in enc_channels
        ])
        
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            attention_type="scse"
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        delta_indices = torch.abs(x2[:, 6:10, :, :] - x1[:, 6:10, :, :])
        x_concat = torch.cat([x1, x2], dim=1)
        features = self.encoder(x_concat)
        
        modulated_features = []
        for feat, gate in zip(features, self.water_gates):
            sp_size = feat.shape[2:]
            d_ind_scaled = F.interpolate(delta_indices, size=sp_size, mode="bilinear", align_corners=False)
            attn = gate(d_ind_scaled)
            modulated_features.append(feat * (1.0 + attn))
            
        decoded = self.decoder(modulated_features)
        return self.segmentation_head(decoded)


def morphological_edge(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Boundary target = dilation - erosion of a binary mask of shape (B, 1, H, W)."""
    pad = k // 2
    dil = F.max_pool2d(mask, k, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


class _EGSRSCSE(nn.Module):
    """Concurrent spatial/channel squeeze-excitation used by the EGSR refinement block."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // r), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(4, c // r), c, 1), nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cse(x) + x * self.sse(x)


@MODEL_REGISTRY.register("hydro_edge_net")
class HydroEdgeNet(BaseModel):
    """
    HydroEdgeNet: HydroEarlyFusionNet + Edge-Guided Shoreline Refinement (EGSR).

    Input: two bi-temporal tensors x1, x2 of shape (B, C, H, W) with channels 6..9 being
    [MNDWI, NDWI, NDVI, LWI]. EGSR predicts the boundary band of the change mask from decoder
    features and |delta indices|, then gates decoder features with sigma(edge_logits).
    forward(x1, x2, return_aux=False) -> seg_logits or (seg_logits, edge_logits).
    Train the edge head with `morphological_edge(change_mask)` as target.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels_per_image: int = 10,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__(
            ModelConfig(
                arch="hydro_edge_net", dims=2, in_channels=in_channels_per_image * 2,
                num_seg_classes=classes, pretrained=(encoder_weights is not None),
                extra={"encoder_name": encoder_name, "attention": "swdg+scse+egsr"},
            )
        )
        import segmentation_models_pytorch as smp
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels_per_image * 2, depth=5, weights=encoder_weights
        )
        enc_ch = self.encoder.out_channels
        self.water_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(4, max(4, ch // 4), 3, padding=1, bias=False), nn.BatchNorm2d(max(4, ch // 4)),
                nn.ReLU(inplace=True), nn.Conv2d(max(4, ch // 4), ch, 1, bias=False),
                nn.BatchNorm2d(ch), nn.Sigmoid(),
            ) for ch in enc_ch
        ])
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_ch, decoder_channels=decoder_channels, n_blocks=5, attention_type="scse"
        )
        c = decoder_channels[-1]
        self.edge_head = nn.Sequential(
            nn.Conv2d(c + 4, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
        )
        self.refine = nn.Sequential(
            _EGSRSCSE(c), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.segmentation_head = smp.base.SegmentationHead(in_channels=c, out_channels=classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, return_aux: bool = False):
        d_ind = torch.abs(x2[:, 6:10] - x1[:, 6:10])
        feats = self.encoder(torch.cat([x1, x2], dim=1))
        mod = []
        for f, g in zip(feats, self.water_gates):
            d = F.interpolate(d_ind, size=f.shape[2:], mode="bilinear", align_corners=False)
            mod.append(f * (1.0 + g(d)))
        dec = self.decoder(mod)
        edge_logits = self.edge_head(torch.cat([dec, d_ind], dim=1))
        dec = self.refine(dec * (1.0 + torch.sigmoid(edge_logits)))
        seg = self.segmentation_head(dec)
        return (seg, edge_logits) if return_aux else seg


# ==============================================================================
# 11. Wang et al. (2026) - DSTCD (Dual-Stage Training for Optical-SAR Few-Shot CD)
# ==============================================================================


class _DSTCDDoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DSTCDOneConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODEL_REGISTRY.register("dstcd", aliases=["opt_sar_cdnet", "dstcd_net"])
class DSTCDNet(BaseModel):
    """
    DSTCD: Dual-Stage Training Framework for Few-Shot Change Detection in Optical and SAR
    Remote Sensing Images (Wang, Ma, Wang, Yang, & Zhang, Int. J. Appl. Earth Obs. Geoinf. 146, 105100, 2026).

    Architecture (`opt_sar_CDNet` in `CDNet.py`):
    1. `CDFE` (`Simple_encoder`): 7x7 stride-2 initial conv + 4 DoubleConv stages with MaxPool downsampling.
    2. `HIFE` (`HIFEResNetFPN`): ASpanFormer's `ResNetFPN_8_2` registration encoder extracting low-level (1/2)
       structural features and deep (1/8) semantic features for T1 and T2.
    3. `STFTB` (`DSTCDSTFTBBlock`): Structure Feature Transfer Block at 1/2 resolution using 4-way split
       cosine-similarity weighting (`T_D_fusion`).
    4. `SEFTB` (`DSTCDSEFTBBlock`): Semantic Feature Transfer Block at 1/8 resolution using 'first cross-task,
       then cross-source' triple Multi-Head Attention.
    5. `Simple_decoder`: Shared decoder producing `aux_seg_logits` (before transfer) and `seg_logits` (after transfer).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channel_list: Sequence[int] = (64, 128, 256, 512),
        transfer_channels: Sequence[int] = (128, 256),
        num_heads: int = 4,
        dropout: float = 0.2,
        transfer: bool = True,
        structure: bool = True,
        semantic: bool = True,
        hife_ckpt_path: Optional[str] = None,
        freeze_hife: bool = True,
    ):
        config = ModelConfig(
            arch="dstcd",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=channel_list[0],
            dropout=dropout,
            extra={
                "channel_list": list(channel_list),
                "transfer_channels": list(transfer_channels),
                "num_heads": num_heads,
                "transfer": transfer,
                "structure": structure,
                "semantic": semantic,
                "hife_ckpt_path": hife_ckpt_path,
                "freeze_hife": freeze_hife,
            },
        )
        super().__init__(config=config)
        self.channels = list(channel_list)
        self.transfer_c = list(transfer_channels)
        self.transfer = transfer
        self.structure = structure
        self.semantic = semantic

        # CDFE (Simple_encoder)
        self.init_conv = nn.Conv2d(in_channels * 2, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.down = nn.MaxPool2d(2)
        self.enc1 = _DSTCDDoubleConv(self.channels[0], self.channels[1])  # 1/2
        self.enc2 = _DSTCDDoubleConv(self.channels[1], self.channels[2])  # 1/4
        self.enc3 = _DSTCDDoubleConv(self.channels[2], self.channels[3])  # 1/8
        self.enc4 = _DSTCDDoubleConv(self.channels[3], self.channels[3])  # 1/8

        # Simple_decoder
        self.f1 = _DSTCDOneConv(self.channels[3], self.channels[3])
        self.f2 = _DSTCDOneConv(self.channels[3] * 2, self.channels[2])
        self.f3 = _DSTCDOneConv(self.channels[2] * 2, self.channels[1])
        self.f4 = _DSTCDOneConv(self.channels[1] * 2, self.channels[0])
        self.outc = nn.Conv2d(self.channels[0], num_classes, kernel_size=1)

        if self.transfer:
            self.hife = HIFEResNetFPN(
                initial_dim=self.transfer_c[0],
                block_dims=(self.transfer_c[0], 196, self.transfer_c[-1]),
                ckpt_path=hife_ckpt_path,
                freeze=freeze_hife,
            )
            if self.structure:
                self.stftb = DSTCDSTFTBBlock(
                    target_channels=self.channels[1],
                    source_channels=self.transfer_c[0],
                )
            else:
                self.low2 = _DSTCDDoubleConv(self.transfer_c[0] * 2, self.channels[1])
                self.low2f = _DSTCDOneConv(self.channels[1] * 2, self.channels[1])

            if self.semantic:
                self.seftb = DSTCDSEFTBBlock(
                    target_channels=self.channels[-1],
                    source_channels=self.transfer_c[-1],
                    num_heads=num_heads,
                    dropout=dropout,
                )
            else:
                self.deep2 = _DSTCDDoubleConv(self.transfer_c[-1] * 2, self.channels[-1])
                self.deep2f = _DSTCDOneConv(self.channels[-1] * 2, self.channels[-1])

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.enc1(self.init_conv(x))
        x2 = self.enc2(self.down(x1))
        x3 = self.enc3(self.down(x2))
        x4 = self.enc4(x3)
        return [x1, x2, x3, x4]

    def _up_cat(self, x_low_res: torch.Tensor, x_high_res: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x_low_res, size=x_high_res.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x_high_res, up], dim=1)

    def _decode(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        feat4 = self.f1(features[-1])
        feat3 = self.f2(torch.cat([feat4, features[-2]], dim=1))
        feat2 = self.f3(self._up_cat(feat3, features[1]))
        feat1 = self.f4(self._up_cat(feat2, features[0]))
        return feat1

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        hw = img_t1.shape[-2:]
        x = torch.cat([img_t1, img_t2], dim=1)
        features = self._encode(x)

        dec1 = self._decode(features)
        logits_1 = self.outc(F.interpolate(dec1, size=hw, mode="bilinear", align_corners=False))

        if self.transfer:
            (t1_low, t2_low), (t1_deep, t2_deep) = self.hife(img_t1, img_t2)
            if self.structure:
                features[0] = self.stftb(t1_low, t2_low, features[0])
            else:
                low_cat = F.interpolate(
                    torch.cat([t1_low, t2_low], dim=1), size=features[0].shape[-2:], mode="bilinear", align_corners=False
                )
                features[0] = self.low2f(torch.cat([features[0], self.low2(low_cat)], dim=1))

            if self.semantic:
                features[-1] = self.seftb(t1_deep, t2_deep, features[-1])
            else:
                deep_cat = F.interpolate(
                    torch.cat([t1_deep, t2_deep], dim=1),
                    size=features[-1].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                features[-1] = self.deep2f(torch.cat([features[-1], self.deep2(deep_cat)], dim=1))

        dec2 = self._decode(features)
        logits_2 = self.outc(F.interpolate(dec2, size=hw, mode="bilinear", align_corners=False))

        return ModelOutput(
            seg_logits=logits_2,
            aux_seg_logits=logits_1,
            pre1_prob=torch.sigmoid(logits_1),
            pre2_prob=torch.sigmoid(logits_2),
        )


# ==============================================================================
# 12. Liu et al. (2026) - GLMamba (Global-Local Mamba Network for CD)
# ==============================================================================


@MODEL_REGISTRY.register("glmamba", aliases=["gl_mamba"])
class GLMamba(BaseModel):
    """
    GLMamba: A Global–Local Mamba Network for Efficient Remote Sensing Change Detection
    (Liu, Zhu, Yin, Qin, Lin, Huang, Xia, & Weng, IEEE JSTARS, Vol. 19, pp. 11344–11360, 2026).

    Architecture (Fig. 1):
    1. Weight-shared 4-stage Visual State Space (VSS) Encoder (`GLMambaVSSBlock`).
    2. Change-Aware Semantic Aggregation Module (`GLMambaCSAMModule`) across adjacent pyramid stages.
    3. Bitemporal Fusion Attention (`GLMambaBFAModule`) combining CBAM difference path and cascaded conv path.
    4. Cross-Space Interaction Enhancement (`GLMambaCSIEModule`) at each scale with top-down summation
       decoder and classifier head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depths: Sequence[int] = (2, 2, 2, 2),
        csie_groups: int = 4,
    ):
        config = ModelConfig(
            arch="glmamba",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=base_channels,
            extra={"depths": list(depths), "csie_groups": csie_groups},
        )
        super().__init__(config=config)
        dims = [base_channels * (2 ** i) for i in range(4)]
        self.dims = dims

        # 4-stage Siamese VSS Encoder (Patch Embedding + Patch Merging)
        self.down_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[0]),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[1]),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[2]),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[3]),
                    nn.GELU(),
                ),
            ]
        )
        self.vss_stages = nn.ModuleList(
            [
                nn.Sequential(*[GLMambaVSSBlock(dims[i]) for _ in range(depths[i])])
                for i in range(4)
            ]
        )

        # CSAM modules (shared across T1 and T2 branches per Fig. 1)
        self.csam_blocks = nn.ModuleList(
            [
                GLMambaCSAMModule(c_mid=dims[0], c_low=None, c_high=dims[1]),
                GLMambaCSAMModule(c_mid=dims[1], c_low=dims[0], c_high=dims[2]),
                GLMambaCSAMModule(c_mid=dims[2], c_low=dims[1], c_high=dims[3]),
                GLMambaCSAMModule(c_mid=dims[3], c_low=dims[2], c_high=None),
            ]
        )

        # BFA modules at all 4 scales
        self.bfa_blocks = nn.ModuleList([GLMambaBFAModule(c) for c in dims])

        # CSIE modules at all 4 scales
        self.csie_blocks = nn.ModuleList([GLMambaCSIEModule(c, groups=csie_groups) for c in dims])

        # Top-down summation projections (Fig. 1 right panel)
        self.up_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dims[i + 1], dims[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(dims[i]),
                    nn.ReLU(inplace=True),
                )
                for i in range(3)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], num_classes, kernel_size=1),
        )

    def _encode_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        cur = x
        for down, stage in zip(self.down_layers, self.vss_stages):
            cur = stage(down(cur))
            feats.append(cur)
        return feats

    def _apply_csam(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        f1, f2, f3, f4 = feats
        out1 = self.csam_blocks[0](f2=f1, f1=None, f3=f2)
        out2 = self.csam_blocks[1](f2=f2, f1=f1, f3=f3)
        out3 = self.csam_blocks[2](f2=f3, f1=f2, f3=f4)
        out4 = self.csam_blocks[3](f2=f4, f1=f3, f3=None)
        return [out1, out2, out3, out4]

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        hw = img_t1.shape[-2:]
        fa_raw = self._encode_single(img_t1)
        fb_raw = self._encode_single(img_t2)

        fa = self._apply_csam(fa_raw)
        fb = self._apply_csam(fb_raw)

        # BFA + CSIE across all 4 stages
        enhanced = [
            csie(bfa(a_i, b_i))
            for a_i, b_i, bfa, csie in zip(fa, fb, self.bfa_blocks, self.csie_blocks)
        ]

        # Top-down summation decoder (Fig. 1)
        cur = enhanced[3]
        for i in range(2, -1, -1):
            up = F.interpolate(self.up_projs[i](cur), size=enhanced[i].shape[-2:], mode="bilinear", align_corners=False)
            cur = enhanced[i] + up

        seg_logits = F.interpolate(self.classifier(cur), size=hw, mode="bilinear", align_corners=False)
        return ModelOutput(seg_logits=seg_logits)


# ==============================================================================
# 13. Su et al. (2026) - Seg2Change (Category-Agnostic Change Head + OVCD Adapter)
# ==============================================================================


@MODEL_REGISTRY.register("seg2change", aliases=["seg2change_cach", "cach_fdr"])
class Seg2ChangeNet(BaseModel):
    """
    Seg2Change: Adapting Open-Vocabulary Semantic Segmentation Model for Remote Sensing
    Change Detection (Su, Song, Chen, & Wen, arXiv:2604.11231, 2026).

    Implements the Category-Agnostic Change Head (`CHead_FPN_FDR` in `change_head_fdr_dino.py`)
    coupled with a multi-layer encoder:
    1. Feature Modulation Module (FMM): Projects multi-layer features into `(48, 64, 80, 96)` channels
       and resizes via `[ConvTranspose4x4(s=4), ConvTranspose2x2(s=2), Identity(), Conv3x3(s=2)]`.
    2. Bi-temporal Difference Fusion Module (`Seg2ChangeBDFM`): Difference-attention guided feature fusion.
    3. Effective Difference Query Attention (`Seg2ChangeEDQA`): Window-based cross-temporal difference query
       attention + 4-expert Mixture-of-Experts FFN (`MoEFFN`).
    4. Residual Upsampler (`Seg2ChangeResUpBlock`): RefineNet-style multi-scale fusion producing:
       - `seg_logits` (category-agnostic change logits)
       - `aux_seg_logits` / `multi_scale_logits` (`[output1, output2, output3]` for `L_ups`)
       - `sem_rep_t1`, `sem_rep_t2` (`outputA`, `outputB` for `L_sim` unchanged-region similarity loss)
       - Optional open-vocabulary semantic change indexing (`ovcd_change_map`) when `sem_mask_t1`/`sem_mask_t2`
         are provided (Eqs. 2, 22, 23).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        enc_dim: int = 128,
        cach_channels: Sequence[int] = (48, 64, 80, 96),
        hidden_dim: int = 64,
        window_size: int = 9,
        num_experts: int = 4,
        num_heads: int = 8,
    ):
        config = ModelConfig(
            arch="seg2change",
            dims=2,
            in_channels=in_channels,
            num_seg_classes=num_classes,
            base_channels=hidden_dim,
            extra={
                "enc_dim": enc_dim,
                "cach_channels": list(cach_channels),
                "hidden_dim": hidden_dim,
                "window_size": window_size,
                "num_experts": num_experts,
                "num_heads": num_heads,
            },
        )
        super().__init__(config=config)
        self.num_classes = num_classes
        self.cach_channels = tuple(cach_channels)
        self.hidden_dim = hidden_dim

        # Lightweight ViT/Conv stem producing 4 intermediate token maps at 1/8 resolution
        self.patch_stem = nn.Sequential(
            nn.Conv2d(in_channels, enc_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(enc_dim // 2),
            nn.GELU(),
            nn.Conv2d(enc_dim // 2, enc_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(enc_dim),
            nn.GELU(),
            nn.Conv2d(enc_dim, enc_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(enc_dim),
            nn.GELU(),
        )
        self.enc_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(enc_dim, enc_dim, kernel_size=3, padding=1, groups=enc_dim, bias=False),
                    nn.BatchNorm2d(enc_dim),
                    nn.GELU(),
                    nn.Conv2d(enc_dim, enc_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(enc_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )

        # FMM (adapterA, adapterB, resize_layers in change_head_fdr_dino.py)
        self.adapter_a = nn.ModuleList([nn.Conv2d(enc_dim, c, kernel_size=1) for c in cach_channels])
        self.adapter_b = nn.ModuleList([nn.Conv2d(enc_dim, c, kernel_size=1) for c in cach_channels])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(cach_channels[0], cach_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(cach_channels[1], cach_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(cach_channels[3], cach_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        # BDFM
        self.diff_fdr = nn.ModuleList([Seg2ChangeBDFM(c, c) for c in cach_channels])

        # EDQA
        self.aggregators = nn.ModuleList(
            [
                Seg2ChangeEDQA(dim=c, num_heads=num_heads, window_size=window_size, num_experts=num_experts)
                for c in cach_channels
            ]
        )

        # Residual Upsampler (ResUp)
        self.layer_rn = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=3, stride=1, padding=1) for c in cach_channels]
        )
        self.refinenet0 = Seg2ChangeResUpBlock(hidden_dim, has_skip=True)
        self.refinenet1 = Seg2ChangeResUpBlock(hidden_dim, has_skip=True)
        self.refinenet2 = Seg2ChangeResUpBlock(hidden_dim, has_skip=True)
        self.refinenet3 = Seg2ChangeResUpBlock(hidden_dim, has_skip=False)

        self.class_embed = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )
        self.class_embed1 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.class_embed2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.class_embed3 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.class_embed_a = nn.Conv2d(hidden_dim, max(num_classes, 2), kernel_size=1)
        self.class_embed_b = nn.Conv2d(hidden_dim, max(num_classes, 2), kernel_size=1)

    def _extract_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        cur = self.patch_stem(x)
        outs: List[torch.Tensor] = []
        for blk in self.enc_blocks:
            cur = cur + blk(cur)
            outs.append(cur)
        return outs

    def _upsampler(
        self, feats: Sequence[torch.Tensor], get_multi: bool = False
    ) -> Any:
        f0, f1, f2, f3 = [rn(f) for rn, f in zip(self.layer_rn, feats)]
        path3 = self.refinenet3(f3, size=f2.shape[-2:])
        path2 = self.refinenet2(path3, f2, size=f1.shape[-2:])
        path1 = self.refinenet1(path2, f1, size=f0.shape[-2:])
        output = self.refinenet0(path1, f0)
        if get_multi:
            return output, path1, path2, path3
        return output

    def forward(
        self,
        img_t1: torch.Tensor,
        img_t2: Optional[torch.Tensor] = None,
        sem_mask_t1: Optional[torch.Tensor] = None,
        sem_mask_t2: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        if img_t2 is None:
            img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)

        hw = img_t1.shape[-2:]
        enc_a = self._extract_layers(img_t1)
        enc_b = self._extract_layers(img_t2)

        feats_a: List[torch.Tensor] = []
        feats_b: List[torch.Tensor] = []
        for i in range(4):
            fa = self.resize_layers[i](self.adapter_a[i](enc_a[i]))
            fb = self.resize_layers[i](self.adapter_b[i](enc_b[i]))
            feats_a.append(fa)
            feats_b.append(fb)

        diff_feats = [
            agg(bdfm(fa, fb), fa, fb)
            for fa, fb, bdfm, agg in zip(feats_a, feats_b, self.diff_fdr, self.aggregators)
        ]

        output, path1, path2, path3 = self._upsampler(diff_feats, get_multi=True)
        out_a_feat = self._upsampler(feats_a, get_multi=False)
        out_b_feat = self._upsampler(feats_b, get_multi=False)

        seg_logits = F.interpolate(self.class_embed(output), size=hw, mode="bilinear", align_corners=False)
        out1 = F.interpolate(self.class_embed1(path1), size=hw, mode="bilinear", align_corners=False)
        out2 = F.interpolate(self.class_embed2(path2), size=hw, mode="bilinear", align_corners=False)
        out3 = F.interpolate(self.class_embed3(path3), size=hw, mode="bilinear", align_corners=False)
        sem_a = F.interpolate(self.class_embed_a(out_a_feat), size=hw, mode="bilinear", align_corners=False)
        sem_b = F.interpolate(self.class_embed_b(out_b_feat), size=hw, mode="bilinear", align_corners=False)

        # Category-agnostic change mask M_ca
        if seg_logits.shape[1] == 1:
            m_ca = (torch.sigmoid(seg_logits) >= 0.5).float()
        else:
            m_ca = torch.argmax(seg_logits, dim=1, keepdim=True).float()

        ovcd_change = None
        if sem_mask_t1 is not None and sem_mask_t2 is not None:
            # Eq. (2): M_ch = M_ca * M_1 + M_ca * M_2
            ovcd_change = ((m_ca * sem_mask_t1.float() + m_ca * sem_mask_t2.float()) > 0).float()

        return ModelOutput(
            seg_logits=seg_logits,
            aux_seg_logits=out1,
            multi_scale_logits=[out1, out2, out3],
            sem_rep_t1=sem_a,
            sem_rep_t2=sem_b,
            category_agnostic_mask=m_ca,
            ovcd_change_map=ovcd_change,
        )

