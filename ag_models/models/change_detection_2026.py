"""
Change-detection models from three 2026 studies (see `ag_models.modules.change_detection_2026`):

- `SSCPCDNet`  (Ma et al., Pattern Recognition 2026): Siamese encoder + SSCP attention on fused difference features.
- `FAPMNet`    (Chen et al., IEEE GRSL 2026): VSS encoder + flow alignment + prototype memory + CSAF decoder.
- `FarmCD`     (Wang et al., IEEE TGRS 2026): region-guided change detection with FFAM/DFM/FSGM/MFRM and a
               three-phase training schedule.

Each model returns a `ModelOutput` with `seg_logits` (B,1,H,W) as the binary change logits.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ag_models.core.base import BaseModel, ModelConfig, ModelOutput
from ag_models.core.registry import MODEL_REGISTRY
from ag_models.modules.change_detection_2026 import (
    CrossScaleAttentionFusion,
    FarmCDDFM,
    FarmCDFFAM,
    FarmCDFSGM,
    FarmCDMFRM,
    FlowAlignmentModule,
    PrototypeMemoryModule,
    SSCPAttention,
    _cbr,
)
from ag_models.modules.remote_sensing import GLMambaVSSBlock


def _smp_encoder(name: str, in_channels: int, weights: Optional[str]):
    import segmentation_models_pytorch as smp

    return smp.encoders.get_encoder(name, in_channels=in_channels, depth=5, weights=weights)


# =============================================================================
# SSCP-CD
# =============================================================================


@MODEL_REGISTRY.register("sscp_cd", aliases=["sscp_cdnet"])
class SSCPCDNet(BaseModel):
    """
    Siamese ResNet encoder (shared weights) -> per-scale fusion of [|f1-f2|, f1, f2] -> `SSCPAttention`
    -> top-down FPN decoder. The SSCP paper presents SSCP as a plug-and-play module inserted into existing CD
    networks (BIT, BAN, ...); this model is a compact CNN host for it, not a re-implementation of a specific host.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        decoder_dim: int = 64,
        head_num: int = 4,
        use_sscp: bool = True,
    ):
        super().__init__(
            ModelConfig(
                arch="sscp_cd",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                pretrained=encoder_weights is not None,
                extra={"encoder_name": encoder_name, "use_sscp": use_sscp},
            )
        )
        self.encoder = _smp_encoder(encoder_name, in_channels, encoder_weights)
        chs = list(self.encoder.out_channels)[2:]  # strides 4, 8, 16, 32
        self.fuse = nn.ModuleList([_cbr(c * 3, c, 3) for c in chs])
        self.sscp = nn.ModuleList([SSCPAttention(c, head_num=head_num) if use_sscp else nn.Identity() for c in chs])
        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_dim, 1) for c in chs])
        self.smooth = nn.ModuleList([_cbr(decoder_dim, decoder_dim, 3) for _ in chs])
        self.head = nn.Sequential(_cbr(decoder_dim, decoder_dim, 3), nn.Conv2d(decoder_dim, num_classes, 1))

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)
        hw = img_t1.shape[-2:]
        f1 = self.encoder(img_t1)[2:]
        f2 = self.encoder(img_t2)[2:]
        feats: List[torch.Tensor] = []
        for a, b, fu, att in zip(f1, f2, self.fuse, self.sscp):
            feats.append(att(fu(torch.cat([torch.abs(a - b), a, b], dim=1))))
        x = self.lateral[-1](feats[-1])
        x = self.smooth[-1](x)
        for i in range(len(feats) - 2, -1, -1):
            x = self.smooth[i](self.lateral[i](feats[i]) + F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False))
        logits = F.interpolate(self.head(x), size=hw, mode="bilinear", align_corners=False)
        return ModelOutput(seg_logits=logits)


# =============================================================================
# FAPMNet
# =============================================================================


@MODEL_REGISTRY.register("fapmnet", aliases=["fapm_net"])
class FAPMNet(BaseModel):
    """
    FAPMNet adapted to binary change detection with per-date semantic maps.

    Faithful: shared VSS (Mamba-style) Siamese encoder, per-level FAM, PMM (K=16 prototypes), CSAF, and the
    multi-task objective (change + T1/T2 semantics). Adaptations: (i) the encoder is a small VSS stack
    (the paper uses VMamba-base); (ii) the semantic branches predict a binary land-cover map per date
    (`sem_logits_t1/t2`), which for this project is the water mask, instead of 6/7-class SCD labels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depths: Sequence[int] = (2, 2, 2, 2),
        decoder_dim: int = 64,
        num_prototypes: int = 16,
    ):
        super().__init__(
            ModelConfig(
                arch="fapmnet",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                base_channels=base_channels,
                extra={"depths": list(depths), "num_prototypes": num_prototypes},
            )
        )
        dims = [base_channels * (2 ** i) for i in range(4)]
        chans = [in_channels] + dims[:-1]
        self.down = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(chans[i], dims[i], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(dims[i]), nn.GELU())
                for i in range(4)
            ]
        )
        self.stages = nn.ModuleList([nn.Sequential(*[GLMambaVSSBlock(dims[i]) for _ in range(depths[i])]) for i in range(4)])
        d = decoder_dim
        self.proj = nn.ModuleList([nn.Conv2d(c, d, 1) for c in dims])
        self.fam = nn.ModuleList([FlowAlignmentModule(d) for _ in dims])
        self.pmm = nn.ModuleList([PrototypeMemoryModule(d, num_prototypes) for _ in dims])
        self.merge_pre = nn.ModuleList([_cbr(d * 2, d, 3) for _ in dims[:-1]])  # deepest level has no previous state
        self.merge_post = nn.ModuleList([_cbr(d * 2, d, 3) for _ in dims[:-1]])  # deepest level has no previous state
        self.gate = nn.ModuleList([nn.Sequential(nn.Conv2d(d * 2, d, 1), nn.ReLU(inplace=True), nn.Conv2d(d, d, 1), nn.Sigmoid()) for _ in dims])
        self.cd_fuse = nn.ModuleList([_cbr(d * 2, d, 3) for _ in dims])
        self.cd_merge = nn.ModuleList([_cbr(d * 2, d, 3) for _ in dims[:-1]])
        self.csaf = nn.ModuleList([CrossScaleAttentionFusion(d) for _ in dims])
        self.sem_head = nn.Conv2d(d, num_classes, 1)
        self.cd_head = nn.Sequential(_cbr(d, d, 3), nn.Conv2d(d, num_classes, 1))

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for dn, st in zip(self.down, self.stages):
            x = st(dn(x))
            out.append(x)
        return out

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)
        hw = img_t1.shape[-2:]
        f1, f2 = self._encode(img_t1), self._encode(img_t2)
        pre_s = post_s = cd_s = None
        sims = []
        for i in range(3, -1, -1):
            pre, post = self.proj[i](f1[i]), self.proj[i](f2[i])
            if pre_s is not None:
                size = pre.shape[-2:]
                up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
                pre = self.merge_pre[i](torch.cat([pre, up(pre_s)], 1))
                post = self.merge_post[i](torch.cat([post, up(post_s)], 1))
            pre, post, _ = self.fam[i](pre, post)
            pre, sim_a = self.pmm[i](pre)
            post, sim_b = self.pmm[i](post)
            sims.append((sim_a, sim_b))
            g = self.gate[i](torch.cat([pre, post], 1))
            cd = self.cd_fuse[i](torch.cat([g * pre, g * post], 1))
            if cd_s is not None:
                cd = self.cd_merge[i](torch.cat([cd, F.interpolate(cd_s, size=cd.shape[-2:], mode="bilinear", align_corners=False)], 1))
            cd = self.csaf[i](cd)
            pre_s, post_s, cd_s = pre, post, cd
        up_full = lambda t: F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
        return ModelOutput(
            seg_logits=up_full(self.cd_head(cd_s)),
            sem_logits_t1=up_full(self.sem_head(pre_s)),
            sem_logits_t2=up_full(self.sem_head(post_s)),
        )


# =============================================================================
# FarmCD
# =============================================================================


class _RegionBranch(nn.Module):
    """Per-date region extractor: encoder + top-down FFAM chain -> multi-scale features and a region logit map."""

    def __init__(self, in_channels: int, name: str, weights: Optional[str]):
        super().__init__()
        self.encoder = _smp_encoder(name, in_channels, weights)
        chs = list(self.encoder.out_channels)[2:]
        self.ffam = nn.ModuleList([FarmCDFFAM(chs[i], chs[i + 1]) for i in range(3)])
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.channels = chs

    def forward(self, x: torch.Tensor):
        f = self.encoder(x)[2:]
        r = [None] * 4
        r[3] = f[3]
        for i in range(2, -1, -1):
            r[i] = self.ffam[i](f[i], r[i + 1])
        return r, self.head(r[0])


@MODEL_REGISTRY.register("farmcd", aliases=["farm_cd"])
class FarmCD(BaseModel):
    """
    FarmCD: region-guided change detection.

    Two *unshared* per-date region branches (farmland in the paper; here the region of interest is supplied by
    the caller through `roi` supervision, e.g. a water mask) feed multi-scale region features into the shared
    Siamese change branch. Change features pass DFM -> FSGM (soft-hard region gating) -> MFRM, and are merged
    at stride 4. Outputs `seg_logits`, `roi_logits_t1`, `roi_logits_t2`.

    Three-phase schedule (`set_phase`): 1 = train region branches only; 2 = freeze region branches, train
    change modules; 3 = joint fine-tuning. The paper pretrains the region branch on weak labels
    (5% hand-annotated, then pseudo-labels); that label-generation step is not replicated here.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        cd_encoder: str = "resnet34",
        roi_encoder: str = "resnet18",
        encoder_weights: Optional[str] = "imagenet",
        decoder_dim: int = 64,
        gate_threshold: float = 0.3,
    ):
        super().__init__(
            ModelConfig(
                arch="farmcd",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                pretrained=encoder_weights is not None,
                extra={"cd_encoder": cd_encoder, "roi_encoder": roi_encoder, "gate_threshold": gate_threshold},
            )
        )
        self.cd_encoder = _smp_encoder(cd_encoder, in_channels, encoder_weights)
        cd_ch = list(self.cd_encoder.out_channels)[2:]
        self.roi1 = _RegionBranch(in_channels, roi_encoder, encoder_weights)
        self.roi2 = _RegionBranch(in_channels, roi_encoder, encoder_weights)
        assert self.roi1.channels == cd_ch, "region and change encoders must share channel layout (use resnet18/34)"
        self.dfm = nn.ModuleList([FarmCDDFM(c) for c in cd_ch])
        self.fsgm = nn.ModuleList([FarmCDFSGM(c, threshold=gate_threshold) for c in cd_ch])
        self.mfrm = nn.ModuleList([FarmCDMFRM(c, decoder_dim, upsample=False) for c in cd_ch])
        self.merge = _cbr(decoder_dim * 4, decoder_dim, 3)
        self.head = nn.Conv2d(decoder_dim, num_classes, 1)
        self.phase = 3

    def roi_modules(self) -> List[nn.Module]:
        return [self.roi1, self.roi2]

    def cd_modules(self) -> List[nn.Module]:
        return [self.cd_encoder, self.dfm, self.fsgm, self.mfrm, self.merge, self.head]

    def set_phase(self, phase: int) -> None:
        """Freezes/unfreezes parameter groups and sets BN modes for the 3-phase schedule (call after `.train()`)."""
        self.phase = phase
        train_roi = phase in (1, 3)
        train_cd = phase in (2, 3)
        for mods, flag in ((self.roi_modules(), train_roi), (self.cd_modules(), train_cd)):
            for m in mods:
                for p in m.parameters():
                    p.requires_grad = flag
                if not flag:
                    m.eval()

    def forward(self, img_t1: torch.Tensor, img_t2: Optional[torch.Tensor] = None) -> ModelOutput:
        if img_t2 is None:
            img_t1, img_t2 = torch.chunk(img_t1, 2, dim=1)
        hw = img_t1.shape[-2:]
        r1, roi_l1 = self.roi1(img_t1)
        r2, roi_l2 = self.roi2(img_t2)
        p_soft = torch.maximum(torch.sigmoid(roi_l1), torch.sigmoid(roi_l2))
        c1 = self.cd_encoder(img_t1)[2:]
        c2 = self.cd_encoder(img_t2)[2:]
        target = c1[0].shape[-2:]
        outs = []
        for i in range(4):
            d = self.dfm[i](c1[i] + r1[i], c2[i] + r2[i])
            d = self.fsgm[i](d, p_soft)
            y = self.mfrm[i](d)
            outs.append(F.interpolate(y, size=target, mode="bilinear", align_corners=False))
        logits = self.head(self.merge(torch.cat(outs, dim=1)))
        up = lambda t: F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
        return ModelOutput(seg_logits=up(logits), roi_logits_t1=up(roi_l1), roi_logits_t2=up(roi_l2))
