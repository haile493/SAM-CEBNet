"""
Standardized 2D Vision, Remote Sensing, and 2.5D Multi-Slice Architectures.

Standardizes and unifies:
1. `SMPSegmentationModel` (`smp_seg`): Unified wrapper around `segmentation_models_pytorch`
   (U-Net, Unet++, DeepLabV3+, SegFormer, UPerNet, FPN, PAN) with built-in `AutoPadWrapper`.
2. `WeakMedSAMCoSegWaveAttnNet` (`weakmedsam_coseg_waveattn`): 2.5D ResNet-34 + HaarDWT2D +
   SOPSeg Second-Order Texture + Co-Seg Mutual Guidance + Sub-Class Exploration (SCE).
3. `ImageClassificationBackbone` (`image_classifier`): Standardized 2D Classifier supporting
   ResNet18/34/50, DenseNet121, and ViT-B/16 with configurable freezing (`full`, `partial`, `frozen`)
   from `bc_mama_pcr`.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from ag_models.core.base import BaseModel, ModelConfig, ModelOutput
from ag_models.core.registry import MODEL_REGISTRY
from ag_models.modules.adapters import AutoPadWrapper
from ag_models.modules.attention import SOPSegTextureBlock2D
from ag_models.modules.blocks import DoubleConv
from ag_models.modules.coseg import MutualGuidanceGate
from ag_models.modules.wavelet import WaveletAttentionGate2D


# ---------------------------------------------------------------------------
# 1. Unified SMP Geospatial / 2D Semantic Segmentation Builder
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("smp_seg", aliases=["unet_2d", "deeplabv3plus", "segformer", "upernet", "fpn"])
class SMPSegmentationModel(BaseModel):
    """
    Standardized 2D Semantic Segmentation Model wrapping `segmentation_models_pytorch`
    with automatic spatial padding (`AutoPadWrapper`) for arbitrary input dimensions.
    """

    def __init__(
        self,
        arch: str = "unet",
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 9,
        classes: int = 1,
        auto_pad_multiple: int = 32,
    ):
        super().__init__(
            ModelConfig(
                arch=arch,
                dims=2,
                in_channels=in_channels,
                num_seg_classes=classes,
                pretrained=(encoder_weights is not None),
                extra={"encoder_name": encoder_name, "encoder_weights": encoder_weights},
            )
        )
        import segmentation_models_pytorch as smp

        arch_key = arch.lower()
        params = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": classes,
            "activation": None,
        }
        arch_map = {
            "unet": smp.Unet,
            "unet_2d": smp.Unet,
            "deeplabv3+": smp.DeepLabV3Plus,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "segformer": smp.Segformer,
            "upernet": smp.UPerNet,
            "fpn": smp.FPN,
            "pan": smp.PAN,
            "unetplusplus": smp.UnetPlusPlus,
        }
        if arch_key not in arch_map:
            raise ValueError(f"Unsupported SMP architecture '{arch}'. Available: {list(arch_map.keys())}")

        raw_model = arch_map[arch_key](**params)
        self.net = AutoPadWrapper(raw_model, multiple=auto_pad_multiple, dims=2) if auto_pad_multiple > 1 else raw_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_segmentation_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 9,
    classes: int = 1,
    auto_pad_multiple: int = 32,
) -> SMPSegmentationModel:
    """Drop-in factory compatible with `cangio/src/models.py`."""
    return SMPSegmentationModel(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        auto_pad_multiple=auto_pad_multiple,
    )


# ---------------------------------------------------------------------------
# 2. WeakMedSAM + Co-Seg + WaveAttn 2.5D Network
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("weakmedsam_coseg_waveattn", aliases=["weakmedsam_25d"])
class WeakMedSAMCoSegWaveAttnNet(BaseModel):
    """
    2.5D Multi-Slice Architecture combining:
    - ResNet-34 Encoder
    - SOPSeg 2D Second-Order Covariance Texture Block
    - Wavelet-Gated Attention Skip Connections (`HaarDWT2D`)
    - Co-Seg Mutual Guidance Gates
    - Sub-Class Exploration (SCE) Multi-Task Classification Heads
    """

    def __init__(self, in_channels: int = 3, child_classes: int = 4, pretrained: bool = False):
        super().__init__(
            ModelConfig(
                arch="weakmedsam_coseg_waveattn",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=2,
                num_cls_classes=2,
                pretrained=pretrained,
                extra={"child_classes": child_classes},
            )
        )
        self.child_classes = child_classes
        weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet34(weights=weights)

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.initial = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.sopseg = SOPSegTextureBlock2D(512, reduced_c=64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.primary_cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.subclass_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2 * child_classes + 1),
        )

        self.att4 = WaveletAttentionGate2D(f_g=256, f_l=256, f_int=128)
        self.att3 = WaveletAttentionGate2D(f_g=128, f_l=128, f_int=64)
        self.att2 = WaveletAttentionGate2D(f_g=64, f_l=64, f_int=32)
        self.att1 = WaveletAttentionGate2D(f_g=32, f_l=64, f_int=32)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(512, 256, dims=2)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128, dims=2)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64, dims=2)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64 + 32, 32, dims=2)

        self.mutual_gate = MutualGuidanceGate(32, dims=2, norm_type="batchnorm", act_type="relu")
        self.up_final = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.head_liver = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.head_tumor = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x0 = self.initial(x)
        e1 = self.layer1(self.maxpool(x0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.sopseg(self.layer4(e3))

        pooled = torch.flatten(self.avgpool(e4), 1)
        primary_logits = self.primary_cls_head(pooled)
        subclass_logits = self.subclass_head(pooled)

        d4 = self.up4(e4)
        d4 = self.conv4(torch.cat([self.att4(d4, e3), d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.conv3(torch.cat([self.att3(d3, e2), d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.conv2(torch.cat([self.att2(d2, e1), d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.conv1(torch.cat([self.att1(d1, x0), d1], dim=1))

        d_full = self.up_final(d1)
        liv_guided, tum_guided = self.mutual_gate(d_full)

        logit_liver = self.head_liver(liv_guided)
        logit_tumor = self.head_tumor(tum_guided)
        seg_logits = torch.cat([logit_liver, logit_tumor], dim=1)

        return ModelOutput(
            seg_logits=seg_logits,
            cls_logits=primary_logits,
            primary_logits=primary_logits,
            subclass_logits=subclass_logits,
            latent_feat=e4,
        )


WeakMedSAM_CoSeg_WaveAttn_Net = WeakMedSAMCoSegWaveAttnNet


# ---------------------------------------------------------------------------
# 3. Standardized 2D Medical/Vision Classifier (from `bc_mama_pcr`)
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register("image_classifier", aliases=["pcr_classifier"])
class ImageClassificationBackbone(BaseModel):
    """
    Standardized 2D Transfer-Learning Classifier supporting ResNet18/34/50, DenseNet121, and ViT-B/16
    with fine-tuning modes (`full`, `partial`, `frozen`) used in `bc_mama_pcr`.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 2,
        in_channels: int = 3,
        finetune_mode: str = "full",
        dropout: float = 0.3,
        pretrained: bool = False,
    ):
        super().__init__(
            ModelConfig(
                arch=f"classifier_{backbone_name}",
                dims=2,
                in_channels=in_channels,
                num_cls_classes=num_classes,
                dropout=dropout,
                pretrained=pretrained,
                extra={"backbone_name": backbone_name, "finetune_mode": finetune_mode},
            )
        )
        bname = backbone_name.lower()
        if bname == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet18(weights=weights)
            in_feat = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
        elif bname == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet34(weights=weights)
            in_feat = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
        elif bname == "densenet121":
            weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
            net = tv_models.densenet121(weights=weights)
            in_feat = net.classifier.in_features
            net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
        elif bname in ("vit_b16", "vit_b_16"):
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
            net = tv_models.vit_b_16(weights=weights)
            in_feat = net.heads.head.in_features
            net.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
        else:
            raise ValueError(f"Unsupported backbone_name='{backbone_name}'.")

        self.backbone = net
        self.apply_finetune_mode(finetune_mode)

    def apply_finetune_mode(self, mode: str = "full") -> None:
        mode = mode.lower()
        if mode == "full":
            self.unfreeze_all()
        elif mode == "frozen":
            for name, p in self.backbone.named_parameters():
                p.requires_grad = any(k in name for k in ("fc", "classifier", "heads"))
        elif mode == "partial":
            for name, p in self.backbone.named_parameters():
                p.requires_grad = any(k in name for k in ("layer4", "denseblock4", "encoder_layer_11", "fc", "classifier", "heads"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
