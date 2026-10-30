"""
Standardized Model Architectures for 3D Medical Imaging, 2D Vision & Remote Sensing,
Bi-Temporal/Sequential Satellite Analysis, and Tabular/Bioinformatics.
"""

from ag_models.models.medical import (
    CoSeg_WaveAttn_SOPSegLiverNet,
    CoSegWaveAttnSOPSegNet,
    MultiTaskLiverNet_MSFF,
    MultiTaskMSFFNet,
    ResVNetFusion,
    WaveAttn_SOPSegLiverNet,
    WaveAttnSOPSegNet,
    WaveCo_MTLNet,
    WaveCoMTLNet,
    WeedSenseLiverNet,
    WeedSenseNet,
)
from ag_models.models.remote_sensing import (
    BiSAMCD,
    BITChangeDetector,
    CBRNet,
    CGCCENet,
    DSTCDNet,
    EdgeRefNet,
    GLMamba,
    GMFIMamba,
    HydroEarlyFusionNet,
    HydroEdgeNet,
    ImprovedMaskRCNNBuilding,
    KGBDCNet,
    SAM2CD,
    SAMCBREdgeNet,
    SAMCEBNet,
    SAM3CEBNet,
    Seg2ChangeNet,
    SemiBCD,
    SNUNetCD,
)
from ag_models.models.sequential import (
    DANNModel,
    HighwayBiLSTM,
    build_mlp,
)
from ag_models.models.temporal import (
    EarlyFusionBiTemporalUNet,
    SiameseUNet,
    TemporalAttentionUNet,
    TemporalConvLSTM_UNet,
    TemporalConvLSTMUNet,
    TemporalSelfAttentionModule,
    build_bitemporal_model,
    build_sequential_model,
)
from ag_models.models.vision_2d import (
    ImageClassificationBackbone,
    SMPSegmentationModel,
    WeakMedSAM_CoSeg_WaveAttn_Net,
    WeakMedSAMCoSegWaveAttnNet,
    build_segmentation_model,
)

__all__ = [
    # 3D / 2D Medical Multi-Task Models
    "WaveCoMTLNet",
    "WaveCo_MTLNet",
    "WaveAttnSOPSegNet",
    "WaveAttn_SOPSegLiverNet",
    "CoSegWaveAttnSOPSegNet",
    "CoSeg_WaveAttn_SOPSegLiverNet",
    "WeedSenseNet",
    "WeedSenseLiverNet",
    "MultiTaskMSFFNet",
    "MultiTaskLiverNet_MSFF",
    "ResVNetFusion",
    # 2D Vision & Remote Sensing Models
    "SMPSegmentationModel",
    "build_segmentation_model",
    "WeakMedSAMCoSegWaveAttnNet",
    "WeakMedSAM_CoSeg_WaveAttn_Net",
    "ImageClassificationBackbone",
    # Bi-Temporal & Spatiotemporal Models
    "SiameseUNet",
    "EarlyFusionBiTemporalUNet",
    "TemporalConvLSTMUNet",
    "TemporalConvLSTM_UNet",
    "TemporalAttentionUNet",
    "TemporalSelfAttentionModule",
    "build_bitemporal_model",
    "build_sequential_model",
    # Sequential & Bioinformatics Models
    "HighwayBiLSTM",
    "DANNModel",
    "build_mlp",
    # Remote Sensing Object/Building Detection, Change Detection & Captioning (2022–2026)
    "ImprovedMaskRCNNBuilding",
    "BiSAMCD",
    "SAM2CD",
    "GMFIMamba",
    "SemiBCD",
    "CGCCENet",
    "KGBDCNet",
    "CBRNet",
    "EdgeRefNet",
    "SAMCEBNet",
    "SAM3CEBNet",
    "SAMCBREdgeNet",
    # Bi-Temporal Earth Observation & Wetland Change Detection (2025-2026)
    "BITChangeDetector",
    "SNUNetCD",
    "HydroEarlyFusionNet",
    "HydroEdgeNet",
    # DSTCD, GLMamba & Seg2Change (2026)
    "DSTCDNet",
    "GLMamba",
    "Seg2ChangeNet",
]


from ag_models.models.change_detection_2026 import FAPMNet, FarmCD, SSCPCDNet  # noqa: E402

__all__ += ["FAPMNet", "FarmCD", "SSCPCDNet"]
