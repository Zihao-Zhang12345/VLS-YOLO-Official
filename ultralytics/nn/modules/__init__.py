# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxslim {f} {f} && open {f}')  # pip install onnxslim
    ```
"""
from .lsnet import LSNetBackbone,MSA_Block,LS_Block,Downsample,Stem,C2f_LS_Steel,C2f_LS
from .ska import SKA
from .block import (
    SAACR,
    CBAM,
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    Silence,MDCR
)
from .conv import (
    DWT_CBS_Fusion,
    WCBS,
    Converse2D,
    PConv,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SPDDown,
    DefectAdaptiveFusion
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect,LiteDecoupledHead,Detect_Lite,DetectQuality,AsymSharedDetect,HybridAsymDetect
from .transformer import (
    AIFI,

    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
from .BiFPNRFB import WeightedAdd
from .mamba_yolo import SimpleStem, VisionClueMerge, VSSBlock, XSSBlock,VSSBlockLite,SteelMambaLite,VSSBlockLite_steel
from .fma_block import BasicBlock
from .CSA import CrossScaleAttention,Detect_IoU
from .CAFMAttention import CAFMAttention
from .VSSLite import VSSBlockLite1
from .detail_neck import (
    DetailLocalRefine,
    P2DetailGuide,
)
__all__ = (
    "HybridAsymDetect",
    "AsymSharedDetect",
    "P2DetailGuide",
    "DetailLocalRefine",
    "DetectQuality",
   "C2f_LS_Steel",
   "DefectAdaptiveFusion",
   "SPDDown",
    "VSSBlockLite_steel",
    "SteelMambaLite",
    "DWT_CBS_Fusion",
    "WCBS",
    "MDCR",
    "SAACR",
    "CrossScaleAttention",
    "Detect_IoU",
    "PConv",
    "Detect_Lite",
    "WeightedAdd",
    "LiteDecoupledHead"
    "C2f_LS",
    "VSSBlockLite_RMSNorm",
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "CAFMAttention",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "SimpleStem", "VisionClueMerge", "VSSBlock", "XSSBlock","VSSBlockLite1",
    "Converse2D",
    "VSSBlockLite",
    "BasicBlock",
    "SKA",
    "LSNetBackbone",
    "Stem",
    "MSA_Block",
    "LS_Block",
    "Downsample",
    "CBAM",
)
