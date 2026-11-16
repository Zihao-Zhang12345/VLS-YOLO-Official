# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
__all__ = (
    "SPDDown",
    "WCBS",
    "PConv",
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Converse2D",
    "DefectAdaptiveFusion"
)

import torch.nn.functional as F

# 假设 autopad 已经导入，如果没有，请保留下面这行注释处的代码
# from ultralytics.utils.ops import autopad 
# 或者使用下面的简化版 autopad:

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectAdaptiveFusion(nn.Module):
    """
    缺陷自适应多尺度特征融合模块。

    输入：
        xs: list/tuple，包含n个尺寸和通道完全相同的特征图
            每个特征形状为 [B, C, H, W]

    输出：
        fused: [B, C, H, W]

    设计：
        1. 为每个空间位置预测不同尺度的融合权重；
        2. 从全部输入分支中提取高频细节，而不是固定使用第一个分支；
        3. 使用可学习权重聚合不同分支的高频信息；
        4. DWConv + PWConv进行轻量细化；
        5. 融合权重初始为等权；
        6. 高频增强和refine分支初始为零，提升训练稳定性。
    """

    def __init__(
        self,
        channels: int,
        n_inputs: int = 2,
        reduction: int = 8
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels必须大于0，当前为{channels}"
            )

        if n_inputs < 2:
            raise ValueError(
                f"n_inputs必须至少为2，当前为{n_inputs}"
            )

        self.channels = channels
        self.n_inputs = n_inputs

        hidden = max(8, channels // reduction)

        # ---------------------------------------------------------
        # 空间自适应融合权重生成器
        #
        # 输入：
        # n个原始特征 + 1个聚合高频特征
        #
        # 输出：
        # 每个空间位置对应n个分支权重
        # ---------------------------------------------------------
        self.weight_generator = nn.Sequential(
            nn.Conv2d(
                channels * (n_inputs + 1),
                hidden,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden,
                n_inputs,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )

        # ---------------------------------------------------------
        # 不同输入分支高频特征的可学习聚合权重
        #
        # 初始化为0，经过softmax以后各分支等权。
        # ---------------------------------------------------------
        self.detail_logits = nn.Parameter(
            torch.zeros(n_inputs)
        )

        # ---------------------------------------------------------
        # 高频细节增强强度
        #
        # 每个通道拥有独立缩放系数。
        # 初始化为0，训练初期不直接增强高频噪声。
        # ---------------------------------------------------------
        self.detail_scale = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        # ---------------------------------------------------------
        # 融合后的轻量特征细化
        # DWConv负责局部空间信息
        # PWConv负责通道混合
        # ---------------------------------------------------------
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(channels)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        稳定初始化：

        1. weight_generator最后一层初始化为0：
           softmax后所有输入分支初始等权。

        2. refine最后一个BN初始化为0：
           refine分支初始不影响主融合结果。

        3. detail_scale本身已初始化为0：
           高频增强分支初始关闭。
        """

        fusion_predictor = self.weight_generator[-1]

        nn.init.zeros_(fusion_predictor.weight)

        if fusion_predictor.bias is not None:
            nn.init.zeros_(fusion_predictor.bias)

        final_bn = self.refine[-1]

        if isinstance(final_bn, nn.BatchNorm2d):
            nn.init.zeros_(final_bn.weight)
            nn.init.zeros_(final_bn.bias)

    @staticmethod
    def _extract_detail(feature: torch.Tensor) -> torch.Tensor:
        """
        使用局部均值残差提取高频信息。

        detail = feature - AvgPool(feature)
        """

        low_frequency = F.avg_pool2d(
            feature,
            kernel_size=3,
            stride=1,
            padding=1
        )

        return feature - low_frequency

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)):
            raise TypeError(
                "DefectAdaptiveFusion输入必须是list或tuple，"
                f"当前输入类型为{type(xs)}"
            )

        if len(xs) != self.n_inputs:
            raise ValueError(
                f"DefectAdaptiveFusion预期{self.n_inputs}个输入，"
                f"实际收到{len(xs)}个输入"
            )

        reference_shape = xs[0].shape

        if len(reference_shape) != 4:
            raise RuntimeError(
                "DefectAdaptiveFusion只支持四维特征图，"
                f"当前形状为{reference_shape}"
            )

        for index, feature in enumerate(xs):
            if feature.shape != reference_shape:
                raise RuntimeError(
                    "DefectAdaptiveFusion所有输入的形状必须相同，"
                    f"第0个输入为{tuple(reference_shape)}，"
                    f"第{index}个输入为{tuple(feature.shape)}"
                )

        _, channels, _, _ = reference_shape

        if channels != self.channels:
            raise RuntimeError(
                f"模块初始化通道数为{self.channels}，"
                f"实际输入通道数为{channels}"
            )

        # =========================================================
        # 1. 提取每一个输入分支的高频信息
        # =========================================================
        detail_features = [
            self._extract_detail(feature)
            for feature in xs
        ]

        # [N, B, C, H, W]
        detail_stack = torch.stack(
            detail_features,
            dim=0
        )

        # 不同输入分支的高频聚合权重
        # 初始为 [1/N, 1/N, ...]
        detail_weights = torch.softmax(
            self.detail_logits,
            dim=0
        )

        detail_weights = detail_weights.view(
            self.n_inputs,
            1,
            1,
            1,
            1
        )

        # 聚合全部输入分支的高频信息
        # [B, C, H, W]
        aggregated_detail = (
            detail_stack * detail_weights
        ).sum(dim=0)

        # =========================================================
        # 2. 预测不同空间位置的多尺度融合权重
        # =========================================================
        fusion_context = torch.cat(
            list(xs) + [aggregated_detail],
            dim=1
        )

        # [B, N, H, W]
        spatial_weights = self.weight_generator(
            fusion_context
        )

        spatial_weights = torch.softmax(
            spatial_weights,
            dim=1
        )

        # =========================================================
        # 3. 对n个输入特征进行空间自适应融合
        # =========================================================

        # [B, N, C, H, W]
        feature_stack = torch.stack(
            list(xs),
            dim=1
        )

        # [B, N, 1, H, W]
        spatial_weights = spatial_weights.unsqueeze(2)

        # [B, C, H, W]
        fused = (
            feature_stack * spatial_weights
        ).sum(dim=1)

        # =========================================================
        # 4. 注入可学习高频细节
        #
        # tanh限制增强系数在[-1, 1]，防止数值过大。
        # =========================================================
        fused = (
            fused
            + torch.tanh(self.detail_scale)
            * aggregated_detail
        )

        # =========================================================
        # 5. 轻量局部细化残差
        #
        # refine最后BN初始为0，所以训练初始：
        # output ≈ 等权融合结果
        # =========================================================
        fused = fused + self.refine(fused)

        return fused

class SPDDown(nn.Module):
    """
    Space-to-Depth detail-preserving downsampling.
    [B,C,H,W] -> [B,4C,H/2,W/2] -> [B,C,H/2,W/2]
    """

    def __init__(self, channels: int):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(
                channels * 4,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        height, width = x.shape[-2:]

        if height % 2 != 0 or width % 2 != 0:
            raise RuntimeError(
                f"SPDDown requires even H/W, got {height}x{width}."
            )

        x = F.pixel_unshuffle(x, downscale_factor=2)
        return self.project(x)


class DWT_Haar(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # 1. 定义 4 个滤波器，形状为 [2, 2]
        f_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        f_lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        f_hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        f_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # 2. 堆叠得到 [4, 2, 2]
        haar = torch.stack([f_ll, f_lh, f_hl, f_hh], dim=0)
        
        # 3. 【关键修复】增加一个维度，变成 [4, 1, 2, 2]
        # 含义: [4个子带, 1个输入通道, kernel_h, kernel_w]
        haar = haar.unsqueeze(1)
        
        # 4. 扩展到所有输入通道
        # groups=in_channels 时，权重形状应为 [in_channels * 4, 1, 2, 2]
        # 这里的 repeat(in_channels, 1, 1, 1) 会将 output_channels 维度变大
        haar = haar.repeat(in_channels, 1, 1, 1)
        
        # 5. 安全检查：确保形状正确，防止报错
        expected_shape = (in_channels * 4, 1, 2, 2)
        assert haar.shape == expected_shape, \
            f"DWT Weight shape error! Expected {expected_shape}, but got {haar.shape}"

        self.register_buffer('weight', haar)

    def forward(self, x):
        # x: [B, C, H, W] -> out: [B, 4*C, H/2, W/2]
        return F.conv2d(x, self.weight, stride=2, groups=self.in_channels)
    
class CBS(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------------------------------
# 👍👍 最推荐：并行融合模块
# DWT 下采样 + CBS 下采样并行，最后融合
# -------------------------------------------------
class DWT_CBS_Fusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        
        self.dwt = DWT_Haar(c1)  # 输出 4*c1
        
        # 在 DWT 之后加入 DWConv（最佳位置）
        self.dw = nn.Sequential(
            nn.Conv2d(4 * c1, 4 * c1, 3, 1, 1, groups=4*c1, bias=False),
            nn.BatchNorm2d(4 * c1),
            nn.SiLU()
        )

        self.cbs = CBS(c1, c2, k=3, s=2)

        # 融合
        self.mix = nn.Conv2d(4 * c1 + c2, c2, 1, 1)

    def forward(self, x):
        xd = self.dwt(x)
        xd = self.dw(xd)   # <-- DWConv 正确位置

        xc = self.cbs(x)
        out = torch.cat([xd, xc], dim=1)
        return self.mix(out)


class WCBS(nn.Module):
    """
    Wavelet-based Downsampling Block.
    Replaces Standard Conv(stride=2) with DWT + Conv(stride=1).
    """
    
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小 (建议为 3)
        s: 步长 (注意：虽然传入可能是2，但内部卷积使用 s=1，因为 DWT 已经缩放了)
        """
        super().__init__()
        
        # 1. 实例化 DWT 模块 (修复点：必须在这里定义)
        self.dwt = DWT_Haar(c1)

        # DWT 会将输入通道数变为 4 倍 (C -> 4C)
        self.dwt_out_channels = c1 * 4
        
        # 这里的卷积 stride 强制为 1，因为尺寸缩减由 DWT 完成
        self.conv = nn.Conv2d(
            self.dwt_out_channels, 
            c2, 
            k, 
            stride=1, 
            padding=autopad(k, p, d), 
            groups=g, 
            dilation=d, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 1. 先进行小波变换
        # (修复点：调用 self.dwt 实例，而不是类名)
        x_dwt = self.dwt(x)
        
        # 2. 再进行卷积特征融合与降维
        return self.act(self.bn(self.conv(x_dwt)))

    def forward_fuse(self, x):
        """融合推理时的逻辑 (可选)"""
        # (修复点：同上)
        x_dwt = self.dwt(x)
        return self.act(self.conv(x_dwt))



# (这个 autopad 函数可能被文件内其他模块使用，保留即可)
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# --- 将下面的完整 class 代码粘贴到你的文件中 ---
class Converse2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale, padding, padding_mode, eps=1e-5):
        super(Converse2D, self).__init__()
        """
        Converse2D Operator for Image Restoration Tasks.
        NOTE: Default values have been removed from the signature to ensure all parameters
        are explicitly passed from the YAML file, preventing ambiguity.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the operation.
            kernel_size (int): Size of the kernel.
            scale (int): Upsampling factor. For example, `scale=2` doubles the resolution.
            padding (int): Padding size.
            padding_mode (str): Padding method. One of {'reflect', 'replicate', 'circular', 'constant'}.
            eps (float, optional): Small value added to denominators for numerical stability.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # ensure depthwise
        assert self.out_channels == self.in_channels, \
            f"Converse2D requires in_channels and out_channels to be the same, but got {self.in_channels} and {self.out_channels}"

        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = F.softmax(self.weight.data.view(1, self.in_channels, -1), dim=-1).view(
            1, self.in_channels, self.kernel_size, self.kernel_size)

    def forward(self, x):
        # Use float32 to avoid precision issues in FFT ops
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float()
            weight_fp32 = self.weight.float()
            bias_fp32 = self.bias.float()

            if self.padding > 0:
                x_fp32 = F.pad(
                    x_fp32,
                    pad=[self.padding, self.padding, self.padding, self.padding],
                    mode=self.padding_mode,
                    value=0,
                )

            # biaseps shape: (1, in_ch_weight, 1, 1)
            biaseps = torch.sigmoid(bias_fp32 - 9.0) + self.eps

            _, _, h, w = x_fp32.shape
            STy = self.upsample(x_fp32, scale=self.scale)  # [B, actual_in, H*scale, W*scale]
            if self.scale != 1:
                x_fp32 = F.interpolate(x_fp32, scale_factor=self.scale, mode='nearest')

            # FB: OTF from PSF -> shape (1, in_ch_weight, Hf, Wf)
            FB = self.p2o(weight_fp32, (h * self.scale, w * self.scale))
            FBC = torch.conj(FB)
            F2B = torch.pow(torch.abs(FB), 2)

            # ---- Compatibility check and adaptation ----
            expected_in = FB.shape[1]
            actual_in = STy.shape[1]

            if expected_in != actual_in:
                if actual_in % expected_in == 0:
                    factor = actual_in // expected_in
                    FB = FB.repeat(1, factor, 1, 1)
                    FBC = FBC.repeat(1, factor, 1, 1)
                    F2B = F2B.repeat(1, factor, 1, 1)
                    biaseps = biaseps.repeat(1, factor, 1, 1)
                else:
                    raise RuntimeError(
                        f"Converse2D channel mismatch: kernel expects {expected_in} channels, "
                        f"but input has {actual_in} channels and it's not an integer multiple. "
                        f"Please ensure the model YAML / upstream Concat produces matching channels."
                    )
            # ---------------------------------------------

            FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
            FR = FBFy + torch.fft.fftn(biaseps * x_fp32, dim=(-2, -1))
            x1 = FB.mul(FR)
            FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
            invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)

            invWBR = FBR.div(invW + biaseps)
            FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
            FX = (FR - FCBinvWBR) / biaseps
            out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

            if self.padding > 0:
                out = out[..., self.padding * self.scale:-self.padding * self.scale,
                      self.padding * self.scale:-self.padding * self.scale]

            return out.to(x.dtype)

    def splits(self, a, scale):
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale
        b = a.view(*leading_dims, scale, W_s, scale, H_s)
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3,
                                                          len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def p2o(self, psf, shape):
        otf = torch.zeros(psf.shape[:-2] + shape, device=psf.device, dtype=psf.dtype)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2, -1))
        return otf

    def upsample(self, x, scale=3):
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale), device=x.device, dtype=x.dtype)
        z[..., st::scale, st::scale].copy_(x)
        return z


# Your original Partial_conv3 module (I've included it for completeness)
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


# =================================================================================
# === HERE IS THE NEW PROPOSED MODULE =============================================
# =================================================================================
class PConv(nn.Module):
    """
    A composite block that uses a 1x1 Conv for downsampling and channel expansion,
    followed by a Partial_conv3 for efficient feature extraction.
    """

    def __init__(self, c1, c2, n_div=4, forward='split_cat'):
        """
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n_div (int): Division factor for the partial convolution. Default is 4.
            forward (str): Forward method for Partial_conv3 ('split_cat' or 'slicing').
        """
        super().__init__()
        # 1x1 convolution with stride 2 for downsampling and channel matching
        self.conv_down = Conv(c1, c2, k=1, s=2)

        # Partial convolution for efficient feature processing
        # It operates on the output of the downsampling conv (c2 channels)
        self.p_conv = Partial_conv3(c2, n_div=n_div, forward=forward)

    def forward(self, x):
        # First, downsample and expand channels
        x = self.conv_down(x)
        # Second, apply the efficient partial convolution
        x = self.p_conv(x)
        return x




class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")




class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
