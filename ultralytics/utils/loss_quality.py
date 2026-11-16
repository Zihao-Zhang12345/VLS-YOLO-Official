"""
保存路径：
    ultralytics/utils/loss_quality.py

适用版本：
    Ultralytics YOLOv8.2.29

功能：
1. 保留YOLOv8原始box、cls、DFL损失；
2. 增加IoU监督的quality质量分支；
3. 对高置信度难负样本进行质量抑制；
4. 增加四边一致性损失，改善mAP75和mAP50-95；
5. 对Trainer仍只返回3项：
       box_loss
       cls_loss
       dfl_loss

其中：
    edge loss合并到box_loss；
    quality loss合并到cls_loss。
"""

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import (
    TaskAlignedAssigner,
    make_anchors,
)


class v8DetectionQualityLoss(v8DetectionLoss):
    """YOLOv8质量校准检测损失。"""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        quality_gain: float = 0.50,
        edge_gain: float = 0.10,
        quality_neg_ratio: int = 3,
        quality_neg_weight: float = 0.25,
        debug_once: bool = True,
    ):
        """
        Args:
            model:
                Ultralytics DetectionModel。

            tal_topk:
                TaskAlignedAssigner正样本候选数量。

            quality_gain:
                质量分支损失权重，最终并入cls_loss。

            edge_gain:
                四边一致性损失权重，最终并入box_loss。

            quality_neg_ratio:
                每个正样本对应选择的高置信度难负样本数量。

            quality_neg_weight:
                难负样本损失权重。

            debug_once:
                是否在第一批数据打印一次正样本分配信息。
        """
        # YOLOv8.2.29父类构造函数只接收model
        super().__init__(model)

        head = model.model[-1]

        # =========================================================
        # 检测头属性检查
        # =========================================================
        required_attributes = (
            "nc",
            "reg_max",
            "nl",
            "stride",
        )

        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(head, attribute)
        ]

        if missing_attributes:
            raise AttributeError(
                "DetectQuality缺少必要属性："
                f"{missing_attributes}"
            )

        head_stride = head.stride.detach().float()

        if head_stride.numel() != int(head.nl):
            raise RuntimeError(
                "DetectQuality stride数量与检测尺度数量不一致："
                f"stride={head_stride.tolist()}, "
                f"head.nl={head.nl}"
            )

        if not torch.isfinite(head_stride).all():
            raise RuntimeError(
                "DetectQuality stride中存在NaN或Inf："
                f"{head_stride.tolist()}"
            )

        if (head_stride <= 0).any():
            raise RuntimeError(
                "DetectQuality stride尚未正确初始化："
                f"{head_stride.tolist()}。"
                "P3/P4/P5检测头的正确值通常应为"
                "[8.0, 16.0, 32.0]。"
            )

        # =========================================================
        # DetectQuality参数
        # =========================================================
        self.nc = int(head.nc)
        self.reg_max = int(head.reg_max)
        self.nl = int(head.nl)

        # 每个检测位置输出：
        # 4 * reg_max个回归通道
        # nc个分类通道
        # 1个quality通道
        self.no = (
            self.reg_max * 4
            + self.nc
            + 1
        )

        self.stride = head.stride

        # 如果topk不是默认10，则重新创建assigner
        if int(tal_topk) != 10:
            self.assigner = TaskAlignedAssigner(
                topk=int(tal_topk),
                num_classes=self.nc,
                alpha=0.5,
                beta=6.0,
            )

        self.quality_gain = float(quality_gain)
        self.edge_gain = float(edge_gain)
        self.quality_neg_ratio = int(
            quality_neg_ratio
        )
        self.quality_neg_weight = float(
            quality_neg_weight
        )

        if self.quality_gain < 0:
            raise ValueError(
                "quality_gain不能小于0。"
            )

        if self.edge_gain < 0:
            raise ValueError(
                "edge_gain不能小于0。"
            )

        if self.quality_neg_ratio < 0:
            raise ValueError(
                "quality_neg_ratio不能小于0。"
            )

        if self.quality_neg_weight < 0:
            raise ValueError(
                "quality_neg_weight不能小于0。"
            )

        self.quality_bce = nn.BCEWithLogitsLoss(
            reduction="none"
        )

        # 修复此前出现的：
        # AttributeError: no attribute '_debug_printed'
        self.debug_once = bool(debug_once)
        self._debug_printed = False

    @staticmethod
    def _aligned_iou_xyxy(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        计算一一对应的xyxy边界框IoU。

        Args:
            boxes1: [N, 4]
            boxes2: [N, 4]

        Returns:
            iou: [N]
        """
        if boxes1.shape != boxes2.shape:
            raise ValueError(
                "IoU输入形状不一致："
                f"boxes1={tuple(boxes1.shape)}, "
                f"boxes2={tuple(boxes2.shape)}"
            )

        if (
            boxes1.ndim != 2
            or boxes1.shape[-1] != 4
        ):
            raise ValueError(
                "IoU输入必须是[N,4]，当前为："
                f"{tuple(boxes1.shape)}"
            )

        left_top = torch.maximum(
            boxes1[:, :2],
            boxes2[:, :2],
        )

        right_bottom = torch.minimum(
            boxes1[:, 2:],
            boxes2[:, 2:],
        )

        intersection_wh = (
            right_bottom - left_top
        ).clamp(min=0)

        intersection = (
            intersection_wh[:, 0]
            * intersection_wh[:, 1]
        )

        area1_wh = (
            boxes1[:, 2:]
            - boxes1[:, :2]
        ).clamp(min=0)

        area2_wh = (
            boxes2[:, 2:]
            - boxes2[:, :2]
        ).clamp(min=0)

        area1 = (
            area1_wh[:, 0]
            * area1_wh[:, 1]
        )

        area2 = (
            area2_wh[:, 0]
            * area2_wh[:, 1]
        )

        union = (
            area1
            + area2
            - intersection
        )

        return intersection / union.clamp(
            min=eps
        )

    def _quality_loss(
        self,
        quality_logits: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算quality质量分支损失。

        正样本：
            监督目标为预测框与匹配GT框的IoU。

        负样本：
            选择分类置信度最高的背景位置，
            quality监督目标设为0。
        """
        if (
            quality_logits.ndim != 3
            or quality_logits.shape[-1] != 1
        ):
            raise ValueError(
                "quality_logits应为[B,A,1]，"
                f"当前为{tuple(quality_logits.shape)}"
            )

        logits = quality_logits.squeeze(-1)

        total_loss = logits.new_zeros(())
        total_weight = logits.new_zeros(())

        # =========================================================
        # 正样本质量监督
        # =========================================================
        if fg_mask.any():
            positive_iou = (
                self._aligned_iou_xyxy(
                    pred_bboxes[fg_mask].detach(),
                    target_bboxes[fg_mask].detach(),
                )
                .clamp(0.0, 1.0)
                .to(logits.dtype)
            )

            positive_loss = self.quality_bce(
                logits[fg_mask],
                positive_iou,
            )

            total_loss = (
                total_loss
                + positive_loss.sum()
            )

            total_weight = (
                total_weight
                + positive_loss.new_tensor(
                    float(positive_loss.numel())
                )
            )

        # =========================================================
        # 高置信度难负样本
        # =========================================================
        batch_size = logits.shape[0]

        for batch_index in range(batch_size):
            negative_mask = ~fg_mask[
                batch_index
            ]

            if not negative_mask.any():
                continue

            negative_logits = logits[
                batch_index
            ][negative_mask]

            # 每个背景位置取最高类别置信度
            negative_confidence = (
                pred_scores[batch_index]
                .detach()
                .sigmoid()
                .amax(dim=-1)[negative_mask]
            )

            positive_count = int(
                fg_mask[batch_index]
                .sum()
                .item()
            )

            # 每张图至少选择32个难负样本
            selected_count = max(
                32,
                positive_count
                * self.quality_neg_ratio,
            )

            selected_count = min(
                selected_count,
                int(
                    negative_confidence.numel()
                ),
            )

            if selected_count <= 0:
                continue

            hard_indices = torch.topk(
                negative_confidence,
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices

            hard_negative_logits = (
                negative_logits[hard_indices]
            )

            hard_negative_targets = (
                torch.zeros_like(
                    hard_negative_logits
                )
            )

            negative_loss = self.quality_bce(
                hard_negative_logits,
                hard_negative_targets,
            )

            total_loss = (
                total_loss
                + self.quality_neg_weight
                * negative_loss.sum()
            )

            total_weight = (
                total_weight
                + negative_loss.new_tensor(
                    self.quality_neg_weight
                    * float(
                        negative_loss.numel()
                    )
                )
            )

        return total_loss / total_weight.clamp(
            min=1.0
        )

    @staticmethod
    def _edge_consistency_loss(
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        四边一致性损失。

        在特征图坐标下分别约束：
            left
            top
            right
            bottom

        并使用GT框宽高归一化。
        """
        if not fg_mask.any():
            return pred_bboxes.new_zeros(())

        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]

        target_width = (
            target[:, 2]
            - target[:, 0]
        ).clamp(min=eps)

        target_height = (
            target[:, 3]
            - target[:, 1]
        ).clamp(min=eps)

        horizontal_error = (
            (
                pred[:, 0]
                - target[:, 0]
            ).abs()
            +
            (
                pred[:, 2]
                - target[:, 2]
            ).abs()
        ) / (
            2.0 * target_width
        )

        vertical_error = (
            (
                pred[:, 1]
                - target[:, 1]
            ).abs()
            +
            (
                pred[:, 3]
                - target[:, 3]
            ).abs()
        ) / (
            2.0 * target_height
        )

        edge_error = (
            horizontal_error
            + vertical_error
        )

        positive_weight = (
            target_scores
            .sum(dim=-1)[fg_mask]
            .detach()
            .to(edge_error.dtype)
        )

        denominator = (
            target_scores_sum
            .to(edge_error.dtype)
            .clamp(min=1.0)
        )

        return (
            edge_error
            * positive_weight
        ).sum() / denominator

    def _extract_features(self, preds):
        """
        提取DetectQuality原始多尺度输出。

        训练阶段：
            preds = [P3, P4, P5]

        验证阶段可能为：
            preds = (decoded, [P3, P4, P5])
        """
        if isinstance(preds, tuple):
            if len(preds) < 2:
                raise ValueError(
                    "预测结果tuple长度不足，"
                    "无法获取原始多尺度输出。"
                )

            feats = preds[1]

        else:
            feats = preds

        if not isinstance(
            feats,
            (list, tuple),
        ):
            raise TypeError(
                "DetectQuality原始输出应为"
                "list或tuple，当前类型为："
                f"{type(feats)}"
            )

        if len(feats) != self.nl:
            raise ValueError(
                "检测尺度数量不一致："
                f"期望{self.nl}个尺度，"
                f"实际收到{len(feats)}个尺度。"
            )

        for level, feature in enumerate(feats):
            if not isinstance(
                feature,
                torch.Tensor,
            ):
                raise TypeError(
                    f"第{level}个尺度不是Tensor："
                    f"{type(feature)}"
                )

            if feature.ndim != 4:
                raise ValueError(
                    f"第{level}个尺度应为"
                    "[B,C,H,W]，当前为"
                    f"{tuple(feature.shape)}"
                )

            if feature.shape[1] != self.no:
                raise ValueError(
                    f"第{level}个尺度输出通道错误："
                    f"实际={feature.shape[1]}，"
                    f"期望={self.no}。"
                    "DetectQuality每个尺度必须输出"
                    "4*reg_max+nc+1个通道。"
                )

        return feats

    def __call__(self, preds, batch):
        """
        计算总损失。

        Returns:
            total_loss:
                用于反向传播的总损失。

            loss_items:
                [box_loss, cls_loss, dfl_loss]
        """
        loss = torch.zeros(
            3,
            device=self.device,
        )

        feats = self._extract_features(
            preds
        )

        batch_size = feats[0].shape[0]

        # =========================================================
        # 合并P3、P4、P5输出
        # [B,no,H,W] -> [B,no,A]
        # =========================================================
        merged = torch.cat(
            [
                feature.reshape(
                    batch_size,
                    self.no,
                    -1,
                )
                for feature in feats
            ],
            dim=2,
        )

        (
            pred_distri,
            pred_scores,
            pred_quality,
        ) = merged.split(
            (
                self.reg_max * 4,
                self.nc,
                1,
            ),
            dim=1,
        )

        # [B,C,A] -> [B,A,C]
        pred_distri = (
            pred_distri
            .permute(0, 2, 1)
            .contiguous()
        )

        pred_scores = (
            pred_scores
            .permute(0, 2, 1)
            .contiguous()
        )

        pred_quality = (
            pred_quality
            .permute(0, 2, 1)
            .contiguous()
        )

        dtype = pred_scores.dtype

        # 输入图像尺寸
        image_size = (
            torch.tensor(
                feats[0].shape[2:],
                device=self.device,
                dtype=dtype,
            )
            * self.stride[0].to(
                device=self.device,
                dtype=dtype,
            )
        )

        anchor_points, stride_tensor = (
            make_anchors(
                feats,
                self.stride,
                0.5,
            )
        )

        # =========================================================
        # 处理GT
        # =========================================================
        targets = torch.cat(
            (
                batch["batch_idx"].view(
                    -1,
                    1,
                ),
                batch["cls"].view(
                    -1,
                    1,
                ),
                batch["bboxes"],
            ),
            dim=1,
        )

        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=image_size[
                [1, 0, 1, 0]
            ],
        )

        gt_labels, gt_bboxes = (
            targets.split(
                (1, 4),
                dim=2,
            )
        )

        mask_gt = (
            gt_bboxes
            .sum(dim=2, keepdim=True)
            .gt_(0)
        )

        # 特征图坐标下的预测框
        pred_bboxes = self.bbox_decode(
            anchor_points,
            pred_distri,
        )

        # =========================================================
        # TaskAlignedAssigner
        # =========================================================
        (
            _,
            target_bboxes,
            target_scores,
            fg_mask,
            _,
        ) = self.assigner(
            pred_scores.detach().sigmoid(),

            (
                pred_bboxes.detach()
                * stride_tensor
            ).type(gt_bboxes.dtype),

            anchor_points * stride_tensor,

            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        raw_target_scores_sum = (
            target_scores.sum()
        )

        target_scores_sum = (
            raw_target_scores_sum
            .float()
            .clamp(min=1.0)
        )

        # =========================================================
        # 第一次前向打印调试信息
        # =========================================================
        if (
            self.debug_once
            and not getattr(
                self,
                "_debug_printed",
                False,
            )
        ):
            print("\n" + "=" * 72)
            print("[LOSS DEBUG]")

            print(
                "stride：",
                self.stride
                .detach()
                .cpu()
                .tolist(),
            )

            print(
                "image_size：",
                image_size
                .detach()
                .float()
                .cpu()
                .tolist(),
            )

            print(
                "GT数量：",
                int(mask_gt.sum().item()),
            )

            print(
                "正样本anchor数量：",
                int(fg_mask.sum().item()),
            )

            print(
                "target_scores_sum：",
                float(
                    raw_target_scores_sum
                    .detach()
                    .float()
                    .cpu()
                ),
            )

            print("=" * 72 + "\n")

            self._debug_printed = True

        # 当前batch含GT但没有正样本时直接停止
        if mask_gt.any() and not fg_mask.any():
            raise RuntimeError(
                "当前batch含有GT，但"
                "TaskAlignedAssigner没有分配到"
                "任何正样本。"
                f"stride="
                f"{self.stride.detach().cpu().tolist()}, "
                f"image_size="
                f"{image_size.detach().cpu().tolist()}, "
                f"GT数量="
                f"{int(mask_gt.sum().item())}。"
                "请检查stride、GT坐标和"
                "DetectQuality输出。"
            )

        # =========================================================
        # 原始分类损失
        # =========================================================
        cls_loss = self.bce(
            pred_scores,
            target_scores.to(dtype),
        ).sum() / target_scores_sum.to(
            pred_scores.dtype
        )

        # assigner返回像素坐标，
        # 转换回特征图坐标
        target_bboxes = (
            target_bboxes
            / stride_tensor
        )

        box_loss = pred_scores.new_zeros(())
        dfl_loss = pred_scores.new_zeros(())
        edge_loss = pred_scores.new_zeros(())

        # =========================================================
        # 原始box、DFL和新增edge损失
        # =========================================================
        if fg_mask.any():
            box_loss, dfl_loss = (
                self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )
            )

            edge_loss = (
                self._edge_consistency_loss(
                    pred_bboxes=pred_bboxes,
                    target_bboxes=target_bboxes,
                    target_scores=target_scores,
                    fg_mask=fg_mask,
                    target_scores_sum=(
                        target_scores_sum
                    ),
                )
            )

        # =========================================================
        # 新增quality损失
        # =========================================================
        quality_loss = self._quality_loss(
            quality_logits=pred_quality,
            pred_scores=pred_scores,
            pred_bboxes=pred_bboxes,
            target_bboxes=target_bboxes,
            fg_mask=fg_mask,
        )

        # =========================================================
        # 合并损失
        # =========================================================
        combined_box_loss = (
            box_loss
            + self.edge_gain
            * edge_loss
        )

        combined_cls_loss = (
            cls_loss
            + self.quality_gain
            * quality_loss
        )

        loss[0] = (
            combined_box_loss
            * self.hyp.box
        )

        loss[1] = (
            combined_cls_loss
            * self.hyp.cls
        )

        loss[2] = (
            dfl_loss
            * self.hyp.dfl
        )

        return (
            loss.sum() * batch_size,
            loss.detach(),
        )