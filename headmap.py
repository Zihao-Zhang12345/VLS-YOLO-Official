
"""YOLOv8n 与 VLS-YOLO 的 Grad-CAM 可视化。"""

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ===================== 配置区域 =====================

# 必须使用在 NEU-DET 上训练得到的权重
MODEL_BASELINE_PATH = ""
MODEL_OURS_PATH = ""

# 如果论文写的是 held-out test set，这里必须指向测试集
IMG_DIR = "VOCdevkit/images/test"
OUTPUT_DIR = "gradcam_results"

# 保留原脚本的可视化尺寸，可改为 512
IMG_SIZE = 512

# 两种 CAM 的平均绝对差异筛选阈值
DIFF_THRESHOLD = 0.15

# 选择用于 Grad-CAM 的特征层
TARGET_LAYER_BASE = -2
TARGET_LAYER_OURS = 25

# 使用前 20 个类别置信度作为梯度目标
TOP_K = 20
EXPECTED_CLASSES = 6

# ===================================================


class YOLOWrapper(torch.nn.Module):
    """提取 YOLO 前向输出中的可微预测张量。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return self._find_prediction(output)

    @classmethod
    def _find_prediction(cls, output):
        if torch.is_tensor(output) and output.ndim == 3:
            return output

        if isinstance(output, (tuple, list)):
            for item in output:
                try:
                    return cls._find_prediction(item)
                except (TypeError, ValueError):
                    pass

        if isinstance(output, dict):
            for item in output.values():
                try:
                    return cls._find_prediction(item)
                except (TypeError, ValueError):
                    pass

        raise TypeError(
            "未找到 [B,4+nc,N] 或 [B,N,4+nc] 格式的预测张量。"
            "自定义检测头可能需要调整 YOLOWrapper。"
        )


class YOLODetectionTarget:
    """将 YOLOv8 检测输出转成可反向传播的标量。"""

    def __init__(self, num_classes, top_k=20):
        self.num_classes = int(num_classes)
        self.top_k = int(top_k)

        if self.top_k <= 0:
            raise ValueError("TOP_K 必须大于 0")

    def __call__(self, model_output):
        if model_output.ndim != 2:
            raise ValueError(
                f"单张图像预测应为二维张量："
                f"{tuple(model_output.shape)}"
            )

        channels = 4 + self.num_classes

        if model_output.shape[0] == channels:
            class_scores = model_output[4:, :]

        elif model_output.shape[1] == channels:
            class_scores = model_output[:, 4:].transpose(0, 1)

        else:
            raise ValueError(
                f"预测维度 {tuple(model_output.shape)} "
                f"与 nc={self.num_classes} 不匹配"
            )

        max_scores = class_scores.max(dim=0).values
        k = min(self.top_k, max_scores.numel())

        return torch.topk(max_scores, k=k).values.sum()


def validate_weights(path, label):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{label} 权重文件不存在：{path}"
        )


def get_class_count(yolo):
    return len(yolo.names)


def build_gradcam(yolo, layer_index, label):
    n = len(yolo.model.model)

    if not -n <= layer_index < n:
        raise IndexError(
            f"{label} 层索引 {layer_index} 超出范围，"
            f"模型共有 {n} 层"
        )

    layer = yolo.model.model[layer_index]

    print(
        f"{label}：Grad-CAM 目标层 = "
        f"{layer_index} ({layer.__class__.__name__})"
    )

    wrapper = YOLOWrapper(yolo.model).eval()

    return GradCAM(
        model=wrapper,
        target_layers=[layer]
    )


def get_cam(cam_engine, input_tensor, target):
    # Grad-CAM 必须启用梯度，不能使用 no_grad
    grayscale_cam = cam_engine(
        input_tensor=input_tensor,
        targets=[target],
        aug_smooth=False,
        eigen_smooth=False
    )[0]

    if not np.isfinite(grayscale_cam).all():
        raise FloatingPointError(
            "Grad-CAM 中出现 NaN 或 Inf"
        )

    if float(grayscale_cam.max()) <= 1e-8:
        raise ValueError(
            "Grad-CAM 全零，请检查目标层与梯度路径"
        )

    return grayscale_cam.astype(np.float32)


def process_images():
    validate_weights(MODEL_BASELINE_PATH, "YOLOv8n")
    validate_weights(MODEL_OURS_PATH, "VLS-YOLO")

    img_dir = Path(IMG_DIR)

    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"图像目录不存在：{img_dir}"
        )

    img_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    )

    if not img_files:
        raise FileNotFoundError(
            f"图像目录中没有图片：{img_dir}"
        )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"设备：{device}")
    print(f"图像数量：{len(img_files)}")
    print(f"输入尺寸：{IMG_SIZE}×{IMG_SIZE}")

    # 加载两个训练好的检测模型
    yolo_base = YOLO(MODEL_BASELINE_PATH)
    yolo_ours = YOLO(MODEL_OURS_PATH)

    for label, model in [
        ("YOLOv8n", yolo_base),
        ("VLS-YOLO", yolo_ours)
    ]:
        nc = get_class_count(model)

        if nc != EXPECTED_CLASSES:
            raise ValueError(
                f"{label} 包含 {nc} 类，"
                f"但 NEU-DET 应包含 "
                f"{EXPECTED_CLASSES} 类。"
                "请使用 NEU-DET 训练权重。"
            )

        model.model.to(device).float().eval()

    target_base = YOLODetectionTarget(
        EXPECTED_CLASSES, TOP_K
    )

    target_ours = YOLODetectionTarget(
        EXPECTED_CLASSES, TOP_K
    )

    saved_count = 0
    error_count = 0

    # 每个模型创建一次 Grad-CAM
    with build_gradcam(
        yolo_base,
        TARGET_LAYER_BASE,
        "YOLOv8n"
    ) as cam_base_engine, build_gradcam(
        yolo_ours,
        TARGET_LAYER_OURS,
        "VLS-YOLO"
    ) as cam_ours_engine:

        for path in tqdm(
            img_files,
            desc="Grad-CAM"
        ):
            try:
                # 读取并转换为 RGB
                bgr_img = cv2.imread(str(path))

                if bgr_img is None:
                    raise ValueError("无法读取图像")

                bgr_img = cv2.resize(
                    bgr_img,
                    (IMG_SIZE, IMG_SIZE)
                )

                rgb_img = cv2.cvtColor(
                    bgr_img,
                    cv2.COLOR_BGR2RGB
                )

                rgb_float = (
                    np.asarray(
                        rgb_img,
                        dtype=np.float32
                    ) / 255.0
                )

                # YOLO 使用 RGB/255，而非 ImageNet 标准化
                chw = np.ascontiguousarray(
                    rgb_float.transpose(2, 0, 1)
                )

                input_tensor = (
                    torch.from_numpy(chw)
                    .unsqueeze(0)
                    .to(device)
                )

                # 生成两个模型的真正 Grad-CAM
                cam_base = get_cam(
                    cam_base_engine,
                    input_tensor,
                    target_base
                )

                cam_ours = get_cam(
                    cam_ours_engine,
                    input_tensor,
                    target_ours
                )

                # 比较独立归一化后的空间响应
                diff_score = float(
                    np.abs(
                        cam_base - cam_ours
                    ).mean()
                )

                if diff_score < DIFF_THRESHOLD:
                    continue

                # 相同 JET 色图、相同融合比例
                vis_base = show_cam_on_image(
                    rgb_float,
                    cam_base,
                    use_rgb=True,
                    colormap=cv2.COLORMAP_JET,
                    image_weight=0.5
                )

                vis_ours = show_cam_on_image(
                    rgb_float,
                    cam_ours,
                    use_rgb=True,
                    colormap=cv2.COLORMAP_JET,
                    image_weight=0.5
                )

                # 原图 | YOLOv8n | VLS-YOLO
                combined_rgb = np.hstack([
                    rgb_img,
                    vis_base,
                    vis_ours
                ])

                combined_bgr = cv2.cvtColor(
                    combined_rgb,
                    cv2.COLOR_RGB2BGR
                )

                category = (
                    path.stem.split("_")[0]
                    if "_" in path.stem
                    else "others"
                )

                save_dir = output_dir / category
                save_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                dst = save_dir / (
                    f"gradcam_diff_{diff_score:.3f}_"
                    f"{path.name}"
                )

                if not cv2.imwrite(
                    str(dst),
                    combined_bgr
                ):
                    raise OSError("保存图像失败")

                saved_count += 1

            except Exception as exc:
                error_count += 1
                print(
                    f"\n处理失败：{path.name}："
                    f"{type(exc).__name__}: {exc}"
                )

    print(
        f"处理完成：保存 {saved_count} 张，"
        f"失败 {error_count} 张。"
    )
    print(f"输出目录：{output_dir}")

    if saved_count == 0:
        print(
            "没有满足差异阈值的图像，"
            "请检查 Grad-CAM 或 DIFF_THRESHOLD。"
        )


if __name__ == "__main__":
    process_images()
