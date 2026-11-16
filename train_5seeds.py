#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配置版：顺序运行5个seed，分别保存独立测试集结果并汇总均值与样本标准差。"""

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import traceback
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# 对齐论文中“deterministic execution was enabled”的描述。
# 部分第三方 CUDA/SSM 算子可能仍存在非确定性，PyTorch 会给出警告。
cudnn.benchmark = False
cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path(__file__).resolve().parent
SUMMARY_KEYS = ("precision_pct", "recall_pct", "f1_pct", "map50_pct", "map75_pct", "map50_95_pct")


def parse_opt():
    parser = argparse.ArgumentParser(description="顺序训练 5 个独立随机种子")
    parser.add_argument("--model", default=str(ROOT / "ultralytics/cfg/models/zzh/yolov8-test-12-1.yaml"))
    parser.add_argument("--data", default=str(ROOT / "ultralytics/cfg/datasets/VOC-test.yaml"))
    parser.add_argument("--project", default=str(ROOT / "output_dir/5-seeds"))
    parser.add_argument("--name", default="NEU-12-1-s_paper_SGD")

    # 与原脚本当前实际生效的参数保持一致
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9388)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_momentum", type=float, default=0.8)
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--close_mosaic", type=int, default=200)
    parser.add_argument("--patience", type=int, default=0)
    # 论文中显式列出的数据增强与训练策略
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)

    # 遇到中断后，可跳过已经完整保存 metrics.json 的种子
    parser.add_argument("--skip-completed", action="store_true")
    # 默认保留训练后 best.pt/last.pt 原有的存储精度（Ultralytics 可能已转为 FP16）。
    parser.add_argument(
        "--weight-precision", choices=["keep", "fp32", "fp16"], default="keep",
        help="最终权重存储精度：keep 不改变原权重数值，fp16 可再压缩（但可能影响精度）。",
    )
    # 论文报告 held-out test 指标，默认对最佳验证集权重再独立评估 test 集。
    parser.add_argument("--eval-split", choices=["train_result", "val", "test"], default="test")
    opt = parser.parse_args()
    opt.optimizer = opt.optimizer.strip()
    if opt.eval_split == "test":
        data_path = Path(opt.data)
        if not data_path.is_file():
            parser.error(f"数据集配置文件不存在：{data_path}")
        with open(data_path, encoding="utf-8") as f:
            dataset_config = yaml.safe_load(f)
        if not isinstance(dataset_config, dict) or not dataset_config.get("test"):
            parser.error("论文需要独立测试集：数据集 YAML 必须具有有效的 test: 字段")
    if len(set(opt.seeds)) != len(opt.seeds):
        parser.error("--seeds 中不能有重复值")
    return opt


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_signature(opt, seed):
    """给每个seed的真实参数生成指纹，防止跳过旧实验时混合不同配置。"""
    config = dict(vars(opt))
    config.pop("skip_completed", None)
    config.pop("seeds", None)
    config["seed"] = int(seed)
    value = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def metrics_from_result(result, seed, best_pt):
    # Ultralytics DetMetrics 中的原始值为 0~1；保存成百分数便于直接用于论文
    box = result.box
    summary = {
        "seed": int(seed),
        "precision_pct": float(result.results_dict["metrics/precision(B)"]) * 100,
        "recall_pct": float(result.results_dict["metrics/recall(B)"]) * 100,
        "f1_pct": float(np.mean(np.asarray(box.f1))) * 100,
        "map50_pct": float(result.results_dict["metrics/mAP50(B)"]) * 100,
        "map50_95_pct": float(result.results_dict["metrics/mAP50-95(B)"]) * 100,
        "map75_pct": float(np.mean(np.asarray(box.all_ap)[:, 5])) * 100,
        "best_pt": str(best_pt),
    }

    names = getattr(result, "names", {})
    if not names:
        names = {}
    ap_ids = getattr(box, "ap_class_index", None)
    if ap_ids is None:
        ap_ids = np.arange(len(box.p))
    class_rows = []
    for i, cls_id in enumerate(np.asarray(ap_ids).astype(int).tolist()):
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (
            names[cls_id] if cls_id < len(names) else str(cls_id)
        )
        class_rows.append({
            "seed": int(seed),
            "class_id": cls_id,
            "class_name": str(cls_name),
            "precision_pct": float(box.p[i]) * 100,
            "recall_pct": float(box.r[i]) * 100,
            "f1_pct": float(box.f1[i]) * 100,
            "map50_pct": float(box.ap50[i]) * 100,
            "map75_pct": float(box.all_ap[i, 5]) * 100,
            "map50_95_pct": float(box.ap[i]) * 100,
        })
    return summary, class_rows

def save_run_reports(seed_dir, summary, class_rows, result, model, best_pt):
    # 即使模型信息/FLOPs 统计失败，也不影响指标保存
    info = {}
    try:
        _, params, _, flops = model_info(model.model)
        info["params"] = int(params)
        info["gflops"] = float(flops)
    except Exception as exc:
        info["model_info_warning"] = str(exc)
    info["weight_size_mb"] = (best_pt.stat().st_size / 1024**2) if best_pt.exists() else None
    speed = getattr(result, "speed", {}) or {}
    info["speed_ms_per_image"] = {k: float(v) for k, v in speed.items() if isinstance(v, (int, float))}
    p_ms = info["speed_ms_per_image"].get("preprocess", 0.0)
    i_ms = info["speed_ms_per_image"].get("inference", 0.0)
    post_ms = info["speed_ms_per_image"].get("postprocess", 0.0)
    info["fps_from_validator_speeds"] = 1000.0 / (p_ms + i_ms + post_ms) if (p_ms + i_ms + post_ms) > 0 else None
    info["fps_inference_only"] = 1000.0 / i_ms if i_ms > 0 else None
    summary.update(info)

    write_json(seed_dir / "metrics.json", summary)
    write_csv(seed_dir / "class_metrics.csv", class_rows,
              ["seed", "class_id", "class_name", "precision_pct", "recall_pct",
               "f1_pct", "map50_pct", "map75_pct", "map50_95_pct"])

    lines = [f"Seed: {summary['seed']}", f"Best checkpoint: {best_pt}",
             "Metrics below are percentages (%)", "", "Overall metrics:"]
    lines += [f"  {key}: {summary[key]:.4f}" for key in SUMMARY_KEYS]
    lines += ["", "Class-wise metrics:"]
    for row in class_rows:
        lines.append(f"  {row['class_name']:<18} " + " ".join(
            f"{key}={row[key]:.4f}" for key in ("precision_pct", "recall_pct", "map50_pct", "map75_pct", "map50_95_pct")
        ))
    lines += ["", f"Model info: {info}",
              "Note: FPS from validator timings is not a controlled end-to-end GPU benchmark."]
    (seed_dir / "paper_data.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def save_aggregates(project_dir, run_name, seeds, eval_split):
    records = []
    all_classes = []
    for seed in seeds:
        seed_dir = project_dir / f"{run_name}_seed{seed}"
        path = seed_dir / "metrics.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
        class_path = seed_dir / "class_metrics.csv"
        if class_path.exists():
            with open(class_path, newline="", encoding="utf-8-sig") as f:
                all_classes.extend(list(csv.DictReader(f)))

    if not records:
        return
    write_csv(project_dir / f"{run_name}_all_seeds.csv", records,
              ["seed", *SUMMARY_KEYS, "best_pt", "params", "gflops", "weight_size_mb",
               "fps_from_validator_speeds", "fps_inference_only"])
    if all_classes:
        write_csv(project_dir / f"{run_name}_all_classes.csv", all_classes,
                  ["seed", "class_id", "class_name", "precision_pct", "recall_pct",
                   "f1_pct", "map50_pct", "map75_pct", "map50_95_pct"])

    # 仅在所有指定种子完成后输出最终统计，避免把部分运行误当成五次结果
    if len(records) != len(seeds):
        print(f"\n已完成 {len(records)}/{len(seeds)} 个 seed；全部完成后生成均值和标准差。")
        return

    summary_rows = []
    for key in SUMMARY_KEYS:
        values = np.asarray([r[key] for r in records], dtype=float)
        summary_rows.append({"metric": key, "n": len(values),
                             "mean_pct": float(values.mean()),
                             "std_pct": float(values.std(ddof=1)) if len(values) > 1 else 0.0})
    write_csv(project_dir / f"{run_name}_summary.csv", summary_rows,
              ["metric", "n", "mean_pct", "std_pct"])

    class_stats = []
    groups = {}
    for row in all_classes:
        key = (int(row["class_id"]), row["class_name"])
        groups.setdefault(key, []).append(row)
    for (cls_id, name), rows in sorted(groups.items()):
        for metric in ("precision_pct", "recall_pct", "f1_pct", "map50_pct", "map75_pct", "map50_95_pct"):
            values = np.asarray([float(r[metric]) for r in rows])
            class_stats.append({"class_id": cls_id, "class_name": name, "metric": metric,
                                "n": len(values), "mean_pct": float(values.mean()),
                                "std_pct": float(values.std(ddof=1)) if len(values) > 1 else 0.0})
    if class_stats:
        write_csv(project_dir / f"{run_name}_class_summary.csv", class_stats,
                  ["class_id", "class_name", "metric", "n", "mean_pct", "std_pct"])

    text = [f"{run_name}: {len(records)} independent runs; seeds={seeds}; evaluation={eval_split}",
            "Mean ± sample SD (percentage points):"]
    text += [f"  {row['metric']}: {row['mean_pct']:.4f} ± {row['std_pct']:.4f}"
             for row in summary_rows]
    (project_dir / f"{run_name}_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n" + "\n".join(text))


def train_one_seed(opt, seed, project_dir):
    seed_dir = project_dir / f"{opt.name}_seed{seed}"
    old_metrics = seed_dir / "metrics.json"
    if seed_dir.exists():
        if opt.skip_completed and old_metrics.is_file() and (seed_dir / "weights/best.pt").is_file():
            with open(old_metrics, encoding="utf-8") as f:
                completed = json.load(f)
            if (completed.get("evaluation_split") != opt.eval_split
                    or completed.get("seed") != seed
                    or completed.get("config_signature") != config_signature(opt, seed)):
                raise ValueError(f"已完成记录与本次 seed/评估划分不一致：{old_metrics}")
            print(f"[跳过] seed={seed}: 已有 metrics.json 和 best.pt")
            return
        raise FileExistsError(f"目录已存在，为保护旧结果不会覆盖：{seed_dir}。如已完成可使用 --skip-completed")

    fix_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n" + "=" * 72)
    print(f"开始 seed={seed} | 模型={opt.model} | epochs={opt.epochs} | batch={opt.batch}")
    print(f"optimizer={opt.optimizer}, lr0={opt.lr0}, cls={opt.cls}, close_mosaic={opt.close_mosaic}")
    print(f"输出目录：{seed_dir}")
    print("=" * 72)

    model = YOLO(opt.model, task="detect")  # 每个 seed 从 YAML 重新构建独立模型
    try:
        result = model.train(
            data=opt.data,
            epochs=opt.epochs,
            batch=opt.batch,
            imgsz=opt.imgsz,
            workers=opt.workers,
            device=opt.device,
            project=str(project_dir),
            name=seed_dir.name,
            exist_ok=True,
            seed=seed,
            deterministic=True,
            amp=False,
            multi_scale=False,
            optimizer=opt.optimizer,
            lr0=opt.lr0,
            lrf=opt.lrf,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            warmup_epochs=opt.warmup_epochs,
            warmup_momentum=opt.warmup_momentum,
            warmup_bias_lr=opt.warmup_bias_lr,
            box=opt.box,
            cls=opt.cls,
            dfl=opt.dfl,
            close_mosaic=opt.close_mosaic,
            mosaic=opt.mosaic,
            hsv_h=opt.hsv_h,
            hsv_s=opt.hsv_s,
            hsv_v=opt.hsv_v,
            translate=opt.translate,
            scale=opt.scale,
            fliplr=opt.fliplr,
            flipud=opt.flipud,
            mixup=opt.mixup,
            degrees=opt.degrees,
            shear=opt.shear,
            perspective=opt.perspective,
            iou=opt.iou,
            max_det=opt.max_det,
            val=True,
            patience=opt.patience,
            plots=True,
            save=True,
        )
        best_pt = seed_dir / "weights" / "best.pt"
        if not best_pt.is_file():
            raise FileNotFoundError(f"训练后未发现 best.pt：{best_pt}")

        # 默认保留旧脚本的训练返回指标（通常是验证集）。
        # 选择 --eval-split test 时，在对应 best.pt 上额外执行独立测试集评估。
        if opt.eval_split != "train_result":
            evaluator = YOLO(str(best_pt), task="detect")
            result = evaluator.val(
                data=opt.data, split=opt.eval_split, imgsz=opt.imgsz,
                batch=opt.batch, workers=opt.workers, device=opt.device,
                iou=opt.iou, max_det=opt.max_det, half=False, plots=True,
                save_txt=True, save_conf=True,
                project=str(seed_dir), name=f"{opt.eval_split}_evaluation", exist_ok=True,
            )
            del evaluator

        # 训练/独立测试结束后，分别精简 best.pt 和 last.pt，保留结果在外部 CSV/JSON。
        # 先保存训练测试指标引用，再对 checkpoint 做推理可用性检查。
        summary, class_rows = metrics_from_result(result, seed, best_pt)
        weight_reports = []
        for filename in ("best.pt", "last.pt"):
            weight_file = seed_dir / "weights" / filename
            weight_reports.append(compact_checkpoint_for_inference(
                weight_file, opt.weight_precision, opt.device, opt.imgsz
            ))
        write_json(seed_dir / "checkpoint_compaction.json", weight_reports)
        summary["evaluation_split"] = opt.eval_split
        summary["config_signature"] = config_signature(opt, seed)
        save_run_reports(seed_dir, summary, class_rows, result, model, best_pt)
        print(f"[完成] seed={seed}，结果已独立保存。")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    opt = parse_opt()
    project_dir = Path(opt.project).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    write_json(project_dir / f"{opt.name}_run_config.json", vars(opt))

    for index, seed in enumerate(opt.seeds, 1):
        print(f"\n===== 第 {index}/{len(opt.seeds)} 次训练 (seed={seed}) =====", flush=True)
        try:
            train_one_seed(opt, seed, project_dir)
        except Exception:
            error_path = project_dir / f"{opt.name}_seed{seed}_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            save_aggregates(project_dir, opt.name, opt.seeds, opt.eval_split)
            print(f"训练中断，错误日志：{error_path}")
            raise  # 出错就停止，避免将不完整的实验伪装成五次成功运行
        save_aggregates(project_dir, opt.name, opt.seeds, opt.eval_split)

    print("\n五个 seed 已全部执行完成，无须再次手动运行训练脚本。")
    print(f"最终汇总：{project_dir / (opt.name + '_summary.csv')}")


if __name__ == "__main__":
    main()
