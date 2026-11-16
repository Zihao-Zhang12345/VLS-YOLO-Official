#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seven-model × five-seed ablation runner (35 independent training experiments).

Change MODEL_PATHS / DATA_YAML at the top if needed, then run:
    python train_ablation_35.py

Runs training, held-out test evaluation, and checkpoint compaction in independent
processes to release CUDA memory between stages. Training/evaluation records are
stored outside best.pt and last.pt; the compact checkpoints only hold {'model': ...}.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent
# Prefer the user's custom Ultralytics fork when launched from the project root.
if (Path.cwd() / "ultralytics").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

# =============== 用户可修改区域 ===============
MODEL_PATHS = {
    "M0": ROOT / "",
    "M1": ROOT / "",
    "M2": ROOT / "",
    "M3": ROOT / "",
    "M4": ROOT / "",
    "M5":  ROOT / "",
    "M6":  ROOT / "",
}
DATA_YAML = Path("ultralytics/cfg/datasets/VOC-test.yaml")
OUTPUT_DIR = ROOT / "output_dir/ablation_35"
# ==============================================

MODEL_ORDER = ("M0", "M1", "M2", "M3", "M4", "A", "B")
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("precision_pct", "recall_pct", "map50_pct", "map50_95_pct")
CLASS_METRICS = ("precision_pct", "recall_pct", "f1_pct", "map50_pct", "map75_pct", "map50_95_pct")

# Train settings correspond to the manuscript: SGD / 400 epochs / batch 10 / 512.
TRAIN = {
    "epochs": 400,
    "batch": 10,
    "imgsz": 512,
    "workers": 4,
    "device": "0",
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 1.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "close_mosaic": 200,
    "patience": 0,
    "mosaic": 1.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mixup": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "amp": False,
    "multi_scale": False,
    "deterministic": True,
    "val": True,
    "plots": True,
    "save": True,
}
EVAL = {
    "split": "test",
    "batch": 1,                  # Separate process + batch 1 avoids prior CUDA OOM.
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temp, path)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def abs_path(path):
    path = Path(path).expanduser()
    return str((Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve())


def ablation_check(model_paths):
    """Stop if A/B do not actually differ from the full model.

    This is structural/YAML checking only; full modules must exist in user's fork.
    """
    import yaml
    docs = {}
    for key, path in model_paths.items():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict) and data.get("nc") == 6, f"{key}: nc must be 6"
        assert isinstance(data.get("backbone"), list) and isinstance(data.get("head"), list), key
        docs[key] = data
    for key, expected in (("M0", False), ("M1", False), ("M2", False),
                          ("M3", True), ("M4", True), ("A", False), ("B", True)):
        has_mdcr = any(layer[2] == "MDCR" for layer in docs[key]["backbone"])
        if has_mdcr != expected:
            raise ValueError(f"{key}: unexpected MDCR setting. Cannot run meaningful cumulative ablation.")
    for key, expect_vss in (("M4", True), ("A", True), ("B", False)):
        has_vss = any(layer[2] == "VSSBlockLite" for layer in docs[key]["head"])
        if has_vss != expect_vss:
            raise ValueError(f"{key}: unexpected VSSBlockLite setting.")
    # A and M4 have identical non-MDCR graph; B and M4 share fusion structure.
    for other in ("A", "B"):
        if docs["M4"] == docs[other]:
            raise ValueError(f"{other} is identical to M4: not an ablation experiment")
    return {key: sha256(path) for key, path in model_paths.items()}


def build_configuration(args):
    paths = {k: abs_path(p) for k, p in MODEL_PATHS.items()}
    for name, p in paths.items():
        if not Path(p).is_file():
            raise FileNotFoundError(f"模型 YAML 不存在（可在 MODEL_PATHS 中修改）：{name}={p}")
    data_path = abs_path(args.data or DATA_YAML)
    if not Path(data_path).is_file():
        raise FileNotFoundError(f"数据 YAML 不存在（可在 DATA_YAML 或 --data 中修改）：{data_path}")
    import yaml
    with open(data_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data.get("test"):
        raise ValueError("需要真实独立测试集：数据 YAML 必须包含 test: 配置")
    hashes = ablation_check(paths)
    train = dict(TRAIN)
    train.update({"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
                  "workers": args.workers, "device": args.device})
    evalu = dict(EVAL)
    evalu.update({"split": args.eval_split, "device": args.device, "imgsz": args.imgsz})
    return {
        "model_paths": paths, "model_hashes": hashes,
        "data": data_path, "data_hash": sha256(data_path),
        "train": train, "eval": evalu,
        "precision": args.weight_precision,
        "models_selected": args.models, "seeds_selected": args.seeds,
        "project": abs_path(args.project or OUTPUT_DIR),
    }


def signature(cfg, model_name, seed):
    item = {"model_name": model_name, "model_hash": cfg["model_hashes"][model_name],
            "data_hash": cfg["data_hash"], "train": cfg["train"],
            "eval": cfg["eval"], "seed": int(seed), "precision": cfg["precision"]}
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


def setup_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return random, np, torch


def train_stage(cfg, model_name, seed):
    random, np, torch = setup_torch()
    from ultralytics import YOLO
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    d = Path(cfg["project"]) / model_name / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    sig = signature(cfg, model_name, seed)
    if (d / "train_done.json").exists():
        if read_json(d / "train_done.json")["signature"] != sig:
            raise ValueError("训练配置指纹不一致，不能复用已有实验")
        return
    # Distinguish a partially trained run from an empty directory.
    if (d / "weights").exists() or (d / "results.csv").exists():
        raise RuntimeError(f"检测到未完成的训练目录：{d}。为避免覆盖，请先人工检查/恢复该实验。")

    shutil.copyfile(cfg["model_paths"][model_name], d / "model_used.yaml")
    shutil.copyfile(cfg["data"], d / "dataset_used.yaml")
    print(f"[训练] {model_name}, seed={seed}, YAML={cfg['model_paths'][model_name]}", flush=True)
    net = YOLO(cfg["model_paths"][model_name], task="detect")
    net.train(
        data=cfg["data"], project=str(d.parent), name=d.name,
        exist_ok=True, seed=seed, **cfg["train"]
    )
    if not (d / "weights" / "best.pt").exists() or not (d / "weights" / "last.pt").exists():
        raise RuntimeError("训练未生成 best.pt 和 last.pt")
    args_file = d / "args.yaml"

    if not args_file.is_file():
        alt_args = d / "1args.yaml"
        if alt_args.is_file():
            shutil.copy2(alt_args, args_file)

    if not (d / "results.csv").is_file() or not args_file.is_file():
        raise RuntimeError("训练未生成 results.csv 或 args.yaml")
    write_json(d / "train_done.json", {"signature": sig, "seed": seed,
                                      "model": model_name, "model_sha256": cfg["model_hashes"][model_name],
                                      "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"[训练完成] {model_name} seed{seed}", flush=True)


def get_metrics(result, model_name, seed):
    import numpy as np
    box = result.box
    def fv(v): return float(v) * 100
    out = {
        "model": model_name, "seed": seed,
        "precision_pct": fv(box.mp), "recall_pct": fv(box.mr),
        "map50_pct": fv(box.map50), "map50_95_pct": fv(box.map),
    }
    aps = np.asarray(box.all_ap)
    out["map75_pct"] = fv(np.mean(aps[:, 5])) if aps.ndim == 2 and aps.shape[1] > 5 else None
    f1 = np.asarray(box.f1)
    out["f1_pct"] = fv(np.mean(f1)) if f1.size else None
    names = getattr(result, "names", {}) or {}
    ids = getattr(box, "ap_class_index", None)
    if ids is None:
        ids = np.arange(len(box.p))
    classes = []
    for i, cid in enumerate(np.asarray(ids, dtype=int).tolist()):
        cname = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
        classes.append({
            "model": model_name, "seed": seed, "class_id": cid, "class_name": str(cname),
            "precision_pct": fv(box.p[i]), "recall_pct": fv(box.r[i]),
            "f1_pct": fv(box.f1[i]), "map50_pct": fv(box.ap50[i]),
            "map75_pct": fv(box.all_ap[i, 5]), "map50_95_pct": fv(box.ap[i]),
        })
    return out, classes


def eval_stage(cfg, model_name, seed):
    _, _, _ = setup_torch()
    from ultralytics import YOLO
    d = Path(cfg["project"]) / model_name / f"seed{seed}"
    sig = signature(cfg, model_name, seed)
    if not (d / "train_done.json").exists():
        raise RuntimeError("训练尚未完成，不能进行独立测试")
    best_pt = d / "weights" / "best.pt"
    model = YOLO(str(best_pt), task="detect")
    ekw = dict(cfg["eval"])
    result = model.val(data=cfg["data"], project=str(d), name=f"{ekw.pop('split')}_evaluation",
                       exist_ok=True, save_txt=True, save_conf=True, plots=False,
                       **{**ekw, "split": cfg["eval"]["split"]})
    metrics, class_rows = get_metrics(result, model_name, seed)
    try:
        from ultralytics.utils.torch_utils import model_info
        _, params, _, flops = model_info(model.model, imgsz=cfg["train"]["imgsz"])
        metrics["params_m"] = round(params / 1e6, 6)
        metrics["gflops"] = float(flops)
    except Exception as ex:
        metrics["params_m"], metrics["gflops"] = None, None
        metrics["profiling_warning"] = str(ex)
    speeds = getattr(result, "speed", {}) or {}
    metrics["speed_ms"] = {k: float(v) for k, v in speeds.items() if isinstance(v, (float, int))}
    metrics["inference_ms"] = metrics["speed_ms"].get("inference")
    metrics["end_to_end_ms"] = (sum(metrics["speed_ms"].get(k, 0) for k in
                                    ("preprocess", "inference", "postprocess"))
                                 if metrics["speed_ms"] else None)
    metrics["evaluation_split"] = cfg["eval"]["split"]
    metrics["signature"] = sig
    write_json(d / "metrics.json", metrics)
    write_csv(d / "class_metrics.csv", class_rows,
              ["model", "seed", "class_id", "class_name", *CLASS_METRICS])
    write_json(d / "eval_done.json", {"signature": sig, "metrics": str(d / "metrics.json"),
                                     "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"[测试完成] {model_name} seed{seed}: mAP50={metrics['map50_pct']:.3f}%", flush=True)

def aggregate_model(cfg, model_name, seeds):
    import statistics
    project = Path(cfg["project"])
    ms = []
    cls = []
    for seed in seeds:
        d = project / model_name / f"seed{seed}"
        if not (d / "compact_done.json").exists() or not (d / "metrics.json").exists():
            continue
        item = read_json(d / "metrics.json")
        if item.get("signature") != signature(cfg, model_name, seed):
            raise RuntimeError(f"记录指纹不一致: {d}")
        ms.append(item)
        with (d / "class_metrics.csv").open("r", encoding="utf-8-sig", newline="") as f:
            cls.extend(csv.DictReader(f))
    md = project / model_name
    if ms:
        write_csv(md / "all_seeds.csv", ms,
                  ["model", "seed", *METRICS, "map75_pct", "f1_pct", "params_m", "gflops",
                   "inference_ms", "end_to_end_ms", "evaluation_split", "signature"])
    if cls:
        write_csv(md / "all_classes.csv", cls,
                  ["model", "seed", "class_id", "class_name", *CLASS_METRICS])
    if len(ms) != len(seeds):
        return None
    rows = []
    for key in [*METRICS, "map75_pct", "f1_pct", "params_m", "gflops", "inference_ms", "end_to_end_ms"]:
        values = [float(m[key]) for m in ms if m.get(key) is not None]
        rows.append({"model": model_name, "metric": key, "n": len(values),
                     "mean": statistics.mean(values) if values else None,
                     "sd": statistics.stdev(values) if len(values) > 1 else (0 if values else None)})
    write_csv(md / "summary.csv", rows, ["model", "metric", "n", "mean", "sd"])
    class_summaries = []
    groups = {}
    for rec in cls:
        groups.setdefault((int(rec["class_id"]), rec["class_name"]), []).append(rec)
    for (cid, cname), group in sorted(groups.items()):
        for key in CLASS_METRICS:
            vals = [float(r[key]) for r in group]
            class_summaries.append({"class_id": cid, "class_name": cname, "metric": key,
                                    "n": len(vals), "mean": statistics.mean(vals),
                                    "sd": statistics.stdev(vals) if len(vals) > 1 else 0})
    write_csv(md / "class_summary.csv", class_summaries,
              ["class_id", "class_name", "metric", "n", "mean", "sd"])
    summary = {r["metric"]: r for r in rows}
    print(f"[五种子汇总] {model_name}: mAP@50="
          f"{summary['map50_pct']['mean']:.3f} ± {summary['map50_pct']['sd']:.3f}%", flush=True)
    return {"model": model_name, "n": len(ms), **{
        k + "_mean": summary[k]["mean"] for k in METRICS}, **{
        k + "_sd": summary[k]["sd"] for k in METRICS},
        "params_m": summary["params_m"]["mean"], "gflops": summary["gflops"]["mean"],
        "inference_ms": summary["inference_ms"]["mean"], "end_to_end_ms": summary["end_to_end_ms"]["mean"]}


def tee_worker(cfgpath, model, seed, stage, logfile):
    """Relay tqdm's raw carriage returns so batch updates stay on one terminal line."""
    command = [sys.executable, str(SCRIPT), "--_stage", stage,
               "--_config", str(cfgpath), "--_model", model, "--_seed", str(seed)]
    logfile.parent.mkdir(parents=True, exist_ok=True)
    # Binary mode is essential: text=True turns tqdm's '\r' into '\n'.
    with logfile.open("ab") as f:
        f.write((f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"stage={stage} ---\n").encode("utf-8"))
        f.flush()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            cwd=str(Path.cwd()),
        )
        try:
            while True:
                chunk = os.read(proc.stdout.fileno(), 8192)
                if not chunk:
                    break
                # Preserve \r rather than translating it to a new line.
                os.write(sys.stdout.fileno(), chunk)
                f.write(chunk)
                f.flush()
        finally:
            proc.stdout.close()
        code = proc.wait()
    if code:
        raise RuntimeError(
            f"{model}/seed{seed}/{stage} failed (exit={code}). Log: {logfile}"
        )


def orchestrate(args):
    cfg = build_configuration(args)
    if args.check_only:
        print("[检查通过] 已找到 7 份 YAML、数据集 YAML，消融结构关系正确。")
        for key in MODEL_ORDER:
            print(f"  {key}: {cfg['model_paths'][key]}")
        return
    project = Path(cfg["project"])
    project.mkdir(parents=True, exist_ok=True)
    cfgfile = project / "current_experiment_configuration.json"
    write_json(cfgfile, cfg)
    total = len(args.models) * len(args.seeds)
    n = 0
    for name in args.models:
        for seed in args.seeds:
            n += 1
            d = project / name / f"seed{seed}"
            done = d / "compact_done.json"
            sig = signature(cfg, name, seed)
            for marker in ("train_done.json", "eval_done.json", "compact_done.json"):
                mp = d / marker
                if mp.exists() and read_json(mp).get("signature") != sig:
                    raise RuntimeError(f"已有实验使用了不同的配置，拒绝混用：{mp}")
            if done.exists():
                if (read_json(done).get("signature") != sig or
                        not (d / "metrics.json").exists() or
                        not (d / "weights" / "best.pt").exists() or
                        not (d / "weights" / "last.pt").exists()):
                    raise RuntimeError(f"已有完成标记但文件缺失或配置不同：{d}")
                print(f"[{n}/{total}] 跳过已完成：{name} seed{seed}", flush=True)
            else:
                print(f"\n========== [{n}/{total}] {name} seed={seed} ==========", flush=True)
                try:
                    if not (d / "train_done.json").exists():
                        tee_worker(cfgfile, name, seed, "train", d / "train.log")
                    if not (d / "eval_done.json").exists():
                        tee_worker(cfgfile, name, seed, "eval", d / "evaluation.log")
                    tee_worker(cfgfile, name, seed, "compact", d / "compaction.log")
                except Exception:
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    print(f"[错误] 已保存日志到：{d / 'ERROR.txt'}", flush=True)
                    raise
            aggregate_model(cfg, name, args.seeds)
    comparative = [aggregate_model(cfg, name, args.seeds) for name in args.models]
    if all(x is not None for x in comparative):
        write_csv(project / "35_experiments_comparison.csv", comparative,
                  ["model", "n", *[k + suffix for suffix in ("_mean", "_sd") for k in METRICS],
                   "params_m", "gflops", "inference_ms", "end_to_end_ms"])
    print(f"\n35项实验流程结束，输出路径：{project}", flush=True)


def get_args():
    p = argparse.ArgumentParser(description="7 ablation models × 5 seeds, separate run/test/compact")
    p.add_argument("--data", default=None, help="数据集 YAML，可覆盖文件开头 DATA_YAML")
    p.add_argument("--project", default=None, help="结果根目录")
    p.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    p.add_argument("--batch", type=int, default=TRAIN["batch"])
    p.add_argument("--imgsz", type=int, default=TRAIN["imgsz"])
    p.add_argument("--workers", type=int, default=TRAIN["workers"])
    p.add_argument("--device", default=TRAIN["device"])
    p.add_argument("--eval-split", choices=("test", "val"), default="test")
    p.add_argument("--weight-precision", choices=("keep", "fp16", "fp32"), default="keep")
    p.add_argument("--check-only", action="store_true", help="只校验配置，不进行训练")
    p.add_argument("--_stage", choices=("train", "eval", "compact"), help=argparse.SUPPRESS)
    p.add_argument("--_config", help=argparse.SUPPRESS)
    p.add_argument("--_model", help=argparse.SUPPRESS)
    p.add_argument("--_seed", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be unique")
    if len(set(args.models)) != len(args.models):
        p.error("models must be unique")
    return args


def main():
    args = get_args()
    if args._stage:
        cfg = read_json(args._config)
        if args._stage == "train":
            train_stage(cfg, args._model, args._seed)
        elif args._stage == "eval":
            eval_stage(cfg, args._model, args._seed)
        else:
            compact_stage(cfg, args._model, args._seed)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
