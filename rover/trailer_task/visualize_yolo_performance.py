#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


Box = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize YOLO training curves and validation predictions.")
    parser.add_argument("--run-dir", default=str(here / "runs_yolo" / "trailer_panel_yolo11n"))
    parser.add_argument("--yolo-dir", default=str(here / "bbox_dataset_mirror_run_20260606_112344" / "yolo_panel"))
    parser.add_argument("--weights", default="", help="Defaults to RUN_DIR/weights/best.pt.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-grid", type=int, default=24)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_training_curves(run_dir: Path, out_dir: Path) -> Path | None:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    x = df["epoch"] if "epoch" in df else np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=130)
    ax = axes[0, 0]
    for col in ("metrics/precision(B)", "metrics/recall(B)"):
        if col in df:
            ax.plot(x, df[col], label=col.replace("metrics/", "").replace("(B)", ""))
    ax.set_title("Precision / Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for col in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if col in df:
            ax.plot(x, df[col], label=col.replace("metrics/", "").replace("(B)", ""))
    ax.set_title("mAP")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for col in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
        if col in df:
            ax.plot(x, df[col], label=col.replace("train/", ""))
    ax.set_title("Train Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    for col in ("val/box_loss", "val/cls_loss", "val/dfl_loss"):
        if col in df:
            ax.plot(x, df[col], label=col.replace("val/", ""))
    ax.set_title("Val Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    last = df.iloc[-1]
    fig.suptitle(
        "Trailer Panel YOLO Training"
        f" | epoch={int(last.get('epoch', len(df) - 1))}"
        f" | P={float(last.get('metrics/precision(B)', 0)):.3f}"
        f" R={float(last.get('metrics/recall(B)', 0)):.3f}"
        f" mAP50={float(last.get('metrics/mAP50(B)', 0)):.3f}"
        f" mAP50-95={float(last.get('metrics/mAP50-95(B)', 0)):.3f}"
    )
    fig.tight_layout()
    out = out_dir / "performance_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def read_yolo_label(path: Path, image_size: Tuple[int, int]) -> List[Box]:
    if not path.exists():
        return []
    width, height = image_size
    boxes: List[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = [float(v) for v in parts]
        x0 = (xc - bw / 2.0) * width
        y0 = (yc - bh / 2.0) * height
        x1 = (xc + bw / 2.0) * width
        y1 = (yc + bh / 2.0) * height
        boxes.append((x0, y0, x1, y1))
    return boxes


def box_iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def draw_boxes(
    image_path: Path,
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Tuple[Box, float]],
    status: str,
    target_size: Tuple[int, int] = (260, 360),
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in gt_boxes:
        draw.rectangle(box, outline=(0, 190, 255), width=2)
    for box, conf in pred_boxes:
        draw.rectangle(box, outline=(255, 210, 0), width=2)
        draw.text((box[0], max(0, box[1] - 12)), f"{conf:.2f}", fill=(255, 210, 0))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = None
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 2), status, fill=(255, 255, 255), font=font)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    img.thumbnail(target_size, resampling)
    canvas = Image.new("RGB", target_size, (28, 28, 28))
    canvas.paste(img, ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2))
    return canvas


def make_grid(images: Sequence[Image.Image], out_path: Path, cols: int = 6) -> None:
    if not images:
        return
    w, h = images[0].size
    rows = int(math.ceil(len(images) / cols))
    grid = Image.new("RGB", (cols * w, rows * h), (18, 18, 18))
    for idx, img in enumerate(images):
        grid.paste(img, ((idx % cols) * w, (idx // cols) * h))
    grid.save(out_path, quality=92)


def val_images_and_labels(yolo_dir: Path) -> List[Tuple[Path, Path]]:
    image_dir = yolo_dir / "images" / "val"
    label_dir = yolo_dir / "labels" / "val"
    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        pairs.append((image_path, label_dir / (image_path.stem + ".txt")))
    return pairs


def predict_and_visualize(args: argparse.Namespace, out_dir: Path) -> Dict[str, float]:
    from ultralytics import YOLO

    weights = Path(args.weights).expanduser().resolve() if args.weights else Path(args.run_dir) / "weights" / "best.pt"
    yolo_dir = Path(args.yolo_dir).expanduser().resolve()
    model = YOLO(str(weights))

    pairs = val_images_and_labels(yolo_dir)
    tp = fp = fn = tn = 0
    visual_items = []
    error_items = []

    for image_path, label_path in pairs:
        with Image.open(image_path) as img:
            image_size = img.size
        gt = read_yolo_label(label_path, image_size)
        result = model.predict(
            source=str(image_path),
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            device=args.device,
            verbose=False,
        )[0]
        preds: List[Tuple[Box, float]] = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            order = np.argsort(-confs)
            for idx in order[:3]:
                box = tuple(float(v) for v in xyxy[idx])
                preds.append((box, float(confs[idx])))

        best_iou = 0.0
        if gt and preds:
            best_iou = max(box_iou(g, p[0]) for g in gt for p in preds)

        if gt and best_iou >= args.iou:
            tp += 1
            status = f"TP iou={best_iou:.2f}"
        elif gt and not preds:
            fn += 1
            status = "FN"
        elif gt and preds:
            fn += 1
            fp += 1
            status = f"MISS iou={best_iou:.2f}"
        elif not gt and preds:
            fp += 1
            status = f"FP {preds[0][1]:.2f}"
        else:
            tn += 1
            status = "TN"

        item = draw_boxes(image_path, gt, preds, status)
        if len(visual_items) < int(args.max_grid):
            visual_items.append(item)
        if status.startswith(("FN", "FP", "MISS")) and len(error_items) < int(args.max_grid):
            error_items.append(item)

    make_grid(visual_items, out_dir / "val_predictions_grid.jpg")
    make_grid(error_items, out_dir / "val_errors_grid.jpg")

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "val_images": float(len(pairs)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = run_dir / "performance_viz"
    ensure_dir(out_dir)
    curves = plot_training_curves(run_dir, out_dir)
    metrics = predict_and_visualize(args, out_dir)
    summary = out_dir / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"curves: {curves}",
                f"val_images: {int(metrics['val_images'])}",
                f"tp: {int(metrics['tp'])}",
                f"fp: {int(metrics['fp'])}",
                f"fn: {int(metrics['fn'])}",
                f"tn: {int(metrics['tn'])}",
                f"precision: {metrics['precision']:.4f}",
                f"recall: {metrics['recall']:.4f}",
                f"specificity: {metrics['specificity']:.4f}",
                f"pred_grid: {out_dir / 'val_predictions_grid.jpg'}",
                f"error_grid: {out_dir / 'val_errors_grid.jpg'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
