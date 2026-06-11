#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from trailer_parking_core import as_float, as_int, load_yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Capture:
    roi_x0: float
    roi_y0: float
    roi_x1: float
    roi_y1: float
    image_w: int
    image_h: int


@dataclass
class Sample:
    image_path: Path
    rel_image_path: str
    camera_key: str
    usable: bool
    box_xyxy: Optional[Tuple[float, float, float, float]]
    capture: Capture


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert manual labels.csv into YOLO crops using the current YAML mirror ROI.")
    parser.add_argument("--dataset", default=str(here / "bbox_dataset_mirror_run_20260606_112344"))
    parser.add_argument("--config", default=str(here / "dotted_lane_following_config.yaml"))
    parser.add_argument("--output", default="", help="Defaults to DATASET/yolo_panel_current_roi.")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-name", default="trailer_panel")
    parser.add_argument("--min-visible-frac", type=float, default=0.35)
    parser.add_argument("--jpg-quality", type=int, default=92)
    return parser.parse_args()


def parse_box(row: Dict[str, str]) -> Optional[Tuple[float, float, float, float]]:
    vals = [as_float(row.get(k, "")) for k in ("x0", "y0", "x1", "y1")]
    if all(v is not None for v in vals):
        x0, y0, x1, y1 = [float(v) for v in vals if v is not None]
        if x1 > x0 and y1 > y0:
            return x0, y0, x1, y1
    return None


def read_captures(dataset: Path) -> Dict[str, Capture]:
    path = dataset / "captures.csv"
    captures: Dict[str, Capture] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            if not rel:
                continue
            captures[rel] = Capture(
                roi_x0=as_float(row.get("roi_x0", "")) or 0.0,
                roi_y0=as_float(row.get("roi_y0", "")) or 0.0,
                roi_x1=as_float(row.get("roi_x1", "")) or 0.0,
                roi_y1=as_float(row.get("roi_y1", "")) or 0.0,
                image_w=as_int(row.get("image_w", ""), 0),
                image_h=as_int(row.get("image_h", ""), 0),
            )
    return captures


def read_samples(dataset: Path) -> List[Sample]:
    captures = read_captures(dataset)
    labels_csv = dataset / "labels.csv"
    samples: List[Sample] = []
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            if not rel:
                continue
            image_path = dataset / rel
            capture = captures.get(rel)
            if capture is None or not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            box = parse_box(row) if row.get("usable", "0") == "1" else None
            samples.append(
                Sample(
                    image_path=image_path,
                    rel_image_path=rel,
                    camera_key=row.get("camera_key", ""),
                    usable=box is not None,
                    box_xyxy=box,
                    capture=capture,
                )
            )
    if not samples:
        raise SystemExit(f"No samples found in {labels_csv}")
    return samples


def current_roi_abs(config: Dict, camera_key: str, fallback_w: int = 640, fallback_h: int = 360) -> Tuple[int, int, int, int]:
    cam_cfg = (((config.get("camera", {}) or {}).get("cameras", {}) or {}).get(camera_key, {}) or {})
    roi = cam_cfg.get("roi", {}) or {}
    full_w = as_int((config.get("camera", {}) or {}).get("output_width", fallback_w), fallback_w)
    full_h = as_int((config.get("camera", {}) or {}).get("output_height", fallback_h), fallback_h)
    x = as_float(roi.get("x", 0.0), 0.0)
    y = as_float(roi.get("y", 0.0), 0.0)
    w = as_float(roi.get("w", 1.0), 1.0)
    h = as_float(roi.get("h", 1.0), 1.0)
    x0 = max(0, min(full_w - 1, int(round(x * full_w))))
    y0 = max(0, min(full_h - 1, int(round(y * full_h))))
    x1 = max(x0 + 1, min(full_w, int(round((x + w) * full_w))))
    y1 = max(y0 + 1, min(full_h, int(round((y + h) * full_h))))
    return x0, y0, x1, y1


def split_samples(samples: Sequence[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    positives = [s for s in samples if s.usable]
    negatives = [s for s in samples if not s.usable]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_group(group: List[Sample]) -> Tuple[List[Sample], List[Sample]]:
        if not group:
            return [], []
        n_val = max(1, int(round(len(group) * val_ratio))) if len(group) > 2 else 1
        n_val = min(len(group) - 1, n_val) if len(group) > 1 else len(group)
        return group[n_val:], group[:n_val]

    train_pos, val_pos = split_group(positives)
    train_neg, val_neg = split_group(negatives)
    train = train_pos + train_neg
    val = val_pos + val_neg
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def clear_output(output: Path) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        path = output / sub
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def safe_name(sample: Sample) -> str:
    return "__".join(Path(sample.rel_image_path).parts)


def crop_label(sample: Sample, roi_full: Tuple[int, int, int, int], min_visible_frac: float) -> Optional[str]:
    if not sample.usable or sample.box_xyxy is None:
        return None
    old = sample.capture
    bx0, by0, bx1, by1 = sample.box_xyxy
    full_box = (old.roi_x0 + bx0, old.roi_y0 + by0, old.roi_x0 + bx1, old.roi_y0 + by1)
    rx0, ry0, rx1, ry1 = roi_full
    ix0 = max(full_box[0], rx0)
    iy0 = max(full_box[1], ry0)
    ix1 = min(full_box[2], rx1)
    iy1 = min(full_box[3], ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    full_area = max(1.0, (full_box[2] - full_box[0]) * (full_box[3] - full_box[1]))
    visible_area = (ix1 - ix0) * (iy1 - iy0)
    if visible_area / full_area < max(0.0, float(min_visible_frac)):
        return None
    cw = max(1.0, rx1 - rx0)
    ch = max(1.0, ry1 - ry0)
    xc = ((ix0 + ix1) * 0.5 - rx0) / cw
    yc = ((iy0 + iy1) * 0.5 - ry0) / ch
    bw = (ix1 - ix0) / cw
    bh = (iy1 - iy0) / ch
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def write_split(
    samples: Sequence[Sample],
    output: Path,
    split: str,
    config: Dict,
    min_visible_frac: float,
    jpg_quality: int,
) -> Tuple[int, int]:
    positives = 0
    negatives = 0
    for sample in samples:
        roi_full = current_roi_abs(config, sample.camera_key)
        rx0, ry0, rx1, ry1 = roi_full
        old = sample.capture
        crop = (rx0 - old.roi_x0, ry0 - old.roi_y0, rx1 - old.roi_x0, ry1 - old.roi_y0)
        name = Path(safe_name(sample)).with_suffix(".jpg").name
        image_dst = output / "images" / split / name
        label_dst = output / "labels" / split / f"{Path(name).stem}.txt"
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(sample.image_path) as raw:
            raw.convert("RGB").crop(tuple(int(round(v)) for v in crop)).save(image_dst, quality=jpg_quality)
        label = crop_label(sample, roi_full, min_visible_frac)
        if label:
            label_dst.write_text(label, encoding="utf-8")
            positives += 1
        else:
            label_dst.write_text("", encoding="utf-8")
            negatives += 1
    return positives, negatives


def write_data_yaml(output: Path, class_name: str) -> Path:
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    config = load_yaml(Path(args.config).expanduser().resolve())
    output = Path(args.output).expanduser().resolve() if args.output else dataset / "yolo_panel_current_roi"
    samples = read_samples(dataset)
    train, val = split_samples(samples, max(0.05, min(0.45, float(args.val_ratio))), int(args.seed))
    clear_output(output)
    train_pos, train_neg = write_split(train, output, "train", config, float(args.min_visible_frac), int(args.jpg_quality))
    val_pos, val_neg = write_split(val, output, "val", config, float(args.min_visible_frac), int(args.jpg_quality))
    data_yaml = write_data_yaml(output, args.class_name)
    print(f"Dataset: {dataset}")
    print(f"Config:  {Path(args.config).expanduser().resolve()}")
    print(f"YOLO output: {output}")
    print(f"train: {len(train)} images, positives={train_pos}, negatives={train_neg}")
    print(f"val:   {len(val)} images, positives={val_pos}, negatives={val_neg}")
    print(f"data:  {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
