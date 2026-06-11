#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    image_path: Path
    rel_image_path: str
    camera_key: str
    usable: bool
    box_xyxy: Optional[Tuple[float, float, float, float]]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert trailer labels.csv into an Ultralytics YOLO dataset.")
    parser.add_argument("--dataset", default="", help="bbox_dataset* directory. Defaults to the newest one.")
    parser.add_argument("--output", default="", help="Output YOLO dataset directory. Defaults to DATASET/yolo_panel.")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-name", default="trailer_panel")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument("--root", default=str(here))
    return parser.parse_args()


def newest_dataset(root: Path) -> Path:
    candidates = sorted(
        (p for p in root.glob("bbox_dataset*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No bbox_dataset* directory found under {root}")
    return candidates[0].resolve()


def as_float(value: str) -> Optional[float]:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def parse_box(row: Dict[str, str]) -> Optional[Tuple[float, float, float, float]]:
    # Prefer the generated bbox fields. They are filled for both bbox and quad4 labels.
    vals = [as_float(row.get(k, "")) for k in ("x0", "y0", "x1", "y1")]
    if all(v is not None for v in vals):
        x0, y0, x1, y1 = [float(v) for v in vals if v is not None]
        if x1 > x0 and y1 > y0:
            return x0, y0, x1, y1

    points: List[Tuple[float, float]] = []
    for idx in range(4):
        x = as_float(row.get(f"q{idx}_x", ""))
        y = as_float(row.get(f"q{idx}_y", ""))
        if x is None or y is None:
            return None
        points.append((x, y))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 > x0 and y1 > y0:
        return x0, y0, x1, y1
    return None


def read_samples(dataset: Path) -> List[Sample]:
    labels_csv = dataset / "labels.csv"
    if not labels_csv.exists():
        raise SystemExit(f"labels.csv not found: {labels_csv}")
    samples: List[Sample] = []
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            if not rel:
                continue
            image_path = dataset / rel
            if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            usable = row.get("usable", "0") == "1"
            box = parse_box(row) if usable else None
            samples.append(
                Sample(
                    image_path=image_path,
                    rel_image_path=rel,
                    camera_key=row.get("camera_key", ""),
                    usable=usable and box is not None,
                    box_xyxy=box,
                )
            )
    if not samples:
        raise SystemExit(f"No usable rows found in {labels_csv}")
    return samples


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
    parts = Path(sample.rel_image_path).parts
    return "__".join(parts)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def yolo_line(sample: Sample) -> str:
    assert sample.box_xyxy is not None
    with Image.open(sample.image_path) as img:
        width, height = img.size
    x0, y0, x1, y1 = sample.box_xyxy
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(x0 + 1.0, min(float(width), x1))
    y1 = max(y0 + 1.0, min(float(height), y1))
    xc = ((x0 + x1) * 0.5) / width
    yc = ((y0 + y1) * 0.5) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def write_split(samples: Sequence[Sample], output: Path, split: str, copy: bool) -> Tuple[int, int]:
    positives = 0
    negatives = 0
    for sample in samples:
        name = safe_name(sample)
        image_dst = output / "images" / split / name
        label_dst = output / "labels" / split / (Path(name).stem + ".txt")
        link_or_copy(sample.image_path, image_dst, copy=copy)
        if sample.usable and sample.box_xyxy is not None:
            label_dst.write_text(yolo_line(sample), encoding="utf-8")
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
    root = Path(args.root).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else newest_dataset(root)
    output = Path(args.output).expanduser().resolve() if args.output else dataset / "yolo_panel"
    samples = read_samples(dataset)
    train, val = split_samples(samples, val_ratio=max(0.05, min(0.45, args.val_ratio)), seed=args.seed)
    clear_output(output)
    train_pos, train_neg = write_split(train, output, "train", copy=args.copy)
    val_pos, val_neg = write_split(val, output, "val", copy=args.copy)
    data_yaml = write_data_yaml(output, args.class_name)

    print(f"Dataset: {dataset}")
    print(f"YOLO output: {output}")
    print(f"train: {len(train)} images, positives={train_pos}, negatives={train_neg}")
    print(f"val:   {len(val)} images, positives={val_pos}, negatives={val_neg}")
    print(f"data:  {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
