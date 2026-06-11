#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


@dataclass
class YoloBox:
    cls_id: int
    xc: float
    yc: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Make ROI-zoomed YOLO data from existing trailer panel labels.")
    parser.add_argument("--source", default=str(here / "bbox_dataset_mirror_run_20260606_112344" / "yolo_panel"))
    parser.add_argument("--output", default=str(here / "bbox_dataset_mirror_run_20260606_112344" / "yolo_panel_roi_zoom"))
    parser.add_argument("--aug-per-positive", type=int, default=8)
    parser.add_argument("--aug-per-negative", type=int, default=1)
    parser.add_argument("--edge-aug-per-positive", type=int, default=0)
    parser.add_argument("--min-visible-frac", type=float, default=0.60)
    parser.add_argument("--edge-min-visible-frac", type=float, default=0.10)
    parser.add_argument("--out-width", type=int, default=360)
    parser.add_argument("--out-height", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-originals", action="store_true", default=True)
    parser.add_argument("--no-copy-originals", dest="copy_originals", action="store_false")
    parser.add_argument("--class-name", default="trailer_panel")
    return parser.parse_args()


def read_boxes(path: Path) -> List[YoloBox]:
    if not path.exists():
        return []
    boxes: List[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, w, h = [float(v) for v in parts[1:]]
        except Exception:
            continue
        if w > 0.0 and h > 0.0:
            boxes.append(YoloBox(cls_id, xc, yc, w, h))
    return boxes


def box_to_abs(box: YoloBox, width: int, height: int) -> Tuple[float, float, float, float]:
    bw = box.w * width
    bh = box.h * height
    cx = box.xc * width
    cy = box.yc * height
    return cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5


def crop_box_to_yolo(
    box: YoloBox,
    image_size: Tuple[int, int],
    crop: Tuple[float, float, float, float],
    min_visible_frac: float = 0.60,
) -> Optional[str]:
    width, height = image_size
    x0, y0, x1, y1 = box_to_abs(box, width, height)
    cx0, cy0, cx1, cy1 = crop
    ix0 = max(x0, cx0)
    iy0 = max(y0, cy0)
    ix1 = min(x1, cx1)
    iy1 = min(y1, cy1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    original_area = max(1.0, (x1 - x0) * (y1 - y0))
    visible_area = (ix1 - ix0) * (iy1 - iy0)
    if visible_area / original_area < max(0.0, float(min_visible_frac)):
        return None
    cw = max(1.0, cx1 - cx0)
    ch = max(1.0, cy1 - cy0)
    xc = ((ix0 + ix1) * 0.5 - cx0) / cw
    yc = ((iy0 + iy1) * 0.5 - cy0) / ch
    bw = (ix1 - ix0) / cw
    bh = (iy1 - iy0) / ch
    if bw <= 0.002 or bh <= 0.002:
        return None
    return f"{box.cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def clamp_crop(
    center_x: float,
    center_y: float,
    crop_w: float,
    crop_h: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    crop_w = min(max(2.0, crop_w), float(width))
    crop_h = min(max(2.0, crop_h), float(height))
    x0 = center_x - crop_w * 0.5
    y0 = center_y - crop_h * 0.5
    x0 = min(max(0.0, x0), max(0.0, width - crop_w))
    y0 = min(max(0.0, y0), max(0.0, height - crop_h))
    return x0, y0, x0 + crop_w, y0 + crop_h


def positive_crop(
    rng: random.Random,
    box: YoloBox,
    image_size: Tuple[int, int],
    roi_aspect: float,
) -> Tuple[float, float, float, float]:
    width, height = image_size
    x0, y0, x1, y1 = box_to_abs(box, width, height)
    bw = max(2.0, x1 - x0)
    bh = max(2.0, y1 - y0)
    target_w_frac = rng.uniform(0.18, 0.42)
    target_h_frac = rng.uniform(0.18, 0.46)
    crop_w = bw / target_w_frac
    crop_h = bh / target_h_frac
    aspect = rng.uniform(0.88, 1.18) * roi_aspect
    if crop_w / max(1.0, crop_h) < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect
    center_x = (x0 + x1) * 0.5 + rng.uniform(-0.14, 0.14) * crop_w
    center_y = (y0 + y1) * 0.5 + rng.uniform(-0.14, 0.14) * crop_h
    return clamp_crop(center_x, center_y, crop_w, crop_h, width, height)


def edge_crop(
    rng: random.Random,
    box: YoloBox,
    image_size: Tuple[int, int],
    roi_aspect: float,
) -> Tuple[float, float, float, float]:
    width, height = image_size
    x0, y0, x1, y1 = box_to_abs(box, width, height)
    bw = max(2.0, x1 - x0)
    bh = max(2.0, y1 - y0)
    crop_h = min(float(height) * rng.uniform(0.72, 1.08), bh / rng.uniform(0.18, 0.42))
    crop_h = max(bh * 1.8, crop_h)
    crop_w = crop_h * roi_aspect * rng.uniform(0.92, 1.12)
    crop_w = min(max(crop_w, bw * 3.5), float(width) * 1.25)
    crop_h = crop_w / max(0.1, roi_aspect)
    crop_h = min(max(crop_h, bh * 1.5), float(height) * 1.25)

    visible_frac = rng.uniform(0.12, 0.58)
    side = rng.choice(("left", "right"))
    if side == "right":
        crop_x1 = x0 + bw * visible_frac
        crop_x0 = crop_x1 - crop_w
    else:
        crop_x0 = x1 - bw * visible_frac
        crop_x1 = crop_x0 + crop_w

    center_y = (y0 + y1) * 0.5 + rng.uniform(-0.18, 0.18) * crop_h
    crop_y0 = center_y - crop_h * 0.5
    crop_y0 = min(max(crop_y0, -0.12 * crop_h), float(height) - 0.88 * crop_h)
    return crop_x0, crop_y0, crop_x1, crop_y0 + crop_h


def negative_crop(rng: random.Random, image_size: Tuple[int, int], roi_aspect: float) -> Tuple[float, float, float, float]:
    width, height = image_size
    crop_w = rng.uniform(0.45, 0.95) * width
    crop_h = crop_w / max(0.1, roi_aspect)
    if crop_h > height:
        crop_h = rng.uniform(0.45, 0.95) * height
        crop_w = crop_h * roi_aspect
    center_x = rng.uniform(0.0, width)
    center_y = rng.uniform(0.0, height)
    return clamp_crop(center_x, center_y, crop_w, crop_h, width, height)


def write_crop(
    image: Image.Image,
    boxes: Iterable[YoloBox],
    crop: Tuple[float, float, float, float],
    image_dst: Path,
    label_dst: Path,
    out_size: Tuple[int, int],
    min_visible_frac: float = 0.60,
) -> bool:
    labels = [
        line
        for box in boxes
        if (line := crop_box_to_yolo(box, image.size, crop, min_visible_frac=min_visible_frac)) is not None
    ]
    cropped = image.crop(tuple(int(round(v)) for v in crop)).resize(out_size, RESAMPLE_BILINEAR)
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(image_dst, quality=90)
    label_dst.write_text("".join(labels), encoding="utf-8")
    return bool(labels)


def clear_output(output: Path) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        path = output / sub
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def image_label_pairs(source: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = source / "images" / split
    label_dir = source / "labels" / split
    pairs = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        pairs.append((image_path, label_dir / f"{image_path.stem}.txt"))
    return pairs


def write_data_yaml(output: Path, class_name: str) -> None:
    (output / "data.yaml").write_text(
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


def build_split(args: argparse.Namespace, source: Path, output: Path, split: str, rng: random.Random) -> Tuple[int, int]:
    out_size = (max(32, int(args.out_width)), max(32, int(args.out_height)))
    roi_aspect = out_size[0] / float(out_size[1])
    positives = 0
    negatives = 0
    for idx, (image_path, label_path) in enumerate(image_label_pairs(source, split)):
        boxes = read_boxes(label_path)
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            if args.copy_originals:
                name = f"{image_path.stem}_orig.jpg"
                image_dst = output / "images" / split / name
                label_dst = output / "labels" / split / f"{Path(name).stem}.txt"
                full_crop = (0.0, 0.0, float(image.width), float(image.height))
                if write_crop(image, boxes, full_crop, image_dst, label_dst, out_size):
                    positives += 1
                else:
                    negatives += 1
            if boxes:
                for aug_idx in range(max(0, int(args.aug_per_positive))):
                    box = rng.choice(boxes)
                    crop = positive_crop(rng, box, image.size, roi_aspect)
                    name = f"{image_path.stem}_zoom{aug_idx:02d}.jpg"
                    image_dst = output / "images" / split / name
                    label_dst = output / "labels" / split / f"{Path(name).stem}.txt"
                    if write_crop(image, boxes, crop, image_dst, label_dst, out_size):
                        positives += 1
                    else:
                        negatives += 1
                for aug_idx in range(max(0, int(args.edge_aug_per_positive))):
                    box = rng.choice(boxes)
                    crop = edge_crop(rng, box, image.size, roi_aspect)
                    name = f"{image_path.stem}_edge{aug_idx:02d}.jpg"
                    image_dst = output / "images" / split / name
                    label_dst = output / "labels" / split / f"{Path(name).stem}.txt"
                    if write_crop(
                        image,
                        boxes,
                        crop,
                        image_dst,
                        label_dst,
                        out_size,
                        min_visible_frac=float(args.edge_min_visible_frac),
                    ):
                        positives += 1
                    else:
                        negatives += 1
            else:
                for aug_idx in range(max(0, int(args.aug_per_negative))):
                    crop = negative_crop(rng, image.size, roi_aspect)
                    name = f"{image_path.stem}_neg{aug_idx:02d}.jpg"
                    image_dst = output / "images" / split / name
                    label_dst = output / "labels" / split / f"{Path(name).stem}.txt"
                    write_crop(image, [], crop, image_dst, label_dst, out_size)
                    negatives += 1
    return positives, negatives


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not (source / "data.yaml").exists():
        raise SystemExit(f"source data.yaml not found: {source / 'data.yaml'}")
    rng = random.Random(int(args.seed))
    clear_output(output)
    train_pos, train_neg = build_split(args, source, output, "train", rng)
    val_pos, val_neg = build_split(args, source, output, "val", rng)
    write_data_yaml(output, args.class_name)
    print(f"source: {source}")
    print(f"output: {output}")
    print(f"train positives={train_pos} negatives={train_neg}")
    print(f"val positives={val_pos} negatives={val_neg}")
    print(f"data: {output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
