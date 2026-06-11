#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


EDGE_COLOR_FIX_WIDTH_RATIO = 0.32
EDGE_COLOR_FIX_STRENGTH = 0.75
EDGE_COLOR_FIX_GREEN_RECOVERY = 0.70


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Apply red/magenta edge color correction to captured trailer datasets.")
    parser.add_argument("datasets", nargs="*", help="Dataset directories. Defaults to every trailer_task/bbox_dataset* directory.")
    parser.add_argument("--root", default=str(here))
    parser.add_argument("--force", action="store_true", help="Re-apply even if a manifest exists.")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--edge-color-fix-width-ratio", type=float, default=EDGE_COLOR_FIX_WIDTH_RATIO)
    parser.add_argument("--edge-color-fix-strength", type=float, default=EDGE_COLOR_FIX_STRENGTH)
    parser.add_argument("--edge-color-fix-green-recovery", type=float, default=EDGE_COLOR_FIX_GREEN_RECOVERY)
    return parser.parse_args()


def discover_datasets(root: Path, requested: Iterable[str]) -> List[Path]:
    if requested:
        return [Path(p).expanduser().resolve() for p in requested]
    return sorted(p.resolve() for p in root.glob("bbox_dataset*") if p.is_dir())


def read_capture_rows(dataset: Path) -> List[Dict[str, str]]:
    csv_path = dataset / "captures.csv"
    if not csv_path.exists():
        rows = []
        for p in sorted((dataset / "images").rglob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(p) as img:
                    w, h = img.size
            except Exception:
                continue
            rows.append(
                {
                    "image_path": str(p.relative_to(dataset)),
                    "save_mode": "full",
                    "image_w": str(w),
                    "image_h": str(h),
                    "roi_x0": "0",
                    "roi_y0": "0",
                    "roi_x1": str(w),
                    "roi_y1": str(h),
                }
            )
        return rows

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def infer_source_width(rows: List[Dict[str, str]]) -> int:
    width = 1
    for row in rows:
        width = max(width, as_int(row.get("image_w"), 1), as_int(row.get("roi_x1"), 1))
    return width


def magenta_excess_mean(frame_rgb: np.ndarray) -> float:
    work = frame_rgb.astype(np.float32)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    return float(np.maximum(np.minimum(red, blue) - green, 0.0).mean())


def red_bias_mean(frame_rgb: np.ndarray) -> float:
    work = frame_rgb.astype(np.float32)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    return float((red - 0.5 * (green + blue)).mean())


def fix_edge_color_cast_global(
    frame_rgb: np.ndarray,
    full_width: int,
    x_offset: int,
    width_ratio: float,
    strength: float,
    green_recovery: float,
) -> np.ndarray:
    full_width = max(1, int(full_width))
    x_offset = int(x_offset)
    crop_width = int(frame_rgb.shape[1])
    edge_width = max(1, int(full_width * max(0.0, min(0.5, width_ratio))))
    x_global = np.arange(x_offset, x_offset + crop_width, dtype=np.float32)
    x_global = np.clip(x_global, 0, full_width - 1)
    distance_to_edge = np.minimum(x_global, full_width - 1 - x_global)
    edge_weight = np.clip((edge_width - distance_to_edge) / edge_width, 0.0, 1.0) ** 2
    weight = edge_weight[np.newaxis, :]
    if float(weight.max()) <= 0.0:
        return frame_rgb

    work = frame_rgb.astype(np.float32)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    magenta_excess = np.maximum(np.minimum(red, blue) - green, 0.0)
    red_excess = np.maximum(red - luma, 0.0)
    blue_excess = np.maximum(blue - luma, 0.0)
    green_deficit = np.maximum(luma - green, 0.0)

    work[:, :, 0] = red - (red_excess + 0.60 * magenta_excess) * strength * weight
    work[:, :, 2] = blue - (0.45 * blue_excess + 0.55 * magenta_excess) * strength * weight
    work[:, :, 1] = green + (green_deficit + 0.35 * magenta_excess) * green_recovery * weight
    return np.clip(work, 0, 255).astype(np.uint8)


def fix_global_red_cast(frame_rgb: np.ndarray) -> np.ndarray:
    work = frame_rgb.astype(np.float32)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    max_ch = np.maximum(np.maximum(red, green), blue)
    min_ch = np.minimum(np.minimum(red, green), blue)
    chroma = max_ch - min_ch

    neutral = (luma > 28.0) & (luma < 235.0) & (chroma < np.maximum(26.0, 0.34 * luma))
    if int(neutral.sum()) < max(64, int(0.04 * neutral.size)):
        neutral = (luma > 28.0) & (luma < 235.0)
    if int(neutral.sum()) >= 16:
        means = work[neutral].reshape(-1, 3).mean(axis=0)
    else:
        means = work.reshape(-1, 3).mean(axis=0)
    means = np.maximum(means, 1.0)
    target = float(np.mean(means))
    gains = np.clip(target / means, 0.55, 1.85)
    gains = 1.0 + (gains - 1.0) * 0.92
    work *= gains.reshape(1, 1, 3)

    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    red_allowed = np.maximum(green, blue) * 1.10 + 8.0
    red_over = np.maximum(red - red_allowed, 0.0)
    work[:, :, 0] = red - 0.72 * red_over
    work[:, :, 1] = green + 0.16 * np.maximum(luma - green, 0.0)
    return np.clip(work, 0, 255).astype(np.uint8)


def save_rgb(path: Path, frame_rgb: np.ndarray, jpg_quality: int) -> None:
    img = Image.fromarray(frame_rgb, "RGB")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(path, quality=max(1, min(95, int(jpg_quality))), optimize=False)
    else:
        img.save(path)


def row_source_geometry(row: Dict[str, str], source_width: int) -> Tuple[int, int]:
    save_mode = row.get("save_mode", "")
    if save_mode == "roi":
        x_offset = as_int(row.get("roi_x0"), 0)
        full_width = max(source_width, as_int(row.get("roi_x1"), source_width))
        return full_width, x_offset
    image_w = as_int(row.get("image_w"), source_width)
    return max(1, image_w), 0


def process_dataset(dataset: Path, args: argparse.Namespace) -> Tuple[int, float, float]:
    manifest = dataset / "edge_color_fix_manifest.csv"
    if manifest.exists() and not args.force:
        print(f"[skip] {dataset} already has {manifest.name}; use --force to re-apply")
        return 0, 0.0, 0.0

    rows = read_capture_rows(dataset)
    source_width = infer_source_width(rows)
    done = 0
    before_sum = 0.0
    after_sum = 0.0
    red_before_sum = 0.0
    red_after_sum = 0.0
    started = time.time()

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "source_width",
                "x_offset",
                "magenta_before",
                "magenta_after",
                "red_bias_before",
                "red_bias_after",
                "updated_at",
            ],
        )
        writer.writeheader()
        for row in rows:
            rel = row.get("image_path", "")
            if not rel:
                continue
            path = dataset / rel
            if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            with Image.open(path) as img:
                frame = np.asarray(img.convert("RGB"), dtype=np.uint8)

            full_width, x_offset = row_source_geometry(row, source_width)
            before = magenta_excess_mean(frame)
            red_before = red_bias_mean(frame)
            fixed = fix_edge_color_cast_global(
                frame,
                full_width=full_width,
                x_offset=x_offset,
                width_ratio=args.edge_color_fix_width_ratio,
                strength=args.edge_color_fix_strength,
                green_recovery=args.edge_color_fix_green_recovery,
            )
            fixed = fix_global_red_cast(fixed)
            after = magenta_excess_mean(fixed)
            red_after = red_bias_mean(fixed)
            save_rgb(path, fixed, args.jpg_quality)
            before_sum += before
            after_sum += after
            red_before_sum += red_before
            red_after_sum += red_after
            done += 1
            writer.writerow(
                {
                    "image_path": rel,
                    "source_width": full_width,
                    "x_offset": x_offset,
                    "magenta_before": f"{before:.4f}",
                    "magenta_after": f"{after:.4f}",
                    "red_bias_before": f"{red_before:.4f}",
                    "red_bias_after": f"{red_after:.4f}",
                    "updated_at": f"{started:.3f}",
                }
            )
            if done % 100 == 0:
                print(f"[{dataset.name}] corrected {done} images")

    mean_before = before_sum / max(1, done)
    mean_after = after_sum / max(1, done)
    red_mean_before = red_before_sum / max(1, done)
    red_mean_after = red_after_sum / max(1, done)
    print(
        f"[done] {dataset}: {done} images, "
        f"magenta {mean_before:.3f} -> {mean_after:.3f}, "
        f"red_bias {red_mean_before:.3f} -> {red_mean_after:.3f}"
    )
    return done, mean_before, mean_after


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    datasets = discover_datasets(root, args.datasets)
    if not datasets:
        print("No dataset directories found.")
        return 1

    total = 0
    for dataset in datasets:
        count, _, _ = process_dataset(dataset, args)
        total += count
    print(f"Total corrected images: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
