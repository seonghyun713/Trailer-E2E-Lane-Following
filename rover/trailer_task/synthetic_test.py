#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from trailer_angle_estimator import TrailerAngleEstimator, draw_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic robustness test for trailer marker detection.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--camera", default="cam0", choices=("cam0", "cam1"))
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--output-dir", default="", help="Optional directory for annotated trial images.")
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def load_marker(estimator: TrailerAngleEstimator) -> np.ndarray:
    marker_path = Path(estimator.config_path).parent / estimator.config["marker"]["image_path"]
    marker = cv2.imread(str(marker_path.resolve()), cv2.IMREAD_COLOR)
    if marker is None:
        raise RuntimeError(f"Could not read marker: {marker_path}")
    return marker


def roi_abs(estimator: TrailerAngleEstimator, camera_key: str, shape):
    cfg = estimator.config["cameras"][camera_key]["roi"]
    h, w = shape[:2]
    x0 = int(cfg["x"] * w)
    y0 = int(cfg["y"] * h)
    x1 = int((cfg["x"] + cfg["w"]) * w)
    y1 = int((cfg["y"] + cfg["h"]) * h)
    return x0, y0, x1, y1


def render_trial(marker: np.ndarray, estimator: TrailerAngleEstimator, camera_key: str, rng: random.Random):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (130, 128, 120)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 7, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = roi_abs(estimator, camera_key, frame.shape)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (110, 110, 110), -1)

    if rng.random() < 0.5:
        marker = cv2.flip(marker, 1)
    mh, mw = marker.shape[:2]
    scale_w = rng.uniform(0.32, 0.78) * (x1 - x0)
    scale_h = scale_w * mh / mw * rng.uniform(0.86, 1.15)
    cx = rng.uniform(x0 + 0.28 * (x1 - x0), x1 - 0.22 * (x1 - x0))
    cy = rng.uniform(y0 + 0.25 * (y1 - y0), y1 - 0.25 * (y1 - y0))
    skew = rng.uniform(-0.34, 0.34) * scale_w
    roll = rng.uniform(-16, 16)
    pts = np.asarray(
        [
            [-scale_w / 2 - skew * 0.25, -scale_h / 2],
            [scale_w / 2 + skew * 0.25, -scale_h / 2],
            [scale_w / 2 - skew * 0.25, scale_h / 2],
            [-scale_w / 2 + skew * 0.25, scale_h / 2],
        ],
        dtype=np.float32,
    )
    rad = np.deg2rad(roll)
    rot = np.asarray([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    dst = pts @ rot.T + np.asarray([cx, cy], dtype=np.float32)
    src = np.asarray([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(marker, h_mat, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpPerspective(np.full((mh, mw), 255, dtype=np.uint8), h_mat, (frame.shape[1], frame.shape[0]))
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    frame = np.clip(warped.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    estimator = TrailerAngleEstimator(args.config)
    marker = load_marker(estimator)
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failures = []
    for idx in range(args.trials):
        frame = render_trial(marker, estimator, args.camera, rng)
        result = estimator.estimate(frame, args.camera)
        if result.ok:
            ok += 1
        elif len(failures) < 6:
            failures.append(idx)
        if out_dir is not None and (idx < 12 or not result.ok):
            draw = frame.copy()
            draw_result(draw, result)
            cv2.imwrite(str(out_dir / f"trial_{idx:03d}_{'ok' if result.ok else 'fail'}.jpg"), draw)
    rate = ok / max(1, args.trials)
    print(f"camera={args.camera} trials={args.trials} ok={ok} rate={rate:.3f}")
    if failures:
        print("first failures:", failures)
    return 0 if rate >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
