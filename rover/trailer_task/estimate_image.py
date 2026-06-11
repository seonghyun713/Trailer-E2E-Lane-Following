#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from trailer_angle_estimator import TrailerAngleEstimator, draw_fused, draw_result, fuse_results, result_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate trailer articulation angle from a saved side-mirror frame.")
    parser.add_argument("image", help="Input image path. Raw camera frames and wide dual-view screenshots are supported.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")), help="Estimator config YAML.")
    parser.add_argument("--camera", default="auto", help="Camera key: cam0, cam1, all, or auto.")
    parser.add_argument("--split-dual", action="store_true", help="Split input image in half: left half -> cam1, right half -> cam0.")
    parser.add_argument("--output", default="", help="Optional annotated output image.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview window.")
    parser.add_argument("--angle-deg", type=float, default=None, help="Known trailer angle for calibration sample.")
    parser.add_argument("--append-calibration", action="store_true", help="Append detected features to calibration CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    estimator = TrailerAngleEstimator(args.config)
    frames_by_camera = {}
    h, w = frame.shape[:2]
    auto_split_dual = args.camera in ("auto", "all") and (w / max(1, h)) > 2.55
    split_dual = args.split_dual or auto_split_dual

    if split_dual:
        mid = w // 2
        frames_by_camera["cam1"] = frame[:, :mid].copy()
        frames_by_camera["cam0"] = frame[:, mid:].copy()
        display = frame.copy()
    else:
        camera_keys = estimator.camera_keys if args.camera in ("auto", "all") else [args.camera]
        frames_by_camera = {key: frame.copy() for key in camera_keys}
        display = frame.copy()

    results = []
    if split_dual:
        left = frames_by_camera.get("cam1")
        right = frames_by_camera.get("cam0")
        res_left = estimator.estimate(left, "cam1")
        res_right = estimator.estimate(right, "cam0")
        draw_result(left, res_left)
        draw_result(right, res_right)
        display = cv2.hconcat([left, right])
        results = [res_left, res_right]
    else:
        for camera_key, cam_frame in frames_by_camera.items():
            result = estimator.estimate(cam_frame, camera_key)
            draw_result(display, result)
            results.append(result)

    fused = fuse_results(results)
    draw_fused(display, fused)

    if args.append_calibration:
        if args.angle_deg is None:
            raise SystemExit("--append-calibration requires --angle-deg.")
        for result in results:
            if result.ok:
                csv_path = estimator.append_calibration_sample(result.camera_key, args.angle_deg, result)
                print(f"calibration sample appended: {csv_path} camera={result.camera_key} angle={args.angle_deg}")

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), display)
        print(f"wrote: {out_path}")

    if args.json or not args.output:
        print(result_json(results, fused))

    if args.show:
        cv2.imshow("trailer angle", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
