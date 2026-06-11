#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trailer_parking_core import as_bool, as_float, as_int, clamp, load_yaml  # noqa: E402
from live_trailer_parking import apply_color_fix, open_gst_tools, rgb_to_bgr  # noqa: E402


def enabled_camera_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cameras = ((config.get("camera", {}) or {}).get("cameras", {}) or {})
    return {key: value or {} for key, value in cameras.items() if as_bool((value or {}).get("enabled"), True)}


def crop_frame(frame_bgr, config: Dict[str, Any], camera_cfg: Dict[str, Any]):
    crop_cfg = camera_cfg.get("crop") or ((config.get("preprocess", {}) or {}).get("crop", {}) or {})
    h, w = frame_bgr.shape[:2]
    x = clamp(as_float(crop_cfg.get("x"), 0.0), 0.0, 0.99)
    y = clamp(as_float(crop_cfg.get("y"), 0.0), 0.0, 0.99)
    cw = clamp(as_float(crop_cfg.get("w"), 1.0), 0.01, 1.0)
    ch = clamp(as_float(crop_cfg.get("h"), 1.0), 0.01, 1.0)
    x0 = int(round(x * w))
    y0 = int(round(y * h))
    x1 = max(x0 + 1, min(w, int(round((x + cw) * w))))
    y1 = max(y0 + 1, min(h, int(round((y + ch) * h))))
    return frame_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def open_camera(config: Dict[str, Any], camera_key: str, camera_cfg: Dict[str, Any], GstCamera):
    root = config.get("camera", {}) or {}
    sensor_id = as_int(camera_cfg.get("sensor_id"), 0)
    cam = GstCamera(
        sensor_id=sensor_id,
        capture_width=as_int(root.get("capture_width"), 1280),
        capture_height=as_int(root.get("capture_height"), 720),
        output_width=as_int(root.get("output_width"), 640),
        output_height=as_int(root.get("output_height"), 360),
        fps=as_int(root.get("fps"), 30),
        sink_format=str(root.get("appsink_format", "RGBA")),
    )
    first = cam.read(timeout_ms=1200)
    if first is None:
        cam.close()
        raise RuntimeError(f"{camera_key}/sensor{sensor_id}: first frame is None")
    print(f"[camera] {camera_key}/sensor{sensor_id} OK shape={first.shape}")
    return cam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture dual-camera crop images for BEV homography calibration.")
    parser.add_argument("--config", type=Path, default=HERE / "dotted_lane_following_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=HERE / "dual_bev_calib_capture")
    parser.add_argument("--warmup-frames", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    out_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except Exception as exc:
        raise SystemExit(f"OpenCV is required to save calibration images: {exc}") from exc

    GstCamera, fix_edge_color_cast = open_gst_tools()
    cameras = {}
    try:
        for key, cfg in enabled_camera_configs(config).items():
            try:
                cameras[key] = (cfg, open_camera(config, key, cfg, GstCamera))
                open_delay = as_float((config.get("camera", {}) or {}).get("open_delay_s"), 0.0)
                if open_delay > 0.0:
                    time.sleep(open_delay)
            except Exception as exc:
                print(f"[camera] {key} skipped: {exc}")
        if not cameras:
            raise SystemExit("No camera is available.")

        for _ in range(max(0, args.warmup_frames)):
            for _key, (_cfg, cam) in cameras.items():
                cam.read(timeout_ms=90)
            time.sleep(0.01)

        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(config_path),
            "cameras": {},
        }
        for key, (cfg, cam) in cameras.items():
            frame_rgb = cam.read(timeout_ms=1200)
            if frame_rgb is None:
                print(f"[capture] {key}: failed")
                continue
            frame_rgb = apply_color_fix(frame_rgb, config, fix_edge_color_cast)
            frame_bgr = rgb_to_bgr(frame_rgb)
            crop_bgr, crop_xyxy = crop_frame(frame_bgr, config, cfg)
            full_path = out_dir / f"{key}_full.jpg"
            crop_path = out_dir / f"{key}_crop.jpg"
            cv2.imwrite(str(full_path), frame_bgr)
            cv2.imwrite(str(crop_path), crop_bgr)
            meta["cameras"][key] = {
                "sensor_id": as_int(cfg.get("sensor_id"), 0),
                "full_image": full_path.name,
                "crop_image": crop_path.name,
                "full_size": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
                "crop_size": [int(crop_bgr.shape[1]), int(crop_bgr.shape[0])],
                "crop_xyxy": list(crop_xyxy),
            }
            print(f"[capture] {key}: {crop_path}")

        (out_dir / "capture_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[done] {out_dir}")
        return 0
    finally:
        for _cfg, cam in cameras.values():
            try:
                cam.close()
            except Exception:
                pass
        delay_s = as_float((config.get("camera", {}) or {}).get("argus_release_delay_s"), 0.0)
        if delay_s > 0.0:
            time.sleep(delay_s)


if __name__ == "__main__":
    raise SystemExit(main())
