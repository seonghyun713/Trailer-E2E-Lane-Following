#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import gi
import numpy as np
from PIL import Image

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402


LEFT_CAMERA_ID = 1
RIGHT_CAMERA_ID = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
SAVE_WIDTH = 640
SAVE_HEIGHT = 360
CAPTURE_FPS = 30
FRAME_STEP = 10
WARMUP_FRAMES = 20
APPSINK_FORMAT = "RGBA"

EDGE_COLOR_FIX_WIDTH_RATIO = 0.32
EDGE_COLOR_FIX_STRENGTH = 0.75
EDGE_COLOR_FIX_GREEN_RECOVERY = 0.70
EDGE_COLOR_FIX = True

_STOP = False
_EDGE_WEIGHT_CACHE: Dict[Tuple[int, float], Tuple[int, np.ndarray, np.ndarray]] = {}


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip3 install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="OpenCV-free CSI frame capture for blue-panel trailer bbox datasets.",
    )
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument("--output", default=str(here / "bbox_dataset"))
    parser.add_argument("--left-camera", type=int, default=LEFT_CAMERA_ID)
    parser.add_argument("--right-camera", type=int, default=RIGHT_CAMERA_ID)
    parser.add_argument("--left-key", default="cam1")
    parser.add_argument("--right-key", default="cam0")
    parser.add_argument("--capture-width", type=positive_int, default=CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=positive_int, default=CAPTURE_HEIGHT)
    parser.add_argument("--save-width", type=positive_int, default=SAVE_WIDTH)
    parser.add_argument("--save-height", type=positive_int, default=SAVE_HEIGHT)
    parser.add_argument("--fps", type=positive_int, default=CAPTURE_FPS)
    parser.add_argument("--appsink-format", choices=("RGBA", "BGRx", "RGBx"), default=APPSINK_FORMAT)
    parser.add_argument("--state-timeout-s", type=float, default=0.8)
    parser.add_argument("--first-frame-timeout-ms", type=int, default=650)
    parser.add_argument("--frame-step", type=positive_int, default=FRAME_STEP, help="Save one image every N grabbed frames.")
    parser.add_argument("--warmup-frames", type=int, default=WARMUP_FRAMES)
    parser.add_argument("--max-images-per-camera", type=int, default=0, help="0 means unlimited until Ctrl+C.")
    parser.add_argument("--save-mode", choices=("roi", "full", "both"), default="roi")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=88)
    parser.add_argument(
        "--edge-color-fix",
        dest="edge_color_fix",
        action="store_true",
        default=EDGE_COLOR_FIX,
        help="Enable red/magenta edge color correction. This is on by default.",
    )
    parser.add_argument(
        "--no-edge-color-fix",
        dest="edge_color_fix",
        action="store_false",
        help="Disable red/magenta edge color correction.",
    )
    parser.add_argument("--edge-color-fix-width-ratio", type=float, default=EDGE_COLOR_FIX_WIDTH_RATIO)
    parser.add_argument("--edge-color-fix-strength", type=float, default=EDGE_COLOR_FIX_STRENGTH)
    parser.add_argument("--edge-color-fix-green-recovery", type=float, default=EDGE_COLOR_FIX_GREEN_RECOVERY)
    parser.add_argument("--single-camera", choices=("left", "right"), default="")
    return parser.parse_args()


def roi_abs(config: Dict[str, Any], camera_key: str, shape: Sequence[int]) -> Tuple[int, int, int, int]:
    h, w = int(shape[0]), int(shape[1])
    cameras = config.get("cameras", {}) or {}
    if not cameras:
        cameras = ((config.get("camera", {}) or {}).get("cameras", {}) or {})
    roi = (cameras.get(camera_key, {}) or {}).get("roi", {}) or {}
    x = float(roi.get("x", 0.0))
    y = float(roi.get("y", 0.0))
    rw = float(roi.get("w", 1.0))
    rh = float(roi.get("h", 1.0))
    x0 = max(0, min(w - 1, int(round(x * w))))
    y0 = max(0, min(h - 1, int(round(y * h))))
    x1 = max(x0 + 1, min(w, int(round((x + rw) * w))))
    y1 = max(y0 + 1, min(h, int(round((y + rh) * h))))
    return x0, y0, x1, y1


def fix_edge_color_cast(
    frame_rgb: np.ndarray,
    enabled: bool,
    width_ratio: float,
    strength: float,
    green_recovery: float,
) -> np.ndarray:
    if not enabled:
        return frame_rgb
    _, width = frame_rgb.shape[:2]
    key = (width, round(float(width_ratio), 4))
    cached = _EDGE_WEIGHT_CACHE.get(key)
    if cached is None:
        edge_width = max(1, int(width * max(0.0, min(0.5, width_ratio))))
        x = np.arange(edge_width, dtype=np.float32)
        left = ((edge_width - x) / edge_width) ** 2
        right = left[::-1].copy()
        cached = (edge_width, left[np.newaxis, :], right[np.newaxis, :])
        _EDGE_WEIGHT_CACHE[key] = cached
    edge_width, left_weight, right_weight = cached

    out = frame_rgb.copy()

    def fix_region(x_slice: slice, weight: np.ndarray) -> None:
        work = out[:, x_slice].astype(np.float32)
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
        out[:, x_slice] = np.clip(work, 0, 255).astype(np.uint8)

    fix_region(slice(0, edge_width), left_weight)
    fix_region(slice(width - edge_width, width), right_weight)
    return fix_global_red_cast(out, enabled=enabled)


def fix_global_red_cast(frame_rgb: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Strong per-frame white balance for the red/pink camera tint."""
    if not enabled:
        return frame_rgb
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


@dataclass
class GstCamera:
    sensor_id: int
    capture_width: int
    capture_height: int
    output_width: int
    output_height: int
    fps: int
    sink_format: str
    state_timeout_s: float = 0.8

    def __post_init__(self) -> None:
        self.pipeline = Gst.parse_launch(self._pipeline_text())
        self.sink = self.pipeline.get_by_name("sink")
        if self.sink is None:
            raise RuntimeError("appsink was not created")
        self.pipeline.set_state(Gst.State.PLAYING)
        timeout_ns = int(max(0.1, float(self.state_timeout_s)) * Gst.SECOND)
        ret, _, _ = self.pipeline.get_state(timeout_ns)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"cam{self.sensor_id}: failed to start GStreamer pipeline")

    def _pipeline_text(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.capture_width}, height=(int){self.capture_height}, "
            f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
            "nvvidconv ! "
            f"video/x-raw, width=(int){self.output_width}, height=(int){self.output_height}, "
            f"format=(string){self.sink_format} ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

    def read(self, timeout_ms: int = 250) -> Optional[np.ndarray]:
        sample = self.sink.emit("try-pull-sample", int(timeout_ms * 1_000_000))
        if sample is None:
            return None
        caps = sample.get_caps()
        caps_struct = caps.get_structure(0)
        width = int(caps_struct.get_value("width"))
        height = int(caps_struct.get_value("height"))
        fmt = str(caps_struct.get_value("format"))
        video_info = GstVideo.VideoInfo.new_from_caps(caps)
        stride = int(video_info.stride[0]) if video_info is not None else width * 4
        offset = int(video_info.offset[0]) if video_info is not None else 0
        stride = max(stride, width * 4)
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            need = offset + stride * max(0, height - 1) + width * 4
            data = np.frombuffer(info.data, dtype=np.uint8)
            if data.size < need:
                return None
            frame4 = np.ndarray(
                shape=(height, width, 4),
                dtype=np.uint8,
                buffer=data,
                offset=offset,
                strides=(stride, 4, 1),
            )
            if fmt == "BGRx":
                return frame4[:, :, [2, 1, 0]].copy()
            return frame4[:, :, :3].copy()
        finally:
            buf.unmap(info)

    def close(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)


def save_image(path: Path, frame_rgb: np.ndarray, fmt: str, jpg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame_rgb, "RGB")
    if fmt == "jpg":
        img.save(path, quality=max(1, min(95, int(jpg_quality))), optimize=False)
    else:
        img.save(path)


def append_metadata(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "camera_key",
                "sensor_id",
                "frame_idx",
                "timestamp",
                "monotonic_s",
                "save_mode",
                "image_w",
                "image_h",
                "roi_x0",
                "roi_y0",
                "roi_x1",
                "roi_y1",
            ],
        )
        if first:
            writer.writeheader()
        writer.writerow(row)


def open_camera(name: str, sensor_id: int, args: argparse.Namespace) -> Optional[GstCamera]:
    try:
        cam = GstCamera(
            sensor_id=sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            output_width=args.save_width,
            output_height=args.save_height,
            fps=args.fps,
            sink_format=args.appsink_format,
            state_timeout_s=args.state_timeout_s,
        )
        first = cam.read(timeout_ms=args.first_frame_timeout_ms)
        if first is None:
            raise RuntimeError("first frame is None")
        print(f"[{name}/cam{sensor_id}] OK shape={first.shape}")
        return cam
    except Exception as exc:
        print(f"[{name}/cam{sensor_id}] open failed: {exc}")
        return None


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    Gst.init(None)
    args = parse_args()
    config = _read_yaml(Path(args.config).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    metadata_csv = output / "captures.csv"
    ext = "jpg" if args.format == "jpg" else "png"

    left_enabled = args.single_camera in ("", "left")
    right_enabled = args.single_camera in ("", "right")
    left = open_camera("left", args.left_camera, args) if left_enabled else None
    right = open_camera("right", args.right_camera, args) if right_enabled else None
    if left is None and right is None:
        print("No camera is available.")
        return 2

    counts = {args.left_key: 0, args.right_key: 0}
    frame_idx = 0
    start = time.monotonic()
    print(f"Saving to: {output}")
    print(f"Mode={args.save_mode}, every {args.frame_step} frames, Ctrl+C to stop.")
    print(f"Edge color fix={'ON' if args.edge_color_fix else 'OFF'}")

    def handle_frame(frame_rgb: np.ndarray, camera_key: str, sensor_id: int) -> None:
        roi = roi_abs(config, camera_key, frame_rgb.shape)
        frame_rgb2 = fix_edge_color_cast(
            frame_rgb,
            enabled=args.edge_color_fix,
            width_ratio=args.edge_color_fix_width_ratio,
            strength=args.edge_color_fix_strength,
            green_recovery=args.edge_color_fix_green_recovery,
        )

        targets = []
        if args.save_mode in ("full", "both"):
            targets.append(("full", frame_rgb2, (0, 0, frame_rgb2.shape[1], frame_rgb2.shape[0])))
        if args.save_mode in ("roi", "both"):
            x0, y0, x1, y1 = roi
            targets.append(("roi", frame_rgb2[y0:y1, x0:x1], roi))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for mode, image, save_roi in targets:
            image = fix_global_red_cast(image, enabled=args.edge_color_fix)
            rel = Path("images") / camera_key / mode / f"{stamp}_{camera_key}_f{frame_idx:07d}.{ext}"
            path = output / rel
            save_image(path, image, args.format, args.jpg_quality)
            append_metadata(
                metadata_csv,
                {
                    "image_path": str(rel),
                    "camera_key": camera_key,
                    "sensor_id": sensor_id,
                    "frame_idx": frame_idx,
                    "timestamp": stamp,
                    "monotonic_s": f"{time.monotonic() - start:.3f}",
                    "save_mode": mode,
                    "image_w": image.shape[1],
                    "image_h": image.shape[0],
                    "roi_x0": save_roi[0],
                    "roi_y0": save_roi[1],
                    "roi_x1": save_roi[2],
                    "roi_y1": save_roi[3],
                },
            )
        counts[camera_key] += 1

    try:
        while not _STOP:
            frame_idx += 1
            active = False
            for cam, key, sensor_id in ((left, args.left_key, args.left_camera), (right, args.right_key, args.right_camera)):
                if cam is None:
                    continue
                frame = cam.read(timeout_ms=90)
                if frame is None:
                    continue
                active = True
                if frame_idx <= max(0, args.warmup_frames):
                    continue
                if frame_idx % args.frame_step != 0:
                    continue
                if args.max_images_per_camera > 0 and counts[key] >= args.max_images_per_camera:
                    continue
                handle_frame(frame, key, sensor_id)

            if not active:
                time.sleep(0.01)
            if args.max_images_per_camera > 0:
                done_left = left is None or counts[args.left_key] >= args.max_images_per_camera
                done_right = right is None or counts[args.right_key] >= args.max_images_per_camera
                if done_left and done_right:
                    break
            if (counts[args.left_key] + counts[args.right_key]) and frame_idx % max(1, args.frame_step * 10) == 0:
                print(f"saved cam1={counts.get(args.left_key, 0)} cam0={counts.get(args.right_key, 0)}")
    finally:
        for cam in (left, right):
            if cam is not None:
                cam.close()

    print(f"Done. saved {counts}")
    print(f"Metadata: {metadata_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
