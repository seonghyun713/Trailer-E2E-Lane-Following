#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from capture_bbox_frames_gst import (
    Gst,
    GstCamera,
    _read_yaml,
    fix_edge_color_cast,
    fix_global_red_cast,
    positive_int,
    roi_abs,
    save_image,
)


STOP = False


def handle_signal(_signum, _frame) -> None:
    global STOP
    STOP = True


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Keep both CSI cameras open and capture neg/pos small-angle ROI phases on Enter."
    )
    parser.add_argument("--config", default=str(here / "dotted_lane_following_config.yaml"))
    parser.add_argument("--output", default=str(here / f"bbox_dataset_small_angle_phases_{stamp}"))
    parser.add_argument("--left-camera", type=int, default=1)
    parser.add_argument("--right-camera", type=int, default=0)
    parser.add_argument("--left-key", default="cam1")
    parser.add_argument("--right-key", default="cam0")
    parser.add_argument("--capture-width", type=positive_int, default=1280)
    parser.add_argument("--capture-height", type=positive_int, default=720)
    parser.add_argument("--save-width", type=positive_int, default=640)
    parser.add_argument("--save-height", type=positive_int, default=360)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--appsink-format", choices=("RGBA", "BGRx", "RGBx"), default="RGBA")
    parser.add_argument("--state-timeout-s", type=float, default=0.8)
    parser.add_argument("--first-frame-timeout-ms", type=int, default=650)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--frame-step", type=positive_int, default=10)
    parser.add_argument("--phases", default="neg,pos", help="Comma-separated phase names. Default: neg,pos")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=92)
    parser.add_argument("--edge-color-fix", dest="edge_color_fix", action="store_true", default=True)
    parser.add_argument("--no-edge-color-fix", dest="edge_color_fix", action="store_false")
    parser.add_argument("--edge-color-fix-width-ratio", type=float, default=0.32)
    parser.add_argument("--edge-color-fix-strength", type=float, default=0.75)
    parser.add_argument("--edge-color-fix-green-recovery", type=float, default=0.70)
    return parser.parse_args()


def append_metadata(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = not csv_path.exists()
    fieldnames = [
        "image_path",
        "camera_key",
        "sensor_id",
        "phase",
        "frame_idx",
        "phase_frame_idx",
        "timestamp",
        "monotonic_s",
        "save_mode",
        "image_w",
        "image_h",
        "roi_x0",
        "roi_y0",
        "roi_x1",
        "roi_y1",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if first:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def save_roi_frame(
    *,
    output: Path,
    metadata_csv: Path,
    config: Dict[str, Any],
    args: argparse.Namespace,
    phase: str,
    frame_rgb: Any,
    camera_key: str,
    sensor_id: int,
    frame_idx: int,
    phase_frame_idx: int,
    start_time: float,
) -> None:
    roi = roi_abs(config, camera_key, frame_rgb.shape)
    fixed = fix_edge_color_cast(
        frame_rgb,
        enabled=args.edge_color_fix,
        width_ratio=args.edge_color_fix_width_ratio,
        strength=args.edge_color_fix_strength,
        green_recovery=args.edge_color_fix_green_recovery,
    )
    x0, y0, x1, y1 = roi
    image = fix_global_red_cast(fixed[y0:y1, x0:x1], enabled=args.edge_color_fix)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = "jpg" if args.format == "jpg" else "png"
    rel = Path("images") / phase / camera_key / "roi" / f"{stamp}_{phase}_{camera_key}_f{frame_idx:07d}.{ext}"
    save_image(output / rel, image, args.format, args.jpg_quality)
    append_metadata(
        metadata_csv,
        {
            "image_path": str(rel),
            "camera_key": camera_key,
            "sensor_id": sensor_id,
            "phase": phase,
            "frame_idx": frame_idx,
            "phase_frame_idx": phase_frame_idx,
            "timestamp": stamp,
            "monotonic_s": f"{time.monotonic() - start_time:.3f}",
            "save_mode": "roi",
            "image_w": image.shape[1],
            "image_h": image.shape[0],
            "roi_x0": x0,
            "roi_y0": y0,
            "roi_x1": x1,
            "roi_y1": y1,
        },
    )


def warmup(cameras: Sequence[Tuple[Optional[GstCamera], str, int]], frames: int) -> None:
    for _ in range(max(0, frames)):
        for cam, _key, _sensor_id in cameras:
            if cam is not None:
                cam.read(timeout_ms=90)


def capture_phase(
    *,
    phase: str,
    cameras: Sequence[Tuple[Optional[GstCamera], str, int]],
    output: Path,
    metadata_csv: Path,
    config: Dict[str, Any],
    args: argparse.Namespace,
    global_frame_idx: int,
    start_time: float,
) -> Tuple[int, Dict[str, int]]:
    counts = {key: 0 for _cam, key, _sensor_id in cameras}
    phase_frame_idx = 0
    deadline = time.monotonic() + max(0.1, float(args.duration_s))
    print(f"[{phase}] capture {args.duration_s:.1f}s, frame_step={args.frame_step}")
    while not STOP and time.monotonic() < deadline:
        global_frame_idx += 1
        phase_frame_idx += 1
        active = False
        for cam, key, sensor_id in cameras:
            if cam is None:
                continue
            frame = cam.read(timeout_ms=90)
            if frame is None:
                continue
            active = True
            if phase_frame_idx % args.frame_step != 0:
                continue
            save_roi_frame(
                output=output,
                metadata_csv=metadata_csv,
                config=config,
                args=args,
                phase=phase,
                frame_rgb=frame,
                camera_key=key,
                sensor_id=sensor_id,
                frame_idx=global_frame_idx,
                phase_frame_idx=phase_frame_idx,
                start_time=start_time,
            )
            counts[key] = counts.get(key, 0) + 1
        if not active:
            time.sleep(0.01)
        if phase_frame_idx % max(1, args.frame_step * 10) == 0:
            print(f"[{phase}] saved cam1={counts.get('cam1', 0)} cam0={counts.get('cam0', 0)}")
    print(f"[{phase}] done {counts}")
    return global_frame_idx, counts


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    Gst.init(None)

    args = parse_args()
    config = _read_yaml(Path(args.config).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_csv = output / "captures.csv"
    phases = [p.strip() for p in str(args.phases).split(",") if p.strip()]
    if not phases:
        phases = ["neg", "pos"]

    left = open_camera("left", args.left_camera, args)
    right = open_camera("right", args.right_camera, args)
    cameras = ((left, args.left_key, args.left_camera), (right, args.right_key, args.right_camera))
    if left is None and right is None:
        print("No camera is available.")
        return 2

    print(f"Saving to: {output}")
    print("Cameras stay open. Set trailer phase, then press Enter.")
    print("Label later with label_bbox_tool.py. Use x/No Marker for the hidden side.")
    start_time = time.monotonic()
    global_frame_idx = 0
    total_counts: Dict[str, Dict[str, int]] = {}

    try:
        warmup(cameras, args.warmup_frames)
        for phase in phases:
            if STOP:
                break
            input(f"\nSet trailer to {phase.upper()} side, then press Enter to capture {args.duration_s:.0f}s...")
            global_frame_idx, counts = capture_phase(
                phase=phase,
                cameras=cameras,
                output=output,
                metadata_csv=metadata_csv,
                config=config,
                args=args,
                global_frame_idx=global_frame_idx,
                start_time=start_time,
            )
            total_counts[phase] = counts
    finally:
        for cam, _key, _sensor_id in cameras:
            if cam is not None:
                cam.close()

    print(f"\nDone. saved {total_counts}")
    print(f"Metadata: {metadata_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
