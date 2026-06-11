#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jetcam.csi_camera import CSICamera  # noqa: E402
from trailer_angle_estimator import AngleFilter, TrailerAngleEstimator, draw_fused, draw_result, fuse_results  # noqa: E402


LEFT_CAMERA_ID = 1
RIGHT_CAMERA_ID = 0
CAPTURE_WIDTH = 960
CAPTURE_HEIGHT = 540
CAPTURE_FPS = 30
FRAME_COLOR = "bgr"
PROCESS_SCALE = 1.0
ESTIMATE_EVERY = 8
INACTIVE_ESTIMATE_EVERY = 24

# Match camera_live_dual.py: reduce red/magenta color cast near the lens edges.
EDGE_COLOR_FIX = True
EDGE_COLOR_FIX_WIDTH_RATIO = 0.32
EDGE_COLOR_FIX_STRENGTH = 0.75
EDGE_COLOR_FIX_GREEN_RECOVERY = 0.70

MAX_CONSECUTIVE_FAILS = 5
_EDGE_WEIGHT_CACHE = {}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual CSI live trailer angle estimator.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--left-camera", type=int, default=LEFT_CAMERA_ID)
    parser.add_argument("--right-camera", type=int, default=RIGHT_CAMERA_ID)
    parser.add_argument("--left-key", default="cam1")
    parser.add_argument("--right-key", default="cam0")
    parser.add_argument("--capture-width", type=positive_int, default=CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=positive_int, default=CAPTURE_HEIGHT)
    parser.add_argument("--fps", type=positive_int, default=CAPTURE_FPS)
    parser.add_argument("--frame-color", choices=("bgr", "rgb"), default=FRAME_COLOR)
    parser.add_argument("--process-scale", type=float, default=PROCESS_SCALE, help="Resize frames by this scale before marker detection.")
    parser.add_argument("--estimate-every", type=positive_int, default=ESTIMATE_EVERY, help="Run marker detection every N display frames.")
    parser.add_argument(
        "--inactive-estimate-every",
        type=positive_int,
        default=INACTIVE_ESTIMATE_EVERY,
        help="When one camera has the marker, check the other camera every N frames.",
    )
    parser.add_argument("--no-edge-color-fix", action="store_true", help="Disable red/magenta edge color correction.")
    parser.add_argument("--edge-color-fix-width-ratio", type=float, default=EDGE_COLOR_FIX_WIDTH_RATIO)
    parser.add_argument("--edge-color-fix-strength", type=float, default=EDGE_COLOR_FIX_STRENGTH)
    parser.add_argument("--edge-color-fix-green-recovery", type=float, default=EDGE_COLOR_FIX_GREEN_RECOVERY)
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--save-prefix", default="trailer_angle_snapshot")
    return parser.parse_args()


def try_open(device_id: int, args: argparse.Namespace):
    try:
        cam = CSICamera(
            capture_device=device_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            downsample=1,
            capture_fps=args.fps,
        )
        frame = cam.read()
        if frame is None:
            raise RuntimeError("first frame is None")
        print(f"[cam{device_id}] OK shape={frame.shape}")
        return cam
    except Exception as exc:
        print(f"[cam{device_id}] open/read failed: {exc}")
        return None


def fix_edge_color_cast(
    frame: np.ndarray,
    enabled: bool = EDGE_COLOR_FIX,
    width_ratio: float = EDGE_COLOR_FIX_WIDTH_RATIO,
    strength: float = EDGE_COLOR_FIX_STRENGTH,
    green_recovery: float = EDGE_COLOR_FIX_GREEN_RECOVERY,
) -> np.ndarray:
    """Reduce red/magenta shading at the left/right edges without flattening the center."""
    if not enabled:
        return frame

    _, width = frame.shape[:2]
    cache_key = (width, round(float(width_ratio), 4))
    cached = _EDGE_WEIGHT_CACHE.get(cache_key)
    if cached is None:
        edge_width = max(1, int(width * max(0.0, min(0.5, width_ratio))))
        x = np.arange(edge_width, dtype=np.float32)
        left_weight = ((edge_width - x) / edge_width) ** 2
        right_weight = left_weight[::-1].copy()
        cached = (edge_width, left_weight[np.newaxis, :], right_weight[np.newaxis, :])
        _EDGE_WEIGHT_CACHE[cache_key] = cached
    edge_width, left_weight, right_weight = cached

    out = frame.copy()

    def fix_region(x_slice: slice, weight: np.ndarray) -> None:
        work = out[:, x_slice].astype(np.float32)
        blue = work[:, :, 0]
        green = work[:, :, 1]
        red = work[:, :, 2]
        luma = 0.114 * blue + 0.587 * green + 0.299 * red
        magenta_excess = np.maximum(np.minimum(red, blue) - green, 0.0)
        red_excess = np.maximum(red - luma, 0.0)
        blue_excess = np.maximum(blue - luma, 0.0)
        green_deficit = np.maximum(luma - green, 0.0)

        work[:, :, 2] = red - (red_excess + 0.60 * magenta_excess) * strength * weight
        work[:, :, 0] = blue - (0.45 * blue_excess + 0.55 * magenta_excess) * strength * weight
        work[:, :, 1] = green + (green_deficit + 0.35 * magenta_excess) * green_recovery * weight
        out[:, x_slice] = np.clip(work, 0, 255).astype(np.uint8)

    fix_region(slice(0, edge_width), left_weight)
    fix_region(slice(width - edge_width, width), right_weight)
    return out


def resize_for_processing(frame: np.ndarray, scale: float) -> np.ndarray:
    scale = max(0.10, min(1.0, float(scale)))
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def scale_result_for_display(result, src_shape, dst_shape):
    if result is None:
        return None
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    scale_x = dst_w / max(1.0, float(src_w))
    scale_y = dst_h / max(1.0, float(src_h))
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return result

    scaled = copy.deepcopy(result)
    x0, y0, x1, y1 = scaled.roi_xyxy
    scaled.roi_xyxy = (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )
    if scaled.corners is not None:
        scaled.corners = scaled.corners.copy()
        scaled.corners[:, 0] *= scale_x
        scaled.corners[:, 1] *= scale_y
    return scaled


def safe_read(cam, device_id: int, fails: int, args: argparse.Namespace):
    if cam is None:
        return None, fails
    try:
        frame = cam.read()
        if frame is None:
            return None, fails + 1
        if args.frame_color == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = fix_edge_color_cast(
            frame,
            enabled=not args.no_edge_color_fix,
            width_ratio=args.edge_color_fix_width_ratio,
            strength=args.edge_color_fix_strength,
            green_recovery=args.edge_color_fix_green_recovery,
        )
        return frame, 0
    except Exception as exc:
        fails += 1
        if fails == 1 or fails % 30 == 0:
            print(f"[cam{device_id}] read failed #{fails}: {exc}")
        return None, fails


def placeholder(name: str, shape=(720, 1280, 3)) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    cv2.putText(img, name, (32, shape[0] // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.putText(img, "UNAVAILABLE", (32, shape[0] // 2 + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    return img


def main() -> int:
    args = parse_args()
    if args.left_camera == args.right_camera:
        raise SystemExit("--left-camera and --right-camera must be different.")

    estimator = TrailerAngleEstimator(args.config)
    filter_cfg = estimator.config.get("filter", {})
    angle_filter = AngleFilter(
        alpha=float(filter_cfg.get("alpha", 0.35)),
        min_confidence=float(filter_cfg.get("min_confidence", 0.18)),
        max_jump_deg=float(filter_cfg.get("max_jump_deg", 28.0)),
    )

    cam_left = try_open(args.left_camera, args)
    cam_right = try_open(args.right_camera, args)
    if cam_left is None and cam_right is None:
        raise SystemExit("Both cameras unavailable.")

    dead_left = cam_left is None
    dead_right = cam_right is None
    fails_left = 0
    fails_right = 0
    last_left = None
    last_right = None
    last_estimate_results = {}
    last_draw_results = {}
    active_key = None
    snapshot_idx = 0
    frame_idx = 0
    prev_t = time.time()
    fps = 0.0
    window = "Trailer angle dual view (q: quit, s: save)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not dead_left:
                frame, fails_left = safe_read(cam_left, args.left_camera, fails_left, args)
                if frame is not None:
                    last_left = frame
                if fails_left >= MAX_CONSECUTIVE_FAILS:
                    dead_left = True
                    try:
                        cam_left.cap.release()
                    except Exception:
                        pass

            if not dead_right:
                frame, fails_right = safe_read(cam_right, args.right_camera, fails_right, args)
                if frame is not None:
                    last_right = frame
                if fails_right >= MAX_CONSECUTIVE_FAILS:
                    dead_right = True
                    try:
                        cam_right.cap.release()
                    except Exception:
                        pass

            if dead_left and dead_right:
                print("Both cameras became unavailable.")
                break

            base_shape = last_left.shape if last_left is not None else (args.capture_height, args.capture_width, 3)
            left = placeholder(f"cam{args.left_camera}", base_shape) if dead_left or last_left is None else last_left.copy()
            right = placeholder(f"cam{args.right_camera}", base_shape) if dead_right or last_right is None else last_right.copy()

            results = []
            if not dead_left and last_left is not None:
                left_period = args.estimate_every if active_key in (None, args.left_key) else args.inactive_estimate_every
                estimate_left = (
                    args.left_key not in last_estimate_results
                    or frame_idx % left_period == 0
                )
                if estimate_left:
                    proc_left = resize_for_processing(last_left, args.process_scale)
                    res_left = estimator.estimate(proc_left, args.left_key)
                    last_estimate_results[args.left_key] = res_left
                    last_draw_results[args.left_key] = scale_result_for_display(res_left, proc_left.shape, left.shape)
                draw_left = last_draw_results.get(args.left_key)
                if draw_left is not None:
                    draw_result(left, draw_left)
                results.append(last_estimate_results[args.left_key])
            if not dead_right and last_right is not None:
                right_period = args.estimate_every if active_key in (None, args.right_key) else args.inactive_estimate_every
                right_offset = max(1, right_period // 2)
                estimate_right = (
                    args.right_key not in last_estimate_results
                    or (frame_idx + right_offset) % right_period == 0
                )
                if estimate_right:
                    proc_right = resize_for_processing(last_right, args.process_scale)
                    res_right = estimator.estimate(proc_right, args.right_key)
                    last_estimate_results[args.right_key] = res_right
                    last_draw_results[args.right_key] = scale_result_for_display(res_right, proc_right.shape, right.shape)
                draw_right = last_draw_results.get(args.right_key)
                if draw_right is not None:
                    draw_result(right, draw_right)
                results.append(last_estimate_results[args.right_key])

            fused = fuse_results(results, min_confidence=angle_filter.min_confidence)
            if fused.ok and fused.source in (args.left_key, args.right_key):
                active_key = fused.source
            filtered = angle_filter.update(fused.angle_deg, fused.confidence)
            if filtered is not None:
                fused.angle_deg = filtered
            draw_fused(left, fused)
            draw_fused(right, fused)

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - prev_t)
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps
            prev_t = now
            cv2.putText(left, f"cam{args.left_camera} {fps:4.1f} FPS", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
            cv2.putText(
                right,
                f"cam{args.right_camera}  proc={args.process_scale:.2f} every={args.estimate_every}",
                (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
            )

            if left.shape[:2] != right.shape[:2]:
                right = cv2.resize(right, (left.shape[1], left.shape[0]))
            combined = cv2.hconcat([left, right])
            if args.display_scale != 1.0:
                combined = cv2.resize(combined, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(window, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = f"{args.save_prefix}_{snapshot_idx:03d}.jpg"
                cv2.imwrite(out, combined)
                print(f"saved: {out}")
                snapshot_idx += 1
            frame_idx += 1
    finally:
        for cam in (cam_left, cam_right):
            if cam is not None:
                try:
                    cam.cap.release()
                except Exception:
                    pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
