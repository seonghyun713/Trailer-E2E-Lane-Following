#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trailer_parking_core import (  # noqa: E402
    AngleEstimate,
    AngleStateFilter,
    CenterTableAngleEstimator,
    PanelDetection,
    as_bool,
    as_float,
    as_int,
    draw_panel_overlay,
    draw_status_overlay,
    load_yaml,
    resolve_path,
)
from live_trailer_parking import (  # noqa: E402
    apply_color_fix,
    best_panel_detection,
    best_panel_detection_in_frame,
    camera_roi,
    enabled_camera_keys,
    enforce_single_panel_detection,
    open_gst_tools,
    rgb_to_bgr,
)


_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight HTTP trailer-angle viewer.")
    parser.add_argument("--config", type=Path, default=HERE / "dotted_lane_following_config.yaml")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8081)
    parser.add_argument("--http-width", type=int, default=960)
    parser.add_argument("--http-height", type=int, default=360)
    parser.add_argument("--http-fps", type=float, default=6.0)
    parser.add_argument("--jpeg-quality", type=int, default=65)
    parser.add_argument("--single-camera", choices=("left", "right", "cam0", "cam1"), default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


class HttpMjpegStreamer:
    def __init__(self, host: str, port: int, width: int, height: int, fps: float, jpeg_quality: int):
        self.host = host
        self.port = int(port)
        self.width = max(320, int(width))
        self.height = max(180, int(height))
        self.quality = max(30, min(95, int(jpeg_quality)))
        self.frame_period_s = 1.0 / max(1.0, float(fps))
        self.next_frame_at = 0.0
        self.condition = threading.Condition()
        self.latest_jpeg: Optional[bytes] = None
        self.frame_id = 0
        self.running = False
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.hostname = socket.gethostname().split(".")[0] or "rover"

    def open(self) -> None:
        handler = self._make_handler()
        self.server = ThreadingHTTPServer((self.host, self.port), handler)
        self.running = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="angle-http-stream", daemon=True)
        self.thread.start()
        print(f"[http] open http://{self.hostname}.local:{self.port}/")
        for ip in self._local_ips():
            print(f"[http] open http://{ip}:{self.port}/")

    def _local_ips(self) -> List[str]:
        ips = set()
        try:
            output = subprocess.check_output(["hostname", "-I"], text=True, timeout=0.5)
            for item in output.split():
                if item and not item.startswith("127.") and not item.startswith("172.17."):
                    ips.add(item)
        except Exception:
            pass
        return sorted(ips)

    def _make_handler(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    self._send_index()
                elif path in {"/stream.mjpg", "/stream"}:
                    self._send_stream()
                else:
                    self.send_error(404)

            def _send_index(self) -> None:
                body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trailer Angle View</title>
  <style>
    body {{ margin: 0; background: #101112; color: #eee; font-family: system-ui, sans-serif; }}
    header {{ padding: 9px 13px; background: #1b1d20; border-bottom: 1px solid #33383d; font-size: 15px; }}
    img {{ display: block; width: min(100vw, {streamer.width}px); max-height: calc(100vh - 42px); height: auto; margin: 0 auto; background: #050505; }}
  </style>
</head>
<body>
  <header>Trailer angle view</header>
  <img src="/stream.mjpg" alt="trailer angle stream">
</body>
</html>
"""
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                last_id = -1
                while streamer.running:
                    with streamer.condition:
                        streamer.condition.wait_for(
                            lambda: (not streamer.running) or streamer.frame_id != last_id,
                            timeout=1.0,
                        )
                        if not streamer.running:
                            break
                        jpeg = streamer.latest_jpeg
                        last_id = streamer.frame_id
                    if not jpeg:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

        return Handler

    def write(self, frame_bgr: np.ndarray) -> None:
        if not self.running:
            return
        now = time.monotonic()
        if now < self.next_frame_at:
            return
        self.next_frame_at = now + self.frame_period_s
        try:
            import cv2

            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                return
            with self.condition:
                self.latest_jpeg = encoded.tobytes()
                self.frame_id += 1
                self.condition.notify_all()
        except Exception as exc:
            print(f"[http] write failed: {exc}")

    def close(self) -> None:
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None


def resize_exact(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


def fit_into(frame_bgr: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, float, int, int]:
    try:
        import cv2
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8), 1.0, 0, 0
    h, w = frame_bgr.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    off_x = (width - new_w) // 2
    off_y = (height - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return out, scale, off_x, off_y


def roi_tile(
    frame_bgr: Optional[np.ndarray],
    config: Dict[str, Any],
    camera_key: str,
    detection: Optional[PanelDetection],
    width: int,
    height: int,
) -> np.ndarray:
    if frame_bgr is None:
        out = np.zeros((height, width, 3), dtype=np.uint8)
        draw_status_overlay(out, [f"{camera_key} unavailable"], origin=(12, 24))
        return out
    try:
        import cv2
    except Exception:
        cv2 = None
    x0, y0, x1, y1 = camera_roi(config, camera_key).to_abs(frame_bgr.shape)
    roi = frame_bgr[y0:y1, x0:x1].copy()
    if roi.size == 0:
        out = np.zeros((height, width, 3), dtype=np.uint8)
        draw_status_overlay(out, [f"{camera_key} empty ROI"], origin=(12, 24))
        return out
    out, scale, off_x, off_y = fit_into(roi, width, height)
    if cv2 is not None and detection is not None and detection.ok:
        p0 = (off_x + int(round(detection.x0 * scale)), off_y + int(round(detection.y0 * scale)))
        p1 = (off_x + int(round(detection.x1 * scale)), off_y + int(round(detection.y1 * scale)))
        cv2.rectangle(out, p0, p1, (0, 255, 255), 2)
        cv2.circle(out, (off_x + int(round(detection.center_x * scale)), off_y + int(round(detection.center_y * scale))), 5, (0, 255, 255), -1)
    det = "no panel" if detection is None or not detection.ok else f"panel conf={detection.confidence:.2f}"
    draw_status_overlay(out, [f"{camera_key} ROI  {det}"], origin=(12, 24))
    return out


def make_view(
    config: Dict[str, Any],
    frames_bgr: Dict[str, np.ndarray],
    detections: Dict[str, PanelDetection],
    fused_angle,
    fps: float,
) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None
    out_w, out_h = 960, 360
    half_w = out_w // 2
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for idx, key in enumerate(("cam1", "cam0")):
        x = idx * half_w
        frame = frames_bgr.get(key)
        canvas[:, x:x + half_w] = roi_tile(frame, config, key, detections.get(key), half_w, out_h)
    angle_text = "--" if fused_angle.angle_deg is None else f"{fused_angle.angle_deg:+.1f} deg"
    status = [
        f"angle={angle_text}  conf={fused_angle.confidence:.2f}  src={fused_angle.source}  fps={fps:.1f}",
    ]
    draw_status_overlay(canvas, status, origin=(12, 24))
    if cv2 is not None:
        cv2.line(canvas, (half_w, 0), (half_w, out_h), (80, 80, 80), 1)
    return canvas


def load_panel_model(config: Dict[str, Any], config_dir: Path):
    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    weights = resolve_path(config_dir, str(panel_cfg.get("weights", "")))
    if not weights.exists():
        raise SystemExit(f"Panel YOLO weights not found: {weights}")
    from ultralytics import YOLO

    print(f"[panel] weights={weights}")
    if weights.suffix.lower() == ".engine":
        return YOLO(str(weights), task="detect")
    return YOLO(str(weights))


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    streamer = HttpMjpegStreamer(
        args.http_host,
        args.http_port,
        args.http_width,
        args.http_height,
        args.http_fps,
        args.jpeg_quality,
    )
    streamer.open()

    panel_model = load_panel_model(config, config_dir)
    angle_estimator = CenterTableAngleEstimator(config, config_dir)
    angle_filter = AngleStateFilter(config)
    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    infer_every = max(1, as_int(panel_cfg.get("infer_every"), 1))

    GstCamera, fix_edge_color_cast = open_gst_tools()
    keys = enabled_camera_keys(config, args.single_camera)
    cameras: Dict[str, Any] = {}

    def open_one(key: str) -> Optional[Any]:
        root = config.get("camera", {}) or {}
        camera_cfg = (root.get("cameras") or {}).get(key, {}) or {}
        sensor_id = as_int(camera_cfg.get("sensor_id"), 0)
        try:
            cam = GstCamera(
                sensor_id=sensor_id,
                capture_width=as_int(root.get("capture_width"), 1280),
                capture_height=as_int(root.get("capture_height"), 720),
                output_width=as_int(root.get("output_width"), 1280),
                output_height=as_int(root.get("output_height"), 720),
                fps=as_int(root.get("fps"), 30),
                sink_format=str(root.get("appsink_format", "RGBA")),
                state_timeout_s=as_float(root.get("state_timeout_s"), 0.8),
            )
            first = cam.read(timeout_ms=as_int(root.get("first_frame_timeout_ms"), 650))
            if first is None:
                raise RuntimeError("first frame is None")
            print(f"[camera] {key}/sensor{sensor_id} OK shape={first.shape}")
            return cam
        except Exception as exc:
            print(f"[camera] {key}/sensor{sensor_id} failed: {exc}")
            return None

    root = config.get("camera", {}) or {}
    if as_bool(root.get("parallel_open"), len(keys) > 1) and len(keys) > 1:
        lock = threading.Lock()

        def worker(camera_key: str) -> None:
            cam = open_one(camera_key)
            if cam is not None:
                with lock:
                    cameras[camera_key] = cam

        threads = [threading.Thread(target=worker, args=(key,), name=f"open-{key}", daemon=True) for key in keys]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        missing = [key for key in keys if key not in cameras]
        if missing:
            print("[camera] retrying missing camera(s) sequentially")
            for key in missing:
                cam = open_one(key)
                if cam is not None:
                    cameras[key] = cam
        cameras = {key: cameras[key] for key in keys if key in cameras}
    else:
        for key in keys:
            cam = open_one(key)
            if cam is not None:
                cameras[key] = cam
    if not cameras:
        raise SystemExit("No camera is available.")

    detections: Dict[str, PanelDetection] = {}
    estimates: Dict[str, AngleEstimate] = {}
    frame_idx = 0
    prev_t = time.monotonic()
    fps = 0.0
    timeout_ms = as_int((config.get("camera", {}) or {}).get("timeout_ms"), 90)

    try:
        while not _STOP:
            frame_idx += 1
            now = time.monotonic()

            frames_rgb: Dict[str, np.ndarray] = {}
            frames_bgr: Dict[str, np.ndarray] = {}
            for key, cam in cameras.items():
                frame = cam.read(timeout_ms=timeout_ms)
                if frame is None:
                    continue
                frame = apply_color_fix(frame, config, fix_edge_color_cast)
                frames_rgb[key] = frame
                frames_bgr[key] = rgb_to_bgr(frame)

            measurements: List[AngleEstimate] = []
            if frame_idx % infer_every == 0:
                for key, frame_rgb in frames_rgb.items():
                    det = best_panel_detection_in_frame(panel_model, frame_rgb, key, config, panel_cfg, now)
                    detections[key] = det
                    estimates[key] = angle_estimator.estimate(det)
                measurements = enforce_single_panel_detection(detections, estimates, config, now)
            fused = angle_filter.update(measurements, now=now)

            view = make_view(config, frames_bgr, detections, fused, fps)
            streamer.write(view)
            done_t = time.monotonic()
            inst_fps = 1.0 / max(1e-6, done_t - prev_t)
            fps = inst_fps if fps <= 0.0 else 0.9 * fps + 0.1 * inst_fps
            prev_t = done_t

            if args.print_every > 0 and frame_idx % args.print_every == 0:
                angle = "--" if fused.angle_deg is None else f"{fused.angle_deg:+.1f}"
                det_text = " ".join(
                    f"{key}:{'ok' if det.ok else 'lost'}:{det.confidence:.2f}"
                    for key, det in sorted(detections.items())
                )
                print(f"[{frame_idx:06d}] angle={angle} conf={fused.confidence:.2f} src={fused.source} fps={fps:.1f} {det_text}")
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        for cam in cameras.values():
            try:
                cam.close()
            except Exception:
                pass
        streamer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
