#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import gi
import numpy as np
from PIL import Image

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_bbox_frames_gst import (  # noqa: E402
    APPSINK_FORMAT,
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    EDGE_COLOR_FIX_GREEN_RECOVERY,
    EDGE_COLOR_FIX_STRENGTH,
    EDGE_COLOR_FIX_WIDTH_RATIO,
    GstCamera,
    fix_edge_color_cast,
    fix_global_red_cast,
    positive_int,
    roi_abs,
)


LEFT_CAMERA_ID = 1
RIGHT_CAMERA_ID = 0
SAVE_WIDTH = 1280
SAVE_HEIGHT = 720
DEFAULT_ANGLES = "-50,-45,-35,-25,-15,-8,0,8,15,25,35,45,50"
DEFAULT_SAMPLES_PER_CAMERA = 10
DEFAULT_FRAME_STEP = 1
DEFAULT_CONF = 0.18

_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip3 install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_angles(value: str) -> List[float]:
    angles: List[float] = []
    for chunk in value.replace(" ", "").split(","):
        if not chunk:
            continue
        angles.append(float(chunk))
    if not angles:
        raise argparse.ArgumentTypeError("at least one angle is required")
    return angles


def normalize_argv(argv: Sequence[str]) -> List[str]:
    """Let argparse accept '--angles -45,-35,...' despite the leading dash."""
    normalized: List[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--angles" and idx + 1 < len(argv):
            value = argv[idx + 1]
            if value.startswith("-") and "," in value:
                normalized.append(f"--angles={value}")
                idx += 2
                continue
        normalized.append(arg)
        idx += 1
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect true hinge-angle calibration samples with YOLO detections.",
    )
    parser.add_argument("--config", default=str(HERE / "dotted_lane_following_config.yaml"))
    parser.add_argument("--weights", default=str(HERE / "runs_yolo" / "trailer_panel_yolo11n" / "weights" / "best_640.engine"))
    parser.add_argument("--output", default=str(HERE / f"angle_calib_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--angles", default=DEFAULT_ANGLES, help="Comma-separated true hinge angles in degrees.")
    parser.add_argument("--samples-per-camera", type=positive_int, default=DEFAULT_SAMPLES_PER_CAMERA)
    parser.add_argument("--frame-step", type=positive_int, default=DEFAULT_FRAME_STEP)
    parser.add_argument("--warmup-frames", type=int, default=4)
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
    parser.add_argument("--parallel-open", dest="parallel_open", action="store_true", default=True)
    parser.add_argument("--no-parallel-open", dest="parallel_open", action="store_false")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--http-stream", dest="http_stream", action="store_true", default=True)
    parser.add_argument("--no-http-stream", dest="http_stream", action="store_false")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8081)
    parser.add_argument("--http-width", type=int, default=960)
    parser.add_argument("--http-height", type=int, default=720)
    parser.add_argument("--http-fps", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--interactive", action="store_true", default=True)
    parser.add_argument("--no-interactive", dest="interactive", action="store_false")
    parser.add_argument("--single-camera", choices=("left", "right"), default="")
    parser.add_argument(
        "--camera-by-sign",
        action="store_true",
        help="Capture negative angles with cam0/right only and positive angles with cam1/left only.",
    )
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        help="Skip 0 deg samples. Useful when zero angle is represented by no panel detection.",
    )
    parser.add_argument("--edge-color-fix-width-ratio", type=float, default=EDGE_COLOR_FIX_WIDTH_RATIO)
    parser.add_argument("--edge-color-fix-strength", type=float, default=EDGE_COLOR_FIX_STRENGTH)
    parser.add_argument("--edge-color-fix-green-recovery", type=float, default=EDGE_COLOR_FIX_GREEN_RECOVERY)
    return parser.parse_args(normalize_argv(sys.argv[1:]))


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


def save_image(path: Path, frame_rgb: np.ndarray, fmt: str, jpg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame_rgb, "RGB")
    if fmt == "jpg":
        img.save(path, quality=max(1, min(95, int(jpg_quality))), optimize=False)
    else:
        img.save(path)


def best_detection(model, image_source: Any, imgsz: int, conf: float, device: str) -> Dict[str, Any]:
    if isinstance(image_source, Path):
        source = str(image_source)
    else:
        source = image_source[:, :, ::-1].copy()
    result = model.predict(source=source, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return {"detected": False}
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    idx = int(np.argmax(confs))
    x0, y0, x1, y1 = [float(v) for v in xyxy[idx]]
    return {
        "detected": True,
        "conf": float(confs[idx]),
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
    }


def fit_into(frame_rgb: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, float, int, int]:
    try:
        import cv2
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8), 1.0, 0, 0
    out = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = frame_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return out, 1.0, 0, 0
    scale = min(width / float(w), height / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    off_x = (width - new_w) // 2
    off_y = (height - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return out, scale, off_x, off_y


def draw_text(frame_rgb: np.ndarray, lines: Sequence[str], origin: Tuple[int, int] = (12, 24)) -> None:
    try:
        import cv2
    except Exception:
        return
    x, y = origin
    for idx, line in enumerate(lines):
        yy = y + idx * 22
        cv2.putText(frame_rgb, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_rgb, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)


class CalibrationHttpPreview:
    def __init__(self, args: argparse.Namespace):
        self.enabled = bool(args.http_stream)
        self.host = str(args.http_host)
        self.port = int(args.http_port)
        self.width = max(320, int(args.http_width))
        self.height = max(240, int(args.http_height))
        self.quality = max(30, min(95, int(args.jpeg_quality)))
        self.frame_period_s = 1.0 / max(1.0, float(args.http_fps))
        self.next_frame_at = 0.0
        self.condition = threading.Condition()
        self.latest_jpeg: Optional[bytes] = None
        self.frame_id = 0
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.latest: Dict[str, Dict[str, Any]] = {}
        self.status: List[str] = []
        self.hostname = socket.gethostname().split(".")[0] or "rover"
        if self.enabled:
            self.open()

    def open(self) -> None:
        try:
            self.server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
            self.running = True
            self.thread = threading.Thread(target=self.server.serve_forever, name="angle-calib-http", daemon=True)
            self.thread.start()
            print(f"[http] angle calibration preview http://{self.hostname}.local:{self.port}/")
            for ip in self._local_ips():
                print(f"[http] angle calibration preview http://{ip}:{self.port}/")
        except Exception as exc:
            print(f"[http] preview disabled: {exc}")
            self.enabled = False
            self.running = False

    def _local_ips(self) -> List[str]:
        ips = set()
        try:
            output = socket.gethostbyname_ex(socket.gethostname())[2]
            ips.update(output)
        except Exception:
            pass
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                ips.add(sock.getsockname()[0])
        except Exception:
            pass
        return sorted(ip for ip in ips if ip and not ip.startswith("127."))

    def _make_handler(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Angle Calibration Preview</title>
<style>
body {{ margin:0; background:#101112; color:#f3f3f3; font-family:system-ui,sans-serif; }}
header {{ padding:10px 14px; background:#1b1d20; border-bottom:1px solid #33383d; }}
h1 {{ margin:0; font-size:16px; }}
img {{ display:block; width:min(100vw,{preview.width}px); height:auto; margin:0 auto; background:#050505; }}
</style></head>
<body><header><h1>Angle calibration preview</h1></header><img src="/stream.mjpg"></body></html>"""
                    data = body.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path in {"/stream.mjpg", "/stream"}:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    last_id = -1
                    while preview.running:
                        with preview.condition:
                            preview.condition.wait_for(lambda: not preview.running or preview.frame_id != last_id, timeout=1.0)
                            if not preview.running:
                                break
                            jpeg = preview.latest_jpeg
                            last_id = preview.frame_id
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
                    return
                self.send_error(404)

        return Handler

    def update_camera(
        self,
        camera_key: str,
        frame_rgb: np.ndarray,
        roi_xyxy: Tuple[int, int, int, int],
        roi_rgb: np.ndarray,
        det: Dict[str, Any],
        status: Sequence[str],
    ) -> None:
        if not self.enabled or not self.running:
            return
        self.latest[camera_key] = {
            "frame": frame_rgb.copy(),
            "roi": roi_rgb.copy(),
            "roi_xyxy": roi_xyxy,
            "det": dict(det),
        }
        self.status = list(status)
        now = time.monotonic()
        if now < self.next_frame_at:
            return
        self.next_frame_at = now + self.frame_period_s
        self._publish()

    def _publish(self) -> None:
        try:
            import cv2
        except Exception:
            return
        canvas = self._render()
        ok, encoded = cv2.imencode(".jpg", canvas[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self.condition:
            self.latest_jpeg = encoded.tobytes()
            self.frame_id += 1
            self.condition.notify_all()

    def _render(self) -> np.ndarray:
        try:
            import cv2
        except Exception:
            cv2 = None
        out_w, out_h = self.width, self.height
        half_w = out_w // 2
        half_h = out_h // 2
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        colors = {"cam1": (255, 220, 80), "cam0": (100, 255, 120)}
        for idx, key in enumerate(("cam1", "cam0")):
            x = idx * half_w
            width = out_w - x if idx == 1 else half_w
            item = self.latest.get(key)
            if item is None:
                tile = np.zeros((half_h, width, 3), dtype=np.uint8)
                draw_text(tile, [f"{key} waiting"], origin=(12, 24))
                canvas[0:half_h, x:x + width] = tile
                canvas[half_h:out_h, x:x + width] = tile.copy()
                continue
            full, scale, off_x, off_y = fit_into(item["frame"], width, half_h)
            rx0, ry0, rx1, ry1 = item["roi_xyxy"]
            color = colors.get(key, (255, 255, 0))
            if cv2 is not None:
                p0 = (off_x + int(round(rx0 * scale)), off_y + int(round(ry0 * scale)))
                p1 = (off_x + int(round(rx1 * scale)), off_y + int(round(ry1 * scale)))
                cv2.rectangle(full, p0, p1, color, 2)
            draw_text(full, [f"{key} full camera"], origin=(12, 24))
            canvas[0:half_h, x:x + width] = full

            roi_tile, roi_scale, roi_off_x, roi_off_y = fit_into(item["roi"], width, half_h)
            det = item["det"]
            if cv2 is not None and det.get("detected"):
                d0 = (roi_off_x + int(round(float(det["x0"]) * roi_scale)), roi_off_y + int(round(float(det["y0"]) * roi_scale)))
                d1 = (roi_off_x + int(round(float(det["x1"]) * roi_scale)), roi_off_y + int(round(float(det["y1"]) * roi_scale)))
                cv2.rectangle(roi_tile, d0, d1, (255, 255, 0), 2)
            det_text = "no panel" if not det.get("detected") else f"panel conf={float(det['conf']):.2f}"
            draw_text(roi_tile, [f"{key} YOLO ROI {item['roi'].shape[1]}x{item['roi'].shape[0]}", det_text], origin=(12, 24))
            canvas[half_h:out_h, x:x + width] = roi_tile
        draw_text(canvas, self.status, origin=(12, max(24, out_h - 68)))
        if cv2 is not None:
            cv2.line(canvas, (half_w, 0), (half_w, out_h), (80, 80, 80), 1)
            cv2.line(canvas, (0, half_h), (out_w, half_h), (80, 80, 80), 1)
        return canvas

    def close(self) -> None:
        self.enabled = False
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


def detection_features(det: Dict[str, Any], image_w: int, image_h: int, camera_key: str) -> Dict[str, Any]:
    if not det.get("detected"):
        return {
            "detected": 0,
            "det_conf": "",
            "x0": "",
            "y0": "",
            "x1": "",
            "y1": "",
            "bbox_w": "",
            "bbox_h": "",
            "center_x": "",
            "center_y": "",
            "width_norm": "",
            "height_norm": "",
            "center_x_norm": "",
            "center_y_norm": "",
            "area_norm": "",
            "log_area": "",
            "aspect": "",
        }
    x0 = max(0.0, min(float(image_w - 1), float(det["x0"])))
    y0 = max(0.0, min(float(image_h - 1), float(det["y0"])))
    x1 = max(x0 + 1.0, min(float(image_w), float(det["x1"])))
    y1 = max(y0 + 1.0, min(float(image_h), float(det["y1"])))
    bw = x1 - x0
    bh = y1 - y0
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    area_norm = (bw * bh) / max(1.0, float(image_w * image_h))
    return {
        "detected": 1,
        "det_conf": f"{float(det['conf']):.5f}",
        "x0": f"{x0:.2f}",
        "y0": f"{y0:.2f}",
        "x1": f"{x1:.2f}",
        "y1": f"{y1:.2f}",
        "bbox_w": f"{bw:.2f}",
        "bbox_h": f"{bh:.2f}",
        "center_x": f"{cx:.2f}",
        "center_y": f"{cy:.2f}",
        "width_norm": f"{bw / image_w:.6f}",
        "height_norm": f"{bh / image_h:.6f}",
        "center_x_norm": f"{cx / image_w:.6f}",
        "center_y_norm": f"{cy / image_h:.6f}",
        "area_norm": f"{area_norm:.8f}",
        "log_area": f"{math.log(max(area_norm, 1e-8)):.6f}",
        "aspect": f"{bw / max(1.0, bh):.6f}",
    }


CSV_FIELDS = [
    "image_path",
    "angle_deg",
    "camera_key",
    "sensor_id",
    "frame_idx",
    "angle_sample_idx",
    "timestamp",
    "monotonic_s",
    "image_w",
    "image_h",
    "roi_x0",
    "roi_y0",
    "roi_x1",
    "roi_y1",
    "detected",
    "det_conf",
    "x0",
    "y0",
    "x1",
    "y1",
    "bbox_w",
    "bbox_h",
    "center_x",
    "center_y",
    "width_norm",
    "height_norm",
    "center_x_norm",
    "center_y_norm",
    "area_norm",
    "log_area",
    "aspect",
]


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if first:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def process_frame(
    model,
    config: Dict[str, Any],
    output: Path,
    csv_path: Path,
    frame_rgb: np.ndarray,
    camera_key: str,
    sensor_id: int,
    angle_deg: float,
    frame_idx: int,
    angle_sample_idx: int,
    start_time: float,
    args: argparse.Namespace,
    preview: Optional[CalibrationHttpPreview] = None,
    status: Sequence[str] = (),
) -> bool:
    roi = roi_abs(config, camera_key, frame_rgb.shape)
    frame_fixed = fix_edge_color_cast(
        frame_rgb,
        enabled=True,
        width_ratio=args.edge_color_fix_width_ratio,
        strength=args.edge_color_fix_strength,
        green_recovery=args.edge_color_fix_green_recovery,
    )
    x0, y0, x1, y1 = roi
    image = fix_global_red_cast(frame_fixed[y0:y1, x0:x1], enabled=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = "jpg" if args.format == "jpg" else "png"
    angle_name = f"{angle_deg:+06.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
    rel = Path("images") / angle_name / camera_key / f"{stamp}_{camera_key}_a{angle_deg:+.1f}_f{frame_idx:07d}.{ext}"
    save_path = output / rel
    save_image(save_path, image, args.format, args.jpg_quality)

    det = best_detection(model, image, imgsz=args.imgsz, conf=args.conf, device=args.device)
    if preview is not None:
        preview.update_camera(camera_key, frame_fixed, roi, image, det, status)
    features = detection_features(det, image_w=image.shape[1], image_h=image.shape[0], camera_key=camera_key)
    append_row(
        csv_path,
        {
            "image_path": str(rel),
            "angle_deg": f"{angle_deg:.3f}",
            "camera_key": camera_key,
            "sensor_id": sensor_id,
            "frame_idx": frame_idx,
            "angle_sample_idx": angle_sample_idx,
            "timestamp": stamp,
            "monotonic_s": f"{time.monotonic() - start_time:.3f}",
            "image_w": image.shape[1],
            "image_h": image.shape[0],
            "roi_x0": x0,
            "roi_y0": y0,
            "roi_x1": x1,
            "roi_y1": y1,
            **features,
        },
    )
    return bool(features["detected"])


def wait_for_angle(angle: float, args: argparse.Namespace) -> str:
    if not args.interactive:
        return "capture"
    print()
    print(f"Set physical hinge angle to {angle:+.1f} deg.")
    print("Press Enter to capture, 's' then Enter to skip, or 'q' then Enter to quit.")
    answer = input("> ").strip().lower()
    if answer == "q":
        return "quit"
    if answer == "s":
        return "skip"
    return "capture"


def start_wait_preview(
    model,
    config: Dict[str, Any],
    cameras: Sequence[Tuple[GstCamera, str, int]],
    angle: float,
    args: argparse.Namespace,
    preview: Optional[CalibrationHttpPreview],
) -> Tuple[threading.Event, Optional[threading.Thread]]:
    stop_event = threading.Event()
    if preview is None or not preview.enabled or not cameras:
        return stop_event, None

    def loop() -> None:
        frame_idx = 0
        while not stop_event.is_set() and not _STOP:
            frame_idx += 1
            for cam, key, _sensor_id in cameras:
                frame = cam.read(timeout_ms=120)
                if frame is None:
                    continue
                roi = roi_abs(config, key, frame.shape)
                frame_fixed = fix_edge_color_cast(
                    frame,
                    enabled=True,
                    width_ratio=args.edge_color_fix_width_ratio,
                    strength=args.edge_color_fix_strength,
                    green_recovery=args.edge_color_fix_green_recovery,
                )
                x0, y0, x1, y1 = roi
                image = fix_global_red_cast(frame_fixed[y0:y1, x0:x1], enabled=True)
                try:
                    det = best_detection(model, image, imgsz=args.imgsz, conf=args.conf, device=args.device)
                except Exception as exc:
                    det = {"detected": False, "error": str(exc)}
                preview.update_camera(
                    key,
                    frame_fixed,
                    roi,
                    image,
                    det,
                    [
                        f"SET ANGLE {angle:+.1f} deg  press Enter in terminal to capture",
                        f"conf>={args.conf:.2f} imgsz={args.imgsz} preview frame={frame_idx}",
                    ],
                )
            time.sleep(0.001)

    thread = threading.Thread(target=loop, name="angle-calib-wait-preview", daemon=True)
    thread.start()
    return stop_event, thread


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    Gst.init(None)
    args = parse_args()
    angles = parse_angles(args.angles)
    config = read_yaml(Path(args.config).expanduser().resolve())
    preview = CalibrationHttpPreview(args) if args.http_stream else None
    output = Path(args.output).expanduser().resolve()
    csv_path = output / "angle_calibration.csv"
    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise SystemExit(f"YOLO weights not found: {weights}")

    from ultralytics import YOLO

    if weights.suffix.lower() == ".engine":
        model = YOLO(str(weights), task="detect")
    else:
        model = YOLO(str(weights))

    left_enabled = args.single_camera in ("", "left")
    right_enabled = args.single_camera in ("", "right")
    left: Optional[GstCamera] = None
    right: Optional[GstCamera] = None
    if args.parallel_open and left_enabled and right_enabled:
        opened: Dict[str, Optional[GstCamera]] = {"left": None, "right": None}

        def worker(name: str, sensor_id: int) -> None:
            opened[name] = open_camera(name, sensor_id, args)

        threads = [
            threading.Thread(target=worker, args=("left", args.left_camera), name="open-left", daemon=True),
            threading.Thread(target=worker, args=("right", args.right_camera), name="open-right", daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        left = opened["left"]
        right = opened["right"]
        if left is None or right is None:
            print("[camera] retrying missing camera(s) sequentially")
            if left is None:
                left = open_camera("left", args.left_camera, args)
            if right is None:
                right = open_camera("right", args.right_camera, args)
    else:
        left = open_camera("left", args.left_camera, args) if left_enabled else None
        right = open_camera("right", args.right_camera, args) if right_enabled else None
    if left is None and right is None:
        print("No camera is available.")
        return 2

    print(f"Output: {output}")
    print(f"CSV: {csv_path}")
    print(f"Weights: {weights}")
    print(f"Angles: {', '.join(f'{a:+.1f}' for a in angles)}")
    print(f"Samples per camera per angle: {args.samples_per_camera}")
    print("Edge color fix=ON")

    cameras = []
    if left is not None:
        cameras.append((left, args.left_key, args.left_camera))
    if right is not None:
        cameras.append((right, args.right_key, args.right_camera))

    def cameras_for_angle(angle_deg: float) -> List[Tuple[GstCamera, str, int]]:
        if not args.camera_by_sign:
            return list(cameras)
        if angle_deg < 0.0:
            return [(cam, key, sid) for cam, key, sid in cameras if key == args.right_key]
        if angle_deg > 0.0:
            return [(cam, key, sid) for cam, key, sid in cameras if key == args.left_key]
        return [] if args.skip_zero else list(cameras)

    frame_idx = 0
    start_time = time.monotonic()
    try:
        for angle in angles:
            if _STOP:
                break
            active_cameras = cameras_for_angle(angle)
            if not active_cameras:
                print(f"skip angle {angle:+.1f}: no active camera")
                continue
            active_names = ",".join(key for _, key, _ in active_cameras)
            print(f"[angle {angle:+.1f}] active camera(s): {active_names}")
            preview_stop, preview_thread = start_wait_preview(model, config, active_cameras, angle, args, preview)
            action = wait_for_angle(angle, args)
            preview_stop.set()
            if preview_thread is not None:
                preview_thread.join(timeout=1.0)
            if action == "quit":
                break
            if action == "skip":
                continue
            for _ in range(max(0, args.warmup_frames)):
                for cam, _, _ in active_cameras:
                    cam.read(timeout_ms=120)
            counts = {key: 0 for _, key, _ in active_cameras}
            detected = {key: 0 for _, key, _ in active_cameras}
            last_report_total = -1
            while not _STOP and any(counts[key] < args.samples_per_camera for _, key, _ in active_cameras):
                frame_idx += 1
                for cam, key, sensor_id in active_cameras:
                    frame = cam.read(timeout_ms=120)
                    if frame is None:
                        continue
                    if frame_idx % args.frame_step != 0:
                        continue
                    if counts[key] >= args.samples_per_camera:
                        continue
                    status = [
                        f"CAPTURING {angle:+.1f} deg",
                        " ".join(
                            f"{k}={counts[k]}/{args.samples_per_camera} det={detected[k]}"
                            for _, k, _ in active_cameras
                        ),
                    ]
                    ok = process_frame(
                        model=model,
                        config=config,
                        output=output,
                        csv_path=csv_path,
                        frame_rgb=frame,
                        camera_key=key,
                        sensor_id=sensor_id,
                        angle_deg=angle,
                        frame_idx=frame_idx,
                        angle_sample_idx=counts[key],
                        start_time=start_time,
                        args=args,
                        preview=preview,
                        status=status,
                    )
                    counts[key] += 1
                    detected[key] += int(ok)
                total = sum(counts.values())
                report_step = max(1, len(active_cameras) * max(1, args.samples_per_camera // 2))
                if total > 0 and total != last_report_total and total % report_step == 0:
                    last_report_total = total
                    print(
                        f"angle {angle:+.1f}: "
                        + " ".join(f"{k}={counts[k]}/{args.samples_per_camera} det={detected[k]}" for _, k, _ in active_cameras)
                    )
            print(
                f"done angle {angle:+.1f}: "
                + " ".join(f"{k} saved={counts[k]} detected={detected[k]}" for _, k, _ in active_cameras)
            )
    finally:
        for cam, _, _ in cameras:
            cam.close()
        if preview is not None:
            preview.close()

    print(f"Done. CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
