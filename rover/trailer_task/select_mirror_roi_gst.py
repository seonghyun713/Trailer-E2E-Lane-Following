#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gi
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_bbox_frames_gst import GstCamera  # noqa: E402


LEFT_CAMERA_ID = 1
RIGHT_CAMERA_ID = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
CAPTURE_FPS = 30
APPSINK_FORMAT = "RGBA"


Box = Tuple[int, int, int, int]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV-free side-mirror ROI selector.")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--left-camera", type=int, default=LEFT_CAMERA_ID)
    parser.add_argument("--right-camera", type=int, default=RIGHT_CAMERA_ID)
    parser.add_argument("--left-key", default="cam1")
    parser.add_argument("--right-key", default="cam0")
    parser.add_argument("--capture-width", type=positive_int, default=CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=positive_int, default=CAPTURE_HEIGHT)
    parser.add_argument("--display-width", type=positive_int, default=DISPLAY_WIDTH)
    parser.add_argument("--display-height", type=positive_int, default=DISPLAY_HEIGHT)
    parser.add_argument("--fps", type=positive_int, default=CAPTURE_FPS)
    parser.add_argument("--appsink-format", choices=("RGBA", "BGRx", "RGBx"), default=APPSINK_FORMAT)
    parser.add_argument("--warmup-frames", type=int, default=6)
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def norm_roi_to_box(config: Dict[str, Any], key: str, width: int, height: int) -> Box:
    roi = config.get("cameras", {}).get(key, {}).get("roi", {})
    x = float(roi.get("x", 0.0))
    y = float(roi.get("y", 0.0))
    w = float(roi.get("w", 1.0))
    h = float(roi.get("h", 1.0))
    x0 = max(0, min(width - 1, int(round(x * width))))
    y0 = max(0, min(height - 1, int(round(y * height))))
    x1 = max(x0 + 1, min(width, int(round((x + w) * width))))
    y1 = max(y0 + 1, min(height, int(round((y + h) * height))))
    return x0, y0, x1, y1


def box_to_norm(box: Box, width: int, height: int) -> Dict[str, float]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return {
        "x": round(x0 / width, 5),
        "y": round(y0 / height, 5),
        "w": round((x1 - x0) / width, 5),
        "h": round((y1 - y0) / height, 5),
    }


def patch_config_roi(config_path: Path, camera_key: str, roi: Dict[str, float]) -> None:
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    current: Optional[str] = None
    replaced = False
    camera_re = re.compile(r"^(\s{2})([A-Za-z0-9_]+):\s*$")
    roi_re = re.compile(r"^(\s*)roi:\s*\{.*\}\s*$")
    new_line = (
        f"    roi: {{x: {roi['x']:.5f}, y: {roi['y']:.5f}, "
        f"w: {roi['w']:.5f}, h: {roi['h']:.5f}}}"
    )

    for i, line in enumerate(lines):
        m = camera_re.match(line)
        if m:
            current = m.group(2)
        if current == camera_key and roi_re.match(line):
            lines[i] = new_line
            replaced = True
            break

    if not replaced:
        raise RuntimeError(f"Could not find roi line for {camera_key} in {config_path}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def placeholder(name: str, width: int, height: int, message: str) -> np.ndarray:
    img = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    draw.text((20, height // 2 - 20), name, fill=(255, 80, 80))
    draw.text((20, height // 2 + 8), message[:60], fill=(255, 80, 80))
    return np.asarray(img, dtype=np.uint8)


def grab_snapshot(name: str, sensor_id: int, args: argparse.Namespace) -> np.ndarray:
    cam: Optional[GstCamera] = None
    try:
        cam = GstCamera(
            sensor_id=sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            output_width=args.display_width,
            output_height=args.display_height,
            fps=args.fps,
            sink_format=args.appsink_format,
        )
        frame = None
        for _ in range(max(1, args.warmup_frames)):
            got = cam.read(timeout_ms=300)
            if got is not None:
                frame = got
        if frame is None:
            raise RuntimeError("first frame is None")
        return frame
    except Exception as exc:
        return placeholder(name, args.display_width, args.display_height, str(exc))
    finally:
        if cam is not None:
            cam.close()
            time.sleep(0.12)


class RoiSelector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = Path(args.config).expanduser().resolve()
        self.config = read_yaml(self.config_path)
        self.left_key = args.left_key
        self.right_key = args.right_key
        self.width = args.display_width
        self.height = args.display_height
        self.gap = 10
        self.frames: Dict[str, np.ndarray] = {}
        self.boxes: Dict[str, Box] = {
            self.left_key: norm_roi_to_box(self.config, self.left_key, self.width, self.height),
            self.right_key: norm_roi_to_box(self.config, self.right_key, self.width, self.height),
        }
        self.drag_key: Optional[str] = None
        self.drag_start: Optional[Tuple[int, int]] = None

        self.root = tk.Tk()
        self.root.title("Mirror ROI Selector")
        self.canvas = tk.Canvas(self.root, width=self.width * 2 + self.gap, height=self.height, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=5, sticky="nsew")
        self.status = tk.StringVar(value="")
        ttk.Button(self.root, text="Refresh", command=self.refresh).grid(row=1, column=0, sticky="ew", padx=4, pady=6)
        ttk.Button(self.root, text="Save", command=self.save).grid(row=1, column=1, sticky="ew", padx=4, pady=6)
        ttk.Button(self.root, text="Quit", command=self.root.destroy).grid(row=1, column=2, sticky="ew", padx=4, pady=6)
        ttk.Label(self.root, textvariable=self.status).grid(row=1, column=3, columnspan=2, sticky="w", padx=8)
        self.root.columnconfigure(4, weight=1)

        self.photo = None
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("s", lambda _e: self.save())
        self.root.bind("r", lambda _e: self.refresh())
        self.root.bind("q", lambda _e: self.root.destroy())
        self.refresh()

    def refresh(self) -> None:
        self.status.set("capturing snapshots...")
        self.root.update_idletasks()
        self.frames = {
            self.left_key: grab_snapshot(f"{self.left_key}/cam{self.args.left_camera}", self.args.left_camera, self.args),
            self.right_key: grab_snapshot(f"{self.right_key}/cam{self.args.right_camera}", self.args.right_camera, self.args),
        }
        self.draw()
        self.status.set(self.status_text())

    def compose(self) -> Image.Image:
        left = Image.fromarray(self.frames[self.left_key], "RGB")
        right = Image.fromarray(self.frames[self.right_key], "RGB")
        combined = Image.new("RGB", (self.width * 2 + self.gap, self.height), (12, 12, 12))
        combined.paste(left, (0, 0))
        combined.paste(right, (self.width + self.gap, 0))
        draw = ImageDraw.Draw(combined)
        draw.text((8, 8), self.left_key, fill=(0, 255, 120))
        draw.text((self.width + self.gap + 8, 8), self.right_key, fill=(0, 255, 120))
        for key, offset_x, color in (
            (self.left_key, 0, (0, 200, 255)),
            (self.right_key, self.width + self.gap, (255, 210, 0)),
        ):
            x0, y0, x1, y1 = self.boxes[key]
            draw.rectangle((offset_x + x0, y0, offset_x + x1, y1), outline=color, width=3)
        return combined

    def draw(self) -> None:
        img = self.compose()
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def pane_for_x(self, x: int) -> Optional[Tuple[str, int]]:
        if 0 <= x < self.width:
            return self.left_key, 0
        right_x0 = self.width + self.gap
        if right_x0 <= x < right_x0 + self.width:
            return self.right_key, right_x0
        return None

    def on_press(self, event) -> None:
        pane = self.pane_for_x(event.x)
        if pane is None:
            return
        key, offset_x = pane
        self.drag_key = key
        self.drag_start = (max(0, min(self.width - 1, event.x - offset_x)), max(0, min(self.height - 1, event.y)))

    def on_drag(self, event) -> None:
        if self.drag_key is None or self.drag_start is None:
            return
        pane = self.pane_for_x(event.x)
        offset_x = 0 if self.drag_key == self.left_key else self.width + self.gap
        x0, y0 = self.drag_start
        x1 = max(0, min(self.width, event.x - offset_x))
        y1 = max(0, min(self.height, event.y))
        xa, xb = sorted((int(x0), int(x1)))
        ya, yb = sorted((int(y0), int(y1)))
        self.boxes[self.drag_key] = (xa, ya, max(xa + 1, xb), max(ya + 1, yb))
        self.draw()
        self.status.set(self.status_text())

    def on_release(self, event) -> None:
        self.on_drag(event)
        self.drag_key = None
        self.drag_start = None

    def save(self) -> None:
        for key in (self.left_key, self.right_key):
            patch_config_roi(self.config_path, key, box_to_norm(self.boxes[key], self.width, self.height))
        self.config = read_yaml(self.config_path)
        self.status.set("saved: " + self.status_text())

    def status_text(self) -> str:
        parts = []
        for key in (self.left_key, self.right_key):
            roi = box_to_norm(self.boxes[key], self.width, self.height)
            parts.append(f"{key}=x{roi['x']:.3f},y{roi['y']:.3f},w{roi['w']:.3f},h{roi['h']:.3f}")
        return " | ".join(parts)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    Gst.init(None)
    args = parse_args()
    RoiSelector(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
