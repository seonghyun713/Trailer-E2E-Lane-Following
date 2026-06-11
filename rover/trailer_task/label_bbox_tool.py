#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_dataset = here / "bbox_dataset"
    parser = argparse.ArgumentParser(description="Simple PIL/tkinter bbox labeler for blue trailer panels.")
    parser.add_argument("--dataset", default=str(default_dataset))
    parser.add_argument("--images", default="", help="Optional image directory. Defaults to DATASET/images.")
    parser.add_argument("--labels", default="", help="Defaults to DATASET/labels.csv.")
    parser.add_argument("--config", default=str(here / "config.yaml"))
    parser.add_argument("--start", default="", help="Start from an image path substring.")
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--max-display-height", type=int, default=820)
    parser.add_argument("--zoom", type=float, default=3.0, help="Initial display zoom.")
    parser.add_argument("--min-zoom", type=float, default=0.5)
    parser.add_argument("--max-zoom", type=float, default=10.0)
    parser.add_argument("--label-mode", choices=("point4", "bbox"), default="point4")
    return parser.parse_args()


@dataclass
class CaptureMeta:
    camera_key: str = ""
    save_mode: str = ""
    roi_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)


class LabelStore:
    fieldnames = [
        "image_path",
        "camera_key",
        "usable",
        "x0",
        "y0",
        "x1",
        "y1",
        "width_px",
        "height_px",
        "center_x",
        "center_y",
        "label_type",
        "q0_x",
        "q0_y",
        "q1_x",
        "q1_y",
        "q2_x",
        "q2_y",
        "q3_x",
        "q3_y",
        "angle_deg",
        "updated_at",
    ]

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.rows: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            return
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row.get("image_path", "")
                if image_path:
                    self.rows[image_path] = row

    def get(self, image_path: str) -> Optional[Dict[str, str]]:
        return self.rows.get(image_path)

    def set_box(
        self,
        image_path: str,
        camera_key: str,
        box: Optional[Tuple[int, int, int, int]],
        angle_deg: str,
        usable: bool,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        row = {name: "" for name in self.fieldnames}
        row["image_path"] = image_path
        row["camera_key"] = camera_key
        row["usable"] = "1" if usable and box is not None else "0"
        row["angle_deg"] = angle_deg.strip()
        row["updated_at"] = f"{time.time():.3f}"
        if points and usable:
            row["label_type"] = "quad4" if len(points) == 4 else "points"
            for idx, (px, py) in enumerate(points[:4]):
                row[f"q{idx}_x"] = str(int(px))
                row[f"q{idx}_y"] = str(int(py))
        elif box is not None and usable:
            row["label_type"] = "bbox"
        else:
            row["label_type"] = "none"
        if box is not None and usable:
            x0, y0, x1, y1 = box
            row["x0"] = str(x0)
            row["y0"] = str(y0)
            row["x1"] = str(x1)
            row["y1"] = str(y1)
            row["width_px"] = str(max(0, x1 - x0))
            row["height_px"] = str(max(0, y1 - y0))
            row["center_x"] = f"{0.5 * (x0 + x1):.2f}"
            row["center_y"] = f"{0.5 * (y0 + y1):.2f}"
        self.rows[image_path] = row
        self.save()

    def delete(self, image_path: str) -> None:
        self.rows.pop(image_path, None)
        self.save()

    def save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for key in sorted(self.rows):
                row = self.rows[key]
                writer.writerow({name: row.get(name, "") for name in self.fieldnames})


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_capture_meta(dataset: Path) -> Dict[str, CaptureMeta]:
    csv_path = dataset / "captures.csv"
    metas: Dict[str, CaptureMeta] = {}
    if not csv_path.exists():
        return metas
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row.get("image_path", "")
            if not image_path:
                continue
            try:
                roi = (
                    int(float(row.get("roi_x0", "0") or 0)),
                    int(float(row.get("roi_y0", "0") or 0)),
                    int(float(row.get("roi_x1", "0") or 0)),
                    int(float(row.get("roi_y1", "0") or 0)),
                )
            except Exception:
                roi = (0, 0, 0, 0)
            metas[image_path] = CaptureMeta(
                camera_key=row.get("camera_key", ""),
                save_mode=row.get("save_mode", ""),
                roi_xyxy=roi,
            )
    return metas


def infer_camera_key(path: Path, rel: str, meta: Optional[CaptureMeta]) -> str:
    if meta and meta.camera_key:
        return meta.camera_key
    parts = set(path.parts)
    for key in ("cam0", "cam1", "left", "right"):
        if key in parts or key in rel:
            return key
    return ""


def clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return x0, y0, x1, y1


def box_from_points(points: Sequence[Tuple[int, int]], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if len(points) < 2:
        return None
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return clamp_box((min(xs), min(ys), max(xs) + 1, max(ys) + 1), width, height)


def parse_quad_points(row: Dict[str, str]) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for idx in range(4):
        x = row.get(f"q{idx}_x", "")
        y = row.get(f"q{idx}_y", "")
        if x == "" or y == "":
            return []
        try:
            points.append((int(float(x)), int(float(y))))
        except Exception:
            return []
    return points


def auto_suggest_blue_box(image: Image.Image, camera_key: str, config: Dict[str, Any], meta: Optional[CaptureMeta]) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    search_offset = (0, 0)
    search = arr

    if meta and meta.save_mode == "full" and camera_key:
        roi = config.get("cameras", {}).get(camera_key, {}).get("roi", {})
        rx = float(roi.get("x", 0.0))
        ry = float(roi.get("y", 0.0))
        rw = float(roi.get("w", 1.0))
        rh = float(roi.get("h", 1.0))
        x0 = max(0, min(w - 1, int(round(rx * w))))
        y0 = max(0, min(h - 1, int(round(ry * h))))
        x1 = max(x0 + 1, min(w, int(round((rx + rw) * w))))
        y1 = max(y0 + 1, min(h, int(round((ry + rh) * h))))
        search = arr[y0:y1, x0:x1]
        search_offset = (x0, y0)

    r = search[:, :, 0].astype(np.int16)
    g = search[:, :, 1].astype(np.int16)
    b = search[:, :, 2].astype(np.int16)
    blue = (b > 85) & (b - r > 28) & (b - g > 8) & ((b + g) > 145)
    white = (r > 185) & (g > 185) & (b > 185)

    if ndi is not None:
        mask = ndi.binary_opening(blue, structure=np.ones((2, 2), dtype=bool))
        labels, count = ndi.label(mask)
        slices = ndi.find_objects(labels)
        best = None
        best_score = -1.0
        for label_id, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            ys, xs = slc
            x0, x1 = int(xs.start), int(xs.stop)
            y0, y1 = int(ys.start), int(ys.stop)
            bw, bh = x1 - x0, y1 - y0
            if bw < 8 or bh < 5:
                continue
            aspect = bw / max(1, bh)
            if aspect < 0.8 or aspect > 10.0:
                continue
            area = int((labels[slc] == label_id).sum())
            white_inside = float(white[y0:y1, x0:x1].mean()) if bw * bh else 0.0
            score = area * (1.0 + 2.5 * white_inside) * min(2.0, aspect)
            if score > best_score:
                best_score = score
                best = (x0, y0, x1, y1)
        if best is not None:
            x0, y0, x1, y1 = best
            pad = max(2, int(0.03 * max(x1 - x0, y1 - y0)))
            ox, oy = search_offset
            return clamp_box((x0 + ox - pad, y0 + oy - pad, x1 + ox + pad, y1 + oy + pad), w, h)

    ys, xs = np.where(blue)
    if len(xs) < 20:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    ox, oy = search_offset
    return clamp_box((x0 + ox, y0 + oy, x1 + ox, y1 + oy), w, h)


class BBoxLabelTool:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = Path(args.dataset).expanduser().resolve()
        self.image_root = Path(args.images).expanduser().resolve() if args.images else self.dataset / "images"
        self.label_path = Path(args.labels).expanduser().resolve() if args.labels else self.dataset / "labels.csv"
        self.config = _read_yaml(Path(args.config).expanduser().resolve())
        self.images = list_images(self.image_root)
        if not self.images:
            raise SystemExit(f"No images found: {self.image_root}")
        self.meta = load_capture_meta(self.dataset)
        self.store = LabelStore(self.label_path)
        self.index = 0
        if args.start:
            for i, path in enumerate(self.images):
                if args.start in str(path):
                    self.index = i
                    break
        else:
            for i, path in enumerate(self.images):
                if self.rel_path(path) not in self.store.rows:
                    self.index = i
                    break

        self.root = tk.Tk()
        self.root.title("Trailer Blue Panel BBox Labeler")
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0, bg="#101010")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        panel = ttk.Frame(self.root, padding=8)
        panel.grid(row=0, column=1, sticky="ns")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.camera_var = tk.StringVar()
        self.angle_var = tk.StringVar()
        self.usable_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar()
        self._action_lock_until = 0.0

        ttk.Label(panel, text="camera").grid(row=0, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.camera_var, width=14).grid(row=1, column=0, sticky="ew")
        ttk.Label(panel, text="angle_deg").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(panel, textvariable=self.angle_var, width=14).grid(row=3, column=0, sticky="ew")
        ttk.Checkbutton(panel, text="usable", variable=self.usable_var).grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.add_button(panel, "Save", self.save_current, row=5, pady=(10, 0))
        self.add_button(panel, "Save+Next", self.save_and_next, row=6)
        self.add_button(panel, "No Marker", self.save_no_marker, row=7)
        self.add_button(panel, "Undo Point", self.undo_point, row=8)
        self.add_button(panel, "Clear Label", self.delete_box, row=9)
        self.add_button(panel, "Zoom In", lambda: self.change_zoom(1.25), row=10, pady=(14, 0))
        self.add_button(panel, "Zoom Out", lambda: self.change_zoom(0.80), row=11)
        self.add_button(panel, "Fit", self.fit_zoom, row=12)
        self.add_button(panel, "Prev", self.prev_image, row=13, pady=(14, 0))
        self.add_button(panel, "Next", self.next_image, row=14)
        ttk.Label(panel, textvariable=self.status_var, wraplength=180).grid(row=15, column=0, sticky="w", pady=(14, 0))

        self.image: Optional[Image.Image] = None
        self.photo = None
        self.scale = max(float(args.min_zoom), min(float(args.max_zoom), float(args.zoom)))
        self.last_display_size: Tuple[int, int] = (1, 1)
        self.box: Optional[Tuple[int, int, int, int]] = None
        self.points: List[Tuple[int, int]] = []
        self.drag_start: Optional[Tuple[int, int]] = None
        self.rect_id: Optional[int] = None
        self.point_item_ids: List[int] = []
        self.poly_item_id: Optional[int] = None
        self.image_id: Optional[int] = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda _e: self.change_zoom(1.15))
        self.canvas.bind("<Button-5>", lambda _e: self.change_zoom(1.0 / 1.15))
        self.root.bind("<Return>", lambda event: self.on_key_action(event, self.save_and_next, allow_entry=True))
        self.root.bind("s", lambda event: self.on_key_action(event, self.save_current))
        self.root.bind("n", lambda event: self.on_key_action(event, self.next_image))
        self.root.bind("p", lambda event: self.on_key_action(event, self.prev_image))
        self.root.bind("d", lambda event: self.on_key_action(event, self.delete_box))
        self.root.bind("u", lambda event: self.on_key_action(event, self.undo_point))
        self.root.bind("<BackSpace>", lambda event: self.on_key_action(event, self.undo_point, allow_entry=True))
        self.root.bind("x", lambda event: self.on_key_action(event, self.save_no_marker))
        self.root.bind("+", lambda event: self.on_key_action(event, lambda: self.change_zoom(1.25)))
        self.root.bind("=", lambda event: self.on_key_action(event, lambda: self.change_zoom(1.25)))
        self.root.bind("-", lambda event: self.on_key_action(event, lambda: self.change_zoom(0.80)))
        self.root.bind("0", lambda event: self.on_key_action(event, lambda: self.set_zoom(float(self.args.zoom))))
        self.root.bind("f", lambda event: self.on_key_action(event, self.fit_zoom))
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

        self.load_current()

    def add_button(self, parent, text: str, command, row: int, pady=(4, 0)) -> None:
        button = ttk.Button(parent, text=text, command=self.guard_action(command), takefocus=False)
        button.grid(row=row, column=0, sticky="ew", pady=pady)

    def guard_action(self, command):
        def wrapped():
            now = time.monotonic()
            if now < self._action_lock_until:
                return "break"
            self._action_lock_until = now + 0.12
            command()
            self.canvas.focus_set()
            return "break"

        return wrapped

    def on_key_action(self, event, command, allow_entry: bool = False):
        if not allow_entry and isinstance(event.widget, tk.Entry):
            return None
        return self.guard_action(command)()

    def rel_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.dataset))
        except Exception:
            return str(path)

    def current_meta(self) -> Optional[CaptureMeta]:
        return self.meta.get(self.rel_path(self.images[self.index]))

    def load_current(self) -> None:
        path = self.images[self.index]
        rel = self.rel_path(path)
        self.image = Image.open(path).convert("RGB")

        meta = self.current_meta()
        camera_key = infer_camera_key(path, rel, meta)
        row = self.store.get(rel)
        self.box = None
        self.points = []
        self.camera_var.set(camera_key)
        self.angle_var.set("")
        self.usable_var.set(True)
        if row:
            self.camera_var.set(row.get("camera_key", camera_key))
            self.angle_var.set(row.get("angle_deg", ""))
            self.usable_var.set(row.get("usable", "1") == "1")
            self.points = parse_quad_points(row)
            try:
                if row.get("usable", "1") == "1":
                    if self.points:
                        self.box = box_from_points(self.points, *self.image.size)
                    else:
                        self.box = (
                            int(float(row["x0"])),
                            int(float(row["y0"])),
                            int(float(row["x1"])),
                            int(float(row["y1"])),
                        )
            except Exception:
                self.box = None
                self.points = []
        self.render_image(reset_view=True)
        self.update_status()

    def update_status(self) -> None:
        path = self.images[self.index]
        rel = self.rel_path(path)
        width_text = ""
        if self.box is not None:
            width_text = f" box_w={self.box[2] - self.box[0]}px"
        point_text = f" points={len(self.points)}/4" if self.args.label_mode == "point4" else " mode=bbox"
        labeled = len(self.store.rows)
        self.status_var.set(
            f"{self.index + 1}/{len(self.images)} labeled={labeled} zoom={self.scale:.2f}x"
            f"{point_text}{width_text}\n{rel}"
        )

    def render_image(self, reset_view: bool = False) -> None:
        if self.image is None:
            return
        w, h = self.image.size
        disp_size = (max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale))))
        self.last_display_size = disp_size
        resampling = getattr(Image, "Resampling", Image).NEAREST if self.scale >= 2.0 else getattr(Image, "Resampling", Image).BILINEAR
        disp = self.image.resize(disp_size, resampling)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.image_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(
            width=min(self.args.max_display_width, disp_size[0]),
            height=min(self.args.max_display_height, disp_size[1]),
            scrollregion=(0, 0, disp_size[0], disp_size[1]),
        )
        self.draw_box()
        if reset_view:
            self.canvas.xview_moveto(0.0)
            self.canvas.yview_moveto(0.0)

    def draw_box(self) -> None:
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        if self.poly_item_id is not None:
            self.canvas.delete(self.poly_item_id)
            self.poly_item_id = None
        for item_id in self.point_item_ids:
            self.canvas.delete(item_id)
        self.point_item_ids = []

        if self.points:
            scaled_points = [(px * self.scale, py * self.scale) for px, py in self.points]
            flat = [coord for point in scaled_points for coord in point]
            if len(scaled_points) >= 2:
                if len(scaled_points) == 4:
                    self.poly_item_id = self.canvas.create_polygon(
                        *flat,
                        outline="#ffd000",
                        fill="",
                        width=3,
                    )
                else:
                    self.poly_item_id = self.canvas.create_line(*flat, fill="#ffd000", width=3)
            radius = max(4.0, 1.8 * self.scale)
            for idx, (sx, sy) in enumerate(scaled_points):
                item = self.canvas.create_oval(
                    sx - radius,
                    sy - radius,
                    sx + radius,
                    sy + radius,
                    outline="#111111",
                    fill="#ffd000",
                    width=2,
                )
                self.point_item_ids.append(item)
                text = self.canvas.create_text(
                    sx + radius + 5,
                    sy - radius - 5,
                    text=str(idx + 1),
                    fill="#ffd000",
                    anchor="nw",
                )
                self.point_item_ids.append(text)

        if self.box is None:
            return
        x0, y0, x1, y1 = self.box
        sx0, sy0, sx1, sy1 = [v * self.scale for v in (x0, y0, x1, y1)]
        self.rect_id = self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline="#00a8ff", width=3)

    def display_to_image(self, x: int, y: int) -> Tuple[int, int]:
        assert self.image is not None
        w, h = self.image.size
        cx = self.canvas.canvasx(x)
        cy = self.canvas.canvasy(y)
        ix = max(0, min(w - 1, int(round(cx / self.scale))))
        iy = max(0, min(h - 1, int(round(cy / self.scale))))
        return ix, iy

    def set_zoom(self, zoom: float) -> None:
        zoom = max(float(self.args.min_zoom), min(float(self.args.max_zoom), float(zoom)))
        if abs(zoom - self.scale) < 1e-6:
            return
        x_frac = self.canvas.xview()[0] if self.last_display_size[0] > 1 else 0.0
        y_frac = self.canvas.yview()[0] if self.last_display_size[1] > 1 else 0.0
        self.scale = zoom
        self.render_image(reset_view=False)
        self.canvas.xview_moveto(x_frac)
        self.canvas.yview_moveto(y_frac)
        self.update_status()

    def change_zoom(self, factor: float) -> None:
        self.set_zoom(self.scale * float(factor))

    def fit_zoom(self) -> None:
        if self.image is None:
            return
        w, h = self.image.size
        fit = min(
            self.args.max_display_width / max(1, w),
            self.args.max_display_height / max(1, h),
        )
        self.set_zoom(fit)

    def on_mousewheel(self, event) -> None:
        factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        self.change_zoom(factor)

    def on_press(self, event) -> None:
        if self.args.label_mode == "point4":
            self.add_point_from_event(event)
            return
        self.drag_start = self.display_to_image(event.x, event.y)

    def on_drag(self, event) -> None:
        if self.args.label_mode == "point4":
            return
        if self.drag_start is None or self.image is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.display_to_image(event.x, event.y)
        self.box = clamp_box((x0, y0, x1, y1), *self.image.size)
        self.draw_box()
        self.update_status()

    def on_release(self, event) -> None:
        if self.args.label_mode == "point4":
            return
        self.on_drag(event)
        self.drag_start = None

    def add_point_from_event(self, event) -> None:
        if self.image is None:
            return
        if len(self.points) >= 4:
            self.points = []
            self.box = None
        point = self.display_to_image(event.x, event.y)
        self.points.append(point)
        if len(self.points) >= 2:
            self.box = box_from_points(self.points, *self.image.size)
        self.usable_var.set(True)
        self.draw_box()
        self.update_status()

    def undo_point(self) -> None:
        if self.points:
            self.points.pop()
            if self.image is not None and len(self.points) >= 2:
                self.box = box_from_points(self.points, *self.image.size)
            else:
                self.box = None
        else:
            self.box = None
        self.draw_box()
        self.update_status()

    def auto_box(self) -> None:
        if self.image is None:
            return
        meta = self.current_meta()
        self.box = auto_suggest_blue_box(self.image, self.camera_var.get().strip(), self.config, meta)
        self.draw_box()
        self.update_status()

    def delete_box(self) -> None:
        self.box = None
        self.points = []
        self.draw_box()
        self.update_status()

    def save_current(self) -> bool:
        rel = self.rel_path(self.images[self.index])
        if self.args.label_mode == "point4" and 0 < len(self.points) < 4:
            self.status_var.set(
                f"{self.index + 1}/{len(self.images)} need 4 points, or Clear Label / x for no marker\n{rel}"
            )
            self.canvas.focus_set()
            return False

        points = self.points if self.args.label_mode == "point4" and len(self.points) == 4 else []
        if self.image is not None and points:
            self.box = box_from_points(points, *self.image.size)

        usable = self.usable_var.get() and self.box is not None and (self.args.label_mode != "point4" or len(points) == 4)
        if not usable:
            self.usable_var.set(False)
        self.store.set_box(
            image_path=rel,
            camera_key=self.camera_var.get().strip(),
            box=self.box,
            angle_deg=self.angle_var.get(),
            usable=usable,
            points=points,
        )
        self.update_status()
        self.canvas.focus_set()
        return True

    def save_and_next(self) -> None:
        if self.save_current():
            self.next_image()

    def save_no_marker(self) -> None:
        self.box = None
        self.points = []
        self.usable_var.set(False)
        if self.save_current():
            self.next_image()

    def delete_current(self) -> None:
        self.store.delete(self.rel_path(self.images[self.index]))
        self.box = None
        self.draw_box()
        self.update_status()

    def prev_image(self) -> None:
        self.index = max(0, self.index - 1)
        self.load_current()

    def next_image(self) -> None:
        self.index = min(len(self.images) - 1, self.index + 1)
        self.load_current()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    tool = BBoxLabelTool(args)
    tool.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
