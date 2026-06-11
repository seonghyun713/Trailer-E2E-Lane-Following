#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trailer_parking_core import load_yaml, resolve_path  # noqa: E402


DEFAULT_POINT_NAMES = ("TL", "TR", "BR", "BL")


def point_names_from_config(config: Dict[str, Any]) -> Tuple[str, ...]:
    dual_cfg = ((config.get("bev", {}) or {}).get("dual", {}) or {})
    raw_names = dual_cfg.get("point_order")
    if isinstance(raw_names, (list, tuple)) and len(raw_names) >= 4:
        names = tuple(str(item) for item in raw_names)
        if all(names):
            return names
    return DEFAULT_POINT_NAMES


def parse_points(text: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_text, y_text = chunk.split(",")
        points.append((float(x_text), float(y_text)))
    if len(points) < 4:
        raise argparse.ArgumentTypeError("Expected 4 points: 'x,y;x,y;x,y;x,y'")
    return points


def latest_capture_dir(root: Path) -> Path:
    candidates = [path for path in root.expanduser().resolve().iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No capture directory under {root}")
    return sorted(candidates)[-1]


def label_padding(width: int, height: int) -> Tuple[int, int, int]:
    pad_x = max(220, int(round(width * 0.60)))
    pad_top = 0
    pad_bottom = 0
    return pad_x, pad_top, pad_bottom


def snap_y_to_edge(y: float, height: int, threshold_px: float = 12.0) -> float:
    if y <= threshold_px:
        return 0.0
    bottom = float(max(0, height - 1))
    if y >= bottom - threshold_px:
        return bottom
    return min(max(float(y), 0.0), bottom)


def click_points(image_path: Path, camera_key: str, point_names: Tuple[str, ...]) -> List[Tuple[float, float]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"OpenCV GUI is required for clicking points: {exc}") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    base = image.copy()
    height, width = base.shape[:2]
    pad_x, pad_top, pad_bottom = label_padding(width, height)
    points: List[Tuple[float, float]] = []
    expected_count = len(point_names)
    window = f"{camera_key}: click {', '.join(point_names)}"

    def redraw() -> None:
        canvas = np.zeros((height + pad_top + pad_bottom, width + pad_x * 2, 3), dtype=base.dtype)
        canvas[:] = (24, 24, 24)
        canvas[pad_top : pad_top + height, pad_x : pad_x + width] = base
        cv2.rectangle(canvas, (pad_x, pad_top), (pad_x + width, pad_top + height), (130, 130, 130), 1, cv2.LINE_AA)
        for index, point in enumerate(points):
            x, y = int(round(point[0] + pad_x)), int(round(point[1] + pad_top))
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, point_names[index], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if len(points) >= 2:
            for left, right in zip(points, points[1:]):
                cv2.line(
                    canvas,
                    (int(round(left[0] + pad_x)), int(round(left[1] + pad_top))),
                    (int(round(right[0] + pad_x)), int(round(right[1] + pad_top))),
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        info = f"{camera_key}: click {point_names[len(points)] if len(points) < expected_count else 'done'} | u=undo r=reset q=quit"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(canvas, info, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < expected_count:
            image_y = snap_y_to_edge(float(y - pad_top), height)
            points.append((float(x - pad_x), image_y))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while len(points) < expected_count:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise RuntimeError(f"Cancelled {camera_key}")
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == ord("r"):
            points.clear()
            redraw()
    redraw()
    cv2.waitKey(250)
    cv2.destroyWindow(window)
    return points


def points_payload(points: List[Tuple[float, float]], width: int, height: int, point_names: Tuple[str, ...]) -> Dict[str, Any]:
    clamped_points = [(float(x), snap_y_to_edge(float(y), height)) for x, y in points]
    return {
        "point_order": list(point_names),
        "image_size": [int(width), int(height)],
        "src_points_px": [[round(x, 2), round(y, 2)] for x, y in clamped_points],
        "src_points_ratio": [[round(x / max(1, width), 6), round(y / max(1, height), 6)] for x, y in clamped_points],
    }


def load_meta(capture_dir: Path) -> Dict[str, Any]:
    meta_path = capture_dir / "capture_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def bev_preview_config(config: Dict[str, Any], point_names: Tuple[str, ...]) -> Tuple[int, int, List[List[float]]]:
    bev_cfg = config.get("bev", {}) or {}
    dual_cfg = bev_cfg.get("dual", {}) or {}
    width = int(bev_cfg.get("output_width", 640))
    height = int(bev_cfg.get("output_height", 720))
    raw_points = dual_cfg.get("dst_points_ratio") or [[0.10, 0.08], [0.90, 0.08], [0.90, 0.95], [0.10, 0.95]]
    points: List[List[float]] = []
    for item in raw_points:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            points.append([float(item[0]), float(item[1])])
    if len(points) != len(point_names):
        points = [[0.10, 0.08], [0.90, 0.08], [0.90, 0.95], [0.10, 0.95]]
    return width, height, points


def make_headless_html(capture_dir: Path, meta: Dict[str, Any], config_path: Path, config: Dict[str, Any], point_names: Tuple[str, ...]) -> Path:
    cards = []
    point_text = ", ".join(point_names)
    bev_width, bev_height, dst_points_ratio = bev_preview_config(config, point_names)
    for camera_key, camera_meta in (meta.get("cameras", {}) or {}).items():
        crop_path = capture_dir / camera_meta["crop_image"]
        if not crop_path.exists():
            continue
        width, height = camera_meta["crop_size"]
        pad_x, pad_top, pad_bottom = label_padding(int(width), int(height))
        canvas_width = int(width) + pad_x * 2
        canvas_height = int(height) + pad_top + pad_bottom
        cards.append(
            f"""
      <section class="card" data-camera="{html.escape(camera_key)}">
        <h2>{html.escape(camera_key)} <span>click {html.escape(point_text)}</span></h2>
        <canvas id="canvas_{html.escape(camera_key)}" width="{canvas_width}" height="{canvas_height}" data-image-w="{int(width)}" data-image-h="{int(height)}" data-pad-x="{pad_x}" data-pad-top="{pad_top}"></canvas>
        <img id="img_{html.escape(camera_key)}" src="{image_data_uri(crop_path)}" hidden>
        <canvas class="zoom" id="zoom_{html.escape(camera_key)}" width="220" height="220"></canvas>
        <div class="points" id="points_{html.escape(camera_key)}"></div>
        <button onclick="undoPoint('{html.escape(camera_key)}')">Undo</button>
        <button onclick="resetPoints('{html.escape(camera_key)}')">Reset</button>
      </section>
"""
        )
    cameras = [key for key in (meta.get("cameras", {}) or {}).keys()]
    preview_card = f"""
      <section class="card bev-card">
        <h2>BEV preview <span>updates after 4 points</span></h2>
        <canvas id="bev_canvas" width="{bev_width}" height="{bev_height}"></canvas>
        <div class="points" id="bev_status"></div>
      </section>
"""
    command_prefix = (
        f"python3 {html.escape(str(HERE / 'label_dual_bev_points.py'))} "
        f"--config {html.escape(str(config_path))} "
        f"--capture-dir {html.escape(str(capture_dir))}"
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dual BEV Point Labeler</title>
  <style>
    body {{ margin: 0; padding: 20px; background: #111; color: #eee; font-family: sans-serif; }}
    h1 {{ margin: 0 0 12px; font-size: 22px; }}
    .wrap {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
    .card {{ background: #1b1b1b; border: 1px solid #333; padding: 12px; }}
    .bev-card {{ position: sticky; top: 12px; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    h2 span {{ color: #aaa; font-size: 13px; font-weight: normal; margin-left: 8px; }}
    canvas {{ display: block; max-width: 100%; height: auto; border: 1px solid #555; cursor: crosshair; }}
    #bev_canvas {{ width: 360px; max-height: 76vh; cursor: default; }}
    .zoom {{ width: 220px; height: 220px; margin-top: 8px; background: #080808; cursor: default; image-rendering: pixelated; }}
    button {{ margin-top: 8px; margin-right: 6px; padding: 6px 10px; }}
    .points {{ min-height: 22px; margin-top: 8px; color: #ddd; font-family: monospace; }}
    textarea {{ width: min(1200px, 100%); height: 96px; margin-top: 16px; background: #050505; color: #d8ffd8; }}
    .hint {{ color: #bbb; margin-bottom: 14px; }}
  </style>
</head>
<body>
  <h1>Dual BEV Point Labeler</h1>
  <div class="hint">For each camera, click matching physical road points in order: {html.escape(point_text)}. The horizontal gray margin is clickable, while Y stays inside the image. Then copy the command below into the Jetson terminal.</div>
  <div class="wrap">
    {''.join(cards)}
    {preview_card}
  </div>
  <textarea id="command" readonly></textarea>
  <script>
    const names = {json.dumps(list(point_names))};
    const expectedCount = names.length;
    const cameras = {json.dumps(cameras)};
    const points = Object.fromEntries(cameras.map(c => [c, []]));
    const hovers = Object.fromEntries(cameras.map(c => [c, null]));
    const commandPrefix = {json.dumps(command_prefix)};
    const bevWidth = {bev_width};
    const bevHeight = {bev_height};
    const dstRatios = {json.dumps(dst_points_ratio)};
    const dstPoints = dstRatios.map(p => [p[0] * bevWidth, p[1] * bevHeight]);
    const sourceData = {{}};
    const ySnapPx = 12;
    const zoomSize = 220;
    const zoomScale = 5;

    function snapImageY(y, imageH) {{
      const bottom = imageH - 1;
      if (y <= ySnapPx) return 0;
      if (y >= bottom - ySnapPx) return bottom;
      return Math.max(0, Math.min(bottom, y));
    }}

    function draw(camera, updatePreview = true) {{
      const canvas = document.getElementById(`canvas_${{camera}}`);
      const img = document.getElementById(`img_${{camera}}`);
      const ctx = canvas.getContext("2d");
      const imageW = Number(canvas.dataset.imageW);
      const imageH = Number(canvas.dataset.imageH);
      const padX = Number(canvas.dataset.padX);
      const padTop = Number(canvas.dataset.padTop);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#181818";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, padX, padTop, imageW, imageH);
      ctx.strokeStyle = "#888";
      ctx.lineWidth = 1;
      ctx.strokeRect(padX, padTop, imageW, imageH);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
      ctx.font = "16px sans-serif";
      const hover = hovers[camera];
      if (hover) {{
        const hoverImageX = hover.x - padX;
        const hoverImageY = snapImageY(hover.y - padTop, imageH);
        const hoverCanvasY = hoverImageY + padTop;
        ctx.save();
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(hover.x, 0);
        ctx.lineTo(hover.x, canvas.height);
        ctx.moveTo(0, hoverCanvasY);
        ctx.lineTo(canvas.width, hoverCanvasY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#00e5ff";
        ctx.font = "15px monospace";
        const labelX = Math.min(canvas.width - 150, Math.max(8, hover.x + 10));
        const labelY = Math.min(canvas.height - 12, Math.max(18, hoverCanvasY - 10));
        ctx.fillText(`x=${{hoverImageX.toFixed(1)}} y=${{hoverImageY.toFixed(1)}}`, labelX, labelY);
        ctx.restore();
      }}
      const ps = points[camera];
      for (let i = 0; i < ps.length; i++) {{
        const [imageX, imageY] = ps[i];
        const x = imageX + padX;
        const y = imageY + padTop;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(names[i], x + 8, y - 8);
        if (i > 0) {{
          ctx.beginPath();
          ctx.moveTo(ps[i - 1][0] + padX, ps[i - 1][1] + padTop);
          ctx.lineTo(x, y);
          ctx.stroke();
        }}
      }}
      document.getElementById(`points_${{camera}}`).textContent =
        ps.map((p, i) => `${{names[i]}}=(${{p[0].toFixed(1)}},${{p[1].toFixed(1)}})`).join(" ");
      drawZoom(camera, canvas, hover);
      updateCommand();
      if (updatePreview) drawBevPreview();
    }}

    function drawZoom(camera, sourceCanvas, hover) {{
      const zoomCanvas = document.getElementById(`zoom_${{camera}}`);
      if (!zoomCanvas) return;
      const ctx = zoomCanvas.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      ctx.fillStyle = "#080808";
      ctx.fillRect(0, 0, zoomCanvas.width, zoomCanvas.height);
      if (!hover) {{
        ctx.fillStyle = "#999";
        ctx.font = "14px monospace";
        ctx.fillText("move mouse over image", 18, 110);
        return;
      }}
      const crop = zoomCanvas.width / zoomScale;
      const sx = Math.max(0, Math.min(sourceCanvas.width - crop, hover.x - crop / 2));
      const sy = Math.max(0, Math.min(sourceCanvas.height - crop, hover.y - crop / 2));
      ctx.drawImage(sourceCanvas, sx, sy, crop, crop, 0, 0, zoomCanvas.width, zoomCanvas.height);
      ctx.save();
      ctx.strokeStyle = "#00e5ff";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(zoomCanvas.width / 2, 0);
      ctx.lineTo(zoomCanvas.width / 2, zoomCanvas.height);
      ctx.moveTo(0, zoomCanvas.height / 2);
      ctx.lineTo(zoomCanvas.width, zoomCanvas.height / 2);
      ctx.stroke();
      ctx.strokeStyle = "#ffffff";
      ctx.strokeRect(0.5, 0.5, zoomCanvas.width - 1, zoomCanvas.height - 1);
      ctx.restore();
    }}

    function canvasPoint(event, canvas) {{
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * canvas.width / rect.width;
      const y = (event.clientY - rect.top) * canvas.height / rect.height;
      const imageY = snapImageY(y - Number(canvas.dataset.padTop), Number(canvas.dataset.imageH));
      return [x - Number(canvas.dataset.padX), imageY];
    }}

    function canvasRawPoint(event, canvas) {{
      const rect = canvas.getBoundingClientRect();
      return {{
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height,
      }};
    }}

    function pointsArg(camera) {{
      return points[camera].map(p => `${{p[0].toFixed(1)}},${{p[1].toFixed(1)}}`).join(";");
    }}

    function updateCommand() {{
      let cmd = commandPrefix;
      for (const camera of cameras) {{
        if (points[camera].length === expectedCount) {{
          cmd += ` --${{camera}}-points="${{pointsArg(camera)}}"`;
        }}
      }}
      document.getElementById("command").value = cmd;
    }}

    function solveLinear(matrix, vector) {{
      const n = vector.length;
      const a = matrix.map((row, i) => row.slice().concat([vector[i]]));
      for (let col = 0; col < n; col++) {{
        let pivot = col;
        for (let row = col + 1; row < n; row++) {{
          if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
        }}
        if (Math.abs(a[pivot][col]) < 1e-9) return null;
        [a[col], a[pivot]] = [a[pivot], a[col]];
        const divisor = a[col][col];
        for (let j = col; j <= n; j++) a[col][j] /= divisor;
        for (let row = 0; row < n; row++) {{
          if (row === col) continue;
          const factor = a[row][col];
          for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
        }}
      }}
      return a.map(row => row[n]);
    }}

    function homography(src, dst) {{
      const a = [];
      const b = [];
      for (let i = 0; i < 4; i++) {{
        const [x, y] = src[i];
        const [u, v] = dst[i];
        a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
        b.push(u);
        a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
        b.push(v);
      }}
      const h = solveLinear(a, b);
      if (!h) return null;
      return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
    }}

    function invert3(m) {{
      const [a, b, c, d, e, f, g, h, i] = m;
      const A = e * i - f * h;
      const B = c * h - b * i;
      const C = b * f - c * e;
      const D = f * g - d * i;
      const E = a * i - c * g;
      const F = c * d - a * f;
      const G = d * h - e * g;
      const H = b * g - a * h;
      const I = a * e - b * d;
      const det = a * A + b * D + c * G;
      if (Math.abs(det) < 1e-9) return null;
      return [A / det, B / det, C / det, D / det, E / det, F / det, G / det, H / det, I / det];
    }}

    function transformPoint(m, x, y) {{
      const den = m[6] * x + m[7] * y + m[8];
      if (Math.abs(den) < 1e-9) return null;
      return [(m[0] * x + m[1] * y + m[2]) / den, (m[3] * x + m[4] * y + m[5]) / den];
    }}

    function getSourceData(camera) {{
      if (sourceData[camera]) return sourceData[camera];
      const img = document.getElementById(`img_${{camera}}`);
      const labelCanvas = document.getElementById(`canvas_${{camera}}`);
      const w = Number(labelCanvas.dataset.imageW);
      const h = Number(labelCanvas.dataset.imageH);
      if (!img.complete || w <= 0 || h <= 0) return null;
      const offscreen = document.createElement("canvas");
      offscreen.width = w;
      offscreen.height = h;
      const ctx = offscreen.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      sourceData[camera] = {{ width: w, height: h, data: ctx.getImageData(0, 0, w, h).data }};
      return sourceData[camera];
    }}

    function warpCameraToBev(camera, output, counts) {{
      if (points[camera].length !== expectedCount || expectedCount !== 4) return false;
      const source = getSourceData(camera);
      if (!source) return false;
      const h = homography(points[camera], dstPoints);
      const inv = h ? invert3(h) : null;
      if (!inv) return false;
      const srcData = source.data;
      for (let y = 0; y < bevHeight; y++) {{
        for (let x = 0; x < bevWidth; x++) {{
          const src = transformPoint(inv, x + 0.5, y + 0.5);
          if (!src) continue;
          const sx = Math.round(src[0]);
          const sy = Math.round(src[1]);
          if (sx < 0 || sx >= source.width || sy < 0 || sy >= source.height) continue;
          const dstIndex = (y * bevWidth + x) * 4;
          const srcIndex = (sy * source.width + sx) * 4;
          const countIndex = y * bevWidth + x;
          const count = counts[countIndex];
          output.data[dstIndex] = Math.round((output.data[dstIndex] * count + srcData[srcIndex]) / (count + 1));
          output.data[dstIndex + 1] = Math.round((output.data[dstIndex + 1] * count + srcData[srcIndex + 1]) / (count + 1));
          output.data[dstIndex + 2] = Math.round((output.data[dstIndex + 2] * count + srcData[srcIndex + 2]) / (count + 1));
          output.data[dstIndex + 3] = 255;
          counts[countIndex] = Math.min(255, count + 1);
        }}
      }}
      return true;
    }}

    function drawDestinationGuide(ctx) {{
      ctx.save();
      ctx.strokeStyle = "#00d1ff";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(dstPoints[0][0], dstPoints[0][1]);
      for (let i = 1; i < dstPoints.length; i++) ctx.lineTo(dstPoints[i][0], dstPoints[i][1]);
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "#00d1ff";
      ctx.font = "18px sans-serif";
      for (let i = 0; i < dstPoints.length; i++) {{
        ctx.beginPath();
        ctx.arc(dstPoints[i][0], dstPoints[i][1], 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(names[i], dstPoints[i][0] + 8, dstPoints[i][1] - 8);
      }}
      ctx.restore();
    }}

    function drawBevPreview() {{
      const canvas = document.getElementById("bev_canvas");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const output = ctx.createImageData(bevWidth, bevHeight);
      for (let i = 0; i < output.data.length; i += 4) {{
        output.data[i] = 16;
        output.data[i + 1] = 16;
        output.data[i + 2] = 16;
        output.data[i + 3] = 255;
      }}
      const counts = new Uint8Array(bevWidth * bevHeight);
      const ready = [];
      for (const camera of cameras) {{
        if (warpCameraToBev(camera, output, counts)) ready.push(camera);
      }}
      ctx.putImageData(output, 0, 0);
      drawDestinationGuide(ctx);
      document.getElementById("bev_status").textContent =
        ready.length ? `ready: ${{ready.join(", ")}}` : "click 4 points on a camera to preview BEV";
    }}

    function undoPoint(camera) {{
      points[camera].pop();
      draw(camera);
    }}

    function resetPoints(camera) {{
      points[camera] = [];
      draw(camera);
    }}

    for (const camera of cameras) {{
      const canvas = document.getElementById(`canvas_${{camera}}`);
      const img = document.getElementById(`img_${{camera}}`);
      img.onload = () => draw(camera);
      canvas.addEventListener("mousemove", event => {{
        hovers[camera] = canvasRawPoint(event, canvas);
        draw(camera, false);
      }});
      canvas.addEventListener("mouseleave", () => {{
        hovers[camera] = null;
        draw(camera, false);
      }});
      canvas.addEventListener("click", event => {{
        if (points[camera].length >= expectedCount) return;
        points[camera].push(canvasPoint(event, canvas));
        draw(camera);
      }});
      if (img.complete) draw(camera);
    }}
    updateCommand();
    drawBevPreview();
  </script>
</body>
</html>
"""
    html_path = capture_dir / "label_dual_bev_points.html"
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label matching road points for dual BEV calibration.")
    parser.add_argument("--config", type=Path, default=HERE / "dotted_lane_following_config.yaml")
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument("--capture-root", type=Path, default=HERE / "dual_bev_calib_capture")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--html-only", action="store_true", help="Create a browser-based labeler instead of opening OpenCV windows.")
    parser.add_argument("--cam0-points", type=parse_points, default=None, help="Optional 'x,y;x,y;x,y;x,y' for cam0 crop image.")
    parser.add_argument("--cam1-points", type=parse_points, default=None, help="Optional 'x,y;x,y;x,y;x,y' for cam1 crop image.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    point_names = point_names_from_config(config)
    config_dir = config_path.parent
    capture_dir = args.capture_dir.expanduser().resolve() if args.capture_dir else latest_capture_dir(args.capture_root)
    meta = load_meta(capture_dir)

    dual_cfg = ((config.get("bev", {}) or {}).get("dual", {}) or {})
    output = args.output
    if output is None:
        output = resolve_path(config_dir, str(dual_cfg.get("calibration_file", "dual_bev_calibration.yaml")))
    else:
        output = output.expanduser().resolve()

    point_overrides = {"cam0": args.cam0_points, "cam1": args.cam1_points}
    for camera_key, points in point_overrides.items():
        if points is not None and len(points) != len(point_names):
            raise SystemExit(f"{camera_key} expected {len(point_names)} points ({', '.join(point_names)}), got {len(points)}")
    if args.html_only or all(value is None for value in point_overrides.values()):
        if args.html_only:
            html_path = make_headless_html(capture_dir, meta, config_path, config, point_names)
            print(f"[html] {html_path}")
            print("Open this HTML in a browser, click points, then copy the generated command.")
            return 0

    cameras: Dict[str, Any] = {}
    for camera_key, camera_meta in (meta.get("cameras", {}) or {}).items():
        crop_path = capture_dir / camera_meta["crop_image"]
        width, height = camera_meta["crop_size"]
        points = point_overrides.get(camera_key)
        if points is None:
            try:
                points = click_points(crop_path, camera_key, point_names)
            except Exception as exc:
                html_path = make_headless_html(capture_dir, meta, config_path, config, point_names)
                print(f"[gui] OpenCV window unavailable: {exc}")
                print(f"[html] {html_path}")
                print(f"Open this HTML in a browser, click {', '.join(point_names)} for each camera, then copy the generated command.")
                return 2
        payload = points_payload(points, width, height, point_names)
        payload.update(
            {
                "capture_dir": str(capture_dir),
                "crop_image": camera_meta["crop_image"],
                "sensor_id": camera_meta.get("sensor_id"),
            }
        )
        cameras[camera_key] = payload
        print(f"[label] {camera_key}: {payload['src_points_ratio']}")

    if not cameras:
        raise SystemExit("No cameras were labeled.")

    dst_points = dual_cfg.get("dst_points_ratio") or [[0.10, 0.08], [0.90, 0.08], [0.90, 0.95], [0.10, 0.95]]
    payload = {
        "dual_bev": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "point_order": list(point_names),
            "capture_dir": str(capture_dir),
            "dst_points_ratio": dst_points,
            "vehicle_center_x_bias": dual_cfg.get("vehicle_center_x_bias", 0.0),
            "drive_mode": dual_cfg.get("drive_mode", "estimate_fusion"),
            "merge_mode": dual_cfg.get("merge_mode", "class_priority_max"),
            "cameras": cameras,
        }
    }

    try:
        import yaml
    except Exception as exc:
        raise SystemExit(f"PyYAML is required to save calibration: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[done] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
