#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip3 install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class FastDetectionResult:
    ok: bool
    camera_key: str
    method: str = "torch_ncc"
    variant: str = ""
    confidence: float = 0.0
    score: float = 0.0
    angle_deg: Optional[float] = None
    raw_angle_deg: Optional[float] = None
    model: str = "fast_torch_template"
    roi_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)
    corners: Optional[np.ndarray] = None
    features: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusedAngle:
    ok: bool
    angle_deg: Optional[float]
    confidence: float
    source: str
    message: str = ""


@dataclass
class _TemplateSpec:
    variant: str
    rotation_deg: float
    nominal_width: int
    kernel: torch.Tensor
    ones: torch.Tensor
    norm: torch.Tensor
    area: float
    height: int
    width: int


class AngleFilter:
    def __init__(self, alpha: float = 0.35, min_confidence: float = 0.18, max_jump_deg: float = 28.0):
        self.alpha = float(alpha)
        self.min_confidence = float(min_confidence)
        self.max_jump_deg = float(max_jump_deg)
        self.value: Optional[float] = None

    def update(self, angle_deg: Optional[float], confidence: float) -> Optional[float]:
        if angle_deg is None or confidence < self.min_confidence:
            return self.value
        angle = float(angle_deg)
        if self.value is None:
            self.value = angle
            return self.value
        delta = angle - self.value
        while delta <= -180.0:
            delta += 360.0
        while delta > 180.0:
            delta -= 360.0
        delta = _clip(delta, -self.max_jump_deg, self.max_jump_deg)
        self.value += self.alpha * delta
        return self.value


class TorchTrailerEstimator:
    """OpenCV-free trailer marker estimator.

    This is intentionally a fast tracker, not a full geometry solver. It uses
    normalized cross-correlation against rotated marker templates on the GPU,
    then converts marker bearing/roll into the same fallback angle convention
    used by config.yaml.
    """

    def __init__(
        self,
        config_path: str | Path,
        device: str = "auto",
        fp16: bool = True,
        template_widths: Optional[Sequence[int]] = None,
        rotations: Optional[Sequence[float]] = None,
    ):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config_dir = self.config_path.parent
        self.config = _read_yaml(self.config_path)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" and fp16 else torch.float32
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        fast_cfg = self.config.get("fast_detection", {})
        if template_widths is None:
            template_widths = fast_cfg.get("template_widths_px", [40, 52, 68, 88, 112, 144, 184])
        if rotations is None:
            rotations = fast_cfg.get("rotations_deg", [-35, -25, -15, -8, 0, 8, 15, 25, 35])
        self.min_score = float(fast_cfg.get("min_score", 0.33))
        self.min_texture = float(fast_cfg.get("min_texture", 0.16))
        self.max_templates_per_estimate = int(fast_cfg.get("max_templates_per_estimate", 96))
        self.templates = self._build_templates(template_widths, rotations)
        if not self.templates:
            raise RuntimeError("No usable marker templates were generated.")

    def _build_templates(self, widths: Sequence[int], rotations: Sequence[float]) -> List[_TemplateSpec]:
        marker_cfg = self.config.get("marker", {})
        marker_path = _resolve_path(self.config_dir, marker_cfg.get("image_path", "../trailer_maker.png"))
        base = self._load_marker_gray(marker_path)
        variants = marker_cfg.get("variants", ["normal", "mirror_x"])
        if not variants:
            variants = ["normal"]

        resampling = getattr(Image, "Resampling", Image).BILINEAR
        specs: List[_TemplateSpec] = []
        for variant in variants:
            img = ImageOps.mirror(base) if variant == "mirror_x" else base
            for nominal_width in sorted({int(w) for w in widths if int(w) >= 16}):
                h = max(8, int(round(nominal_width * img.height / max(1, img.width))))
                resized = img.resize((nominal_width, h), resampling)
                for rotation in rotations:
                    rotated = resized.rotate(float(rotation), resample=resampling, expand=True, fillcolor=255)
                    arr = np.asarray(rotated, dtype=np.float32) / 255.0
                    if arr.shape[0] < 8 or arr.shape[1] < 16:
                        continue
                    t = torch.from_numpy(arr).to(device=self.device, dtype=torch.float32)
                    t0 = t - t.mean()
                    norm = torch.linalg.vector_norm(t0).clamp_min(1e-4)
                    kernel = t0[None, None].to(dtype=self.dtype).contiguous()
                    ones = torch.ones_like(kernel, device=self.device, dtype=self.dtype)
                    specs.append(
                        _TemplateSpec(
                            variant=str(variant),
                            rotation_deg=float(rotation),
                            nominal_width=int(nominal_width),
                            kernel=kernel,
                            ones=ones,
                            norm=norm.to(dtype=self.dtype),
                            area=float(t0.numel()),
                            height=int(arr.shape[0]),
                            width=int(arr.shape[1]),
                        )
                    )
        specs.sort(key=lambda s: (s.nominal_width, abs(s.rotation_deg), s.variant))
        return specs[: max(1, self.max_templates_per_estimate)]

    @staticmethod
    def _load_marker_gray(path: Path) -> Image.Image:
        rgba = Image.open(path).convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(white, rgba).convert("L")
        arr = np.asarray(composed)
        mask = arr < 245
        if mask.any():
            ys, xs = np.where(mask)
            pad_x = max(4, int(0.02 * composed.width))
            pad_y = max(4, int(0.02 * composed.height))
            x0 = max(0, int(xs.min()) - pad_x)
            y0 = max(0, int(ys.min()) - pad_y)
            x1 = min(composed.width, int(xs.max()) + pad_x + 1)
            y1 = min(composed.height, int(ys.max()) + pad_y + 1)
            composed = composed.crop((x0, y0, x1, y1))
        return composed

    def _roi_abs(self, frame_shape: Sequence[int], camera_key: str) -> Tuple[int, int, int, int]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        roi = self.config.get("cameras", {}).get(camera_key, {}).get("roi", {})
        x = _as_float(roi.get("x"), 0.0)
        y = _as_float(roi.get("y"), 0.0)
        rw = _as_float(roi.get("w"), 1.0)
        rh = _as_float(roi.get("h"), 1.0)
        x0 = max(0, min(w - 1, int(round(x * w))))
        y0 = max(0, min(h - 1, int(round(y * h))))
        x1 = max(x0 + 1, min(w, int(round((x + rw) * w))))
        y1 = max(y0 + 1, min(h, int(round((y + rh) * h))))
        return x0, y0, x1, y1

    def estimate(self, frame_rgb: np.ndarray, camera_key: str) -> FastDetectionResult:
        if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
            return FastDetectionResult(False, camera_key, message="invalid frame")

        frame_h, frame_w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
        roi_xyxy = self._roi_abs(frame_rgb.shape, camera_key)
        x0, y0, x1, y1 = roi_xyxy
        roi_np = np.ascontiguousarray(frame_rgb[y0:y1, x0:x1, :3])
        if roi_np.shape[0] < 16 or roi_np.shape[1] < 24:
            return FastDetectionResult(False, camera_key, roi_xyxy=roi_xyxy, message="roi too small")

        with torch.inference_mode():
            img = torch.from_numpy(roi_np).to(device=self.device, dtype=self.dtype, non_blocking=True)
            img = img.permute(2, 0, 1).unsqueeze(0) / 255.0
            gray = img[:, 0:1] * 0.299 + img[:, 1:2] * 0.587 + img[:, 2:3] * 0.114
            gray_sq = gray * gray
            roi_h, roi_w = int(gray.shape[-2]), int(gray.shape[-1])

            scores: List[torch.Tensor] = []
            indices: List[torch.Tensor] = []
            out_widths: List[int] = []
            eligible: List[_TemplateSpec] = []

            for spec in self.templates:
                if spec.height >= roi_h or spec.width >= roi_w:
                    continue
                ncc = self._ncc_map(gray, gray_sq, spec)
                flat = ncc.flatten()
                max_score, max_idx = torch.max(flat, dim=0)
                scores.append(max_score.float())
                indices.append(max_idx)
                out_widths.append(int(ncc.shape[-1]))
                eligible.append(spec)

            if not scores:
                return FastDetectionResult(False, camera_key, roi_xyxy=roi_xyxy, message="no template fits roi")

            stacked = torch.stack(scores)
            best_score_t, best_template_idx_t = torch.max(stacked, dim=0)
            best_template_idx = int(best_template_idx_t.item())
            spec = eligible[best_template_idx]
            flat_idx = int(indices[best_template_idx].item())
            out_w = out_widths[best_template_idx]
            yy = flat_idx // max(1, out_w)
            xx = flat_idx - yy * max(1, out_w)

            window = gray[0, 0, yy : yy + spec.height, xx : xx + spec.width].float()
            bright_ratio_t = (window > 0.62).float().mean()
            dark_ratio_t = (window < 0.35).float().mean()
            texture_t = window.std(unbiased=False)
            texture_score_t = torch.minimum(texture_t / self.min_texture, torch.ones_like(texture_t))
            mix_t = torch.minimum(bright_ratio_t / 0.35, torch.ones_like(bright_ratio_t))
            mix_t = mix_t * torch.minimum(dark_ratio_t / 0.10, torch.ones_like(dark_ratio_t)) * texture_score_t
            final_score_t = 0.82 * best_score_t.float() + 0.18 * mix_t.float()

            best_score = float(best_score_t.item())
            final_score = float(final_score_t.item())
            texture = float(texture_t.item())
            bright_ratio = float(bright_ratio_t.item())
            dark_ratio = float(dark_ratio_t.item())

        confidence = _clip((final_score - self.min_score) / max(0.001, 0.72 - self.min_score), 0.0, 1.0)
        ok = final_score >= self.min_score and texture >= self.min_texture * 0.55 and dark_ratio >= 0.035

        bx0 = x0 + xx
        by0 = y0 + yy
        bx1 = x0 + xx + spec.width
        by1 = y0 + yy + spec.height
        corners = np.asarray([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=np.float32)
        features = self._features_from_box(corners, roi_xyxy, (frame_h, frame_w), camera_key, spec.rotation_deg)
        raw_angle = self._fallback_angle(camera_key, features)

        return FastDetectionResult(
            ok=bool(ok),
            camera_key=camera_key,
            variant=spec.variant,
            confidence=float(confidence if ok else min(confidence, 0.17)),
            score=float(final_score),
            angle_deg=float(raw_angle) if ok else None,
            raw_angle_deg=float(raw_angle),
            roi_xyxy=roi_xyxy,
            corners=corners,
            features=features,
            message=(
                f"ncc={best_score:.3f} final={final_score:.3f} tex={texture:.3f} "
                f"b={bright_ratio:.2f} d={dark_ratio:.2f} rot={spec.rotation_deg:g}"
            ),
        )

    def _ncc_map(self, gray: torch.Tensor, gray_sq: torch.Tensor, spec: _TemplateSpec) -> torch.Tensor:
        numerator = F.conv2d(gray, spec.kernel)
        local_sum = F.conv2d(gray, spec.ones)
        local_sq_sum = F.conv2d(gray_sq, spec.ones)
        variance_sum = (local_sq_sum - (local_sum * local_sum) / spec.area).clamp_min(1e-5)
        denom = variance_sum.sqrt() * spec.norm
        return numerator / denom

    def _features_from_box(
        self,
        corners: np.ndarray,
        roi_xyxy: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        camera_key: str,
        roll_deg: float,
    ) -> Dict[str, float]:
        x0, y0, x1, y1 = roi_xyxy
        frame_h, frame_w = frame_shape
        rw = max(1.0, float(x1 - x0))
        rh = max(1.0, float(y1 - y0))
        center = corners.mean(axis=0)
        width = float(abs(corners[1, 0] - corners[0, 0]))
        height = float(abs(corners[3, 1] - corners[0, 1]))
        area_norm = (width * height) / max(1.0, rw * rh)
        cam_model = self.config.get("camera_model", {})
        fx = _as_float(cam_model.get("fx_px"), _as_float(cam_model.get("fx_ratio_of_width"), 0.92) * frame_w)
        bearing_deg = math.degrees(math.atan2(float(center[0] - frame_w * 0.5), max(1.0, fx)))
        return {
            "center_x": float((center[0] - x0) / rw),
            "center_y": float((center[1] - y0) / rh),
            "log_area": float(math.log(max(area_norm, 1e-6))),
            "area_norm": float(area_norm),
            "aspect": float(width / max(1e-6, height)),
            "width_ratio": 1.0,
            "height_ratio": 1.0,
            "roll_deg": float(roll_deg),
            "left_tilt_deg": 0.0,
            "right_tilt_deg": 0.0,
            "skew_deg": 0.0,
            "bearing_deg": float(bearing_deg),
            "pnp_yaw_deg": 0.0,
            "roi_x0": float(x0),
            "roi_y0": float(y0),
            "roi_w": float(rw),
            "roi_h": float(rh),
        }

    def _fallback_angle(self, camera_key: str, features: Dict[str, float]) -> float:
        camera_cfg = self.config.get("cameras", {}).get(camera_key, {})
        weights = camera_cfg.get("fallback_weights", {})
        sign = _as_float(camera_cfg.get("mirror_sign"), 1.0)
        raw = 0.0
        for name, weight in weights.items():
            raw += float(weight) * float(features.get(name, 0.0))
        clip_angle = _as_float(self.config.get("calibration", {}).get("clip_angle_deg"), 70.0)
        return _clip(sign * raw, -clip_angle, clip_angle)


def fuse_results(results: Iterable[FastDetectionResult], min_confidence: float = 0.18) -> FusedAngle:
    usable = [r for r in results if r.ok and r.angle_deg is not None and r.confidence >= min_confidence]
    if not usable:
        return FusedAngle(False, None, 0.0, "none", "no confident marker detection")
    if len(usable) == 1:
        r = usable[0]
        return FusedAngle(True, float(r.angle_deg), float(r.confidence), r.camera_key, r.model)
    weights = np.asarray([max(0.001, r.confidence) for r in usable], dtype=np.float64)
    angles = np.asarray([float(r.angle_deg) for r in usable], dtype=np.float64)
    fused = float(np.sum(weights * angles) / np.sum(weights))
    confidence = float(np.clip(np.mean(weights) + 0.12 * (len(usable) - 1), 0.0, 1.0))
    source = "+".join(r.camera_key for r in usable)
    return FusedAngle(True, fused, confidence, source, "confidence weighted")
