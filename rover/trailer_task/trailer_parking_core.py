#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - useful on minimal Jetson images.
    yaml = None

try:
    import serial
except Exception:  # pragma: no cover
    serial = None


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip3 install pyyaml")
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(base_dir: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def norm_angle_delta(angle_deg: float) -> float:
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def truthy_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class CameraROI:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    @classmethod
    def from_config(cls, data: Dict[str, Any]) -> "CameraROI":
        return cls(
            x=as_float(data.get("x"), 0.0),
            y=as_float(data.get("y"), 0.0),
            w=as_float(data.get("w"), 1.0),
            h=as_float(data.get("h"), 1.0),
        )

    def to_abs(self, frame_shape: Sequence[int]) -> Tuple[int, int, int, int]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x0 = int(round(self.x * width))
        y0 = int(round(self.y * height))
        x1 = int(round((self.x + self.w) * width))
        y1 = int(round((self.y + self.h) * height))
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        return x0, y0, x1, y1


@dataclass
class PanelDetection:
    ok: bool
    camera_key: str
    confidence: float = 0.0
    xyxy: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    image_size: Tuple[int, int] = (0, 0)
    class_id: int = -1
    class_name: str = ""
    timestamp: float = field(default_factory=time.monotonic)
    source: str = "yolo"

    @property
    def x0(self) -> float:
        return float(self.xyxy[0])

    @property
    def y0(self) -> float:
        return float(self.xyxy[1])

    @property
    def x1(self) -> float:
        return float(self.xyxy[2])

    @property
    def y1(self) -> float:
        return float(self.xyxy[3])

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    @property
    def image_w(self) -> int:
        return int(self.image_size[0])

    @property
    def image_h(self) -> int:
        return int(self.image_size[1])

    def features(self) -> Dict[str, float]:
        image_w = max(1.0, float(self.image_w))
        image_h = max(1.0, float(self.image_h))
        width_norm = self.width / image_w
        height_norm = self.height / image_h
        area_norm = width_norm * height_norm
        return {
            "det_conf": float(self.confidence),
            "width_norm": width_norm,
            "height_norm": height_norm,
            "center_x_norm": self.center_x / image_w,
            "center_y_norm": self.center_y / image_h,
            "area_norm": area_norm,
            "log_area": math.log(max(area_norm, 1e-9)),
            "aspect": self.width / max(1.0, self.height),
        }

    def clipped(self, margin_px: int) -> bool:
        if self.image_w <= 0 or self.image_h <= 0:
            return False
        margin = max(0, int(margin_px))
        return (
            self.x0 <= margin
            or self.y0 <= margin
            or self.x1 >= self.image_w - margin
            or self.y1 >= self.image_h - margin
        )


@dataclass
class AngleAnchor:
    camera_key: str
    angle_deg: float
    center_x_norm: float
    center_y_norm: float
    width_norm: float
    height_norm: float
    det_conf: float
    samples: int


@dataclass
class AngleEstimate:
    ok: bool
    camera_key: str
    angle_deg: Optional[float] = None
    confidence: float = 0.0
    source: str = "none"
    message: str = ""
    detection: Optional[PanelDetection] = None
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)

    def to_log_dict(self, prefix: str = "") -> Dict[str, Any]:
        p = prefix
        return {
            f"{p}ok": int(self.ok),
            f"{p}camera": self.camera_key,
            f"{p}angle_deg": "" if self.angle_deg is None else f"{self.angle_deg:.3f}",
            f"{p}confidence": f"{self.confidence:.4f}",
            f"{p}source": self.source,
            f"{p}message": self.message,
        }


@dataclass
class FusedAngleEstimate:
    ok: bool
    angle_deg: Optional[float]
    confidence: float
    source: str
    message: str = ""
    age_s: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
    measurements: List[AngleEstimate] = field(default_factory=list)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "angle_ok": int(self.ok),
            "angle_deg": "" if self.angle_deg is None else f"{self.angle_deg:.3f}",
            "angle_confidence": f"{self.confidence:.4f}",
            "angle_source": self.source,
            "angle_age_s": f"{self.age_s:.3f}",
            "angle_message": self.message,
        }


class LinearFeatureAngleModel:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).expanduser().resolve()
        self.feature_names: List[str] = []
        self.models: Dict[str, Dict[str, List[float]]] = {}
        self.available = False
        if self.model_path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.feature_names = list(data.get("feature_names") or [])
        for camera_key, model in (data.get("cameras") or {}).items():
            coef = [as_float(v) for v in model.get("coef") or []]
            mean = [as_float(v) for v in model.get("mean") or []]
            std = [max(1e-8, as_float(v, 1.0)) for v in model.get("std") or []]
            if len(coef) == len(self.feature_names) + 1 and len(mean) == len(self.feature_names) and len(std) == len(self.feature_names):
                self.models[str(camera_key)] = {"coef": coef, "mean": mean, "std": std}
        self.available = bool(self.models)

    def predict(self, camera_key: str, features: Dict[str, float]) -> Optional[float]:
        model = self.models.get(camera_key)
        if model is None:
            return None
        coef = model["coef"]
        mean = model["mean"]
        std = model["std"]
        value = coef[0]
        for idx, name in enumerate(self.feature_names):
            x = (float(features.get(name, 0.0)) - mean[idx]) / std[idx]
            value += coef[idx + 1] * x
        if not math.isfinite(value):
            return None
        return float(value)


class CenterTableAngleEstimator:
    def __init__(self, config: Dict[str, Any], config_dir: str | Path):
        self.config = config
        self.config_dir = Path(config_dir).expanduser().resolve()
        self.angle_cfg = config.get("angle", {}) or {}
        self.csv_path = resolve_path(self.config_dir, str(self.angle_cfg.get("calibration_csv", "")))
        self.min_detection_conf = as_float(self.angle_cfg.get("min_detection_conf"), 0.25)
        self.min_samples_per_anchor = max(1, as_int(self.angle_cfg.get("min_samples_per_anchor"), 5))
        self.reliable_min_abs_deg = as_float(self.angle_cfg.get("reliable_min_abs_deg"), 15.0)
        self.reliable_max_abs_deg = as_float(self.angle_cfg.get("reliable_max_abs_deg"), 58.0)
        self.extrapolate_margin_norm = as_float(self.angle_cfg.get("extrapolate_margin_norm"), 0.08)
        self.clip_margin_px = as_int(self.angle_cfg.get("clip_margin_px"), 3)
        self.clipped_conf_scale = as_float(self.angle_cfg.get("clipped_conf_scale"), 0.55)
        self.edge_extrapolated_conf_scale = as_float(self.angle_cfg.get("edge_extrapolated_conf_scale"), 0.45)
        table_cfg = self.angle_cfg.get("table", {}) or {}
        self.width_hint_weight = clamp(as_float(table_cfg.get("width_hint_weight"), 0.0), 0.0, 0.5)
        self.y_sanity_weight = clamp(as_float(table_cfg.get("y_sanity_weight"), 0.0), 0.0, 0.3)
        linear_path_value = str(self.angle_cfg.get("linear_model_json", "") or "")
        self.linear_weight = clamp(as_float(self.angle_cfg.get("linear_model_weight"), 0.0), 0.0, 0.8)
        self.linear_model: Optional[LinearFeatureAngleModel] = None
        if linear_path_value:
            self.linear_model = LinearFeatureAngleModel(resolve_path(self.config_dir, linear_path_value))
        self.anchors_by_camera = self._load_anchor_table()
        if not self.anchors_by_camera:
            raise RuntimeError(f"No usable angle anchors found in {self.csv_path}")

    def _load_anchor_table(self) -> Dict[str, List[AngleAnchor]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Angle calibration CSV not found: {self.csv_path}")
        grouped: Dict[Tuple[str, float], List[Dict[str, float]]] = {}
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                if not truthy_csv(raw.get("detected")):
                    continue
                conf = as_float(raw.get("det_conf"), -1.0)
                if conf < self.min_detection_conf:
                    continue
                camera_key = str(raw.get("camera_key", "")).strip()
                angle = as_float(raw.get("angle_deg"), float("nan"))
                if not camera_key or not math.isfinite(angle):
                    continue
                values = {
                    "center_x_norm": as_float(raw.get("center_x_norm"), float("nan")),
                    "center_y_norm": as_float(raw.get("center_y_norm"), float("nan")),
                    "width_norm": as_float(raw.get("width_norm"), float("nan")),
                    "height_norm": as_float(raw.get("height_norm"), float("nan")),
                    "det_conf": conf,
                }
                if not all(math.isfinite(v) for v in values.values()):
                    continue
                grouped.setdefault((camera_key, round(angle, 3)), []).append(values)

        out: Dict[str, List[AngleAnchor]] = {}
        for (camera_key, angle), rows in sorted(grouped.items()):
            if len(rows) < self.min_samples_per_anchor:
                continue
            anchor = AngleAnchor(
                camera_key=camera_key,
                angle_deg=float(angle),
                center_x_norm=float(median([r["center_x_norm"] for r in rows])),
                center_y_norm=float(median([r["center_y_norm"] for r in rows])),
                width_norm=float(median([r["width_norm"] for r in rows])),
                height_norm=float(median([r["height_norm"] for r in rows])),
                det_conf=float(median([r["det_conf"] for r in rows])),
                samples=len(rows),
            )
            out.setdefault(camera_key, []).append(anchor)
        for camera_key in list(out):
            out[camera_key] = sorted(out[camera_key], key=lambda a: a.center_x_norm)
            if len(out[camera_key]) < 2:
                del out[camera_key]
        return out

    def camera_keys(self) -> List[str]:
        return sorted(self.anchors_by_camera)

    def debug_anchor_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for camera_key, anchors in sorted(self.anchors_by_camera.items()):
            for anchor in anchors:
                rows.append(
                    {
                        "camera_key": camera_key,
                        "angle_deg": anchor.angle_deg,
                        "center_x_norm": anchor.center_x_norm,
                        "width_norm": anchor.width_norm,
                        "det_conf": anchor.det_conf,
                        "samples": anchor.samples,
                    }
                )
        return rows

    def estimate(self, detection: PanelDetection) -> AngleEstimate:
        now = detection.timestamp
        if not detection.ok:
            return AngleEstimate(False, detection.camera_key, source="no_detection", message="panel not detected", detection=detection, timestamp=now)
        anchors = self.anchors_by_camera.get(detection.camera_key)
        if not anchors:
            return AngleEstimate(False, detection.camera_key, source="no_table", message="no calibration table for camera", detection=detection, timestamp=now)

        features = detection.features()
        cx = features["center_x_norm"]
        angle, interp_quality, segment = self._interpolate_center_x(anchors, cx)
        width_angle, width_quality = self._width_hint_angle(anchors, features["width_norm"], angle)
        if width_angle is not None and self.width_hint_weight > 0.0:
            w = self.width_hint_weight * width_quality
            angle = (1.0 - w) * angle + w * width_angle
            interp_quality *= 0.85 + 0.15 * width_quality

        linear_angle = None
        if self.linear_model is not None and self.linear_model.available and self.linear_weight > 0.0:
            linear_angle = self.linear_model.predict(detection.camera_key, features)
            if linear_angle is not None:
                w = self.linear_weight
                angle = (1.0 - w) * angle + w * linear_angle

        conf = clamp(float(detection.confidence), 0.0, 1.0)
        conf *= clamp(interp_quality, 0.0, 1.0)
        conf *= self._feature_sanity(anchors, features)
        if detection.clipped(self.clip_margin_px):
            conf *= self.clipped_conf_scale

        abs_angle = abs(angle)
        if abs_angle < self.reliable_min_abs_deg:
            conf *= 0.55
        if abs_angle > self.reliable_max_abs_deg:
            over = min(1.0, (abs_angle - self.reliable_max_abs_deg) / 10.0)
            conf *= 1.0 - 0.55 * over

        source = "center_table"
        if width_angle is not None and self.width_hint_weight > 0.0:
            source += "+width"
        if linear_angle is not None and self.linear_weight > 0.0:
            source += "+linear"
        return AngleEstimate(
            ok=conf > 0.0 and math.isfinite(angle),
            camera_key=detection.camera_key,
            angle_deg=float(angle),
            confidence=clamp(conf, 0.0, 1.0),
            source=source,
            message=segment,
            detection=detection,
            features=features,
            timestamp=now,
        )

    def _interpolate_center_x(self, anchors: Sequence[AngleAnchor], cx: float) -> Tuple[float, float, str]:
        if cx <= anchors[0].center_x_norm:
            dist = anchors[0].center_x_norm - cx
            quality = self._edge_quality(dist)
            return anchors[0].angle_deg, quality, "left_edge"
        if cx >= anchors[-1].center_x_norm:
            dist = cx - anchors[-1].center_x_norm
            quality = self._edge_quality(dist)
            return anchors[-1].angle_deg, quality, "right_edge"
        for idx in range(len(anchors) - 1):
            left = anchors[idx]
            right = anchors[idx + 1]
            if left.center_x_norm <= cx <= right.center_x_norm:
                span = max(1e-6, right.center_x_norm - left.center_x_norm)
                t = (cx - left.center_x_norm) / span
                angle = left.angle_deg + t * (right.angle_deg - left.angle_deg)
                return angle, 1.0, f"segment:{left.angle_deg:.1f}->{right.angle_deg:.1f}"
        return anchors[-1].angle_deg, self.edge_extrapolated_conf_scale, "fallback_edge"

    def _edge_quality(self, distance_norm: float) -> float:
        if distance_norm <= 0.0:
            return 1.0
        margin = max(1e-6, self.extrapolate_margin_norm)
        return clamp(1.0 - distance_norm / margin, self.edge_extrapolated_conf_scale, 1.0)

    def _width_hint_angle(self, anchors: Sequence[AngleAnchor], width_norm: float, fallback_angle: float) -> Tuple[Optional[float], float]:
        candidates = sorted(anchors, key=lambda a: abs(a.width_norm - width_norm))[:2]
        if not candidates:
            return None, 0.0
        best = candidates[0]
        width_span = max(0.015, max(a.width_norm for a in anchors) - min(a.width_norm for a in anchors))
        quality = clamp(1.0 - abs(best.width_norm - width_norm) / width_span, 0.0, 1.0)
        if len(candidates) == 1 or abs(candidates[0].width_norm - candidates[1].width_norm) < 1e-6:
            return best.angle_deg, quality
        a0, a1 = candidates[0], candidates[1]
        denom = a1.width_norm - a0.width_norm
        t = clamp((width_norm - a0.width_norm) / denom, 0.0, 1.0) if abs(denom) > 1e-6 else 0.0
        angle = a0.angle_deg + t * (a1.angle_deg - a0.angle_deg)
        if abs(angle - fallback_angle) > 15.0:
            quality *= 0.45
        return angle, quality

    def _feature_sanity(self, anchors: Sequence[AngleAnchor], features: Dict[str, float]) -> float:
        score = 1.0
        if self.y_sanity_weight > 0.0:
            y_values = [a.center_y_norm for a in anchors]
            y_med = float(median(y_values))
            y_span = max(0.02, max(y_values) - min(y_values))
            y_penalty = clamp(abs(features["center_y_norm"] - y_med) / (3.0 * y_span), 0.0, 1.0)
            score *= 1.0 - self.y_sanity_weight * y_penalty
        width_min = min(a.width_norm for a in anchors) * 0.45
        width_max = max(a.width_norm for a in anchors) * 1.65
        if not (width_min <= features["width_norm"] <= width_max):
            score *= 0.60
        height_min = min(a.height_norm for a in anchors) * 0.45
        height_max = max(a.height_norm for a in anchors) * 1.65
        if not (height_min <= features["height_norm"] <= height_max):
            score *= 0.65
        return clamp(score, 0.0, 1.0)


class AngleStateFilter:
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("angle", {}) or {}
        self.alpha = clamp(as_float(cfg.get("filter_alpha"), 0.34), 0.0, 1.0)
        self.max_jump_deg = max(0.0, as_float(cfg.get("max_jump_deg"), 18.0))
        self.min_output_confidence = as_float(cfg.get("min_output_confidence"), 0.18)
        self.hold_timeout_s = as_float(cfg.get("hold_timeout_s"), 0.55)
        self.stale_timeout_s = as_float(cfg.get("stale_timeout_s"), 1.20)
        self.near_zero_abs_deg = as_float(cfg.get("near_zero_abs_deg"), 13.0)
        self.near_zero_confidence = as_float(cfg.get("near_zero_confidence"), 0.22)
        self.no_detection_zero_after_s = as_float(cfg.get("no_detection_zero_after_s"), 0.20)
        self.snap_to_measurement = as_bool(cfg.get("snap_to_measurement"), False)
        self.no_detection_policy = str(cfg.get("no_detection_policy", "hold_then_zero")).strip().lower()
        self.edge_missing_start_abs_deg = max(0.0, as_float(cfg.get("edge_missing_start_abs_deg"), 45.0))
        self.edge_missing_output_abs_deg = max(0.0, as_float(cfg.get("edge_missing_output_abs_deg"), 63.0))
        self.edge_missing_confidence = as_float(cfg.get("edge_missing_confidence"), 0.30)
        self.center_missing_confidence = as_float(cfg.get("center_missing_confidence"), self.near_zero_confidence)
        self.value: Optional[float] = None
        self.last_confidence = 0.0
        self.last_update_time: Optional[float] = None
        self.last_measurement_time: Optional[float] = None

    def update(self, measurements: Iterable[AngleEstimate], now: Optional[float] = None) -> FusedAngleEstimate:
        now = time.monotonic() if now is None else float(now)
        valid = [m for m in measurements if m.ok and m.angle_deg is not None and m.confidence >= self.min_output_confidence]
        if valid:
            fused_raw = self._fuse_measurements(valid)
            angle = float(fused_raw.angle_deg or 0.0)
            if self.snap_to_measurement or self.value is None:
                self.value = angle
            else:
                delta = norm_angle_delta(angle - self.value)
                if self.max_jump_deg > 0.0:
                    delta = clamp(delta, -self.max_jump_deg, self.max_jump_deg)
                self.value += self.alpha * delta
            self.last_confidence = fused_raw.confidence
            self.last_update_time = now
            self.last_measurement_time = now
            return FusedAngleEstimate(
                True,
                self.value,
                fused_raw.confidence,
                fused_raw.source,
                fused_raw.message,
                age_s=0.0,
                timestamp=now,
                measurements=list(valid),
            )

        age = float("inf") if self.last_measurement_time is None else now - self.last_measurement_time
        if self.no_detection_policy in {"edge_or_zero", "snap_edge_or_zero", "limit_or_zero"}:
            return self._missing_as_edge_or_zero(now, age)
        if self.value is not None and age <= self.hold_timeout_s:
            conf = self.last_confidence * clamp(1.0 - age / max(1e-6, self.hold_timeout_s), 0.18, 1.0)
            return FusedAngleEstimate(True, self.value, conf, "hold", "holding last mirror angle", age_s=age, timestamp=now)
        if self.value is None or abs(self.value) <= self.near_zero_abs_deg or age >= self.no_detection_zero_after_s:
            if self.value is None or abs(self.value) <= self.near_zero_abs_deg:
                self.value = 0.0
                self.last_update_time = now
                return FusedAngleEstimate(True, 0.0, self.near_zero_confidence, "near_zero_deadband", "panel invisible near straight angle", age_s=age, timestamp=now)
        if age <= self.stale_timeout_s and self.value is not None:
            return FusedAngleEstimate(False, self.value, 0.0, "stale", "angle is stale", age_s=age, timestamp=now)
        return FusedAngleEstimate(False, None, 0.0, "lost", "no reliable angle", age_s=age, timestamp=now)

    def _missing_as_edge_or_zero(self, now: float, age: float) -> FusedAngleEstimate:
        if self.value is not None and abs(self.value) >= self.edge_missing_start_abs_deg:
            sign = 1.0 if self.value >= 0.0 else -1.0
            self.value = sign * self.edge_missing_output_abs_deg
            self.last_update_time = now
            return FusedAngleEstimate(
                True,
                self.value,
                self.edge_missing_confidence,
                "edge_missing_limit",
                "panel invisible near angle limit",
                age_s=age,
                timestamp=now,
            )
        self.value = 0.0
        self.last_update_time = now
        return FusedAngleEstimate(
            True,
            0.0,
            self.center_missing_confidence,
            "center_missing_zero",
            "panel invisible away from angle limit",
            age_s=age,
            timestamp=now,
        )

    def _fuse_measurements(self, measurements: Sequence[AngleEstimate]) -> FusedAngleEstimate:
        if len(measurements) == 1:
            m = measurements[0]
            return FusedAngleEstimate(True, m.angle_deg, m.confidence, m.source, m.message, timestamp=m.timestamp, measurements=[m])
        total = sum(max(1e-6, m.confidence) for m in measurements)
        angle = sum(float(m.angle_deg or 0.0) * max(1e-6, m.confidence) for m in measurements) / total
        conf = clamp(total / len(measurements), 0.0, 1.0)
        source = "+".join(sorted({m.camera_key for m in measurements}))
        return FusedAngleEstimate(True, angle, conf, f"fused:{source}", "weighted mirror fusion", measurements=list(measurements))


@dataclass
class SlotEstimate:
    ok: bool = False
    confidence: float = 0.0
    center_x_norm: Optional[float] = None
    width_norm: Optional[float] = None
    source: str = "none"
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class ParkingCommand:
    speed: float
    steer: float
    brake: bool
    state: str
    target_angle_deg: Optional[float]
    reason: str
    active: bool = False

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "parking_state": self.state,
            "parking_speed": f"{self.speed:.4f}",
            "parking_steer": f"{self.steer:.4f}",
            "parking_brake": int(self.brake),
            "parking_target_angle_deg": "" if self.target_angle_deg is None else f"{self.target_angle_deg:.3f}",
            "parking_reason": self.reason,
        }


@dataclass
class WheelCommand:
    left: float
    right: float
    sent: bool
    reason: str

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "wheel_left": f"{self.left:.4f}",
            "wheel_right": f"{self.right:.4f}",
            "wheel_sent": int(self.sent),
            "wheel_reason": self.reason,
        }


class TrailerParkingController:
    STATES = ("IDLE", "BREAK_ANGLE", "HOLD_ANGLE", "CHASE_TRAILER", "STRAIGHTEN", "STOPPED", "FAULT")

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cfg = config.get("parking", {}) or {}
        self.safety_cfg = config.get("safety", {}) or {}
        self.state = "IDLE"
        self.start_time: Optional[float] = None
        self.phase_start_time: Optional[float] = None
        self.lost_since: Optional[float] = None
        self.last_angle: Optional[float] = None
        self.last_angle_time: Optional[float] = None
        self.last_error: Optional[float] = None
        self.last_error_time: Optional[float] = None
        self.fault_reason = ""

    def start(self, now: Optional[float] = None, side: Optional[str] = None) -> None:
        now = time.monotonic() if now is None else float(now)
        if side:
            self.cfg["start_side"] = side
        self.state = "BREAK_ANGLE"
        self.start_time = now
        self.phase_start_time = now
        self.lost_since = None
        self.last_error = None
        self.fault_reason = ""

    def stop(self, reason: str = "manual_stop") -> None:
        self.state = "STOPPED"
        self.fault_reason = reason
        self.phase_start_time = time.monotonic()

    def reset(self) -> None:
        self.state = "IDLE"
        self.start_time = None
        self.phase_start_time = None
        self.lost_since = None
        self.last_error = None
        self.fault_reason = ""

    def fault(self, reason: str) -> None:
        self.state = "FAULT"
        self.fault_reason = reason
        self.phase_start_time = time.monotonic()

    def update(
        self,
        angle: FusedAngleEstimate,
        slot: Optional[SlotEstimate] = None,
        now: Optional[float] = None,
    ) -> ParkingCommand:
        now = time.monotonic() if now is None else float(now)
        if self.state == "IDLE" and as_bool(self.cfg.get("auto_start"), False):
            self.start(now)

        if self.state in {"IDLE", "STOPPED", "FAULT"}:
            reason = self.fault_reason or self.state.lower()
            return ParkingCommand(0.0, 0.0, True, self.state, None, reason, active=False)

        max_time = as_float(self.cfg.get("max_parking_time_s"), 12.0)
        if self.start_time is not None and now - self.start_time > max_time:
            self.fault(f"parking_timeout>{max_time:.1f}s")
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.fault_reason, active=False)

        usable_angle = self._usable_angle(angle, now)
        if usable_angle is None:
            return ParkingCommand(0.0, 0.0, True, self.state, None, "angle_lost", active=False)

        hard_stop = as_float(self.safety_cfg.get("hard_stop_abs_angle_deg"), 66.0)
        if abs(usable_angle) >= hard_stop:
            self.fault(f"hard_angle_limit:{usable_angle:.1f}")
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.fault_reason, active=False)

        if self._slot_required(slot, now):
            return ParkingCommand(0.0, 0.0, True, self.state, None, "slot_required_or_lost", active=False)

        phase_elapsed = self._phase_elapsed(now)
        target = self._target_angle(phase_elapsed)
        speed = -abs(as_float(self.cfg.get("reverse_speed"), 0.34))
        if self.state == "STRAIGHTEN":
            speed = -abs(as_float(self.cfg.get("reverse_speed_slow"), 0.22))

        command = self._control_to_target(usable_angle, target, speed, now)
        self._advance_state(usable_angle, target, phase_elapsed, now)
        command.state = self.state if command.brake else command.state
        return command

    def _usable_angle(self, angle: FusedAngleEstimate, now: float) -> Optional[float]:
        min_conf = as_float(self.cfg.get("min_angle_confidence"), 0.16)
        grace = as_float(self.cfg.get("lost_angle_grace_s"), 0.65)
        if angle.ok and angle.angle_deg is not None and angle.confidence >= min_conf:
            self.last_angle = float(angle.angle_deg)
            self.last_angle_time = now
            self.lost_since = None
            return self.last_angle
        if self.lost_since is None:
            self.lost_since = now
        if self.last_angle is not None and now - self.lost_since <= grace:
            return self.last_angle
        if as_bool(self.safety_cfg.get("stop_on_angle_stale"), True):
            self.fault("angle_stale")
        return None

    def _slot_required(self, slot: Optional[SlotEstimate], now: float) -> bool:
        box_required_start = as_bool(self.cfg.get("require_box_for_start"), False)
        box_required_run = as_bool(self.cfg.get("require_box_during_parking"), False)
        if not box_required_start and not box_required_run:
            return False
        if slot is None or not slot.ok:
            return True
        if box_required_run:
            max_age = as_float(self.cfg.get("box_lost_grace_s"), 0.8)
            return now - slot.timestamp > max_age
        return False

    def _phase_elapsed(self, now: float) -> float:
        if self.phase_start_time is None:
            self.phase_start_time = now
            return 0.0
        return max(0.0, now - self.phase_start_time)

    def _target_sign(self) -> float:
        side = str(self.cfg.get("start_side", "right")).strip().lower()
        if side in {"left", "l", "cam1"}:
            return 1.0
        return -1.0

    def _target_angle(self, phase_elapsed: float) -> float:
        sign = self._target_sign()
        break_angle = abs(as_float(self.cfg.get("break_angle_deg"), 30.0))
        hold_angle = abs(as_float(self.cfg.get("hold_angle_deg"), 28.0))
        if self.state == "BREAK_ANGLE":
            return sign * break_angle
        if self.state == "HOLD_ANGLE":
            return sign * hold_angle
        if self.state == "CHASE_TRAILER":
            duration = max(0.1, as_float(self.cfg.get("chase_phase_timeout_s"), 5.0))
            start = sign * hold_angle
            end = as_float(self.cfg.get("straighten_target_deg"), 0.0)
            t = clamp(phase_elapsed / duration, 0.0, 1.0)
            return start + t * (end - start)
        return as_float(self.cfg.get("straighten_target_deg"), 0.0)

    def _control_to_target(self, angle_deg: float, target_deg: float, speed: float, now: float) -> ParkingCommand:
        error = norm_angle_delta(target_deg - angle_deg)
        kp = as_float(self.cfg.get("kp_angle"), 0.035)
        kd = as_float(self.cfg.get("kd_angle"), 0.006)
        derivative = 0.0
        if self.last_error is not None and self.last_error_time is not None:
            dt = max(1e-3, now - self.last_error_time)
            derivative = norm_angle_delta(error - self.last_error) / dt
        self.last_error = error
        self.last_error_time = now
        steer_sign = as_float(self.cfg.get("reverse_steer_sign"), 1.0)
        steer = steer_sign * (kp * error + kd * derivative)
        steer = clamp(steer, -abs(as_float(self.cfg.get("max_steer"), 0.82)), abs(as_float(self.cfg.get("max_steer"), 0.82)))
        return ParkingCommand(speed, steer, False, self.state, target_deg, f"angle_error={error:.1f}", active=True)

    def _advance_state(self, angle_deg: float, target_deg: float, phase_elapsed: float, now: float) -> None:
        tol = as_float(self.cfg.get("angle_tolerance_deg"), 4.0)
        if self.state == "BREAK_ANGLE":
            timeout = as_float(self.cfg.get("break_phase_timeout_s"), 4.5)
            if abs(norm_angle_delta(target_deg - angle_deg)) <= tol or phase_elapsed >= timeout:
                self.state = "HOLD_ANGLE"
                self.phase_start_time = now
                self.last_error = None
        elif self.state == "HOLD_ANGLE":
            hold_s = as_float(self.cfg.get("hold_phase_s"), 1.4)
            if phase_elapsed >= hold_s:
                self.state = "CHASE_TRAILER"
                self.phase_start_time = now
                self.last_error = None
        elif self.state == "CHASE_TRAILER":
            threshold = as_float(self.cfg.get("straighten_threshold_deg"), 7.0)
            timeout = as_float(self.cfg.get("chase_phase_timeout_s"), 5.0)
            if abs(angle_deg) <= threshold or phase_elapsed >= timeout:
                self.state = "STRAIGHTEN"
                self.phase_start_time = now
                self.last_error = None
        elif self.state == "STRAIGHTEN":
            final_s = as_float(self.cfg.get("straighten_phase_s"), 1.0)
            if phase_elapsed >= final_s:
                self.stop("parking_complete_scripted")


class DifferentialMixer:
    """Mix signed speed and virtual yaw/differential command into skid-steer PWM.

    The rover has no steering servo. `command.steer` is a normalized
    left/right wheel-speed differential: positive increases left-side PWM and
    decreases right-side PWM before optional inversion.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("rover", {}) or {}

    def mix(self, command: ParkingCommand) -> WheelCommand:
        max_wheel = abs(as_float(self.cfg.get("max_wheel_speed"), 0.11))
        if command.brake or not command.active or max_wheel <= 0.0:
            return WheelCommand(0.0, 0.0, False, f"brake:{command.reason}")
        speed_gain = as_float(self.cfg.get("speed_gain"), 1.0)
        base = clamp(command.speed * speed_gain, -1.0, 1.0) * max_wheel
        steer = clamp(command.steer, -1.0, 1.0)
        turn = steer * max_wheel * clamp(as_float(self.cfg.get("steer_mix"), 0.82), 0.0, 3.0)
        left = clamp(base + turn, -max_wheel, max_wheel)
        right = clamp(base - turn, -max_wheel, max_wheel)
        min_abs = max(0.0, as_float(self.cfg.get("min_abs_wheel_command"), 0.0))
        if min_abs > 0.0:
            left = self._apply_min_abs(left, min_abs, max_wheel)
            right = self._apply_min_abs(right, min_abs, max_wheel)
        if as_bool(self.cfg.get("invert_left"), False):
            left = -left
        if as_bool(self.cfg.get("invert_right"), False):
            right = -right
        sent = abs(left) > 1e-6 or abs(right) > 1e-6
        return WheelCommand(left, right, sent, command.reason)

    @staticmethod
    def _apply_min_abs(value: float, min_abs: float, max_abs: float) -> float:
        if abs(value) <= 1e-6:
            return 0.0
        return math.copysign(clamp(abs(value), min_abs, max_abs), value)


class RoverSerial:
    def __init__(self, serial_path: str, baud: int, armed: bool, write_timeout_s: float = 0.2) -> None:
        self.serial_path = serial_path
        self.baud = int(baud)
        self.armed = bool(armed)
        self.write_timeout_s = max(0.02, float(write_timeout_s))
        self._serial = None
        self._lock = threading.Lock()
        self.write_count = 0
        self.error_count = 0
        self.last_payload = ""
        self.last_error = ""
        self.last_write_monotonic = 0.0
        if not self.armed:
            print("[rover] dry-run: use --arm to send motor commands.")
            return
        if serial is None:
            raise RuntimeError("pyserial is required for armed rover output. Install with: pip3 install pyserial")
        self._serial = serial.Serial(self.serial_path, self.baud, timeout=0.0, write_timeout=self.write_timeout_s)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        print(f"[rover] ARMED: connected to {self.serial_path} at {self.baud} baud")

    def send(self, left: float, right: float) -> bool:
        if not self.armed:
            return False
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise RuntimeError("Rover serial port is not open.")
            payload = {"T": 1, "L": round(float(left), 3), "R": round(float(right), 3)}
            text = json.dumps(payload, separators=(",", ":"))
            try:
                self._serial.write((text + "\n").encode("ascii"))
                self.write_count += 1
                self.last_payload = text
                self.last_error = ""
                self.last_write_monotonic = time.monotonic()
            except Exception as exc:
                self.error_count += 1
                self.last_error = str(exc)
                raise
        return True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "armed": bool(self.armed),
                "serial_path": self.serial_path,
                "baud": int(self.baud),
                "write_timeout_s": float(self.write_timeout_s),
                "write_count": int(self.write_count),
                "error_count": int(self.error_count),
                "last_payload": self.last_payload,
                "last_error": self.last_error,
                "last_write_age_s": max(0.0, time.monotonic() - self.last_write_monotonic) if self.last_write_monotonic > 0.0 else None,
            }

    def stop(self) -> None:
        try:
            self.send(0.0, 0.0)
        except Exception as exc:
            print(f"[rover] warning: failed to send stop: {exc}")

    def close(self) -> None:
        self.stop()
        with self._lock:
            if self._serial is not None:
                self._serial.close()


class CommandRepeater:
    def __init__(self, rover: RoverSerial, rate_hz: float) -> None:
        self.rover = rover
        self.period_sec = 1.0 / max(1.0, float(rate_hz))
        self._left = 0.0
        self._right = 0.0
        self._active = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trailer-rover-command-repeater", daemon=True)
        self._last_warning_at = -999.0
        self._last_sent_inactive_zero = False

    def start(self) -> None:
        self._thread.start()

    def update(self, left: float, right: float, active: bool) -> bool:
        with self._lock:
            self._left = float(left) if active else 0.0
            self._right = float(right) if active else 0.0
            self._active = bool(active)
        self._wake.set()
        return self.rover.armed

    def stop(self) -> None:
        self.update(0.0, 0.0, False)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)

    def _snapshot(self) -> Tuple[float, float, bool]:
        with self._lock:
            return self._left, self._right, self._active

    def _run(self) -> None:
        while not self._stop.is_set():
            left, right, active = self._snapshot()
            try:
                if active:
                    self.rover.send(left, right)
                    self._last_sent_inactive_zero = False
                elif not self._last_sent_inactive_zero:
                    self.rover.send(0.0, 0.0)
                    self._last_sent_inactive_zero = True
                else:
                    self._wake.wait(self.period_sec)
                    self._wake.clear()
                    continue
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_warning_at >= 1.0:
                    print(f"[rover] warning: repeated command send failed: {exc}")
                    self._last_warning_at = now
            self._wake.wait(self.period_sec)
            self._wake.clear()


def make_repeater_if_enabled(config: Dict[str, Any], rover: RoverSerial) -> Optional[CommandRepeater]:
    cfg = config.get("rover", {}) or {}
    if not as_bool(cfg.get("repeat_last_command"), True):
        return None
    repeater = CommandRepeater(rover, as_float(cfg.get("command_rate_hz"), 20.0))
    repeater.start()
    return repeater


def draw_panel_overlay(frame_bgr: Any, roi_xyxy: Tuple[int, int, int, int], detection: Optional[PanelDetection], color: Tuple[int, int, int]) -> None:
    try:
        import cv2
    except Exception:
        return
    x0, y0, x1, y1 = roi_xyxy
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 1)
    if detection is None or not detection.ok:
        return
    dx0, dy0, dx1, dy1 = detection.xyxy
    p0 = (int(round(x0 + dx0)), int(round(y0 + dy0)))
    p1 = (int(round(x0 + dx1)), int(round(y0 + dy1)))
    cv2.rectangle(frame_bgr, p0, p1, color, 2)
    cv2.circle(frame_bgr, (int(round(x0 + detection.center_x)), int(round(y0 + detection.center_y))), 4, color, -1)


def draw_status_overlay(frame_bgr: Any, lines: Sequence[str], origin: Tuple[int, int] = (12, 24)) -> None:
    try:
        import cv2
    except Exception:
        return
    x, y = origin
    for idx, line in enumerate(lines):
        yy = y + idx * 22
        cv2.putText(frame_bgr, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
