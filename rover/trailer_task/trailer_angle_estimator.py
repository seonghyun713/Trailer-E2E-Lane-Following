from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - useful on minimal Jetson images.
    yaml = None


PointArray = np.ndarray


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for config files. Install with: pip3 install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    return math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))


def _wrap_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def _line_angle_around_horizontal(angle: float) -> float:
    angle = _wrap_angle_deg(angle)
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return angle


def _order_quad(points: PointArray) -> PointArray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _quad_area(points: PointArray) -> float:
    return float(abs(cv2.contourArea(np.asarray(points, dtype=np.float32).reshape(4, 2))))


def _quad_is_convex(points: PointArray) -> bool:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    return bool(cv2.isContourConvex(pts.astype(np.int32)))


def _composite_alpha_on_white(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    if img.shape[2] != 4:
        raise ValueError(f"Unsupported marker channel count: {img.shape}")
    bgr = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    out = bgr * alpha + 255.0 * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


@dataclass
class CameraROI:
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_config(cls, data: Dict[str, Any]) -> "CameraROI":
        return cls(
            x=_as_float(data.get("x"), 0.0),
            y=_as_float(data.get("y"), 0.0),
            w=_as_float(data.get("w"), 1.0),
            h=_as_float(data.get("h"), 1.0),
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
class TemplateVariant:
    name: str
    gray: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    corners: np.ndarray


@dataclass
class DetectionResult:
    ok: bool
    camera_key: str
    method: str = "none"
    variant: str = ""
    confidence: float = 0.0
    score: float = 0.0
    angle_deg: Optional[float] = None
    raw_angle_deg: Optional[float] = None
    model: str = "none"
    roi_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)
    corners: Optional[np.ndarray] = None
    features: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "camera_key": self.camera_key,
            "method": self.method,
            "variant": self.variant,
            "confidence": self.confidence,
            "score": self.score,
            "angle_deg": self.angle_deg,
            "raw_angle_deg": self.raw_angle_deg,
            "model": self.model,
            "roi_xyxy": list(self.roi_xyxy),
            "corners": None if self.corners is None else self.corners.round(2).tolist(),
            "features": {k: round(float(v), 5) for k, v in self.features.items()},
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class FusedAngle:
    ok: bool
    angle_deg: Optional[float]
    confidence: float
    source: str
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "angle_deg": self.angle_deg,
            "confidence": self.confidence,
            "source": self.source,
            "message": self.message,
        }


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
        delta = _wrap_angle_deg(angle - self.value)
        delta = _clip(delta, -self.max_jump_deg, self.max_jump_deg)
        self.value = self.value + self.alpha * delta
        return self.value


class LinearCalibrationModel:
    def __init__(
        self,
        config_dir: Path,
        csv_path: str,
        feature_names: Sequence[str],
        ridge_lambda: float,
        min_samples: int,
        clip_angle_deg: float,
    ):
        self.config_dir = config_dir
        self.csv_path = _resolve_path(config_dir, csv_path)
        self.feature_names = list(feature_names)
        self.ridge_lambda = float(ridge_lambda)
        self.min_samples = int(min_samples)
        self.clip_angle_deg = float(clip_angle_deg)
        self.weights_by_camera: Dict[str, np.ndarray] = {}
        self.sample_count_by_camera: Dict[str, int] = {}
        self._fit_if_available()

    def _fit_if_available(self) -> None:
        if not self.csv_path.exists():
            return
        rows: Dict[str, List[Dict[str, str]]] = {}
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                camera = row.get("camera_key", "")
                rows.setdefault(camera, []).append(row)

        for camera, camera_rows in rows.items():
            xs: List[List[float]] = []
            ys: List[float] = []
            for row in camera_rows:
                try:
                    y = float(row["angle_deg"])
                    x = [1.0] + [float(row.get(name, "nan")) for name in self.feature_names]
                except Exception:
                    continue
                if not np.all(np.isfinite(x)) or not math.isfinite(y):
                    continue
                xs.append(x)
                ys.append(y)
            self.sample_count_by_camera[camera] = len(xs)
            if len(xs) < self.min_samples:
                continue
            x_arr = np.asarray(xs, dtype=np.float64)
            y_arr = np.asarray(ys, dtype=np.float64)
            reg = self.ridge_lambda * np.eye(x_arr.shape[1], dtype=np.float64)
            reg[0, 0] = 0.0
            try:
                weights = np.linalg.solve(x_arr.T @ x_arr + reg, x_arr.T @ y_arr)
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(x_arr.T @ x_arr + reg) @ (x_arr.T @ y_arr)
            self.weights_by_camera[camera] = weights

    def available_for(self, camera_key: str) -> bool:
        return camera_key in self.weights_by_camera

    def predict(self, camera_key: str, features: Dict[str, float]) -> Optional[float]:
        weights = self.weights_by_camera.get(camera_key)
        if weights is None:
            return None
        x = np.asarray([1.0] + [float(features.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)
        angle = float(x @ weights)
        return _clip(angle, -self.clip_angle_deg, self.clip_angle_deg)

    def append_sample(self, camera_key: str, angle_deg: float, result: DetectionResult) -> Path:
        if not result.ok:
            raise ValueError("Cannot append calibration sample from a failed detection.")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["camera_key", "angle_deg", "method", "confidence", "timestamp"] + self.feature_names
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {
                "camera_key": camera_key,
                "angle_deg": float(angle_deg),
                "method": result.method,
                "confidence": float(result.confidence),
                "timestamp": time.time(),
            }
            for name in self.feature_names:
                row[name] = float(result.features.get(name, 0.0))
            writer.writerow(row)
        self._fit_if_available()
        return self.csv_path


class TrailerAngleEstimator:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config_dir = self.config_path.parent
        self.config = _read_yaml(self.config_path)
        detection_cfg = self.config.get("detection", {})
        self.feature_name, self.feature_detector, self.feature_norm = self._create_feature_detector(detection_cfg)
        self.matcher = cv2.BFMatcher(self.feature_norm, crossCheck=False)
        self.templates = self._load_templates()
        cal_cfg = self.config.get("calibration", {})
        self.calibration = LinearCalibrationModel(
            config_dir=self.config_dir,
            csv_path=str(cal_cfg.get("samples_csv", "calibration_samples.csv")),
            feature_names=cal_cfg.get("features", []),
            ridge_lambda=_as_float(cal_cfg.get("ridge_lambda"), 0.08),
            min_samples=int(cal_cfg.get("min_samples_per_camera", 5)),
            clip_angle_deg=_as_float(cal_cfg.get("clip_angle_deg"), 70.0),
        )

    def _create_feature_detector(self, detection_cfg: Dict[str, Any]):
        requested = str(detection_cfg.get("feature_detector", "sift")).lower()
        if requested == "sift" and hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create(nfeatures=int(detection_cfg.get("sift_features", 1800)))
            return "sift", detector, cv2.NORM_L2
        if requested in ("akaze", "sift") and hasattr(cv2, "AKAZE_create"):
            detector = cv2.AKAZE_create()
            return "akaze", detector, cv2.NORM_HAMMING
        detector = cv2.ORB_create(
            nfeatures=int(detection_cfg.get("orb_features", 2500)),
            fastThreshold=int(detection_cfg.get("orb_fast_threshold", 7)),
            edgeThreshold=8,
            patchSize=31,
        )
        return "orb", detector, cv2.NORM_HAMMING

    @property
    def camera_keys(self) -> List[str]:
        cameras = self.config.get("cameras", {})
        return [key for key, value in cameras.items() if value.get("enabled", True)]

    def _load_templates(self) -> List[TemplateVariant]:
        marker_cfg = self.config.get("marker", {})
        marker_path = _resolve_path(self.config_dir, str(marker_cfg.get("image_path", "../trailer_maker.png")))
        img = cv2.imread(str(marker_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Marker image not found: {marker_path}")
        img = _composite_alpha_on_white(img)
        variants = marker_cfg.get("variants", ["normal", "mirror_x"])
        max_width = int(marker_cfg.get("max_template_width_px", 900))
        prepared: List[TemplateVariant] = []

        for name in variants:
            var_img = img.copy()
            if name == "mirror_x":
                var_img = cv2.flip(var_img, 1)
            elif name == "mirror_y":
                var_img = cv2.flip(var_img, 0)
            elif name == "rotate_180":
                var_img = cv2.rotate(var_img, cv2.ROTATE_180)
            elif name != "normal":
                continue
            if var_img.shape[1] > max_width:
                scale = max_width / float(var_img.shape[1])
                var_img = cv2.resize(var_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(var_img, cv2.COLOR_BGR2GRAY)
            gray = self._preprocess_gray(gray)
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            h, w = gray.shape[:2]
            corners = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            prepared.append(TemplateVariant(name, gray, keypoints, descriptors, corners))
        if not prepared:
            raise RuntimeError("No marker template variants were prepared.")
        return prepared

    def _preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        if gray.ndim != 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        detection_cfg = self.config.get("detection", {})
        if bool(detection_cfg.get("clahe", True)):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        return gray

    def estimate(self, frame_bgr: np.ndarray, camera_key: str) -> DetectionResult:
        camera_cfg = self.config.get("cameras", {}).get(camera_key)
        if camera_cfg is None:
            raise KeyError(f"Unknown camera key: {camera_key}")
        roi = CameraROI.from_config(camera_cfg.get("roi", {}))
        x0, y0, x1, y1 = roi.to_abs(frame_bgr.shape)
        roi_bgr = frame_bgr[y0:y1, x0:x1]
        result = self._detect_in_roi(roi_bgr, camera_key, (x0, y0, x1, y1))
        if result.ok and result.corners is not None:
            result.corners = result.corners + np.asarray([x0, y0], dtype=np.float32)
            result.features = self._compute_features(result.corners, (x0, y0, x1, y1), frame_bgr.shape, camera_key)
            result.raw_angle_deg = self._fallback_angle(camera_key, result.features)
            calibrated = self.calibration.predict(camera_key, result.features)
            if calibrated is not None:
                result.angle_deg = calibrated
                result.model = "linear_calibration"
            else:
                result.angle_deg = result.raw_angle_deg
                result.model = "visual_proxy"
        return result

    def _detect_in_roi(
        self, roi_bgr: np.ndarray, camera_key: str, roi_xyxy: Tuple[int, int, int, int]
    ) -> DetectionResult:
        orb_result = self._detect_orb(roi_bgr, camera_key, roi_xyxy)
        if orb_result.ok:
            return orb_result

        detection_cfg = self.config.get("detection", {})
        if bool(detection_cfg.get("feature_cluster_fallback", True)):
            cluster_result = self._detect_feature_cluster(roi_bgr, camera_key, roi_xyxy)
            if cluster_result.ok:
                return cluster_result

        contour_enabled = bool(self.config.get("detection", {}).get("contour_fallback", True))
        if not contour_enabled:
            return orb_result
        contour_result = self._detect_contour(roi_bgr, camera_key, roi_xyxy)
        if contour_result.ok and contour_result.confidence > orb_result.confidence:
            return contour_result
        return orb_result

    def _detect_orb(
        self, roi_bgr: np.ndarray, camera_key: str, roi_xyxy: Tuple[int, int, int, int]
    ) -> DetectionResult:
        detection_cfg = self.config.get("detection", {})
        min_good = int(detection_cfg.get("min_good_matches", 18))
        min_inliers = int(detection_cfg.get("min_inliers", 12))
        match_ratio = _as_float(detection_cfg.get("match_ratio"), 0.76)
        ransac_thresh = _as_float(detection_cfg.get("ransac_reproj_threshold_px"), 4.0)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_gray(gray)
        kp, des = self.feature_detector.detectAndCompute(gray, None)
        if des is None or len(kp) < min_good:
            return DetectionResult(False, camera_key, method=self.feature_name, roi_xyxy=roi_xyxy, message="not enough ROI features")

        best: Optional[DetectionResult] = None
        best_corners: Optional[np.ndarray] = None
        for tmpl in self.templates:
            if tmpl.descriptors is None or len(tmpl.keypoints) < min_good:
                continue
            matches = self.matcher.knnMatch(tmpl.descriptors, des, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < match_ratio * n.distance:
                    good.append(m)
            if len(good) < min_good:
                continue
            src = np.float32([tmpl.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
            if homography is None or mask is None:
                continue
            inliers = int(mask.ravel().sum())
            if inliers < min_inliers:
                continue
            corners = cv2.perspectiveTransform(tmpl.corners.reshape(-1, 1, 2), homography).reshape(4, 2)
            corners = _order_quad(corners)
            valid, reason = self._validate_quad(corners, roi_bgr.shape)
            if not valid:
                continue
            area = _quad_area(corners)
            area_ratio = area / max(1.0, float(roi_bgr.shape[0] * roi_bgr.shape[1]))
            inlier_ratio = inliers / max(1, len(good))
            confidence = _clip(0.48 * inlier_ratio + 0.34 * min(inliers / 55.0, 1.0) + 0.18 * min(area_ratio / 0.08, 1.0), 0.0, 1.0)
            result = DetectionResult(
                ok=True,
                camera_key=camera_key,
                method=self.feature_name,
                variant=tmpl.name,
                confidence=confidence,
                score=confidence,
                roi_xyxy=roi_xyxy,
                corners=corners,
                message=f"good={len(good)} inliers={inliers}",
            )
            if best is None or result.confidence > best.confidence:
                best = result
                best_corners = corners
        if best is None:
            return DetectionResult(False, camera_key, method=self.feature_name, roi_xyxy=roi_xyxy, message="marker homography not found")
        best.corners = best_corners
        return best

    def _detect_feature_cluster(
        self, roi_bgr: np.ndarray, camera_key: str, roi_xyxy: Tuple[int, int, int, int]
    ) -> DetectionResult:
        detection_cfg = self.config.get("detection", {})
        cluster_cfg = detection_cfg.get("feature_cluster", {})
        match_ratio = _as_float(cluster_cfg.get("match_ratio"), 0.90)
        min_points = int(cluster_cfg.get("min_points", 12))
        min_affine_inliers = int(cluster_cfg.get("min_affine_inliers", 6))
        min_score = _as_float(cluster_cfg.get("min_score"), 0.72)
        min_width_ratio = _as_float(cluster_cfg.get("min_width_ratio"), 0.18)
        min_span_ratio = _as_float(cluster_cfg.get("min_span_ratio"), 0.18)
        max_y_ratio = _as_float(cluster_cfg.get("max_y_ratio"), 0.52)
        point_radius = int(cluster_cfg.get("point_radius_px", 13))
        patch_radius = int(cluster_cfg.get("local_patch_radius_px", 18))
        min_bright = _as_float(cluster_cfg.get("min_local_bright_ratio"), 0.10)
        min_std = _as_float(cluster_cfg.get("min_local_std"), 18.0)
        bright_threshold = int(cluster_cfg.get("bright_threshold", 92))
        dark_threshold = int(cluster_cfg.get("dark_threshold", 75))
        expand_x = _as_float(cluster_cfg.get("expand_x"), 1.22)
        expand_y = _as_float(cluster_cfg.get("expand_y"), 1.55)
        kernel_value = cluster_cfg.get("connect_kernel", [19, 9])
        if not isinstance(kernel_value, (list, tuple)) or len(kernel_value) != 2:
            kernel_value = [19, 9]
        connect_kernel = (max(1, int(kernel_value[0])), max(1, int(kernel_value[1])))

        roi_h, roi_w = roi_bgr.shape[:2]
        raw_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_gray(raw_gray)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < min_points:
            return DetectionResult(False, camera_key, method="feature_cluster", roi_xyxy=roi_xyxy, message="not enough ROI features")

        best: Optional[DetectionResult] = None
        for tmpl in self.templates:
            if tmpl.descriptors is None:
                continue
            matches = self.matcher.knnMatch(tmpl.descriptors, descriptors, k=2)
            match_records = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                match, neighbor = pair
                if match.distance >= match_ratio * neighbor.distance:
                    continue
                x, y = keypoints[match.trainIdx].pt
                if y > max_y_ratio * roi_h:
                    continue
                x0 = max(0, int(round(x)) - patch_radius)
                y0 = max(0, int(round(y)) - patch_radius)
                x1 = min(roi_w, int(round(x)) + patch_radius + 1)
                y1 = min(roi_h, int(round(y)) + patch_radius + 1)
                patch = raw_gray[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                if float(np.mean(patch > bright_threshold)) < min_bright or float(np.std(patch)) < min_std:
                    continue
                match_records.append((match, float(x), float(y)))

            if len(match_records) < min_points:
                continue

            point_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            for _, x, y in match_records:
                cv2.circle(point_mask, (int(round(x)), int(round(y))), point_radius, 255, -1)
            point_mask = cv2.dilate(point_mask, cv2.getStructuringElement(cv2.MORPH_RECT, connect_kernel), iterations=1)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(point_mask, 8)

            for label_idx in range(1, num_labels):
                x, y, width, height, _area = stats[label_idx]
                component_records = [
                    record
                    for record in match_records
                    if labels[int(round(record[2])), int(round(record[1]))] == label_idx
                ]
                point_count = len(component_records)
                if point_count < min_points:
                    continue
                if width < min_width_ratio * roi_w or height < 15:
                    continue
                aspect = width / max(1.0, float(height))
                if aspect < 1.0 or aspect > 8.0:
                    continue

                dst_points = np.asarray([[record[1], record[2]] for record in component_records], dtype=np.float32)
                src_points = np.asarray(
                    [tmpl.keypoints[record[0].queryIdx].pt for record in component_records],
                    dtype=np.float32,
                )
                dst_span = float((dst_points[:, 0].max() - dst_points[:, 0].min()) / max(1, roi_w))
                if dst_span < min_span_ratio:
                    continue

                affine, affine_mask = cv2.estimateAffinePartial2D(
                    src_points.reshape(-1, 1, 2),
                    dst_points.reshape(-1, 1, 2),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=8.0,
                    maxIters=2000,
                    confidence=0.97,
                )
                affine_inliers = int(affine_mask.ravel().sum()) if affine_mask is not None else 0
                if affine is None or affine_inliers < min_affine_inliers:
                    continue

                px0 = max(0, x - 8)
                py0 = max(0, y - 8)
                px1 = min(roi_w, x + width + 8)
                py1 = min(roi_h, y + height + 8)
                patch = raw_gray[py0:py1, px0:px1]
                bright_ratio = float(np.mean(patch > bright_threshold)) if patch.size else 0.0
                dark_ratio = float(np.mean(patch < dark_threshold)) if patch.size else 0.0
                contrast = float(np.std(patch) / 55.0) if patch.size else 0.0
                top_bonus = max(0.0, 1.0 - y / max(1.0, max_y_ratio * roi_h))
                width_score = min(width / max(1.0, 0.42 * roi_w), 1.0)
                point_score = min(point_count / 24.0, 1.0)
                inlier_score = min(affine_inliers / max(1.0, 0.65 * point_count), 1.0)
                brightness_score = min(bright_ratio / 0.35, 1.0)
                score = (
                    0.26 * point_score
                    + 0.20 * width_score
                    + 0.18 * inlier_score
                    + 0.14 * min(contrast, 1.0)
                    + 0.12 * top_bonus
                    + 0.10 * brightness_score
                )
                if score < min_score:
                    continue

                rect = cv2.minAreaRect(dst_points.astype(np.float32))
                center, size, angle = rect
                rect_w = max(float(size[0]), 1.0)
                rect_h = max(float(size[1]), 1.0)
                if rect_h > rect_w:
                    rect_w, rect_h = rect_h, rect_w
                    angle += 90.0
                rect_w = max(rect_w * expand_x, width * 0.95)
                rect_h = max(rect_h * expand_y, height * 0.80, 20.0)
                rotated_corners = _order_quad(cv2.boxPoints((center, (rect_w, rect_h), angle)))
                valid, _reason = self._validate_quad(rotated_corners, roi_bgr.shape)
                if valid:
                    corners = rotated_corners
                else:
                    pad_x = max(8, int(round(0.10 * width)))
                    pad_y = max(6, int(round(0.06 * height)))
                    ax0 = max(0, x - pad_x)
                    ay0 = max(0, y - pad_y)
                    ax1 = min(roi_w - 1, x + width + pad_x)
                    ay1 = min(roi_h - 1, y + height + pad_y)
                    corners = np.asarray(
                        [[ax0, ay0], [ax1, ay0], [ax1, ay1], [ax0, ay1]],
                        dtype=np.float32,
                    )
                    valid, _reason = self._validate_quad(corners, roi_bgr.shape)
                    if not valid:
                        continue

                confidence = _clip(0.35 + 0.55 * score + 0.10 * min(dst_span / 0.35, 1.0), 0.0, 0.92)
                result = DetectionResult(
                    ok=True,
                    camera_key=camera_key,
                    method="feature_cluster",
                    variant=tmpl.name,
                    confidence=confidence,
                    score=score,
                    roi_xyxy=roi_xyxy,
                    corners=corners,
                    message=(
                        f"points={point_count} affine_inliers={affine_inliers} "
                        f"span={dst_span:.2f} bright={bright_ratio:.2f} dark={dark_ratio:.2f}"
                    ),
                )
                if best is None or result.confidence > best.confidence:
                    best = result

        if best is None:
            return DetectionResult(False, camera_key, method="feature_cluster", roi_xyxy=roi_xyxy, message="marker feature cluster not found")
        return best

    def _detect_contour(
        self, roi_bgr: np.ndarray, camera_key: str, roi_xyxy: Tuple[int, int, int, int]
    ) -> DetectionResult:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_gray(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 45, 135)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[DetectionResult] = None
        best_corners: Optional[np.ndarray] = None
        roi_area = float(roi_bgr.shape[0] * roi_bgr.shape[1])
        for contour in contours:
            area = abs(cv2.contourArea(contour))
            if area < roi_area * float(self.config.get("detection", {}).get("min_area_ratio", 0.002)):
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
            if len(approx) != 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                corners = _order_quad(box)
            else:
                corners = _order_quad(approx.reshape(4, 2))
            valid, reason = self._validate_quad(corners, roi_bgr.shape)
            if not valid:
                continue
            x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
            patch = gray[max(y, 0) : min(y + h, gray.shape[0]), max(x, 0) : min(x + w, gray.shape[1])]
            contrast = float(np.std(patch)) / 80.0 if patch.size else 0.0
            area_ratio = _quad_area(corners) / max(1.0, roi_area)
            confidence = _clip(0.45 * min(area_ratio / 0.10, 1.0) + 0.35 * min(contrast, 1.0) + 0.20, 0.0, 0.72)
            result = DetectionResult(
                ok=True,
                camera_key=camera_key,
                method="contour",
                variant="rectangle",
                confidence=confidence,
                score=confidence,
                roi_xyxy=roi_xyxy,
                corners=corners,
                message="contour fallback",
            )
            if best is None or result.confidence > best.confidence:
                best = result
                best_corners = corners
        if best is None:
            return DetectionResult(False, camera_key, method="contour", roi_xyxy=roi_xyxy, message="marker rectangle not found")
        best.corners = best_corners
        return best

    def _validate_quad(self, corners: np.ndarray, roi_shape: Sequence[int]) -> Tuple[bool, str]:
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        h, w = int(roi_shape[0]), int(roi_shape[1])
        margin_x = 0.25 * w
        margin_y = 0.25 * h
        if np.any(corners[:, 0] < -margin_x) or np.any(corners[:, 0] > w + margin_x):
            return False, "quad outside x bounds"
        if np.any(corners[:, 1] < -margin_y) or np.any(corners[:, 1] > h + margin_y):
            return False, "quad outside y bounds"
        area = _quad_area(corners)
        area_ratio = area / max(1.0, float(w * h))
        det_cfg = self.config.get("detection", {})
        if area_ratio < float(det_cfg.get("min_area_ratio", 0.002)):
            return False, "quad too small"
        if area_ratio > float(det_cfg.get("max_area_ratio", 0.85)):
            return False, "quad too large"
        if not _quad_is_convex(corners):
            return False, "quad not convex"
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        mean_w = 0.5 * (top + bottom)
        mean_h = 0.5 * (left + right)
        if mean_w < 12.0 or mean_h < 8.0:
            return False, "quad edges too short"
        aspect = mean_w / max(mean_h, 1e-6)
        if aspect < float(det_cfg.get("min_aspect", 1.25)) or aspect > float(det_cfg.get("max_aspect", 5.2)):
            return False, f"aspect out of range: {aspect:.2f}"
        return True, "ok"

    def _compute_features(
        self,
        corners_full: np.ndarray,
        roi_xyxy: Tuple[int, int, int, int],
        frame_shape: Sequence[int],
        camera_key: str,
    ) -> Dict[str, float]:
        x0, y0, x1, y1 = roi_xyxy
        rw = max(1.0, float(x1 - x0))
        rh = max(1.0, float(y1 - y0))
        corners = _order_quad(corners_full)
        tl, tr, br, bl = corners
        center = corners.mean(axis=0)
        top = float(np.linalg.norm(tr - tl))
        bottom = float(np.linalg.norm(br - bl))
        left = float(np.linalg.norm(bl - tl))
        right = float(np.linalg.norm(br - tr))
        mean_w = 0.5 * (top + bottom)
        mean_h = 0.5 * (left + right)
        area = _quad_area(corners)
        top_angle = _line_angle_around_horizontal(_angle_deg(tl, tr))
        bottom_angle = _line_angle_around_horizontal(_angle_deg(bl, br))
        roll_deg = _line_angle_around_horizontal(0.5 * (top_angle + bottom_angle))
        left_tilt_deg = math.degrees(math.atan2(float(bl[0] - tl[0]), max(1.0, float(bl[1] - tl[1]))))
        right_tilt_deg = math.degrees(math.atan2(float(br[0] - tr[0]), max(1.0, float(br[1] - tr[1]))))
        skew_deg = _wrap_angle_deg(right_tilt_deg - left_tilt_deg)

        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        cam_model = self.config.get("camera_model", {})
        fx = _as_float(cam_model.get("fx_px"), _as_float(cam_model.get("fx_ratio_of_width"), 0.92) * frame_w)
        bearing_deg = math.degrees(math.atan2(float(center[0] - frame_w * 0.5), max(1.0, fx)))
        pnp_yaw = self._pnp_yaw_deg(corners, frame_shape)

        area_norm = area / max(1.0, rw * rh)
        return {
            "center_x": float((center[0] - x0) / rw),
            "center_y": float((center[1] - y0) / rh),
            "log_area": float(math.log(max(area_norm, 1e-6))),
            "area_norm": float(area_norm),
            "aspect": float(mean_w / max(mean_h, 1e-6)),
            "width_ratio": float(top / max(bottom, 1e-6)),
            "height_ratio": float(left / max(right, 1e-6)),
            "roll_deg": float(roll_deg),
            "left_tilt_deg": float(left_tilt_deg),
            "right_tilt_deg": float(right_tilt_deg),
            "skew_deg": float(skew_deg),
            "bearing_deg": float(bearing_deg),
            "pnp_yaw_deg": float(pnp_yaw),
            "roi_x0": float(x0),
            "roi_y0": float(y0),
            "roi_w": float(rw),
            "roi_h": float(rh),
        }

    def _pnp_yaw_deg(self, corners: np.ndarray, frame_shape: Sequence[int]) -> float:
        marker_cfg = self.config.get("marker", {})
        marker_w = _as_float(marker_cfg.get("physical_width_m"), 0.22)
        template = self.templates[0]
        marker_h = marker_w * (template.gray.shape[0] / max(1.0, float(template.gray.shape[1])))
        obj = np.asarray(
            [
                [-marker_w / 2, -marker_h / 2, 0.0],
                [marker_w / 2, -marker_h / 2, 0.0],
                [marker_w / 2, marker_h / 2, 0.0],
                [-marker_w / 2, marker_h / 2, 0.0],
            ],
            dtype=np.float32,
        )
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        cam_model = self.config.get("camera_model", {})
        fx = _as_float(cam_model.get("fx_px"), _as_float(cam_model.get("fx_ratio_of_width"), 0.92) * frame_w)
        fy = _as_float(cam_model.get("fy_px"), _as_float(cam_model.get("fy_ratio_of_width"), 0.92) * frame_w)
        cx = _as_float(cam_model.get("cx_px"), frame_w * 0.5)
        cy = _as_float(cam_model.get("cy_px"), frame_h * 0.5)
        camera_matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((5, 1), dtype=np.float64)
        try:
            ok, rvec, _tvec = cv2.solvePnP(obj, corners.astype(np.float32), camera_matrix, dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                ok, rvec, _tvec = cv2.solvePnP(obj, corners.astype(np.float32), camera_matrix, dist)
            if not ok:
                return 0.0
            rot, _ = cv2.Rodrigues(rvec)
            normal = rot @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            return math.degrees(math.atan2(float(normal[0]), max(1e-6, float(normal[2]))))
        except Exception:
            return 0.0

    def _fallback_angle(self, camera_key: str, features: Dict[str, float]) -> float:
        camera_cfg = self.config.get("cameras", {}).get(camera_key, {})
        weights = camera_cfg.get("fallback_weights", {})
        sign = _as_float(camera_cfg.get("mirror_sign"), 1.0)
        raw = 0.0
        for name, weight in weights.items():
            raw += float(weight) * float(features.get(name, 0.0))
        clip_angle = _as_float(self.config.get("calibration", {}).get("clip_angle_deg"), 70.0)
        return _clip(sign * raw, -clip_angle, clip_angle)

    def append_calibration_sample(self, camera_key: str, angle_deg: float, result: DetectionResult) -> Path:
        return self.calibration.append_sample(camera_key, angle_deg, result)


def fuse_results(results: Iterable[DetectionResult], min_confidence: float = 0.18) -> FusedAngle:
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


def draw_result(frame_bgr: np.ndarray, result: DetectionResult, color: Tuple[int, int, int] = (0, 230, 255)) -> np.ndarray:
    out = frame_bgr
    x0, y0, x1, y1 = result.roi_xyxy
    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 180, 255), 2)
    if result.corners is not None:
        pts = result.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
        for idx, p in enumerate(result.corners.astype(np.int32)):
            cv2.circle(out, tuple(p), 4, (0, 255, 0), -1)
            cv2.putText(out, str(idx), tuple(p + np.asarray([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if result.ok and result.angle_deg is not None:
        text = f"{result.camera_key}: {result.angle_deg:+5.1f} deg  conf={result.confidence:.2f}  {result.model}"
        text_color = (0, 255, 0)
    else:
        text = f"{result.camera_key}: no marker ({result.message})"
        text_color = (0, 80, 255)
    cv2.putText(out, text, (max(8, x0 + 8), max(28, y0 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, text_color, 2, cv2.LINE_AA)
    return out


def draw_fused(frame_bgr: np.ndarray, fused: FusedAngle, y: int = 34) -> np.ndarray:
    if fused.ok and fused.angle_deg is not None:
        text = f"TRAILER ANGLE {fused.angle_deg:+5.1f} deg  conf={fused.confidence:.2f}  src={fused.source}"
        color = (0, 255, 0)
    else:
        text = "TRAILER ANGLE unavailable"
        color = (0, 0, 255)
    cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    return frame_bgr


def result_json(results: Sequence[DetectionResult], fused: Optional[FusedAngle] = None) -> str:
    data: Dict[str, Any] = {"results": [r.to_dict() for r in results]}
    if fused is not None:
        data["fused"] = fused.to_dict()
    return json.dumps(data, ensure_ascii=False, indent=2)
