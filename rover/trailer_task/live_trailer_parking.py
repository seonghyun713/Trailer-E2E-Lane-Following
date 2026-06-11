#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trailer_parking_core import (  # noqa: E402
    AngleEstimate,
    AngleStateFilter,
    CameraROI,
    CenterTableAngleEstimator,
    DifferentialMixer,
    FusedAngleEstimate,
    PanelDetection,
    ParkingCommand,
    RoverSerial,
    SlotEstimate,
    TrailerParkingController,
    as_bool,
    as_float,
    as_int,
    draw_panel_overlay,
    draw_status_overlay,
    load_yaml,
    make_repeater_if_enabled,
    resolve_path,
)


_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual side-mirror trailer angle and reverse parking runner.")
    parser.add_argument("--config", type=Path, default=HERE / "trailer_parking_config.yaml")
    parser.add_argument("--panel-weights", default="", help="Override models.panel.weights.")
    parser.add_argument("--box-weights", default="", help="Override models.parking_box.weights and enable box detector.")
    parser.add_argument("--start-parking", action="store_true", help="Start the scripted parking FSM immediately.")
    parser.add_argument("--side", choices=("left", "right"), default="", help="Override parking.start_side.")
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--arm", dest="arm_override", action="store_true", default=None)
    arm_group.add_argument("--no-arm", dest="arm_override", action="store_false", default=None)
    parser.add_argument("--display", dest="display_override", action="store_true", default=None)
    parser.add_argument("--no-display", dest="display_override", action="store_false", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument("--single-camera", choices=("left", "right", "cam0", "cam1"), default="")
    parser.add_argument("--save-video", action="store_true")
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream_override", action="store_true", default=None)
    stream_group.add_argument("--no-stream", dest="stream_override", action="store_false", default=None)
    parser.add_argument("--udp-host", default="", help="Override stream.host.")
    parser.add_argument("--udp-port", type=int, default=0, help="Override stream.port.")
    parser.add_argument("--stream-codec", choices=("h264", "h265"), default="")
    parser.add_argument("--stream-width", type=int, default=0)
    parser.add_argument("--stream-height", type=int, default=0)
    parser.add_argument("--stream-fps", type=int, default=0)
    parser.add_argument("--stream-bitrate-kbps", type=int, default=0)
    return parser.parse_args()


def yolo_predict_kwargs(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "imgsz": int(model_cfg.get("imgsz", 320)),
        "conf": float(model_cfg.get("conf", 0.25)),
        "iou": float(model_cfg.get("iou", 0.55)),
        "verbose": False,
    }
    device = str(model_cfg.get("device", "auto")).strip().lower()
    if device and device != "auto":
        kwargs["device"] = device
    return kwargs


def open_gst_tools():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        from capture_bbox_frames_gst import GstCamera, fix_edge_color_cast

        Gst.init(None)
        return GstCamera, fix_edge_color_cast
    except Exception as exc:
        raise RuntimeError(f"Failed to import GStreamer camera helpers: {exc}") from exc


def open_camera(camera_key: str, cfg: Dict[str, Any], GstCamera) -> Any:
    root = cfg.get("camera", {}) or {}
    camera_cfg = (root.get("cameras") or {}).get(camera_key, {}) or {}
    sensor_id = as_int(camera_cfg.get("sensor_id"), 0)
    cam = GstCamera(
        sensor_id=sensor_id,
        capture_width=as_int(root.get("capture_width"), 1280),
        capture_height=as_int(root.get("capture_height"), 720),
        output_width=as_int(root.get("output_width"), 640),
        output_height=as_int(root.get("output_height"), 360),
        fps=as_int(root.get("fps"), 30),
        sink_format=str(root.get("appsink_format", "RGBA")),
        state_timeout_s=as_float(root.get("state_timeout_s"), 0.8),
    )
    first = cam.read(timeout_ms=as_int(root.get("first_frame_timeout_ms"), 650))
    if first is None:
        raise RuntimeError(f"{camera_key}/sensor{sensor_id}: first frame is None")
    print(f"[camera] {camera_key}/sensor{sensor_id} OK shape={first.shape}")
    return cam


def open_enabled_cameras(config: Dict[str, Any], keys: List[str], GstCamera) -> Dict[str, Any]:
    root = config.get("camera", {}) or {}
    if not as_bool(root.get("parallel_open"), len(keys) > 1) or len(keys) <= 1:
        cameras: Dict[str, Any] = {}
        for key in keys:
            try:
                cameras[key] = open_camera(key, config, GstCamera)
            except Exception as exc:
                print(f"[camera] {key} open failed: {exc}")
        return cameras

    cameras: Dict[str, Any] = {}
    lock = threading.Lock()

    def worker(key: str) -> None:
        try:
            cam = open_camera(key, config, GstCamera)
        except Exception as exc:
            print(f"[camera] {key} open failed: {exc}")
            return
        with lock:
            cameras[key] = cam

    threads = [threading.Thread(target=worker, args=(key,), name=f"open-{key}", daemon=True) for key in keys]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    missing = [key for key in keys if key not in cameras]
    if missing:
        print("[camera] retrying missing camera(s) sequentially")
        for key in missing:
            try:
                cameras[key] = open_camera(key, config, GstCamera)
            except Exception as exc:
                print(f"[camera] {key} open failed: {exc}")
    return {key: cameras[key] for key in keys if key in cameras}


def enabled_camera_keys(config: Dict[str, Any], single_camera: str = "") -> List[str]:
    cameras = (config.get("camera", {}) or {}).get("cameras", {}) or {}
    keys = [key for key, value in cameras.items() if as_bool((value or {}).get("enabled"), True)]
    alias = {"left": "cam1", "right": "cam0"}
    if single_camera:
        wanted = alias.get(single_camera, single_camera)
        keys = [key for key in keys if key == wanted]
    return keys


def camera_roi(config: Dict[str, Any], camera_key: str) -> CameraROI:
    data = ((config.get("camera", {}) or {}).get("cameras", {}) or {}).get(camera_key, {}) or {}
    return CameraROI.from_config(data.get("roi", {}) or {})


def apply_color_fix(frame_rgb: np.ndarray, config: Dict[str, Any], fix_edge_color_cast) -> np.ndarray:
    fix_cfg = ((config.get("camera", {}) or {}).get("edge_color_fix", {}) or {})
    return fix_edge_color_cast(
        frame_rgb,
        enabled=as_bool(fix_cfg.get("enabled"), True),
        width_ratio=as_float(fix_cfg.get("width_ratio"), 0.32),
        strength=as_float(fix_cfg.get("strength"), 0.75),
        green_recovery=as_float(fix_cfg.get("green_recovery"), 0.70),
    )


def rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return frame_rgb[:, :, ::-1].copy()


def best_panel_detection(model: Any, roi_rgb: np.ndarray, camera_key: str, model_cfg: Dict[str, Any], now: float) -> PanelDetection:
    if roi_rgb.size == 0:
        return PanelDetection(False, camera_key, timestamp=now, source="empty_roi")
    roi_bgr = rgb_to_bgr(roi_rgb)
    result = model.predict(source=roi_bgr, **yolo_predict_kwargs(model_cfg))[0]
    if result.boxes is None or len(result.boxes) == 0:
        return PanelDetection(False, camera_key, image_size=(roi_rgb.shape[1], roi_rgb.shape[0]), timestamp=now, source="yolo")
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else np.zeros(len(confs))
    idx = int(np.argmax(confs))
    names = getattr(result, "names", {}) or {}
    cls_id = int(classes[idx])
    class_name = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
    x0, y0, x1, y1 = [float(v) for v in xyxy[idx]]
    h, w = roi_rgb.shape[:2]
    x0 = max(0.0, min(float(w - 1), x0))
    y0 = max(0.0, min(float(h - 1), y0))
    x1 = max(x0 + 1.0, min(float(w), x1))
    y1 = max(y0 + 1.0, min(float(h), y1))
    return PanelDetection(
        True,
        camera_key,
        confidence=float(confs[idx]),
        xyxy=(x0, y0, x1, y1),
        image_size=(w, h),
        class_id=cls_id,
        class_name=class_name,
        timestamp=now,
        source="yolo",
    )


def _expanded_roi_abs(
    frame_shape: Tuple[int, ...],
    base_xyxy: Tuple[int, int, int, int],
    model_cfg: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x0, y0, x1, y1 = base_xyxy
    base_w = max(1, x1 - x0)
    base_h = max(1, y1 - y0)
    expand_x = max(0.0, as_float(model_cfg.get("roi_expand_x"), 0.45))
    expand_y = max(0.0, as_float(model_cfg.get("roi_expand_y"), 0.35))
    min_w = max(1.0, as_float(model_cfg.get("roi_min_width_px"), 360.0))
    min_h = max(1.0, as_float(model_cfg.get("roi_min_height_px"), 260.0))
    pad_x = max(base_w * expand_x, 0.5 * max(0.0, min_w - base_w))
    pad_y = max(base_h * expand_y, 0.5 * max(0.0, min_h - base_h))
    ex0 = max(0, int(round(x0 - pad_x)))
    ey0 = max(0, int(round(y0 - pad_y)))
    ex1 = min(w, int(round(x1 + pad_x)))
    ey1 = min(h, int(round(y1 + pad_y)))
    if ex1 <= ex0 or ey1 <= ey0:
        return base_xyxy
    return ex0, ey0, ex1, ey1


def panel_detection_roi_abs(
    frame_shape: Tuple[int, ...],
    base_xyxy: Tuple[int, int, int, int],
    model_cfg: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    if not as_bool(model_cfg.get("roi_expand_enabled"), True):
        return base_xyxy
    return _expanded_roi_abs(frame_shape, base_xyxy, model_cfg)


def best_panel_detection_in_frame(
    model: Any,
    frame_rgb: np.ndarray,
    camera_key: str,
    config: Dict[str, Any],
    model_cfg: Dict[str, Any],
    now: float,
) -> PanelDetection:
    bx0, by0, bx1, by1 = camera_roi(config, camera_key).to_abs(frame_rgb.shape)
    base_w = max(1, bx1 - bx0)
    base_h = max(1, by1 - by0)
    ex0, ey0, ex1, ey1 = panel_detection_roi_abs(frame_rgb.shape, (bx0, by0, bx1, by1), model_cfg)
    if (ex0, ey0, ex1, ey1) == (bx0, by0, bx1, by1):
        return best_panel_detection(model, frame_rgb[by0:by1, bx0:bx1], camera_key, model_cfg, now)

    det = best_panel_detection(model, frame_rgb[ey0:ey1, ex0:ex1], camera_key, model_cfg, now)
    if not det.ok:
        return PanelDetection(
            False,
            camera_key,
            image_size=(base_w, base_h),
            timestamp=now,
            source=f"{det.source}_expanded",
        )

    mapped_xyxy = (
        det.x0 + ex0 - bx0,
        det.y0 + ey0 - by0,
        det.x1 + ex0 - bx0,
        det.y1 + ey0 - by0,
    )
    return PanelDetection(
        True,
        camera_key,
        confidence=det.confidence,
        xyxy=mapped_xyxy,
        image_size=(base_w, base_h),
        class_id=det.class_id,
        class_name=det.class_name,
        timestamp=now,
        source="yolo_expanded",
    )


def enforce_single_panel_detection(
    detections: Dict[str, PanelDetection],
    estimates: Dict[str, AngleEstimate],
    config: Dict[str, Any],
    now: float,
) -> List[AngleEstimate]:
    angle_cfg = config.get("angle", {}) or {}
    if not as_bool(angle_cfg.get("mutual_exclusion_enabled"), True):
        return list(estimates.values())

    valid_keys = [
        key
        for key, det in detections.items()
        if det.ok and key in estimates and estimates[key].ok and estimates[key].angle_deg is not None
    ]
    if len(valid_keys) <= 1:
        return list(estimates.values())

    def score(key: str) -> float:
        est = estimates[key]
        det = detections[key]
        return max(float(est.confidence), float(det.confidence))

    best_key = max(valid_keys, key=score)
    for key in valid_keys:
        if key == best_key:
            continue
        old = detections[key]
        suppressed = PanelDetection(
            False,
            key,
            image_size=old.image_size,
            timestamp=old.timestamp,
            source=f"mutual_exclusion:{best_key}",
        )
        detections[key] = suppressed
        estimates[key] = AngleEstimate(
            False,
            key,
            source="mutual_exclusion",
            message=f"suppressed by {best_key}",
            detection=suppressed,
            timestamp=now,
        )
    return list(estimates.values())


class ParkingBoxDetector:
    def __init__(self, config: Dict[str, Any], config_dir: Path):
        self.cfg = ((config.get("models", {}) or {}).get("parking_box", {}) or {})
        self.enabled = as_bool(self.cfg.get("enabled"), False)
        weights_value = str(self.cfg.get("weights", "") or "")
        self.model = None
        if weights_value:
            self.weights = resolve_path(config_dir, weights_value)
        else:
            self.weights = Path("")
        if self.enabled:
            if not self.weights.exists():
                raise FileNotFoundError(f"parking_box.enabled=true but weights not found: {self.weights}")
            from ultralytics import YOLO

            if self.weights.suffix.lower() == ".engine":
                self.model = YOLO(str(self.weights), task="detect")
            else:
                self.model = YOLO(str(self.weights))
            print(f"[box] enabled weights={self.weights}")

    def detect(self, frame_bgr: np.ndarray, now: float) -> SlotEstimate:
        if not self.enabled or self.model is None:
            return SlotEstimate(False, source="disabled", timestamp=now)
        result = self.model.predict(source=frame_bgr, **yolo_predict_kwargs(self.cfg))[0]
        if result.boxes is None or len(result.boxes) < 2:
            return SlotEstimate(False, source="box_yolo", timestamp=now)
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        order = np.argsort(-confs)[:2]
        boxes = xyxy[order]
        h, w = frame_bgr.shape[:2]
        centers = 0.5 * (boxes[:, 0] + boxes[:, 2])
        widths = boxes[:, 2] - boxes[:, 0]
        return SlotEstimate(
            True,
            confidence=float(np.mean(confs[order])),
            center_x_norm=float(np.mean(centers) / max(1, w)),
            width_norm=float(np.mean(widths) / max(1, w)),
            source="box_yolo",
            timestamp=now,
        )


class UdpVideoStreamer:
    def __init__(self, stream_cfg: Dict[str, Any]):
        self.cfg = stream_cfg
        self.enabled = as_bool(stream_cfg.get("enabled"), False)
        self.writer = None
        self.frame_period_s = 1.0 / max(1.0, as_float(stream_cfg.get("fps"), 15.0))
        self.next_frame_at = 0.0
        self.width = max(160, as_int(stream_cfg.get("width"), 960))
        self.height = max(90, as_int(stream_cfg.get("height"), 270))
        self.host = str(stream_cfg.get("host", "127.0.0.1"))
        self.port = as_int(stream_cfg.get("port"), 5000)
        self.codec = str(stream_cfg.get("codec", "h264")).lower()
        self.encoder = str(stream_cfg.get("encoder", "auto")).lower()
        self.bitrate_kbps = max(100, as_int(stream_cfg.get("bitrate_kbps"), 1800))
        self.bitrate = self.bitrate_kbps * 1000
        if self.enabled:
            self._open()

    def _open(self) -> None:
        try:
            import cv2
        except Exception as exc:
            print(f"[stream] disabled: OpenCV import failed: {exc}")
            self.enabled = False
            return
        for label, pipeline in self._pipelines():
            writer = cv2.VideoWriter(
                pipeline,
                cv2.CAP_GSTREAMER,
                0,
                1.0 / self.frame_period_s,
                (self.width, self.height),
                True,
            )
            if writer.isOpened():
                self.writer = writer
                print(
                    f"[stream] udp://{self.host}:{self.port} {self.codec} "
                    f"{self.width}x{self.height}@{1.0 / self.frame_period_s:.0f}fps encoder={label}"
                )
                return
            writer.release()
            print(f"[stream] writer failed with encoder={label}")
        print("[stream] disabled: failed to open GStreamer UDP writer")
        self.enabled = False
        self.writer = None

    def _pipelines(self) -> List[Tuple[str, str]]:
        requested = self.encoder
        if requested in {"auto", ""}:
            order = ["nvv4l2", "x264"]
        elif requested in {"software", "sw", "x264"}:
            order = ["x264"]
        else:
            order = [requested]
            if requested == "nvv4l2":
                order.append("x264")
        pipelines = []
        for encoder in order:
            pipeline = self._pipeline(encoder)
            if pipeline:
                pipelines.append((encoder, pipeline))
        return pipelines

    def _pipeline(self, encoder_name: str) -> str:
        fps = max(1, int(round(1.0 / self.frame_period_s)))
        appsrc = (
            "appsrc is-live=true block=false do-timestamp=true format=time ! "
            f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={fps}/1 ! "
            "queue leaky=downstream max-size-buffers=2 ! "
        )
        if encoder_name in {"x264", "software", "sw"}:
            if self.codec in {"h265", "hevc"}:
                return ""
            return (
                appsrc +
                "videoconvert ! video/x-raw,format=I420 ! "
                f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={self.bitrate_kbps} "
                f"key-int-max={fps} byte-stream=true sliced-threads=true ! "
                "video/x-h264,stream-format=byte-stream,alignment=au ! "
                f"udpsink host={self.host} port={self.port} sync=false async=false"
            )
        if self.codec in {"h265", "hevc"}:
            encoder = (
                f"nvv4l2h265enc bitrate={self.bitrate} iframeinterval={fps} "
                "insert-sps-pps=true maxperf-enable=true"
            )
            parser = "h265parse"
        else:
            encoder = (
                f"nvv4l2h264enc bitrate={self.bitrate} iframeinterval={fps} "
                "insert-sps-pps=true maxperf-enable=true"
            )
            parser = "h264parse config-interval=1"
        return (
            appsrc +
            "videoconvert ! video/x-raw,format=BGRx ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"{encoder} ! {parser} ! mpegtsmux alignment=7 ! "
            f"udpsink host={self.host} port={self.port} sync=false async=false"
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if not self.enabled or self.writer is None:
            return
        now = time.monotonic()
        if now < self.next_frame_at:
            return
        self.next_frame_at = now + self.frame_period_s
        try:
            import cv2

            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.writer.write(frame_bgr)
        except Exception as exc:
            print(f"[stream] disabled after write failure: {exc}")
            self.close()
            self.enabled = False

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def make_blank_frame(shape: Tuple[int, int, int] = (360, 640, 3)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def combined_view(frames_bgr: Dict[str, np.ndarray], config: Dict[str, Any]) -> np.ndarray:
    cam1 = frames_bgr.get("cam1")
    cam0 = frames_bgr.get("cam0")
    if cam1 is None and cam0 is None:
        return make_blank_frame()
    template = cam1 if cam1 is not None else cam0
    blank = make_blank_frame(template.shape)
    left = cam1 if cam1 is not None else blank.copy()
    right = cam0 if cam0 is not None else blank.copy()
    try:
        import cv2

        cv2.putText(left, "cam1 left mirror", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(right, "cam0 right mirror", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    except Exception:
        pass
    return np.hstack([left, right])


def make_log_writer(run_dir: Path) -> Tuple[csv.DictWriter, Any]:
    csv_path = run_dir / "trailer_parking_log.csv"
    f = csv_path.open("w", newline="", encoding="utf-8")
    fields = [
        "frame_idx",
        "timestamp_monotonic",
        "wall_time",
        "cam0_det",
        "cam0_conf",
        "cam0_cx_norm",
        "cam0_w_norm",
        "cam0_angle_deg",
        "cam0_angle_conf",
        "cam1_det",
        "cam1_conf",
        "cam1_cx_norm",
        "cam1_w_norm",
        "cam1_angle_deg",
        "cam1_angle_conf",
        "angle_ok",
        "angle_deg",
        "angle_confidence",
        "angle_source",
        "angle_age_s",
        "angle_message",
        "parking_state",
        "parking_speed",
        "parking_steer",
        "parking_brake",
        "parking_target_angle_deg",
        "parking_reason",
        "wheel_left",
        "wheel_right",
        "wheel_sent",
        "wheel_reason",
        "slot_ok",
        "slot_confidence",
        "slot_center_x_norm",
        "slot_source",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    print(f"[log] {csv_path}")
    return writer, f


def detection_log_values(detection: Optional[PanelDetection], estimate: Optional[AngleEstimate], prefix: str) -> Dict[str, Any]:
    if detection is None or not detection.ok:
        return {
            f"{prefix}_det": 0,
            f"{prefix}_conf": "",
            f"{prefix}_cx_norm": "",
            f"{prefix}_w_norm": "",
            f"{prefix}_angle_deg": "",
            f"{prefix}_angle_conf": "",
        }
    features = detection.features()
    return {
        f"{prefix}_det": 1,
        f"{prefix}_conf": f"{detection.confidence:.4f}",
        f"{prefix}_cx_norm": f"{features['center_x_norm']:.5f}",
        f"{prefix}_w_norm": f"{features['width_norm']:.5f}",
        f"{prefix}_angle_deg": "" if estimate is None or estimate.angle_deg is None else f"{estimate.angle_deg:.3f}",
        f"{prefix}_angle_conf": "" if estimate is None else f"{estimate.confidence:.4f}",
    }


def write_log_row(
    writer: csv.DictWriter,
    frame_idx: int,
    detections: Dict[str, PanelDetection],
    estimates: Dict[str, AngleEstimate],
    fused: FusedAngleEstimate,
    parking: ParkingCommand,
    wheel: Any,
    slot: SlotEstimate,
) -> None:
    row: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "timestamp_monotonic": f"{time.monotonic():.3f}",
        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
    }
    row.update(detection_log_values(detections.get("cam0"), estimates.get("cam0"), "cam0"))
    row.update(detection_log_values(detections.get("cam1"), estimates.get("cam1"), "cam1"))
    row.update(fused.to_log_dict())
    row.update(parking.to_log_dict())
    row.update(wheel.to_log_dict())
    row.update(
        {
            "slot_ok": int(slot.ok),
            "slot_confidence": f"{slot.confidence:.4f}",
            "slot_center_x_norm": "" if slot.center_x_norm is None else f"{slot.center_x_norm:.5f}",
            "slot_source": slot.source,
        }
    )
    writer.writerow(row)


def save_snapshot(snapshot_dir: Path, frame_bgr: np.ndarray) -> Path:
    import cv2

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"trailer_parking_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    if args.panel_weights:
        config.setdefault("models", {}).setdefault("panel", {})["weights"] = args.panel_weights
    if args.box_weights:
        config.setdefault("models", {}).setdefault("parking_box", {})["weights"] = args.box_weights
        config.setdefault("models", {}).setdefault("parking_box", {})["enabled"] = True
    if args.side:
        config.setdefault("parking", {})["start_side"] = args.side
    if args.arm_override is not None:
        config.setdefault("rover", {})["arm"] = bool(args.arm_override)
    if args.display_override is not None:
        config.setdefault("runtime", {})["display"] = bool(args.display_override)
    if args.save_video:
        config.setdefault("runtime", {})["save_video"] = True
    if args.stream_override is not None:
        config.setdefault("stream", {})["enabled"] = bool(args.stream_override)
    if args.udp_host:
        config.setdefault("stream", {})["host"] = args.udp_host
    if args.udp_port > 0:
        config.setdefault("stream", {})["port"] = int(args.udp_port)
    if args.stream_codec:
        config.setdefault("stream", {})["codec"] = args.stream_codec
    if args.stream_width > 0:
        config.setdefault("stream", {})["width"] = int(args.stream_width)
    if args.stream_height > 0:
        config.setdefault("stream", {})["height"] = int(args.stream_height)
    if args.stream_fps > 0:
        config.setdefault("stream", {})["fps"] = int(args.stream_fps)
    if args.stream_bitrate_kbps > 0:
        config.setdefault("stream", {})["bitrate_kbps"] = int(args.stream_bitrate_kbps)

    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    panel_weights = resolve_path(config_dir, str(panel_cfg.get("weights", "")))
    if not panel_weights.exists():
        raise SystemExit(f"Panel YOLO weights not found: {panel_weights}")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is required. Install with: pip3 install ultralytics\n{exc}") from exc
    if panel_weights.suffix.lower() == ".engine":
        panel_model = YOLO(str(panel_weights), task="detect")
    else:
        panel_model = YOLO(str(panel_weights))
    print(f"[panel] weights={panel_weights}")

    angle_estimator = CenterTableAngleEstimator(config, config_dir)
    print("[angle] anchors:")
    for row in angle_estimator.debug_anchor_rows():
        print(
            "  {camera_key} angle={angle_deg:>6.1f} cx={center_x_norm:.3f} "
            "w={width_norm:.3f} conf={det_conf:.2f} n={samples}".format(**row)
        )
    angle_filter = AngleStateFilter(config)
    controller = TrailerParkingController(config)
    mixer = DifferentialMixer(config)
    box_detector = ParkingBoxDetector(config, config_dir)

    GstCamera, fix_edge_color_cast = open_gst_tools()
    keys = enabled_camera_keys(config, args.single_camera)
    cameras = open_enabled_cameras(config, keys, GstCamera)
    if not cameras:
        raise SystemExit("No camera is available.")

    rover_cfg = config.get("rover", {}) or {}
    rover = RoverSerial(
        str(rover_cfg.get("serial", "/dev/ttyUSB0")),
        as_int(rover_cfg.get("baud"), 115200),
        as_bool(rover_cfg.get("arm"), False),
    )
    repeater = make_repeater_if_enabled(config, rover)

    runtime_cfg = config.get("runtime", {}) or {}
    display = as_bool(runtime_cfg.get("display"), True)
    print_every = args.print_every if args.print_every > 0 else as_int(runtime_cfg.get("print_every"), 10)
    log_root = resolve_path(config_dir, str(runtime_cfg.get("log_dir", "logs/trailer_parking_run")))
    run_dir = log_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_writer, log_file = make_log_writer(run_dir)
    snapshot_dir = resolve_path(config_dir, str(runtime_cfg.get("snapshot_dir", "logs/trailer_parking_snapshots")))
    streamer = UdpVideoStreamer(config.get("stream", {}) or {})

    video_writer = None
    if as_bool(runtime_cfg.get("save_video"), False):
        import cv2

        video_path = run_dir / "trailer_parking_debug.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (1280, 360))
        print(f"[video] {video_path}")

    if args.start_parking:
        controller.start(side=args.side or None)
        print("[parking] started")

    detections: Dict[str, PanelDetection] = {}
    estimates: Dict[str, AngleEstimate] = {}
    frame_idx = 0
    last_slot = SlotEstimate(False, source="init")
    infer_every = max(1, as_int(panel_cfg.get("infer_every"), 2))

    try:
        while not _STOP:
            frame_idx += 1
            now = time.monotonic()
            frames_rgb: Dict[str, np.ndarray] = {}
            frames_bgr: Dict[str, np.ndarray] = {}
            for key, cam in cameras.items():
                frame = cam.read(timeout_ms=as_int((config.get("camera", {}) or {}).get("timeout_ms"), 90))
                if frame is None:
                    continue
                frame = apply_color_fix(frame, config, fix_edge_color_cast)
                frames_rgb[key] = frame
                frames_bgr[key] = rgb_to_bgr(frame)

            if not frames_rgb:
                time.sleep(0.01)
                continue

            current_measurements: List[AngleEstimate] = []
            if frame_idx % infer_every == 0:
                for key, frame_rgb in frames_rgb.items():
                    det = best_panel_detection_in_frame(panel_model, frame_rgb, key, config, panel_cfg, now)
                    detections[key] = det
                    estimates[key] = angle_estimator.estimate(det)
                current_measurements = enforce_single_panel_detection(detections, estimates, config, now)
            fused = angle_filter.update(current_measurements, now=now)

            view = combined_view(frames_bgr, config)
            if box_detector.enabled and frame_idx % infer_every == 0:
                last_slot = box_detector.detect(view, now)

            parking_command = controller.update(fused, last_slot, now=now)
            wheel = mixer.mix(parking_command)
            if repeater is not None:
                repeater.update(wheel.left, wheel.right, wheel.sent)
            else:
                if wheel.sent:
                    rover.send(wheel.left, wheel.right)
                else:
                    rover.send(0.0, 0.0)

            for key, frame_bgr in frames_bgr.items():
                roi_xyxy = camera_roi(config, key).to_abs(frame_bgr.shape)
                color = (0, 220, 255) if key == "cam1" else (0, 170, 255)
                draw_panel_overlay(frame_bgr, roi_xyxy, detections.get(key), color)
            view = combined_view(frames_bgr, config)
            angle_display = "--" if fused.angle_deg is None else f"{fused.angle_deg:5.1f} deg"
            target_display = "--" if parking_command.target_angle_deg is None else f"{parking_command.target_angle_deg:.1f}"
            status_lines = [
                f"angle: {angle_display} conf={fused.confidence:.2f} src={fused.source}",
                f"parking: {parking_command.state} target={target_display} diff={parking_command.steer:.2f} speed={parking_command.speed:.2f}",
                f"wheel: L={wheel.left:.3f} R={wheel.right:.3f} {'ARMED' if rover.armed else 'dry-run'}",
                "keys: p=start, x=stop, r=reset, s=snapshot, q=quit",
            ]
            draw_status_overlay(view, status_lines)
            streamer.write(view)

            write_log_row(log_writer, frame_idx, detections, estimates, fused, parking_command, wheel, last_slot)
            if frame_idx % 20 == 0:
                log_file.flush()
            if video_writer is not None:
                if view.shape[1] != 1280 or view.shape[0] != 360:
                    import cv2

                    view_video = cv2.resize(view, (1280, 360))
                else:
                    view_video = view
                video_writer.write(view_video)

            if print_every > 0 and frame_idx % print_every == 0:
                angle_text = "--" if fused.angle_deg is None else f"{fused.angle_deg:.1f}"
                print(
                    f"[{frame_idx:06d}] angle={angle_text} conf={fused.confidence:.2f} "
                    f"state={parking_command.state} diff={parking_command.steer:.2f} "
                    f"L={wheel.left:.3f} R={wheel.right:.3f} {wheel.reason}"
                )

            if display:
                try:
                    import cv2

                    scale = as_float(runtime_cfg.get("display_scale"), 1.0)
                    shown = view
                    if scale > 0.0 and abs(scale - 1.0) > 1e-3:
                        shown = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("trailer_parking", shown)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as exc:
                    print(f"[display] disabled: {exc}")
                    display = False
                    key = 255
                if key == ord("q") or key == 27:
                    break
                if key == ord("p"):
                    controller.start(side=args.side or None)
                    print("[parking] started")
                elif key == ord("x") or key == ord(" "):
                    controller.stop("manual_stop")
                    print("[parking] stopped")
                elif key == ord("r"):
                    controller.reset()
                    print("[parking] reset")
                elif key == ord("s"):
                    path = save_snapshot(snapshot_dir, view)
                    print(f"[snapshot] {path}")

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        if repeater is not None:
            repeater.stop()
        rover.close()
        for cam in cameras.values():
            try:
                cam.close()
            except Exception:
                pass
        log_file.flush()
        log_file.close()
        if video_writer is not None:
            video_writer.release()
        streamer.close()
        if display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
