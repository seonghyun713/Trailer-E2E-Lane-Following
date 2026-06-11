#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TRACK_SCRIPTS = REPO_ROOT / "track_riding" / "model_car_jetson" / "scripts"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if TRACK_SCRIPTS.exists() and str(TRACK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRACK_SCRIPTS))

from trailer_parking_core import (  # noqa: E402
    AngleEstimate,
    AngleStateFilter,
    CenterTableAngleEstimator,
    FusedAngleEstimate,
    PanelDetection,
    RoverSerial,
    as_bool,
    as_float,
    as_int,
    clamp,
    draw_panel_overlay,
    draw_status_overlay,
    load_yaml,
    make_repeater_if_enabled,
    resolve_path,
)
from live_trailer_parking import (  # noqa: E402
    UdpVideoStreamer,
    apply_color_fix,
    best_panel_detection,
    best_panel_detection_in_frame,
    camera_roi,
    enforce_single_panel_detection,
    open_gst_tools,
    panel_detection_roi_abs,
    rgb_to_bgr,
)
from trailer_route_controller import (  # noqa: E402
    LaneSignal,
    RouteAwareTrailerController,
    RouteControlDebug,
    TrailerAngleRateFilter,
    TrailerAngleState,
    corner_features_from_rows,
    lane_signal_from_estimate,
    lane_signal_from_results,
)
from lane_following_core import (  # noqa: E402
    BevConfig,
    LaneEstimate,
    LaneFollowerConfig,
    LaneFollowerState,
    estimate_lane,
    estimate_to_dict,
    draw_bev_debug,
    make_debug_panel,
    perspective_matrices,
    warp_to_bev,
)


_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


@dataclass
class WheelCommand:
    left: float
    right: float
    sent: bool
    reason: str


@dataclass
class LaneDriveCommand:
    active: bool
    valid: bool
    confidence: float
    state: str
    steer: float
    speed: float
    brake: bool
    reason: str


@dataclass
class CameraLaneResult:
    camera_key: str
    frame_bgr: np.ndarray
    crop_bgr: np.ndarray
    mask: np.ndarray
    bev_mask: np.ndarray
    estimate: LaneEstimate
    panel: np.ndarray


@dataclass
class DualBevCalibration:
    enabled: bool
    cameras: Dict[str, Dict[str, Any]]
    dst_points_ratio: Tuple[Tuple[float, float], ...]
    vehicle_center_x_bias: float
    merge_mode: str
    source: str = "disabled"
    point_order: Tuple[str, ...] = ("TL", "TR", "BR", "BL")
    drive_mode: str = "merged_mask"


class LaneWheelMixer:
    def __init__(self, config: Dict[str, Any]):
        self.rover_cfg = config.get("rover", {}) or {}
        self.safety_cfg = config.get("safety", {}) or {}
        self.corner_pivot_cfg = config.get("corner_pivot", {}) or {}
        self.lane_pivot_cfg = config.get("lane_pivot_assist", {}) or {}
        self.last_output_steer: Optional[float] = None
        self.last_valid_steer: Optional[float] = None
        self.last_valid_speed: float = 0.0
        self.lane_lost_since: Optional[float] = None
        self.assist_until: float = 0.0
        self.assist_cooldown_until: float = 0.0
        self.assist_direction: float = 0.0
        self.assist_start_steer: float = 0.0
        self.assist_blend: float = 0.0
        self.assist_turn_dir: float = 0.0
        self.last_corner_state_time: Optional[float] = None

    def mix(self, command: LaneDriveCommand, now: float) -> WheelCommand:
        if command.brake or not command.active:
            self._clear_lane_memory()
            return WheelCommand(0.0, 0.0, False, f"brake:{command.reason}")

        if command.state.startswith("CORNER_PIVOT"):
            self.last_corner_state_time = now
            self._clear_lane_memory()
            return self._pivot_wheel_from_values(command.steer, command.reason)

        min_conf = as_float(self.safety_cfg.get("min_drive_confidence"), as_float(command.confidence, 0.0))
        lane_fault = (not command.valid) or command.confidence < min_conf
        if lane_fault:
            hold = self._lane_lost_hold(command, now)
            if hold is not None:
                return hold
            self._clear_lane_memory()
            return WheelCommand(0.0, 0.0, False, f"stop:{command.state}:conf={command.confidence:.2f}")

        steer = self._limit_steer_delta(clamp(command.steer, -1.0, 1.0))
        speed = max(0.0, command.speed)
        assist = self._lane_pivot_assist(command, steer, speed, now)
        if assist is not None:
            self.lane_lost_since = None
            self.last_valid_steer = steer
            self.last_valid_speed = speed
            return assist
        wheel = self._wheel_from_values(steer, speed, command.reason)
        if wheel.sent:
            self.lane_lost_since = None
            self.last_valid_steer = steer
            self.last_valid_speed = speed
        else:
            self._clear_lane_memory()
        return wheel

    def _lane_lost_hold(self, command: LaneDriveCommand, now: float) -> Optional[WheelCommand]:
        grace_s = max(0.0, as_float(self.safety_cfg.get("lane_lost_grace_s"), 0.0))
        if grace_s <= 0.0 or self.last_valid_steer is None or self.last_valid_speed <= 0.0:
            return None
        if self.lane_lost_since is None:
            self.lane_lost_since = now
        elapsed = now - self.lane_lost_since
        if elapsed > grace_s:
            return None
        scale = max(0.0, as_float(self.safety_cfg.get("lane_lost_speed_scale"), 0.55))
        if scale <= 0.0:
            return None
        speed = self.last_valid_speed * scale
        min_speed = max(0.0, as_float(self.safety_cfg.get("lane_lost_min_speed"), 0.0))
        if min_speed > 0.0:
            speed = max(speed, min_speed)
        speed = min(speed, self.last_valid_speed)
        reason = f"lane_lost_hold:{command.state}:{elapsed:.2f}/{grace_s:.2f}"
        return self._wheel_from_values(self.last_valid_steer, speed, reason)

    def _limit_steer_delta(self, steer: float) -> float:
        max_delta = max(0.0, as_float(self.safety_cfg.get("max_steer_delta_per_frame"), 0.0))
        if max_delta > 0.0 and self.last_output_steer is not None:
            steer = clamp(steer, self.last_output_steer - max_delta, self.last_output_steer + max_delta)
        self.last_output_steer = steer
        return steer

    def _lane_pivot_assist(
        self,
        command: LaneDriveCommand,
        steer: float,
        speed: float,
        now: float,
    ) -> Optional[WheelCommand]:
        if not as_bool(self.lane_pivot_cfg.get("enabled"), False):
            self._clear_lane_pivot_assist()
            return None
        mode = str(self.lane_pivot_cfg.get("mode", "blended")).lower()
        if mode == "pulse":
            return self._lane_pivot_pulse_assist(command, steer, speed, now)
        return self._lane_pivot_blended_assist(command, steer, speed, now)

    def _lane_pivot_blended_assist(
        self,
        command: LaneDriveCommand,
        steer: float,
        speed: float,
        now: float,
    ) -> Optional[WheelCommand]:
        if speed <= 0.0:
            self.assist_blend = 0.0
            return None
        if not self._lane_pivot_state_allowed(command.state, now):
            self.assist_blend = 0.0
            return None
        min_conf = clamp(as_float(self.lane_pivot_cfg.get("min_confidence"), 0.35), 0.0, 1.0)
        if command.confidence < min_conf:
            self.assist_blend = 0.0
            return None

        start_abs = clamp(as_float(self.lane_pivot_cfg.get("start_abs_steer"), 0.06), 0.0, 1.0)
        full_abs = clamp(as_float(self.lane_pivot_cfg.get("full_abs_steer"), 0.55), start_abs + 1e-3, 1.0)
        steer_abs = abs(steer)
        smoothing = clamp(as_float(self.lane_pivot_cfg.get("blend_smoothing"), 0.45), 0.0, 0.98)
        if steer_abs < start_abs:
            self.assist_blend = 0.0
            self.assist_turn_dir = 0.0
            return None

        direction_sign = -1.0 if as_float(self.lane_pivot_cfg.get("direction_sign"), 1.0) < 0.0 else 1.0
        turn_dir = (1.0 if steer > 0.0 else -1.0) * direction_sign
        if self.assist_turn_dir != 0.0 and turn_dir != self.assist_turn_dir:
            self.assist_blend = 0.0
        self.assist_turn_dir = turn_dir

        raw_blend = clamp((steer_abs - start_abs) / max(1e-6, full_abs - start_abs), 0.0, 1.0)
        target_blend = raw_blend * raw_blend * (3.0 - 2.0 * raw_blend)
        self.assist_blend = smoothing * self.assist_blend + (1.0 - smoothing) * target_blend
        min_blend = clamp(as_float(self.lane_pivot_cfg.get("min_blend_to_send"), 0.03), 0.0, 0.50)
        if self.assist_blend < min_blend:
            return None

        base_pair = self._base_wheel_pair_from_values(steer, speed)
        if base_pair is None:
            return None
        max_wheel = abs(as_float(self.rover_cfg.get("max_wheel_speed"), 0.60))
        outer = clamp(as_float(self.lane_pivot_cfg.get("outer_pwm"), 0.62), 0.0, max_wheel)
        inner = clamp(as_float(self.lane_pivot_cfg.get("inner_reverse_pwm"), 0.30), 0.0, max_wheel)
        target_left = outer if turn_dir > 0.0 else -inner
        target_right = -inner if turn_dir > 0.0 else outer
        base_left, base_right = base_pair
        blend = self.assist_blend
        left = base_left * (1.0 - blend) + target_left * blend
        right = base_right * (1.0 - blend) + target_right * blend
        reason = (
            f"lane_pivot_assist:blend={blend:.2f} steer={steer:+.2f} "
            f"conf={command.confidence:.2f} state={command.state}"
        )
        return self._finalize_wheel_pair(left, right, reason)

    def _lane_pivot_pulse_assist(
        self,
        command: LaneDriveCommand,
        steer: float,
        speed: float,
        now: float,
    ) -> Optional[WheelCommand]:
        if not self._lane_pivot_state_allowed(command.state, now):
            return None
        if now < self.assist_until and self.assist_direction != 0.0:
            return self._lane_pivot_wheel_from_direction(
                self.assist_direction,
                f"lane_pivot_assist:pulse steer={self.assist_start_steer:+.2f} conf={command.confidence:.2f} state={command.state}",
            )
        if speed <= 0.0 or now < self.assist_cooldown_until:
            return None
        enter_abs_steer = clamp(as_float(self.lane_pivot_cfg.get("enter_abs_steer"), 0.40), 0.0, 1.0)
        if abs(steer) < enter_abs_steer:
            return None
        min_conf = clamp(as_float(self.lane_pivot_cfg.get("min_confidence"), 0.55), 0.0, 1.0)
        if command.confidence < min_conf:
            return None

        direction_sign = -1.0 if as_float(self.lane_pivot_cfg.get("direction_sign"), 1.0) < 0.0 else 1.0
        self.assist_direction = (1.0 if steer > 0.0 else -1.0) * direction_sign
        self.assist_start_steer = steer
        pulse_duration = max(0.0, as_float(self.lane_pivot_cfg.get("pulse_duration_s"), 0.12))
        cooldown = max(0.0, as_float(self.lane_pivot_cfg.get("cooldown_s"), 0.30))
        self.assist_until = now + pulse_duration
        self.assist_cooldown_until = self.assist_until + cooldown
        return self._lane_pivot_wheel_from_direction(
            self.assist_direction,
            f"lane_pivot_assist:start steer={steer:+.2f} conf={command.confidence:.2f} state={command.state}",
        )

    def _lane_pivot_state_allowed(self, state: str, now: float) -> bool:
        state_upper = str(state).upper()
        if state_upper.startswith("CORNER_PIVOT"):
            self.last_corner_state_time = now
            return False
        if "LOST" in state_upper or "DASHED_RECOVERY" in state_upper:
            return False
        is_corner = "CORNER" in state_upper
        if is_corner:
            self.last_corner_state_time = now
        if is_corner and not as_bool(self.lane_pivot_cfg.get("allow_in_corner"), True):
            return False
        post_corner_disable_s = max(0.0, as_float(self.lane_pivot_cfg.get("post_corner_disable_s"), 0.0))
        if not is_corner and post_corner_disable_s > 0.0 and self.last_corner_state_time is not None:
            if now - self.last_corner_state_time < post_corner_disable_s:
                return False
        if not is_corner and not as_bool(self.lane_pivot_cfg.get("allow_in_straight"), True):
            return False
        return True

    def _lane_pivot_wheel_from_direction(self, direction: float, reason: str) -> WheelCommand:
        max_wheel = abs(as_float(self.rover_cfg.get("max_wheel_speed"), 0.60))
        if max_wheel <= 0.0:
            return WheelCommand(0.0, 0.0, False, f"stop:{reason}:max_wheel_zero")
        outer = clamp(as_float(self.lane_pivot_cfg.get("outer_pwm"), 0.75), 0.0, max_wheel)
        inner = clamp(as_float(self.lane_pivot_cfg.get("inner_reverse_pwm"), 0.35), 0.0, max_wheel)
        if outer <= 0.0 and inner <= 0.0:
            return WheelCommand(0.0, 0.0, False, f"stop:{reason}:assist_pwm_zero")

        turn_dir = 1.0 if direction >= 0.0 else -1.0
        left = outer if turn_dir > 0.0 else -inner
        right = -inner if turn_dir > 0.0 else outer
        min_abs = max(0.0, as_float(self.rover_cfg.get("min_abs_wheel_command"), 0.0))
        if min_abs > 0.0:
            left = self._apply_min_abs(left, min_abs, max_wheel)
            right = self._apply_min_abs(right, min_abs, max_wheel)
        if as_bool(self.rover_cfg.get("invert_left"), False):
            left = -left
        if as_bool(self.rover_cfg.get("invert_right"), False):
            right = -right
        return WheelCommand(left, right, True, reason)

    def _wheel_from_values(self, steer: float, speed: float, reason: str) -> WheelCommand:
        pair = self._base_wheel_pair_from_values(steer, speed)
        if pair is None:
            return WheelCommand(0.0, 0.0, False, f"stop:{reason}")
        left, right = pair
        return self._finalize_wheel_pair(left, right, reason)

    def _base_wheel_pair_from_values(self, steer: float, speed: float) -> Optional[Tuple[float, float]]:
        max_wheel = abs(as_float(self.rover_cfg.get("max_wheel_speed"), 0.60))
        forward = clamp(speed * as_float(self.rover_cfg.get("speed_gain"), 1.0), 0.0, max_wheel)
        if forward <= 0.0 or max_wheel <= 0.0:
            return None

        min_forward = clamp(as_float(self.rover_cfg.get("min_forward_speed"), 0.0), 0.0, max_wheel)
        turn_threshold = clamp(as_float(self.rover_cfg.get("turn_in_place_threshold"), 1.0), 0.0, 1.0)
        if min_forward > 0.0 and abs(steer) < turn_threshold:
            forward = max(forward, min_forward)

        turn = steer * max_wheel * clamp(as_float(self.rover_cfg.get("steer_mix"), 1.0), 0.0, 3.0)
        if turn_threshold < 1.0 and abs(steer) >= turn_threshold:
            forward *= max(0.0, 1.0 - abs(steer))

        min_inner_ratio = clamp(as_float(self.rover_cfg.get("min_inner_wheel_ratio"), 0.0), 0.0, 0.95)
        if min_inner_ratio > 0.0 and forward > 0.0:
            turn_limit = forward * (1.0 - min_inner_ratio)
            turn = clamp(turn, -turn_limit, turn_limit)

        left = clamp(forward + turn, -max_wheel, max_wheel)
        right = clamp(forward - turn, -max_wheel, max_wheel)
        return left, right

    def _finalize_wheel_pair(self, left: float, right: float, reason: str) -> WheelCommand:
        max_wheel = abs(as_float(self.rover_cfg.get("max_wheel_speed"), 0.60))
        min_abs = max(0.0, as_float(self.rover_cfg.get("min_abs_wheel_command"), 0.0))
        if min_abs > 0.0:
            left = self._apply_min_abs(left, min_abs, max_wheel)
            right = self._apply_min_abs(right, min_abs, max_wheel)
        if as_bool(self.rover_cfg.get("invert_left"), False):
            left = -left
        if as_bool(self.rover_cfg.get("invert_right"), False):
            right = -right
        return WheelCommand(left, right, True, reason)

    def _pivot_wheel_from_values(self, steer: float, reason: str) -> WheelCommand:
        max_wheel = abs(as_float(self.rover_cfg.get("max_wheel_speed"), 0.60))
        pwm = clamp(as_float(self.corner_pivot_cfg.get("pivot_pwm"), 0.55), 0.0, max_wheel)
        if pwm <= 0.0 or max_wheel <= 0.0:
            return WheelCommand(0.0, 0.0, False, f"stop:{reason}:pivot_pwm_zero")
        direction = -1.0 if steer < 0.0 else 1.0
        left = pwm * direction
        right = -pwm * direction
        min_abs = max(0.0, as_float(self.rover_cfg.get("min_abs_wheel_command"), 0.0))
        if min_abs > 0.0:
            left = self._apply_min_abs(left, min_abs, max_wheel)
            right = self._apply_min_abs(right, min_abs, max_wheel)
        if as_bool(self.rover_cfg.get("invert_left"), False):
            left = -left
        if as_bool(self.rover_cfg.get("invert_right"), False):
            right = -right
        return WheelCommand(left, right, True, f"corner_pivot:{reason}")

    @staticmethod
    def _apply_min_abs(value: float, min_abs: float, max_abs: float) -> float:
        if abs(value) <= 1e-6:
            return 0.0
        return float(np.copysign(clamp(abs(value), min_abs, max_abs), value))

    def _clear_lane_memory(self) -> None:
        self.last_output_steer = None
        self.last_valid_steer = None
        self.last_valid_speed = 0.0
        self.lane_lost_since = None
        self._clear_lane_pivot_assist()

    def _clear_lane_pivot_assist(self) -> None:
        self.assist_until = 0.0
        self.assist_cooldown_until = 0.0
        self.assist_direction = 0.0
        self.assist_start_steer = 0.0
        self.assist_blend = 0.0
        self.assist_turn_dir = 0.0


def enabled_camera_configs(config: Dict[str, Any], single_camera: str = "") -> Dict[str, Dict[str, Any]]:
    root = config.get("camera", {}) or {}
    cameras = root.get("cameras", {}) or {}
    alias = {"left": "cam1", "right": "cam0"}
    wanted = alias.get(single_camera, single_camera)
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in cameras.items():
        cfg = value or {}
        if not as_bool(cfg.get("enabled"), True):
            continue
        if wanted and key != wanted:
            continue
        out[key] = cfg
    return out


class CsiFrameSource:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        root = config.get("camera", {}) or {}
        GstCamera, fix_edge_color_cast = open_gst_tools()
        self.GstCamera = GstCamera
        self.fix_edge_color_cast = fix_edge_color_cast
        self.cameras: Dict[str, Any] = {}
        camera_cfgs = enabled_camera_configs(config, str(root.get("single_camera", "")))
        parallel_open = as_bool(root.get("parallel_open"), len(camera_cfgs) > 1)
        if parallel_open and len(camera_cfgs) > 1:
            self._open_cameras_parallel(root, camera_cfgs)
            missing_cfgs = {key: cfg for key, cfg in camera_cfgs.items() if key not in self.cameras}
            if missing_cfgs:
                print("[camera] retrying missing camera(s) sequentially")
                for camera_key, camera_cfg in missing_cfgs.items():
                    opened = self._open_camera(root, camera_key, camera_cfg)
                    if opened is not None:
                        cam, first = opened
                        self.cameras[camera_key] = cam
                        print(f"[camera] {camera_key}/sensor{cam.sensor_id} OK shape={first.shape}")
        else:
            for camera_key, camera_cfg in camera_cfgs.items():
                opened = self._open_camera(root, camera_key, camera_cfg)
                if opened is not None:
                    cam, first = opened
                    self.cameras[camera_key] = cam
                    print(f"[camera] {camera_key}/sensor{cam.sensor_id} OK shape={first.shape}")
                    open_delay = as_float(root.get("open_delay_s"), 0.0)
                    if open_delay > 0.0:
                        time.sleep(open_delay)
        if not self.cameras:
            raise RuntimeError("No CSI camera is available.")

    def _open_camera(
        self,
        root: Dict[str, Any],
        camera_key: str,
        camera_cfg: Dict[str, Any],
    ) -> Optional[Tuple[Any, np.ndarray]]:
        sensor_id = as_int(camera_cfg.get("sensor_id"), as_int(root.get("sensor_id"), 0))
        try:
            cam = self.GstCamera(
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
                print(f"[camera] {camera_key}/sensor{sensor_id} open failed: first frame is None")
                try:
                    cam.close()
                except Exception:
                    pass
                return None
            return cam, first
        except Exception as exc:
            print(f"[camera] {camera_key}/sensor{sensor_id} open failed: {exc}")
            return None

    def _open_cameras_parallel(self, root: Dict[str, Any], camera_cfgs: Dict[str, Dict[str, Any]]) -> None:
        results: Dict[str, Tuple[Any, np.ndarray]] = {}
        lock = threading.Lock()

        def worker(camera_key: str, camera_cfg: Dict[str, Any]) -> None:
            opened = self._open_camera(root, camera_key, camera_cfg)
            if opened is not None:
                with lock:
                    results[camera_key] = opened

        threads = [
            threading.Thread(target=worker, args=(camera_key, camera_cfg), name=f"open-{camera_key}", daemon=True)
            for camera_key, camera_cfg in camera_cfgs.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for camera_key, camera_cfg in camera_cfgs.items():
            opened = results.get(camera_key)
            if opened is None:
                continue
            cam, first = opened
            self.cameras[camera_key] = cam
            print(f"[camera] {camera_key}/sensor{cam.sensor_id} OK shape={first.shape}")

    def read(self) -> Dict[str, np.ndarray]:
        root = self.config.get("camera", {}) or {}
        timeout_ms = as_int((self.config.get("camera", {}) or {}).get("timeout_ms"), 90)
        parallel_read = as_bool(root.get("parallel_read"), len(self.cameras) > 1)
        frames: Dict[str, np.ndarray] = {}
        lock = threading.Lock()

        def read_one(camera_key: str, camera: Any) -> None:
            frame_rgb = camera.read(timeout_ms=timeout_ms)
            if frame_rgb is None:
                return
            frame_rgb = apply_color_fix(frame_rgb, self.config, self.fix_edge_color_cast)
            frame_bgr = rgb_to_bgr(frame_rgb)
            with lock:
                frames[camera_key] = frame_bgr

        if parallel_read and len(self.cameras) > 1:
            threads = [
                threading.Thread(target=read_one, args=(camera_key, camera), name=f"read-{camera_key}", daemon=True)
                for camera_key, camera in self.cameras.items()
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for camera_key, camera in self.cameras.items():
                read_one(camera_key, camera)
        return frames

    def close(self) -> None:
        for camera in self.cameras.values():
            try:
                camera.close()
            except Exception:
                pass
        delay_s = as_float((self.config.get("camera", {}) or {}).get("argus_release_delay_s"), 0.0)
        if delay_s > 0.0:
            time.sleep(delay_s)


class VideoFrameSource:
    def __init__(self, path: Path):
        import cv2

        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        print(f"[video] {path}")

    def read(self) -> Dict[str, np.ndarray]:
        ok, frame = self.capture.read()
        return {"video": frame} if ok else {}

    def close(self) -> None:
        self.capture.release()


class HttpMjpegStreamer:
    def __init__(
        self,
        stream_cfg: Dict[str, Any],
        app_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
        mirror_config_paths: Optional[Sequence[Path]] = None,
    ):
        self.cfg = stream_cfg
        self.app_config = app_config or {}
        self.config_path = config_path
        self.mirror_config_paths = list(mirror_config_paths or [])
        self.config_lock = threading.RLock()
        self.enabled = as_bool(stream_cfg.get("enabled"), False)
        self.host = str(stream_cfg.get("host", "0.0.0.0"))
        self.port = as_int(stream_cfg.get("port"), 8081)
        self.width = max(160, as_int(stream_cfg.get("width"), 960))
        self.height = max(90, as_int(stream_cfg.get("height"), 360))
        self.keep_aspect = as_bool(stream_cfg.get("keep_aspect"), True)
        self.quality = max(30, min(95, as_int(stream_cfg.get("jpeg_quality"), 75)))
        self.frame_period_s = 1.0 / max(1.0, as_float(stream_cfg.get("fps"), 8.0))
        self.next_frame_at = 0.0
        self.condition = threading.Condition()
        self.latest_jpeg: Optional[bytes] = None
        self.frame_id = 0
        self.running = False
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.mdns_process: Optional[subprocess.Popen] = None
        self.mdns_advertised = False
        self.hostname = socket.gethostname().split(".")[0] or "rover"
        if self.enabled:
            self._open()

    def _camera_rois(self) -> Dict[str, Dict[str, float]]:
        with self.config_lock:
            cameras = ((self.app_config.get("camera", {}) or {}).get("cameras", {}) or {})
            out: Dict[str, Dict[str, float]] = {}
            for key in ("cam1", "cam0"):
                roi = ((cameras.get(key, {}) or {}).get("roi", {}) or {})
                out[key] = {
                    "x": as_float(roi.get("x"), 0.0),
                    "y": as_float(roi.get("y"), 0.0),
                    "w": as_float(roi.get("w"), 1.0),
                    "h": as_float(roi.get("h"), 1.0),
                }
            return out

    def _roi_payload(self, message: str = "") -> Dict[str, Any]:
        return {
            "ok": True,
            "message": message,
            "view": str(self.cfg.get("view", "fused_bev")),
            "rois": self._camera_rois(),
            "config_path": "" if self.config_path is None else str(self.config_path),
            "mirror_config_paths": [str(path) for path in self.mirror_config_paths],
            "network": self._network_payload(),
        }

    def _network_payload(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "mdns_url": f"http://{self.hostname}.local:{self.port}/",
            "bind_host": self.host,
            "port": self.port,
            "access_urls": self._access_urls(),
        }

    def _set_view(self, view: str) -> Dict[str, Any]:
        view = str(view or "").strip().lower()
        allowed = {"roi_edit", "roi_editor", "edit_roi", "yolo", "camera", "fused_bev", "camera_bev"}
        if view not in allowed:
            raise ValueError(f"unsupported view: {view}")
        if view in {"roi_editor", "edit_roi"}:
            view = "roi_edit"
        with self.config_lock:
            self.cfg["view"] = view
            self.app_config.setdefault("http_stream", {})["view"] = view
        return self._roi_payload(f"view={view}")

    def _set_roi(self, camera_key: str, roi_values: Dict[str, Any]) -> Dict[str, Any]:
        if camera_key not in {"cam1", "cam0"}:
            raise ValueError(f"unsupported camera: {camera_key}")
        current = self._camera_rois().get(camera_key, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
        x = as_float(roi_values.get("x"), current["x"])
        y = as_float(roi_values.get("y"), current["y"])
        w = as_float(roi_values.get("w"), current["w"])
        h = as_float(roi_values.get("h"), current["h"])
        x = clamp(x, 0.0, 0.99)
        y = clamp(y, 0.0, 0.99)
        w = clamp(w, 0.01, 1.0 - x)
        h = clamp(h, 0.01, 1.0 - y)
        clean = {"x": round(float(x), 5), "y": round(float(y), 5), "w": round(float(w), 5), "h": round(float(h), 5)}
        with self.config_lock:
            cameras = self.app_config.setdefault("camera", {}).setdefault("cameras", {})
            cameras.setdefault(camera_key, {}).setdefault("roi", {}).update(clean)
        return clean

    def _save_roi_configs(self) -> List[str]:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(f"PyYAML is required to save ROI: {exc}") from exc

        rois = self._camera_rois()
        paths: List[Path] = []
        if self.config_path is not None:
            paths.append(self.config_path)
        paths.extend(self.mirror_config_paths)
        saved: List[str] = []
        seen = set()
        for path in paths:
            path = Path(path).expanduser().resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            data = load_yaml(path)
            cameras = data.setdefault("camera", {}).setdefault("cameras", {})
            for key, roi in rois.items():
                cameras.setdefault(key, {}).setdefault("roi", {}).update(roi)
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            saved.append(str(path))
        return saved

    def _local_ipv4_addresses(self) -> List[str]:
        addresses = set()
        try:
            for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                addresses.add(item[4][0])
        except Exception:
            pass
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                addresses.add(sock.getsockname()[0])
        except Exception:
            pass
        try:
            output = subprocess.check_output(
                ["ip", "-o", "-4", "addr", "show", "scope", "global"],
                text=True,
                timeout=0.5,
            )
            for line in output.splitlines():
                parts = line.split()
                if len(parts) < 4:
                    continue
                iface = parts[1]
                if iface.startswith(("docker", "br-", "veth")):
                    continue
                addresses.add(parts[3].split("/", 1)[0])
        except Exception:
            pass
        return sorted(ip for ip in addresses if ip and not ip.startswith("127."))

    def _access_urls(self) -> List[str]:
        if self.host not in {"", "0.0.0.0", "::"}:
            return [f"http://{self.host}:{self.port}/"]
        urls = [f"http://{self.hostname}.local:{self.port}/"]
        urls.extend(f"http://{ip}:{self.port}/" for ip in self._local_ipv4_addresses())
        seen = set()
        return [url for url in urls if not (url in seen or seen.add(url))]

    def _print_access_banner(self) -> None:
        urls = self._access_urls()
        mdns_url = f"http://{self.hostname}.local:{self.port}/"
        numeric_urls = [url for url in urls if url != mdns_url]
        view = str(self.cfg.get("view", "fused_bev"))
        print("")
        print("=" * 78)
        print("[http] LAN VIEW READY - 다른 노트북에서 실시간 추론 화면 보기")
        print("=" * 78)
        print("[http] 1순위 주소: 같은 LAN/Wi-Fi 노트북 브라우저에서 여세요")
        print(f"[http]   {mdns_url}")
        print("[http]")
        print("[http] .local 주소가 안 열리면 현재 숫자 IP 주소를 쓰세요")
        if numeric_urls:
            for url in numeric_urls:
                print(f"[http]   {url}")
        else:
            print("[http]   아직 LAN IPv4 주소가 없습니다. Wi-Fi/Ethernet 연결을 확인하세요.")
        print("[http]")
        print(f"[http] Stream: {self.width}x{self.height} @ {1.0 / self.frame_period_s:.0f} fps, view={view}")
        print(f"[http] Bind: {self.host}:{self.port}, mDNS={'on' if self.mdns_advertised else 'off/unavailable'}")
        print("[http] 다른 LAN으로 옮겨도 먼저 .local 주소를 쓰면 됩니다.")
        print("[http] 웹 페이지 상단의 숫자 IP 주소는 5초마다 자동 갱신됩니다.")
        print("=" * 78)
        print("")

    def _open(self) -> None:
        try:
            handler = self._make_handler()
            self.server = ThreadingHTTPServer((self.host, self.port), handler)
            self.running = True
            self.thread = threading.Thread(target=self.server.serve_forever, name="http-mjpeg-stream", daemon=True)
            self.thread.start()
            self._start_mdns()
            self._print_access_banner()
        except Exception as exc:
            print(f"[http] disabled: failed to start HTTP MJPEG server: {exc}")
            self.enabled = False
            self.running = False
            self.server = None

    def _start_mdns(self) -> None:
        self.mdns_advertised = False
        if not as_bool(self.cfg.get("mdns_enabled"), True):
            return
        try:
            self.mdns_process = subprocess.Popen(
                [
                    "avahi-publish-service",
                    "Rover Trailer View",
                    "_http._tcp",
                    str(self.port),
                    "path=/",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.mdns_advertised = True
            print(f"[http] mdns broadcast enabled: http://{self.hostname}.local:{self.port}/")
        except FileNotFoundError:
            print("[http] mdns broadcast unavailable: avahi-publish-service not found")
        except Exception as exc:
            print(f"[http] mdns broadcast unavailable: {exc}")

    def _make_handler(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                if path in {"/", "/index.html"}:
                    self._send_index()
                    return
                if path in {"/stream.mjpg", "/stream"}:
                    self._send_stream()
                    return
                if path == "/api/roi":
                    self._send_json(streamer._roi_payload())
                    return
                if path == "/api/network":
                    self._send_json({"ok": True, "network": streamer._network_payload()})
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    payload = self._read_json()
                    query = parse_qs(parsed.query)
                    if path == "/api/roi":
                        camera_key = str(payload.get("camera") or (query.get("camera", [""])[0]))
                        streamer._set_roi(camera_key, payload)
                        self._send_json(streamer._roi_payload(f"{camera_key} ROI updated"))
                        return
                    if path == "/api/save":
                        saved = streamer._save_roi_configs()
                        self._send_json(streamer._roi_payload(f"saved {len(saved)} file(s)") | {"saved": saved})
                        return
                    if path == "/api/view":
                        view = str(payload.get("view") or (query.get("view", [""])[0]))
                        self._send_json(streamer._set_view(view))
                        return
                    self.send_error(404)
                except Exception as exc:
                    self._send_json({"ok": False, "message": str(exc)}, status=400)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                return json.loads(raw.decode("utf-8"))

            def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_index(self) -> None:
                links = "\n".join(f'<a href="{url}">{url}</a>' for url in streamer._access_urls())
                body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rover Trailer View</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #101112; color: #f3f3f3; font-family: system-ui, sans-serif; }}
    header {{ display: flex; gap: 12px; align-items: center; justify-content: space-between; padding: 10px 14px; background: #1b1d20; border-bottom: 1px solid #33383d; }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 650; }}
    .urls {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 13px; }}
    a {{ color: #7fd0ff; text-decoration: none; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 12px; padding: 12px; }}
    .stage {{ position: relative; width: min(100%, {streamer.width}px); margin: 0 auto; background: #050505; touch-action: none; }}
    #stream {{ display: block; width: 100%; height: auto; background: #050505; user-select: none; -webkit-user-drag: none; }}
    #overlay {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }}
    aside {{ background: #181a1d; border: 1px solid #30343a; padding: 10px; min-width: 0; }}
    button {{ background: #2b3036; color: #f4f4f4; border: 1px solid #4b535d; padding: 8px 10px; margin: 3px; font: inherit; cursor: pointer; }}
    button.primary {{ background: #166534; border-color: #22c55e; }}
    .row {{ display: grid; grid-template-columns: 48px repeat(4, 1fr); gap: 5px; align-items: center; margin: 8px 0; }}
    input {{ width: 100%; background: #0f1113; color: #f3f3f3; border: 1px solid #3a4048; padding: 6px; }}
    .hint, #status, #networkStatus {{ color: #bac3cf; font-size: 13px; line-height: 1.35; }}
    code {{ color: #a7f3d0; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} aside {{ order: -1; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Rover trailer view</h1>
    <div class="urls" id="networkLinks">{links}</div>
  </header>
  <main>
    <section class="stage" id="stage">
      <img id="stream" src="/stream.mjpg" alt="rover trailer stream">
      <canvas id="overlay"></canvas>
    </section>
    <aside>
      <div>
        <button data-view="roi_edit" class="primary">ROI Edit</button>
        <button data-view="yolo">YOLO</button>
        <button data-view="camera">Camera</button>
        <button data-view="fused_bev">BEV</button>
      </div>
      <p class="hint">노트북은 <code>http://{streamer.hostname}.local:{streamer.port}/</code>로 접속하면 LAN이 바뀌어도 IP를 외울 필요가 없다.</p>
      <p id="networkStatus"></p>
      <p class="hint">ROI Edit에서 원하는 카메라 위를 드래그하면 ROI가 바로 바뀐다. 저장해야 YAML에 남는다.</p>
      <div id="roiRows"></div>
      <div>
        <button id="saveBtn" class="primary">Save YAML</button>
        <button id="reloadBtn">Reload</button>
      </div>
      <p id="status"></p>
    </aside>
  </main>
  <script>
    const streamW = {streamer.width};
    const streamH = {streamer.height};
    const frameW = {as_int(((streamer.app_config.get("camera", {}) or {}).get("output_width")), 640)};
    const frameH = {as_int(((streamer.app_config.get("camera", {}) or {}).get("output_height")), 360)};
    const img = document.getElementById('stream');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const networkStatusEl = document.getElementById('networkStatus');
    const networkLinksEl = document.getElementById('networkLinks');
    const roiRows = document.getElementById('roiRows');
    let state = {{ view: 'roi_edit', rois: {{ cam1: {{x:0,y:0,w:1,h:1}}, cam0: {{x:0,y:0,w:1,h:1}} }} }};
    let drag = null;

    function setStatus(text) {{ statusEl.textContent = text || ''; }}

    function renderNetwork(network) {{
      if (!network) return;
      const urls = network.access_urls || [];
      networkLinksEl.innerHTML = urls.map(url => `<a href="${{url}}">${{url}}</a>`).join('');
      networkStatusEl.textContent = `LAN auto: ${{network.mdns_url}}  current: ${{urls.join('  ')}}`;
    }}

    async function api(path, payload) {{
      const opts = payload === undefined ? {{}} : {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }};
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!data.ok) throw new Error(data.message || 'request failed');
      return data;
    }}

    function imageScale() {{
      const rect = img.getBoundingClientRect();
      return {{ x: rect.width / streamW, y: rect.height / streamH, w: rect.width, h: rect.height }};
    }}

    function camBox(camera) {{
      const idx = camera === 'cam1' ? 0 : 1;
      const tileW = streamW / 2;
      const tileH = streamH;
      const scale = Math.min(tileW / frameW, tileH / frameH);
      const drawW = frameW * scale;
      const drawH = frameH * scale;
      return {{
        x: idx * tileW + (tileW - drawW) / 2,
        y: (tileH - drawH) / 2,
        w: drawW,
        h: drawH,
        scale
      }};
    }}

    function normToCanvas(camera, roi) {{
      const box = camBox(camera);
      const sc = imageScale();
      return {{
        x: (box.x + roi.x * box.w) * sc.x,
        y: (box.y + roi.y * box.h) * sc.y,
        w: roi.w * box.w * sc.x,
        h: roi.h * box.h * sc.y
      }};
    }}

    function pointerToNorm(evt) {{
      const rect = canvas.getBoundingClientRect();
      const sc = imageScale();
      const sx = (evt.clientX - rect.left) / sc.x;
      const sy = (evt.clientY - rect.top) / sc.y;
      const camera = sx < streamW / 2 ? 'cam1' : 'cam0';
      const box = camBox(camera);
      const nx = Math.max(0, Math.min(1, (sx - box.x) / box.w));
      const ny = Math.max(0, Math.min(1, (sy - box.y) / box.h));
      return {{ camera, x: nx, y: ny }};
    }}

    function draw() {{
      const rect = img.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width));
      canvas.height = Math.max(1, Math.round(rect.height));
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (state.view !== 'roi_edit') return;
      for (const camera of ['cam1', 'cam0']) {{
        const r = normToCanvas(camera, state.rois[camera]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = camera === 'cam1' ? '#ffdc5e' : '#63f27a';
        ctx.fillStyle = camera === 'cam1' ? 'rgba(255,220,94,.12)' : 'rgba(99,242,122,.12)';
        ctx.fillRect(r.x, r.y, r.w, r.h);
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        ctx.fillStyle = '#050505';
        ctx.fillRect(r.x, Math.max(0, r.y - 22), 118, 20);
        ctx.fillStyle = '#fff';
        ctx.font = '14px system-ui';
        ctx.fillText(camera + ' ROI', r.x + 6, Math.max(14, r.y - 7));
      }}
      if (drag) {{
        const x0 = Math.min(drag.start.x, drag.current.x);
        const y0 = Math.min(drag.start.y, drag.current.y);
        const x1 = Math.max(drag.start.x, drag.current.x);
        const y1 = Math.max(drag.start.y, drag.current.y);
        const preview = normToCanvas(drag.camera, {{ x: x0, y: y0, w: x1 - x0, h: y1 - y0 }});
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 3;
        ctx.strokeRect(preview.x, preview.y, preview.w, preview.h);
      }}
    }}

    function renderRows() {{
      roiRows.innerHTML = '';
      for (const camera of ['cam1', 'cam0']) {{
        const roi = state.rois[camera];
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = `<strong>${{camera}}</strong>` + ['x','y','w','h'].map(k =>
          `<input data-camera="${{camera}}" data-key="${{k}}" type="number" step="0.0001" min="0" max="1" value="${{Number(roi[k]).toFixed(5)}}">`
        ).join('');
        roiRows.appendChild(row);
      }}
      roiRows.querySelectorAll('input').forEach(input => {{
        input.addEventListener('change', async () => {{
          const camera = input.dataset.camera;
          const roi = {{ ...state.rois[camera], [input.dataset.key]: Number(input.value) }};
          const data = await api('/api/roi', {{ camera, ...roi }});
          state = data;
          renderRows();
          draw();
        }});
      }});
    }}

    async function refresh() {{
      state = await api('/api/roi');
      renderNetwork(state.network);
      renderRows();
      draw();
      setStatus(state.message || `view=${{state.view}}`);
    }}

    async function refreshNetwork() {{
      try {{
        const data = await api('/api/network');
        renderNetwork(data.network);
      }} catch (err) {{
        networkStatusEl.textContent = 'network status unavailable';
      }}
    }}

    canvas.addEventListener('pointerdown', evt => {{
      if (state.view !== 'roi_edit') return;
      const p = pointerToNorm(evt);
      drag = {{ camera: p.camera, start: {{x:p.x, y:p.y}}, current: {{x:p.x, y:p.y}} }};
      canvas.setPointerCapture(evt.pointerId);
      draw();
    }});
    canvas.addEventListener('pointermove', evt => {{
      if (!drag) return;
      const p = pointerToNorm(evt);
      if (p.camera !== drag.camera) return;
      drag.current = {{x:p.x, y:p.y}};
      draw();
    }});
    canvas.addEventListener('pointerup', async evt => {{
      if (!drag) return;
      const done = drag;
      drag = null;
      const x0 = Math.min(done.start.x, done.current.x);
      const y0 = Math.min(done.start.y, done.current.y);
      const x1 = Math.max(done.start.x, done.current.x);
      const y1 = Math.max(done.start.y, done.current.y);
      if (x1 - x0 > 0.01 && y1 - y0 > 0.01) {{
        const data = await api('/api/roi', {{ camera: done.camera, x: x0, y: y0, w: x1 - x0, h: y1 - y0 }});
        state = data;
        renderRows();
      }}
      draw();
    }});
    window.addEventListener('resize', draw);
    document.querySelectorAll('button[data-view]').forEach(btn => {{
      btn.addEventListener('click', async () => {{
        state = await api('/api/view', {{ view: btn.dataset.view }});
        renderRows();
        draw();
        setStatus(state.message);
      }});
    }});
    document.getElementById('saveBtn').addEventListener('click', async () => {{
      const data = await api('/api/save', {{}});
      state = data;
      setStatus(data.message + ': ' + (data.saved || []).join(', '));
    }});
    document.getElementById('reloadBtn').addEventListener('click', refresh);
    img.addEventListener('load', draw);
    setInterval(refreshNetwork, 5000);
    refresh().catch(err => setStatus(err.message));
  </script>
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
        if not self.enabled or not self.running:
            return
        now = time.monotonic()
        if now < self.next_frame_at:
            return
        self.next_frame_at = now + self.frame_period_s
        try:
            import cv2

            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                if self.keep_aspect:
                    frame_bgr, _scale, _off_x, _off_y = fit_into(frame_bgr, self.width, self.height)
                else:
                    frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                return
            with self.condition:
                self.latest_jpeg = encoded.tobytes()
                self.frame_id += 1
                self.condition.notify_all()
        except Exception as exc:
            print(f"[http] disabled after write failure: {exc}")
            self.close()

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
        if self.mdns_process is not None:
            try:
                self.mdns_process.terminate()
                self.mdns_process.wait(timeout=1.0)
            except Exception:
                try:
                    self.mdns_process.kill()
                except Exception:
                    pass
            self.mdns_process = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegFormer dotted-centerline lane follower for trailer_task.")
    parser.add_argument("--config", type=Path, default=HERE / "dotted_lane_following_config.yaml")
    parser.add_argument("--video", type=Path, default=None, help="Use a video file instead of CSI camera.")
    parser.add_argument("--single-camera", choices=("left", "right", "cam0", "cam1"), default="")
    parser.add_argument("--sensor-id", type=int, default=None, help="Override the selected CSI camera sensor id.")
    parser.add_argument("--start-driving", action="store_true", help="Start sending lane commands immediately.")
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--arm", dest="arm_override", action="store_true", default=None)
    arm_group.add_argument("--no-arm", dest="arm_override", action="store_false", default=None)
    parser.add_argument("--base-speed", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument("--display", dest="display_override", action="store_true", default=None)
    parser.add_argument("--no-display", dest="display_override", action="store_false", default=None)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream_override", action="store_true", default=None)
    stream_group.add_argument("--no-stream", dest="stream_override", action="store_false", default=None)
    parser.add_argument("--udp-host", default="")
    parser.add_argument("--udp-port", type=int, default=0)
    http_group = parser.add_mutually_exclusive_group()
    http_group.add_argument("--http-stream", dest="http_stream_override", action="store_true", default=None)
    http_group.add_argument("--no-http-stream", dest="http_stream_override", action="store_false", default=None)
    parser.add_argument("--http-host", default="")
    parser.add_argument("--http-port", type=int, default=0)
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.single_camera:
        config.setdefault("camera", {})["single_camera"] = args.single_camera
    if args.sensor_id is not None:
        camera = config.setdefault("camera", {})
        camera_cfgs = enabled_camera_configs(config, args.single_camera)
        if len(camera_cfgs) == 1:
            camera_key = next(iter(camera_cfgs.keys()))
        else:
            camera_key = "cam0"
        camera.setdefault("cameras", {}).setdefault(camera_key, {})["sensor_id"] = int(args.sensor_id)
    if args.arm_override is not None:
        config.setdefault("rover", {})["arm"] = bool(args.arm_override)
    if args.display_override is not None:
        config.setdefault("runtime", {})["display"] = bool(args.display_override)
    if args.stream_override is not None:
        config.setdefault("stream", {})["enabled"] = bool(args.stream_override)
    if args.udp_host:
        config.setdefault("stream", {})["host"] = args.udp_host
    if args.udp_port > 0:
        config.setdefault("stream", {})["port"] = int(args.udp_port)
    if args.http_stream_override is not None:
        config.setdefault("http_stream", {})["enabled"] = bool(args.http_stream_override)
    if args.http_host:
        config.setdefault("http_stream", {})["host"] = args.http_host
    if args.http_port > 0:
        config.setdefault("http_stream", {})["port"] = int(args.http_port)
    if args.base_speed > 0.0:
        config.setdefault("lane", {})["base_speed"] = float(args.base_speed)
    if args.max_frames > 0:
        config.setdefault("runtime", {})["max_frames"] = int(args.max_frames)


def parse_ratio_points(value: Any, min_points: int = 4) -> Optional[Tuple[Tuple[float, float], ...]]:
    points: List[Tuple[float, float]] = []
    for item in value or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        points.append((float(item[0]), float(item[1])))
    if len(points) < min_points:
        return None
    return tuple(points)


def tuple_points(value: Any) -> Tuple[Tuple[float, float], ...]:
    parsed = parse_ratio_points(value, min_points=4)
    if parsed is None:
        return BevConfig().src_points_ratio
    return tuple(parsed[:4])


def optional_tuple_points(value: Any, min_points: int = 4) -> Optional[Tuple[Tuple[float, float], ...]]:
    return parse_ratio_points(value, min_points=min_points)


def point_order_from_value(value: Any, fallback_len: int = 4) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        names = tuple(str(item) for item in value)
        if all(names):
            return names
    default = ("TL", "TR", "BR", "BL", "C")
    return default[: max(4, fallback_len)]


def points_to_array(points: Any, width: int, height: int, default_ratio: Tuple[Tuple[float, float], ...]) -> np.ndarray:
    parsed = optional_tuple_points(points) or default_ratio
    return np.array([(x * width, y * height) for x, y in parsed], dtype=np.float32)


def make_bev_config(config: Dict[str, Any]) -> BevConfig:
    data = config.get("bev", {}) or {}
    bev = BevConfig(
        output_width=as_int(data.get("output_width"), 640),
        output_height=as_int(data.get("output_height"), 720),
    )
    bev.src_points_ratio = tuple_points(data.get("src_points_ratio"))
    return bev


def load_dual_bev_calibration(config: Dict[str, Any], config_dir: Path) -> DualBevCalibration:
    bev_cfg = config.get("bev", {}) or {}
    dual_cfg = bev_cfg.get("dual", {}) or {}
    mode = str(bev_cfg.get("mode", "per_camera_fusion")).lower()
    requested = as_bool(dual_cfg.get("enabled"), False) or mode in {"dual", "dual_bev", "dual_homography"}
    if not requested:
        dst_points = optional_tuple_points(dual_cfg.get("dst_points_ratio"), min_points=4) or tuple_points(dual_cfg.get("dst_points_ratio"))
        return DualBevCalibration(False, {}, dst_points, 0.0, "disabled")

    calibration_data: Dict[str, Any] = {}
    calibration_value = str(dual_cfg.get("calibration_file", "") or "")
    source = "inline"
    if calibration_value:
        calibration_path = resolve_path(config_dir, calibration_value)
        if calibration_path.exists():
            calibration_data = load_yaml(calibration_path)
            source = str(calibration_path)
        else:
            source = f"missing:{calibration_path}"

    file_root = calibration_data.get("dual_bev", calibration_data) if calibration_data else {}
    file_cameras = file_root.get("cameras", {}) or {}
    inline_cameras = dual_cfg.get("cameras", {}) or {}
    cameras: Dict[str, Dict[str, Any]] = {}
    for key in set(inline_cameras.keys()) | set(file_cameras.keys()):
        merged: Dict[str, Any] = {}
        merged.update(inline_cameras.get(key, {}) or {})
        merged.update(file_cameras.get(key, {}) or {})
        if merged.get("src_points_ratio") or merged.get("src_points_px"):
            cameras[key] = merged

    default_dst = ((0.10, 0.95), (0.90, 0.95), (0.90, 0.08), (0.10, 0.08))
    dst_points = (
        optional_tuple_points(file_root.get("dst_points_ratio"), min_points=4)
        or optional_tuple_points(dual_cfg.get("dst_points_ratio"), min_points=4)
        or default_dst
    )
    point_order = point_order_from_value(file_root.get("point_order", dual_cfg.get("point_order")), len(dst_points))
    vehicle_center_bias = as_float(
        file_root.get("vehicle_center_x_bias"),
        as_float(dual_cfg.get("vehicle_center_x_bias"), 0.0),
    )
    merge_mode = str(file_root.get("merge_mode", dual_cfg.get("merge_mode", "class_priority_max")))
    drive_mode = str(file_root.get("drive_mode", dual_cfg.get("drive_mode", "merged_mask"))).lower()
    if drive_mode not in {"merged_mask", "estimate_fusion"}:
        drive_mode = "merged_mask"
    enabled = requested and bool(cameras)
    return DualBevCalibration(enabled, cameras, dst_points, vehicle_center_bias, merge_mode, source, point_order, drive_mode)


def make_lane_config(config: Dict[str, Any], camera_cfg: Optional[Dict[str, Any]] = None) -> LaneFollowerConfig:
    data = config.get("lane", {}) or {}
    camera_cfg = camera_cfg or {}
    lane = LaneFollowerConfig()
    for key in (
        "control_mode",
        "nominal_lane_width_px",
        "min_lane_width_px",
        "max_lane_width_px",
        "row_samples",
        "row_y_min_ratio",
        "row_y_max_ratio",
        "row_band_px",
        "min_component_pixels",
        "lookahead_y_ratio",
        "pure_pursuit_gain",
        "lateral_gain",
        "heading_gain",
        "steer_smoothing",
        "center_smoothing",
        "vehicle_center_x_bias",
        "min_confidence",
        "base_speed",
        "min_speed",
        "max_speed",
    ):
        if key in data:
            current = getattr(lane, key)
            if isinstance(current, bool):
                setattr(lane, key, as_bool(data.get(key), current))
            elif isinstance(current, int):
                setattr(lane, key, as_int(data.get(key), current))
            elif isinstance(current, float):
                setattr(lane, key, as_float(data.get(key), current))
            else:
                setattr(lane, key, str(data.get(key)))
    if "vehicle_center_x_bias" in camera_cfg:
        lane.vehicle_center_x_bias = as_float(camera_cfg.get("vehicle_center_x_bias"), lane.vehicle_center_x_bias)
    return lane


def crop_frame(
    frame_bgr: np.ndarray,
    config: Dict[str, Any],
    camera_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    camera_cfg = camera_cfg or {}
    crop_cfg = camera_cfg.get("crop") or ((config.get("preprocess", {}) or {}).get("crop", {}) or {})
    h, w = frame_bgr.shape[:2]
    x0 = int(round(clamp(as_float(crop_cfg.get("x"), 0.0), 0.0, 0.99) * w))
    y0 = int(round(clamp(as_float(crop_cfg.get("y"), 0.0), 0.0, 0.99) * h))
    cw = clamp(as_float(crop_cfg.get("w"), 1.0), 0.01, 1.0)
    ch = clamp(as_float(crop_cfg.get("h"), 1.0), 0.01, 1.0)
    x1 = max(x0 + 1, min(w, int(round((as_float(crop_cfg.get("x"), 0.0) + cw) * w))))
    y1 = max(y0 + 1, min(h, int(round((as_float(crop_cfg.get("y"), 0.0) + ch) * h))))
    return frame_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def dual_src_points(camera_calib: Dict[str, Any], width: int, height: int) -> Optional[np.ndarray]:
    ratio = optional_tuple_points(camera_calib.get("src_points_ratio"), min_points=4)
    if ratio is not None:
        return np.array([(x * width, y * height) for x, y in ratio], dtype=np.float32)
    if camera_calib.get("src_points_px"):
        points = camera_calib.get("src_points_px")
        if isinstance(points, list) and len(points) >= 4:
            return np.array(points, dtype=np.float32)
    return None


def dual_dst_points(dual: DualBevCalibration, bev: BevConfig) -> np.ndarray:
    return np.array(
        [(x * bev.output_width, y * bev.output_height) for x, y in dual.dst_points_ratio],
        dtype=np.float32,
    )


def dual_warp_mask(mask: np.ndarray, camera_key: str, dual: DualBevCalibration, bev: BevConfig) -> Optional[np.ndarray]:
    camera_calib = dual.cameras.get(camera_key)
    if not camera_calib:
        return None
    src = dual_src_points(camera_calib, mask.shape[1], mask.shape[0])
    if src is None:
        return None
    dst = dual_dst_points(dual, bev)
    if len(src) != len(dst):
        print(f"[dual_bev] {camera_key} point count mismatch: src={len(src)} dst={len(dst)}")
        return None
    try:
        import cv2

        if len(src) == 4:
            matrix = cv2.getPerspectiveTransform(src, dst)
        else:
            matrix, _status = cv2.findHomography(src, dst, method=0)
            if matrix is None:
                print(f"[dual_bev] {camera_key} homography failed with {len(src)} points")
                return None
        return cv2.warpPerspective(
            mask,
            matrix,
            (bev.output_width, bev.output_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except Exception as exc:
        print(f"[dual_bev] {camera_key} warp failed: {exc}")
        return None


def merge_bev_masks(masks: List[np.ndarray], dual: DualBevCalibration, bev: BevConfig) -> np.ndarray:
    if not masks:
        return np.zeros((bev.output_height, bev.output_width), dtype=np.uint8)
    merged = np.zeros((bev.output_height, bev.output_width), dtype=np.uint8)
    for mask in masks:
        if dual.merge_mode == "overwrite":
            active = mask > 0
            merged[active] = mask[active]
        else:
            # Class priority: background=0, solid=1, dashed=2.
            merged = np.maximum(merged, mask)
    return merged


def open_source(config: Dict[str, Any], args: argparse.Namespace):
    if args.video is not None:
        return VideoFrameSource(args.video.expanduser().resolve())
    source = str((config.get("camera", {}) or {}).get("source", "csi")).lower()
    if source != "csi":
        raise RuntimeError(f"Unsupported camera.source={source!r}; use --video for files.")
    return CsiFrameSource(config)


def get_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_predictor(config: Dict[str, Any], config_dir: Path):
    model_cfg = ((config.get("models", {}) or {}).get("segmentation", {}) or {})
    weights = resolve_path(config_dir, str(model_cfg.get("weights", "")))
    if not weights.exists():
        raise SystemExit(f"Segmentation model not found: {weights}")
    device = get_device(str(model_cfg.get("device", "auto")))
    half = as_bool(model_cfg.get("half"), True)
    width = as_int(model_cfg.get("model_width"), 512)
    height = as_int(model_cfg.get("model_height"), 288)
    backend = str(model_cfg.get("backend", "auto") or "auto").strip().lower()
    onnx_value = str(model_cfg.get("onnx", "") or "")
    if backend in {"auto", "onnx", "onnxruntime", "trt", "tensorrt"} and onnx_value:
        onnx_path = resolve_path(config_dir, onnx_value)
        if onnx_path.exists():
            try:
                from optimized_segformer import OnnxSegFormerYellowLinePredictor

                cache_value = str(model_cfg.get("onnx_trt_cache", "") or "")
                cache_path = resolve_path(config_dir, cache_value) if cache_value else onnx_path.parent / "ort_trt_cache"
                predictor = OnnxSegFormerYellowLinePredictor(
                    onnx_path,
                    width,
                    height,
                    trt_cache_dir=cache_path,
                    prefer_tensorrt=backend in {"auto", "trt", "tensorrt"},
                    fast_postprocess=as_bool(model_cfg.get("fast_postprocess"), True),
                )
                print(f"[seg] backend=onnx providers={predictor.providers}")
                print(f"[seg] weights={onnx_path}")
                return predictor
            except Exception as exc:
                print(f"[seg] onnx backend unavailable, falling back: {exc}")
        elif backend != "auto":
            raise SystemExit(f"Segmentation ONNX model not found: {onnx_path}")
    torchscript_value = str(model_cfg.get("torchscript", "") or "")
    if backend in {"auto", "torchscript", "jit"} and torchscript_value:
        script_path = resolve_path(config_dir, torchscript_value)
        if script_path.exists():
            try:
                from optimized_segformer import TorchScriptSegFormerYellowLinePredictor

                print(f"[seg] backend=torchscript weights={script_path}")
                print(f"[seg] device={device} half={bool(half and device.type == 'cuda')}")
                return TorchScriptSegFormerYellowLinePredictor(script_path, device, width, height, half)
            except Exception as exc:
                print(f"[seg] torchscript backend unavailable, falling back: {exc}")
        elif backend != "auto":
            raise SystemExit(f"Segmentation TorchScript model not found: {script_path}")
    print(f"[seg] weights={weights}")
    print(f"[seg] device={device} half={bool(half and device.type == 'cuda')}")
    from jetson_lane_only_runner import SegFormerYellowLinePredictor

    return SegFormerYellowLinePredictor(
        weights,
        device,
        width,
        height,
        half,
    )


def load_panel_model(config: Dict[str, Any], config_dir: Path):
    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    if not as_bool(panel_cfg.get("enabled"), False):
        return None
    weights_value = str(panel_cfg.get("weights", "") or "")
    if not weights_value:
        print("[panel] disabled: models.panel.weights is empty")
        return None
    weights = resolve_path(config_dir, weights_value)
    if not weights.exists():
        print(f"[panel] disabled: weights not found: {weights}")
        return None
    try:
        from ultralytics import YOLO

        print(f"[panel] weights={weights}")
        if weights.suffix.lower() == ".engine":
            return YOLO(str(weights), task="detect")
        return YOLO(str(weights))
    except Exception as exc:
        print(f"[panel] disabled: failed to load YOLO model: {exc}")
        return None


def estimate_trailer_angle(
    config: Dict[str, Any],
    panel_model: Any,
    angle_estimator: Optional[CenterTableAngleEstimator],
    angle_filter: Optional[AngleStateFilter],
    detections: Dict[str, PanelDetection],
    estimates: Dict[str, AngleEstimate],
    frames_bgr: Dict[str, np.ndarray],
    frame_idx: int,
    now: float,
) -> FusedAngleEstimate:
    if panel_model is None or angle_estimator is None or angle_filter is None:
        return FusedAngleEstimate(False, None, 0.0, "disabled", "panel angle disabled", timestamp=now)
    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    infer_every = max(1, as_int(panel_cfg.get("infer_every"), 2))
    measurements: List[AngleEstimate] = []
    if frame_idx % infer_every == 0:
        for key, frame_bgr in frames_bgr.items():
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            det = best_panel_detection_in_frame(panel_model, frame_rgb, key, config, panel_cfg, now)
            detections[key] = det
            estimates[key] = angle_estimator.estimate(det)
        measurements = enforce_single_panel_detection(detections, estimates, config, now)
    return angle_filter.update(measurements, now=now)


def apply_lane_lookahead(lane_cfgs: Dict[str, LaneFollowerConfig], fused_lane_cfg: LaneFollowerConfig, lookahead_y_ratio: float) -> None:
    value = float(clamp(lookahead_y_ratio, 0.05, 0.98))
    fused_lane_cfg.lookahead_y_ratio = value
    for lane_cfg in lane_cfgs.values():
        lane_cfg.lookahead_y_ratio = value


def use_simple_dashed_priority(
    config: Dict[str, Any],
    route_controller: RouteAwareTrailerController,
    lane_signal: LaneSignal,
    angle_state: TrailerAngleState,
    base_command: LaneDriveCommand,
) -> bool:
    cfg = config.get("route_controller", {}) or {}
    if not as_bool(cfg.get("simple_dashed_priority_enabled"), True):
        return False
    if route_controller.mode != "STRAIGHT":
        return False
    if not base_command.valid or base_command.brake:
        return False
    state = str(lane_signal.state).upper()
    if "LANE_FOLLOW" not in state and "DASHED_PARTIAL" not in state:
        return False
    if lane_signal.confidence < as_float(cfg.get("simple_dashed_min_confidence"), 0.62):
        return False
    if lane_signal.row_count < as_int(cfg.get("simple_dashed_min_rows"), 8):
        return False
    if lane_signal.corner_confidence > as_float(cfg.get("simple_dashed_max_corner_confidence"), 0.18):
        return False
    max_angle = as_float(cfg.get("simple_dashed_max_abs_angle_deg"), 45.0)
    if angle_state.angle_deg is not None and abs(float(angle_state.angle_deg)) > max_angle:
        return False
    return True


def simple_dashed_command(base_command: LaneDriveCommand, active: bool, config: Dict[str, Any]) -> LaneDriveCommand:
    cfg = config.get("route_controller", {}) or {}
    speed_scale = max(0.0, as_float(cfg.get("simple_dashed_speed_scale"), 1.0))
    max_steer = abs(as_float(cfg.get("simple_dashed_max_steer"), 1.0))
    return LaneDriveCommand(
        active=active,
        valid=base_command.valid,
        confidence=base_command.confidence,
        state=f"SIMPLE_DASHED:{base_command.state}",
        steer=clamp(base_command.steer, -max_steer, max_steer),
        speed=max(0.0, base_command.speed * speed_scale),
        brake=not active,
        reason=f"simple_dashed_priority {base_command.reason}",
    )


def estimate_lane_with_solid_fallback(
    bev_mask: np.ndarray,
    lane_config: LaneFollowerConfig,
    lane_state: LaneFollowerState,
    app_config: Dict[str, Any],
    route_hint: str = "none",
) -> LaneEstimate:
    estimate = estimate_lane(bev_mask, lane_config, lane_state, route_hint=route_hint)
    lane_cfg = app_config.get("lane", {}) or {}
    if not as_bool(lane_cfg.get("solid_fallback_enabled"), True):
        return estimate
    if estimate.valid:
        return estimate
    if estimate.state not in {"DASHED_RECOVERY", "LOW_CONFIDENCE"}:
        return estimate

    rows = list(estimate.row_estimates)
    pair_rows = sum(1 for row in rows if row.method == "solid_pair_midpoint")
    single_rows = sum(1 for row in rows if row.method == "single_solid_recovery")
    solid_rows = pair_rows + single_rows
    dashed_rows = sum(1 for row in rows if row.dashed_x is not None)
    min_rows = max(1, as_int(lane_cfg.get("solid_fallback_min_rows"), 4))
    min_pair_rows = max(0, as_int(lane_cfg.get("solid_fallback_min_pair_rows"), 2))
    min_confidence = max(0.0, as_float(lane_cfg.get("solid_fallback_min_confidence"), 0.16))
    if solid_rows < min_rows or pair_rows < min_pair_rows or estimate.confidence < min_confidence:
        return estimate

    min_drive_confidence = as_float(lane_cfg.get("min_drive_confidence"), 0.08)
    confidence_scale = clamp(as_float(lane_cfg.get("solid_fallback_confidence_scale"), 0.78), 0.05, 1.0)
    confidence = max(min_drive_confidence, min(1.0, estimate.confidence * confidence_scale))
    fallback_speed = as_float(lane_cfg.get("solid_fallback_speed"), 0.0)
    if fallback_speed <= 0.0:
        fallback_speed = max(float(estimate.speed), lane_config.min_speed)
    speed = clamp(fallback_speed, 0.0, lane_config.max_speed)

    return LaneEstimate(
        valid=True,
        confidence=float(confidence),
        state="SOLID_FALLBACK",
        center_x=estimate.center_x,
        lookahead_y=estimate.lookahead_y,
        lateral_error_px=estimate.lateral_error_px,
        heading_error_rad=estimate.heading_error_rad,
        raw_steer=estimate.raw_steer,
        steer=estimate.steer,
        speed=float(speed),
        lane_width_px=estimate.lane_width_px,
        row_estimates=rows,
        poly_coefficients=estimate.poly_coefficients,
        reason=(
            f"solid_fallback pair_rows={pair_rows} single_rows={single_rows} "
            f"dashed_rows={dashed_rows}/{len(rows)} base={estimate.state}:{estimate.reason}"
        ),
        route_hint=estimate.route_hint,
    )


class CornerPivotController:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.cfg = config.get("corner_pivot", {}) or {}
        self.enabled = as_bool(self.cfg.get("enabled"), False)
        self.active = False
        self.direction = 0.0
        self.started_at = 0.0
        self.enter_count = 0
        self.exit_count = 0
        self.last_candidate_sign = 0.0
        self.last_exit_turn_sign = 0.0
        self.last_debug = "idle"

    def override(
        self,
        estimate: Optional[LaneEstimate],
        results: Dict[str, CameraLaneResult],
        angle_state: TrailerAngleState,
        command: LaneDriveCommand,
        active: bool,
        bev_width: int,
        bev_height: int,
        now: float,
    ) -> LaneDriveCommand:
        if not self.enabled or not active:
            self._reset()
            return command

        if self._suppress_during_post_corner_bias(command):
            self._reset()
            self.last_debug = "reject:post_corner_bias"
            return command

        lane_estimate = self._select_estimate(estimate, results, bev_width, bev_height)
        if lane_estimate is None:
            self._reset()
            self.last_debug = "reject:no_estimate"
            return command

        stats = self._lane_stats(lane_estimate, bev_width, bev_height)
        angle_abs = abs(angle_state.angle_deg) if angle_state.angle_deg is not None else None
        angle_stop_enabled = as_bool(self.cfg.get("angle_stop_enabled"), True)
        stop_angle = abs(as_float(self.cfg.get("stop_abs_trailer_angle_deg"), 46.0))
        if self.active:
            self.last_debug = f"active:{self._format_stats(stats)}"
            if angle_stop_enabled and angle_abs is not None and angle_abs >= stop_angle:
                self._reset()
                self.last_debug = f"fault:angle:{angle_abs:.1f}>={stop_angle:.1f}"
                return LaneDriveCommand(
                    active=True,
                    valid=False,
                    confidence=0.0,
                    state="CORNER_PIVOT_FAULT",
                    steer=0.0,
                    speed=0.0,
                    brake=True,
                    reason=f"corner_pivot_stop_angle:{angle_abs:.1f}>={stop_angle:.1f}",
                )
            max_duration = max(0.0, as_float(self.cfg.get("max_duration_s"), 4.0))
            if max_duration > 0.0 and now - self.started_at >= max_duration:
                self._reset()
                self.last_debug = f"fault:timeout>{max_duration:.1f}s"
                return LaneDriveCommand(
                    active=True,
                    valid=False,
                    confidence=0.0,
                    state="CORNER_PIVOT_TIMEOUT",
                    steer=0.0,
                    speed=0.0,
                    brake=True,
                    reason=f"corner_pivot_timeout>{max_duration:.1f}s",
                )
            if self._exit_ready(stats):
                self.exit_count += 1
            else:
                self.exit_count = 0
            if self.exit_count >= max(1, as_int(self.cfg.get("exit_confirm_frames"), 2)):
                self.last_exit_turn_sign = self.direction
                self._reset()
                self.last_debug = f"exit:dashed_reacquired:{self._format_stats(stats)}"
                return command
            return self._pivot_command(stats)

        reject_reason = self._entry_reject_reason(stats, angle_abs)
        if reject_reason is None:
            sign = float(stats["turn_sign"])
            if sign == self.last_candidate_sign:
                self.enter_count += 1
            else:
                self.last_candidate_sign = sign
                self.enter_count = 1
            self.last_debug = (
                f"candidate:{self.enter_count}/{max(1, as_int(self.cfg.get('enter_confirm_frames'), 2))}:"
                f"{self._format_stats(stats)}"
            )
            if self.enter_count >= max(1, as_int(self.cfg.get("enter_confirm_frames"), 2)):
                self.active = True
                self.direction = sign
                self.last_exit_turn_sign = 0.0
                self.started_at = now
                self.exit_count = 0
                return self._pivot_command(stats)
        else:
            self.enter_count = 0
            self.last_candidate_sign = 0.0
            self.last_debug = f"reject:{reject_reason}:{self._format_stats(stats)}"
        return command

    def _select_estimate(
        self,
        estimate: Optional[LaneEstimate],
        results: Dict[str, CameraLaneResult],
        bev_width: int,
        bev_height: int,
    ) -> Optional[LaneEstimate]:
        candidates: List[LaneEstimate] = []
        if estimate is not None and estimate.row_estimates:
            candidates.append(estimate)
        candidates.extend(result.estimate for result in results.values() if result.estimate.row_estimates)
        if not candidates:
            return estimate
        dashed_limit = as_int(self.cfg.get("dashed_enter_max_rows"), 1)
        scored: List[Tuple[float, LaneEstimate]] = []
        for candidate in candidates:
            stats = self._lane_stats(candidate, bev_width, bev_height)
            dash_penalty = 0.0 if stats["dashed_rows"] <= dashed_limit else -2.0
            score = (
                dash_penalty
                + 3.0 * stats["corner_confidence"]
                + 0.25 * min(8.0, stats["solid_rows"])
                + 0.04 * min(15.0, stats["rows"])
                + 0.25 * stats["confidence"]
            )
            scored.append((float(score), candidate))
        return max(scored, key=lambda item: item[0])[1]

    def _lane_stats(self, estimate: LaneEstimate, bev_width: int, bev_height: int) -> Dict[str, float]:
        rows = list(estimate.row_estimates)
        near_y = bev_height * clamp(as_float(self.cfg.get("near_y_min_ratio"), 0.70), 0.0, 1.0)
        dashed_rows = sum(1 for row in rows if row.dashed_x is not None)
        near_dashed_rows = sum(1 for row in rows if row.dashed_x is not None and float(row.y) >= near_y)
        pair_rows = sum(1 for row in rows if row.method == "solid_pair_midpoint")
        single_rows = sum(1 for row in rows if row.method == "single_solid_recovery")
        solid_rows = pair_rows + single_rows
        curvature_proxy, turn_sign, corner_confidence = corner_features_from_rows(rows, self.config, bev_width)
        return {
            "confidence": float(estimate.confidence),
            "rows": float(len(rows)),
            "dashed_rows": float(dashed_rows),
            "near_dashed_rows": float(near_dashed_rows),
            "pair_rows": float(pair_rows),
            "single_rows": float(single_rows),
            "solid_rows": float(solid_rows),
            "curvature_proxy": float(curvature_proxy),
            "turn_sign": float(turn_sign),
            "corner_confidence": float(corner_confidence),
        }

    def _suppress_during_post_corner_bias(self, command: LaneDriveCommand) -> bool:
        route_cfg = self.config.get("route_controller", {}) or {}
        if not as_bool(route_cfg.get("post_corner_bias_disable_corner_pivot"), True):
            return False
        return "POST_CORNER_BIAS" in str(command.state).upper()

    def _entry_reject_reason(self, stats: Dict[str, float], angle_abs: Optional[float]) -> Optional[str]:
        if stats["turn_sign"] == 0.0:
            return "turn_sign"
        if stats["confidence"] < as_float(self.cfg.get("min_lane_confidence"), 0.12):
            return "lane_conf"
        if stats["dashed_rows"] > as_int(self.cfg.get("dashed_enter_max_rows"), 1):
            return "dashed_rows"
        if stats["solid_rows"] < as_int(self.cfg.get("solid_enter_min_rows"), 4):
            return "solid_rows"
        if stats["pair_rows"] < as_int(self.cfg.get("solid_enter_min_pair_rows"), 2):
            return "pair_rows"
        if stats["corner_confidence"] < as_float(self.cfg.get("corner_enter_confidence"), 0.55):
            return "corner_conf"
        if not as_bool(self.cfg.get("angle_stop_enabled"), True):
            return None
        max_angle = abs(as_float(self.cfg.get("max_abs_trailer_angle_deg"), 40.0))
        if angle_abs is not None and angle_abs >= max_angle:
            return "angle"
        return None

    def _exit_ready(self, stats: Dict[str, float]) -> bool:
        dashed_exit = as_int(self.cfg.get("dashed_exit_min_rows"), 4)
        near_exit = as_int(self.cfg.get("near_exit_min_dashed_rows"), 2)
        return stats["dashed_rows"] >= dashed_exit or stats["near_dashed_rows"] >= near_exit

    def _pivot_command(self, stats: Dict[str, float]) -> LaneDriveCommand:
        raw_direction = self.direction if self.direction != 0.0 else float(stats["turn_sign"])
        direction_sign = -1.0 if as_float(self.cfg.get("direction_sign"), 1.0) < 0.0 else 1.0
        direction = raw_direction * direction_sign
        state = "CORNER_PIVOT_LEFT" if direction < 0.0 else "CORNER_PIVOT_RIGHT"
        reason = (
            f"dash={int(stats['dashed_rows'])} near_dash={int(stats['near_dashed_rows'])} "
            f"solid={int(stats['solid_rows'])} pair={int(stats['pair_rows'])} "
            f"corner={stats['corner_confidence']:.2f} curv={stats['curvature_proxy']:+.2f} "
            f"raw_dir={raw_direction:+.0f} sign={direction_sign:+.0f}"
        )
        self.last_debug = f"active:{reason}"
        return LaneDriveCommand(
            active=True,
            valid=True,
            confidence=max(as_float(self.cfg.get("min_lane_confidence"), 0.12), stats["confidence"]),
            state=state,
            steer=-1.0 if direction < 0.0 else 1.0,
            speed=0.0,
            brake=False,
            reason=reason,
        )

    def _reset(self) -> None:
        self.active = False
        self.direction = 0.0
        self.started_at = 0.0
        self.enter_count = 0
        self.exit_count = 0
        self.last_candidate_sign = 0.0

    @staticmethod
    def _format_stats(stats: Dict[str, float]) -> str:
        return (
            f"dash={int(stats['dashed_rows'])} near={int(stats['near_dashed_rows'])} "
            f"solid={int(stats['solid_rows'])} pair={int(stats['pair_rows'])} "
            f"corner={stats['corner_confidence']:.2f} curv={stats['curvature_proxy']:+.2f} "
            f"dir={stats['turn_sign']:+.0f} conf={stats['confidence']:.2f}"
        )


def make_drive_command(estimate: LaneEstimate, config: Dict[str, Any], active: bool) -> LaneDriveCommand:
    min_drive_conf = as_float((config.get("lane", {}) or {}).get("min_drive_confidence"), 0.08)
    valid = bool(estimate.valid and estimate.confidence >= min_drive_conf)
    return LaneDriveCommand(
        active=active,
        valid=valid,
        confidence=float(estimate.confidence),
        state=estimate.state,
        steer=float(estimate.steer),
        speed=float(estimate.speed),
        brake=not active,
        reason=estimate.reason,
    )


def lane_result_weight(result: CameraLaneResult, config: Dict[str, Any]) -> float:
    lane_cfg = config.get("lane", {}) or {}
    min_drive_conf = as_float(lane_cfg.get("min_drive_confidence"), 0.08)
    estimate = result.estimate
    if not estimate.valid or estimate.confidence < min_drive_conf:
        return 0.0
    row_target = max(1.0, as_float(lane_cfg.get("row_samples"), 15.0))
    row_ratio = min(1.0, len(estimate.row_estimates) / row_target)
    return max(0.0, estimate.confidence) * max(0.25, row_ratio)


def fuse_lane_results(results: Dict[str, CameraLaneResult], config: Dict[str, Any], active: bool) -> LaneDriveCommand:
    weighted = []
    for result in results.values():
        weight = lane_result_weight(result, config)
        if weight > 0.0:
            weighted.append((weight, result))
    if not weighted:
        if results:
            best = max(results.values(), key=lambda item: item.estimate.confidence)
            command = make_drive_command(best.estimate, config, active)
            command.valid = False
            command.state = f"FUSED_LOST:{best.camera_key}:{best.estimate.state}"
            command.reason = f"dual_lane_lost best={best.camera_key}:{best.estimate.reason}"
            return command
        return LaneDriveCommand(active, False, 0.0, "NO_CAMERA_FRAME", 0.0, 0.0, True, "no_camera_frame")

    total = sum(weight for weight, _ in weighted)
    steer = sum(weight * result.estimate.steer for weight, result in weighted) / max(1e-6, total)
    confidence = sum(weight * result.estimate.confidence for weight, result in weighted) / max(1e-6, total)
    speed = min(result.estimate.speed for _, result in weighted)
    cameras = ",".join(result.camera_key for _, result in weighted)
    states = ",".join(f"{result.camera_key}:{result.estimate.state}" for _, result in weighted)
    return LaneDriveCommand(
        active=active,
        valid=True,
        confidence=float(confidence),
        state=f"FUSED_LANE:{cameras}",
        steer=float(clamp(steer, -1.0, 1.0)),
        speed=float(max(0.0, speed)),
        brake=not active,
        reason=f"confidence_weighted_average {states}",
    )


def make_combined_debug_panel(
    results: Dict[str, CameraLaneResult],
    fused: LaneDriveCommand,
    wheel: WheelCommand,
    armed: bool,
    fused_bev_panel: Optional[np.ndarray] = None,
    route_debug: Optional[RouteControlDebug] = None,
    angle_state: Optional[TrailerAngleState] = None,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None
    panels = []
    for key in sorted(results.keys(), reverse=True):
        panel = results[key].panel.copy()
        if cv2 is not None:
            cv2.putText(panel, key, (12, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    if fused_bev_panel is not None:
        panel = fused_bev_panel.copy()
        if cv2 is not None:
            cv2.putText(panel, "fused BEV", (12, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    if not panels:
        return np.zeros((360, 960, 3), dtype=np.uint8)
    target_h = min(panel.shape[0] for panel in panels)
    resized = []
    for panel in panels:
        if cv2 is not None and panel.shape[0] != target_h:
            width = int(round(panel.shape[1] * target_h / panel.shape[0]))
            panel = cv2.resize(panel, (width, target_h), interpolation=cv2.INTER_AREA)
        resized.append(panel)
    out = np.hstack(resized) if len(resized) > 1 else resized[0]
    status = [
        f"dotted lane: {'RUNNING' if fused.active else 'PREVIEW'} {fused.state} conf={fused.confidence:.2f}",
        f"fused steer={fused.steer:+.3f} speed={fused.speed:.3f} valid={fused.valid}",
        f"wheel: L={wheel.left:+.3f} R={wheel.right:+.3f} {'ARMED' if armed else 'dry-run'}",
        "keys: p=start, x=stop, q=quit",
    ]
    if route_debug is not None:
        alpha = "--" if route_debug.alpha_deg is None else f"{route_debug.alpha_deg:+.1f}"
        status.insert(
            1,
            f"route={route_debug.mode} band={route_debug.angle_band} "
            f"alpha={alpha}/{route_debug.alpha_ref_deg:+.1f} look={route_debug.lookahead_y_ratio:.2f}",
        )
        status.insert(
            2,
            f"lane={route_debug.lane_state} rows={route_debug.lane_row_count}",
        )
        status.insert(
            3,
            f"u lane={route_debug.u_lane:+.2f} curve={route_debug.u_curve:+.2f} "
            f"trailer={route_debug.u_trailer:+.2f} bias={route_debug.target_bias_norm:+.2f}/{route_debug.target_bias_fraction:.2f} "
            f"corner={route_debug.corner_confidence:.2f}",
        )
    elif angle_state is not None and angle_state.angle_deg is not None:
        status.insert(1, f"trailer angle={angle_state.angle_deg:+.1f} deg src={angle_state.source}")
    draw_status_overlay(out, status)
    draw_trailer_angle_hud(out, config or {}, route_debug)
    return out


def draw_trailer_angle_hud(
    frame_bgr: np.ndarray,
    config: Dict[str, Any],
    route_debug: Optional[RouteControlDebug],
) -> None:
    if route_debug is None:
        return
    try:
        import cv2
    except Exception:
        return

    h, w = frame_bgr.shape[:2]
    rcfg = config.get("route_controller", {}) or {}
    deadband = abs(as_float(rcfg.get("straight_deadband_abs_angle_deg"), 5.0))
    caution = abs(as_float(rcfg.get("straight_caution_abs_angle_deg"), 15.0))
    corner = abs(as_float(rcfg.get("corner_allow_abs_angle_deg"), 45.0))
    recovery = abs(as_float(rcfg.get("recovery_start_abs_angle_deg"), 48.0))
    hard_stop = max(1.0, abs(as_float(rcfg.get("hard_stop_abs_angle_deg"), 62.0)))
    deadband = min(deadband, hard_stop)
    caution = min(max(caution, deadband), hard_stop)
    corner = min(max(corner, caution), hard_stop)
    recovery = min(max(recovery, corner), hard_stop)

    margin = 12
    box_w = min(340, max(240, w - 2 * margin))
    box_h = 92
    x0 = max(margin, w - box_w - margin)
    y0 = max(margin, h - box_h - margin)
    x1 = min(w - margin, x0 + box_w)
    y1 = min(h - margin, y0 + box_h)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame_bgr, 0.32, 0.0, frame_bgr)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (230, 230, 230), 1)

    alpha_text = "--" if route_debug.alpha_deg is None else f"{route_debug.alpha_deg:+.1f}"
    title = f"trailer angle {alpha_text} deg  band={route_debug.angle_band}"
    cv2.putText(frame_bgr, title, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, title, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (245, 245, 245), 1, cv2.LINE_AA)

    bar_x0 = x0 + 18
    bar_x1 = x1 - 18
    bar_y = y0 + 56
    bar_w = max(1, bar_x1 - bar_x0)

    def x_for(angle_deg: float) -> int:
        angle = clamp(float(angle_deg), -hard_stop, hard_stop)
        return int(round(bar_x0 + ((angle + hard_stop) / (2.0 * hard_stop)) * bar_w))

    segments = [
        (-hard_stop, -recovery, (25, 25, 210)),
        (-recovery, -corner, (0, 125, 255)),
        (-corner, -caution, (0, 190, 255)),
        (-caution, -deadband, (0, 215, 215)),
        (-deadband, deadband, (80, 205, 80)),
        (deadband, caution, (0, 215, 215)),
        (caution, corner, (0, 190, 255)),
        (corner, recovery, (0, 125, 255)),
        (recovery, hard_stop, (25, 25, 210)),
    ]
    for start, end, color in segments:
        sx = x_for(start)
        ex = x_for(end)
        if ex > sx:
            cv2.line(frame_bgr, (sx, bar_y), (ex, bar_y), color, 9, cv2.LINE_AA)

    zero_x = x_for(0.0)
    ref_x = x_for(route_debug.alpha_ref_deg)
    cv2.line(frame_bgr, (zero_x, bar_y - 14), (zero_x, bar_y + 14), (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (ref_x, bar_y - 13), (ref_x, bar_y + 13), (255, 255, 0), 2, cv2.LINE_AA)
    if route_debug.alpha_deg is not None:
        alpha_x = x_for(route_debug.alpha_deg)
        cv2.circle(frame_bgr, (alpha_x, bar_y), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (alpha_x, bar_y), 7, (0, 0, 0), 1, cv2.LINE_AA)

    footer = f"ref={route_debug.alpha_ref_deg:+.1f}  rate={route_debug.alpha_rate_deg_s:+.1f} deg/s"
    cv2.putText(frame_bgr, footer, (x0 + 12, y1 - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, footer, (x0 + 12, y1 - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)


def resize_exact(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


def make_status_frame(width: int, height: int, lines: List[str]) -> np.ndarray:
    frame = np.zeros((max(90, height), max(160, width), 3), dtype=np.uint8)
    draw_status_overlay(frame, lines, origin=(14, 30))
    return frame


def fit_into(frame_bgr: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, float, int, int]:
    try:
        import cv2
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8), 1.0, 0, 0
    out = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return out, 1.0, 0, 0
    scale = min(width / float(w), height / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    off_x = (width - new_w) // 2
    off_y = (height - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return out, scale, off_x, off_y


def draw_label(frame_bgr: np.ndarray, text: str, xy: Tuple[int, int], scale: float = 0.58) -> None:
    try:
        import cv2
    except Exception:
        return
    cv2.putText(frame_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def make_roi_zoom(
    frame_bgr: Optional[np.ndarray],
    config: Dict[str, Any],
    camera_key: str,
    detection: Optional[PanelDetection],
    estimate: Optional[AngleEstimate],
    width: int,
    height: int,
) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None
    if frame_bgr is None:
        out = np.zeros((height, width, 3), dtype=np.uint8)
        draw_label(out, f"{camera_key} unavailable", (14, 30))
        return out

    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    rx0, ry0, rx1, ry1 = camera_roi(config, camera_key).to_abs(frame_bgr.shape)
    ex0, ey0, ex1, ey1 = panel_detection_roi_abs(frame_bgr.shape, (rx0, ry0, rx1, ry1), panel_cfg)
    roi = frame_bgr[ey0:ey1, ex0:ex1].copy()
    if roi.size == 0:
        out = np.zeros((height, width, 3), dtype=np.uint8)
        draw_label(out, f"{camera_key} empty ROI", (14, 30))
        return out

    out, scale, off_x, off_y = fit_into(roi, width, height)
    if cv2 is not None:
        base_p0 = (off_x + int(round((rx0 - ex0) * scale)), off_y + int(round((ry0 - ey0) * scale)))
        base_p1 = (off_x + int(round((rx1 - ex0) * scale)), off_y + int(round((ry1 - ey0) * scale)))
        cv2.rectangle(out, base_p0, base_p1, (255, 220, 0), 1)
    if cv2 is not None and detection is not None and detection.ok:
        det_abs = (
            rx0 + detection.x0,
            ry0 + detection.y0,
            rx0 + detection.x1,
            ry0 + detection.y1,
        )
        p0 = (off_x + int(round((det_abs[0] - ex0) * scale)), off_y + int(round((det_abs[1] - ey0) * scale)))
        p1 = (off_x + int(round((det_abs[2] - ex0) * scale)), off_y + int(round((det_abs[3] - ey0) * scale)))
        center = (
            off_x + int(round((rx0 + detection.center_x - ex0) * scale)),
            off_y + int(round((ry0 + detection.center_y - ey0) * scale)),
        )
        cv2.rectangle(out, p0, p1, (0, 255, 255), 2)
        cv2.circle(out, center, 5, (0, 255, 255), -1)
    det_text = "no panel" if detection is None or not detection.ok else f"panel conf={detection.confidence:.2f}"
    angle_text = "--" if estimate is None or estimate.angle_deg is None else f"{estimate.angle_deg:+.1f} deg"
    src_text = "none" if estimate is None else estimate.source
    draw_label(out, f"{camera_key} YOLO crop  {det_text}  angle={angle_text} src={src_text}", (14, 28))
    return out


def panel_estimate_lines(detection: Optional[PanelDetection], estimate: Optional[AngleEstimate]) -> List[str]:
    det_ok = detection is not None and detection.ok
    det_conf = "" if detection is None else f"{detection.confidence:.2f}"
    det_source = "none" if detection is None else str(detection.source)
    est_ok = estimate is not None and estimate.ok
    angle_text = "--" if estimate is None or estimate.angle_deg is None else f"{estimate.angle_deg:+.1f}"
    est_conf = "" if estimate is None else f"{estimate.confidence:.2f}"
    est_source = "none" if estimate is None else str(estimate.source)
    message = "" if estimate is None else str(estimate.message)
    if len(message) > 54:
        message = message[:51] + "..."
    lines = [
        f"det={'Y' if det_ok else 'N'} conf={det_conf or '--'} src={det_source}",
        f"angle={angle_text} deg est_conf={est_conf or '--'} est={'OK' if est_ok else 'NO'} src={est_source}",
    ]
    if message:
        lines.append(f"msg={message}")
    return lines


def panel_console_token(camera_key: str, detection: Optional[PanelDetection], estimate: Optional[AngleEstimate]) -> str:
    if detection is None:
        det_text = "none"
    elif detection.ok:
        det_text = f"det{detection.confidence:.2f}"
    else:
        det_text = str(detection.source)
    angle_text = "--" if estimate is None or estimate.angle_deg is None else f"{estimate.angle_deg:+.1f}"
    est_source = "none" if estimate is None else str(estimate.source)
    return f"{camera_key}:{det_text}:{angle_text}:{est_source}"


def make_yolo_camera_panel(
    config: Dict[str, Any],
    frames_bgr: Dict[str, np.ndarray],
    detections: Dict[str, PanelDetection],
    angle_estimates: Dict[str, AngleEstimate],
    fused: LaneDriveCommand,
    wheel: WheelCommand,
    armed: bool,
    angle_state: TrailerAngleState,
    route_debug: Optional[RouteControlDebug],
) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None

    http_cfg = config.get("http_stream", {}) or {}
    out_w = max(320, as_int(http_cfg.get("width"), 960))
    out_h = max(180, as_int(http_cfg.get("height"), 360))
    keys = ("cam1", "cam0")
    tile_w = max(1, out_w // len(keys))
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    imgsz = as_int(panel_cfg.get("imgsz"), 416)

    for idx, key in enumerate(keys):
        x = idx * tile_w
        width = out_w - x if idx == len(keys) - 1 else tile_w
        frame = frames_bgr.get(key)
        if frame is None:
            tile = np.zeros((out_h, width, 3), dtype=np.uint8)
            draw_status_overlay(tile, [f"{key} unavailable"], origin=(12, 24))
            canvas[:, x:x + width] = tile
            continue

        rx0, ry0, rx1, ry1 = camera_roi(config, key).to_abs(frame.shape)
        ex0, ey0, ex1, ey1 = panel_detection_roi_abs(frame.shape, (rx0, ry0, rx1, ry1), panel_cfg)
        roi_cfg = (((config.get("camera", {}) or {}).get("cameras", {}) or {}).get(key, {}) or {}).get("roi", {}) or {}
        roi_text = (
            f"roi x={as_float(roi_cfg.get('x'), 0.0):.5f} y={as_float(roi_cfg.get('y'), 0.0):.5f} "
            f"w={as_float(roi_cfg.get('w'), 1.0):.5f} h={as_float(roi_cfg.get('h'), 1.0):.5f}"
        )
        roi_px_text = f"base px [{rx0},{ry0}] [{rx1},{ry1}]"
        detect_px_text = f"detect px [{ex0},{ey0}] [{ex1},{ey1}]"
        roi = frame[ey0:ey1, ex0:ex1].copy()
        if roi.size == 0:
            tile = np.zeros((out_h, width, 3), dtype=np.uint8)
            draw_status_overlay(tile, [f"{key} empty YOLO ROI"], origin=(12, 24))
            canvas[:, x:x + width] = tile
            continue

        tile, scale, off_x, off_y = fit_into(roi, width, out_h)
        det = detections.get(key)
        if cv2 is not None:
            base_p0 = (off_x + int(round((rx0 - ex0) * scale)), off_y + int(round((ry0 - ey0) * scale)))
            base_p1 = (off_x + int(round((rx1 - ex0) * scale)), off_y + int(round((ry1 - ey0) * scale)))
            cv2.rectangle(tile, base_p0, base_p1, (255, 220, 0), 1)
        if cv2 is not None and det is not None and det.ok:
            det_abs = (rx0 + det.x0, ry0 + det.y0, rx0 + det.x1, ry0 + det.y1)
            p0 = (off_x + int(round((det_abs[0] - ex0) * scale)), off_y + int(round((det_abs[1] - ey0) * scale)))
            p1 = (off_x + int(round((det_abs[2] - ex0) * scale)), off_y + int(round((det_abs[3] - ey0) * scale)))
            center = (
                off_x + int(round((rx0 + det.center_x - ex0) * scale)),
                off_y + int(round((ry0 + det.center_y - ey0) * scale)),
            )
            cv2.rectangle(tile, p0, p1, (0, 255, 255), 2)
            cv2.circle(tile, center, 5, (0, 255, 255), -1)
        det_text = "no panel" if det is None or not det.ok else f"panel conf={det.confidence:.2f}"
        est = angle_estimates.get(key)
        draw_status_overlay(
            tile,
            [
                f"{key} YOLO crop {roi.shape[1]}x{roi.shape[0]} -> imgsz {imgsz}",
                roi_text,
                roi_px_text,
                detect_px_text,
                f"{det_text}  source={getattr(det, 'source', '-') if det is not None else '-'}",
            ]
            + panel_estimate_lines(det, est),
            origin=(12, 24),
        )
        canvas[:, x:x + width] = tile

    angle_text = "--" if angle_state.angle_deg is None else f"{angle_state.angle_deg:+.1f} deg"
    mode_text = route_debug.mode if route_debug is not None else "NO_ROUTE"
    band_text = route_debug.angle_band if route_debug is not None else angle_state.reason
    draw_status_overlay(
        canvas,
        [
            f"YOLO camera view  angle={angle_text} conf={angle_state.confidence:.2f} src={angle_state.source}",
            f"{'RUNNING' if fused.active else 'PREVIEW'} {mode_text}/{band_text} steer={fused.steer:+.2f} wheel={wheel.left:+.2f}/{wheel.right:+.2f} {'ARMED' if armed else 'dry-run'}",
        ],
        origin=(12, max(24, out_h - 46)),
    )
    if cv2 is not None:
        for idx in range(1, len(keys)):
            xx = idx * tile_w
            cv2.line(canvas, (xx, 0), (xx, out_h), (80, 80, 80), 1)
    return canvas


def make_roi_editor_panel(
    config: Dict[str, Any],
    frames_bgr: Dict[str, np.ndarray],
    detections: Dict[str, PanelDetection],
    angle_state: TrailerAngleState,
) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None

    http_cfg = config.get("http_stream", {}) or {}
    out_w = max(320, as_int(http_cfg.get("width"), 960))
    out_h = max(180, as_int(http_cfg.get("height"), 360))
    keys = ("cam1", "cam0")
    tile_w = max(1, out_w // len(keys))
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for idx, key in enumerate(keys):
        x = idx * tile_w
        width = out_w - x if idx == len(keys) - 1 else tile_w
        frame = frames_bgr.get(key)
        if frame is None:
            tile = np.zeros((out_h, width, 3), dtype=np.uint8)
            draw_status_overlay(tile, [f"{key} unavailable"], origin=(12, 24))
            canvas[:, x:x + width] = tile
            continue

        tile, scale, off_x, off_y = fit_into(frame, width, out_h)
        rx0, ry0, rx1, ry1 = camera_roi(config, key).to_abs(frame.shape)
        roi_cfg = (((config.get("camera", {}) or {}).get("cameras", {}) or {}).get(key, {}) or {}).get("roi", {}) or {}
        color = (0, 220, 255) if key == "cam1" else (80, 255, 110)
        if cv2 is not None:
            p0 = (off_x + int(round(rx0 * scale)), off_y + int(round(ry0 * scale)))
            p1 = (off_x + int(round(rx1 * scale)), off_y + int(round(ry1 * scale)))
            cv2.rectangle(tile, p0, p1, color, 2)
            det = detections.get(key)
            if det is not None and det.ok:
                d0 = (off_x + int(round((rx0 + det.x0) * scale)), off_y + int(round((ry0 + det.y0) * scale)))
                d1 = (off_x + int(round((rx0 + det.x1) * scale)), off_y + int(round((ry0 + det.y1) * scale)))
                center = (
                    off_x + int(round((rx0 + det.center_x) * scale)),
                    off_y + int(round((ry0 + det.center_y) * scale)),
                )
                cv2.rectangle(tile, d0, d1, (0, 255, 255), 2)
                cv2.circle(tile, center, 5, (0, 255, 255), -1)
        roi_text = (
            f"x={as_float(roi_cfg.get('x'), 0.0):.5f} y={as_float(roi_cfg.get('y'), 0.0):.5f} "
            f"w={as_float(roi_cfg.get('w'), 1.0):.5f} h={as_float(roi_cfg.get('h'), 1.0):.5f}"
        )
        det_text = "no panel" if detections.get(key) is None or not detections[key].ok else f"panel conf={detections[key].confidence:.2f}"
        draw_status_overlay(
            tile,
            [
                f"{key} ROI EDIT drag on this camera",
                roi_text,
                det_text,
            ],
            origin=(12, 24),
        )
        canvas[:, x:x + width] = tile

    angle_text = "--" if angle_state.angle_deg is None else f"{angle_state.angle_deg:+.1f} deg"
    draw_status_overlay(
        canvas,
        [f"ROI editor  angle={angle_text} conf={angle_state.confidence:.2f} src={angle_state.source}"],
        origin=(12, max(24, out_h - 22)),
    )
    if cv2 is not None:
        for idx in range(1, len(keys)):
            xx = idx * tile_w
            cv2.line(canvas, (xx, 0), (xx, out_h), (80, 80, 80), 1)
    return canvas


def make_angle_camera_panel(
    config: Dict[str, Any],
    frames_bgr: Dict[str, np.ndarray],
    detections: Dict[str, PanelDetection],
    angle_estimates: Dict[str, AngleEstimate],
    fused: LaneDriveCommand,
    wheel: WheelCommand,
    armed: bool,
    angle_state: TrailerAngleState,
    route_debug: Optional[RouteControlDebug],
) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None

    out_w, out_h = 1280, 720
    half_w = out_w // 2
    half_h = out_h // 2
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    keys = ("cam1", "cam0")
    colors = {"cam1": (60, 220, 255), "cam0": (120, 255, 120)}

    for idx, key in enumerate(keys):
        x = idx * half_w
        frame = frames_bgr.get(key)
        if frame is None:
            view = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            draw_label(view, f"{key} unavailable", (14, 30))
        else:
            view = frame.copy()
            roi_xyxy = camera_roi(config, key).to_abs(view.shape)
            draw_panel_overlay(view, roi_xyxy, detections.get(key), colors.get(key, (0, 255, 255)))
            view = resize_exact(view, half_w, half_h)
            est = angle_estimates.get(key)
            angle_text = "--" if est is None or est.angle_deg is None else f"{est.angle_deg:+.1f} deg"
            draw_label(view, f"{key} full camera  angle={angle_text}", (14, 30))
        canvas[0:half_h, x:x + half_w] = view

        zoom = make_roi_zoom(frame, config, key, detections.get(key), angle_estimates.get(key), half_w, half_h)
        canvas[half_h:out_h, x:x + half_w] = zoom

    angle_text = "--" if angle_state.angle_deg is None else f"{angle_state.angle_deg:+.1f} deg"
    rate_text = f"{angle_state.angle_rate_deg_s:+.1f} deg/s"
    mode_text = route_debug.mode if route_debug is not None else "NO_ROUTE"
    band_text = route_debug.angle_band if route_debug is not None else angle_state.reason
    status = [
        f"trailer angle={angle_text}  conf={angle_state.confidence:.2f}  src={angle_state.source}  rate={rate_text}",
        f"{'RUNNING' if fused.active else 'PREVIEW'} {mode_text}/{band_text}  lane_conf={fused.confidence:.2f}  steer={fused.steer:+.2f}  wheel={wheel.left:+.2f}/{wheel.right:+.2f} {'ARMED' if armed else 'dry-run'}",
    ]
    draw_status_overlay(canvas, status)
    if cv2 is not None:
        cv2.line(canvas, (half_w, 0), (half_w, out_h), (80, 80, 80), 1)
        cv2.line(canvas, (0, half_h), (out_w, half_h), (80, 80, 80), 1)
    return canvas


def make_http_debug_panel(
    config: Dict[str, Any],
    full_panel: np.ndarray,
    results: Dict[str, CameraLaneResult],
    fused: LaneDriveCommand,
    wheel: WheelCommand,
    armed: bool,
    fused_bev_panel: Optional[np.ndarray],
    route_debug: Optional[RouteControlDebug] = None,
    frames_bgr: Optional[Dict[str, np.ndarray]] = None,
    detections: Optional[Dict[str, PanelDetection]] = None,
    angle_estimates: Optional[Dict[str, AngleEstimate]] = None,
    angle_state: Optional[TrailerAngleState] = None,
) -> np.ndarray:
    view = str((config.get("http_stream", {}) or {}).get("view", "fused_bev")).lower()
    if view in {"roi_edit", "roi_editor", "edit_roi"} and frames_bgr is not None and angle_state is not None:
        return make_roi_editor_panel(
            config,
            frames_bgr,
            detections or {},
            angle_state,
        )
    if view in {"yolo", "yolo_camera", "panel", "panel_yolo", "roi", "mirror_roi"} and frames_bgr is not None and angle_state is not None:
        return make_yolo_camera_panel(
            config,
            frames_bgr,
            detections or {},
            angle_estimates or {},
            fused,
            wheel,
            armed,
            angle_state,
            route_debug,
        )
    if view in {"angle_camera", "angle", "camera", "full_camera"} and frames_bgr is not None and angle_state is not None:
        return make_angle_camera_panel(
            config,
            frames_bgr,
            detections or {},
            angle_estimates or {},
            fused,
            wheel,
            armed,
            angle_state,
            route_debug,
        )
    if view in {"fused_bev", "bev", "dual_bev"} and fused_bev_panel is not None:
        panel = fused_bev_panel.copy()
        status = [
            f"HTTP BEV {'RUNNING' if fused.active else 'PREVIEW'} {fused.state}",
            f"steer={fused.steer:+.3f} speed={fused.speed:.3f} conf={fused.confidence:.2f}",
            f"wheel L={wheel.left:+.3f} R={wheel.right:+.3f} {'ARMED' if armed else 'dry-run'}",
        ]
        if route_debug is not None:
            alpha = "--" if route_debug.alpha_deg is None else f"{route_debug.alpha_deg:+.1f}"
            status.insert(
                1,
                f"route={route_debug.mode} alpha={alpha}/{route_debug.alpha_ref_deg:+.1f} "
                f"look={route_debug.lookahead_y_ratio:.2f} bias={route_debug.target_bias_norm:+.2f}/{route_debug.target_bias_fraction:.2f} "
                f"corner={route_debug.corner_confidence:.2f}",
            )
            status.insert(
                2,
                f"lane={route_debug.lane_state} rows={route_debug.lane_row_count}",
            )
            status.insert(
                3,
                f"u lane={route_debug.u_lane:+.2f} curve={route_debug.u_curve:+.2f} trailer={route_debug.u_trailer:+.2f}",
            )
        draw_status_overlay(panel, status)
        draw_trailer_angle_hud(panel, config, route_debug)
        return panel
    if view in {"camera_bev", "per_camera_bev"}:
        panels = [result.panel for _key, result in sorted(results.items(), reverse=True)]
        if panels:
            try:
                import cv2

                target_h = min(panel.shape[0] for panel in panels)
                resized = []
                for panel in panels:
                    if panel.shape[0] != target_h:
                        width = int(round(panel.shape[1] * target_h / panel.shape[0]))
                        panel = cv2.resize(panel, (width, target_h), interpolation=cv2.INTER_AREA)
                    resized.append(panel)
                panel = np.hstack(resized)
                draw_trailer_angle_hud(panel, config, route_debug)
                return panel
            except Exception:
                panel = panels[0].copy()
                draw_trailer_angle_hud(panel, config, route_debug)
                return panel
    panel = full_panel.copy()
    draw_trailer_angle_hud(panel, config, route_debug)
    return panel


def panel_log_values(detection: Optional[PanelDetection], estimate: Optional[AngleEstimate], prefix: str) -> Dict[str, Any]:
    det_ok = detection is not None and detection.ok
    features = detection.features() if det_ok and detection is not None else {}
    return {
        f"{prefix}_panel_det": int(det_ok),
        f"{prefix}_panel_conf": "" if detection is None else f"{detection.confidence:.4f}",
        f"{prefix}_panel_cx_norm": "" if not features else f"{features['center_x_norm']:.5f}",
        f"{prefix}_panel_cy_norm": "" if not features else f"{features['center_y_norm']:.5f}",
        f"{prefix}_panel_w_norm": "" if not features else f"{features['width_norm']:.5f}",
        f"{prefix}_panel_h_norm": "" if not features else f"{features['height_norm']:.5f}",
        f"{prefix}_panel_source": "" if detection is None else detection.source,
        f"{prefix}_panel_angle_ok": "" if estimate is None else int(bool(estimate.ok)),
        f"{prefix}_panel_angle_deg": "" if estimate is None or estimate.angle_deg is None else f"{estimate.angle_deg:.4f}",
        f"{prefix}_panel_angle_conf": "" if estimate is None else f"{estimate.confidence:.4f}",
        f"{prefix}_panel_angle_source": "" if estimate is None else estimate.source,
        f"{prefix}_panel_angle_message": "" if estimate is None else estimate.message,
    }


def make_log_writer(run_dir: Path) -> Tuple[csv.DictWriter, Any]:
    csv_path = run_dir / "dotted_lane_following_log.csv"
    f = csv_path.open("w", newline="", encoding="utf-8")
    fields = [
        "frame_idx",
        "timestamp_monotonic",
        "wall_time",
        "fps",
        "active",
        "fused_valid",
        "fused_confidence",
        "fused_state",
        "fused_steer",
        "fused_speed",
        "trailer_angle_deg",
        "trailer_angle_confidence",
        "trailer_angle_rate_deg_s",
        "trailer_angle_source",
        "trailer_angle_age_s",
        "cam0_valid",
        "cam0_confidence",
        "cam0_state",
        "cam0_steer",
        "cam0_speed",
        "cam0_rows",
        "cam0_dashed_rows",
        "cam0_panel_det",
        "cam0_panel_conf",
        "cam0_panel_cx_norm",
        "cam0_panel_cy_norm",
        "cam0_panel_w_norm",
        "cam0_panel_h_norm",
        "cam0_panel_source",
        "cam0_panel_angle_ok",
        "cam0_panel_angle_deg",
        "cam0_panel_angle_conf",
        "cam0_panel_angle_source",
        "cam0_panel_angle_message",
        "cam1_valid",
        "cam1_confidence",
        "cam1_state",
        "cam1_steer",
        "cam1_speed",
        "cam1_rows",
        "cam1_dashed_rows",
        "cam1_panel_det",
        "cam1_panel_conf",
        "cam1_panel_cx_norm",
        "cam1_panel_cy_norm",
        "cam1_panel_w_norm",
        "cam1_panel_h_norm",
        "cam1_panel_source",
        "cam1_panel_angle_ok",
        "cam1_panel_angle_deg",
        "cam1_panel_angle_conf",
        "cam1_panel_angle_source",
        "cam1_panel_angle_message",
        "route_mode",
        "route_lane_state",
        "route_lane_rows",
        "route_angle_deg",
        "route_angle_rate_deg_s",
        "route_alpha_ref_deg",
        "route_angle_band",
        "route_lookahead_y_ratio",
        "route_lane_scale",
        "route_trailer_scale",
        "route_speed_scale",
        "route_u_lane",
        "route_u_curve",
        "route_u_trailer",
        "route_u_total",
        "route_target_bias_norm",
        "route_target_bias_fraction",
        "route_curvature_proxy",
        "route_corner_confidence",
        "wheel_left",
        "wheel_right",
        "wheel_sent",
        "reason",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    print(f"[log] {csv_path}")
    return writer, f


def write_log_row(
    writer: csv.DictWriter,
    frame_idx: int,
    fps: float,
    fused: LaneDriveCommand,
    results: Dict[str, CameraLaneResult],
    wheel: WheelCommand,
    angle_state: Optional[TrailerAngleState] = None,
    route_debug: Optional[RouteControlDebug] = None,
    detections: Optional[Dict[str, PanelDetection]] = None,
    angle_estimates: Optional[Dict[str, AngleEstimate]] = None,
) -> None:
    row = {
        "frame_idx": frame_idx,
        "timestamp_monotonic": f"{time.monotonic():.3f}",
        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        "fps": f"{fps:.2f}",
        "active": int(fused.active),
        "fused_valid": int(fused.valid),
        "fused_confidence": f"{fused.confidence:.4f}",
        "fused_state": fused.state,
        "fused_steer": f"{fused.steer:.4f}",
        "fused_speed": f"{fused.speed:.4f}",
        "trailer_angle_deg": "",
        "trailer_angle_confidence": "",
        "trailer_angle_rate_deg_s": "",
        "trailer_angle_source": "",
        "trailer_angle_age_s": "",
        "wheel_left": f"{wheel.left:.4f}",
        "wheel_right": f"{wheel.right:.4f}",
        "wheel_sent": int(wheel.sent),
        "reason": wheel.reason,
        "route_mode": "",
        "route_lane_state": "",
        "route_lane_rows": "",
        "route_angle_deg": "",
        "route_angle_rate_deg_s": "",
        "route_alpha_ref_deg": "",
        "route_angle_band": "",
        "route_lookahead_y_ratio": "",
        "route_lane_scale": "",
        "route_trailer_scale": "",
        "route_speed_scale": "",
        "route_u_lane": "",
        "route_u_curve": "",
        "route_u_trailer": "",
        "route_u_total": "",
        "route_curvature_proxy": "",
        "route_corner_confidence": "",
    }
    if angle_state is not None:
        row.update(
            {
                "trailer_angle_deg": "" if angle_state.angle_deg is None else f"{angle_state.angle_deg:.4f}",
                "trailer_angle_confidence": f"{angle_state.confidence:.4f}",
                "trailer_angle_rate_deg_s": f"{angle_state.angle_rate_deg_s:.4f}",
                "trailer_angle_source": angle_state.source,
                "trailer_angle_age_s": f"{angle_state.age_s:.4f}",
            }
        )
    if route_debug is not None:
        row.update(
            {
                "route_mode": route_debug.mode,
                "route_lane_state": route_debug.lane_state,
                "route_lane_rows": int(route_debug.lane_row_count),
                "route_angle_deg": "" if route_debug.alpha_deg is None else f"{route_debug.alpha_deg:.4f}",
                "route_angle_rate_deg_s": f"{route_debug.alpha_rate_deg_s:.4f}",
                "route_alpha_ref_deg": f"{route_debug.alpha_ref_deg:.4f}",
                "route_angle_band": route_debug.angle_band,
                "route_lookahead_y_ratio": f"{route_debug.lookahead_y_ratio:.4f}",
                "route_lane_scale": f"{route_debug.lane_scale:.4f}",
                "route_trailer_scale": f"{route_debug.trailer_scale:.4f}",
                "route_speed_scale": f"{route_debug.speed_scale:.4f}",
                "route_u_lane": f"{route_debug.u_lane:.4f}",
                "route_u_curve": f"{route_debug.u_curve:.4f}",
                "route_u_trailer": f"{route_debug.u_trailer:.4f}",
                "route_u_total": f"{route_debug.u_total:.4f}",
                "route_target_bias_norm": f"{route_debug.target_bias_norm:.4f}",
                "route_target_bias_fraction": f"{route_debug.target_bias_fraction:.4f}",
                "route_curvature_proxy": f"{route_debug.curvature_proxy:.4f}",
                "route_corner_confidence": f"{route_debug.corner_confidence:.4f}",
            }
        )
    for camera_key in ("cam0", "cam1"):
        result = results.get(camera_key)
        prefix = camera_key
        if result is None:
            row.update(
                {
                    f"{prefix}_valid": 0,
                    f"{prefix}_confidence": "",
                    f"{prefix}_state": "",
                    f"{prefix}_steer": "",
                    f"{prefix}_speed": "",
                    f"{prefix}_rows": "",
                    f"{prefix}_dashed_rows": "",
                }
            )
            continue
        lane = estimate_to_dict(result.estimate)
        row.update(
            {
                f"{prefix}_valid": int(bool(lane["valid"])),
                f"{prefix}_confidence": f"{float(lane['confidence']):.4f}",
                f"{prefix}_state": lane["state"],
                f"{prefix}_steer": f"{float(lane['steer']):.4f}",
                f"{prefix}_speed": f"{float(lane['speed']):.4f}",
                f"{prefix}_rows": int(lane["row_count"]),
                f"{prefix}_dashed_rows": int(lane["dashed_row_count"]),
            }
        )
        row.update(panel_log_values((detections or {}).get(camera_key), (angle_estimates or {}).get(camera_key), prefix))
    writer.writerow(row)


def save_run_config(run_dir: Path, config: Dict[str, Any], bev: BevConfig, lane: LaneFollowerConfig) -> None:
    payload = {
        "config": config,
        "bev": {
            "output_width": bev.output_width,
            "output_height": bev.output_height,
            "src_points_ratio": bev.src_points_ratio,
        },
        "lane": lane.__dict__,
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class E2EDatasetWriter:
    def __init__(self, run_dir: Path, config: Dict[str, Any], bev: BevConfig) -> None:
        self.cfg = config.get("e2e_dataset", {}) or {}
        self.enabled = as_bool(self.cfg.get("enabled"), False)
        self.sample_every = max(1, as_int(self.cfg.get("sample_every"), 1))
        self.save_only_active = as_bool(self.cfg.get("save_only_active"), True)
        self.save_only_sent = as_bool(self.cfg.get("save_only_sent"), True)
        self.save_invalid = as_bool(self.cfg.get("save_invalid"), False)
        self.save_input_gray = as_bool(self.cfg.get("save_input_gray"), True)
        self.save_color_preview = as_bool(self.cfg.get("save_color_preview"), False)
        self.sample_index = 0
        self.csv_file = None
        self.writer: Optional[csv.DictWriter] = None
        self.cv2 = None
        self.bev = bev
        if not self.enabled:
            return

        try:
            import cv2

            self.cv2 = cv2
        except Exception as exc:
            self.enabled = False
            print(f"[e2e] disabled: OpenCV unavailable for PNG writing: {exc}")
            return

        base_dir = run_dir / str(self.cfg.get("dir", "e2e_dataset"))
        self.base_dir = base_dir
        self.mask_dir = base_dir / str(self.cfg.get("bev_mask_dir", "bev_masks"))
        self.input_dir = base_dir / str(self.cfg.get("bev_input_dir", "bev_input"))
        self.color_dir = base_dir / str(self.cfg.get("bev_color_dir", "bev_color"))
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        if self.save_input_gray:
            self.input_dir.mkdir(parents=True, exist_ok=True)
        if self.save_color_preview:
            self.color_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = base_dir / str(self.cfg.get("metadata_csv", "metadata.csv"))
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self._fields())
        self.writer.writeheader()
        self._write_manifest(config)
        print(f"[e2e] dataset enabled: {base_dir}")

    @staticmethod
    def _fields() -> List[str]:
        fields = [
            "sample_index",
            "frame_idx",
            "timestamp_monotonic",
            "wall_time",
            "bev_mask_path",
            "bev_input_path",
            "bev_color_path",
            "bev_source",
            "bev_width",
            "bev_height",
            "active",
            "command_valid",
            "command_confidence",
            "command_state",
            "command_steer",
            "command_speed",
            "command_brake",
            "wheel_left",
            "wheel_right",
            "wheel_sent",
            "wheel_reason",
            "trailer_angle_ok",
            "trailer_angle_deg",
            "trailer_angle_confidence",
            "trailer_angle_rate_deg_s",
            "trailer_angle_source",
            "trailer_angle_age_s",
            "trailer_angle_reason",
            "route_mode",
            "route_lane_state",
            "route_curvature_proxy",
            "route_corner_confidence",
            "route_target_bias_norm",
            "route_target_bias_fraction",
            "fused_state",
            "fused_confidence",
            "fused_rows",
            "fused_dashed_rows",
        ]
        for camera_key in ("cam0", "cam1"):
            fields.extend(
                [
                    f"{camera_key}_state",
                    f"{camera_key}_confidence",
                    f"{camera_key}_rows",
                    f"{camera_key}_dashed_rows",
                ]
            )
        return fields

    def _write_manifest(self, config: Dict[str, Any]) -> None:
        manifest = {
            "format_version": 1,
            "description": "E2E driving dataset: BEV segmentation mask plus trailer angle metadata to final motor outputs.",
            "inputs": {
                "bev_mask": {
                    "encoding": "uint8 PNG",
                    "classes": {"0": "background", "1": "solid_line", "2": "dashed_line"},
                    "width": self.bev.output_width,
                    "height": self.bev.output_height,
                },
                "bev_input_gray": {
                    "encoding": "uint8 PNG",
                    "values": {"0": "background", "127": "solid_line", "255": "dashed_line"},
                    "note": "High-contrast grayscale image intended for direct E2E image loading.",
                },
                "trailer_angle_deg": "Scalar angle estimate from the separate trailer panel detector/calibration pipeline.",
            },
            "labels": {
                "wheel_left": "Final left motor command after wheel mixer.",
                "wheel_right": "Final right motor command after wheel mixer.",
                "command_steer": "Pre-mixer normalized steer command.",
                "command_speed": "Pre-mixer normalized speed command.",
            },
            "policy": {
                "save_only_active": self.save_only_active,
                "save_only_sent": self.save_only_sent,
                "save_invalid": self.save_invalid,
                "sample_every": self.sample_every,
            },
            "model_boundaries": {
                "lane_segmentation": "external/separate model; saved BEV mask is its downstream output",
                "trailer_angle_detection": "external/separate detector/calibration; saved angle is its downstream output",
            },
            "config_project": (config.get("project", {}) or {}).get("name", ""),
        }
        (self.base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None

    def write(
        self,
        frame_idx: int,
        fused_bev_mask: Optional[np.ndarray],
        results: Dict[str, CameraLaneResult],
        fused_estimate: Optional[LaneEstimate],
        command: LaneDriveCommand,
        wheel: WheelCommand,
        angle_state: TrailerAngleState,
        route_debug: Optional[RouteControlDebug],
    ) -> None:
        if not self.enabled or self.writer is None or self.cv2 is None:
            return
        if frame_idx % self.sample_every != 0:
            return
        if self.save_only_active and not command.active:
            return
        if self.save_only_sent and not wheel.sent:
            return
        if not self.save_invalid and (not command.valid or command.brake):
            return

        bev_mask, bev_source = self._select_bev_mask(fused_bev_mask, results)
        if bev_mask is None:
            return

        self.sample_index += 1
        stem = f"{frame_idx:08d}"
        mask_path = self.mask_dir / f"{stem}.png"
        mask_uint8 = np.asarray(bev_mask, dtype=np.uint8)
        self.cv2.imwrite(str(mask_path), mask_uint8)

        input_rel = ""
        if self.save_input_gray:
            input_path = self.input_dir / f"{stem}.png"
            self.cv2.imwrite(str(input_path), self._mask_to_input_gray(mask_uint8))
            input_rel = input_path.relative_to(self.base_dir).as_posix()

        color_rel = ""
        if self.save_color_preview:
            color_path = self.color_dir / f"{stem}.png"
            self.cv2.imwrite(str(color_path), self._mask_to_color(mask_uint8))
            color_rel = color_path.relative_to(self.base_dir).as_posix()

        row = {
            "sample_index": self.sample_index,
            "frame_idx": frame_idx,
            "timestamp_monotonic": f"{time.monotonic():.3f}",
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "bev_mask_path": mask_path.relative_to(self.base_dir).as_posix(),
            "bev_input_path": input_rel,
            "bev_color_path": color_rel,
            "bev_source": bev_source,
            "bev_width": int(mask_uint8.shape[1]),
            "bev_height": int(mask_uint8.shape[0]),
            "active": int(command.active),
            "command_valid": int(command.valid),
            "command_confidence": f"{command.confidence:.4f}",
            "command_state": command.state,
            "command_steer": f"{command.steer:.4f}",
            "command_speed": f"{command.speed:.4f}",
            "command_brake": int(command.brake),
            "wheel_left": f"{wheel.left:.4f}",
            "wheel_right": f"{wheel.right:.4f}",
            "wheel_sent": int(wheel.sent),
            "wheel_reason": wheel.reason,
            "trailer_angle_ok": int(angle_state.ok),
            "trailer_angle_deg": "" if angle_state.angle_deg is None else f"{angle_state.angle_deg:.4f}",
            "trailer_angle_confidence": f"{angle_state.confidence:.4f}",
            "trailer_angle_rate_deg_s": f"{angle_state.angle_rate_deg_s:.4f}",
            "trailer_angle_source": angle_state.source,
            "trailer_angle_age_s": f"{angle_state.age_s:.4f}",
            "trailer_angle_reason": angle_state.reason,
            "route_mode": "" if route_debug is None else route_debug.mode,
            "route_lane_state": "" if route_debug is None else route_debug.lane_state,
            "route_curvature_proxy": "" if route_debug is None else f"{route_debug.curvature_proxy:.4f}",
            "route_corner_confidence": "" if route_debug is None else f"{route_debug.corner_confidence:.4f}",
            "route_target_bias_norm": "" if route_debug is None else f"{route_debug.target_bias_norm:.4f}",
            "route_target_bias_fraction": "" if route_debug is None else f"{route_debug.target_bias_fraction:.4f}",
        }
        self._add_estimate_fields(row, "fused", fused_estimate)
        for camera_key in ("cam0", "cam1"):
            result = results.get(camera_key)
            self._add_estimate_fields(row, camera_key, None if result is None else result.estimate)
        self.writer.writerow(row)
        if self.sample_index % 20 == 0 and self.csv_file is not None:
            self.csv_file.flush()

    def _select_bev_mask(
        self,
        fused_bev_mask: Optional[np.ndarray],
        results: Dict[str, CameraLaneResult],
    ) -> Tuple[Optional[np.ndarray], str]:
        if fused_bev_mask is not None:
            return fused_bev_mask, "fused_bev_mask"
        masks = [result.bev_mask for result in results.values() if result.bev_mask is not None]
        if not masks:
            return None, "none"
        if len(masks) == 1:
            key = next(iter(results.keys()), "single")
            return masks[0], f"{key}_bev_mask"
        merged = np.maximum.reduce(masks)
        return merged, "merged_camera_bev_masks"

    @staticmethod
    def _add_estimate_fields(row: Dict[str, Any], prefix: str, estimate: Optional[LaneEstimate]) -> None:
        if estimate is None:
            row.update(
                {
                    f"{prefix}_state": "",
                    f"{prefix}_confidence": "",
                    f"{prefix}_rows": "",
                    f"{prefix}_dashed_rows": "",
                }
            )
            return
        row.update(
            {
                f"{prefix}_state": estimate.state,
                f"{prefix}_confidence": f"{float(estimate.confidence):.4f}",
                f"{prefix}_rows": len(estimate.row_estimates),
                f"{prefix}_dashed_rows": sum(1 for lane_row in estimate.row_estimates if lane_row.dashed_x is not None),
            }
        )

    @staticmethod
    def _mask_to_color(mask: np.ndarray) -> np.ndarray:
        color = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
        color[mask == 1] = (42, 211, 255)
        color[mask == 2] = (0, 138, 255)
        return color

    @staticmethod
    def _mask_to_input_gray(mask: np.ndarray) -> np.ndarray:
        out = np.zeros(mask.shape[:2], dtype=np.uint8)
        out[mask == 1] = 127
        out[mask == 2] = 255
        return out


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    apply_overrides(config, args)

    mirror_configs: List[Path] = []
    parking_config = config_dir / "trailer_parking_config.yaml"
    if parking_config.exists() and parking_config.resolve() != config_path:
        mirror_configs.append(parking_config)
    http_streamer = HttpMjpegStreamer(
        config.get("http_stream", {}) or {},
        app_config=config,
        config_path=config_path,
        mirror_config_paths=mirror_configs,
    )
    http_streamer.write(
        make_status_frame(
            http_streamer.width,
            http_streamer.height,
            ["Rover trailer view", "initializing models and cameras..."],
        )
    )

    predictor = load_predictor(config, config_dir)
    panel_model = load_panel_model(config, config_dir)
    angle_estimator: Optional[CenterTableAngleEstimator] = None
    angle_filter: Optional[AngleStateFilter] = None
    if panel_model is not None:
        try:
            angle_estimator = CenterTableAngleEstimator(config, config_dir)
            angle_filter = AngleStateFilter(config)
            print("[angle] anchors:")
            for row in angle_estimator.debug_anchor_rows():
                print(
                    f"  {row['camera_key']} angle={row['angle_deg']:6.1f} "
                    f"cx={row['center_x_norm']:.3f} w={row['width_norm']:.3f} "
                    f"conf={row['det_conf']:.2f} n={row['samples']}"
                )
        except Exception as exc:
            print(f"[angle] disabled: {exc}")
            angle_estimator = None
            angle_filter = None
    bev = make_bev_config(config)
    dual_bev = load_dual_bev_calibration(config, config_dir)
    if dual_bev.enabled:
        print(
            f"[dual_bev] enabled source={dual_bev.source} "
            f"cameras={','.join(sorted(dual_bev.cameras.keys()))} drive_mode={dual_bev.drive_mode}"
        )
    else:
        print(f"[dual_bev] disabled/fallback source={dual_bev.source}")
    camera_cfgs = enabled_camera_configs(config, str((config.get("camera", {}) or {}).get("single_camera", "")))
    lane_cfgs: Dict[str, LaneFollowerConfig] = {
        key: make_lane_config(config, camera_cfg) for key, camera_cfg in camera_cfgs.items()
    }
    lane_states: Dict[str, LaneFollowerState] = {key: LaneFollowerState() for key in lane_cfgs.keys()}
    fused_lane_cfg = make_lane_config(config)
    fused_lane_cfg.vehicle_center_x_bias = dual_bev.vehicle_center_x_bias
    fused_lane_state = LaneFollowerState()
    if args.video is not None and not lane_cfgs:
        lane_cfgs["video"] = make_lane_config(config)
        lane_states["video"] = LaneFollowerState()
    matrix_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
    route_controller = RouteAwareTrailerController(config)
    angle_rate_filter = TrailerAngleRateFilter(config)
    corner_pivot_controller = CornerPivotController(config)
    if route_controller.enabled:
        print("[route] angle-aware gain-scheduled controller enabled")
    if corner_pivot_controller.enabled:
        pivot_cfg = config.get("corner_pivot", {}) or {}
        print(
            f"[pivot] enabled pwm={as_float(pivot_cfg.get('pivot_pwm'), 0.55):.2f} "
            f"direction_sign={as_float(pivot_cfg.get('direction_sign'), 1.0):+.0f}"
        )
    wheel_mixer = LaneWheelMixer(config)
    source = open_source(config, args)

    rover_cfg = config.get("rover", {}) or {}
    rover = RoverSerial(
        str(rover_cfg.get("serial", "/dev/ttyUSB0")),
        as_int(rover_cfg.get("baud"), 115200),
        as_bool(rover_cfg.get("arm"), False),
    )
    repeater = make_repeater_if_enabled(config, rover)
    if repeater is not None:
        print(
            f"[rover] command repeater enabled: {as_float(rover_cfg.get('command_rate_hz'), 20.0):.1f} Hz "
            f"(control updates follow camera/inference loop, target camera fps={as_float((config.get('camera', {}) or {}).get('fps'), 0.0):.1f})"
        )
    else:
        print("[rover] command repeater disabled: commands are sent only when a new frame is processed")

    runtime_cfg = config.get("runtime", {}) or {}
    active = bool(args.start_driving or as_bool(runtime_cfg.get("auto_start"), False))
    display = as_bool(runtime_cfg.get("display"), False)
    print_every = args.print_every if args.print_every > 0 else as_int(runtime_cfg.get("print_every"), 5)
    max_frames = args.max_frames if args.max_frames > 0 else as_int(runtime_cfg.get("max_frames"), 0)
    log_root = resolve_path(config_dir, str(runtime_cfg.get("log_dir", "logs/dotted_lane_following_run")))
    run_dir = log_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_writer, log_file = make_log_writer(run_dir)
    primary_lane_cfg = next(iter(lane_cfgs.values())) if lane_cfgs else make_lane_config(config)
    save_run_config(run_dir, config, bev, primary_lane_cfg)
    e2e_writer = E2EDatasetWriter(run_dir, config, bev)
    streamer = UdpVideoStreamer(config.get("stream", {}) or {})
    save_video_enabled = as_bool(runtime_cfg.get("save_video"), False)
    debug_enabled = bool(display or streamer.enabled or http_streamer.enabled or save_video_enabled)
    need_full_debug_panel = bool(display or streamer.enabled or save_video_enabled)
    if not debug_enabled:
        print("[perf] debug rendering disabled: no display/udp/http/video")
    else:
        print(
            "[perf] debug rendering enabled: "
            f"display={display} udp={streamer.enabled} http={http_streamer.enabled} video={save_video_enabled}"
        )

    video_writer = None
    detections: Dict[str, PanelDetection] = {}
    angle_estimates: Dict[str, AngleEstimate] = {}
    started_at = time.monotonic()
    prev_time = started_at
    frame_idx = 0
    command_repeat_hz = as_float(rover_cfg.get("command_rate_hz"), 20.0) if repeater is not None else 0.0
    last_wheel_signature: Optional[Tuple[float, float, bool]] = None
    last_command_change_time = started_at
    command_change_hz = 0.0
    if active:
        print("[lane] driving started")
    else:
        print("[lane] preview only: use --start-driving to send wheel commands")

    try:
        while not _STOP:
            frames = source.read()
            if not frames:
                if as_bool((config.get("safety", {}) or {}).get("stop_on_camera_failure"), True):
                    print("[camera] frame read failed")
                    break
                time.sleep(0.01)
                continue

            frame_idx += 1
            now = time.monotonic()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now

            fused_angle = estimate_trailer_angle(
                config,
                panel_model,
                angle_estimator,
                angle_filter,
                detections,
                angle_estimates,
                frames,
                frame_idx,
                now,
            )
            angle_state = angle_rate_filter.update(fused_angle, now=now)
            if route_controller.enabled:
                apply_lane_lookahead(lane_cfgs, fused_lane_cfg, route_controller.scheduled_lookahead(angle_state))

            results: Dict[str, CameraLaneResult] = {}
            warped_masks: List[np.ndarray] = []
            fused_bev_mask: Optional[np.ndarray] = None
            fused_bev_panel: Optional[np.ndarray] = None
            fused_estimate: Optional[LaneEstimate] = None
            for camera_key, frame in frames.items():
                camera_cfg = camera_cfgs.get(camera_key, {})
                lane_cfg = lane_cfgs.get(camera_key)
                if lane_cfg is None:
                    lane_cfg = make_lane_config(config, camera_cfg)
                    lane_cfgs[camera_key] = lane_cfg
                lane_state = lane_states.setdefault(camera_key, LaneFollowerState())
                crop_bgr, _crop_xyxy = crop_frame(frame, config, camera_cfg)
                h, w = crop_bgr.shape[:2]
                cache_key = (camera_key, w, h)
                if cache_key not in matrix_cache:
                    matrix_cache[cache_key], _ = perspective_matrices(w, h, bev)

                mask = predictor.predict(crop_bgr)
                if dual_bev.enabled:
                    bev_mask = dual_warp_mask(mask, camera_key, dual_bev, bev)
                    if bev_mask is None:
                        bev_mask = warp_to_bev(mask, bev, matrix_cache[cache_key], is_mask=True)
                    else:
                        warped_masks.append(bev_mask)
                    estimate = estimate_lane_with_solid_fallback(bev_mask, fused_lane_cfg, lane_state, config)
                else:
                    bev_mask = warp_to_bev(mask, bev, matrix_cache[cache_key], is_mask=True)
                    estimate = estimate_lane_with_solid_fallback(bev_mask, lane_cfg, lane_state, config)
                panel = (
                    make_debug_panel(crop_bgr, mask, bev_mask, estimate, bev)
                    if debug_enabled
                    else np.zeros((1, 1, 3), dtype=np.uint8)
                )
                results[camera_key] = CameraLaneResult(
                    camera_key=camera_key,
                    frame_bgr=frame,
                    crop_bgr=crop_bgr,
                    mask=mask,
                    bev_mask=bev_mask,
                    estimate=estimate,
                    panel=panel,
                )

            if dual_bev.enabled and warped_masks:
                fused_bev_mask = merge_bev_masks(warped_masks, dual_bev, bev)
                fused_estimate = estimate_lane_with_solid_fallback(fused_bev_mask, fused_lane_cfg, fused_lane_state, config)
                if dual_bev.drive_mode == "estimate_fusion":
                    drive_command = fuse_lane_results(results, config, active)
                    drive_command.state = f"DUAL_BEV_EST_FUSION:{drive_command.state}"
                    drive_command.reason = f"dual_bev_estimate_fusion {drive_command.reason}"
                else:
                    drive_command = make_drive_command(fused_estimate, config, active)
                    drive_command.state = f"DUAL_BEV:{fused_estimate.state}"
                    drive_command.reason = f"dual_bev {fused_estimate.reason}"
                if debug_enabled:
                    fused_bev_panel = draw_bev_debug(fused_bev_mask, fused_estimate, bev)
            else:
                drive_command = fuse_lane_results(results, config, active)
            base_drive_command = drive_command
            route_debug: Optional[RouteControlDebug] = None
            lane_signal: Optional[LaneSignal] = None
            if route_controller.enabled:
                fused_lane_signal: Optional[LaneSignal] = None
                if fused_estimate is not None:
                    fused_lane_signal = lane_signal_from_estimate(fused_estimate, config, bev.output_width)
                lane_signal = (
                    fused_lane_signal
                    if fused_lane_signal is not None and fused_lane_signal.valid
                    else lane_signal_from_results(results, config, bev.output_width)
                )
                if use_simple_dashed_priority(config, route_controller, lane_signal, angle_state, base_drive_command):
                    drive_command = simple_dashed_command(base_drive_command, active, config)
                else:
                    route_output = route_controller.control(lane_signal, angle_state, active, now=now)
                    drive_command = LaneDriveCommand(
                        active=route_output.active,
                        valid=route_output.valid,
                        confidence=route_output.confidence,
                        state=route_output.state,
                        steer=route_output.steer,
                        speed=route_output.speed,
                        brake=route_output.brake,
                        reason=route_output.reason,
                    )
                    route_debug = route_controller.last_debug
            drive_command = corner_pivot_controller.override(
                fused_estimate,
                results,
                angle_state,
                drive_command,
                active,
                bev.output_width,
                bev.output_height,
                now,
            )
            pivot_exited = corner_pivot_controller.last_debug.startswith("exit:dashed_reacquired")
            pivot_exit_triggers_bias = as_bool(
                ((config.get("route_controller", {}) or {}).get("post_corner_bias_trigger_on_pivot_exit")),
                True,
            )
            if route_controller.enabled and pivot_exited and pivot_exit_triggers_bias and lane_signal is not None:
                route_controller.force_post_corner_bias(now, corner_pivot_controller.last_exit_turn_sign)
                route_output = route_controller.control(lane_signal, angle_state, active, now=now)
                drive_command = LaneDriveCommand(
                    active=route_output.active,
                    valid=route_output.valid,
                    confidence=route_output.confidence,
                    state=route_output.state,
                    steer=route_output.steer,
                    speed=route_output.speed,
                    brake=route_output.brake,
                    reason=f"pivot_exit_bias {route_output.reason}",
                )
                route_debug = route_controller.last_debug
            wheel = wheel_mixer.mix(drive_command, now)
            wheel_signature = (round(wheel.left, 3), round(wheel.right, 3), bool(wheel.sent))
            if wheel_signature != last_wheel_signature:
                command_change_now = time.monotonic()
                if last_wheel_signature is not None:
                    command_change_hz = 1.0 / max(1e-6, command_change_now - last_command_change_time)
                last_command_change_time = command_change_now
                last_wheel_signature = wheel_signature

            if repeater is not None:
                repeater.update(wheel.left, wheel.right, wheel.sent)
            elif wheel.sent:
                rover.send(wheel.left, wheel.right)
            else:
                rover.send(0.0, 0.0)

            e2e_writer.write(
                frame_idx,
                fused_bev_mask,
                results,
                fused_estimate,
                drive_command,
                wheel,
                angle_state,
                route_debug,
            )

            panel: Optional[np.ndarray] = None
            if need_full_debug_panel:
                panel = make_combined_debug_panel(results, drive_command, wheel, rover.armed, fused_bev_panel, route_debug, angle_state, config)
                streamer.write(panel)

            if http_streamer.enabled:
                fallback_panel = panel
                if fallback_panel is None:
                    fallback_panel = np.zeros((http_streamer.height, http_streamer.width, 3), dtype=np.uint8)
                http_panel = make_http_debug_panel(
                    config,
                    fallback_panel,
                    results,
                    drive_command,
                    wheel,
                    rover.armed,
                    fused_bev_panel,
                    route_debug,
                    frames,
                    detections,
                    angle_estimates,
                    angle_state,
                )
                http_streamer.write(http_panel)

            if save_video_enabled:
                import cv2

                if panel is None:
                    panel = make_combined_debug_panel(results, drive_command, wheel, rover.armed, fused_bev_panel, route_debug, angle_state, config)
                if video_writer is None:
                    video_path = run_dir / "dotted_lane_debug.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (panel.shape[1], panel.shape[0]))
                    print(f"[video] {video_path}")
                video_writer.write(panel)

            write_log_row(
                log_writer,
                frame_idx,
                fps,
                drive_command,
                results,
                wheel,
                angle_state,
                route_debug,
                detections,
                angle_estimates,
            )
            if frame_idx % 20 == 0:
                log_file.flush()

            if print_every > 0 and frame_idx % print_every == 0:
                cam_text = " ".join(
                    f"{key}:{result.estimate.state[:8]}:{result.estimate.confidence:.2f}"
                    for key, result in sorted(results.items())
                )
                mode_text = drive_command.state
                band_text = ""
                ref_text = ""
                if route_debug is not None:
                    if not drive_command.state.startswith("CORNER_PIVOT"):
                        mode_text = route_debug.mode
                    else:
                        mode_text = f"{drive_command.state}|route={route_debug.mode}"
                    band_text = f"/{route_debug.angle_band}"
                    ref_text = f" ref={route_debug.alpha_ref_deg:+.1f} bias={route_debug.target_bias_norm:+.2f}/{route_debug.target_bias_fraction:.2f}"
                angle_text = "--" if angle_state.angle_deg is None else f"{angle_state.angle_deg:+.1f}"
                panel_text = " ".join(
                    panel_console_token(key, detections.get(key), angle_estimates.get(key))
                    for key in ("cam1", "cam0")
                )
                pivot_text = f" pivot={corner_pivot_controller.last_debug}" if corner_pivot_controller.enabled else ""
                assist_text = " assist=lane_pivot" if wheel.reason.startswith("lane_pivot_assist") else ""
                print(
                    f"[{frame_idx:06d}] {'RUN' if active else 'PRE'} {mode_text}{band_text} "
                    f"angle={angle_text}{ref_text} c={angle_state.confidence:.2f} src={angle_state.source} "
                    f"lane={drive_command.confidence:.2f} steer={drive_command.steer:+.2f} speed={drive_command.speed:.2f} "
                    f"wheel={wheel.left:+.2f}/{wheel.right:+.2f} "
                    f"ctrlHz={fps:.1f} cmdChangeHz={command_change_hz:.1f}/{command_repeat_hz:.0f}"
                    f"{assist_text}{pivot_text} panel={panel_text} {cam_text}"
                )

            if display:
                try:
                    import cv2

                    scale = as_float(runtime_cfg.get("display_scale"), 1.0)
                    shown = panel
                    if scale > 0.0 and abs(scale - 1.0) > 1e-3:
                        shown = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("dotted_lane_following", shown)
                    key_code = cv2.waitKey(1) & 0xFF
                except Exception as exc:
                    print(f"[display] disabled: {exc}")
                    display = False
                    key_code = 255
                if key_code in (ord("q"), 27):
                    break
                if key_code == ord("p"):
                    active = True
                    print("[lane] driving started")
                elif key_code in (ord("x"), ord(" ")):
                    active = False
                    print("[lane] driving stopped")

            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        if repeater is not None:
            repeater.stop()
        rover.close()
        source.close()
        streamer.close()
        http_streamer.close()
        e2e_writer.close()
        if video_writer is not None:
            video_writer.release()
        log_file.flush()
        log_file.close()
        if display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    print(f"frames={frame_idx}")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
