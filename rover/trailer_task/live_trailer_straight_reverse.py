#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from dataclasses import dataclass
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
    CenterTableAngleEstimator,
    DifferentialMixer,
    FusedAngleEstimate,
    ParkingCommand,
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
    norm_angle_delta,
    resolve_path,
)
from live_trailer_parking import (  # noqa: E402
    UdpVideoStreamer,
    apply_color_fix,
    best_panel_detection,
    camera_roi,
    combined_view,
    enabled_camera_keys,
    open_camera,
    open_gst_tools,
    rgb_to_bgr,
)


_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


@dataclass
class StraightReverseDebug:
    angle_deg: Optional[float]
    error_deg: float
    derivative_deg_s: float
    speed_scale: float
    steering_sign: float


class StraightReverseController:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("straight_reverse", {}) or {}
        self.rover_cfg = config.get("rover", {}) or {}
        self.safety_cfg = config.get("safety", {}) or {}
        self.state = "IDLE"
        self.start_time: Optional[float] = None
        self.last_angle: Optional[float] = None
        self.last_angle_time: Optional[float] = None
        self.lost_since: Optional[float] = None
        self.last_error: Optional[float] = None
        self.last_error_time: Optional[float] = None
        self.integral_error = 0.0
        self.last_steer = 0.0
        self.complete_since: Optional[float] = None
        self.reason = "idle"
        self.debug = StraightReverseDebug(None, 0.0, 0.0, 1.0, self.steering_sign)

    @property
    def steering_sign(self) -> float:
        return 1.0 if as_float(self.cfg.get("reverse_steer_sign"), 1.0) >= 0.0 else -1.0

    def start(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.state = "RUNNING"
        self.start_time = now
        self.lost_since = None
        self.last_error = None
        self.last_error_time = None
        self.integral_error = 0.0
        self.last_steer = 0.0
        self.complete_since = None
        self.reason = "running"

    def stop(self, reason: str = "manual_stop") -> None:
        self.state = "STOPPED"
        self.reason = reason

    def reset(self) -> None:
        self.state = "IDLE"
        self.start_time = None
        self.lost_since = None
        self.last_error = None
        self.last_error_time = None
        self.integral_error = 0.0
        self.last_steer = 0.0
        self.complete_since = None
        self.reason = "idle"

    def fault(self, reason: str) -> None:
        self.state = "FAULT"
        self.reason = reason

    def flip_steering_sign(self) -> None:
        self.cfg["reverse_steer_sign"] = -self.steering_sign
        self.last_error = None
        self.last_error_time = None
        self.integral_error = 0.0
        self.last_steer = 0.0
        self.reason = f"steering_sign={self.steering_sign:+.0f}"

    def update(self, angle: FusedAngleEstimate, now: Optional[float] = None) -> ParkingCommand:
        now = time.monotonic() if now is None else float(now)
        if self.state == "IDLE" and as_bool(self.cfg.get("auto_start"), False):
            self.start(now)
        if self.state in {"IDLE", "STOPPED", "FAULT", "COMPLETE"}:
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.reason, active=False)

        max_run = as_float(self.cfg.get("max_run_time_s"), 20.0)
        if self.start_time is not None and max_run > 0.0 and now - self.start_time > max_run:
            self.stop(f"max_run_time>{max_run:.1f}s")
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.reason, active=False)

        theta = self._usable_angle(angle, now)
        if theta is None:
            blind_command = self._blind_reverse_command(now)
            if blind_command is not None:
                return blind_command
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.reason, active=False)

        hard_stop = as_float(
            self.cfg.get("hard_stop_abs_angle_deg"),
            as_float(self.safety_cfg.get("hard_stop_abs_angle_deg"), 58.0),
        )
        if abs(theta) >= hard_stop:
            self.fault(f"hard_angle_limit:{theta:.1f}")
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.reason, active=False)

        max_abs = as_float(self.cfg.get("max_abs_angle_deg"), 45.0)
        if abs(theta) >= max_abs:
            self.fault(f"angle_limit:{theta:.1f}")
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.reason, active=False)

        target = as_float(self.cfg.get("target_angle_deg"), 0.0)
        error = norm_angle_delta(target - theta)
        steer, derivative = self._pid_steer(error, now)
        speed, speed_scale = self._reverse_speed(theta)
        steer = self._limit_diff_for_reverse(steer, speed)
        self._maybe_complete(error, now)
        self.debug = StraightReverseDebug(theta, error, derivative, speed_scale, self.steering_sign)
        return ParkingCommand(speed, steer, False, self.state, target, f"theta={theta:.1f},error={error:.1f}", active=True)

    def _usable_angle(self, angle: FusedAngleEstimate, now: float) -> Optional[float]:
        min_conf = as_float(self.cfg.get("min_angle_confidence"), 0.16)
        grace = as_float(self.cfg.get("lost_angle_grace_s"), 0.55)
        if angle.ok and angle.angle_deg is not None and angle.confidence >= min_conf:
            self.last_angle = float(angle.angle_deg)
            self.last_angle_time = now
            self.lost_since = None
            if self.reason == "angle_lost_waiting":
                self.reason = "angle_recovered"
            return self.last_angle
        if self.lost_since is None:
            self.lost_since = now
        if self.last_angle is not None and now - self.lost_since <= grace:
            self.reason = "holding_last_angle"
            return self.last_angle
        if as_bool(self.cfg.get("fault_on_angle_lost"), False):
            self.fault("angle_lost")
        else:
            self.reason = "angle_lost_waiting"
        return None

    def _blind_reverse_command(self, now: float) -> Optional[ParkingCommand]:
        if not as_bool(self.cfg.get("continue_when_angle_lost"), False):
            return None
        if self.state != "RUNNING" or self.last_angle is None or self.lost_since is None:
            return None
        max_last_angle = abs(as_float(self.cfg.get("lost_continue_max_abs_angle_deg"), 20.0))
        if abs(self.last_angle) > max_last_angle:
            self.reason = f"angle_lost_last_angle_too_large:{self.last_angle:.1f}"
            return None
        lost_age = now - self.lost_since
        blind_timeout = as_float(self.cfg.get("blind_reverse_timeout_s"), 5.0)
        if blind_timeout > 0.0 and lost_age > blind_timeout:
            self.stop(f"blind_reverse_timeout>{blind_timeout:.1f}s")
            return ParkingCommand(0.0, 0.0, True, self.state, None, self.reason, active=False)
        speed = -abs(as_float(self.cfg.get("lost_continue_speed"), 0.12))
        base = abs(as_float(self.cfg.get("reverse_speed"), 0.18))
        self.debug = StraightReverseDebug(None, self.last_error or 0.0, 0.0, abs(speed) / max(1e-6, base), self.steering_sign)
        self.reason = f"blind_base_reverse:last_angle={self.last_angle:.1f},lost={lost_age:.1f}s"
        return ParkingCommand(speed, 0.0, False, self.state, 0.0, self.reason, active=True)

    def _pid_steer(self, error: float, now: float) -> Tuple[float, float]:
        kp = as_float(self.cfg.get("kp_angle"), 0.045)
        ki = as_float(self.cfg.get("ki_angle"), 0.0)
        kd = as_float(self.cfg.get("kd_angle"), 0.010)
        derivative = 0.0
        dt = 0.0
        if self.last_error is not None and self.last_error_time is not None:
            dt = max(1e-3, now - self.last_error_time)
            derivative = norm_angle_delta(error - self.last_error) / dt
            self.integral_error += error * dt
        self.integral_error = clamp(self.integral_error, -180.0, 180.0)
        self.last_error = error
        self.last_error_time = now

        raw = self.steering_sign * (kp * error + ki * self.integral_error + kd * derivative)
        max_steer = abs(as_float(self.cfg.get("max_steer"), 0.70))
        steer = clamp(raw, -max_steer, max_steer)
        max_delta = abs(as_float(self.cfg.get("max_steer_delta_per_frame"), 0.08))
        if max_delta > 0.0:
            steer = clamp(steer, self.last_steer - max_delta, self.last_steer + max_delta)
        self.last_steer = steer
        return steer, derivative

    def _limit_diff_for_reverse(self, steer: float, speed: float) -> float:
        if as_bool(self.cfg.get("allow_counter_rotation"), False) or speed >= 0.0:
            return steer
        min_ratio = clamp(as_float(self.cfg.get("min_inner_reverse_ratio"), 0.25), 0.0, 0.95)
        speed_gain = as_float(self.rover_cfg.get("speed_gain"), 1.0)
        steer_mix = max(1e-6, as_float(self.rover_cfg.get("steer_mix"), 1.0))
        base_norm = abs(speed * speed_gain)
        max_diff = max(0.0, base_norm * (1.0 - min_ratio) / steer_mix)
        return clamp(steer, -max_diff, max_diff)

    def _reverse_speed(self, angle_deg: float) -> Tuple[float, float]:
        base = abs(as_float(self.cfg.get("reverse_speed"), 0.24))
        minimum = abs(as_float(self.cfg.get("min_reverse_speed"), 0.14))
        start = max(0.0, as_float(self.cfg.get("angle_slowdown_start_deg"), 10.0))
        full = max(start + 1e-3, as_float(self.cfg.get("angle_slowdown_full_deg"), 32.0))
        amount = clamp((abs(angle_deg) - start) / (full - start), 0.0, 1.0)
        speed = base + amount * (minimum - base)
        return -abs(speed), speed / max(1e-6, base)

    def _maybe_complete(self, error: float, now: float) -> None:
        if not as_bool(self.cfg.get("stop_when_complete"), False):
            return
        threshold = abs(as_float(self.cfg.get("complete_abs_angle_deg"), 3.0))
        hold_s = as_float(self.cfg.get("complete_hold_s"), 2.0)
        if abs(error) <= threshold:
            if self.complete_since is None:
                self.complete_since = now
            if now - self.complete_since >= hold_s:
                self.state = "COMPLETE"
                self.reason = "complete"
        else:
            self.complete_since = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Practice straight reverse with a trailer using mirror angle feedback.")
    parser.add_argument("--config", type=Path, default=HERE / "trailer_parking_config.yaml")
    parser.add_argument("--panel-weights", default="", help="Override models.panel.weights.")
    parser.add_argument("--start-reverse", action="store_true", help="Start reversing immediately.")
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--arm", dest="arm_override", action="store_true", default=None)
    arm_group.add_argument("--no-arm", dest="arm_override", action="store_false", default=None)
    parser.add_argument("--display", dest="display_override", action="store_true", default=None)
    parser.add_argument("--no-display", dest="display_override", action="store_false", default=None)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream_override", action="store_true", default=None)
    stream_group.add_argument("--no-stream", dest="stream_override", action="store_false", default=None)
    parser.add_argument("--udp-host", default="")
    parser.add_argument("--udp-port", type=int, default=0)
    parser.add_argument("--reverse-speed", type=float, default=0.0)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--diff-sign", "--steer-sign", dest="steer_sign", type=float, choices=(-1.0, 1.0), default=None)
    parser.add_argument("--max-run-time", type=float, default=None, help="Seconds before auto-stop. Use 0 to disable.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument("--single-camera", choices=("left", "right", "cam0", "cam1"), default="")
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.panel_weights:
        config.setdefault("models", {}).setdefault("panel", {})["weights"] = args.panel_weights
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
    straight = config.setdefault("straight_reverse", {})
    if args.reverse_speed > 0.0:
        straight["reverse_speed"] = float(args.reverse_speed)
    if args.kp is not None:
        straight["kp_angle"] = float(args.kp)
    if args.kd is not None:
        straight["kd_angle"] = float(args.kd)
    if args.steer_sign is not None:
        straight["reverse_steer_sign"] = float(args.steer_sign)
    if args.max_run_time is not None:
        straight["max_run_time_s"] = float(args.max_run_time)


def make_log_writer(run_dir: Path) -> Tuple[csv.DictWriter, Any]:
    csv_path = run_dir / "straight_reverse_log.csv"
    f = csv_path.open("w", newline="", encoding="utf-8")
    fields = [
        "frame_idx",
        "timestamp_monotonic",
        "wall_time",
        "angle_ok",
        "angle_deg",
        "angle_confidence",
        "angle_source",
        "angle_age_s",
        "controller_state",
        "target_angle_deg",
        "error_deg",
        "derivative_deg_s",
        "diff_cmd",
        "reverse_speed",
        "speed_scale",
        "diff_sign",
        "wheel_left",
        "wheel_right",
        "wheel_sent",
        "reason",
        "cam0_det",
        "cam0_conf",
        "cam0_cx_norm",
        "cam0_angle_deg",
        "cam1_det",
        "cam1_conf",
        "cam1_cx_norm",
        "cam1_angle_deg",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    print(f"[log] {csv_path}")
    return writer, f


def camera_values(det: Optional[PanelDetection], est: Optional[AngleEstimate], prefix: str) -> Dict[str, Any]:
    if det is None or not det.ok:
        return {
            f"{prefix}_det": 0,
            f"{prefix}_conf": "",
            f"{prefix}_cx_norm": "",
            f"{prefix}_angle_deg": "",
        }
    feats = det.features()
    return {
        f"{prefix}_det": 1,
        f"{prefix}_conf": f"{det.confidence:.4f}",
        f"{prefix}_cx_norm": f"{feats['center_x_norm']:.5f}",
        f"{prefix}_angle_deg": "" if est is None or est.angle_deg is None else f"{est.angle_deg:.3f}",
    }


def write_log_row(
    writer: csv.DictWriter,
    frame_idx: int,
    detections: Dict[str, PanelDetection],
    estimates: Dict[str, AngleEstimate],
    fused: FusedAngleEstimate,
    controller: StraightReverseController,
    command: ParkingCommand,
    wheel: Any,
) -> None:
    row: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "timestamp_monotonic": f"{time.monotonic():.3f}",
        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        "angle_ok": int(fused.ok),
        "angle_deg": "" if fused.angle_deg is None else f"{fused.angle_deg:.3f}",
        "angle_confidence": f"{fused.confidence:.4f}",
        "angle_source": fused.source,
        "angle_age_s": f"{fused.age_s:.3f}",
        "controller_state": command.state,
        "target_angle_deg": "" if command.target_angle_deg is None else f"{command.target_angle_deg:.3f}",
        "error_deg": f"{controller.debug.error_deg:.3f}",
        "derivative_deg_s": f"{controller.debug.derivative_deg_s:.3f}",
        "diff_cmd": f"{command.steer:.4f}",
        "reverse_speed": f"{command.speed:.4f}",
        "speed_scale": f"{controller.debug.speed_scale:.4f}",
        "diff_sign": f"{controller.debug.steering_sign:+.0f}",
        "wheel_left": f"{wheel.left:.4f}",
        "wheel_right": f"{wheel.right:.4f}",
        "wheel_sent": int(wheel.sent),
        "reason": command.reason,
    }
    row.update(camera_values(detections.get("cam0"), estimates.get("cam0"), "cam0"))
    row.update(camera_values(detections.get("cam1"), estimates.get("cam1"), "cam1"))
    writer.writerow(row)


def load_panel_model(config: Dict[str, Any], config_dir: Path):
    panel_cfg = ((config.get("models", {}) or {}).get("panel", {}) or {})
    weights = resolve_path(config_dir, str(panel_cfg.get("weights", "")))
    if not weights.exists():
        raise SystemExit(f"Panel YOLO weights not found: {weights}")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is required. Install with: pip3 install ultralytics\n{exc}") from exc
    print(f"[panel] weights={weights}")
    if weights.suffix.lower() == ".engine":
        return YOLO(str(weights), task="detect"), panel_cfg
    return YOLO(str(weights)), panel_cfg


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    apply_overrides(config, args)

    panel_model, panel_cfg = load_panel_model(config, config_dir)
    angle_estimator = CenterTableAngleEstimator(config, config_dir)
    angle_filter = AngleStateFilter(config)
    controller = StraightReverseController(config)
    mixer = DifferentialMixer(config)

    GstCamera, fix_edge_color_cast = open_gst_tools()
    keys = enabled_camera_keys(config, args.single_camera)
    cameras: Dict[str, Any] = {}
    for key in keys:
        try:
            cameras[key] = open_camera(key, config, GstCamera)
        except Exception as exc:
            print(f"[camera] {key} open failed: {exc}")
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
    display = as_bool(runtime_cfg.get("display"), False)
    print_every = args.print_every if args.print_every > 0 else as_int(runtime_cfg.get("print_every"), 10)
    log_root = resolve_path(config_dir, "logs/straight_reverse_run")
    run_dir = log_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_writer, log_file = make_log_writer(run_dir)
    streamer = UdpVideoStreamer(config.get("stream", {}) or {})

    if args.start_reverse:
        controller.start()
        print("[straight] reverse started")

    detections: Dict[str, PanelDetection] = {}
    estimates: Dict[str, AngleEstimate] = {}
    frame_idx = 0
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
                    x0, y0, x1, y1 = camera_roi(config, key).to_abs(frame_rgb.shape)
                    det = best_panel_detection(panel_model, frame_rgb[y0:y1, x0:x1], key, panel_cfg, now)
                    detections[key] = det
                    estimates[key] = angle_estimator.estimate(det)
                    current_measurements.append(estimates[key])
            fused = angle_filter.update(current_measurements, now=now)
            command = controller.update(fused, now=now)
            wheel = mixer.mix(command)

            if repeater is not None:
                repeater.update(wheel.left, wheel.right, wheel.sent)
            elif wheel.sent:
                rover.send(wheel.left, wheel.right)
            else:
                rover.send(0.0, 0.0)

            for key, frame_bgr in frames_bgr.items():
                roi_xyxy = camera_roi(config, key).to_abs(frame_bgr.shape)
                color = (0, 220, 255) if key == "cam1" else (0, 170, 255)
                draw_panel_overlay(frame_bgr, roi_xyxy, detections.get(key), color)
            view = combined_view(frames_bgr, config)
            angle_text = "--" if fused.angle_deg is None else f"{fused.angle_deg:5.1f} deg"
            lines = [
                f"straight reverse: {command.state} angle={angle_text} conf={fused.confidence:.2f} src={fused.source}",
                f"target={command.target_angle_deg} err={controller.debug.error_deg:.1f} diff={command.steer:.2f} speed={command.speed:.2f}",
                f"wheel/pwm: L={wheel.left:.3f} R={wheel.right:.3f} diff_sign={controller.debug.steering_sign:+.0f} {'ARMED' if rover.armed else 'dry-run'}",
                "keys: p=start, x=stop, r=reset, f=flip diff sign, q=quit",
            ]
            draw_status_overlay(view, lines)
            streamer.write(view)
            write_log_row(log_writer, frame_idx, detections, estimates, fused, controller, command, wheel)
            if frame_idx % 20 == 0:
                log_file.flush()

            if print_every > 0 and frame_idx % print_every == 0:
                print(
                    f"[{frame_idx:06d}] state={command.state} angle={angle_text} "
                    f"err={controller.debug.error_deg:.1f} diff={command.steer:.2f} "
                    f"L={wheel.left:.3f} R={wheel.right:.3f} {command.reason}"
                )

            if display:
                try:
                    import cv2

                    scale = as_float(runtime_cfg.get("display_scale"), 1.0)
                    shown = view
                    if scale > 0.0 and abs(scale - 1.0) > 1e-3:
                        shown = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("trailer_straight_reverse", shown)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as exc:
                    print(f"[display] disabled: {exc}")
                    display = False
                    key = 255
                if key == ord("q") or key == 27:
                    break
                if key == ord("p"):
                    controller.start()
                    print("[straight] reverse started")
                elif key == ord("x") or key == ord(" "):
                    controller.stop("manual_stop")
                    print("[straight] stopped")
                elif key == ord("r"):
                    controller.reset()
                    print("[straight] reset")
                elif key == ord("f"):
                    controller.flip_steering_sign()
                    print(f"[straight] differential sign flipped to {controller.steering_sign:+.0f}")

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
        streamer.close()
        log_file.flush()
        log_file.close()
        if display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
