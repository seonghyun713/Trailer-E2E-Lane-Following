#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import live_dotted_lane_following as live  # noqa: E402
from e2e_temporal_policy.model import TemporalPolicyNet  # noqa: E402
from trailer_parking_core import (  # noqa: E402
    AngleStateFilter,
    CenterTableAngleEstimator,
    PanelDetection,
    RoverSerial,
    as_bool,
    as_float,
    as_int,
    clamp,
    load_yaml,
    make_repeater_if_enabled,
    resolve_path,
)
from trailer_route_controller import (  # noqa: E402
    LaneSignal,
    RouteAwareTrailerController,
    RouteControlDebug,
    TrailerAngleRateFilter,
    TrailerAngleState,
    lane_signal_from_estimate,
    lane_signal_from_results,
)
from lane_following_core import (  # noqa: E402
    LaneEstimate,
    LaneFollowerConfig,
    LaneFollowerState,
    draw_bev_debug,
    perspective_matrices,
    warp_to_bev,
)


@dataclass
class E2EPolicyOutput:
    ready: bool
    raw_left: float = 0.0
    raw_right: float = 0.0
    safe_left: float = 0.0
    safe_right: float = 0.0
    sent: bool = False
    mode: str = "unknown"
    mode_confidence: float = 0.0
    history_len: int = 0
    reason: str = "not_ready"


class LiveTemporalPolicy:
    def __init__(self, config: Dict[str, Any], config_dir: Path) -> None:
        self.cfg = config.get("e2e_live_policy", {}) or {}
        checkpoint_value = str(self.cfg.get("checkpoint", "") or "")
        if not checkpoint_value:
            raise RuntimeError("e2e_live_policy.checkpoint is empty")
        self.checkpoint_path = resolve_path(config_dir, checkpoint_value)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"E2E checkpoint not found: {self.checkpoint_path}")

        self.device = self._device(str(self.cfg.get("device", "auto")))
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
        self.policy_cfg = checkpoint["config"]
        data_cfg = self.policy_cfg["data"]
        model_cfg = self.policy_cfg.get("model", {})
        self.mode_names = list(checkpoint.get("mode_names") or ["normal", "straight", "corner", "pivot", "bias"])
        self.history = int(data_cfg.get("history", 10))
        self.image_size = (int(data_cfg.get("image_size", [192, 160])[0]), int(data_cfg.get("image_size", [192, 160])[1]))
        self.scalar_features = list(data_cfg.get("scalar_features", []))
        self.max_dt_s = float(data_cfg.get("max_dt_s", 0.5))
        self.require_full_history = as_bool(self.cfg.get("require_full_history"), True)
        self.frames: Deque[np.ndarray] = deque(maxlen=self.history)
        self.scalars: Deque[np.ndarray] = deque(maxlen=self.history)
        self.prev_time: Optional[float] = None

        self.model = TemporalPolicyNet(
            scalar_dim=len(self.scalar_features),
            num_modes=len(self.mode_names),
            in_channels=2,
            cnn_channels=model_cfg.get("cnn_channels", [24, 32, 48, 64]),
            frame_feature_dim=int(model_cfg.get("frame_feature_dim", 128)),
            scalar_embed_dim=int(model_cfg.get("scalar_embed_dim", 32)),
            temporal_hidden_dim=int(model_cfg.get("temporal_hidden_dim", 128)),
            temporal_layers=int(model_cfg.get("temporal_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            max_motor=float(model_cfg.get("max_motor", 1.0)),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[e2e] checkpoint={self.checkpoint_path} epoch={self.checkpoint_epoch} "
            f"history={self.history} image={self.image_size[0]}x{self.image_size[1]} device={self.device}"
        )

    @staticmethod
    def _device(name: str) -> torch.device:
        value = str(name or "auto").strip().lower()
        if value in {"", "auto"}:
            value = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(value)

    def reset(self) -> None:
        self.frames.clear()
        self.scalars.clear()
        self.prev_time = None

    def _bev_to_tensor(self, mask: np.ndarray) -> np.ndarray:
        import cv2

        img = np.asarray(mask)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = self.image_size
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        max_value = int(img.max()) if img.size else 0
        if max_value <= 2:
            solid = img == 1
            dashed = img == 2
        else:
            solid = (img >= 64) & (img < 192)
            dashed = img >= 192
        return np.stack([solid, dashed], axis=0).astype(np.float32)

    def _scalar_value(self, angle_state: TrailerAngleState, name: str, dt_s: float) -> float:
        if name == "angle_deg":
            return as_float(angle_state.angle_deg, 0.0) / 50.0 if angle_state.angle_deg is not None else 0.0
        if name == "angle_rate_deg_s":
            return float(np.clip(as_float(angle_state.angle_rate_deg_s, 0.0), -150.0, 150.0) / 150.0)
        if name == "angle_confidence":
            return float(np.clip(as_float(angle_state.confidence, 0.0), 0.0, 1.0))
        if name == "angle_age_s":
            return float(np.clip(as_float(angle_state.age_s, 0.0), 0.0, 1.0))
        if name == "angle_ok":
            return 1.0 if bool(angle_state.ok) else 0.0
        if name == "dt_s":
            return float(np.clip(dt_s, 0.0, self.max_dt_s) / max(self.max_dt_s, 1e-6))
        raise KeyError(f"Unknown scalar feature: {name}")

    def _scalar_vector(self, angle_state: TrailerAngleState, now: float) -> np.ndarray:
        dt_s = 0.0 if self.prev_time is None else max(0.0, now - self.prev_time)
        self.prev_time = now
        return np.asarray([self._scalar_value(angle_state, name, dt_s) for name in self.scalar_features], dtype=np.float32)

    @torch.no_grad()
    def predict(self, bev_mask: Optional[np.ndarray], angle_state: TrailerAngleState, now: float) -> E2EPolicyOutput:
        if bev_mask is None:
            return E2EPolicyOutput(False, history_len=len(self.frames), reason="missing_bev_mask")
        try:
            self.frames.append(self._bev_to_tensor(bev_mask))
            self.scalars.append(self._scalar_vector(angle_state, now))
        except Exception as exc:
            return E2EPolicyOutput(False, history_len=len(self.frames), reason=f"preprocess_failed:{exc}")

        history_len = len(self.frames)
        if self.require_full_history and history_len < self.history:
            return E2EPolicyOutput(False, history_len=history_len, reason=f"warmup:{history_len}/{self.history}")

        frame_seq = list(self.frames)
        scalar_seq = list(self.scalars)
        if history_len < self.history:
            frame_seq = [frame_seq[0]] * (self.history - history_len) + frame_seq
            scalar_seq = [scalar_seq[0]] * (self.history - history_len) + scalar_seq

        bev = torch.from_numpy(np.stack(frame_seq, axis=0)[None]).to(self.device).float()
        scalars = torch.from_numpy(np.stack(scalar_seq, axis=0)[None]).to(self.device).float()
        outputs = self.model(bev, scalars)
        wheels = outputs["wheels"][0].detach().cpu().numpy()
        probs = torch.softmax(outputs["mode_logits"], dim=1)[0].detach().cpu().numpy()
        mode_idx = int(probs.argmax())
        return E2EPolicyOutput(
            ready=True,
            raw_left=float(wheels[0]),
            raw_right=float(wheels[1]),
            mode=self.mode_names[mode_idx],
            mode_confidence=float(probs[mode_idx]),
            history_len=history_len,
            reason="ok",
        )


class E2EWheelSafety:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config.get("e2e_live_policy", {}) or {}
        rover_cfg = config.get("rover", {}) or {}
        self.output_scale = as_float(self.cfg.get("output_scale"), 1.0)
        self.max_abs = abs(as_float(self.cfg.get("max_abs_wheel"), as_float(rover_cfg.get("max_wheel_speed"), 1.0)))
        self.max_delta = max(0.0, as_float(self.cfg.get("max_delta_per_frame"), 0.30))
        self.deadband = max(0.0, as_float(self.cfg.get("deadband"), 0.02))
        self.min_mode_confidence = max(0.0, as_float(self.cfg.get("min_mode_confidence"), 0.0))
        self.last_left: Optional[float] = None
        self.last_right: Optional[float] = None

    def reset(self) -> None:
        self.last_left = None
        self.last_right = None

    def apply(self, output: E2EPolicyOutput, active: bool) -> live.WheelCommand:
        if not active:
            self.reset()
            output.sent = False
            output.safe_left = 0.0
            output.safe_right = 0.0
            return live.WheelCommand(0.0, 0.0, False, f"e2e_inactive:{output.reason}")
        if not output.ready:
            self.reset()
            return live.WheelCommand(0.0, 0.0, False, f"e2e_not_ready:{output.reason}")
        if output.mode_confidence < self.min_mode_confidence:
            self.reset()
            return live.WheelCommand(0.0, 0.0, False, f"e2e_low_mode_conf:{output.mode_confidence:.2f}")

        left = clamp(output.raw_left * self.output_scale, -self.max_abs, self.max_abs)
        right = clamp(output.raw_right * self.output_scale, -self.max_abs, self.max_abs)
        if self.last_left is not None and self.max_delta > 0.0:
            left = clamp(left, self.last_left - self.max_delta, self.last_left + self.max_delta)
        if self.last_right is not None and self.max_delta > 0.0:
            right = clamp(right, self.last_right - self.max_delta, self.last_right + self.max_delta)
        if abs(left) < self.deadband:
            left = 0.0
        if abs(right) < self.deadband:
            right = 0.0
        self.last_left = float(left)
        self.last_right = float(right)
        output.safe_left = float(left)
        output.safe_right = float(right)
        output.sent = True
        return live.WheelCommand(
            float(left),
            float(right),
            True,
            f"e2e:{output.mode}:conf={output.mode_confidence:.2f}:scale={self.output_scale:.2f}",
        )


def select_bev_mask(
    fused_bev_mask: Optional[np.ndarray],
    results: Dict[str, live.CameraLaneResult],
) -> Tuple[Optional[np.ndarray], str]:
    if fused_bev_mask is not None:
        return fused_bev_mask, "fused_bev_mask"
    masks = [result.bev_mask for result in results.values() if result.bev_mask is not None]
    if not masks:
        return None, "none"
    if len(masks) == 1:
        return masks[0], "single_camera_bev_mask"
    return np.maximum.reduce(masks), "merged_camera_bev_masks"


def pseudo_command_from_wheel(
    wheel: live.WheelCommand,
    active: bool,
    confidence: float,
    state: str,
    reason: str,
    max_abs_wheel: float,
) -> live.LaneDriveCommand:
    forward = max(0.0, 0.5 * (wheel.left + wheel.right))
    steer = 0.0
    if max_abs_wheel > 1e-6:
        steer = clamp((wheel.left - wheel.right) / (2.0 * max_abs_wheel), -1.0, 1.0)
    return live.LaneDriveCommand(
        active=active,
        valid=bool(wheel.sent),
        confidence=confidence,
        state=state,
        steer=float(steer),
        speed=float(forward),
        brake=not active or not wheel.sent,
        reason=reason,
    )


def blend_wheels(rule: live.WheelCommand, e2e: live.WheelCommand, blend: float) -> live.WheelCommand:
    b = clamp(blend, 0.0, 1.0)
    if not e2e.sent:
        return live.WheelCommand(rule.left, rule.right, rule.sent, f"assist_fallback_rule:{e2e.reason}")
    if not rule.sent:
        return live.WheelCommand(e2e.left, e2e.right, e2e.sent, f"assist_fallback_e2e:{rule.reason}")
    left = rule.left * (1.0 - b) + e2e.left * b
    right = rule.right * (1.0 - b) + e2e.right * b
    return live.WheelCommand(left, right, True, f"assist_blend={b:.2f}:rule={rule.reason}:e2e={e2e.reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live temporal E2E policy runner for dotted lane trailer task.")
    parser.add_argument("--config", type=Path, default=HERE / "dotted_lane_following_config.yaml")
    parser.add_argument("--video", type=Path, default=None, help="Use a video file instead of CSI camera.")
    parser.add_argument("--single-camera", choices=("left", "right", "cam0", "cam1"), default="")
    parser.add_argument("--sensor-id", type=int, default=None)
    parser.add_argument("--start-driving", action="store_true")
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
    parser.add_argument("--e2e-checkpoint", default="")
    parser.add_argument("--e2e-mode", choices=("shadow", "assist", "drive"), default="")
    parser.add_argument("--e2e-output-scale", type=float, default=0.0)
    parser.add_argument("--e2e-max-abs-wheel", type=float, default=0.0)
    parser.add_argument("--e2e-max-delta", type=float, default=-1.0)
    parser.add_argument("--e2e-assist-blend", type=float, default=-1.0)
    parser.add_argument("--save-e2e-dataset", action="store_true", help="Allow E2E dataset labels from selected live output.")
    return parser.parse_args()


def apply_e2e_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    cfg = config.setdefault("e2e_live_policy", {})
    if args.e2e_checkpoint:
        cfg["checkpoint"] = args.e2e_checkpoint
    if args.e2e_mode:
        cfg["mode"] = args.e2e_mode
    if args.e2e_output_scale > 0.0:
        cfg["output_scale"] = float(args.e2e_output_scale)
    if args.e2e_max_abs_wheel > 0.0:
        cfg["max_abs_wheel"] = float(args.e2e_max_abs_wheel)
    if args.e2e_max_delta >= 0.0:
        cfg["max_delta_per_frame"] = float(args.e2e_max_delta)
    if args.e2e_assist_blend >= 0.0:
        cfg["assist_blend"] = float(args.e2e_assist_blend)
    if not args.save_e2e_dataset:
        config.setdefault("e2e_dataset", {})["enabled"] = False


def make_policy_log_writer(run_dir: Path) -> Tuple[csv.DictWriter, Any]:
    path = run_dir / "e2e_policy_live_log.csv"
    f = path.open("w", newline="", encoding="utf-8")
    fields = [
        "frame_idx",
        "timestamp_monotonic",
        "wall_time",
        "active",
        "e2e_mode",
        "bev_source",
        "history_len",
        "e2e_ready",
        "e2e_pred_mode",
        "e2e_mode_confidence",
        "e2e_raw_left",
        "e2e_raw_right",
        "e2e_safe_left",
        "e2e_safe_right",
        "rule_left",
        "rule_right",
        "rule_sent",
        "selected_left",
        "selected_right",
        "selected_sent",
        "trailer_angle_deg",
        "trailer_angle_confidence",
        "route_mode",
        "selected_reason",
        "e2e_reason",
        "rule_reason",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    print(f"[e2e-log] {path}")
    return writer, f


def write_policy_log(
    writer: csv.DictWriter,
    frame_idx: int,
    active: bool,
    mode: str,
    bev_source: str,
    e2e: E2EPolicyOutput,
    rule_wheel: live.WheelCommand,
    selected_wheel: live.WheelCommand,
    angle_state: TrailerAngleState,
    route_debug: Optional[RouteControlDebug],
) -> None:
    writer.writerow(
        {
            "frame_idx": frame_idx,
            "timestamp_monotonic": f"{time.monotonic():.3f}",
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "active": int(active),
            "e2e_mode": mode,
            "bev_source": bev_source,
            "history_len": e2e.history_len,
            "e2e_ready": int(e2e.ready),
            "e2e_pred_mode": e2e.mode,
            "e2e_mode_confidence": f"{e2e.mode_confidence:.4f}",
            "e2e_raw_left": f"{e2e.raw_left:.4f}",
            "e2e_raw_right": f"{e2e.raw_right:.4f}",
            "e2e_safe_left": f"{e2e.safe_left:.4f}",
            "e2e_safe_right": f"{e2e.safe_right:.4f}",
            "rule_left": f"{rule_wheel.left:.4f}",
            "rule_right": f"{rule_wheel.right:.4f}",
            "rule_sent": int(rule_wheel.sent),
            "selected_left": f"{selected_wheel.left:.4f}",
            "selected_right": f"{selected_wheel.right:.4f}",
            "selected_sent": int(selected_wheel.sent),
            "trailer_angle_deg": "" if angle_state.angle_deg is None else f"{angle_state.angle_deg:.4f}",
            "trailer_angle_confidence": f"{angle_state.confidence:.4f}",
            "route_mode": "" if route_debug is None else route_debug.mode,
            "selected_reason": selected_wheel.reason,
            "e2e_reason": e2e.reason,
            "rule_reason": rule_wheel.reason,
        }
    )


def main() -> int:
    signal.signal(signal.SIGINT, live._handle_signal)
    signal.signal(signal.SIGTERM, live._handle_signal)
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    live.apply_overrides(config, args)
    apply_e2e_overrides(config, args)

    e2e_cfg = config.get("e2e_live_policy", {}) or {}
    e2e_mode = str(e2e_cfg.get("mode", "shadow")).strip().lower()
    if e2e_mode not in {"shadow", "assist", "drive"}:
        e2e_mode = "shadow"
    print(f"[e2e] live mode={e2e_mode}")
    if e2e_mode == "drive" and not args.start_driving:
        print("[e2e] drive mode selected, but --start-driving is not set; motors stay inactive until active.")

    mirror_configs: List[Path] = []
    parking_config = config_dir / "trailer_parking_config.yaml"
    if parking_config.exists() and parking_config.resolve() != config_path:
        mirror_configs.append(parking_config)
    http_streamer = live.HttpMjpegStreamer(
        config.get("http_stream", {}) or {},
        app_config=config,
        config_path=config_path,
        mirror_config_paths=mirror_configs,
    )
    http_streamer.write(
        live.make_status_frame(
            http_streamer.width,
            http_streamer.height,
            ["Rover E2E temporal policy", "initializing models and cameras..."],
        )
    )

    predictor = live.load_predictor(config, config_dir)
    panel_model = live.load_panel_model(config, config_dir)
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

    bev = live.make_bev_config(config)
    dual_bev = live.load_dual_bev_calibration(config, config_dir)
    print(
        f"[dual_bev] {'enabled' if dual_bev.enabled else 'disabled/fallback'} "
        f"source={dual_bev.source} drive_mode={dual_bev.drive_mode}"
    )
    camera_cfgs = live.enabled_camera_configs(config, str((config.get("camera", {}) or {}).get("single_camera", "")))
    lane_cfgs: Dict[str, LaneFollowerConfig] = {
        key: live.make_lane_config(config, camera_cfg) for key, camera_cfg in camera_cfgs.items()
    }
    lane_states: Dict[str, LaneFollowerState] = {key: LaneFollowerState() for key in lane_cfgs.keys()}
    fused_lane_cfg = live.make_lane_config(config)
    fused_lane_cfg.vehicle_center_x_bias = dual_bev.vehicle_center_x_bias
    fused_lane_state = LaneFollowerState()
    if args.video is not None and not lane_cfgs:
        lane_cfgs["video"] = live.make_lane_config(config)
        lane_states["video"] = LaneFollowerState()

    e2e_policy = LiveTemporalPolicy(config, config_dir)
    e2e_safety = E2EWheelSafety(config)
    max_abs_for_display = max(1e-6, e2e_safety.max_abs)
    matrix_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
    route_controller = RouteAwareTrailerController(config)
    angle_rate_filter = TrailerAngleRateFilter(config)
    corner_pivot_controller = live.CornerPivotController(config)
    rule_wheel_mixer = live.LaneWheelMixer(config)
    source = live.open_source(config, args)

    rover_cfg = config.get("rover", {}) or {}
    rover = RoverSerial(
        str(rover_cfg.get("serial", "/dev/ttyUSB0")),
        as_int(rover_cfg.get("baud"), 115200),
        as_bool(rover_cfg.get("arm"), False),
    )
    repeater = make_repeater_if_enabled(config, rover)
    runtime_cfg = config.get("runtime", {}) or {}
    active = bool(args.start_driving or as_bool(runtime_cfg.get("auto_start"), False))
    display = as_bool(runtime_cfg.get("display"), False)
    print_every = args.print_every if args.print_every > 0 else as_int(runtime_cfg.get("print_every"), 5)
    max_frames = args.max_frames if args.max_frames > 0 else as_int(runtime_cfg.get("max_frames"), 0)
    log_root = resolve_path(config_dir, str(runtime_cfg.get("log_dir", "logs/dotted_lane_following_run")))
    run_dir = log_root / f"e2e_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_writer, log_file = live.make_log_writer(run_dir)
    policy_writer, policy_log_file = make_policy_log_writer(run_dir)
    primary_lane_cfg = next(iter(lane_cfgs.values())) if lane_cfgs else live.make_lane_config(config)
    live.save_run_config(run_dir, config, bev, primary_lane_cfg)
    e2e_writer = live.E2EDatasetWriter(run_dir, config, bev)
    streamer = live.UdpVideoStreamer(config.get("stream", {}) or {})
    save_video_enabled = as_bool(runtime_cfg.get("save_video"), False)
    debug_enabled = bool(display or streamer.enabled or http_streamer.enabled or save_video_enabled)
    need_full_debug_panel = bool(display or streamer.enabled or save_video_enabled)
    video_writer = None
    detections: Dict[str, PanelDetection] = {}
    angle_estimates: Dict[str, live.AngleEstimate] = {}
    frame_idx = 0
    prev_time = time.monotonic()
    last_wheel_signature: Optional[Tuple[float, float, bool]] = None
    last_command_change_time = prev_time
    command_change_hz = 0.0
    command_repeat_hz = as_float(rover_cfg.get("command_rate_hz"), 20.0) if repeater is not None else 0.0
    print("[lane] driving started" if active else "[lane] preview only: use --start-driving to send wheel commands")

    try:
        while not live._STOP:
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

            fused_angle = live.estimate_trailer_angle(
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
                live.apply_lane_lookahead(lane_cfgs, fused_lane_cfg, route_controller.scheduled_lookahead(angle_state))

            results: Dict[str, live.CameraLaneResult] = {}
            warped_masks: List[np.ndarray] = []
            fused_bev_mask: Optional[np.ndarray] = None
            fused_bev_panel: Optional[np.ndarray] = None
            fused_estimate: Optional[LaneEstimate] = None
            for camera_key, frame in frames.items():
                camera_cfg = camera_cfgs.get(camera_key, {})
                lane_cfg = lane_cfgs.get(camera_key)
                if lane_cfg is None:
                    lane_cfg = live.make_lane_config(config, camera_cfg)
                    lane_cfgs[camera_key] = lane_cfg
                lane_state = lane_states.setdefault(camera_key, LaneFollowerState())
                crop_bgr, _crop_xyxy = live.crop_frame(frame, config, camera_cfg)
                h, w = crop_bgr.shape[:2]
                cache_key = (camera_key, w, h)
                if cache_key not in matrix_cache:
                    matrix_cache[cache_key], _ = perspective_matrices(w, h, bev)

                mask = predictor.predict(crop_bgr)
                if dual_bev.enabled:
                    bev_mask = live.dual_warp_mask(mask, camera_key, dual_bev, bev)
                    if bev_mask is None:
                        bev_mask = warp_to_bev(mask, bev, matrix_cache[cache_key], is_mask=True)
                    else:
                        warped_masks.append(bev_mask)
                    estimate = live.estimate_lane_with_solid_fallback(bev_mask, fused_lane_cfg, lane_state, config)
                else:
                    bev_mask = warp_to_bev(mask, bev, matrix_cache[cache_key], is_mask=True)
                    estimate = live.estimate_lane_with_solid_fallback(bev_mask, lane_cfg, lane_state, config)
                panel = (
                    live.make_debug_panel(crop_bgr, mask, bev_mask, estimate, bev)
                    if debug_enabled
                    else np.zeros((1, 1, 3), dtype=np.uint8)
                )
                results[camera_key] = live.CameraLaneResult(
                    camera_key=camera_key,
                    frame_bgr=frame,
                    crop_bgr=crop_bgr,
                    mask=mask,
                    bev_mask=bev_mask,
                    estimate=estimate,
                    panel=panel,
                )

            if dual_bev.enabled and warped_masks:
                fused_bev_mask = live.merge_bev_masks(warped_masks, dual_bev, bev)
                fused_estimate = live.estimate_lane_with_solid_fallback(fused_bev_mask, fused_lane_cfg, fused_lane_state, config)
                if dual_bev.drive_mode == "estimate_fusion":
                    base_drive_command = live.fuse_lane_results(results, config, active)
                    base_drive_command.state = f"DUAL_BEV_EST_FUSION:{base_drive_command.state}"
                    base_drive_command.reason = f"dual_bev_estimate_fusion {base_drive_command.reason}"
                else:
                    base_drive_command = live.make_drive_command(fused_estimate, config, active)
                    base_drive_command.state = f"DUAL_BEV:{fused_estimate.state}"
                    base_drive_command.reason = f"dual_bev {fused_estimate.reason}"
                if debug_enabled:
                    fused_bev_panel = draw_bev_debug(fused_bev_mask, fused_estimate, bev)
            else:
                base_drive_command = live.fuse_lane_results(results, config, active)

            rule_drive_command = base_drive_command
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
                if live.use_simple_dashed_priority(config, route_controller, lane_signal, angle_state, base_drive_command):
                    rule_drive_command = live.simple_dashed_command(base_drive_command, active, config)
                else:
                    route_output = route_controller.control(lane_signal, angle_state, active, now=now)
                    rule_drive_command = live.LaneDriveCommand(
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

            rule_drive_command = corner_pivot_controller.override(
                fused_estimate,
                results,
                angle_state,
                rule_drive_command,
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
                rule_drive_command = live.LaneDriveCommand(
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

            rule_wheel = rule_wheel_mixer.mix(rule_drive_command, now)
            policy_bev_mask, bev_source = select_bev_mask(fused_bev_mask, results)
            e2e_output = e2e_policy.predict(policy_bev_mask, angle_state, now)
            e2e_wheel = e2e_safety.apply(e2e_output, active)

            if e2e_mode == "drive":
                selected_wheel = e2e_wheel
                selected_command = pseudo_command_from_wheel(
                    selected_wheel,
                    active,
                    e2e_output.mode_confidence,
                    f"E2E_DRIVE:{e2e_output.mode}",
                    selected_wheel.reason,
                    max_abs_for_display,
                )
            elif e2e_mode == "assist":
                selected_wheel = blend_wheels(rule_wheel, e2e_wheel, as_float(e2e_cfg.get("assist_blend"), 0.5))
                selected_command = pseudo_command_from_wheel(
                    selected_wheel,
                    active,
                    max(rule_drive_command.confidence, e2e_output.mode_confidence),
                    f"E2E_ASSIST:{e2e_output.mode}|rule={rule_drive_command.state}",
                    selected_wheel.reason,
                    max_abs_for_display,
                )
            else:
                selected_wheel = rule_wheel
                selected_command = rule_drive_command

            wheel_signature = (round(selected_wheel.left, 3), round(selected_wheel.right, 3), bool(selected_wheel.sent))
            if wheel_signature != last_wheel_signature:
                command_change_now = time.monotonic()
                if last_wheel_signature is not None:
                    command_change_hz = 1.0 / max(1e-6, command_change_now - last_command_change_time)
                last_command_change_time = command_change_now
                last_wheel_signature = wheel_signature

            if repeater is not None:
                repeater.update(selected_wheel.left, selected_wheel.right, selected_wheel.sent)
            elif selected_wheel.sent:
                rover.send(selected_wheel.left, selected_wheel.right)
            else:
                rover.send(0.0, 0.0)

            e2e_writer.write(
                frame_idx,
                fused_bev_mask,
                results,
                fused_estimate,
                selected_command,
                selected_wheel,
                angle_state,
                route_debug,
            )
            write_policy_log(
                policy_writer,
                frame_idx,
                active,
                e2e_mode,
                bev_source,
                e2e_output,
                rule_wheel,
                selected_wheel,
                angle_state,
                route_debug,
            )

            panel: Optional[np.ndarray] = None
            if need_full_debug_panel:
                panel = live.make_combined_debug_panel(
                    results,
                    selected_command,
                    selected_wheel,
                    rover.armed,
                    fused_bev_panel,
                    route_debug,
                    angle_state,
                    config,
                )
                streamer.write(panel)

            if http_streamer.enabled:
                fallback_panel = panel
                if fallback_panel is None:
                    fallback_panel = np.zeros((http_streamer.height, http_streamer.width, 3), dtype=np.uint8)
                http_panel = live.make_http_debug_panel(
                    config,
                    fallback_panel,
                    results,
                    selected_command,
                    selected_wheel,
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
                    panel = live.make_combined_debug_panel(
                        results,
                        selected_command,
                        selected_wheel,
                        rover.armed,
                        fused_bev_panel,
                        route_debug,
                        angle_state,
                        config,
                    )
                if video_writer is None:
                    video_path = run_dir / "e2e_temporal_debug.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (panel.shape[1], panel.shape[0]))
                    print(f"[video] {video_path}")
                video_writer.write(panel)

            live.write_log_row(
                log_writer,
                frame_idx,
                fps,
                selected_command,
                results,
                selected_wheel,
                angle_state,
                route_debug,
                detections,
                angle_estimates,
            )
            if frame_idx % 20 == 0:
                log_file.flush()
                policy_log_file.flush()

            if print_every > 0 and frame_idx % print_every == 0:
                mode_text = selected_command.state
                if route_debug is not None and not mode_text.startswith("E2E"):
                    mode_text = route_debug.mode
                angle_text = "--" if angle_state.angle_deg is None else f"{angle_state.angle_deg:+.1f}"
                panel_text = " ".join(
                    live.panel_console_token(key, detections.get(key), angle_estimates.get(key))
                    for key in ("cam1", "cam0")
                )
                cam_text = " ".join(
                    f"{key}:{result.estimate.state[:8]}:{result.estimate.confidence:.2f}"
                    for key, result in sorted(results.items())
                )
                print(
                    f"[{frame_idx:06d}] {'RUN' if active else 'PRE'} {e2e_mode.upper()} {mode_text} "
                    f"angle={angle_text} e2e={e2e_output.mode}:{e2e_output.mode_confidence:.2f} "
                    f"raw={e2e_output.raw_left:+.2f}/{e2e_output.raw_right:+.2f} "
                    f"safe={e2e_wheel.left:+.2f}/{e2e_wheel.right:+.2f} "
                    f"rule={rule_wheel.left:+.2f}/{rule_wheel.right:+.2f} "
                    f"wheel={selected_wheel.left:+.2f}/{selected_wheel.right:+.2f} "
                    f"hist={e2e_output.history_len}/{e2e_policy.history} "
                    f"ctrlHz={fps:.1f} cmdChangeHz={command_change_hz:.1f}/{command_repeat_hz:.0f} "
                    f"bev={bev_source} reason={selected_wheel.reason} panel={panel_text} {cam_text}"
                )

            if display:
                try:
                    import cv2

                    shown = panel
                    if shown is None:
                        shown = np.zeros((720, 960, 3), dtype=np.uint8)
                    scale = as_float(runtime_cfg.get("display_scale"), 1.0)
                    if scale > 0.0 and abs(scale - 1.0) > 1e-3:
                        shown = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("e2e_temporal_policy", shown)
                    key_code = cv2.waitKey(1) & 0xFF
                except Exception as exc:
                    print(f"[display] disabled: {exc}")
                    display = False
                    key_code = 255
                if key_code in (ord("q"), 27):
                    break
                if key_code == ord("p"):
                    active = True
                    e2e_policy.reset()
                    e2e_safety.reset()
                    print("[lane] driving started")
                elif key_code in (ord("x"), ord(" ")):
                    active = False
                    e2e_policy.reset()
                    e2e_safety.reset()
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
        policy_log_file.flush()
        policy_log_file.close()
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

