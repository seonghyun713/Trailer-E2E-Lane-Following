#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trailer_parking_core import FusedAngleEstimate, as_bool, as_float, as_int, clamp


@dataclass
class TrailerAngleState:
    ok: bool
    angle_deg: Optional[float]
    angle_rate_deg_s: float
    confidence: float
    source: str
    age_s: float
    reason: str = ""


@dataclass
class LaneSignal:
    valid: bool
    confidence: float
    e_y_norm: float
    e_psi_rad: float
    speed_base: float
    row_count: int
    curvature_proxy: float
    turn_sign: float
    corner_confidence: float
    state: str
    reason: str
    left_solid_offset_norm: Optional[float] = None
    right_solid_offset_norm: Optional[float] = None


@dataclass
class RouteDriveOutput:
    active: bool
    valid: bool
    confidence: float
    state: str
    steer: float
    speed: float
    brake: bool
    reason: str


@dataclass
class RouteControlDebug:
    mode: str
    lane_state: str
    lane_row_count: int
    alpha_deg: Optional[float]
    alpha_rate_deg_s: float
    alpha_ref_deg: float
    angle_band: str
    lookahead_y_ratio: float
    lane_scale: float
    trailer_scale: float
    speed_scale: float
    u_lane: float
    u_curve: float
    u_trailer: float
    u_total: float
    curvature_proxy: float
    corner_confidence: float
    target_bias_norm: float = 0.0
    target_bias_fraction: float = 0.0


class TrailerAngleRateFilter:
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("route_controller", {}) or {}
        self.min_confidence = as_float(cfg.get("min_angle_confidence"), 0.16)
        self.lost_grace_s = as_float(cfg.get("angle_lost_grace_s"), 0.75)
        self.rate_alpha = clamp(as_float(cfg.get("alpha_rate_filter"), 0.35), 0.0, 1.0)
        self.max_rate = max(1.0, as_float(cfg.get("max_alpha_rate_deg_s"), 160.0))
        self.last_angle: Optional[float] = None
        self.last_time: Optional[float] = None
        self.last_good_time: Optional[float] = None
        self.filtered_rate = 0.0

    def update(self, fused: Optional[FusedAngleEstimate], now: Optional[float] = None) -> TrailerAngleState:
        now = time.monotonic() if now is None else float(now)
        if fused is not None and fused.ok and fused.angle_deg is not None and fused.confidence >= self.min_confidence:
            angle = float(fused.angle_deg)
            rate = self.filtered_rate
            if self.last_angle is not None and self.last_time is not None:
                dt = max(1e-3, now - self.last_time)
                raw_rate = (angle - self.last_angle) / dt
                raw_rate = clamp(raw_rate, -self.max_rate, self.max_rate)
                rate = (1.0 - self.rate_alpha) * self.filtered_rate + self.rate_alpha * raw_rate
            self.last_angle = angle
            self.last_time = now
            self.last_good_time = now
            self.filtered_rate = rate
            return TrailerAngleState(
                ok=True,
                angle_deg=angle,
                angle_rate_deg_s=rate,
                confidence=float(fused.confidence),
                source=str(fused.source),
                age_s=0.0,
                reason=str(fused.message),
            )

        age = float("inf") if self.last_good_time is None else now - self.last_good_time
        if self.last_angle is not None and age <= self.lost_grace_s:
            self.filtered_rate *= 0.80
            return TrailerAngleState(
                ok=True,
                angle_deg=self.last_angle,
                angle_rate_deg_s=self.filtered_rate,
                confidence=0.0,
                source="hold",
                age_s=age,
                reason="angle_lost_hold",
            )

        return TrailerAngleState(
            ok=False,
            angle_deg=self.last_angle,
            angle_rate_deg_s=0.0,
            confidence=0.0,
            source="lost",
            age_s=age,
            reason="angle_lost",
        )


class RouteAwareTrailerController:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("route_controller", {}) or {}
        self.lane_cfg = config.get("lane", {}) or {}
        self.enabled = as_bool(self.cfg.get("enabled"), False)
        self.mode = "STRAIGHT"
        self.corner_sign = 0.0
        self.corner_low_count = 0
        self.frames_in_mode = 0
        self.post_corner_started_at: Optional[float] = None
        self.post_corner_bias_started_at: Optional[float] = None
        self.post_corner_bias_sign = 0.0
        self.corner_blind_until = 0.0
        self.filtered_lookahead: Optional[float] = None
        self.last_debug = RouteControlDebug(
            mode="DISABLED",
            lane_state="",
            lane_row_count=0,
            alpha_deg=None,
            alpha_rate_deg_s=0.0,
            alpha_ref_deg=0.0,
            angle_band="unknown",
            lookahead_y_ratio=as_float(self.lane_cfg.get("lookahead_y_ratio"), 0.50),
            lane_scale=1.0,
            trailer_scale=0.0,
            speed_scale=1.0,
            u_lane=0.0,
            u_curve=0.0,
            u_trailer=0.0,
            u_total=0.0,
            curvature_proxy=0.0,
            corner_confidence=0.0,
        )

    def scheduled_lookahead(self, angle: TrailerAngleState) -> float:
        base = as_float(self.lane_cfg.get("lookahead_y_ratio"), 0.50)
        if not self.enabled:
            return base
        target = base
        alpha_abs = abs(angle.angle_deg) if angle.angle_deg is not None else 0.0
        recovery_start = as_float(self.cfg.get("recovery_start_abs_angle_deg"), 48.0)
        caution = as_float(self.cfg.get("straight_caution_abs_angle_deg"), 15.0)
        if self.mode in {"CORNER_APPROACH", "CORNERING"}:
            target = as_float(self.cfg.get("corner_lookahead_y_ratio"), 0.34)
        elif self.mode == "POST_CORNER_BIAS":
            target = as_float(self.cfg.get("post_corner_bias_lookahead_y_ratio"), base)
        elif self.mode == "POST_CORNER_SETTLE":
            target = as_float(self.cfg.get("post_corner_lookahead_y_ratio"), 0.70)
        elif alpha_abs >= recovery_start:
            target = as_float(self.cfg.get("recovery_lookahead_y_ratio"), 0.32)
        elif alpha_abs >= caution:
            t = smoothstep(caution, recovery_start, alpha_abs)
            target = lerp(base, as_float(self.cfg.get("straight_far_lookahead_y_ratio"), 0.38), t)

        smoothing = clamp(as_float(self.cfg.get("lookahead_smoothing"), 0.65), 0.0, 0.98)
        if self.filtered_lookahead is None:
            self.filtered_lookahead = target
        else:
            self.filtered_lookahead = smoothing * self.filtered_lookahead + (1.0 - smoothing) * target
        return float(clamp(self.filtered_lookahead, 0.05, 0.98))

    def control(self, lane: LaneSignal, angle: TrailerAngleState, active: bool, now: Optional[float] = None) -> RouteDriveOutput:
        _ = time.monotonic() if now is None else float(now)
        if not self.enabled:
            return RouteDriveOutput(active, lane.valid, lane.confidence, lane.state, 0.0, lane.speed_base, not active, "route_controller_disabled")
        alpha = float(angle.angle_deg) if angle.angle_deg is not None else 0.0
        alpha_abs = abs(alpha)
        angle_band = self.angle_band(alpha_abs)
        if active and not lane.valid:
            lookahead = (
                float(self.filtered_lookahead)
                if self.filtered_lookahead is not None
                else as_float(self.lane_cfg.get("lookahead_y_ratio"), 0.50)
            )
            self.last_debug = self._make_debug(
                lane,
                angle,
                0.0,
                lookahead,
                0.0,
                0.0,
                0.0,
                f"LOST:{lane.state}",
                0.0,
                0.0,
                angle_band,
                u_trailer=0.0,
                u_total=0.0,
            )
            return RouteDriveOutput(True, False, lane.confidence, f"ROUTE_LOST:{lane.state}", 0.0, 0.0, True, lane.reason)

        hard_stop_enabled = as_bool(self.cfg.get("hard_stop_enabled"), True)
        hard_stop = as_float(self.cfg.get("hard_stop_abs_angle_deg"), 62.0)
        if hard_stop_enabled and active and angle.ok and alpha_abs >= hard_stop:
            debug = self._make_debug(lane, angle, 0.0, 0.0, 0.0, 0.0, 0.0, "FAULT", 0.0, 0.0, angle_band)
            self.last_debug = debug
            return RouteDriveOutput(
                True,
                False,
                lane.confidence,
                f"ROUTE_FAULT:{angle_band}",
                0.0,
                0.0,
                True,
                f"trailer_hard_angle:{alpha:.1f}>={hard_stop:.1f}",
            )

        self._update_mode(lane, alpha_abs, _)
        lookahead = self.scheduled_lookahead(angle)
        alpha_ref = self._alpha_reference(lane)
        lane_scale = self._lane_scale(alpha_abs)
        trailer_scale = self._trailer_scale(alpha_abs)
        speed_scale = self._speed_scale(alpha_abs, lane.corner_confidence)
        target_bias_norm, target_bias_fraction = self._post_corner_target_bias(lane, _)
        e_y_control = clamp(lane.e_y_norm + target_bias_norm, -1.5, 1.5)

        ky = as_float(self.cfg.get("lateral_gain"), as_float(self.lane_cfg.get("lateral_gain"), 0.60))
        kpsi = as_float(self.cfg.get("heading_gain"), as_float(self.lane_cfg.get("heading_gain"), 0.47))
        kcurve = as_float(self.cfg.get("curve_ff_gain"), 0.30)
        kalpha = self._alpha_gain()
        krate = as_float(self.cfg.get("alpha_rate_gain"), 0.0016)
        lane_sign = as_float(self.cfg.get("lane_feedback_sign"), 1.0)
        curve_sign = as_float(self.cfg.get("curve_feedback_sign"), 1.0)
        alpha_sign = as_float(self.cfg.get("alpha_feedback_sign"), 1.0)

        u_lane = lane_sign * lane_scale * (ky * e_y_control + kpsi * lane.e_psi_rad)
        u_curve = curve_sign * kcurve * lane.curvature_proxy * self._curve_scale()
        alpha_error = alpha - alpha_ref
        u_trailer = alpha_sign * trailer_scale * (kalpha * alpha_error + krate * angle.angle_rate_deg_s)
        max_steer = self._max_steer()
        u_total = clamp(u_lane + u_curve + u_trailer, -max_steer, max_steer)

        base_speed = lane.speed_base
        if as_float(self.cfg.get("base_speed"), 0.0) > 0.0:
            base_speed = as_float(self.cfg.get("base_speed"), base_speed)
        min_speed = as_float(self.cfg.get("min_speed"), as_float(self.lane_cfg.get("min_speed"), 0.08))
        max_speed = as_float(self.cfg.get("max_speed"), as_float(self.lane_cfg.get("max_speed"), 0.26))
        speed = clamp(base_speed * speed_scale, min_speed, max_speed)
        if lane.confidence < as_float(self.cfg.get("slow_confidence_threshold"), 0.40):
            speed *= clamp(lane.confidence / max(1e-6, as_float(self.cfg.get("slow_confidence_threshold"), 0.40)), 0.35, 1.0)

        self.last_debug = self._make_debug(
            lane,
            angle,
            alpha_ref,
            lookahead,
            lane_scale,
            trailer_scale,
            speed_scale,
            angle_band,
            u_lane,
            u_curve,
            angle_band,
            u_trailer=u_trailer,
            u_total=u_total,
            target_bias_norm=target_bias_norm,
            target_bias_fraction=target_bias_fraction,
        )
        reason = (
            f"{self.mode} {angle_band} "
            f"ey={lane.e_y_norm:+.3f}->{e_y_control:+.3f} bias={target_bias_norm:+.3f} "
            f"epsi={lane.e_psi_rad:+.3f} "
            f"curv={lane.curvature_proxy:+.3f} alpha={alpha:+.1f}/{alpha_ref:+.1f}"
        )
        if not active:
            return RouteDriveOutput(False, lane.valid, lane.confidence, f"ROUTE:{self.mode}:PREVIEW", 0.0, lane.speed_base, True, "preview")
        return RouteDriveOutput(True, True, lane.confidence, f"ROUTE:{self.mode}", float(u_total), float(speed), False, reason)

    def _update_mode(self, lane: LaneSignal, alpha_abs: float, now: float) -> None:
        enter = as_float(self.cfg.get("corner_enter_confidence"), 0.45)
        exit_conf = as_float(self.cfg.get("corner_exit_confidence"), 0.22)
        min_corner_frames = max(0, as_int(self.cfg.get("min_corner_frames"), 8))
        recovery_start = as_float(self.cfg.get("recovery_start_abs_angle_deg"), 48.0)
        recovery_exit = as_float(self.cfg.get("recovery_exit_abs_angle_deg"), 35.0)
        post_bias_blind = as_bool(self.cfg.get("post_corner_bias_ignore_corner_confidence"), True)
        corner_entry_conf = 0.0 if post_bias_blind and now < self.corner_blind_until else lane.corner_confidence

        previous = self.mode
        if alpha_abs >= recovery_start:
            self.mode = "RECOVERY"
        elif self.mode == "RECOVERY":
            if alpha_abs <= recovery_exit:
                self.mode = "CORNER_EXIT" if lane.corner_confidence > exit_conf else self._post_corner_or_straight(now)
        elif self.mode in {"CORNER_APPROACH", "CORNERING"}:
            if lane.corner_confidence <= exit_conf:
                self.corner_low_count += 1
            else:
                self.corner_low_count = 0
            if self.corner_low_count >= max(2, min_corner_frames // 2) and self.frames_in_mode >= min_corner_frames:
                self.mode = "CORNER_EXIT"
            elif self.frames_in_mode >= max(2, min_corner_frames // 2):
                self.mode = "CORNERING"
        elif self.mode == "CORNER_EXIT":
            if lane.corner_confidence >= enter:
                self.mode = "CORNER_APPROACH"
            elif lane.corner_confidence <= exit_conf or alpha_abs <= as_float(self.cfg.get("straight_caution_abs_angle_deg"), 15.0):
                self.mode = self._post_corner_or_straight(now)
        elif self.mode == "POST_CORNER_SETTLE":
            if corner_entry_conf >= enter:
                self.mode = "CORNER_APPROACH"
                self.post_corner_started_at = None
            elif self._post_corner_done(alpha_abs, now):
                self.mode = "STRAIGHT"
                self.post_corner_started_at = None
        elif self.mode == "POST_CORNER_BIAS":
            bias_elapsed = self._post_corner_bias_elapsed(now)
            bias_min_s = max(0.0, as_float(self.cfg.get("post_corner_bias_min_s"), 0.8))
            if post_bias_blind:
                blind_after_s = max(0.0, as_float(self.cfg.get("post_corner_bias_corner_blind_after_s"), 0.4))
                self.corner_blind_until = max(self.corner_blind_until, now + blind_after_s)
            if self._post_corner_bias_done(now):
                self.mode = "STRAIGHT"
                self.post_corner_bias_started_at = None
                self.post_corner_bias_sign = 0.0
            elif not post_bias_blind and bias_elapsed >= bias_min_s and lane.corner_confidence >= enter:
                self.mode = "CORNER_APPROACH"
                self.post_corner_bias_started_at = None
                self.post_corner_bias_sign = 0.0
        else:
            if corner_entry_conf >= enter:
                self.mode = "CORNER_APPROACH"
            else:
                self.mode = "STRAIGHT"

        if self.mode != "POST_CORNER_BIAS" and lane.turn_sign != 0.0 and lane.corner_confidence >= exit_conf:
            self.corner_sign = lane.turn_sign
        if self.mode == "STRAIGHT":
            self.corner_sign = 0.0
            self.corner_low_count = 0
            self.post_corner_started_at = None
            self.post_corner_bias_started_at = None
            self.post_corner_bias_sign = 0.0
        if previous != self.mode:
            self.frames_in_mode = 0
        else:
            self.frames_in_mode += 1

    def _post_corner_or_straight(self, now: float) -> str:
        if as_bool(self.cfg.get("post_corner_bias_enabled"), False):
            self.post_corner_started_at = None
            self.post_corner_bias_started_at = now
            self.post_corner_bias_sign = self.corner_sign
            return "POST_CORNER_BIAS"
        if not as_bool(self.cfg.get("post_corner_settle_enabled"), False):
            self.post_corner_started_at = None
            self.post_corner_bias_started_at = None
            self.post_corner_bias_sign = 0.0
            return "STRAIGHT"
        self.post_corner_bias_started_at = None
        self.post_corner_bias_sign = 0.0
        self.post_corner_started_at = now
        return "POST_CORNER_SETTLE"

    def force_post_corner_bias(self, now: Optional[float] = None, corner_sign: float = 0.0) -> bool:
        if not self.enabled or not as_bool(self.cfg.get("post_corner_bias_enabled"), False):
            return False
        now = time.monotonic() if now is None else float(now)
        self.mode = "POST_CORNER_BIAS"
        self.post_corner_started_at = None
        self.post_corner_bias_started_at = now
        self.corner_low_count = 0
        self.frames_in_mode = 0
        if corner_sign != 0.0:
            self.corner_sign = float(corner_sign)
        self.post_corner_bias_sign = self.corner_sign
        return True

    def _post_corner_done(self, alpha_abs: float, now: float) -> bool:
        started = now if self.post_corner_started_at is None else self.post_corner_started_at
        self.post_corner_started_at = started
        elapsed = max(0.0, now - started)
        min_s = max(0.0, as_float(self.cfg.get("post_corner_settle_min_s"), 1.2))
        max_s = max(min_s, as_float(self.cfg.get("post_corner_settle_max_s"), 3.0))
        exit_angle = max(0.0, as_float(self.cfg.get("post_corner_settle_exit_abs_angle_deg"), 12.0))
        return elapsed >= max_s or (elapsed >= min_s and alpha_abs <= exit_angle)

    def _post_corner_bias_elapsed(self, now: float) -> float:
        started = now if self.post_corner_bias_started_at is None else self.post_corner_bias_started_at
        self.post_corner_bias_started_at = started
        return max(0.0, now - started)

    def _post_corner_bias_done(self, now: float) -> bool:
        elapsed = self._post_corner_bias_elapsed(now)
        min_s = max(0.0, as_float(self.cfg.get("post_corner_bias_min_s"), 0.8))
        max_s = max(min_s, as_float(self.cfg.get("post_corner_bias_max_s"), 2.4))
        ramp_s = max(1e-3, as_float(self.cfg.get("post_corner_bias_ramp_s"), max_s))
        return elapsed >= max_s or (elapsed >= min_s and elapsed >= ramp_s)

    def _post_corner_target_bias(self, lane: LaneSignal, now: float) -> Tuple[float, float]:
        if self.mode != "POST_CORNER_BIAS":
            return 0.0, 0.0
        started = now if self.post_corner_bias_started_at is None else self.post_corner_bias_started_at
        self.post_corner_bias_started_at = started
        elapsed = max(0.0, now - started)
        ramp_s = max(1e-3, as_float(self.cfg.get("post_corner_bias_ramp_s"), 2.0))
        start_fraction = clamp(as_float(self.cfg.get("post_corner_bias_start_fraction"), 0.50), 0.0, 1.0)
        fraction = start_fraction * (1.0 - smoothstep(0.0, ramp_s, elapsed))
        if fraction <= 1e-4:
            return 0.0, 0.0

        direction_sign = -1.0 if as_float(self.cfg.get("post_corner_bias_direction_sign"), 1.0) < 0.0 else 1.0
        if self.post_corner_bias_sign == 0.0 and lane.turn_sign != 0.0:
            self.post_corner_bias_sign = lane.turn_sign
        turn = self.post_corner_bias_sign
        side = turn * direction_sign
        if side == 0.0:
            return 0.0, 0.0

        if side > 0.0:
            full_offset = lane.right_solid_offset_norm
        else:
            full_offset = lane.left_solid_offset_norm
        if full_offset is None or abs(full_offset) < 1e-4:
            fallback = abs(as_float(self.cfg.get("post_corner_bias_fallback_solid_offset_norm"), 0.60))
            full_offset = fallback if side > 0.0 else -fallback

        max_bias = abs(as_float(self.cfg.get("post_corner_bias_max_norm"), 0.36))
        target_bias = clamp(float(full_offset) * fraction, -max_bias, max_bias)
        return float(target_bias), float(fraction)

    def _alpha_reference(self, lane: LaneSignal) -> float:
        if self.mode not in {"CORNER_APPROACH", "CORNERING"}:
            return 0.0
        sign = self.corner_sign or lane.turn_sign
        if sign == 0.0:
            return 0.0
        ref = abs(as_float(self.cfg.get("corner_alpha_ref_deg"), 22.0))
        scale = clamp(lane.corner_confidence, 0.0, 1.0)
        return float(sign * ref * scale)

    def _lane_scale(self, alpha_abs: float) -> float:
        deadband = as_float(self.cfg.get("straight_deadband_abs_angle_deg"), 5.0)
        caution = as_float(self.cfg.get("straight_caution_abs_angle_deg"), 15.0)
        if self.mode in {"CORNER_APPROACH", "CORNERING"}:
            return as_float(self.cfg.get("corner_lane_scale"), 0.75)
        if self.mode == "RECOVERY":
            return as_float(self.cfg.get("recovery_lane_scale"), 0.35)
        if self.mode == "POST_CORNER_BIAS":
            return as_float(self.cfg.get("post_corner_bias_lane_scale"), 0.85)
        if self.mode == "POST_CORNER_SETTLE":
            return as_float(self.cfg.get("post_corner_lane_scale"), 0.55)
        t = smoothstep(deadband, caution, alpha_abs)
        return lerp(1.0, as_float(self.cfg.get("straight_angle_lane_scale"), 0.55), t)

    def _trailer_scale(self, alpha_abs: float) -> float:
        deadband = as_float(self.cfg.get("straight_deadband_abs_angle_deg"), 5.0)
        caution = as_float(self.cfg.get("straight_caution_abs_angle_deg"), 15.0)
        if self.mode in {"CORNER_APPROACH", "CORNERING"}:
            return as_float(self.cfg.get("corner_trailer_scale"), 0.45)
        if self.mode == "RECOVERY":
            return as_float(self.cfg.get("recovery_trailer_scale"), 1.00)
        if self.mode == "POST_CORNER_BIAS":
            return as_float(self.cfg.get("post_corner_bias_trailer_scale"), 0.10)
        if self.mode == "POST_CORNER_SETTLE":
            return as_float(self.cfg.get("post_corner_trailer_scale"), 0.55)
        t = smoothstep(deadband, caution, alpha_abs)
        return lerp(as_float(self.cfg.get("straight_min_trailer_scale"), 0.15), 1.0, t)

    def _speed_scale(self, alpha_abs: float, corner_conf: float) -> float:
        if self.mode == "RECOVERY":
            return as_float(self.cfg.get("recovery_speed_scale"), 0.35)
        if self.mode in {"CORNER_APPROACH", "CORNERING"}:
            return lerp(1.0, as_float(self.cfg.get("corner_speed_scale"), 0.52), clamp(corner_conf, 0.0, 1.0))
        if self.mode == "POST_CORNER_BIAS":
            return as_float(self.cfg.get("post_corner_bias_speed_scale"), 0.95)
        if self.mode == "POST_CORNER_SETTLE":
            return as_float(self.cfg.get("post_corner_speed_scale"), 0.85)
        caution = as_float(self.cfg.get("straight_caution_abs_angle_deg"), 15.0)
        recovery = as_float(self.cfg.get("recovery_start_abs_angle_deg"), 48.0)
        t = smoothstep(caution, recovery, alpha_abs)
        return lerp(1.0, as_float(self.cfg.get("angle_slow_speed_scale"), 0.55), t)

    def _curve_scale(self) -> float:
        if self.mode in {"CORNER_APPROACH", "CORNERING"}:
            return 1.0
        if self.mode == "POST_CORNER_BIAS":
            return as_float(self.cfg.get("post_corner_bias_curve_scale"), 0.10)
        if self.mode == "POST_CORNER_SETTLE":
            return as_float(self.cfg.get("post_corner_curve_scale"), 0.15)
        return as_float(self.cfg.get("straight_curve_scale"), 0.35)

    def _alpha_gain(self) -> float:
        if self.mode in {"CORNER_APPROACH", "CORNERING"}:
            return as_float(self.cfg.get("corner_alpha_gain"), 0.006)
        if self.mode == "RECOVERY":
            return as_float(self.cfg.get("recovery_alpha_gain"), 0.018)
        if self.mode == "POST_CORNER_BIAS":
            return as_float(self.cfg.get("post_corner_bias_alpha_gain"), 0.003)
        if self.mode == "POST_CORNER_SETTLE":
            return as_float(self.cfg.get("post_corner_alpha_gain"), 0.006)
        return as_float(self.cfg.get("straight_alpha_gain"), 0.011)

    def _max_steer(self) -> float:
        if self.mode == "POST_CORNER_BIAS":
            return abs(as_float(self.cfg.get("post_corner_bias_max_steer"), 0.75))
        if self.mode == "POST_CORNER_SETTLE":
            return abs(as_float(self.cfg.get("post_corner_max_steer"), 0.55))
        return abs(as_float(self.cfg.get("max_steer"), 0.85))

    def angle_band(self, alpha_abs: float) -> str:
        if alpha_abs < as_float(self.cfg.get("straight_deadband_abs_angle_deg"), 5.0):
            return "straight_deadband"
        if alpha_abs < as_float(self.cfg.get("straight_caution_abs_angle_deg"), 15.0):
            return "straight_caution"
        if alpha_abs < as_float(self.cfg.get("corner_allow_abs_angle_deg"), 45.0):
            return "corner_allowed"
        if alpha_abs < as_float(self.cfg.get("recovery_start_abs_angle_deg"), 48.0):
            return "corner_high"
        if alpha_abs < as_float(self.cfg.get("hard_stop_abs_angle_deg"), 62.0):
            return "recovery"
        return "hard_stop"

    def _make_debug(
        self,
        lane: LaneSignal,
        angle: TrailerAngleState,
        alpha_ref: float,
        lookahead: float,
        lane_scale: float,
        trailer_scale: float,
        speed_scale: float,
        mode: str,
        u_lane: float,
        u_curve: float,
        angle_band: str,
        u_trailer: float = 0.0,
        u_total: float = 0.0,
        target_bias_norm: float = 0.0,
        target_bias_fraction: float = 0.0,
    ) -> RouteControlDebug:
        return RouteControlDebug(
            mode=self.mode if mode == angle_band else str(mode),
            lane_state=str(lane.state),
            lane_row_count=int(lane.row_count),
            alpha_deg=angle.angle_deg,
            alpha_rate_deg_s=float(angle.angle_rate_deg_s),
            alpha_ref_deg=float(alpha_ref),
            angle_band=str(angle_band),
            lookahead_y_ratio=float(lookahead),
            lane_scale=float(lane_scale),
            trailer_scale=float(trailer_scale),
            speed_scale=float(speed_scale),
            u_lane=float(u_lane),
            u_curve=float(u_curve),
            u_trailer=float(u_trailer),
            u_total=float(u_total),
            curvature_proxy=float(lane.curvature_proxy),
            corner_confidence=float(lane.corner_confidence),
            target_bias_norm=float(target_bias_norm),
            target_bias_fraction=float(target_bias_fraction),
        )


def lane_side_offsets_from_rows(rows: Sequence[Any], bev_width: int) -> Tuple[Optional[float], Optional[float]]:
    half_width = max(1.0, float(bev_width) * 0.5)
    left_offsets: List[float] = []
    right_offsets: List[float] = []
    width_offsets: List[float] = []
    for row in rows:
        dashed_x = getattr(row, "dashed_x", None)
        lane_width_px = getattr(row, "lane_width_px", None)
        if lane_width_px is not None:
            width_offsets.append(abs(float(lane_width_px)) / half_width)
        if dashed_x is None:
            continue
        dashed = float(dashed_x)
        left = getattr(row, "solid_left_x", None)
        right = getattr(row, "solid_right_x", None)
        if left is not None:
            left_offsets.append((float(left) - dashed) / half_width)
        if right is not None:
            right_offsets.append((float(right) - dashed) / half_width)

    width_norm = float(np.median(width_offsets)) if width_offsets else None
    left_norm = float(np.median(left_offsets)) if left_offsets else (-width_norm if width_norm is not None else None)
    right_norm = float(np.median(right_offsets)) if right_offsets else (width_norm if width_norm is not None else None)
    return left_norm, right_norm


def lane_signal_from_results(results: Dict[str, Any], config: Dict[str, Any], bev_width: int) -> LaneSignal:
    weighted: List[Tuple[float, Any]] = []
    lane_cfg = config.get("lane", {}) or {}
    min_drive_conf = as_float(lane_cfg.get("min_drive_confidence"), 0.08)
    row_target = max(1.0, as_float(lane_cfg.get("row_samples"), 15.0))
    for result in results.values():
        estimate = result.estimate
        if not estimate.valid or estimate.confidence < min_drive_conf:
            continue
        row_ratio = min(1.0, len(estimate.row_estimates) / row_target)
        weight = max(0.0, float(estimate.confidence)) * max(0.25, row_ratio)
        if weight > 0.0:
            weighted.append((weight, estimate))

    if not weighted:
        if results:
            best = max(results.values(), key=lambda item: item.estimate.confidence).estimate
            half_width = max(1.0, bev_width * 0.5)
            lateral_error = float(getattr(best, "lateral_error_px", 0.0) or 0.0)
            heading_error = float(getattr(best, "heading_error_rad", 0.0) or 0.0)
            rows = list(getattr(best, "row_estimates", []) or [])
            curvature_proxy, turn_sign, corner_conf = corner_features_from_rows(rows, config, bev_width)
            left_offset, right_offset = lane_side_offsets_from_rows(rows, bev_width)
            return LaneSignal(
                False,
                float(best.confidence),
                float(clamp(lateral_error / half_width, -1.5, 1.5)),
                float(clamp(heading_error, -1.5, 1.5)),
                max(0.0, float(getattr(best, "speed", 0.0) or 0.0)),
                len(rows),
                float(curvature_proxy),
                float(turn_sign),
                float(corner_conf),
                best.state,
                best.reason,
                left_offset,
                right_offset,
            )
        return LaneSignal(False, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, "NO_CAMERA_FRAME", "no_camera_frame")

    total = sum(w for w, _ in weighted)
    half_width = max(1.0, bev_width * 0.5)
    e_y = 0.0
    e_psi = 0.0
    conf = 0.0
    speed = min(max(0.0, float(est.speed)) for _, est in weighted)
    row_count = 0
    left_sum = 0.0
    left_weight = 0.0
    right_sum = 0.0
    right_weight = 0.0
    for weight, estimate in weighted:
        e_y += weight * ((float(estimate.lateral_error_px or 0.0)) / half_width)
        e_psi += weight * float(estimate.heading_error_rad or 0.0)
        conf += weight * float(estimate.confidence)
        row_count += len(estimate.row_estimates)
        left_offset, right_offset = lane_side_offsets_from_rows(estimate.row_estimates, bev_width)
        if left_offset is not None:
            left_sum += weight * left_offset
            left_weight += weight
        if right_offset is not None:
            right_sum += weight * right_offset
            right_weight += weight
    e_y /= max(1e-6, total)
    e_psi /= max(1e-6, total)
    conf /= max(1e-6, total)
    left_offset = left_sum / left_weight if left_weight > 1e-6 else None
    right_offset = right_sum / right_weight if right_weight > 1e-6 else None

    best_estimate = max((est for _, est in weighted), key=lambda est: (est.confidence, len(est.row_estimates)))
    curvature_proxy, turn_sign, corner_conf = corner_features_from_rows(best_estimate.row_estimates, config, bev_width)
    states = ",".join(str(est.state) for _, est in weighted)
    reason = f"lane_signal {states}"
    return LaneSignal(
        True,
        float(conf),
        float(clamp(e_y, -1.5, 1.5)),
        float(clamp(e_psi, -1.5, 1.5)),
        float(speed),
        int(row_count),
        float(curvature_proxy),
        float(turn_sign),
        float(corner_conf),
        f"LANE_SIGNAL:{states}",
        reason,
        left_offset,
        right_offset,
    )


def lane_signal_from_estimate(
    estimate: Any,
    config: Dict[str, Any],
    bev_width: int,
    source: str = "FUSED_BEV",
) -> LaneSignal:
    lane_cfg = config.get("lane", {}) or {}
    min_drive_conf = as_float(lane_cfg.get("min_drive_confidence"), 0.08)
    confidence = float(getattr(estimate, "confidence", 0.0))
    rows = list(getattr(estimate, "row_estimates", []) or [])
    state = str(getattr(estimate, "state", "UNKNOWN"))
    reason = str(getattr(estimate, "reason", ""))
    speed = max(0.0, float(getattr(estimate, "speed", 0.0)))
    valid = bool(getattr(estimate, "valid", False)) and confidence >= min_drive_conf
    half_width = max(1.0, bev_width * 0.5)
    lateral_error = float(getattr(estimate, "lateral_error_px", 0.0) or 0.0)
    heading_error = float(getattr(estimate, "heading_error_rad", 0.0) or 0.0)
    e_y = float(clamp(lateral_error / half_width, -1.5, 1.5))
    e_psi = float(clamp(heading_error, -1.5, 1.5))
    curvature_proxy, turn_sign, corner_conf = corner_features_from_rows(rows, config, bev_width)
    left_offset, right_offset = lane_side_offsets_from_rows(rows, bev_width)
    if not valid:
        return LaneSignal(
            False,
            confidence,
            e_y,
            e_psi,
            speed,
            len(rows),
            float(curvature_proxy),
            float(turn_sign),
            float(corner_conf),
            f"{source}:{state}",
            reason or f"{source.lower()} invalid",
            left_offset,
            right_offset,
        )

    return LaneSignal(
        True,
        confidence,
        e_y,
        e_psi,
        speed,
        len(rows),
        float(curvature_proxy),
        float(turn_sign),
        float(corner_conf),
        f"{source}:{state}",
        reason or f"{source.lower()} lane_signal",
        left_offset,
        right_offset,
    )


def corner_features_from_rows(rows: Sequence[Any], config: Dict[str, Any], bev_width: int) -> Tuple[float, float, float]:
    cfg = config.get("route_controller", {}) or {}
    usable = [row for row in rows if getattr(row, "center_x", None) is not None]
    if len(usable) < max(4, as_int(cfg.get("corner_min_rows"), 5)):
        return 0.0, 0.0, 0.0
    ordered = sorted(usable, key=lambda row: float(row.y))
    if str(cfg.get("corner_detection_mode", "quadratic")).strip().lower() not in {"quadratic", "curve", "poly2"}:
        return corner_features_from_shift(ordered, config, bev_width)

    ys = np.array([float(row.y) for row in ordered], dtype=np.float64)
    xs = np.array([float(row.center_x) / max(1.0, float(bev_width)) for row in ordered], dtype=np.float64)
    y_span = float(np.max(ys) - np.min(ys))
    if y_span <= 1e-6 or not np.all(np.isfinite(xs)):
        return 0.0, 0.0, 0.0
    t = 2.0 * (ys - float(np.min(ys))) / y_span - 1.0
    weights = np.array([max(0.05, float(getattr(row, "confidence", 0.5) or 0.5)) for row in ordered], dtype=np.float64)
    try:
        a, b, _c = np.polyfit(t, xs, 2, w=weights)
    except Exception:
        return corner_features_from_shift(ordered, config, bev_width)

    # x(t)=a*t^2+b*t+c.  b is mostly diagonal heading, while 4a is
    # the derivative change across the visible row span, i.e. curve.
    sign_invert = as_float(cfg.get("curvature_sign"), 1.0)
    curve_norm = clamp(float(4.0 * a * sign_invert), -1.0, 1.0)
    linear_slope = abs(float(b))
    start = max(0.0, as_float(cfg.get("corner_curve_start_norm"), as_float(cfg.get("corner_shift_start_norm"), 0.08)))
    full = max(start + 1e-3, as_float(cfg.get("corner_curve_full_norm"), as_float(cfg.get("corner_shift_full_norm"), 0.28)))
    confidence = clamp((abs(curve_norm) - start) / (full - start), 0.0, 1.0)

    diagonal_start = max(0.0, as_float(cfg.get("corner_diagonal_slope_start_norm"), 0.16))
    ratio_min = max(0.0, as_float(cfg.get("corner_diagonal_curvature_ratio"), 0.35))
    if linear_slope >= diagonal_start:
        curve_to_line = abs(curve_norm) / max(1e-6, linear_slope)
        if curve_to_line < ratio_min:
            confidence *= clamp(as_float(cfg.get("corner_diagonal_conf_scale"), 0.25), 0.0, 1.0)

    single_rows = sum(1 for row in ordered if getattr(row, "method", "") == "single_solid_recovery")
    dashed_or_pair_rows = sum(
        1
        for row in ordered
        if getattr(row, "dashed_x", None) is not None or getattr(row, "method", "") == "solid_pair_midpoint"
    )
    if single_rows > 0 and dashed_or_pair_rows == 0:
        confidence *= clamp(as_float(cfg.get("corner_single_solid_only_conf_scale"), 0.60), 0.0, 1.0)

    row_ratio = clamp(len(ordered) / max(1.0, as_float((config.get("lane", {}) or {}).get("row_samples"), 15.0)), 0.0, 1.0)
    confidence *= 0.50 + 0.50 * row_ratio
    turn_sign = 0.0 if abs(curve_norm) < start else float(np.sign(curve_norm))
    gain = as_float(cfg.get("curvature_proxy_gain"), 2.2)
    return clamp(curve_norm * gain, -1.0, 1.0), turn_sign, float(confidence)


def corner_features_from_shift(ordered: Sequence[Any], config: Dict[str, Any], bev_width: int) -> Tuple[float, float, float]:
    cfg = config.get("route_controller", {}) or {}
    third = max(1, len(ordered) // 3)
    far = ordered[:third]
    near = ordered[-third:]
    far_x = float(np.median([float(row.center_x) for row in far]))
    near_x = float(np.median([float(row.center_x) for row in near]))
    shift_norm = (far_x - near_x) / max(1.0, float(bev_width))
    sign_invert = as_float(cfg.get("curvature_sign"), 1.0)
    curvature = clamp(shift_norm * sign_invert, -1.0, 1.0)
    start = max(0.0, as_float(cfg.get("corner_shift_start_norm"), 0.10))
    full = max(start + 1e-3, as_float(cfg.get("corner_shift_full_norm"), 0.28))
    confidence = clamp((abs(curvature) - start) / (full - start), 0.0, 1.0)
    row_ratio = clamp(len(ordered) / max(1.0, as_float((config.get("lane", {}) or {}).get("row_samples"), 15.0)), 0.0, 1.0)
    confidence *= 0.50 + 0.50 * row_ratio
    turn_sign = 0.0 if abs(curvature) < start else float(np.sign(curvature))
    gain = as_float(cfg.get("curvature_proxy_gain"), 2.2)
    return clamp(curvature * gain, -1.0, 1.0), turn_sign, float(confidence)


def lerp(a: float, b: float, t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return float(a + (b - a) * t)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))
