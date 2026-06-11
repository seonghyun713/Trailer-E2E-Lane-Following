#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SOLID_CLASS = 1
DASHED_CLASS = 2


@dataclass
class BevConfig:
    output_width: int = 640
    output_height: int = 720
    # Points are ordered: bottom-left, bottom-right, top-right, top-left.
    src_points_ratio: Tuple[Tuple[float, float], ...] = (
        (0.08, 0.95),
        (0.92, 0.95),
        (0.64, 0.43),
        (0.36, 0.43),
    )


@dataclass
class LaneFollowerConfig:
    lane_side: str = "right"
    path_mode: str = "dashed_centerline"
    control_mode: str = "lateral_heading"
    nominal_lane_width_px: float = 220.0
    min_lane_width_px: float = 80.0
    max_lane_width_px: float = 420.0
    row_samples: int = 15
    row_y_min_ratio: float = 0.42
    row_y_max_ratio: float = 0.92
    row_band_px: int = 18
    min_component_pixels: int = 8
    lookahead_y_ratio: float = 0.64
    pure_pursuit_gain: float = 0.75
    lateral_gain: float = 0.85
    heading_gain: float = 0.45
    steer_smoothing: float = 0.62
    center_smoothing: float = 0.60
    vehicle_center_x_bias: float = 0.0
    roundabout_route_aware: bool = True
    roundabout_approach_x_bias: float = 0.24
    roundabout_circulate_x_bias: float = -0.24
    roundabout_left_circulate_x_bias: float = -0.24
    roundabout_right_circulate_x_bias: float = 0.24
    roundabout_exit_x_bias: float = 0.24
    roundabout_left_exit_x_bias: float = 0.24
    roundabout_right_exit_x_bias: float = 0.24
    roundabout_route_select_strength: float = 0.75
    roundabout_route_far_weight: float = 1.40
    roundabout_route_center_smoothing: float = 0.30
    lane_width_smoothing: float = 0.80
    min_confidence: float = 0.25
    base_speed: float = 0.28
    min_speed: float = 0.06
    max_speed: float = 0.35


@dataclass
class RowEstimate:
    y: float
    center_x: float
    dashed_x: Optional[float]
    solid_x: Optional[float]
    lane_width_px: Optional[float]
    confidence: float
    method: str
    solid_left_x: Optional[float] = None
    solid_right_x: Optional[float] = None


@dataclass
class LaneEstimate:
    valid: bool
    confidence: float
    state: str
    center_x: Optional[float]
    lookahead_y: float
    lateral_error_px: Optional[float]
    heading_error_rad: Optional[float]
    raw_steer: float
    steer: float
    speed: float
    lane_width_px: Optional[float]
    row_estimates: List[RowEstimate] = field(default_factory=list)
    poly_coefficients: Optional[List[float]] = None
    reason: str = ""
    route_hint: str = "none"


@dataclass
class LaneFollowerState:
    prev_center_x: Optional[float] = None
    prev_steer: float = 0.0
    lane_width_px: Optional[float] = None
    lost_count: int = 0
    prev_route_hint: str = "none"


def source_points(image_width: int, image_height: int, bev: BevConfig) -> np.ndarray:
    return np.array(
        [(x * image_width, y * image_height) for x, y in bev.src_points_ratio],
        dtype=np.float32,
    )


def destination_points(bev: BevConfig) -> np.ndarray:
    w = float(bev.output_width)
    h = float(bev.output_height)
    return np.array([(0.0, h), (w, h), (w, 0.0), (0.0, 0.0)], dtype=np.float32)


def perspective_matrices(image_width: int, image_height: int, bev: BevConfig) -> Tuple[np.ndarray, np.ndarray]:
    src = source_points(image_width, image_height, bev)
    dst = destination_points(bev)
    matrix = cv2.getPerspectiveTransform(src, dst)
    inv_matrix = cv2.getPerspectiveTransform(dst, src)
    return matrix, inv_matrix


def warp_to_bev(image_or_mask: np.ndarray, bev: BevConfig, matrix: np.ndarray, is_mask: bool) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpPerspective(
        image_or_mask,
        matrix,
        (bev.output_width, bev.output_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _components_from_strip(class_strip: np.ndarray, min_pixels: int) -> List[Tuple[float, int, int, int]]:
    hist = (class_strip > 0).sum(axis=0)
    active = hist >= max(1, min_pixels // max(1, class_strip.shape[0] // 3))
    components: List[Tuple[float, int, int, int]] = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            weight = int(hist[start : end + 1].sum())
            if weight >= min_pixels:
                xs = np.arange(start, end + 1, dtype=np.float32)
                center = float((xs * hist[start : end + 1]).sum() / max(1, weight))
                components.append((center, start, end, weight))
            start = None
    if start is not None:
        end = len(active) - 1
        weight = int(hist[start : end + 1].sum())
        if weight >= min_pixels:
            xs = np.arange(start, end + 1, dtype=np.float32)
            center = float((xs * hist[start : end + 1]).sum() / max(1, weight))
            components.append((center, start, end, weight))
    return components


def _best_component(components: Sequence[Tuple[float, int, int, int]], target_x: float) -> Optional[Tuple[float, int, int, int]]:
    if not components:
        return None
    return min(components, key=lambda item: abs(item[0] - target_x))


def _nearest_component_on_side(
    components: Sequence[Tuple[float, int, int, int]],
    anchor_x: float,
    side: str,
    min_distance: float,
    max_distance: float,
) -> Optional[Tuple[float, int, int, int]]:
    sign = 1.0 if side == "right" else -1.0
    candidates = []
    for component in components:
        distance = sign * (component[0] - anchor_x)
        if min_distance <= distance <= max_distance:
            candidates.append((distance, component))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _solid_edges_around(
    solid_components: Sequence[Tuple[float, int, int, int]],
    anchor_x: float,
    min_distance: float,
    max_distance: float,
) -> Tuple[Optional[Tuple[float, int, int, int]], Optional[Tuple[float, int, int, int]]]:
    left_candidates = []
    right_candidates = []
    for component in solid_components:
        distance = component[0] - anchor_x
        abs_distance = abs(distance)
        if not (min_distance <= abs_distance <= max_distance):
            continue
        if distance < 0:
            left_candidates.append((abs_distance, component))
        else:
            right_candidates.append((abs_distance, component))
    left = min(left_candidates, key=lambda item: item[0])[1] if left_candidates else None
    right = min(right_candidates, key=lambda item: item[0])[1] if right_candidates else None
    return left, right


def _best_solid_pair(
    solid_components: Sequence[Tuple[float, int, int, int]],
    target_x: float,
    min_half_width: float,
    max_half_width: float,
) -> Tuple[Optional[Tuple[float, int, int, int]], Optional[Tuple[float, int, int, int]]]:
    best_pair = (None, None)
    best_score = float("inf")
    for left in solid_components:
        for right in solid_components:
            if left[0] >= right[0]:
                continue
            center = (left[0] + right[0]) * 0.5
            half_width = (right[0] - left[0]) * 0.5
            if not (min_half_width <= half_width <= max_half_width):
                continue
            score = abs(center - target_x)
            if score < best_score:
                best_score = score
                best_pair = (left, right)
    return best_pair


def _row_estimate(
    bev_mask: np.ndarray,
    y: float,
    image_center_x: float,
    config: LaneFollowerConfig,
    state: LaneFollowerState,
    route_ref_x: Optional[float] = None,
    route_select_strength: float = 0.0,
) -> Optional[RowEstimate]:
    y0 = max(0, int(round(y - config.row_band_px)))
    y1 = min(bev_mask.shape[0], int(round(y + config.row_band_px + 1)))
    if y1 <= y0:
        return None

    strip = bev_mask[y0:y1]
    solid_components = _components_from_strip(strip == SOLID_CLASS, config.min_component_pixels)
    dashed_components = _components_from_strip(strip == DASHED_CLASS, config.min_component_pixels)

    if state.prev_center_x is not None:
        center_ref = state.prev_center_x
    else:
        center_ref = image_center_x
    if route_ref_x is not None:
        strength = float(np.clip(route_select_strength, 0.0, 1.0))
        center_ref = (1.0 - strength) * center_ref + strength * route_ref_x
    if state.lane_width_px is not None:
        lane_width_ref = state.lane_width_px
    else:
        lane_width_ref = config.nominal_lane_width_px

    dashed = _best_component(dashed_components, center_ref)
    if dashed is None:
        dashed = _best_component(dashed_components, image_center_x)

    if dashed is not None:
        left_solid, right_solid = _solid_edges_around(
            solid_components,
            anchor_x=dashed[0],
            min_distance=config.min_lane_width_px,
            max_distance=config.max_lane_width_px,
        )
        solid_support = 0
        measured_widths = []
        for solid in (left_solid, right_solid):
            if solid is None:
                continue
            solid_support += solid[3]
            measured_widths.append(abs(solid[0] - dashed[0]))
        lane_width = float(np.median(measured_widths)) if measured_widths else lane_width_ref
        width_score = 1.0 - min(1.0, abs(lane_width - lane_width_ref) / max(1.0, lane_width_ref))
        boundary_bonus = 0.18 if left_solid is not None and right_solid is not None else 0.08 if measured_widths else 0.0
        support_score = min(1.0, dashed[3] / 220.0)
        solid_support_score = min(1.0, solid_support / 360.0)
        confidence = 0.52 + 0.20 * support_score + 0.10 * solid_support_score + boundary_bonus
        if measured_widths:
            confidence += 0.08 * width_score
        return RowEstimate(
            y=y,
            center_x=dashed[0],
            dashed_x=dashed[0],
            solid_x=right_solid[0] if right_solid is not None else left_solid[0] if left_solid is not None else None,
            lane_width_px=lane_width,
            confidence=float(min(1.0, confidence)),
            method="dashed_centerline",
            solid_left_x=left_solid[0] if left_solid is not None else None,
            solid_right_x=right_solid[0] if right_solid is not None else None,
        )

    left_solid, right_solid = _best_solid_pair(
        solid_components,
        target_x=center_ref,
        min_half_width=config.min_lane_width_px,
        max_half_width=config.max_lane_width_px,
    )
    if left_solid is not None and right_solid is not None:
        center_x = (left_solid[0] + right_solid[0]) * 0.5
        half_width = (right_solid[0] - left_solid[0]) * 0.5
        width_score = 1.0 - min(1.0, abs(half_width - lane_width_ref) / max(1.0, lane_width_ref))
        support_score = min(1.0, (left_solid[3] + right_solid[3]) / 520.0)
        confidence = 0.38 + 0.24 * width_score + 0.18 * support_score
        return RowEstimate(
            y=y,
            center_x=center_x,
            dashed_x=None,
            solid_x=right_solid[0],
            lane_width_px=half_width,
            confidence=float(confidence),
            method="solid_pair_midpoint",
            solid_left_x=left_solid[0],
            solid_right_x=right_solid[0],
        )

    if solid_components:
        solid = _best_component(solid_components, center_ref)
        if solid is not None:
            # Low-confidence recovery: infer the dashed centerline from one
            # visible outer solid line and the last known/nominal half width.
            edge_sign = 1.0 if solid[0] < center_ref else -1.0
            center_x = solid[0] + edge_sign * lane_width_ref
            support_score = min(1.0, solid[3] / 260.0)
            return RowEstimate(
                y=y,
                center_x=center_x,
                dashed_x=None,
                solid_x=solid[0],
                lane_width_px=lane_width_ref,
                confidence=0.18 + 0.18 * support_score,
                method="single_solid_recovery",
                solid_left_x=solid[0] if solid[0] < center_x else None,
                solid_right_x=solid[0] if solid[0] > center_x else None,
            )

    return None


def estimate_lane(
    bev_mask: np.ndarray,
    config: LaneFollowerConfig,
    state: LaneFollowerState,
    route_hint: str = "none",
) -> LaneEstimate:
    height, width = bev_mask.shape[:2]
    camera_center_x = width * 0.5
    vehicle_center_x_bias = float(np.clip(config.vehicle_center_x_bias, -0.5, 0.5))
    vehicle_center_x = camera_center_x + width * vehicle_center_x_bias
    route_hints = {
        "approach",
        "circulate",
        "left_circulate",
        "right_circulate",
        "exit",
        "left_exit",
        "right_exit",
    }
    route_hint = route_hint if route_hint in route_hints and config.roundabout_route_aware else "none"
    route_ref_x: Optional[float] = None
    route_select_strength = 0.0
    if route_hint == "approach":
        route_ref_x = vehicle_center_x + width * config.roundabout_approach_x_bias
        route_select_strength = config.roundabout_route_select_strength
    elif route_hint == "left_circulate":
        route_ref_x = vehicle_center_x + width * config.roundabout_left_circulate_x_bias
        route_select_strength = config.roundabout_route_select_strength
    elif route_hint == "right_circulate":
        route_ref_x = vehicle_center_x + width * config.roundabout_right_circulate_x_bias
        route_select_strength = config.roundabout_route_select_strength
    elif route_hint == "circulate":
        route_ref_x = vehicle_center_x + width * config.roundabout_circulate_x_bias
        route_select_strength = config.roundabout_route_select_strength
    elif route_hint == "left_exit":
        route_ref_x = vehicle_center_x + width * config.roundabout_left_exit_x_bias
        route_select_strength = config.roundabout_route_select_strength
    elif route_hint == "right_exit":
        route_ref_x = vehicle_center_x + width * config.roundabout_right_exit_x_bias
        route_select_strength = config.roundabout_route_select_strength
    elif route_hint == "exit":
        route_ref_x = vehicle_center_x + width * config.roundabout_exit_x_bias
        route_select_strength = config.roundabout_route_select_strength

    y_values = np.linspace(
        height * config.row_y_min_ratio,
        height * config.row_y_max_ratio,
        config.row_samples,
    )
    row_estimates = []
    for y in y_values:
        estimate = _row_estimate(
            bev_mask,
            float(y),
            vehicle_center_x,
            config,
            state,
            route_ref_x=route_ref_x,
            route_select_strength=route_select_strength,
        )
        if estimate is not None:
            if 0 <= estimate.center_x <= width:
                row_estimates.append(estimate)

    if not row_estimates:
        state.lost_count += 1
        steer = state.prev_steer * 0.75
        state.prev_steer = steer
        return LaneEstimate(
            valid=False,
            confidence=0.0,
            state="LANE_LOST",
            center_x=None,
            lookahead_y=height * config.lookahead_y_ratio,
            lateral_error_px=None,
            heading_error_rad=None,
            raw_steer=steer,
            steer=steer,
            speed=0.0,
            lane_width_px=state.lane_width_px,
            row_estimates=[],
            reason="no_row_estimates",
            route_hint=route_hint,
        )

    xs = np.array([row.center_x for row in row_estimates], dtype=np.float64)
    ys = np.array([row.y for row in row_estimates], dtype=np.float64)
    weights = np.array([max(0.05, row.confidence) for row in row_estimates], dtype=np.float64)
    if route_hint != "none":
        far_factor = np.clip((height - ys) / max(1.0, float(height)), 0.0, 1.0)
        weights *= 1.0 + max(0.0, config.roundabout_route_far_weight) * far_factor
    lookahead_y = height * config.lookahead_y_ratio
    # For the first driving baseline, prefer a stable local centerline over a
    # high-order curve. The turn behavior will be handled by state-machine
    # maneuvers; lane following should stay smooth and predictable.
    if len(row_estimates) >= 2 and float(np.ptp(ys)) >= 1.0:
        fit_ys = (ys - lookahead_y) / max(1.0, float(height))
        try:
            norm_coefficients = np.polyfit(fit_ys, xs, 1, w=weights)
        except np.linalg.LinAlgError:
            norm_coefficients = np.array([0.0, float(np.average(xs, weights=weights))], dtype=np.float64)
        derivative = float(norm_coefficients[0] / max(1.0, float(height)))
        coefficients = np.array(
            [
                derivative,
                float(norm_coefficients[1] - derivative * lookahead_y),
            ],
            dtype=np.float64,
        )
    else:
        center = float(np.average(xs, weights=weights))
        derivative = 0.0
        coefficients = np.array([0.0, center], dtype=np.float64)

    target_x = float(np.polyval(coefficients, lookahead_y))
    center_smoothing = config.center_smoothing
    if route_hint != "none":
        center_smoothing = config.roundabout_route_center_smoothing
        if state.prev_route_hint != route_hint:
            center_smoothing = min(center_smoothing, 0.12)
    if state.prev_center_x is not None:
        target_x = center_smoothing * state.prev_center_x + (1.0 - center_smoothing) * target_x

    lateral_error = target_x - vehicle_center_x
    vehicle_y = float(height)
    forward_distance = max(1.0, vehicle_y - lookahead_y)
    heading_error = float(np.arctan(derivative))
    lateral_norm = lateral_error / max(1.0, camera_center_x)
    if config.control_mode == "pure_pursuit":
        lookahead_distance_sq = lateral_error * lateral_error + forward_distance * forward_distance
        pursuit_term = (2.0 * lateral_error * forward_distance) / max(1.0, lookahead_distance_sq)
        raw_steer = config.pure_pursuit_gain * pursuit_term
    else:
        raw_steer = config.lateral_gain * lateral_norm + config.heading_gain * heading_error
    raw_steer = float(np.clip(raw_steer, -1.0, 1.0))

    if state.prev_center_x is None:
        steer = raw_steer
    else:
        steer = config.steer_smoothing * state.prev_steer + (1.0 - config.steer_smoothing) * raw_steer
    steer = float(np.clip(steer, -1.0, 1.0))

    row_coverage = len(row_estimates) / max(1, config.row_samples)
    mean_row_conf = float(np.mean([row.confidence for row in row_estimates]))
    confidence = float(np.clip(0.55 * mean_row_conf + 0.45 * row_coverage, 0.0, 1.0))
    dashed_row_count = sum(1 for row in row_estimates if row.dashed_x is not None)
    dashed_ratio = dashed_row_count / max(1, len(row_estimates))

    measured_widths = [
        row.lane_width_px
        for row in row_estimates
        if row.lane_width_px is not None and row.method in {"dashed_centerline", "solid_pair_midpoint"}
    ]
    if measured_widths:
        measured_width = float(np.median(measured_widths))
        if state.lane_width_px is None:
            state.lane_width_px = measured_width
        else:
            state.lane_width_px = (
                config.lane_width_smoothing * state.lane_width_px
                + (1.0 - config.lane_width_smoothing) * measured_width
            )

    state.prev_center_x = target_x
    state.prev_steer = steer
    state.prev_route_hint = route_hint
    if confidence < config.min_confidence:
        state.lost_count += 1
        drive_state = "LOW_CONFIDENCE"
    elif dashed_ratio < 0.20:
        state.lost_count = 0
        drive_state = "DASHED_RECOVERY"
    elif dashed_ratio < 0.45:
        state.lost_count = 0
        drive_state = "DASHED_PARTIAL"
    else:
        state.lost_count = 0
        drive_state = "LANE_FOLLOW"

    curve_scale = float(np.clip(1.0 - 0.45 * abs(steer), 0.35, 1.0))
    confidence_scale = float(np.clip((confidence - 0.15) / 0.75, 0.0, 1.0))
    speed = config.base_speed * curve_scale * confidence_scale
    if drive_state == "LANE_FOLLOW":
        speed = max(config.min_speed, speed)
    elif drive_state == "DASHED_PARTIAL":
        speed = min(max(config.min_speed, speed), config.min_speed * 1.8)
    elif drive_state == "DASHED_RECOVERY":
        speed = min(config.min_speed, speed)
    else:
        speed = min(config.min_speed, speed)
    speed = float(np.clip(speed, 0.0, config.max_speed))

    return LaneEstimate(
        valid=drive_state in {"LANE_FOLLOW", "DASHED_PARTIAL"},
        confidence=confidence,
        state=drive_state,
        center_x=target_x,
        lookahead_y=lookahead_y,
        lateral_error_px=lateral_error,
        heading_error_rad=heading_error,
        raw_steer=raw_steer,
        steer=steer,
        speed=speed,
        lane_width_px=state.lane_width_px,
        row_estimates=row_estimates,
        poly_coefficients=[float(value) for value in np.atleast_1d(coefficients)],
        reason=f"{config.control_mode} route={route_hint} dashed_rows={dashed_row_count}/{len(row_estimates)}",
        route_hint=route_hint,
    )


def colorize_segmentation(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    color[mask == SOLID_CLASS] = (42, 211, 255)
    color[mask == DASHED_CLASS] = (0, 138, 255)
    return color


def overlay_segmentation(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    color = colorize_segmentation(mask).astype(np.float32)
    image = image_bgr.astype(np.float32)
    active = (mask == SOLID_CLASS) | (mask == DASHED_CLASS)
    out = image.copy()
    out[active] = image[active] * (1.0 - alpha) + color[active] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_bev_debug(bev_mask: np.ndarray, estimate: LaneEstimate, bev: BevConfig) -> np.ndarray:
    canvas = colorize_segmentation(bev_mask)
    canvas = cv2.addWeighted(canvas, 1.0, np.full_like(canvas, 24), 0.25, 0.0)
    h, w = bev_mask.shape[:2]
    cv2.line(canvas, (w // 2, 0), (w // 2, h - 1), (220, 220, 220), 1, cv2.LINE_AA)

    for row in estimate.row_estimates:
        y = int(round(row.y))
        x = int(round(row.center_x))
        cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if row.dashed_x is not None:
            cv2.circle(canvas, (int(round(row.dashed_x)), y), 3, (0, 138, 255), -1, cv2.LINE_AA)
        for solid_x in (row.solid_left_x, row.solid_right_x):
            if solid_x is not None:
                cv2.circle(canvas, (int(round(solid_x)), y), 3, (42, 211, 255), -1, cv2.LINE_AA)
        if row.solid_left_x is None and row.solid_right_x is None and row.solid_x is not None:
            cv2.circle(canvas, (int(round(row.solid_x)), y), 3, (42, 211, 255), -1, cv2.LINE_AA)

    if estimate.poly_coefficients is not None:
        ys = np.linspace(h * 0.35, h * 0.95, 60)
        xs = np.polyval(np.array(estimate.poly_coefficients, dtype=np.float64), ys)
        points = []
        for x, y in zip(xs, ys):
            if 0 <= x < w:
                points.append((int(round(x)), int(round(y))))
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (255, 255, 255), 3, cv2.LINE_AA)

    if estimate.center_x is not None:
        target = (int(round(estimate.center_x)), int(round(estimate.lookahead_y)))
        cv2.circle(canvas, target, 8, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (w // 2, h - 1), target, (255, 0, 255), 2, cv2.LINE_AA)

    text = f"{estimate.state} route={estimate.route_hint} conf={estimate.confidence:.2f} steer={estimate.steer:+.2f} speed={estimate.speed:.2f}"
    cv2.rectangle(canvas, (8, 8), (w - 8, 42), (0, 0, 0), -1)
    cv2.putText(canvas, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_original_debug(image_bgr: np.ndarray, mask: np.ndarray, bev: BevConfig) -> np.ndarray:
    out = overlay_segmentation(image_bgr, mask)
    h, w = image_bgr.shape[:2]
    pts = source_points(w, h, bev).astype(np.int32)
    cv2.polylines(out, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)
    labels = ["BL", "BR", "TR", "TL"]
    for label, point in zip(labels, pts):
        cv2.circle(out, tuple(point), 5, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, label, tuple(point + np.array([7, -7])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    return out


def make_debug_panel(
    image_bgr: np.ndarray,
    pred_mask: np.ndarray,
    bev_mask: np.ndarray,
    estimate: LaneEstimate,
    bev: BevConfig,
) -> np.ndarray:
    original_debug = draw_original_debug(image_bgr, pred_mask, bev)
    bev_debug = draw_bev_debug(bev_mask, estimate, bev)
    target_h = 360
    orig_w = int(round(original_debug.shape[1] * target_h / original_debug.shape[0]))
    original_resized = cv2.resize(original_debug, (orig_w, target_h), interpolation=cv2.INTER_AREA)
    bev_resized = cv2.resize(bev_debug, (320, target_h), interpolation=cv2.INTER_AREA)
    panel = np.zeros((target_h, orig_w + 320, 3), dtype=np.uint8)
    panel[:, :orig_w] = original_resized
    panel[:, orig_w:] = bev_resized
    return panel


def estimate_to_dict(estimate: LaneEstimate) -> Dict[str, object]:
    return {
        "valid": estimate.valid,
        "confidence": estimate.confidence,
        "state": estimate.state,
        "center_x": estimate.center_x,
        "lookahead_y": estimate.lookahead_y,
        "lateral_error_px": estimate.lateral_error_px,
        "heading_error_rad": estimate.heading_error_rad,
        "raw_steer": estimate.raw_steer,
        "steer": estimate.steer,
        "speed": estimate.speed,
        "lane_width_px": estimate.lane_width_px,
        "reason": estimate.reason,
        "route_hint": estimate.route_hint,
        "row_count": len(estimate.row_estimates),
        "dashed_row_count": sum(1 for row in estimate.row_estimates if row.dashed_x is not None),
    }
