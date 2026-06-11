#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


FEATURE_COUNT = 11
DEFAULT_FEATURE_INDICES = [0, 1, 2, 6, 7]


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def feature_indices(policy_cfg: Dict[str, Any]) -> List[int]:
    raw = policy_cfg.get("linear_feature_indices", DEFAULT_FEATURE_INDICES)
    if isinstance(raw, str):
        parts: Iterable[Any] = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, Sequence):
        parts = raw
    else:
        parts = DEFAULT_FEATURE_INDICES
    out: List[int] = []
    for part in parts:
        idx = as_int(part, -1)
        if 0 <= idx < FEATURE_COUNT and idx not in out:
            out.append(idx)
    return out or list(DEFAULT_FEATURE_INDICES)


def csv_files(log_root: Path) -> List[Path]:
    if log_root.is_file():
        return [log_root]
    return sorted(log_root.glob("**/dotted_lane_following_log.csv"), key=lambda p: p.stat().st_mtime)


def has_feature_columns(fieldnames: Optional[Sequence[str]]) -> bool:
    fields = set(fieldnames or [])
    return all(f"policy_feature_{idx}" in fields for idx in range(FEATURE_COUNT))


def row_features(row: Dict[str, str]) -> Optional[np.ndarray]:
    vals = [as_float(row.get(f"policy_feature_{idx}"), math.nan) for idx in range(FEATURE_COUNT)]
    if any(not math.isfinite(v) for v in vals):
        return None
    return np.asarray(vals, dtype=np.float64)


def collect_transitions(
    files: Sequence[Path],
    indices: Sequence[int],
    max_dt_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    episode_starts: List[np.ndarray] = []
    missing_feature_files = 0

    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not has_feature_columns(reader.fieldnames):
                missing_feature_files += 1
                continue
            prev: Optional[Tuple[int, float, np.ndarray, np.ndarray]] = None
            seen_episode_starts: set[int] = set()
            for row in reader:
                if row.get("active") != "1" or row.get("policy_learning") != "1":
                    prev = None
                    continue
                features = row_features(row)
                if features is None:
                    prev = None
                    continue
                ep = as_int(row.get("policy_episode"), -1)
                ts = as_float(row.get("timestamp_monotonic"), math.nan)
                if not math.isfinite(ts) or ep < 0:
                    prev = None
                    continue
                steer = as_float(row.get("policy_action_steer"), math.nan)
                speed = as_float(row.get("policy_action_speed"), math.nan)
                if not math.isfinite(steer) or not math.isfinite(speed):
                    prev = None
                    continue
                z = features[list(indices)]
                action = np.asarray([steer, speed], dtype=np.float64)
                if ep not in seen_episode_starts:
                    episode_starts.append(z.copy())
                    seen_episode_starts.add(ep)
                if prev is not None:
                    prev_ep, prev_ts, prev_z, prev_action = prev
                    if ep == prev_ep and 0.0 < ts - prev_ts <= max_dt_s:
                        states.append(prev_z)
                        actions.append(prev_action)
                        next_states.append(z)
                prev = (ep, ts, z, action)

    if missing_feature_files:
        print(f"[temporal] skipped {missing_feature_files} old log file(s) without policy_feature columns")

    if not states:
        dim = len(indices)
        return (
            np.empty((0, dim), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, dim), dtype=np.float64),
            episode_starts,
        )
    return np.stack(states), np.stack(actions), np.stack(next_states), episode_starts


def fit_linear_dynamics(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, ridge: float) -> np.ndarray:
    ones = np.ones((states.shape[0], 1), dtype=np.float64)
    x = np.concatenate([states, actions, ones], axis=1)
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ next_states)


def load_initial_theta(path: Path, expected_size: int) -> np.ndarray:
    if not path.exists():
        return np.zeros(expected_size, dtype=np.float64)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        theta = np.asarray(data.get("theta", []), dtype=np.float64)
        arch = str(data.get("architecture", "")).strip().lower()
        if arch in {"linear", "linear_feedback"} and theta.shape == (expected_size,):
            print(f"[temporal] loaded initial policy {path}")
            return theta
    except Exception as exc:
        print(f"[temporal] ignored existing weights {path}: {exc}")
    return np.zeros(expected_size, dtype=np.float64)


def rollout_loss(
    theta: np.ndarray,
    starts: np.ndarray,
    dyn_w: np.ndarray,
    horizon: int,
    steer_limit: float,
    fixed_speed: float,
    min_speed: float,
    max_speed: float,
    learn_speed: bool,
    heading_weight: float,
    action_weight: float,
    motion_loss_weight: float,
    motion_target_speed: float,
    zero_bias: bool,
) -> float:
    return float(
        rollout_losses_batch(
            theta[None, :],
            starts,
            dyn_w,
            horizon,
            steer_limit,
            fixed_speed,
            min_speed,
            max_speed,
            learn_speed,
            heading_weight,
            action_weight,
            motion_loss_weight,
            motion_target_speed,
            zero_bias,
        )[0]
    )


def rollout_losses_batch(
    candidates: np.ndarray,
    starts: np.ndarray,
    dyn_w: np.ndarray,
    horizon: int,
    steer_limit: float,
    fixed_speed: float,
    min_speed: float,
    max_speed: float,
    learn_speed: bool,
    heading_weight: float,
    action_weight: float,
    motion_loss_weight: float,
    motion_target_speed: float,
    zero_bias: bool,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float64)
    state_dim = starts.shape[1]
    z = np.broadcast_to(starts[None, :, :], (candidates.shape[0], starts.shape[0], starts.shape[1])).copy()
    steer_weights = candidates[:, :state_dim]
    steer_bias = np.zeros(candidates.shape[0], dtype=np.float64) if zero_bias else candidates[:, state_dim]
    if learn_speed:
        speed_offset = state_dim + 1
        speed_weights = candidates[:, speed_offset:speed_offset + state_dim]
        speed_bias = candidates[:, speed_offset + state_dim]
    else:
        speed_weights = None
        speed_bias = None
    ones = np.ones((candidates.shape[0], starts.shape[0], 1), dtype=np.float64)
    total = np.zeros(candidates.shape[0], dtype=np.float64)
    prev_steer = np.zeros((candidates.shape[0], starts.shape[0]), dtype=np.float64)
    prev_speed = np.full((candidates.shape[0], starts.shape[0]), fixed_speed, dtype=np.float64)
    for _ in range(horizon):
        raw = np.einsum("csd,cd->cs", z, steer_weights) + steer_bias[:, None]
        steer = steer_limit * np.tanh(raw)
        if learn_speed and speed_weights is not None and speed_bias is not None:
            speed_raw = np.einsum("csd,cd->cs", z, speed_weights) + speed_bias[:, None]
            speed01 = 0.5 * (np.tanh(speed_raw) + 1.0)
            speed = min_speed + np.clip(speed01, 0.0, 1.0) * (max_speed - min_speed)
        else:
            speed = np.full_like(steer, fixed_speed)
        action_penalty = action_weight * (np.abs(steer - prev_steer) + 0.25 * np.abs(speed - prev_speed))
        if motion_loss_weight > 0.0 and motion_target_speed > 1e-6:
            motion_loss = motion_loss_weight * np.maximum(0.0, motion_target_speed - speed) / motion_target_speed
        else:
            motion_loss = 0.0
        heading_loss = np.abs(z[:, :, 1]) if z.shape[2] > 1 and heading_weight > 0.0 else 0.0
        step_loss = np.abs(z[:, :, 0]) + heading_weight * heading_loss + motion_loss + action_penalty
        total += np.mean(step_loss, axis=1)
        prev_steer = steer
        prev_speed = speed
        x = np.concatenate([z, steer[:, :, None], speed[:, :, None], ones], axis=2)
        z = np.clip(np.einsum("csk,kd->csd", x, dyn_w), -2.0, 2.0)
    return total / max(1, horizon)


def rollout_loss_slow_reference(
    theta: np.ndarray,
    starts: np.ndarray,
    dyn_w: np.ndarray,
    horizon: int,
    steer_limit: float,
    fixed_speed: float,
    min_speed: float,
    max_speed: float,
    learn_speed: bool,
    heading_weight: float,
    action_weight: float,
    motion_loss_weight: float,
    motion_target_speed: float,
    zero_bias: bool,
) -> float:
    z = starts.copy()
    state_dim = z.shape[1]
    total = 0.0
    prev_steer = np.zeros(z.shape[0], dtype=np.float64)
    prev_speed = np.full(z.shape[0], fixed_speed, dtype=np.float64)
    steer_bias = 0.0 if zero_bias else float(theta[state_dim])
    for _ in range(horizon):
        raw = z @ theta[:state_dim] + steer_bias
        steer = steer_limit * np.tanh(raw)
        if learn_speed:
            speed_offset = state_dim + 1
            speed_raw = z @ theta[speed_offset:speed_offset + state_dim] + theta[speed_offset + state_dim]
            speed01 = 0.5 * (np.tanh(speed_raw) + 1.0)
            speed = min_speed + np.clip(speed01, 0.0, 1.0) * (max_speed - min_speed)
        else:
            speed = np.full_like(steer, fixed_speed)
        action_penalty = action_weight * (np.abs(steer - prev_steer) + 0.25 * np.abs(speed - prev_speed))
        if motion_loss_weight > 0.0 and motion_target_speed > 1e-6:
            motion_loss = motion_loss_weight * np.maximum(0.0, motion_target_speed - speed) / motion_target_speed
        else:
            motion_loss = 0.0
        heading_loss = np.abs(z[:, 1]) if z.shape[1] > 1 and heading_weight > 0.0 else 0.0
        step_loss = np.abs(z[:, 0]) + heading_weight * heading_loss + motion_loss + action_penalty
        total += float(np.mean(step_loss))
        prev_steer = steer
        prev_speed = speed
        x = np.concatenate([z, steer[:, None], speed[:, None], np.ones((z.shape[0], 1))], axis=1)
        z = np.clip(x @ dyn_w, -2.0, 2.0)
    return total / max(1, horizon)


def optimize_policy(
    initial_theta: np.ndarray,
    starts: np.ndarray,
    dyn_w: np.ndarray,
    steer_limit: float,
    fixed_speed: float,
    min_speed: float,
    max_speed: float,
    learn_speed: bool,
    heading_weight: float,
    iterations: int,
    population: int,
    sigma: float,
    learning_rate: float,
    horizon: int,
    action_weight: float,
    motion_loss_weight: float,
    motion_target_speed: float,
    param_weight: float,
    max_param_abs: float,
    seed: int,
    zero_bias: bool,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    state_dim = starts.shape[1]
    theta = np.clip(initial_theta.astype(np.float64, copy=True), -max_param_abs, max_param_abs)
    if zero_bias and theta.size > state_dim:
        theta[state_dim] = 0.0
    best_theta = theta.copy()
    best_loss = rollout_loss(
        theta,
        starts,
        dyn_w,
        horizon,
        steer_limit,
        fixed_speed,
        min_speed,
        max_speed,
        learn_speed,
        heading_weight,
        action_weight,
        motion_loss_weight,
        motion_target_speed,
        zero_bias,
    )
    best_loss += param_weight * float(np.mean(theta * theta))
    print(f"[temporal] initial imagined_loss={best_loss:.5f} theta={theta.tolist()}")
    pop = max(4, int(population))
    if pop % 2:
        pop += 1
    half = pop // 2
    for it in range(1, iterations + 1):
        eps = rng.standard_normal((half, theta.size))
        if zero_bias and eps.shape[1] > state_dim:
            eps[:, state_dim] = 0.0
        candidates = np.clip(np.concatenate([theta + sigma * eps, theta - sigma * eps], axis=0), -max_param_abs, max_param_abs)
        if zero_bias and candidates.shape[1] > state_dim:
            candidates[:, state_dim] = 0.0
        losses = rollout_losses_batch(
            candidates,
            starts,
            dyn_w,
            horizon,
            steer_limit,
            fixed_speed,
            min_speed,
            max_speed,
            learn_speed,
            heading_weight,
            action_weight,
            motion_loss_weight,
            motion_target_speed,
            zero_bias,
        )
        if param_weight > 0.0:
            losses = losses + param_weight * np.mean(candidates * candidates, axis=1)
        rewards = -losses
        std = float(np.std(rewards))
        if std > 1e-9:
            norm = (rewards - float(np.mean(rewards))) / std
            grad = (norm[:, None] * np.concatenate([eps, -eps], axis=0)).mean(axis=0) / max(1e-9, sigma)
            theta = np.clip(theta + learning_rate * grad, -max_param_abs, max_param_abs)
        if zero_bias and theta.size > state_dim:
            theta[state_dim] = 0.0
        current_loss = rollout_loss(
            theta,
            starts,
            dyn_w,
            horizon,
            steer_limit,
            fixed_speed,
            min_speed,
            max_speed,
            learn_speed,
            heading_weight,
            action_weight,
            motion_loss_weight,
            motion_target_speed,
            zero_bias,
        )
        current_loss += param_weight * float(np.mean(theta * theta))
        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta.copy()
        sigma = max(0.01, sigma * 0.995)
        if it == 1 or it % 50 == 0 or it == iterations:
            print(f"[temporal] iter={it:04d} loss={current_loss:.5f} best={best_loss:.5f} sigma={sigma:.3f}")
    return best_theta, best_loss


def save_policy(
    output: Path,
    theta: np.ndarray,
    policy_cfg: Dict[str, Any],
    indices: Sequence[int],
    best_loss: float,
    transition_count: int,
    dynamics_w: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "mode": str(policy_cfg.get("mode", "es_online")),
        "architecture": "linear_feedback",
        "input_size": FEATURE_COUNT,
        "hidden_units": as_int(policy_cfg.get("hidden_units"), 16),
        "linear_feature_indices": [int(v) for v in indices],
        "learn_speed": as_bool(policy_cfg.get("learn_speed"), False),
        "zero_bias": as_bool(policy_cfg.get("zero_bias"), False),
        "theta": [float(v) for v in theta],
        "best_return": -float(best_loss),
        "saved_at": time.time(),
        "trainer": {
            "type": "temporal_linear_dynamics_es",
            "transition_count": int(transition_count),
            "imagined_loss": float(best_loss),
            "dynamics_shape": list(dynamics_w.shape),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[temporal] saved {output} imagined_loss={best_loss:.5f} theta={payload['theta']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear AI lane policy from logged temporal transitions.")
    parser.add_argument("--config", type=Path, default=Path("ai_lane_following_config.yaml"))
    parser.add_argument("--logs", type=Path, default=Path("logs/ai_lane_following_run"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-dt", type=float, default=1.0)
    parser.add_argument("--max-starts", type=int, default=512)
    parser.add_argument("--action-weight", type=float, default=0.015)
    parser.add_argument("--param-weight", type=float, default=0.02)
    parser.add_argument("--max-param-abs", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    policy_cfg = config.get("policy", {}) or {}
    indices = feature_indices(policy_cfg)
    output = args.output
    if output is None:
        output = resolve_path(config_path.parent, str(policy_cfg.get("weights", "learned_policies/ai_lane_policy_linear.json")))
    else:
        output = output.expanduser().resolve()

    files = csv_files(args.logs.expanduser())
    print(f"[temporal] log_files={len(files)} feature_indices={indices}")
    states, actions, next_states, starts_list = collect_transitions(files, indices, max_dt_s=max(0.05, args.max_dt))
    if states.shape[0] < max(12, len(indices) + 4):
        print("[temporal] not enough signed transition data.")
        print("[temporal] Run live_ai_lane_following.py once with the updated logger, collect at least 20-50 episodes, then rerun this script.")
        print(f"[temporal] usable_transitions={states.shape[0]}")
        return 2

    starts = np.stack(starts_list) if starts_list else states.copy()
    if starts.shape[0] > args.max_starts:
        rng = np.random.default_rng(args.seed)
        starts = starts[rng.choice(starts.shape[0], size=args.max_starts, replace=False)]

    dyn_w = fit_linear_dynamics(states, actions, next_states, ridge=max(0.0, args.ridge))
    policy_size = len(indices) + 1
    learn_speed = as_bool(policy_cfg.get("learn_speed"), False)
    if learn_speed:
        policy_size += len(indices) + 1
    initial_theta = load_initial_theta(output, policy_size)
    steer_limit = max(0.05, min(1.0, as_float(policy_cfg.get("steer_limit"), 0.75)))
    min_speed = max(0.0, as_float(policy_cfg.get("min_speed"), 0.07))
    max_speed = max(min_speed, as_float(policy_cfg.get("max_speed"), 0.18))
    fixed_speed = as_float(policy_cfg.get("fixed_speed"), as_float(policy_cfg.get("min_speed"), 0.20))
    heading_weight = max(0.0, as_float(policy_cfg.get("line_heading_loss_weight"), 0.5))
    motion_loss_weight = max(0.0, as_float(policy_cfg.get("motion_loss_weight"), 0.0))
    motion_target_speed = max(0.0, as_float(policy_cfg.get("motion_target_speed"), 0.0))
    zero_bias = as_bool(policy_cfg.get("zero_bias"), False)

    print(
        f"[temporal] transitions={states.shape[0]} starts={starts.shape[0]} "
        f"steer_limit={steer_limit:.2f} fixed_speed={fixed_speed:.2f} learn_speed={learn_speed} "
        f"motion_target={motion_target_speed:.2f} motion_weight={motion_loss_weight:.2f}"
    )
    theta, best_loss = optimize_policy(
        initial_theta,
        starts,
        dyn_w,
        steer_limit=steer_limit,
        fixed_speed=fixed_speed,
        min_speed=min_speed,
        max_speed=max_speed,
        learn_speed=learn_speed,
        heading_weight=heading_weight,
        iterations=max(1, args.iterations),
        population=max(4, args.population),
        sigma=max(1e-4, args.sigma),
        learning_rate=max(0.0, args.learning_rate),
        horizon=max(1, args.horizon),
        action_weight=max(0.0, args.action_weight),
        motion_loss_weight=motion_loss_weight,
        motion_target_speed=motion_target_speed,
        param_weight=max(0.0, args.param_weight),
        max_param_abs=max(0.05, args.max_param_abs),
        seed=args.seed,
        zero_bias=zero_bias,
    )
    save_policy(output, theta, policy_cfg, indices, best_loss, states.shape[0], dyn_w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
