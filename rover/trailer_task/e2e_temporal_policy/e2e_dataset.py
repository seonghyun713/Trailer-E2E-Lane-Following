#!/usr/bin/env python3
from __future__ import annotations

import csv
import fnmatch
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


MODE_NAMES = ["normal", "straight", "corner", "pivot", "bias"]
MODE_TO_INDEX = {name: idx for idx, name in enumerate(MODE_NAMES)}
TURN_DIRECTIONS = ["left", "right", "mixed", "unknown"]


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
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def mode_from_row(row: Dict[str, str]) -> str:
    state = f"{row.get('command_state', '')} {row.get('route_mode', '')}".upper()
    if "PIVOT" in state:
        return "pivot"
    if "POST_CORNER_BIAS" in state or "BIAS" in state:
        return "bias"
    if "CORNER" in state:
        return "corner"
    if "STRAIGHT" in state:
        return "straight"
    return "normal"


def discover_metadata(log_root: Path, glob_pattern: str = "*/e2e_dataset/metadata.csv") -> List[Path]:
    root = log_root.expanduser().resolve()
    if root.is_file():
        return [root]
    return sorted(root.glob(glob_pattern))


def run_name_from_metadata(path: Path) -> str:
    return Path(path).parent.parent.name


def infer_run_turn_direction(
    metadata_path: Path,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    path = Path(metadata_path)
    run_name = run_name_from_metadata(path)
    override = (overrides or {}).get(run_name)
    if override:
        direction = str(override).strip().lower()
        if direction not in TURN_DIRECTIONS:
            direction = "unknown"
        return {
            "run": run_name,
            "direction": direction,
            "source": "override",
            "left_votes": 0,
            "right_votes": 0,
            "mean_pivot_wheel_right_minus_left": 0.0,
            "mean_corner_steer": 0.0,
            "rows": 0,
        }

    rows = 0
    left_votes = 0
    right_votes = 0
    pivot_diffs: List[float] = []
    corner_steers: List[float] = []
    text_left = 0
    text_right = 0

    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows += 1
                state = f"{row.get('command_state', '')} {row.get('route_mode', '')}".upper()
                if "LEFT" in state:
                    text_left += 1
                if "RIGHT" in state:
                    text_right += 1
                wheel_left = as_float(row.get("wheel_left"), math.nan)
                wheel_right = as_float(row.get("wheel_right"), math.nan)
                diff = wheel_right - wheel_left if math.isfinite(wheel_left) and math.isfinite(wheel_right) else math.nan
                is_pivot = "PIVOT" in state or (math.isfinite(diff) and abs(diff) >= 0.8)
                if is_pivot and math.isfinite(diff):
                    pivot_diffs.append(diff)
                    if diff > 0.25:
                        left_votes += 1
                    elif diff < -0.25:
                        right_votes += 1
                if "CORNER" in state or "BIAS" in state or "PIVOT" in state:
                    steer = as_float(row.get("command_steer"), math.nan)
                    if math.isfinite(steer):
                        corner_steers.append(steer)

    total_votes = left_votes + right_votes
    mean_diff = float(np.mean(pivot_diffs)) if pivot_diffs else 0.0
    mean_steer = float(np.mean(corner_steers)) if corner_steers else 0.0
    source = "motor_pivot"

    if total_votes >= 5:
        left_ratio = left_votes / total_votes
        right_ratio = right_votes / total_votes
        if left_ratio >= 0.75:
            direction = "left"
        elif right_ratio >= 0.75:
            direction = "right"
        else:
            direction = "mixed"
    elif abs(mean_steer) >= 0.05:
        direction = "right" if mean_steer > 0.0 else "left"
        source = "corner_steer"
    elif text_left + text_right >= 5:
        source = "state_text"
        if text_left >= 3 * max(text_right, 1):
            direction = "left"
        elif text_right >= 3 * max(text_left, 1):
            direction = "right"
        else:
            direction = "mixed"
    else:
        direction = "unknown"
        source = "insufficient_signal"

    return {
        "run": run_name,
        "direction": direction,
        "source": source,
        "left_votes": left_votes,
        "right_votes": right_votes,
        "text_left": text_left,
        "text_right": text_right,
        "mean_pivot_wheel_right_minus_left": mean_diff,
        "mean_corner_steer": mean_steer,
        "rows": rows,
    }


def direction_report(
    metadata_paths: Sequence[Path],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    runs = [infer_run_turn_direction(Path(p), overrides) for p in metadata_paths]
    counts = {name: 0 for name in TURN_DIRECTIONS}
    for run in runs:
        counts[str(run["direction"])] = counts.get(str(run["direction"]), 0) + 1
    return {"runs": runs, "counts": counts}


def split_metadata_paths(
    metadata_paths: Sequence[Path],
    val_fraction: float,
    seed: int,
    val_run_globs: Optional[Sequence[str]] = None,
    stratify_turn_direction: bool = True,
    run_direction_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[List[Path], List[Path], Dict[str, Any]]:
    paths = sorted(Path(p) for p in metadata_paths)
    report = direction_report(paths, run_direction_overrides)
    report["warnings"] = []
    if not paths:
        return [], [], report

    if val_run_globs:
        val_patterns = [str(p) for p in val_run_globs]
        val_paths = [
            p for p in paths
            if any(fnmatch.fnmatch(p.parent.parent.name, pat) for pat in val_patterns)
        ]
        train_paths = [p for p in paths if p not in set(val_paths)]
        if train_paths and val_paths:
            _append_split_warnings(report, train_paths, val_paths)
            return train_paths, val_paths, report

    if len(paths) == 1 or val_fraction <= 0.0:
        _append_split_warnings(report, paths, [])
        return paths, [], report

    runs = list(paths)
    rng = random.Random(seed)
    val_set = set()

    if stratify_turn_direction:
        by_direction: Dict[str, List[Path]] = {}
        direction_by_run = {item["run"]: item["direction"] for item in report["runs"]}
        rows_by_run = {item["run"]: int(item.get("rows", 0)) for item in report["runs"]}
        for path in runs:
            by_direction.setdefault(str(direction_by_run.get(run_name_from_metadata(path), "unknown")), []).append(path)
        for direction, group in by_direction.items():
            if direction in {"left", "right"} and len(group) <= 1:
                report["warnings"].append(
                    f"Only one {direction} run is available; keeping it in train so the policy sees that direction."
                )
                continue
            if len(group) <= 1:
                continue
            group = sorted(group, key=lambda p: rows_by_run.get(run_name_from_metadata(p), 0), reverse=True)
            protected_train = group[0]
            candidates = group[1:]
            rng.shuffle(candidates)
            count = max(1, int(round(len(group) * val_fraction)))
            count = min(count, len(candidates))
            val_set.update(candidates[:count])
            report["warnings"].append(
                f"Keeping largest {direction} run {run_name_from_metadata(protected_train)} in train for direction coverage."
            )
    else:
        rng.shuffle(runs)
        val_count = max(1, int(round(len(runs) * val_fraction)))
        val_set = set(runs[:val_count])

    train_paths = [p for p in paths if p not in val_set]
    val_paths = [p for p in paths if p in val_set]
    if not train_paths:
        train_paths, val_paths = paths[:-1], paths[-1:]
    if not val_paths and len(paths) > 1:
        candidates = [p for p in runs if p in train_paths]
        if len(candidates) > 1:
            val_paths = [candidates[0]]
            train_paths = [p for p in train_paths if p != candidates[0]]
    _append_split_warnings(report, train_paths, val_paths)
    return train_paths, val_paths, report


def _split_direction_counts(report: Dict[str, Any], paths: Sequence[Path]) -> Dict[str, int]:
    wanted = {run_name_from_metadata(p) for p in paths}
    counts = {name: 0 for name in TURN_DIRECTIONS}
    for item in report.get("runs", []):
        if item.get("run") in wanted:
            direction = str(item.get("direction", "unknown"))
            counts[direction] = counts.get(direction, 0) + 1
    return counts


def _append_split_warnings(report: Dict[str, Any], train_paths: Sequence[Path], val_paths: Sequence[Path]) -> None:
    train_counts = _split_direction_counts(report, train_paths)
    val_counts = _split_direction_counts(report, val_paths)
    report["train_direction_counts"] = train_counts
    report["val_direction_counts"] = val_counts
    warnings = report.setdefault("warnings", [])
    for direction in ("left", "right"):
        total = int(report.get("counts", {}).get(direction, 0))
        if total > 0 and train_counts.get(direction, 0) == 0:
            warnings.append(f"Train split has no {direction} runs although {total} exist in the dataset.")
        if total >= 2 and val_counts.get(direction, 0) == 0:
            warnings.append(f"Validation split has no {direction} runs even though at least two exist.")
        if total == 1 and val_counts.get(direction, 0) == 0:
            warnings.append(f"Validation split has no {direction} run because only one exists; collect another {direction} run for proper validation.")


@dataclass(frozen=True)
class SequenceSample:
    run_name: str
    dataset_dir: Path
    rows: Tuple[Dict[str, str], ...]
    target_row: Dict[str, str]
    mode_name: str
    mode_index: int


class E2ETemporalDataset(Dataset):
    def __init__(
        self,
        metadata_paths: Sequence[Path],
        history: int,
        image_size: Tuple[int, int],
        scalar_features: Sequence[str],
        training: bool = False,
        require_active: bool = True,
        require_valid: bool = True,
        require_sent: bool = True,
        max_dt_s: float = 0.5,
        use_bev_mask: bool = True,
        augment: Optional[Dict[str, Any]] = None,
        mode_weights: Optional[Dict[str, float]] = None,
        max_samples: int = 0,
    ) -> None:
        if history < 1:
            raise ValueError("history must be >= 1")
        self.metadata_paths = [Path(p).expanduser().resolve() for p in metadata_paths]
        self.history = int(history)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.scalar_features = list(scalar_features)
        self.training = bool(training)
        self.require_active = bool(require_active)
        self.require_valid = bool(require_valid)
        self.require_sent = bool(require_sent)
        self.max_dt_s = float(max_dt_s)
        self.use_bev_mask = bool(use_bev_mask)
        self.augment = dict(augment or {})
        self.mode_weights = {name: float(mode_weights.get(name, 1.0)) for name in MODE_NAMES} if mode_weights else {
            name: 1.0 for name in MODE_NAMES
        }
        self.samples = self._load_samples()
        if max_samples and max_samples > 0:
            self.samples = self.samples[: int(max_samples)]

    def _is_valid_row(self, row: Dict[str, str]) -> bool:
        if self.require_active and not as_bool(row.get("active"), False):
            return False
        if self.require_valid and not as_bool(row.get("command_valid"), False):
            return False
        if self.require_sent and not as_bool(row.get("wheel_sent"), False):
            return False
        if not math.isfinite(as_float(row.get("wheel_left"), math.nan)):
            return False
        if not math.isfinite(as_float(row.get("wheel_right"), math.nan)):
            return False
        return self._resolve_bev_path(row, row.get("_dataset_dir")) is not None

    def _load_samples(self) -> List[SequenceSample]:
        samples: List[SequenceSample] = []
        for metadata_path in self.metadata_paths:
            if not metadata_path.exists():
                continue
            dataset_dir = metadata_path.parent
            run_name = metadata_path.parent.parent.name
            with metadata_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row["_dataset_dir"] = str(dataset_dir)

            segment: List[Dict[str, str]] = []
            prev_ts: Optional[float] = None
            for row in rows:
                ts = as_float(row.get("timestamp_monotonic"), math.nan)
                gap = prev_ts is not None and math.isfinite(ts) and ts - prev_ts > self.max_dt_s
                if gap or not self._is_valid_row(row):
                    segment = []
                    prev_ts = ts if math.isfinite(ts) else None
                    continue

                segment.append(row)
                prev_ts = ts if math.isfinite(ts) else prev_ts

                seq = segment[-self.history :]
                if len(seq) < self.history:
                    pad = [seq[0]] * (self.history - len(seq))
                    seq = pad + seq
                mode_name = mode_from_row(row)
                samples.append(
                    SequenceSample(
                        run_name=run_name,
                        dataset_dir=dataset_dir,
                        rows=tuple(seq),
                        target_row=row,
                        mode_name=mode_name,
                        mode_index=MODE_TO_INDEX[mode_name],
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def mode_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in MODE_NAMES}
        for sample in self.samples:
            counts[sample.mode_name] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        counts = self.mode_counts()
        nonzero = [v for v in counts.values() if v > 0]
        if not nonzero:
            return torch.ones(len(MODE_NAMES), dtype=torch.float32)
        median = float(np.median(nonzero))
        weights = []
        for name in MODE_NAMES:
            count = max(counts[name], 1)
            weights.append(math.sqrt(median / count))
        return torch.tensor(weights, dtype=torch.float32)

    def _resolve_bev_path(self, row: Dict[str, str], dataset_dir_value: Any) -> Optional[Path]:
        dataset_dir = Path(str(dataset_dir_value)) if dataset_dir_value else None
        if dataset_dir is None:
            return None
        candidates: List[str] = []
        if self.use_bev_mask:
            candidates.extend([row.get("bev_mask_path", ""), row.get("bev_input_path", "")])
        else:
            candidates.extend([row.get("bev_input_path", ""), row.get("bev_mask_path", "")])
        for value in candidates:
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = dataset_dir / path
            if path.exists():
                return path
        return None

    def _load_bev(self, row: Dict[str, str]) -> np.ndarray:
        path = self._resolve_bev_path(row, row.get("_dataset_dir"))
        if path is None:
            raise FileNotFoundError(f"Missing BEV mask for frame {row.get('frame_idx')}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read BEV mask: {path}")
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

    def _scalar_value(self, row: Dict[str, str], name: str, prev_row: Optional[Dict[str, str]]) -> float:
        if name == "angle_deg":
            return as_float(row.get("trailer_angle_deg"), 0.0) / 50.0
        if name == "angle_rate_deg_s":
            return np.clip(as_float(row.get("trailer_angle_rate_deg_s"), 0.0), -150.0, 150.0) / 150.0
        if name == "angle_confidence":
            return np.clip(as_float(row.get("trailer_angle_confidence"), 0.0), 0.0, 1.0)
        if name == "angle_age_s":
            return np.clip(as_float(row.get("trailer_angle_age_s"), 0.0), 0.0, 1.0)
        if name == "angle_ok":
            return 1.0 if as_bool(row.get("trailer_angle_ok"), False) else 0.0
        if name == "dt_s":
            if prev_row is None:
                return 0.0
            now = as_float(row.get("timestamp_monotonic"), 0.0)
            prev = as_float(prev_row.get("timestamp_monotonic"), now)
            return np.clip(now - prev, 0.0, self.max_dt_s) / max(self.max_dt_s, 1e-6)
        raise KeyError(f"Unknown scalar feature: {name}")

    def _load_scalars(self, rows: Sequence[Dict[str, str]]) -> np.ndarray:
        values: List[List[float]] = []
        prev_row: Optional[Dict[str, str]] = None
        for row in rows:
            values.append([self._scalar_value(row, name, prev_row) for name in self.scalar_features])
            prev_row = row
        scalars = np.asarray(values, dtype=np.float32)
        if self.training:
            angle_noise_std = float(self.augment.get("angle_noise_std_deg", 0.0))
            if angle_noise_std > 0.0 and "angle_deg" in self.scalar_features:
                idx = self.scalar_features.index("angle_deg")
                scalars[:, idx] += np.random.normal(0.0, angle_noise_std / 50.0, size=scalars.shape[0]).astype(np.float32)
            rate_noise_std = float(self.augment.get("angle_rate_noise_std_deg_s", 0.0))
            if rate_noise_std > 0.0 and "angle_rate_deg_s" in self.scalar_features:
                idx = self.scalar_features.index("angle_rate_deg_s")
                scalars[:, idx] += np.random.normal(0.0, rate_noise_std / 150.0, size=scalars.shape[0]).astype(np.float32)
        return scalars

    def _augment_bev(self, bev: np.ndarray) -> np.ndarray:
        if not self.training:
            return bev

        pixel_drop = float(self.augment.get("mask_pixel_dropout", 0.0))
        if pixel_drop > 0.0:
            keep = np.random.random_sample(bev.shape) >= pixel_drop
            bev = bev * keep.astype(np.float32)

        rect_prob = float(self.augment.get("mask_rect_dropout_prob", 0.0))
        rect_max_frac = float(self.augment.get("mask_rect_max_frac", 0.12))
        if rect_prob > 0.0 and np.random.random() < rect_prob:
            t, _, h, w = bev.shape
            rh = max(1, int(h * np.random.uniform(0.03, rect_max_frac)))
            rw = max(1, int(w * np.random.uniform(0.03, rect_max_frac)))
            y0 = np.random.randint(0, max(1, h - rh + 1))
            x0 = np.random.randint(0, max(1, w - rw + 1))
            frame0 = np.random.randint(0, t)
            bev[frame0, :, y0 : y0 + rh, x0 : x0 + rw] = 0.0

        morph_prob = float(self.augment.get("mask_morph_prob", 0.0))
        if morph_prob > 0.0 and np.random.random() < morph_prob:
            kernel = np.ones((3, 3), dtype=np.uint8)
            op = cv2.dilate if np.random.random() < 0.5 else cv2.erode
            out = bev.copy()
            for ti in range(out.shape[0]):
                for ci in range(out.shape[1]):
                    out[ti, ci] = op((out[ti, ci] > 0.5).astype(np.uint8), kernel, iterations=1).astype(np.float32)
            bev = out

        return bev

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        bev = np.stack([self._load_bev(row) for row in sample.rows], axis=0)
        bev = self._augment_bev(bev)
        scalars = self._load_scalars(sample.rows)
        row = sample.target_row
        wheels = np.asarray(
            [as_float(row.get("wheel_left"), 0.0), as_float(row.get("wheel_right"), 0.0)],
            dtype=np.float32,
        )
        command = np.asarray(
            [as_float(row.get("command_steer"), 0.0), as_float(row.get("command_speed"), 0.0)],
            dtype=np.float32,
        )
        return {
            "bev": torch.from_numpy(bev),
            "scalars": torch.from_numpy(scalars),
            "wheels": torch.from_numpy(wheels),
            "command": torch.from_numpy(command),
            "mode": torch.tensor(sample.mode_index, dtype=torch.long),
            "sample_weight": torch.tensor(self.mode_weights[sample.mode_name], dtype=torch.float32),
            "run_name": sample.run_name,
            "frame_idx": as_int(row.get("frame_idx"), -1),
        }
