#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


FEATURE_NAMES = [
    "det_conf",
    "width_norm",
    "height_norm",
    "center_x_norm",
    "center_y_norm",
    "area_norm",
    "log_area",
    "aspect",
]


@dataclass
class Row:
    camera_key: str
    angle_deg: float
    features: np.ndarray


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fit bbox-feature to hinge-angle calibration models.")
    parser.add_argument("--csv", default="", help="angle_calibration.csv. Defaults to newest angle_calib_run*/angle_calibration.csv.")
    parser.add_argument("--output", default="", help="Defaults to CSV directory/angle_model.json.")
    parser.add_argument("--min-conf", type=float, default=0.18)
    parser.add_argument("--ridge-lambda", type=float, default=0.10)
    return parser.parse_args()


def newest_csv(root: Path) -> Path:
    candidates = sorted(
        root.glob("angle_calib_run_*/angle_calibration.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No angle_calib_run_*/angle_calibration.csv found under {root}")
    return candidates[0].resolve()


def as_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        value_f = float(value)
        if not math.isfinite(value_f):
            return None
        return value_f
    except Exception:
        return None


def read_rows(csv_path: Path, min_conf: float) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if raw.get("detected") != "1":
                continue
            conf = as_float(raw.get("det_conf", ""))
            angle = as_float(raw.get("angle_deg", ""))
            if conf is None or angle is None or conf < min_conf:
                continue
            feats: List[float] = []
            ok = True
            for name in FEATURE_NAMES:
                value = as_float(raw.get(name, ""))
                if value is None:
                    ok = False
                    break
                feats.append(value)
            if not ok:
                continue
            rows.append(Row(camera_key=raw.get("camera_key", ""), angle_deg=angle, features=np.asarray(feats, dtype=np.float64)))
    return rows


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xz, mean, std = standardize_fit(x)
    xb = np.concatenate([np.ones((xz.shape[0], 1)), xz], axis=1)
    reg = np.eye(xb.shape[1], dtype=np.float64) * float(ridge_lambda)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return coef, mean, std


def predict(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xz = (x - mean) / std
    xb = np.concatenate([np.ones((xz.shape[0], 1)), xz], axis=1)
    return xb @ coef


def metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - y
    return {
        "count": int(len(y)),
        "mae_deg": float(np.mean(np.abs(err))) if len(y) else float("nan"),
        "rmse_deg": float(np.sqrt(np.mean(err * err))) if len(y) else float("nan"),
        "max_abs_err_deg": float(np.max(np.abs(err))) if len(y) else float("nan"),
    }


def grouped_angle_cv(rows: Sequence[Row], ridge_lambda: float) -> Dict[str, float]:
    angles = sorted({round(r.angle_deg, 3) for r in rows})
    if len(angles) < 3:
        return {"folds": len(angles), "mae_deg": float("nan"), "rmse_deg": float("nan"), "max_abs_err_deg": float("nan")}
    preds: List[float] = []
    ys: List[float] = []
    for held_angle in angles:
        train = [r for r in rows if round(r.angle_deg, 3) != held_angle]
        test = [r for r in rows if round(r.angle_deg, 3) == held_angle]
        if len(train) < 4 or not test:
            continue
        x_train = np.stack([r.features for r in train])
        y_train = np.asarray([r.angle_deg for r in train], dtype=np.float64)
        coef, mean, std = ridge_fit(x_train, y_train, ridge_lambda)
        x_test = np.stack([r.features for r in test])
        pred = predict(x_test, coef, mean, std)
        preds.extend(float(v) for v in pred)
        ys.extend(r.angle_deg for r in test)
    if not ys:
        return {"folds": len(angles), "mae_deg": float("nan"), "rmse_deg": float("nan"), "max_abs_err_deg": float("nan")}
    out = metrics(np.asarray(ys), np.asarray(preds))
    out["folds"] = len(angles)
    return out


def fit_camera(rows: Sequence[Row], ridge_lambda: float) -> Dict[str, object]:
    x = np.stack([r.features for r in rows])
    y = np.asarray([r.angle_deg for r in rows], dtype=np.float64)
    coef, mean, std = ridge_fit(x, y, ridge_lambda)
    pred = predict(x, coef, mean, std)
    return {
        "count": int(len(rows)),
        "angles": sorted(float(a) for a in {round(r.angle_deg, 3) for r in rows}),
        "train_metrics": metrics(y, pred),
        "leave_one_angle_out": grouped_angle_cv(rows, ridge_lambda),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": coef.tolist(),
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else newest_csv(root)
    output = Path(args.output).expanduser().resolve() if args.output else csv_path.parent / "angle_model.json"
    rows = read_rows(csv_path, min_conf=float(args.min_conf))
    if not rows:
        raise SystemExit(f"No detected calibration rows found in {csv_path}")

    by_camera: Dict[str, List[Row]] = {}
    for row in rows:
        by_camera.setdefault(row.camera_key, []).append(row)

    model: Dict[str, object] = {
        "source_csv": str(csv_path),
        "feature_names": FEATURE_NAMES,
        "min_conf": float(args.min_conf),
        "ridge_lambda": float(args.ridge_lambda),
        "cameras": {},
    }

    for camera_key, camera_rows in sorted(by_camera.items()):
        if len(camera_rows) < 8:
            print(f"[skip] {camera_key}: only {len(camera_rows)} rows")
            continue
        model["cameras"][camera_key] = fit_camera(camera_rows, ridge_lambda=float(args.ridge_lambda))

    output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Rows used: {len(rows)}")
    print(f"Saved: {output}")
    for camera_key, camera_model in model["cameras"].items():
        cv = camera_model["leave_one_angle_out"]
        tr = camera_model["train_metrics"]
        print(
            f"{camera_key}: count={camera_model['count']} "
            f"train_MAE={tr['mae_deg']:.2f}deg "
            f"LOAO_MAE={cv['mae_deg']:.2f}deg "
            f"LOAO_RMSE={cv['rmse_deg']:.2f}deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
