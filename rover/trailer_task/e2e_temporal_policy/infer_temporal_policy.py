#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e2e_dataset import MODE_NAMES, E2ETemporalDataset, discover_metadata, split_metadata_paths
from model import TemporalPolicyNet


def auto_device(value: str) -> torch.device:
    name = str(value or "auto").strip().lower()
    if name in {"", "auto"}:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_model_from_checkpoint(checkpoint: Dict[str, Any]) -> TemporalPolicyNet:
    cfg = checkpoint["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    model = TemporalPolicyNet(
        scalar_dim=len(data_cfg.get("scalar_features", [])),
        num_modes=len(checkpoint.get("mode_names") or MODE_NAMES),
        in_channels=2,
        cnn_channels=model_cfg.get("cnn_channels", [24, 32, 48, 64]),
        frame_feature_dim=int(model_cfg.get("frame_feature_dim", 128)),
        scalar_embed_dim=int(model_cfg.get("scalar_embed_dim", 32)),
        temporal_hidden_dim=int(model_cfg.get("temporal_hidden_dim", 128)),
        temporal_layers=int(model_cfg.get("temporal_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.10)),
        max_motor=float(model_cfg.get("max_motor", 1.0)),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def selected_metadata_paths(
    cfg: Dict[str, Any],
    log_root: Optional[str],
    metadata_paths: Optional[Sequence[str]],
    split: str,
) -> tuple[List[Path], Dict[str, Any]]:
    data_cfg = cfg["data"]
    if metadata_paths:
        paths = [Path(p).expanduser().resolve() for p in metadata_paths]
        return paths, {"split": "explicit", "paths": [str(p) for p in paths]}

    root = Path(log_root or data_cfg.get("log_root", "logs/dotted_lane_following_run"))
    paths = discover_metadata(root, data_cfg.get("metadata_glob", "*/e2e_dataset/metadata.csv"))
    if split == "all":
        return paths, {"split": "all", "paths": [str(p) for p in paths]}

    train_paths, val_paths, report = split_metadata_paths(
        paths,
        float(data_cfg.get("val_fraction", 0.2)),
        int(cfg.get("seed", 20260611)),
        data_cfg.get("val_run_globs") or None,
        bool(data_cfg.get("stratify_turn_direction", True)),
        data_cfg.get("run_direction_overrides") or None,
    )
    chosen = train_paths if split == "train" else val_paths
    report["split"] = split
    report["paths"] = [str(p) for p in chosen]
    return chosen, report


def make_dataset(cfg: Dict[str, Any], paths: Sequence[Path], max_samples: int) -> E2ETemporalDataset:
    data_cfg = cfg["data"]
    return E2ETemporalDataset(
        paths,
        history=int(data_cfg.get("history", 10)),
        image_size=tuple(data_cfg.get("image_size", [192, 160])),
        scalar_features=data_cfg.get("scalar_features", []),
        training=False,
        require_active=bool(data_cfg.get("require_active", True)),
        require_valid=bool(data_cfg.get("require_valid", True)),
        require_sent=bool(data_cfg.get("require_sent", True)),
        max_dt_s=float(data_cfg.get("max_dt_s", 0.5)),
        use_bev_mask=bool(data_cfg.get("use_bev_mask", True)),
        augment={},
        mode_weights=None,
        max_samples=max_samples,
    )


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"samples": 0}
    left = np.asarray([float(r["abs_err_left"]) for r in rows], dtype=np.float64)
    right = np.asarray([float(r["abs_err_right"]) for r in rows], dtype=np.float64)
    mode_ok = np.asarray([int(r["true_mode"] == r["pred_mode"]) for r in rows], dtype=np.float64)
    summary: Dict[str, Any] = {
        "samples": len(rows),
        "wheel_left_mae": float(left.mean()),
        "wheel_right_mae": float(right.mean()),
        "avg_wheel_mae": float((left.mean() + right.mean()) * 0.5),
        "mode_acc": float(mode_ok.mean()),
    }
    for mode in MODE_NAMES:
        subset = [r for r in rows if r["true_mode"] == mode]
        if subset:
            l = np.asarray([float(r["abs_err_left"]) for r in subset], dtype=np.float64)
            rr = np.asarray([float(r["abs_err_right"]) for r in subset], dtype=np.float64)
            summary[f"{mode}_samples"] = len(subset)
            summary[f"{mode}_wheel_mae"] = float((l.mean() + rr.mean()) * 0.5)
    return summary


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    mode_names = checkpoint.get("mode_names") or MODE_NAMES

    paths, split_report = selected_metadata_paths(cfg, args.log_root, args.metadata, args.split)
    if not paths:
        raise FileNotFoundError("No metadata.csv files selected for inference")

    dataset = make_dataset(cfg, paths, args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Selected inference dataset is empty after filtering")

    device = auto_device(args.device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_csv = Path(args.output_csv).expanduser()
    if not output_csv.is_absolute():
        output_csv = checkpoint_path.parent / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for batch in loader:
        bev = batch["bev"].to(device, non_blocking=True).float()
        scalars = batch["scalars"].to(device, non_blocking=True).float()
        wheels = batch["wheels"].cpu().numpy()
        modes = batch["mode"].cpu().numpy()
        outputs = model(bev, scalars)
        pred_wheels = outputs["wheels"].detach().cpu().numpy()
        probs = torch.softmax(outputs["mode_logits"], dim=1).detach().cpu().numpy()
        pred_modes = probs.argmax(axis=1)
        run_names = batch["run_name"]
        frame_indices = batch["frame_idx"].cpu().numpy()
        for idx in range(pred_wheels.shape[0]):
            true_left = float(wheels[idx, 0])
            true_right = float(wheels[idx, 1])
            pred_left = float(pred_wheels[idx, 0])
            pred_right = float(pred_wheels[idx, 1])
            true_mode = mode_names[int(modes[idx])]
            pred_mode = mode_names[int(pred_modes[idx])]
            rows.append(
                {
                    "run_name": str(run_names[idx]),
                    "frame_idx": int(frame_indices[idx]),
                    "true_wheel_left": true_left,
                    "true_wheel_right": true_right,
                    "pred_wheel_left": pred_left,
                    "pred_wheel_right": pred_right,
                    "err_left": pred_left - true_left,
                    "err_right": pred_right - true_right,
                    "abs_err_left": abs(pred_left - true_left),
                    "abs_err_right": abs(pred_right - true_right),
                    "true_mode": true_mode,
                    "pred_mode": pred_mode,
                    "pred_mode_conf": float(probs[idx, int(pred_modes[idx])]),
                }
            )

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "output_csv": str(output_csv),
            "split_report": split_report,
        }
    )
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline inference for the temporal E2E policy.")
    parser.add_argument(
        "--checkpoint",
        default="e2e_temporal_policy/runs/temporal_gru_20260611_182905/best.pt",
        help="Path to best.pt or last.pt.",
    )
    parser.add_argument("--log-root", default=None, help="Override checkpoint data.log_root.")
    parser.add_argument("--metadata", action="append", default=None, help="Explicit metadata.csv path. Can be repeated.")
    parser.add_argument("--split", choices=("all", "train", "val"), default="val")
    parser.add_argument("--output-csv", default="inference_val_predictions.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    summary = run_inference(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

