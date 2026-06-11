#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e2e_dataset import MODE_NAMES, E2ETemporalDataset, discover_metadata, split_metadata_paths
from model import TemporalPolicyNet


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> Dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def build_datasets(cfg: Dict[str, Any]) -> Tuple[E2ETemporalDataset, Optional[E2ETemporalDataset], List[Path], List[Path], Dict[str, Any]]:
    data_cfg = cfg["data"]
    log_root = Path(data_cfg.get("log_root", "logs/dotted_lane_following_run"))
    metadata_paths = discover_metadata(log_root, data_cfg.get("metadata_glob", "*/e2e_dataset/metadata.csv"))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata.csv files found under {log_root}")
    train_paths, val_paths, split_report = split_metadata_paths(
        metadata_paths,
        float(data_cfg.get("val_fraction", 0.2)),
        int(cfg.get("seed", 20260611)),
        data_cfg.get("val_run_globs") or None,
        bool(data_cfg.get("stratify_turn_direction", True)),
        data_cfg.get("run_direction_overrides") or None,
    )
    common = dict(
        history=int(data_cfg.get("history", 6)),
        image_size=tuple(data_cfg.get("image_size", [192, 160])),
        scalar_features=data_cfg.get("scalar_features", ["angle_deg", "angle_rate_deg_s", "angle_confidence", "angle_age_s", "angle_ok", "dt_s"]),
        require_active=bool(data_cfg.get("require_active", True)),
        require_valid=bool(data_cfg.get("require_valid", True)),
        require_sent=bool(data_cfg.get("require_sent", True)),
        max_dt_s=float(data_cfg.get("max_dt_s", 0.5)),
        use_bev_mask=bool(data_cfg.get("use_bev_mask", True)),
        mode_weights=cfg.get("loss", {}).get("motor_mode_weights", {}),
    )
    train_ds = E2ETemporalDataset(
        train_paths,
        training=True,
        augment=cfg.get("augmentation", {}),
        max_samples=int(data_cfg.get("max_train_samples", 0)),
        **common,
    )
    val_ds = None
    if val_paths:
        val_ds = E2ETemporalDataset(
            val_paths,
            training=False,
            augment={},
            max_samples=int(data_cfg.get("max_val_samples", 0)),
            **common,
        )
    return train_ds, val_ds, train_paths, val_paths, split_report


def build_model(cfg: Dict[str, Any], scalar_dim: int) -> TemporalPolicyNet:
    model_cfg = cfg.get("model", {})
    return TemporalPolicyNet(
        scalar_dim=scalar_dim,
        num_modes=len(MODE_NAMES),
        in_channels=2,
        cnn_channels=model_cfg.get("cnn_channels", [24, 32, 48, 64]),
        frame_feature_dim=int(model_cfg.get("frame_feature_dim", 128)),
        scalar_embed_dim=int(model_cfg.get("scalar_embed_dim", 32)),
        temporal_hidden_dim=int(model_cfg.get("temporal_hidden_dim", 128)),
        temporal_layers=int(model_cfg.get("temporal_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.10)),
        max_motor=float(model_cfg.get("max_motor", 1.0)),
    )


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    class_weights: Optional[torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = cfg.get("loss", {})
    pred = outputs["wheels"]
    target = batch["wheels"]
    per_sample = F.smooth_l1_loss(pred, target, reduction="none", beta=float(loss_cfg.get("huber_beta", 0.05))).mean(dim=1)
    weights = batch.get("sample_weight")
    if torch.is_tensor(weights):
        motor_loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        motor_loss = per_sample.mean()

    mode_loss = F.cross_entropy(outputs["mode_logits"], batch["mode"], weight=class_weights)
    total = motor_loss + float(loss_cfg.get("mode_loss_weight", 0.20)) * mode_loss
    return total, {
        "motor_loss": float(motor_loss.detach().cpu()),
        "mode_loss": float(mode_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: TemporalPolicyNet,
    loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "motor_loss": 0.0, "mode_loss": 0.0}
    count = 0
    correct = 0
    mae_sum = torch.zeros(2, dtype=torch.float64)
    per_mode_abs = {name: torch.zeros(2, dtype=torch.float64) for name in MODE_NAMES}
    per_mode_count = {name: 0 for name in MODE_NAMES}

    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(batch["bev"].float(), batch["scalars"].float())
        loss, parts = compute_loss(outputs, batch, class_weights, cfg)
        batch_size = int(batch["wheels"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["motor_loss"] += parts["motor_loss"] * batch_size
        totals["mode_loss"] += parts["mode_loss"] * batch_size
        count += batch_size
        abs_err = (outputs["wheels"] - batch["wheels"]).abs().detach().cpu().double()
        mae_sum += abs_err.sum(dim=0)
        mode_pred = outputs["mode_logits"].argmax(dim=1)
        correct += int((mode_pred == batch["mode"]).sum().detach().cpu())
        modes_cpu = batch["mode"].detach().cpu().tolist()
        for idx, mode_idx in enumerate(modes_cpu):
            name = MODE_NAMES[int(mode_idx)]
            per_mode_abs[name] += abs_err[idx]
            per_mode_count[name] += 1

    if count == 0:
        return {"loss": float("inf"), "samples": 0}
    metrics: Dict[str, Any] = {
        "loss": totals["loss"] / count,
        "motor_loss": totals["motor_loss"] / count,
        "mode_loss": totals["mode_loss"] / count,
        "wheel_left_mae": float(mae_sum[0] / count),
        "wheel_right_mae": float(mae_sum[1] / count),
        "mode_acc": correct / count,
        "samples": count,
    }
    for name in MODE_NAMES:
        n = per_mode_count[name]
        if n:
            metrics[f"{name}_wheel_mae"] = float(per_mode_abs[name].mean() / n)
    return metrics


def make_loader(dataset: E2ETemporalDataset, cfg: Dict[str, Any], shuffle: bool) -> DataLoader:
    train_cfg = cfg.get("train", {})
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= int(train_cfg.get("batch_size", 16)),
    )


def save_checkpoint(
    path: Path,
    model: TemporalPolicyNet,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "mode_names": MODE_NAMES,
            "metrics": metrics,
        },
        path,
    )


def train(cfg: Dict[str, Any]) -> Path:
    set_seed(int(cfg.get("seed", 20260611)))
    train_cfg = cfg.get("train", {})
    output_dir = Path(train_cfg.get("output_dir", "e2e_temporal_policy/runs/temporal_gru")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    train_ds, val_ds, train_paths, val_paths, split_report = build_datasets(cfg)
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty after filtering")
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False) if val_ds and len(val_ds) else None

    device_name = str(train_cfg.get("device", "auto")).strip().lower()
    if device_name in {"", "auto"}:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = build_model(cfg, scalar_dim=len(train_ds.scalar_features)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg.get("epochs", 80))),
        eta_min=float(train_cfg.get("min_lr", 1.0e-5)),
    )
    class_weights = train_ds.class_weights().to(device) if bool(cfg.get("loss", {}).get("balanced_mode_loss", True)) else None

    info = {
        "train_runs": [p.parent.parent.name for p in train_paths],
        "val_runs": [p.parent.parent.name for p in val_paths],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "mode_names": MODE_NAMES,
        "train_mode_counts": train_ds.mode_counts(),
        "val_mode_counts": val_ds.mode_counts() if val_ds else {},
        "turn_direction_report": split_report,
        "scalar_features": train_ds.scalar_features,
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    for warning in split_report.get("warnings", []):
        print(f"[split-warning] {warning}")

    metrics_path = output_dir / "metrics.jsonl"
    best_metric = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    epochs = int(train_cfg.get("epochs", 80))
    max_train_batches = int(train_cfg.get("max_train_batches", 0))
    max_val_batches = int(train_cfg.get("max_val_batches", 0))
    log_every = int(train_cfg.get("log_every", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_totals = {"loss": 0.0, "motor_loss": 0.0, "mode_loss": 0.0}
        sample_count = 0
        start = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["bev"].float(), batch["scalars"].float())
            loss, parts = compute_loss(outputs, batch, class_weights, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 5.0)))
            optimizer.step()

            batch_size = int(batch["wheels"].shape[0])
            epoch_totals["loss"] += float(loss.detach().cpu()) * batch_size
            epoch_totals["motor_loss"] += parts["motor_loss"] * batch_size
            epoch_totals["mode_loss"] += parts["mode_loss"] * batch_size
            sample_count += batch_size
            if log_every > 0 and batch_idx % log_every == 0:
                print(f"epoch={epoch} batch={batch_idx} loss={float(loss.detach().cpu()):.5f}")
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

        scheduler.step()
        train_metrics = {
            key: value / max(sample_count, 1)
            for key, value in epoch_totals.items()
        }
        train_metrics["samples"] = sample_count
        train_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        train_metrics["seconds"] = time.time() - start

        val_metrics: Dict[str, Any] = {}
        if val_loader is not None:
            if max_val_batches > 0:
                limited_batches = []
                for idx, batch in enumerate(val_loader, start=1):
                    limited_batches.append(batch)
                    if idx >= max_val_batches:
                        break
                val_metrics = evaluate(model, limited_batches, device, class_weights, cfg)  # type: ignore[arg-type]
            else:
                val_metrics = evaluate(model, val_loader, device, class_weights, cfg)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        metric = float(val_metrics.get("loss", train_metrics["loss"]))
        if metric < best_metric:
            best_metric = metric
            save_checkpoint(best_path, model, optimizer, cfg, epoch, record)
        save_checkpoint(last_path, model, optimizer, cfg, epoch, record)

        if val_metrics:
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"val_mae=({val_metrics['wheel_left_mae']:.4f},{val_metrics['wheel_right_mae']:.4f}) "
                f"mode_acc={val_metrics['mode_acc']:.3f}"
            )
        else:
            print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.5f}")

    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal E2E motor policy from dotted-lane E2E logs.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "configs" / "temporal_gru.yaml"))
    parser.add_argument("--log-root", default=None, help="Override data.log_root")
    parser.add_argument("--output-dir", default=None, help="Override train.output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--device", default=None, help="Override train.device")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny train/eval pass for validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    overrides: Dict[str, Any] = {}
    if args.log_root:
        overrides.setdefault("data", {})["log_root"] = args.log_root
    if args.output_dir:
        overrides.setdefault("train", {})["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides.setdefault("train", {})["epochs"] = args.epochs
    if args.device:
        overrides.setdefault("train", {})["device"] = args.device
    if args.smoke_test:
        overrides = deep_update(overrides, {
            "train": {
                "epochs": 1,
                "batch_size": 4,
                "num_workers": 0,
                "max_train_batches": 2,
                "max_val_batches": 1,
                "output_dir": str(SCRIPT_DIR / "runs" / "smoke"),
            },
            "data": {
                "max_train_samples": 0,
                "max_val_samples": 0,
            },
        })
    cfg = deep_update(cfg, overrides)
    best_path = train(cfg)
    print(f"best_checkpoint={best_path}")


if __name__ == "__main__":
    main()
