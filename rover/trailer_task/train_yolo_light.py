#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a lightweight YOLO detector for the trailer blue panel.")
    parser.add_argument("--dataset", default="", help="bbox_dataset* directory. Defaults to newest one.")
    parser.add_argument("--yolo-dir", default="", help="Prepared YOLO dataset dir. Defaults to DATASET/yolo_panel.")
    parser.add_argument("--model", default="yolo11n.pt", help="Use yolo11n.pt for best light default; yolov8n.pt also works.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", default="8", help="Batch size or -1 for Ultralytics autobatch.")
    parser.add_argument("--device", default="0", help="0 for CUDA, cpu for CPU.")
    parser.add_argument("--project", default=str(here / "runs_yolo"))
    parser.add_argument("--name", default="trailer_panel_yolo11n")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prepare", action="store_true", help="Skip prepare_yolo_dataset.py.")
    parser.add_argument("--resume", action="store_true", help="Resume from PROJECT/NAME/weights/last.pt unless --model points to a checkpoint.")
    parser.add_argument("--export", action="store_true", help="Export ONNX after training.")
    return parser.parse_args()


def newest_dataset(root: Path) -> Path:
    candidates = sorted(
        (p for p in root.glob("bbox_dataset*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No bbox_dataset* directory found under {root}")
    return candidates[0].resolve()


def run_prepare(script_dir: Path, dataset: Path, yolo_dir: Path, seed: int) -> None:
    cmd = [
        sys.executable,
        str(script_dir / "prepare_yolo_dataset.py"),
        "--dataset",
        str(dataset),
        "--output",
        str(yolo_dir),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else newest_dataset(script_dir)
    yolo_dir = Path(args.yolo_dir).expanduser().resolve() if args.yolo_dir else dataset / "yolo_panel"
    if not args.no_prepare:
        run_prepare(script_dir, dataset, yolo_dir, args.seed)

    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is required. Install with: pip3 install ultralytics\n{exc}") from exc

    model_path = Path(args.model).expanduser()
    if args.resume and not model_path.exists():
        candidate = Path(args.project).expanduser().resolve() / args.name / "weights" / "last.pt"
        if candidate.exists():
            model_path = candidate
    model = YOLO(str(model_path) if model_path.exists() else args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch,
        device=args.device,
        project=str(Path(args.project).expanduser().resolve()),
        name=args.name,
        patience=int(args.patience),
        workers=int(args.workers),
        seed=int(args.seed),
        single_cls=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        cache=False,
        plots=True,
        verbose=True,
        resume=bool(args.resume),
    )

    run_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = run_dir / "weights" / "best.pt"
    print(f"Run dir: {run_dir}")
    print(f"Best weights: {best}")

    if args.export:
        export_model = YOLO(str(best if best.exists() else args.model))
        export_model.export(format="onnx", imgsz=int(args.imgsz), simplify=True, opset=12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
