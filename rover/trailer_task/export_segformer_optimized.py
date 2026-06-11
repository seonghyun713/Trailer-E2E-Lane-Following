#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from transformers import SegformerForSemanticSegmentation


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "track_riding" / "model_car_jetson" / "weights" / "segmentation" / "segformer_b0_yellow_line_best_model"
DEFAULT_OUTPUT_DIR = HERE / "optimized_models"


class SegFormerLogitsWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the lane SegFormer model to TorchScript and ONNX.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-torchscript", action="store_true")
    parser.add_argument("--no-onnx", action="store_true")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def export_torchscript(model_dir: Path, out_path: Path, width: int, height: int, device: torch.device) -> None:
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device).eval()
    if device.type == "cuda":
        model.half()
        dummy = torch.randn(1, 3, height, width, device=device, dtype=torch.float16)
    else:
        dummy = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)
    wrapper = SegFormerLogitsWrapper(model).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, dummy, strict=False)
        traced = torch.jit.freeze(traced)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"[torchscript] saved {out_path}")


def export_onnx(model_dir: Path, out_path: Path, width: int, height: int, device: torch.device, opset: int) -> None:
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device).eval()
    wrapper = SegFormerLogitsWrapper(model).eval()
    dummy = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=int(opset),
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(out_path))
    print(f"[onnx] saved {out_path}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True

    name = f"segformer_b0_yellow_line_{args.width}x{args.height}"
    if not args.no_torchscript:
        export_torchscript(args.model_dir, args.output_dir / f"{name}_fp16.ts", args.width, args.height, device)
    if not args.no_onnx:
        export_onnx(args.model_dir, args.output_dir / f"{name}.onnx", args.width, args.height, device, args.opset)


if __name__ == "__main__":
    main()
