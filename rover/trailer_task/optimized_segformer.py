from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _resize_logits_to_mask(logits: np.ndarray, width: int, height: int) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 4:
        logits = logits[0]
    planes = [
        cv2.resize(logits[i], (width, height), interpolation=cv2.INTER_LINEAR)
        for i in range(int(logits.shape[0]))
    ]
    return np.argmax(np.stack(planes, axis=0), axis=0).astype(np.uint8)


def _fast_logits_to_mask(logits: np.ndarray, width: int, height: int) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 4:
        logits = logits[0]
    mask = np.argmax(logits, axis=0).astype(np.uint8)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


class OnnxSegFormerYellowLinePredictor:
    def __init__(
        self,
        onnx_path: Path,
        width: int,
        height: int,
        trt_cache_dir: Optional[Path] = None,
        prefer_tensorrt: bool = True,
        fast_postprocess: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.width = int(width)
        self.height = int(height)
        self.fast_postprocess = bool(fast_postprocess)
        self.session = self._make_session(ort, onnx_path, trt_cache_dir, prefer_tensorrt)
        self.input_name = self.session.get_inputs()[0].name
        self.providers = list(self.session.get_providers())

    def _make_session(self, ort: Any, onnx_path: Path, trt_cache_dir: Optional[Path], prefer_tensorrt: bool):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        available = set(ort.get_available_providers())
        providers: List[Any] = []
        if prefer_tensorrt and "TensorrtExecutionProvider" in available:
            cache_dir = trt_cache_dir or onnx_path.parent / "ort_trt_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": "True",
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": str(cache_dir),
                        "trt_timing_cache_enable": "True",
                        "trt_timing_cache_path": str(cache_dir),
                        "trt_max_workspace_size": str(1 << 30),
                    },
                )
            )
        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": "1",
                    },
                )
            )
        providers.append("CPUExecutionProvider")
        try:
            return ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
        except Exception:
            fallback = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available or p == "CPUExecutionProvider"]
            return ort.InferenceSession(str(onnx_path), sess_options=options, providers=fallback)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        original_h, original_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        tensor = np.ascontiguousarray((tensor - IMAGENET_MEAN) / IMAGENET_STD)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        if self.fast_postprocess:
            return _fast_logits_to_mask(logits, original_w, original_h)
        return _resize_logits_to_mask(logits, original_w, original_h)


class TorchScriptSegFormerYellowLinePredictor:
    def __init__(self, script_path: Path, device: Any, width: int, height: int, half: bool) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.use_half = bool(half and getattr(device, "type", "") == "cuda")
        self.model = torch.jit.load(str(script_path), map_location=device).eval()
        if self.use_half:
            self.model = self.model.half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        if self.use_half:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        original_h, original_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        if self.use_half:
            tensor = tensor.half()
        tensor = (tensor - self.mean) / self.std
        with torch.inference_mode():
            logits = self.model(tensor)
            logits = torch.nn.functional.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
            return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
