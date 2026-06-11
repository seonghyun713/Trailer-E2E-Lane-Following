# Optimized Segmentation Models

Optimized model exports are not tracked in Git.

Expected runtime files:

```text
rover/trailer_task/optimized_models/
├── segformer_b0_yellow_line_512x288.onnx
├── segformer_b0_yellow_line_512x288_fp16.ts
└── ort_trt_cache/
```

The raw SegFormer checkpoint can be downloaded separately and exported with
`export_segformer_optimized.py`.
