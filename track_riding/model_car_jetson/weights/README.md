# Shared Segmentation Weights

The trailer runtime can reuse the yellow-line SegFormer checkpoint from the
model-car project. Model binaries are not tracked in Git.

Expected path:

```text
track_riding/model_car_jetson/weights/segmentation/
└── segformer_b0_yellow_line_best_model/
    ├── config.json
    └── model.safetensors
```

Optimized ONNX/TorchScript/TensorRT exports can be generated from this model or
downloaded from a GitHub Release.
