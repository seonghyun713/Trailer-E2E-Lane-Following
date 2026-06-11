# Trailer Panel Detector Weights

YOLO/TensorRT detector artifacts are not tracked in Git.

Expected runtime path from `dotted_lane_following_config.yaml`:

```text
rover/trailer_task/runs_yolo/trailer_panel_yolo11n/weights/best.engine
```

Recommended release assets:

- `best.pt` for portable PyTorch inference and export.
- `best.engine` for the Jetson TensorRT runtime used during the demo.

TensorRT engine files can be hardware and software-version dependent, so keep a
portable `.pt` checkpoint available when possible.
