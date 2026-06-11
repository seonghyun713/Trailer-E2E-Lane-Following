# Blue Panel BBox Workflow

This path is for the new trailer marker plan:

- blue background panel
- white text
- detect only the panel bbox
- use bbox width, plus optional camera/center terms, to estimate trailer angle

## 1. Capture Images

Default mode saves fixed side-mirror ROI crops. This keeps the dataset small and
makes the detector much easier to train.

```bash
python3 trailer_task/capture_bbox_frames_gst.py \
  --frame-step 10 \
  --max-images-per-camera 200 \
  --save-mode roi
```

Useful variants:

```bash
# Capture only one side.
python3 trailer_task/capture_bbox_frames_gst.py --single-camera left

# Save full frame and ROI crop together.
python3 trailer_task/capture_bbox_frames_gst.py --save-mode both

# If edge red/magenta cast hurts the blue panel color.
python3 trailer_task/capture_bbox_frames_gst.py

# Edge color correction is always on by default. Use this only for debugging.
python3 trailer_task/capture_bbox_frames_gst.py --no-edge-color-fix
```

Images are written under:

```text
trailer_task/bbox_dataset/images/
```

Capture metadata is written to:

```text
trailer_task/bbox_dataset/captures.csv
```

## 2. Label BBoxes

```bash
python3 trailer_task/label_bbox_tool.py
```

Controls:

- drag: draw/replace bbox
- `s`: save
- `Enter`: save and next
- `n` / `p`: next / previous
- `x`: save as no marker
- `d`: delete current bbox
- `Esc`: quit

Labels are saved to:

```text
trailer_task/bbox_dataset/labels.csv
```

Each label stores original image coordinates and `width_px`, so the angle
calibration step can fit per-camera width-to-angle curves later.

## Notes For Angle Estimation

Using bbox width alone is fast, but width only measures apparent scale. It is
stable when trailer distance is controlled. For parking and reversing, use at
least these terms in the final calibration:

- `camera_key`
- `width_px / image_width`
- `center_x / image_width`
- optional left/right camera fusion

That keeps the runtime detector tiny while avoiding the worst ambiguity from
distance and mirror perspective.

## 3. Train A Lightweight YOLO Detector

Convert the current `labels.csv` into Ultralytics YOLO format:

```bash
python3 trailer_task/prepare_yolo_dataset.py \
  --dataset trailer_task/bbox_dataset_mirror_run_YYYYMMDD_HHMMSS
```

Train a small one-class detector:

```bash
python3 trailer_task/train_yolo_light.py \
  --dataset trailer_task/bbox_dataset_mirror_run_YYYYMMDD_HHMMSS \
  --model yolo11n.pt \
  --imgsz 320 \
  --epochs 80 \
  --batch 16 \
  --device 0
```

Use `--batch 8` if the Jetson runs out of memory. Use `--model yolov8n.pt` if
you want the older nano baseline.
