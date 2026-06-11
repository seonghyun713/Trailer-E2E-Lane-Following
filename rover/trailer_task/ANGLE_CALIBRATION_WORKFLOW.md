# Angle Calibration Workflow

Detection labels only teach the model where the trailer panel is. Angle
calibration is a separate supervised regression dataset:

```text
YOLO bbox features + camera_key -> true hinge angle_deg
```

## Angle Set

Recommended first pass:

```text
-45, -35, -25, -15, -8, 0, +8, +15, +25, +35, +45
```

For each angle, collect 20-30 frames per camera. This gives roughly 440-660
calibration rows for both mirrors.

## Hardware Rules

- Define `0 deg` first: truck and trailer centerlines straight.
- Keep the sign convention fixed for the whole dataset.
- Approach each target angle from the same direction when possible to reduce
  hinge/backlash error.
- Wait 1-2 seconds after touching the trailer before capturing.
- Do not move the cameras, mirrors, printed panel, or ROI after collecting the
  calibration dataset.
- If possible, repeat the whole angle list once in the opposite approach
  direction and keep both passes. That exposes mechanical hysteresis.

## Collection Command

Run from the rover root:

```bash
cd /home/ircv02/HYU-ECL3003/rover

python3 trailer_task/collect_angle_calibration_gst.py \
  --angles "-45,-35,-25,-15,-8,0,8,15,25,35,45" \
  --samples-per-camera 24 \
  --frame-step 4 \
  --weights trailer_task/runs_yolo/trailer_panel_yolo11n/weights/best.pt
```

The script pauses before each angle. Set the hinge angle with the physical
angle gauge, then press Enter.

Output example:

```text
trailer_task/angle_calib_run_YYYYMMDD_HHMMSS/
  angle_calibration.csv
  images/
```

`angle_calibration.csv` stores:

- `angle_deg`: true hinge angle from the physical gauge
- `camera_key`: `cam1` or `cam0`
- ROI image path
- YOLO detection confidence
- bbox pixel coordinates
- normalized bbox features for regression

## What To Watch

For a good calibration dataset:

- Every angle should have rows from both cameras.
- Most rows should have `detected=1`.
- Detection confidence should usually be above `0.5`.
- Avoid collecting only one continuous motion sequence; hold each target angle
  still while capturing.

Detection can look perfect but angle regression can still be bad if all samples
come from one distance, one lighting condition, or one direction of approach.
