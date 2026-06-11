# Trailer Parking Algorithm

This project uses two front CSI cameras and two side mirrors. The cameras see the mirror ROIs, and the mirror ROIs see the trailer side panel when the hinge angle becomes large enough.

## Current Signal

The trained YOLO model detects the blue `AI` panel on the trailer side:

```text
runs_yolo/trailer_panel_yolo11n/weights/best.pt
```

The first production estimator should use the bbox center position inside the mirror ROI as the main signal. The calibration CSV already shows a strong monotonic relation:

- `cam0`: negative trailer angles, reliable around `-62` to `-15` deg.
- `cam1`: positive trailer angles, reliable around `+15` to `+58` deg.
- Around `-15` to `+15` deg, the panel is often invisible. This is a visual dead zone, not a model bug.

Because of that dead zone, the controller must not treat "no detection" as a large angle. It should:

1. hold the last valid angle for a short time,
2. classify no-detection near straight as `near_zero_deadband`,
3. stop if a large last angle becomes stale.

That behavior is implemented in `AngleStateFilter`.

## Runtime Pipeline

1. Read both CSI cameras.
2. Apply the same mirror ROI and color correction used during data collection.
3. Run YOLO only on each mirror ROI.
4. Convert the best bbox to normalized features.
5. Estimate trailer hinge angle from the calibration table.
6. Smooth/fuse the left and right mirror estimates.
7. Run the reverse parking state machine.
8. Mix signed speed and left/right PWM differential into skid-steer rover wheel commands.
9. Log every frame for tuning.

## Parking Controller

The initial controller is intentionally scripted because there is not yet a measured parking-slot pose.

States:

```text
IDLE -> BREAK_ANGLE -> HOLD_ANGLE -> CHASE_TRAILER -> STRAIGHTEN -> STOPPED
```

For a right-side parking maneuver, the controller targets a negative trailer angle first. For left-side parking, it targets a positive angle. If the left/right PWM differential acts in the opposite direction on the real rover, tune `parking.reverse_steer_sign` in `trailer_parking_config.yaml`.

## Box Detection Recommendation

Yes, train a separate detector for the two parking boxes if the final task requires reliable closed-loop parking between them.

Without box detection, the system can only do a scripted reverse maneuver from a fixed start pose. That can work in a controlled demo, but it will be sensitive to:

- starting position,
- trailer load and tire slip,
- box spacing,
- floor friction,
- camera/mirror alignment.

With box detection, the parking module can add:

- slot center alignment,
- distance/scale estimate from box bbox size,
- stop condition when the trailer reaches the slot,
- recovery if the rover starts slightly off.

The runtime already has a disabled `models.parking_box` hook. Once a box model is trained, set:

```yaml
models:
  parking_box:
    enabled: true
    weights: runs_yolo/parking_box_yolo/weights/best.pt
```

Then enable stronger checks such as:

```yaml
parking:
  require_box_for_start: true
  require_box_during_parking: true
```
