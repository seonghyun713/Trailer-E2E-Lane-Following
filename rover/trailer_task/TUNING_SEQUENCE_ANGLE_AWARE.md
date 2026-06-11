# Angle-Aware Lane Controller Tuning Sequence

Current config:

- Main config: `dotted_lane_following_config.yaml`
- Angle calibration: `angle_calib_run_sign_filtered_20260607_223304`
- BEV calibration: `dual_bev_calibration.yaml`
- Effective angle anchors:
  - `cam0`: `-10, -15, -25, -35, -45, -50 deg`
  - `cam1`: `+10, +15, +25, +35, +45, +50 deg`

The new trailer-size calibration is reliable up to about `+-50 deg`.
Use `44 deg` as recovery entry and keep hard stop slightly above the
calibration edge so the controller can attempt recovery at the edge.

## 1. Verify Vision Before Motors

Run without motors first.

```bash
cd /home/ircv02/HYU-ECL3003

python3 rover/trailer_task/live_dotted_lane_following.py \
  --config rover/trailer_task/dotted_lane_following_config.yaml \
  --no-arm \
  --no-display \
  --http-stream \
  --http-host 0.0.0.0 \
  --http-port 8081
```

Check:

- BEV lane center is stable.
- `route_angle_deg` changes with real trailer angle.
- `route_corner_confidence` stays near zero on straight tiles.
- `route_corner_confidence` rises before a right-angle corner.

## 2. Sign Test

Tune these first:

```yaml
route_controller:
  lane_feedback_sign: 1.0
  alpha_feedback_sign: 1.0
  curve_feedback_sign: 1.0
  curvature_sign: 1.0
```

Expected behavior:

- If the rover moves away from the lane center, flip `lane_feedback_sign`.
- If trailer correction makes `abs(route_angle_deg)` grow, flip `alpha_feedback_sign`.
- If corner feedforward turns opposite to the visible corner, flip `curve_feedback_sign`.
- If `route_corner_confidence`/curvature sign is reversed before corners, flip `curvature_sign`.

Do not tune gains until the signs are correct.

## 3. Make The Rover Move Reliably

Tune motor-side parameters:

```yaml
rover:
  speed_gain: 1.10
  steer_mix: 1.50
  min_forward_speed: 0.12
  min_inner_wheel_ratio: 0.05
  min_abs_wheel_command: 0.08
```

Symptoms:

- Does not move under trailer load: increase `min_forward_speed` or `min_abs_wheel_command`.
- Does not turn enough: increase `steer_mix`.
- Inner wheel stops and rover stalls in turns: increase `min_inner_wheel_ratio`.
- Too twitchy: reduce `steer_mix` or `max_steer`.

## 4. Tune Straight Driving

Tune only on straight tiles first.

```yaml
route_controller:
  straight_alpha_gain: 0.011
  alpha_rate_gain: 0.0016
  straight_angle_lane_scale: 0.55
  straight_far_lookahead_y_ratio: 0.38
  lookahead_smoothing: 0.65
```

Symptoms:

- Trailer stays bent on straight: increase `straight_alpha_gain`.
- Rover ignores lane too much while correcting trailer: increase `straight_angle_lane_scale`.
- Rover follows lane but trailer becomes unstable: decrease `straight_angle_lane_scale`.
- Straight path oscillates: decrease `straight_alpha_gain`, decrease `heading_gain`, or increase `lookahead_smoothing`.
- Rover reacts too late: increase `straight_far_lookahead_y_ratio` toward `0.42-0.46`.

## 5. Tune Corner Detection

Tune before increasing corner steering force.

```yaml
route_controller:
  corner_enter_confidence: 0.45
  corner_exit_confidence: 0.22
  corner_shift_start_norm: 0.10
  corner_shift_full_norm: 0.28
```

Symptoms:

- Corner mode enters too late: lower `corner_enter_confidence` or `corner_shift_start_norm`.
- Straight is falsely detected as corner: raise `corner_enter_confidence` or `corner_shift_start_norm`.
- Corner mode exits too early: lower `corner_exit_confidence`.
- Corner mode stays too long: raise `corner_exit_confidence`.

## 6. Tune Cornering

```yaml
route_controller:
  corner_alpha_ref_deg: 20.0
  corner_lane_scale: 0.75
  corner_trailer_scale: 0.45
  curve_ff_gain: 0.30
  corner_speed_scale: 0.52
  corner_lookahead_y_ratio: 0.34
```

Symptoms:

- Rover cannot make the right-angle corner: increase `curve_ff_gain`, `corner_lane_scale`, or `rover.steer_mix`.
- Rover turns too aggressively: decrease `curve_ff_gain` or `corner_lane_scale`.
- Trailer folds too much in corner: decrease `corner_alpha_ref_deg` or increase `corner_trailer_scale`.
- Trailer correction blocks cornering: decrease `corner_trailer_scale`.
- Corner is too slow but stable: increase `corner_speed_scale`.
- Corner is unstable: decrease `corner_speed_scale`.

## 7. Tune Recovery And Safety

Current recovery range:

```yaml
route_controller:
  corner_allow_abs_angle_deg: 40.0
  recovery_start_abs_angle_deg: 44.0
  recovery_exit_abs_angle_deg: 35.0
  hard_stop_abs_angle_deg: 52.0
  recovery_alpha_gain: 0.018
  recovery_speed_scale: 0.35
```

Symptoms:

- Recovery starts too often during normal corners: increase `recovery_start_abs_angle_deg` slightly.
- Trailer reaches high angle too often: decrease `recovery_start_abs_angle_deg` or increase `recovery_alpha_gain`.
- Recovery is too harsh: reduce `recovery_alpha_gain`.
- Need a safer first test: lower `hard_stop_abs_angle_deg` to `50.0`.

## 8. Log Columns To Watch

Use the run CSV under `logs/dotted_lane_following_run/...`.

Important columns:

- `route_mode`
- `route_angle_deg`
- `route_alpha_ref_deg`
- `route_angle_band`
- `route_lookahead_y_ratio`
- `route_u_lane`
- `route_u_curve`
- `route_u_trailer`
- `route_u_total`
- `route_curvature_proxy`
- `route_corner_confidence`
