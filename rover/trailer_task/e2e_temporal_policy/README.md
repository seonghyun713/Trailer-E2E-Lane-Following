# E2E Temporal Policy

This folder trains a temporal motor policy from dotted-lane E2E logs without changing
the existing driving stack.

## Policy boundary

Inputs:

- BEV lane mask sequence, solid and dashed as two one-hot channels.
- Trailer angle scalar sequence: angle, angle rate, confidence, age, valid flag.
- Frame delta time.

Outputs:

- `wheel_left`, `wheel_right`.
- Auxiliary mode logits for `normal`, `straight`, `corner`, `pivot`, `bias`.

The rule/controller state is not used as an input. It is only used as an auxiliary
label, so the trained policy still learns from lane geometry plus trailer angle.

## Why this setup

- A 10-frame temporal window is about 2 seconds on the current logs. It covers
  typical pivot and post-corner bias intervals without making the model too large.
- The model is intentionally small because the current dataset has limited pivot
  samples. A large transformer-like model is more likely to memorize current runs.
- No left-right flip augmentation is used. The rover is physically asymmetric, so
  flipped samples would create false dynamics.
- Noise is mild and sensor-like: small angle noise, mask dropout, and tiny mask
  morphology.

## Smoke test

```bash
python3 e2e_temporal_policy/train_temporal_policy.py --smoke-test --device cpu
```

## Train

```bash
python3 e2e_temporal_policy/train_temporal_policy.py \
  --config e2e_temporal_policy/configs/temporal_gru.yaml
```

## Offline Inference

Run the best checkpoint on the validation split and write per-frame predictions:

```bash
python3 e2e_temporal_policy/infer_temporal_policy.py \
  --checkpoint e2e_temporal_policy/runs/temporal_gru_20260611_182905/best.pt \
  --split val \
  --output-csv inference_val_predictions.csv
```

Run on every current E2E log:

```bash
python3 e2e_temporal_policy/infer_temporal_policy.py \
  --checkpoint e2e_temporal_policy/runs/temporal_gru_20260611_182905/best.pt \
  --split all \
  --output-csv inference_all_predictions.csv
```

## Live Inference

The live E2E runner is separate from the original rule-based runner:

```bash
python3 live_e2e_temporal_policy.py --help
```

Shadow mode drives with the existing controller and logs E2E predictions:

```bash
python3 live_e2e_temporal_policy.py \
  --config dotted_lane_following_config.yaml \
  --e2e-mode shadow \
  --start-driving \
  --arm \
  --no-display \
  --no-stream \
  --no-http-stream
```

Direct E2E drive mode sends the model wheel outputs after warmup and safety clamps:

```bash
python3 live_e2e_temporal_policy.py \
  --config dotted_lane_following_config.yaml \
  --e2e-mode drive \
  --e2e-output-scale 1.20 \
  --e2e-max-abs-wheel 0.87 \
  --e2e-max-delta 0.30 \
  --start-driving \
  --arm \
  --no-display \
  --no-stream \
  --no-http-stream
```

By default, the live E2E runner disables `e2e_dataset` writing to avoid saving
model outputs as expert labels. Add `--save-e2e-dataset` only when that is
intentional.

Outputs are written to:

```text
e2e_temporal_policy/runs/temporal_gru/
```

Important files:

- `best.pt`: best validation checkpoint.
- `last.pt`: last checkpoint.
- `metrics.jsonl`: per-epoch metrics.
- `dataset_info.json`: train/validation runs and mode counts.
- `resolved_config.yaml`: exact config used for the run.

For real evaluation, keep validation split at the run level. Do not randomly split
individual frames, because adjacent temporal samples are highly correlated.

The trainer also estimates each run's turn direction from motor commands and keeps
single available left/right runs in train. If only one direction has a single run,
validation for that direction is not meaningful until another run is collected.
