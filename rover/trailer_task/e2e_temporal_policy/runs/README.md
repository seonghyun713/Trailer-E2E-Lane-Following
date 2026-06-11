# Temporal Policy Checkpoints

Training outputs and checkpoints are not tracked in Git.

Expected live policy checkpoint:

```text
rover/trailer_task/e2e_temporal_policy/runs/
└── temporal_gru_20260611_182905/
    └── best.pt
```

Attach `best.pt` to a GitHub Release or store it with Git LFS. The README reports
the current offline validation numbers so the repository stays understandable
even before the checkpoint is downloaded.
