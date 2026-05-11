# LightZero Live Eval Queue Helper - 2026-05-09

Use `scripts/lightzero_live_eval_queue.py` to poll the `curvyzero-runs` Modal
Volume for new official visual Pong `iteration_*.pth.tar` checkpoints and print
eval commands only for checkpoints that do not already have an output directory
under the chosen eval id.

Dry run:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --eval-id faithful-short-32768-live-gpu-stockeval-s0
```

The defaults are the live GPU stock-ish low pass: `--compute gpu-l4-t4`,
`--eval-pass low`, `--low-detail-max-eval-steps 512`, strict checkpoint load,
no model fallback, and stock evaluator enabled.

Execute the printed eval commands:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --eval-id faithful-short-32768-live-gpu-stockeval-s0 \
  --execute
```

Duplicate check: for `iteration_<N>.pth.tar`, the helper skips the checkpoint
when this directory already exists:

```text
training/lightzero-official-visual-pong/<run-id>/attempts/<attempt-id>/eval/<eval-id>/iteration_<N>_low_steps512_seed0
```

Use `--checkpoint-dir` or `--eval-root` only if a run used a nonstandard Volume
layout.
