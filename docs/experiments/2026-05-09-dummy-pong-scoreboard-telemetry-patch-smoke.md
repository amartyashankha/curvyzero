# Dummy Pong Scoreboard Telemetry Patch Smoke - 2026-05-09

## Question

Can the Pong checkpoint scoreboard report useful progress signals when wins are
still zero?

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/run_dummy_pong_checkpoint_scoreboard.py
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 1 \
  --seed 0 \
  --split-id telemetry_patch_smoke \
  --split-role smoke \
  --checkpoint smoke=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-scoreboard-telemetry-patch-smoke-2026-05-09
```

No pytest was run.

## Result

The compile and scoreboard smoke passed.

The scoreboard now includes these fields in matchup and scoreboard rows:

- `truncation_rate`;
- `median_steps`, `p90_steps`, `std_steps`;
- `survival_steps`;
- `score_return_stats_by_policy`;
- `mean_shaped_loss_delay_return_by_policy`;
- `shaped_loss_delay_return_stats_by_policy`.

Example useful zero-win row from the smoke:

```text
learned_smoke_vs_track_ball:
  wins_by_policy: learned_smoke 0, track_ball 0
  truncation_rate: 1.0
  mean_steps: 120.0
  p90_steps: 120.0
  mean_shaped_loss_delay_return_by_policy:
    learned_smoke: 0.0
    track_ball: 0.0
```

This means a future `0/N wins` result is no longer a dead read. We can see
whether the policy is losing quickly, surviving longer, forcing timeouts, or
moving the shaped loss-delay signal.

## Interpretation

This is telemetry plumbing, not MuZero progress. It fixes the repeated process
mistake where Pong evals collapsed to only wins/losses.

The next real training claim is still LightZero custom-env MuZero on dummy Pong
as a Modal whole-job.
