# LightZero Pong Post-Seed-Fix Run Plan - 2026-05-09

Role: post-fix run planner only. No pytest and no implementation work.

## Current Status

This plan is superseded by the deeper-seed-fix run recorded in
`docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md`.
Do not use this file as the current next step.

Latest true state:

- The deeper seed fix passed the modest train-side seed diversity gate:
  131 unique seeds, top seed 2/148 rows, and
  `seed_dominance_warning=false`.
- Strict independent MCTS checkpoint loading passed.
- Paired MCTS still failed checkpoint quality: random `27-37`, lagged
  `27-33`, track `0-59`, learned actions `[7285,4781,0]`.
- Player0-only MCTS did not rescue it: random `14-18`, lagged `16-15`, track
  `0-29`, learned actions `[3353,2073,0]`.
- Current blocker is policy learning/control signal, not seed dominance,
  checkpoint loading, horizon mismatch, or seat pairing.

## Goal

Run one modest CPU Modal training attempt after the seed-handling fix, then
decide whether varied seeds make the LightZero dummy Pong telemetry and the
independent scorecard more honest.

This is not a scale-up run. It is a trust-boundary check.

## Recommended Train Run

Use the existing scaled train wrapper, but keep it smaller than the failed
4096/64 read:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --opponent-policy random_uniform \
  --max-env-step 1024 \
  --max-train-iter 16 \
  --num-simulations 8 \
  --batch-size 32 \
  --update-per-collect 1 \
  --n-evaluator-episode 8 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --seed 3
```

CPU is fine. The 1024/16 cap is large enough to expose repeated-seed or
constant-action behavior, while staying cheap enough to discard. Do not jump
back to 4096/64 unless this run shows seed diversity and a non-degenerate
independent scorecard.

## Required Independent Read

After training, score the final/latest checkpoint, expected
`iteration_16.pth.tar`, through the independent MCTS/eval-mode scorecard:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt \
  --checkpoints lightzero:iter16=ref:training/lightzero-dummy-pong/<RUN_ID>/checkpoints/lightzero/iteration_16.pth.tar \
  --episodes 32 \
  --seed 1701 \
  --split-id dummy_pong_post_seed_fix_heldout_v0 \
  --split-role heldout \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --eval-id mcts-scoreboard-post-seed-fix-1024x16-iter16 \
  --max-env-step 1024 \
  --num-simulations 8
```

Required opponents: `random_uniform`, `lagged_track_ball_1`, and `track_ball`.
If the wrapper default ladder changes, force or patch the invocation before
trusting the result; the pass/fail read depends on all three rows.

## Pass/Fail Read

Pass only if all of these are true:

- Held-out independent MCTS scorecard completes against `random_uniform`,
  `lagged_track_ball_1`, and `track_ball`, using the same 1024 horizon.
- Trainer-side `episodes.jsonl` seed histogram is not dominated by one seed.
  Practical threshold: no single `episode_seed` should account for more than
  half the rows unless the row source is explicitly a fixed-seed debug stream
  excluded from claims.
- Independent scorecard action histogram eventually includes both directions:
  `up` and `down` must both appear for the LightZero policy. `stay` alone is
  not enough. If `down` remains zero, call the run a behavioral fail even if
  random win rate is noisy-positive.
- Scorecard vs `random_uniform` is not materially worse than random and is
  reported with wins/losses, truncations, survival stats, shaped return, score
  return, and action histogram. Do not use win rate alone.

Fail if any of these occur:

- Trainer telemetry is again dominated by one seed.
- Independent MCTS still never chooses one direction.
- The held-out scorecard cannot load the checkpoint strictly from the intended
  state dict.
- Trainer-side wins look good but held-out random/lagged/track rows remain
  degenerate or worse than baseline.

## Telemetry Claim Boundaries

Trainer-side telemetry may claim:

- LightZero training ran on the custom dummy Pong env with the recorded config.
- The trainer collected/evaluated episodes with the recorded seed distribution.
- Env-side rows show in-process behavior during training/eval, useful for
  debugging seed diversity, action usage, rewards, survival, and checkpoint
  creation.

Trainer-side telemetry may not claim:

- Final checkpoint quality.
- Held-out generalization.
- Improvement over `random_uniform`, `lagged_track_ball_1`, or `track_ball`.
- That a high win count is meaningful if rows are dominated by one seed or by
  collector/evaluator streams mixed together.
- That shaped loss-delay return is the true environment reward.

The independent MCTS scorecard is the authority for checkpoint quality. Trainer
rows are supporting diagnostics.

## Checkpoint State Dict Choice

Use checkpoint key `model` for the main scorecard. Current finding says
LightZero eval-mode uses the main model path, and prior scorecards have been
loading `model`.

`target_model` is optional as a small control only. If run, label it explicitly
as `target_model_control`, score only a tiny random-opponent check unless it
surprises us, and do not let it replace the main `model` read.

## Stop Rule

The original run governed by this plan found a seed-plumbing issue. The later
deeper-seed-fix run passed seed diversity but still failed independent MCTS
quality. Stop scaling LightZero Pong in this shape. The next work should
diagnose policy learning/control signal, observation/action wiring, and
objective signal, not another seed-config pass or a larger Modal train.
