# Coach Next Training Run Recommendations

Date: 2026-05-12

Copy-paste handoff: After the renderer changes land, do not start with a long
learning run. First run the canonical two-seat self-play path in profiling
shape on L4/T4 with `browser_lines`, background eval/GIF disabled, and
`profile_no_death` to confirm the new render/cache cost. Optimizer launched a
wait-mode profile matrix named `opt-render-cache-wait-*`; use those results
before choosing the next real run. Future background matrices should use
`modal run --detach`. If the sweep shows render no longer dominates, start the
next real Coach run on normal death mode with L4/T4, `batch_size=32` or `64`,
`collect_steps_per_iteration=64`, `updates_per_iteration=4`,
`num_simulations=8` or `16`, sparse checkpoints, and background checkpoint
eval/GIF enabled only at checkpoint cadence. Treat H100, batch 128, sim32, and
multi-GPU as follow-up scaling experiments, not defaults.

## Current Truth

- Canonical Coach path:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
- This is current-policy two-seat self-play. One shared LightZero MuZero policy
  chooses both players from the same pre-step observation; CurvyTron advances
  once with the joint action; later learner updates mutate that same policy.
- Two-seat mode supports `--compute gpu-l4-t4` and `--compute gpu-h100-cpu40`.
  The old `gpu-l4-t4-cpu40` and `gpu-h100x2-cpu40` shapes are for other
  stock/control paths, not this canonical mode.
- Default training render is `two_seat_trail_render_mode=browser_lines`.
  `body_circles_fast` is only a speed comparison unless Environment changes the
  training surface.
- `profile_no_death` is optimizer timing only. Real training should use normal
  death/reset semantics.

## Evidence So Far

- Fixed-opponent stock-control profiles found useful single-container width up
  to about c128, but those are controls, not direct two-seat learning facts.
- Active two-seat no-death profiles before the latest cache work were
  render-bound: `browser_lines` spent about `191.6s` in visual stack update out
  of `219.4s` elapsed for a small L4 run. Named timers put visual stack near
  `95%` of the measured hot-loop time.
- H100 did not solve the pre-cache render bottleneck. That points to CPU render
  and Python/NumPy orchestration, not raw GPU model throughput, as the current
  Amdahl limit.
- Perspective reuse and a conservative browser-line trail cache have landed or
  are landing in the renderer lane, but the end-to-end two-seat profile evidence
  is still pending.
- The first non-wait `opt-render-cache-*` launch printed function-call IDs but
  did not produce attempt/progress files after the local app exited. Treat that
  as a bad profile launch pattern, not speed evidence.
- Current wait-mode profile run IDs from the main thread:
  `opt-render-cache-wait-l4-b16-sim16-20260512a`,
  `opt-render-cache-wait-l4-b64-sim16-20260512a`,
  `opt-render-cache-wait-l4-b128-sim16-20260512a`,
  `opt-render-cache-wait-h100-b128-sim16-20260512a`,
  `opt-render-cache-wait-l4-b64-sim32-20260512a`.
  Treat them as pending until summaries land.

## Run First

1. Profiling gate, not learning:
   L4/T4, `browser_lines`, `profile_no_death`, background eval/GIF off, no
   initial checkpoint, enough iterations for trails to get long. Compare
   `visual_stack_update_sec`, `policy_search_sec`, `env_step_sec`, learner time,
   elapsed wall time, action batching, and replay rows.
2. If the gate still shows render above about `70%` of hot-loop time, keep the
   first real run conservative: `batch_size=16-32`, `num_simulations=8`,
   `collect_steps_per_iteration=64`, `updates_per_iteration=4`.
3. If the cache sweep shows render is no longer dominant, use
   `batch_size=64`, `num_simulations=16`, `collect_steps_per_iteration=64`,
   `updates_per_iteration=4`, `learner_sample_size=128-256`.
4. Only try `batch_size=128`, `num_simulations=32`, or H100 after the matching
   pending profiles prove the extra rows/search are not just multiplying render
   time.

## Training Shape

Recommended first real run after the profiling gate:

```text
--mode two-seat-selfplay
--compute gpu-l4-t4
--batch-size 32 or 64
--max-train-iter long enough for several checkpoints
--num-simulations 8 or 16
--two-seat-collect-steps-per-iteration 64
--two-seat-updates-per-iteration 4
--two-seat-replay-scope accumulated
--two-seat-learner-sample-size 128 or 256
--two-seat-max-replay-rows 65536
--two-seat-death-mode normal
--two-seat-trail-render-mode browser_lines
--save-ckpt-after-iter 100 to 250
```

Keep progress writes frequent enough to see early movement. Keep stock
LightZero in-loop eval off; it is separate from CurvyZero checkpoint survival
eval/inspection/GIF.

## Cadence

- Profiling: use `--no-background-eval-enabled` and
  `--no-background-gif-enabled`. This also suppresses the GIF browser marker in
  current code. Avoid initial checkpoints unless checking artifact plumbing.
- First real training: allow CurvyZero checkpoint eval/inspection/GIF, but make
  checkpoint cadence sparse enough that artifacts do not become the run. Default
  `100` is okay for a canary; use `250` or `500` for longer runs once stable.
- GIFs are observability, not speed evidence. If the goal is wall-clock
  profiling, turn them off.

## Scaling Read

- Larger batches increase self-play volume and search batching opportunities,
  but they also multiply visual stack updates. Until the cache profiles close,
  batch 128 is a hypothesis, not a default.
- More simulations increase search cost roughly linearly. While render dominates,
  sim16 may be affordable; sim32 is only justified after the pending sim32
  profile.
- H100 is useful for quick sweeps if search/model becomes a larger share. It is
  not required for the first real run if render remains CPU-bound.
- Multi-GPU is not a current two-seat recommendation. The canonical launcher
  does not expose H100x2 for two-seat mode, and the measured bottleneck is not
  yet a multi-GPU model throughput problem.

## Caveats For Coach

- Optimizer is making speed/setup recommendations only. Do not read these as
  claims about reward quality, learning progress, or policy strength.
- `profile_no_death` deliberately exaggerates long-trail render cost and hides
  normal reset/episode dynamics. Use it to expose Amdahl bottlenecks, then train
  with normal death.
- Compare training results to `iteration_0` with the same eval settings. Report
  trainer reward, sparse outcome, survival length, terminal causes, and action
  histograms separately.
- If early progress still shows mean episode length around 10-11 steps with no
  action collapse, that is a debugging signal, not a final learning verdict.
