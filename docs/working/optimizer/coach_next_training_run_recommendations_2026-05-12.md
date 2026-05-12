# Coach Next Training Run Recommendations

Date: 2026-05-12

Copy-paste handoff: Current canonical two-seat profiles are still render-bound.
Use the canonical path, but do not expect bigger GPUs or B128 to fix speed yet.
If Coach must start a real run before the next render pass, use L4/T4,
`browser_lines`, normal death, `batch_size=32` or `64`,
`collect_steps_per_iteration=64`, `updates_per_iteration=4`,
`num_simulations=8` or `16`, sparse checkpoints, and checkpoint eval/GIF only at
checkpoint cadence. B64 gives more replay rows per wall clock than B16, but the
gain is modest because render dominates. Do not default to B128/H100/multi-GPU
until dirty-block render work lowers visual time. Future background matrices
should use `modal run --detach`.

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
- Perspective reuse, a conservative browser-line trail cache, exact downsample
  scratch, and a copy/recolor fast path have landed and passed focused tests.
- The first non-wait `opt-render-cache-*` launch printed function-call IDs but
  did not produce attempt/progress files after the local app exited. Treat that
  as a bad profile launch pattern, not speed evidence.
- Current wait-mode matrix summary:

```text
run                                      B    sim  wall     visual   search  replay_rows
opt-render-cache-wait-l4-b16-sim16      16   16   198.6s   136.2s   14.0s   2356
opt-render-cache-wait-l4-b64-sim16      64   16   559.4s   494.9s   21.2s   7957
opt-render-cache-wait-l4-b64-sim32      64   32   562.1s   493.5s   28.4s   7922
opt-render-cache-wait-l4-b128-sim16     128  16   1112.7s  990.8s   61.0s   15623
opt-render-cache-wait-h100-b128-sim16   128  16   978.9s   853.9s   57.8s   15611
```

- Plain read: B64 and B128 produce more replay rows per iteration, but render
  grows enough that replay-row throughput only improves modestly. Sim32 at B64
  is not much slower than sim16 because render is already dominating; it is not
  proof that search is cheap in general. H100 helps B128 a bit, but the loop is
  still CPU-render-bound.

## Run First

1. Profiling gate, not learning:
   L4/T4, `browser_lines`, `profile_no_death`, background eval/GIF off, no
   initial checkpoint, enough iterations for trails to get long. Compare
   `visual_stack_update_sec`, `policy_search_sec`, `env_step_sec`, learner time,
   elapsed wall time, action batching, and replay rows.
2. If the gate still shows render above about `70%` of hot-loop time, keep the
   first real run conservative: `batch_size=32`, `num_simulations=8`,
   `collect_steps_per_iteration=64`, `updates_per_iteration=4`.
3. Use `batch_size=64`, `num_simulations=16` only if Coach wants a data-volume
   trade-off and accepts slower iterations for a modest replay-row throughput
   gain.
4. Do not use `batch_size=128`, H100, or multi-GPU as the first real training
   default. Revisit after dirty-block render work lowers visual time.

## Training Shape

Recommended first real run after the profiling gate:

```text
--mode two-seat-selfplay
--compute gpu-l4-t4
--batch-size 32
--max-train-iter long enough for several checkpoints
--num-simulations 8
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
  but they also multiply visual stack updates. In the current matrix B64 is a
  modest throughput trade-off, B128 is too render-heavy for a default.
- More simulations increase search cost. While render dominates, sim16 is
  affordable at B64, but the next real run can start at sim8 unless Coach wants
  stronger search immediately.
- H100 is useful for quick sweeps if search/model becomes a larger share. It is
  not required for the first real run while render remains CPU-bound.
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
