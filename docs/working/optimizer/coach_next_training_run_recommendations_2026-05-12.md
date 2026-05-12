# Coach Next Training Run Recommendations

Date: 2026-05-12

Copy-paste handoff: For the overnight learning run, use the canonical two-seat
path with `fast_gray64_direct`, normal death, L4/T4, `batch_size=64`,
`num_simulations=8`, `collect_steps_per_iteration=64`,
`updates_per_iteration=4`, accumulated replay, learner sample size `256`,
sparse checkpoints, and normal CurvyZero checkpoint eval/GIF observability.
Also run a smaller `browser_lines` control if capacity permits. The fast mode
is a strong semantic visual approximation, not browser pixel fidelity: it keeps
trail/head positions, self/other contrast, bonus presence, and bonus type luma,
but drops connected line rasterization, sprite texture, antialiasing, and exact
downsample coverage. Use `profile_no_death` only for timing. Do not default to
B128, H100, or multi-GPU for the overnight run unless the B64 L4 lane is already
healthy and Coach wants an expensive scale probe.

## Current Truth

- Canonical Coach path:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
- This is current-policy two-seat self-play. One shared LightZero MuZero policy
  chooses both players from the same pre-step observation; CurvyTron advances
  once with the joint action; later learner updates mutate that same policy.
- Two-seat mode supports `--compute gpu-l4-t4` and `--compute gpu-h100-cpu40`.
  The old `gpu-l4-t4-cpu40` and `gpu-h100x2-cpu40` shapes are for other
  stock/control paths, not this canonical mode.
- Default/render-control mode is `two_seat_trail_render_mode=browser_lines`.
  `fast_gray64_direct` is the optimizer speed lane. It is acceptable for a
  speed-first overnight canary if labeled as approximation. `body_circles_fast`
  is only a historical/speed comparison and is not recommended.
- Dirty-render production path is the default optimizer recommendation. It is
  exact-pixel intended and falls back to full render on reset or unsupported
  cache state.
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
  scratch, copy/recolor, and dirty-block rendering have landed. Latest focused
  validation: ruff passed; `pytest tests/test_curvytron_two_seat_render_mode.py
  tests/test_vector_visual_observation.py
  tests/test_benchmark_render_lane_microbench.py -q` reported `60 passed`.
- Dirty render reuses previous RGB/gray frames and recomposes only dirty 11x11
  source blocks. Local CPU dynamic stack profiles show about `2.7x-4.3x` over
  full render in the tested B16/B32 long-trail cases.
- Fast direct luma mode passed focused render tests and gives the clearest
  wall-clock win. B64/L4 no-death full loop dropped from roughly `768s` with
  `browser_lines` to about `203s` with `fast_gray64_direct`; the visual stack
  bucket dropped from about `40s/iteration` to about `2s/iteration`.
- Fast mode keeps the main semantic visual facts, but it is not an Environment
  fidelity claim. Treat it as a speed-first training surface until Coach and
  Environment accept it.
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

- Plain read: this pre-dirty matrix says to scale B upward only while replay
  rows/sec improves and learner/search are not starved. B64 is next; B128 waits
  until render is no longer dominant. Sim32 at B64 was not proof that search is
  cheap in general.

## Run First

1. Profiling gate, not learning:
   L4/T4, `browser_lines`, `profile_no_death`, background eval/GIF off, no
   initial checkpoint, enough iterations for trails to get long. Compare
   `visual_stack_update_sec`, `policy_search_sec`, `env_step_sec`, learner time,
   elapsed wall time, action batching, and replay rows.
2. Overnight speed-first run:
   L4/T4, `fast_gray64_direct`, normal death, B64, sim8, collect64, updates4,
   accumulated replay, learner sample 256, sparse checkpoints.
3. Browser-lines control:
   L4/T4, `browser_lines`, normal death, B16 or B32, sim8, collect64, updates4,
   accumulated replay, learner sample 128 or 256. This is the closest current
   visual control, not the fastest lane.
4. Do not use B128, H100, or multi-GPU as the default overnight lane. B128 on
   L4 was worse per replay row in the fast profile; B128 on H100 is only a
   scale probe if there is spare budget and B64 looks healthy.

## Training Shape

Recommended overnight speed-first run:

```text
--mode two-seat-selfplay
--compute gpu-l4-t4
--batch-size 64
--max-train-iter long enough for several checkpoints
--num-simulations 8
--two-seat-collect-steps-per-iteration 64
--two-seat-updates-per-iteration 4
--two-seat-replay-scope accumulated
--two-seat-learner-sample-size 256
--two-seat-max-replay-rows 65536
--two-seat-death-mode normal
--two-seat-trail-render-mode fast_gray64_direct
--save-ckpt-after-iter 100 to 250
```

Recommended browser-lines control:

```text
--mode two-seat-selfplay
--compute gpu-l4-t4
--batch-size 16 or 32
--max-train-iter long enough for at least one checkpoint
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
  but they are not automatically better. In the measured fast profiles, B64/L4
  was the clean default. B128/L4 was slower per replay row; B128/H100 was closer
  but not enough to make H100 the default.
- More simulations increase search cost. While render dominates, sim16 is
  affordable in some profiles, but the overnight speed-first run should start
  at sim8 unless Coach explicitly wants stronger search.
- H100 is useful for quick scale sweeps after B64 is healthy. It is not required
  for the first overnight run.
- Multi-GPU is not a current two-seat recommendation. The canonical launcher
  does not expose H100x2 for two-seat mode, and the measured bottleneck is not
  yet a multi-GPU model throughput problem.

## Caveats For Coach

- Optimizer is making speed/setup recommendations only. Do not read these as
  claims about reward quality, learning progress, or policy strength.
- `fast_gray64_direct` is a strong semantic approximation, not exact browser
  fidelity. Keep the browser-lines control around so we can catch approximation
  damage.
- `profile_no_death` deliberately exaggerates long-trail render cost and hides
  normal reset/episode dynamics. Use it to expose Amdahl bottlenecks, then train
  with normal death.
- Compare training results to `iteration_0` with the same eval settings. Report
  trainer reward, sparse outcome, survival length, terminal causes, and action
  histograms separately.
- If early progress still shows mean episode length around 10-11 steps with no
  action collapse, that is a debugging signal, not a final learning verdict.
