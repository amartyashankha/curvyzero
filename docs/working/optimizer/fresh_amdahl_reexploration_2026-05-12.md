# Fresh Amdahl Re-Exploration

Date: 2026-05-12

Purpose: rerun a small, isolated profile matrix after the stock-vs-custom speed
panic was resolved. This is Optimizer evidence only, not a learning claim.

## Postmortem Scope Correction

These measurements are for the custom `--mode two-seat-selfplay` adapter unless
explicitly stated otherwise. After the Coach architecture reset, this path is
not the trusted learning lane. Treat the numbers below as postmortem/profiling
evidence for the custom adapter, not as recommended Coach training settings.

Current trusted lane: stock LightZero `train_muzero` with
`source_state_fixed_opponent`.

## Questions

1. In the current `fast_gray64_direct` training-like path, is the main wall time
   search, render, observation noise, autoreset/env, learner, or artifacts?
2. In long-survival `browser_lines`, is rendering still the main wall time?
3. Does a bigger search budget or bigger GPU change the fast-direct bottleneck?
4. How much wall is named work versus host/artifact overhead?

## First Fresh Matrix

All runs use:

```text
--mode two-seat-selfplay
--wait-for-train
background eval/GIF off
no initial checkpoint
sparse checkpoint/progress commit cadence
```

Cells:

| Cell | Shape | What It Tests |
| --- | --- | --- |
| fast-main | L4/T4, B64, sim8, normal death, fast-direct, learner on | current main Amdahl shape |
| fast-no-noise | same, obs noise 0 | how much noise costs now |
| fast-sim16 | L4/T4, B64, sim16 | search scaling |
| h100-b128 | H100, B128, sim8 | whether bigger batch/GPU helps |
| browser-nodeath | L4/T4, B8, sim2, no-death, browser-lines, collect-only | rich-render long-survival sentinel |
| fast-nodeath | matched no-death fast-direct sentinel | isolate render effect from search |

## Interpretation Rule

Use replay/policy rows and named timing buckets, not raw `steps/s`, when
comparing custom two-seat profiles. For no-death sentinels, focus on
`visual_stack_update_sec` versus `policy_search_sec`. For main fast-direct
profiles, focus on search sub-buckets, noise, autoreset/env, learner timings,
and progress/checkpoint overhead.

## Fresh Result

Summarized from saved Modal artifacts after warm-up-heavy first iterations were
included. Use this as direction, not as a final steady-state benchmark.

| Cell | Shape | Wall | Collect | Search | Policy Forward | Visual | Noise | Env + Autoreset | Learner | Commit/Ckpt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast-main | L4/T4, B64, sim8, normal, fast-direct, learner on | 58.69s | 36.13s | 14.87s | 13.40s | 4.16s | 9.52s | 4.38s | 2.79s | 5.99s |
| fast-no-noise | L4/T4, B64, sim8, obs noise 0 | 45.61s | 28.24s | 15.46s | 14.15s | 4.24s | 0.00s | 4.60s | 2.55s | 3.06s |
| fast-sim16 | L4/T4, B64, sim16, fast-direct | 51.62s | 29.99s | 14.96s | 14.15s | 2.81s | 6.57s | 3.42s | 2.28s | 6.28s |
| h100-b128 | H100, B128, sim8, fast-direct | 62.49s | 42.11s | 12.68s | 11.04s | 3.62s | 14.94s | 5.49s | 2.10s | 6.17s |
| browser-nodeath | L4/T4, B8, sim2, no-death, browser-lines, collect only | 58.24s | 47.61s | 9.01s | 8.35s | 31.37s | 3.82s | 4.38s | - | 0.04s |
| fast-nodeath | L4/T4, B8, sim2, no-death, fast-direct, collect only | 33.31s | 21.69s | 9.80s | 9.06s | 2.59s | 4.98s | 5.77s | - | 0.04s |

Notes:

- `policy_forward` is nested inside `search`; do not add it to `search`.
- `noise` is `observation_noise_sec + replay_observation_noise_sec`.
- `env + autoreset` is `env_step_sec + initial_autoreset_sec + loop_autoreset_sec`.
- `commit/ckpt` is progress commit plus checkpoint save; profiling commands
  should keep this sparse.

## Current Read

For the current `fast_gray64_direct` main training surface, rendering is not
the main bottleneck. The largest useful buckets are:

1. policy/search, mostly LightZero collect forward;
2. CPU-side observation noise when it is enabled;
3. reset/env/bookkeeping;
4. learner and artifacts, now smaller but still visible.

For rich long-survival `browser_lines`, rendering is still the main bottleneck.
The matched no-death pair says the same search workload spent `31.37s` in rich
rendering versus `2.59s` in fast-direct rendering. That is a separate render
optimization lane and a fidelity sentinel, not the current fast-direct main
training bottleneck.

## Next Experiments

Run these only as isolated Optimizer profiles, not by mutating live Coach runs:

| Cell | Purpose |
| --- | --- |
| collect-only B64 sim8 with default noise | clean denominator without learner/checkpoint noise |
| collect-only B64 sim8 with noise 0 | exact Amdahl ceiling for CPU observation noise |
| B64 sim16 and sim32 | search scaling after warm-up |
| B32/B64/B128 on L4/T4 | replay rows/sec and search rows/sec scaling |
| B128 sim8/sim16 on H100 | whether the bigger GPU pays when batch/search is large |
| matched browser-lines/fast-direct no-death sentinel | keep rich-render cost and approximation gap visible |

If the next isolated runs show fast-direct search dominates, inspect the
LightZero collect/MCTS call path and batching. If CPU noise dominates after
larger batches, optimize or move that augmentation. If browser-lines is required
for a main training run, return to render optimization first.

## Wave 2 Plan

Purpose: isolate search/batch/GPU scaling on the current fast-direct surface.
These are not learning runs.

Common controls:

```text
--mode two-seat-selfplay
--wait-for-train
--two-seat-trail-render-mode fast_gray64_direct
--two-seat-death-mode normal
--two-seat-collect-steps-per-iteration 64
--two-seat-updates-per-iteration 0
--no-two-seat-save-initial-checkpoint
--two-seat-progress-commit-every-iterations 1000
--no-background-eval-enabled
--no-background-gif-enabled
```

Cells:

| Attempt | Compute | B | Sims | Iterations | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| b32-sim8-l4 | L4/T4 | 32 | 8 | 12 | lower-width throughput canary |
| b64-sim8-l4 | L4/T4 | 64 | 8 | 12 | main denominator |
| b64-sim8-l4-noise0 | L4/T4 | 64 | 8 | 12 | CPU noise Amdahl bound |
| b64-sim16-l4 | L4/T4 | 64 | 16 | 12 | first search-budget scaling point |
| b64-sim32-l4 | L4/T4 | 64 | 32 | 8 | search stress at sane width |
| b128-sim8-l4 | L4/T4 | 128 | 8 | 8 | test wider batch on cheaper GPU |
| b128-sim8-h100 | H100 | 128 | 8 | 8 | paired hardware check |
| b128-sim16-h100 | H100 | 128 | 16 | 8 | best chance for H100/search to matter |

Decision rule: use replay rows/sec, policy rows/sec, and named buckets. Ignore
raw iteration count. If B128/H100 does not materially beat B64/L4 by replay rows
per second or search rows per second, the simple scale-out path is many B64 L4
runs, not bigger single containers.

## Bucket Notes

Current two-seat timing buckets are useful but still a little coarse:

- `policy_search_sec` is the outer fresh-row action-selection block. It includes
  tensor prep, `policy.collect_mode.forward(...)`, output decode, search-record
  construction, legal-action checks, and fallback if batched collect fails.
- `policy_collect_forward_sec` is nested inside `policy_search_sec`; it is the
  LightZero collect/MCTS call itself. Do not add it to `policy_search_sec`.
- `observation_noise_sec` and `replay_observation_noise_sec` each allocate and
  perturb the full observation tensor when noise is enabled. Noise `0.0` is a
  no-op bound, not a training recommendation.
- `visual_stack_update_sec` is stack roll plus render/write. In fast-direct this
  is direct gray64 rendering; in browser-lines it includes the richer 704 RGB
  render and downsample path.
- `loop_autoreset_sec` combines env reset work and visual stack reset-row
  refresh. If reset remains large, split env reset from reset-render refresh.

## Wave 2 Result

All cells were collect-only, fast-direct, normal death, no background eval/GIF,
no initial checkpoint. Some L4 jobs were preempted and restarted by Modal; the
table uses final saved artifacts.

| Attempt | Compute | B | Sims | Replay Rows | Wall | Replay Rows/s | Search | Policy Rows/s | Visual | Noise | Env + Autoreset |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b32-sim8-l4 | L4/T4 | 32 | 8 | 49,152 | 94.97s | 517 | 36.61s | 1,343 | 8.26s | 20.40s | 10.17s |
| b64-sim8-l4 | L4/T4 | 64 | 8 | 98,304 | 217.39s | 452 | 75.44s | 1,303 | 19.68s | 52.20s | 26.46s |
| b64-sim8-l4-noise0 | L4/T4 | 64 | 8 | 98,304 | 130.49s | 753 | 59.77s | 1,645 | 15.92s | 0.01s | 19.91s |
| b64-sim16-l4 | L4/T4 | 64 | 16 | 98,304 | 194.04s | 507 | 87.71s | 1,121 | 16.33s | 35.63s | 20.41s |
| b64-sim32-l4 | L4/T4 | 64 | 32 | 65,536 | 175.50s | 373 | 102.16s | 641 | 11.27s | 23.07s | 13.68s |
| b128-sim8-l4 | L4/T4 | 128 | 8 | 131,072 | 206.46s | 635 | 68.57s | 1,912 | 21.35s | 49.24s | 23.17s |
| b128-sim8-h100 | H100 | 128 | 8 | 131,072 | 182.58s | 718 | 48.22s | 2,718 | 14.63s | 58.11s | 23.43s |
| b128-sim16-h100 | H100 | 128 | 16 | 131,072 | 211.99s | 618 | 69.41s | 1,888 | 15.24s | 60.05s | 24.48s |

Plain read:

- `fast_gray64_direct` rendering is not the main bottleneck in this wave.
- Default observation noise is now a major Amdahl target. The B64 sim8 no-noise
  bound improved replay throughput from about `452` rows/s to `753` rows/s.
  That does not mean Coach should disable noise; it means Optimizer should make
  the augmentation cheaper.
- H100 helps search throughput at B128/sim8 (`2,718` policy rows/s versus
  `1,912` on L4/T4), but CPU noise then becomes larger than search. Bigger GPUs
  alone cannot fix the current wall-clock mix.
- B128 is not automatically bad in this clean collect-only wave. It did more
  replay rows per wall second than B64 on L4/T4, but it also increases CPU noise,
  reset, replay-build, and visual work. Treat it as a candidate, not a default.
- Higher simulation counts reduce policy rows/s as expected. Sim32 is mainly a
  quality/search-budget probe, not a speed setting.

Next optimizer target: keep MuZero behavior unchanged and optimize CPU
observation-noise implementation first. Then reprofile B64/B128 on L4 and H100.

## Noise Workspace Reprofile

Patch: the custom two-seat collector now reuses two float32 noise workspaces in
the collect loop, one for current observations and one for replay-next
observations. It preserves full-shape noise draws and keeps the no-noise fast
path.

Validation:

```text
uv run ruff check src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py tests/test_curvytron_two_seat_render_mode.py
uv run pytest tests/test_curvytron_two_seat_render_mode.py -q
```

Both passed.

Post-patch custom-adapter profile matrix:

| Attempt | Compute | B | Sims | Replay Rows | Wall | Replay Rows/s | Search | Policy Rows/s | Visual | Noise | Env + Autoreset |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b64-sim8-l4-noiseopt | L4/T4 | 64 | 8 | 98,304 | 235.54s | 417 | 83.48s | 1,178 | 23.88s | 49.45s | 30.20s |
| b64-sim8-l4-noise0 | L4/T4 | 64 | 8 | 98,304 | 115.54s | 851 | 53.81s | 1,827 | 15.98s | ~0s | 17.43s |
| b128-sim8-l4-noiseopt | L4/T4 | 128 | 8 | 131,072 | 288.69s | 454 | 91.84s | 1,427 | 31.37s | 66.39s | 36.56s |
| b128-sim8-h100-noiseopt | H100 | 128 | 8 | 131,072 | 185.47s | 707 | 50.43s | 2,599 | 14.94s | 52.75s | 24.10s |
| b64-sim16-l4-noiseopt | L4/T4 | 64 | 16 | 98,304 | 184.99s | 531 | 82.79s | 1,187 | 16.14s | 32.81s | 19.77s |
| b128-sim16-h100-noiseopt | H100 | 128 | 16 | 131,072 | 228.43s | 574 | 81.84s | 1,602 | 16.59s | 56.70s | 26.10s |

Plain read:

- The workspace patch is correct and worth keeping, but it is not a large
  whole-loop win. Run-to-run variance and search/render/reset terms are larger
  than the saved allocation overhead.
- The no-noise bound is still much faster, so observation augmentation remains
  an Amdahl term. A bigger fix would need a different augmentation strategy or
  a training decision about the noise knob, not just buffer reuse.
- These rows are now postmortem-only for training recommendations because they
  use the custom adapter. The next trusted profiling lane is stock
  `train_muzero` with `source_state_fixed_opponent`.
