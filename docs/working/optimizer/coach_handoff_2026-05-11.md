# Optimizer to Captain/Coach Handoff - CurvyTron Native LightZero

Date: 2026-05-11

## Plain Version

Use the native CurvyTron LightZero trainer path for CurvyTron work:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

The filename is stale, but the current default path is the source-state visual
trainer:

```text
env_variant=source_state_fixed_opponent
non-ALE [4,64,64] source-state visual stack
fixed-straight or frozen-checkpoint opponent
lzero.entry.train_muzero
LightZero collector / MCTS / replay / learner / checkpoint path
```

This is the path the optimizer has been profiling. It is the right starting
point for Coach. It may still have bugs and it is not a learning proof, but it
is the repo path closest to the promising Pong-style LightZero runs.

## What Changed

- The source-state visual renderer/stack path was optimized. Long-trail
  synthetic render+normalize cost dropped from tens of milliseconds to
  sub-millisecond in the tested renderer path.
- The native trainer now has profile-mode hooks for collector, MCTS/search,
  model initial/recurrent inference, replay, learner, evaluator, env step,
  render, stack, telemetry, checkpoints, GPU samples, and model device.
- The profiler now reports derived batch means from full counters, so search
  batching is easier to read.
- A small env helper breakage was fixed for the current source-state path:
  `VectorMultiplayerEnv` now has the boolean/natural-bonus helper
  definitions needed at construction time. The source-state visual path tests
  pass. Broader natural-bonus tests still need Environment-owner review.
- The old no-death/profile blocker on naturally spawned `BonusAllColor` /
  `BonusSelfMaster` is resolved. Optimizer also raised the finite natural-bonus
  placement retry slab from `16` to `256` so long no-death profiles do not
  crash in crowded states before source would.
- New no-death/profile reruns now exercise the real loop: source-state visual
  env, LightZero collector, MCTS/search, replay, learner, evaluator, and sparse
  checkpoint hooks.
- The trainer now has an explicit `--env-manager-type` knob. Default is
  `subprocess`, matching the stock LightZero/Pong pattern. Use
  `--env-manager-type base` only when Optimizer needs detailed in-process env
  timers.
- Env-wrapper audit found safe plumbing fixes in the source-state path: align
  env knobs, metadata, telemetry, stack schema hash, and ensure render does not
  mutate the observation stack.
- Coach cleanup smoke for `env_variant=source_state_turn_commit` passed:
  `curvytron-source-state-turncommit-smoke-s20260511b` /
  `profile-smoke-sim2-c2-steps64-20260511b`. It used stock `train_muzero`,
  GPU model/search, MCTS, replay sample, one learner step, copied
  `iteration_0`, and wrote env-step telemetry.
- Boundary: `source_state_turn_commit` is a plumbing smoke/control path right
  now, not a learning-quality self-play path. Reward credit remains untrusted
  because player 0 has a pending/no-physics scalar step and player 1 has the
  physical commit scalar step. Profile it as stock-LightZero plumbing; do not
  optimize or scale it as a proven trainer until reward targets are audited or
  fixed.

## What The Profiles Say

GPU is real, but underfed:

- GPU runs see model parameter device samples on `cuda:0`.
- L4 max utilization stayed around `22-30%`.
- Search batches are small unless collector width is raised.

Best immediate speed knob:

```text
collector_env_num == n_episode
```

Short-episode profile results:

```text
profile     sims  collectors  wall    steps  steps/s  root batch  MCTS    GPU max
c4          16    4           8.38s   39     4.66     1.89        1.19s   26%
c8          16    8           7.99s   81     10.14    3.41        1.11s   23%
c16         16    16          9.02s   178    19.74    6.68        1.23s   27%
c32         16    32          7.75s   290    37.41    10.75       1.23s   29%
c4          50    4           14.03s  39     2.78     2.00        4.34s   22%
c16         50    16          12.15s  137    11.28    4.67        4.12s   30%
```

Plain read: wider collectors gave about `8x` collected-step throughput for
sim16 and about `4x` for sim50 in these short-episode profiles, without
changing the MuZero algorithm.

Long no-death source-state profile results, with natural source-default bonuses
on and `source_max_steps=240`:

```text
manager     sims  collectors  wall     steps  steps/s  root batch  MCTS
base-c16    16    16          56.86s   3840   67.5     8.5         28.79s
sub-c16     16    16          48.20s   3840   79.7     8.5         28.86s
base-c32    16    32          74.76s   7680   102.7    16.5        30.96s
sub-c32     16    32          52.50s   7680   146.3    16.5        29.28s
base-c64    16    64          109.47s  15360  140.3    32.5        32.82s
sub-c64     16    64          68.05s   15360  225.7    32.5        32.40s
```

Plain read: subprocess is the quick win. It sped up the matched long no-death
profiles by about `1.18x` at c16, `1.42x` at c32, and `1.61x` at c64. Search
time stayed similar; the win is from not serializing all env work through the
base manager. In subprocess mode, detailed env timers are missing because the
env methods run in worker processes.

## Suggested Coach Command Shape

For a small real training run now that the Environment bonus-catch blocker is
resolved:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --env-variant source_state_fixed_opponent \
  --opponent-policy-kind fixed_straight \
  --env-manager-type subprocess \
  --collector-env-num 16 \
  --n-episode 16 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 16 \
  --batch-size 32 \
  --env-telemetry-stride 50 \
  --save-ckpt-after-iter 1000
```

For serious Pong-comparable MuZero search pressure, use `--num-simulations 50`
and expect search to dominate more. For throughput sweeps, test
`collector-env-num/n-episode` at `32/32` and `64/64`.

Do not launch a long Coach run assuming the current natural source-default
surface is a browser-pixel fidelity claim. It is source-state backed, non-ALE,
and non-browser-pixel. If a future run hits another natural-bonus semantics
issue, keep the repro and hand it to Environment.

For optimizer profiling, use `--mode profile`; that installs debug timing hooks
and stops after a small number of learner calls. Do not treat profile-mode
output as a learning run.

## Validation Done By Optimizer

Focused local checks passed:

```text
uv run pytest \
  tests/test_lightzero_phase_profiler.py \
  tests/test_vector_visual_observation.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_curvyzero_stacked_debug_visual_survival_lightzero_smoke.py -q

23 passed
```

The source-state `wait-for-train` smoke also completed on Modal:

```text
run_id: opt-native-train-smoke-c16-s1121-wait
attempt_id: train-smoke-c16-sim16-sparse-wait
mode: train
compute: gpu-l4-t4
called_train_muzero: true
ok: true
phase_profile_enabled: false
checkpoint refs copied: iteration_0, iteration_35, ckpt_best
sampled telemetry rows: 34
```

This proves the normal trainer path runs without profile stop hooks. It does
not prove learning quality.

Subprocess train smoke also completed:

```text
run_id: opt-source-state-train-smoke-subproc-s1191
attempt_id: train-smoke-subproc-c4-sim8
mode: train
env_manager_type: subprocess
compute: gpu-l4-t4
called_train_muzero: true
ok: true
checkpoints copied: iteration_0, iteration_9, ckpt_best
```

Small subprocess runs can print ignored `BrokenPipeError` lines during
LightZero/DI-engine teardown. The summary still returned `ok=true`; treat that
as cleanup noise unless it turns into a failed run.

Focused post-bonus local checks passed:

```text
uv run pytest \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_vector_multiplayer_env.py -q \
  -k 'source_state_visual_survival_profile_no_death or bonus_all_color or self_master or natural_bonus or death_mode'

15 passed

uv run pytest \
  tests/test_vector_runtime.py \
  tests/test_vector_multiplayer_env.py \
  tests/test_multiplayer_replay_contract.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -q \
  -k 'bonus or natural_bonus or source_state_visual_survival_profile_no_death'

68 passed
```

## Boundaries

- Coach owns learning claims, eval gates, checkpoint quality, and whether
  survival improves.
- Environment owns source fidelity and any promotion of visual tensor semantics.
- Optimizer owns speed/setup/profile evidence and recommended runtime knobs.
- This native path is fixed-opponent single-ego training today, not true
  current-policy self-play.
- `source_state_turn_commit` is stock LightZero plumbing/control only for now.
  Its pending/player0 scalar step gets reward `0`, while the commit/player1
  scalar step advances physics and gets survival reward; normal GameSegment
  value targets may mis-credit player0 states.
- Do not change MuZero semantics to get speed. Lowering `num_simulations` is a
  profile/quality knob, not a silent optimization.
- The current source-state path uses the natural source-default runtime surface.
  Optimizer will not narrow bonus types as a speed fix. Environment owns any
  future source-fidelity claim or natural-bonus semantics issue.

## Docs To Read

- `docs/working/optimizer/curvytron_native_lightzero_profile_2026-05-11.md`
- `docs/working/optimizer/environment_handoff_bonus_runtime_blocker_2026-05-11.md`
  is historical/resolved, not the current blocker.
- `docs/working/optimizer/current_status_2026-05-09.md`
- `docs/working/optimizer/distributed_muzero_architecture_research_2026-05-11.md`
- `docs/working/optimizer/framework_reassessment_2026-05-11.md`
- `docs/working/training_coach_active_board_2026-05-10.md`
- `docs/working/coach_north_star_2026-05-10.md`

## Open Optimizer Next Steps

- Compare the active `64/64` no-death profile against the `16/16` and `32/32`
  runs.
- If `64/64` subprocess still underfeeds GPU in serious sim50 runs, design
  actor/search fanout instead of more single-process polish.
- Keep LightZero as the near-term base. MiniZero/OpenSpiel/EfficientZero are
  architecture references; MCTX is a possible future batched search primitive,
  not an immediate full trainer replacement.
