# CurvyTron Native LightZero Profile

Date: 2026-05-11

Status: optimizer evidence note. This is about speed/setup, not learning
quality. Timers below are debug profiler timers and can be nested; do not add
phase buckets together as exclusive wall time.

## Plain Read

The current CurvyTron native LightZero path is the coach-facing trainer path to
use for CurvyTron work unless a new decision replaces it:

```text
source_state_fixed_opponent
-> non-ALE source-state visual stack [4,64,64]
-> LightZero train_muzero
-> collector/search, replay, learner, evaluator, checkpoint hooks
```

It is still fixed-opponent single-ego training, not simultaneous current-policy
self-play. The env metadata says that explicitly.

`source_state_turn_commit` should be read as stock LightZero plumbing/control,
not a learning-quality self-play proof. Its pending/player0 scalar step has no
physics advance and reward `0`; the commit/player1 scalar step advances physics
and gets survival reward, so normal GameSegment value targets may create reward
credit leakage.

The useful speed picture after the 2026-05-11 optimizer pass:

- Renderer/stack/obs packing was a real bottleneck and is now much cheaper.
- Current env-wrapper audit fixes are plumbing-only: align source-state knobs,
  metadata, telemetry, stack schema hash, and keep render from mutating the
  observation stack.
- The train model is actually on CUDA in GPU runs; observed model parameter
  device samples include `cuda:0`.
- The GPU is still underfed because LightZero search is receiving small root
  batches. Wider collector batches improve throughput without changing MuZero.
- `collector_env_num` and `n_episode` are the immediate speed knobs. Keep them
  equal for clean sweeps.
- `source_max_steps` is only a cap in normal runs. For profiling long games,
  `disable_death_for_profile=true` now lets the same source-state visual path
  run to timeout with natural source-default bonuses active. That knob is
  profile-only and is not source-fidelity evidence.

Latest no-death reruns now reach collect, search, replay, learner, evaluator,
and checkpoint hooks. The old `BonusAllColor` / `BonusSelfMaster` catch gap is
resolved in the runtime, and the remaining optimizer-side fix was to raise the
finite natural-bonus placement retry slab from `16` to `256` so the vector path
better matches the scalar source loop, which retries until placement succeeds.

After the no-death profile unblocked, the biggest simple speed win was switching
the LightZero env manager from `base` to `subprocess`. This matches the
LightZero/Pong pattern and is now the trainer default. Use
`--env-manager-type base` when detailed in-process env timers are needed.

## Cadence

CurvyTron source physics use elapsed milliseconds. A real five-minute human game
does not mean the simulator should take five wall-clock minutes.

Current wrapper default:

```text
decision_ms = 300
5 real minutes = about 1,000 RL decisions
10 real minutes = about 2,000 RL decisions
```

The 60Hz-ish stress profiles used:

```text
decision_ms = 16.6666667
source_max_steps = 600
```

That is a profiling stress shape, not the current recommended training cadence.
Raising `source_max_steps` only raises the cap. It does not force bad policies
to survive.

## Renderer Fix

Files:

- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `tests/test_vector_visual_observation.py`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`

Changes:

- Source-state wrapper uses a trusted renderer path after env construction has
  already fixed shapes.
- Renderer iterates active body slots directly.
- Radius-0 and radius-1 body circles are drawn with vectorized
  `np.maximum.at`.
- Larger circles keep the scalar mask path.
- Circle masks are cached by pixel radius.
- Player-perspective remap uses a cached 256-entry LUT.
- Public validated renderer now rejects invalid `body_write_cursor`.

Local synthetic long-trail render+normalize stress:

```text
active body slots | before patch | after patch
0                 | 1.42 ms      | 0.03 ms
128               | 2.85 ms      | 0.06 ms
512               | 7.16 ms      | 0.09 ms
1024              | 13.37 ms     | 0.12 ms
2048              | 24.86 ms     | 0.18 ms
4096              | 46.80 ms     | 0.30 ms
```

The stress test is synthetic, but it hits the right failure mode: long surviving
games create dense trails. The fix is semantics-preserving for the tested
source-state renderer.

Validation:

```text
uv run pytest \
  tests/test_vector_visual_observation.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_curvyzero_stacked_debug_visual_survival_lightzero_smoke.py \
  tests/test_lightzero_phase_profiler.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py -q

29 passed
```

Post-cleanup focused source-state validation:

```text
uv run pytest \
  tests/test_lightzero_phase_profiler.py \
  tests/test_vector_visual_observation.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_curvyzero_stacked_debug_visual_survival_lightzero_smoke.py -q

23 passed
```

Normal train-mode smoke, without profiler hooks:

```text
run_id=opt-native-train-smoke-c16-s1121-wait
attempt_id=train-smoke-c16-sim16-sparse-wait
mode=train
compute=gpu-l4-t4
called_train_muzero=true
ok=true
checkpoints copied: iteration_0, iteration_35, ckpt_best
```

Earlier full `tests/test_vector_multiplayer_env.py` runs exposed natural-bonus
runtime gaps. Environment has since filled the source-default catch/effect gap
for `BonusAllColor` and `BonusSelfMaster`. Optimizer's remaining change here is
a capacity fix for long no-death profiles: source bonus placement retries until
it finds a free spot, while the vector path needs a finite preallocated draw
budget.

## Full-Loop Profiles

All rows below are native `train_muzero` profiles on `gpu-l4-t4`, env variant
`source_state_fixed_opponent`, `batch_size=32`, checkpoint sparse, stopped
after `5` learner train calls. Timers are useful for Amdahl reads, but nested.

### Profile-No-Death Long Run After Bonus Fixes

Goal: redo profiling against the current source-state visual path, with death
suppressed only in profile mode, so bad policies do not end the rollout before
the environment/search loop gets stressed.

Command shape for the matched c16/c32 runs:

```text
mode=profile
compute=gpu-l4-t4
env_variant=source_state_fixed_opponent
underlying_env_class=VectorMultiplayerEnv
runtime_env_impl_id=curvyzero_vector_multiplayer_env/v0
visual_surface=source_state_visual_tensor
disable_death_for_profile=true
death_mode=profile_no_death
collector_env_num=16 or 32
n_episode=16 or 32
num_simulations=16
source_max_steps=240
decision_ms=300
stop_after_learner_train_calls=5
env_telemetry_stride=20
env_manager_type=base or subprocess
```

Results:

```text
c16:
  run_id=opt-source-state-nodeath-profile-c16-sim16-s1181-posfix
  attempt_id=profile-nodeath-c16-sim16-steps240-posfix
  ok=true
  stopped_by_optimizer_profile_cap=true
  env_steps_collected=3840
  learner_train_calls=5
  replay_sample_calls=5
  collector_collect_calls=1

c32:
  run_id=opt-source-state-nodeath-profile-c32-sim16-s1181-posfix
  attempt_id=profile-nodeath-c32-sim16-steps240-posfix
  ok=true
  stopped_by_optimizer_profile_cap=true
  env_steps_collected=7680
  learner_train_calls=5
  replay_sample_calls=5
  collector_collect_calls=1
```

Plain read: the full profile path is now being exercised. This is still not a
learning proof and it is still fixed-opponent single-ego training, but it is the
right loop for optimizer timing: source-state visual env, LightZero collector,
MCTS/search, replay, learner, evaluator, and sparse checkpoint hooks.

Matched base-env-manager timing:

```text
bucket                         c16          c32          c64
train_muzero wall              56.86s       74.76s       109.47s
env steps collected            3840         7680         15360
collected steps/s              67.5         102.7        140.3
collector.collect              32.48s       50.34s       83.24s
evaluator.eval                 17.24s       17.58s       17.80s
MCTS search                    28.79s       30.96s       32.82s
model recurrent inference      16.63s       17.39s       17.06s
policy forward collect         18.32s       21.85s       26.58s
policy forward eval            15.67s       16.06s       16.27s
env.step                       14.17s       27.63s       54.37s
vector step                    8.79s        17.22s       33.74s
runtime step_many              5.43s        10.71s       20.39s
render gray64                  1.36s        2.62s        5.20s
stack update                   1.63s        3.16s        6.24s
LightZero obs pack             1.78s        3.55s        6.93s
learner train                  2.43s        1.96s        1.98s
replay sample                  0.17s        0.17s        0.15s
checkpoint save                0.24s        0.25s        0.25s
MCTS root sum                  4080         7920         15600
MCTS recurrent batch mean      8.5          16.5         32.5
GPU max util                   19%          19%          22%
```

Read: c32 doubled collected steps for only about `1.31x` wall time, so wider
collector batches are still useful. Search/model time barely grew because the
root batch got wider. Env step grew almost linearly with env decisions, so long
survival profiles now make the vector env/runtime path visible again. The
biggest current wall-clock buckets are collect/search and env stepping; learner
and replay are small in this profile.

Matched subprocess-env-manager timing:

```text
bucket                         c16          c32          c64
train_muzero wall              48.20s       52.50s       68.05s
env steps collected            3840         7680         15360
collected steps/s              79.7         146.3        225.7
collector.collect              21.84s       26.59s       37.77s
evaluator.eval                 18.21s       16.87s       17.68s
MCTS search                    28.86s       29.28s       32.40s
policy forward collect         18.18s       20.76s       26.81s
policy forward eval            16.00s       15.17s       15.66s
learner train                  2.04s        1.80s        1.92s
replay sample                  0.16s        0.14s        0.15s
MCTS root sum                  4080         7920         15600
MCTS recurrent batch mean      8.5          16.5         32.5
```

Read: subprocess is a real speed win for long profiles: about `1.18x` at c16,
`1.42x` at c32, and `1.61x` at c64 versus the base manager. The tradeoff is
observability: env method monkeypatch timers do not fire inside subprocess
workers, so detailed env-step breakdowns require `--env-manager-type base`.
The end-to-end summaries and action telemetry stayed clean. Profile-mode and
small train-mode runs can print ignored `BrokenPipeError` messages during
LightZero subprocess cleanup; the summaries still returned `ok=true`.

Historical blocker: before Environment added the missing source-default bonus
effects, the same route failed before learner/replay on unsupported
`BonusAllColor`. That old failure is kept only as history in
[environment_handoff_bonus_runtime_blocker_2026-05-11.md](environment_handoff_bonus_runtime_blocker_2026-05-11.md).

### Matched c4/sim8 Before And After Renderer Fix

```text
collector_env_num=4
n_episode=4
num_simulations=8
```

```text
bucket                         before       after
train_muzero wall              22.05s       18.73s
MCTS search                    8.07s        8.07s
collector.collect              9.41s        7.46s
evaluator.eval                 6.91s        6.45s
policy forward collect         5.69s        5.43s
policy forward eval            5.45s        5.42s
env.step                       4.55s        2.47s
render gray64                  2.12s        0.19s
stack update                   2.18s        0.23s
LightZero obs pack             2.20s        0.25s
vector step                    0.97s        0.91s
telemetry write                1.03s        0.99s
learner train                  1.81s        1.60s
replay sample                  0.15s        0.11s
GPU max util                   29%          29%
```

Read: renderer/stack/obs packing was a real bottleneck and is now mostly gone.
Wall time improved about `15%` on this matched profile. Amdahl wins are now
limited by MCTS/search, policy forward, evaluator, vector step, telemetry, and
learner.

### Telemetry Stride

Same c4/sim8 profile with:

```text
--env-telemetry-stride 20
```

```text
telemetry write: 0.99s -> 0.07s
env.step:        2.47s -> 1.51s
wall:            18.73s -> 19.10s
```

Read: sparse telemetry removes almost a second from env-step overhead in this
matched profile. The single-run wall did not improve; paired repeats would be
needed to separate profiler variance from the telemetry win. Keep sparse
telemetry for long coach runs when dense action JSONL is not needed.

Caveat: with sparse telemetry, action histograms are sampled. Summaries now
label this with `counts_scope`, `telemetry_sampled`, `telemetry_stride`, and
`sampled_row_count`.

### Search And Collector Sweeps

```text
profile          wall    MCTS    MCTS/call  env.step  GPU max
c4 sim8          18.73s  8.07s   19.2 ms    2.47s     29%
c8 sim8          31.58s  14.46s  26.4 ms    5.73s     23%
c4 sim16         22.84s  12.34s  36.5 ms    2.25s     31%
```

Read:

- Doubling `num_simulations` increases search cost as expected. This is not an
  optimization bug; it is the MuZero planning budget.
- `collector_env_num=8` collects more steps but did not improve GPU saturation.
  It lowered wall per telemetry row but increased MCTS/call and total wall.
- The current single-process LightZero loop is still underfeeding the GPU.

### CUDA And Batch-Width Recheck

These runs add train-time model-device samples, model inference timings, MCTS
root counts, and derived batch means. They used `decision_ms=300`,
`source_max_steps=1000`, sparse telemetry, and `5` learner calls.

```text
profile     sims  collectors  wall    steps  steps/s  root batch  MCTS    env.step  GPU max
c4          16    4           8.38s   39     4.66     1.89        1.19s   0.12s     26%
c8          16    8           7.99s   81     10.14    3.41        1.11s   0.18s     23%
c16         16    16          9.02s   178    19.74    6.68        1.23s   0.35s     27%
c32         16    32          7.75s   290    37.41    10.75       1.23s   0.56s     29%
c4          50    4           14.03s  39     2.78     2.00        4.34s   0.15s     22%
c16         50    16          12.15s  137    11.28    4.67        4.12s   0.32s     30%
```

Read:

- The model is on CUDA, but search is small-batch. GPU max utilization stayed
  around `22-30%`.
- Increasing `collector_env_num` from `4` to `32` improved collected-step
  throughput about `8x` in the short-episode sim16 profile, without changing
  the MuZero algorithm.
- Increasing `collector_env_num` from `4` to `16` improved serious-search
  sim50 throughput about `4x` in this profile.
- `num_simulations` changes search budget, not batch width. It should be a
  quality/profile knob, not a hidden speed optimization.
- `batch_size` mostly affects learner sampling/update width; it does not fix
  MCTS root underbatching.

## Current Bottleneck Ranking

For this profile shape:

1. Collect/search plus policy forward. This remains the largest combined bucket.
2. Env step/runtime work in long no-death profiles. Subprocess hides detailed
   env timers but gives the fastest wall-clock; base-manager profiles show this
   bucket scales almost linearly with decisions.
3. Underbatched search/model inference. Wider collectors help immediately.
4. Evaluator time, because profiles still include initial eval work.
5. Learner train for small `5`-update profiles.
6. Replay sample is small here.

For future long-surviving policies:

- Environment work grows with decisions and trail density.
- Renderer long-trail cost is much lower after the patch.
- MCTS still grows with decisions times `num_simulations`.
- If serious runs use `50` simulations, search will dominate unless batching or
  architecture changes improve it.

## MCTS And GPU Recheck

Plain answer: the current CurvyTron native path is using LightZero's MuZero
machinery, not a repo-owned MCTS implementation.

Verified local path:

```text
lightzero_curvyzero_stacked_debug_visual_survival_train.py
-> env_variant=source_state_fixed_opponent
-> CurvyZeroSourceStateVisualSurvivalLightZeroEnv
-> VectorMultiplayerEnv
-> [4,64,64] source-state visual stack
-> lzero.entry.train_muzero
-> LightZero collector / MCTS / replay / learner / evaluator
```

CUDA/GPU status:

- `--compute gpu-l4-t4` requests a Modal L4/T4 GPU function.
- The config builder passes `cuda=compute != "cpu"` and sets
  `policy.cuda=True` for GPU runs.
- The profile samples `nvidia-smi`; recent runs showed max GPU utilization only
  about `22-30%`.
- The profiler records model parameter device samples; GPU profile summaries
  include `cuda:0`.
- That means the run is GPU-configured, but the GPU is underfed by small
  synchronous search batches.

LightZero search status:

- The profiler hooks `MuZeroMCTSCtree.search` and `MuZeroMCTSPtree.search` from
  LightZero modules.
- LightZero's own docs describe the C++ `ctree` search as batched over root
  nodes and parallel in model inference.
- Therefore, "MCTS is expensive" should not be read as "we wrote bad MCTS."
  The better current read is: we are probably running a small, synchronous,
  underbatched LightZero search workload.

Framework read:

- LightZero remains the near-term base because it already gives the working
  CurvyTron visual path plus MuZero collector, replay, learner, checkpoint, and
  native MCTS.
- MCTX/JAX is the strongest alternative search primitive if LightZero cannot be
  batched well enough. It is JIT/batch-first search, not a full trainer.
- MiniZero is the strongest full-system architecture reference, but likely not
  a drop-in CurvyTron base.

Next proof needed:

1. Add a profile-only scripted-survivor stress mode if we need a true
   long-survival renderer/env/search profile. Do not change training semantics.
2. If search still dominates and GPU stays underused, test actor/search fanout
   before a framework rewrite.
3. Keep LightZero as the near-term base; MiniZero/OpenSpiel/EfficientZero are
   architecture references, and MCTX is a possible future search primitive, not
   an immediate full trainer replacement.

## What To Do Next

Near term:

- Treat the source-state visual no-death profile as unblocked for optimizer
  timing. Keep the death suppression label explicit:
  `profile_only_not_source_fidelity`.
- Use the default `env_manager_type=subprocess` for training and throughput
  profiles. Use `--env-manager-type base` only when detailed env internals are
  the thing being measured.
- Do not narrow source-default bonus types from the optimizer lane. If future
  no-death profiles hit another natural-bonus semantics issue, hand it back to
  Environment with a minimal repro.
- Use `--env-telemetry-stride 20` or larger for throughput profiles and long
  coach runs unless dense per-step action telemetry is needed.
- Keep `save_ckpt_after_iter` sparse for profiling and long runs.
- For Coach-facing native CurvyTron runs, start with
  `collector_env_num=32`, `n_episode=32` as the conservative fast default; use
  `64/64` for throughput sweeps if memory/latency stays stable.
- Keep `evaluator_env_num=1` for training-throughput profiles. Eval throughput
  is a separate question.
- Use `num_simulations=16` for fast optimizer profiles and `50` for serious
  MuZero/Pong-style proof lanes when Coach wants comparability.
- Add deeper tree/transfer split only if the next bottleneck remains unclear.

Bigger picture:

- A 10x wall-clock win probably does not come from more single-loop polish.
- The likely 10x path is many actor/search workers producing searched chunks,
  explicit policy versions, replay merge, learner update, checkpoint publish,
  and actor refresh cadence.
- Do not change the MuZero algorithm to get speed. Lowering simulations is a
  profiling/quality knob, not a silent optimization.

## Coach-Facing Defaults

For current native source-state CurvyTron profiles:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute gpu-l4-t4 \
  --env-variant source_state_fixed_opponent \
  --opponent-policy-kind fixed_straight \
  --env-manager-type subprocess \
  --ego-action-straight-override-probability 0 \
  --control-noise-profile-id none \
  --env-telemetry-stride 20 \
  --save-ckpt-after-iter 1000
```

For learning-quality runs, Coach owns the budget/eval gates. Optimizer advice is
only: keep telemetry/checkpoints sparse unless debugging, and preserve
`env_variant` plus schema metadata end to end.
