# Stock Frozen Optimizer Pivot

Date: 2026-05-12

Purpose: keep the Optimizer lane aligned after the Coach architecture reset.

## Plain Read

The current trusted CurvyTron lane is stock LightZero `train_muzero` with
`env_variant=source_state_fixed_opponent`,
`opponent_policy_kind=frozen_lightzero_checkpoint`, and a frozen checkpoint
opponent on CPU by default.

The custom `--mode two-seat-selfplay` path is not the trusted learning path.
It is useful as a profiler and as postmortem evidence, but the failed scaled
runs mostly test our custom adapter, not CurvyTron or LightZero.

Optimizer should now measure and improve the stock frozen-opponent lane first.

## What Just Changed

Recent custom two-seat profiling still taught us useful speed facts:

| Cell | Wall | Search | Obs Noise | Visual | Replay Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| B64/L4/sim8/noise | 235.54s | 83.48s | 49.45s | 23.88s | 98,304 |
| B64/L4/sim8/no-noise | 115.54s | 53.81s | ~0s | 15.98s | 98,304 |
| B128/L4/sim8/noise | 288.69s | 91.84s | 66.39s | 31.37s | 131,072 |
| B128/H100/sim8/noise | 185.47s | 50.43s | 52.75s | 14.94s | 131,072 |
| B64/L4/sim16/noise | 184.99s | 82.79s | 32.81s | 16.14s | 98,304 |
| B128/H100/sim16/noise | 228.43s | 81.84s | 56.70s | 16.59s | 131,072 |

Those are postmortem/custom-adapter timings. They should not be handed to Coach
as recommended training settings.

The safe conclusions are narrower:

- CPU observation noise can be expensive.
- Bigger GPU helps policy/search but does not remove CPU terms.
- Batch width changes the CPU/env/render/noise pressure, not just GPU search.
- Profiling commands must name the trainer path, not just batch/GPU/sims.

## Current Trusted Path

Use stock LightZero `train_muzero` through the train route. The frozen opponent
should stay on CPU unless a specific experiment says otherwise:

```text
--mode train
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--opponent-use-cuda=false
```

Known proof runs:

```text
stock-frozen-canary-source-state-s304-20260512
stock-frozen-gpu-base-canary-source-state-s304-20260512b
```

These prove stock `train_muzero` plumbing and strict checkpoint opponent load.
They do not prove learning.

## Optimizer Patch

The stock config builder now separates learner CUDA from frozen-opponent CUDA.
Before this, GPU compute set both `policy.cuda=true` and
`env.opponent_use_cuda=true`. That made subprocess env workers try to own CUDA
when the env constructed its frozen checkpoint opponent.

Current behavior:

- GPU learner still uses `policy.cuda=true`.
- Frozen opponent defaults to `opponent_use_cuda=false`.
- The CLI still allows `--opponent-use-cuda` for explicit experiments.
- The profile/summary surface records `opponent_use_cuda`.

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_curvytron_live_checkpoint_eval_plumbing.py
All checks passed

uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py -q
31 passed, 1 skipped

uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_frozen_opponent_cuda_is_decoupled_from_gpu_learner tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_frozen_opponent_cuda_can_still_be_explicitly_enabled -q
2 passed
```

## Immediate Optimizer Questions

1. What is the exact code path for stock frozen CurvyTron training?
2. What timing buckets already exist for that path?
3. What is slow in a small stock frozen GPU profile: env/render, MCTS/search,
   learner, replay/buffer, eval/checkpoint, or Modal artifact work?
4. Does GPU learner plus subprocess env manager now work with the CPU frozen
   opponent flag?
5. If yes, how far can collector env count scale before CPU opponent inference,
   env/render, or search becomes the limiter?

## Next Small Gates

Do not launch a big training run from the Optimizer lane.

Do launch small optimizer profiles once the command is understood:

- CPU/base stock-frozen tiny profile, as a denominator.
- L4/base stock-frozen tiny profile, matching the passed canary.
- L4/subprocess stock-frozen tiny profile with `opponent_use_cuda=false`.
- L4/subprocess wider collector profile if the tiny profile passes.

Each profile must report:

- `called_train_muzero`;
- `env_variant`;
- `opponent_policy_kind`;
- `env_manager_type`;
- collector/evaluator env count;
- learner device;
- opponent device;
- `opponent_use_cuda`;
- phase timings;
- checkpoint/eval/artifact cadence.

Use this known opponent checkpoint for the first profiles unless Coach gives a
newer trusted frozen opponent:

```text
training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar
```

## Profile Results So Far

All rows below used stock `train_muzero`,
`env_variant=source_state_fixed_opponent`,
`opponent_policy_kind=frozen_lightzero_checkpoint`, browser-lines source-state
visuals, sparse outcome reward, LightZero eval off, background eval/GIF off,
checkpoint save interval `9999`, and `opponent_use_cuda=false`.

| Run | Manager | Collectors | Roots | Wall | Collect | MCTS | Learner | Steps/s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `opt-stock-frozen-l4-base-b16-sim8-s304-20260512d` | base | 1 | 66 | 10.79s | 4.63s | 1.20s | 1.67s | 3.06 | component attribution lens |
| `opt-stock-frozen-l4-subproc-c2-b16-sim8-s304-20260512e` | subprocess | 2 | 22 | 9.90s | 2.84s | 0.34s | 2.08s | 2.22 | subprocess works after CPU-opponent split |
| `opt-stock-frozen-l4-subproc-c4-b16-sim8-s304-20260512f` | subprocess | 4 | 38 | 9.94s | 3.00s | 0.37s | 1.87s | 3.82 | wider roots improve throughput |
| `opt-stock-frozen-l4cpu40-subproc-c8-b16-sim8-s304-20260512f` | subprocess | 8 | 87 | 12.63s | 4.56s | 0.39s | 2.11s | 6.89 | C8 scales in tiny profile |
| `opt-stock-frozen-l4cpu40-subproc-c16-b16-sim8-s304-20260512g` | subprocess | 16 | 159 | 13.80s | 5.15s | 0.42s | 2.06s | 11.52 | C16 still scales |
| `opt-stock-frozen-l4cpu40-subproc-c32-b16-sim8-s304-20260512h` | subprocess | 32 | 314 | 13.89s | 4.78s | 0.36s | 1.67s | 22.61 | C32 still scales for short normal-death episodes |
| `opt-stock-frozen-l4cpu40-subproc-c16-b16-sim16-s304-20260512h` | subprocess | 16 | 153 | 14.14s | 5.40s | 0.75s | 2.09s | 10.82 | doubling sims roughly doubles named MCTS but not wall |
| `opt-stock-frozen-l4cpu40-subproc-c64-b16-sim8-s304-20260512i` | subprocess | 64 | 605 | 21.21s | 9.29s | 0.49s | 1.63s | 28.52 | C64 still improves steps/s, but wall grows |
| `opt-stock-frozen-l4cpu40-subproc-c32-b16-sim16-s304-20260512i` | subprocess | 32 | 305 | 16.28s | 6.74s | 0.83s | 1.68s | 18.73 | sim16 at C32 is slower than sim8 but still scales |

Plain read:

- The small device split did what it was supposed to do: GPU learner plus
  subprocess env manager now runs with a frozen checkpoint opponent on CPU.
- Base manager is still needed for fine-grained env/render/opponent attribution;
  subprocess hides env worker internals.
- Subprocess collector width improves tiny-profile throughput through C64 with
  40 CPUs for short normal-death profiles. C64 is not free: wall grew from
  `13.89s` at C32 to `21.21s`, but work done rose enough that steps/s still
  improved.
- More simulations are visible but not dominant in these tiny profiles:
  C32/sim16 reached `18.73` steps/s versus C32/sim8 at `22.61` steps/s.
- These are small plumbing/profile runs, not learning claims.

Base-manager attribution snippets:

- Frozen checkpoint opponent, browser-lines, C1: `env_step=2.11s`,
  `env_stack_update=0.74s`, `env_render_rgb_canvas=0.64s`,
  `policy_forward_eval=0.76s`, `collector_collect=4.63s`.
- Fixed-straight opponent, browser-lines, C1:
  `collector_collect=3.69s`, `policy_forward_eval` absent, `env_step=1.51s`.
  Plain read: the frozen checkpoint opponent costs real time, but it is not the
  only limiter.
- Frozen checkpoint opponent, `body_circles_fast`, C1:
  `env_stack_update=0.50s`, `env_render_rgb_canvas=0.38s`,
  `collector_collect=4.41s`. Plain read: fast trail rendering helps this tiny
  base attribution row a little, but it is not a decisive short-episode win.

Long-survival base attribution with death disabled and `source_max_steps=256`:

| Render Mode | Wall | Collect | Env Step | Stack/Obs Pack | RGB Render | Gray64 | Runtime Step-Many | MCTS | Frozen Opponent Eval | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `browser_lines` | 73.73s | 68.26s | 61.51s | 51.01s | 50.37s | 0.62s | 4.39s | 8.42s | 5.28s | 3.47 |
| `body_circles_fast` | 36.31s | 30.43s | 23.58s | 13.19s | 12.52s | 0.66s | 4.32s | 8.50s | 5.21s | 7.05 |

Plain read: once trajectories survive for a while, rich browser-lines rendering
is the biggest exposed cost. The physics/runtime step-many work is only about
`4.3s` in both rows; the main difference is RGB render plus stack packing.
`body_circles_fast` is about 2x faster end to end on this matched profile, but
it is a visual-fidelity tradeoff and should not quietly replace the trusted
training visual without an explicit Coach/Environment decision. 2026-05-15
correction: current production policy observations are CPU `cpu_oracle`
`browser_lines + simple_symbols`; this `body_circles_fast` row is historical
control evidence only.

Long-survival subprocess scaling check, C4, death disabled, 1,024 total env
steps:

| Render Mode | Wall | Collect | MCTS | Policy Forward Collect | Learner | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `browser_lines` | 98.92s | 91.77s | 7.78s | 10.82s | 1.57s | 10.35 |
| `body_circles_fast` | 48.18s | 41.00s | 7.92s | 11.01s | 1.69s | 21.25 |

Plain read: the same 2x render-path effect survives subprocess collection.
MCTS and learner barely move; the difference is folded into collector/env
worker time. This confirms that render/observation work is the right Amdahl
target for long-survival rich-visual profiles.

Worker-side timing patch:

- Profile-mode stock envs now emit sampled `profile_env_timing_sec` in env-step
  telemetry. Train mode does not enable it.
- This exists because `env_manager_type=subprocess` hides env-worker internals
  from the parent-process monkeypatch profiler.
- The timing sums are worker CPU-seconds across env workers. They can be larger
  than `collector_collect` wall time when workers run in parallel.
- Tooling: `scripts/summarize_curvytron_lightzero_profiles.py` now surfaces
  `telem_obs`, `telem_opp`, and `telem_vec` when those telemetry sums exist.

Validation profile:

| Run | Manager | Collectors | Steps | Wall | Collect | MCTS | Worker Obs/Render | Worker Frozen Opponent | Worker Vector Step | Steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `opt-stock-frozen-l4cpu40-subproc-c4-b16-sim8-nodeath-browser-worker-timing-s304-20260512j` | subprocess | 4 | 256 | 20.62s | 13.24s | 1.92s | 20.60s | 9.20s | 4.62s | 12.42 |

Plain read: on this 4-worker browser-lines no-death profile, the env-worker
CPU time is mostly observation/render, then frozen-opponent inference/action
selection, then physics/vector stepping. This matches the base-manager Amdahl
read and gives us a way to profile real subprocess widths without guessing.

Long-survival warning:

- `opt-stock-frozen-l4cpu40-subproc-c2-b16-sim8-nodeath-s304-20260512f` reached
  1,591 collected env steps before failing with LightZero
  `ValueError: 'a' and 'p' must have same size`.
- It spent `480.54s` wall and `475.06s` in collector collect; named MCTS was
  only `29.63s` and policy-forward collect was `37.20s`.
- Because subprocess hides env worker internals, this is not a clean component
  breakdown. But the Amdahl read is clear enough: long-survival stock browser
  lines are probably dominated by env/render/opponent worker work, not learner
  or named MCTS alone.

Next stock-path profiling questions:

- Where does collector width stop helping: C64 still helped, but C96/C128 may
  hit CPU/process overhead on a 40-CPU container.
- Does a longer no-death run fail only for browser-lines, or does the LightZero
  probability-size error appear independent of render mode?
- Can the browser-lines renderer be made closer to `body_circles_fast` speed
  without changing the observation contract?
- Can frozen-opponent CPU inference be batched or made cheaper without putting
  CUDA inside subprocess env workers?
- If render remains dominant in long-survival runs, bigger GPUs will not help
  much until the env/render work is reduced or parallelized.

## Responsibility Split

Optimizer owns speed, measurement, profiling commands, CPU/GPU split, and
small safe plumbing needed for measurement.

Coach owns learning claims, reward/eval interpretation, and whether a curve is
good enough.

Environment owns source fidelity.
