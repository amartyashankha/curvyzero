# Optimizer Speed Axis

## Purpose

Optimizer recommendations answer a different question than survival or
leaderboard analysis:

```text
What can run fast enough at scale without changing the learning/evaluator contract?
```

This doc keeps speed advice separate from policy-quality advice.

2026-05-15 correction: the policy observation surface is
`browser_lines + simple_symbols`. The current reliable backend is CPU
`cpu_oracle`. The scalar `jax_gpu` backend now reaches stock `train_muzero`, but
it is slower than CPU and fails in subprocess workers. The optimizer
implementation target is a batched GPU backend or render service for the same
surface. H100 learner/search compute is not GPU rendering.

2026-05-16 update: current handoff lives in
`../r18fresh_postmortem_2026-05-16/H100_L4_OPTIMIZER_HANDOFF.md`. Float32 is now
the aggressive default for the profile-only batched GPU observation boundary
sidecar; float64 is the exact-parity/debug reference. This does not change
production training yet, because the sidecar is not wired into the real trainer
and scalar `policy_observation_backend=jax_gpu` remains out of production.

2026-05-16 current launch default update: broad training defaults moved to
`gpu-l4-t4-cpu40`, `collector_env_num=256`, `n_episode=256`, `batch_size=64`,
`num_simulations=8`, `browser_lines + simple_symbols + cpu_oracle`. Fresh
current-surface profiles measured best L4 `713.83` env steps/s and best H100
`1001.94`; L4 throughput is about `28.8%` lower and acceptable for cheaper broad
runs. Batch64 helped the L4/C256 row but hurt H100, so treat it as an L4-lane
default, not a universal rule.

2026-05-21 supersession: the active optimizer source of truth is now
`docs/working/optimizer/active_working_memory_2026-05-14.md` plus
`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/`.
The older L4/C256 launch guidance above is historical. Current profiling says
renderer work helped, but the main optimizer wall is public LightZero
collect/MCTS/search/output handling. Do not copy the older C256/L4 defaults as
the current optimizer recommendation without checking the active optimizer
docs.

## Current Trusted Optimizer Context

Source docs:

- `docs/working/optimizer/README.md`
- `docs/working/optimizer/current_plate_map_2026-05-13.md`
- `docs/working/optimizer/stock_full_loop_profile_2026-05-13.md`

Trusted lane:

```text
stock LightZero train_muzero
env_variant=source_state_fixed_opponent
visual surface=CPU-reference browser_lines
input=[4,64,64] grayscale stack
```

Do not use old `fast_gray64_direct`; that belongs to the superseded custom
two-seat adapter. The target policy observation surface is
`browser_lines + simple_symbols`; the current reliable backend is
`cpu_oracle`. `body_circles_fast + simple_symbols` is historical CPU
ablation/control evidence only.

## Current Speed Read

Fresh full-loop profile:

| Collector envs | Approx env steps/sec | Read |
| ---: | ---: | --- |
| C1 | 10.8 | too narrow |
| C32 | 153.6 | useful baseline |
| C64 | 408.4 | strong throughput |
| C96 | 487.3 | marginal improvement over C64 |

Other reads:

- C64 L4/T4 sim16 costs only about 10% throughput vs sim8 in one profile shape.
- H100 was worse at sim8 C64 but may matter at higher search pressure.
- H100 rows measure learner/search compute placement, not GPU rendering.
- Render remains important for long-survival/no-death regimes, but full-loop
  bottleneck is not only render.
- GPU render is the implementation target only for faithful
  `browser_lines` trails/heads/downsample plus `simple_symbols` bonuses.
- CPU browser-lines plus symbols is the trusted parity oracle/fallback.

## Interaction With Training Recommendations

| Setting | Learning/leaderboard read | Optimizer read | Current stance |
| --- | --- | --- | --- |
| `batch64` on L4/C256 | old learning evidence was mixed/negative for batch64 on other lanes | fresh L4 current-surface profile says it is the fastest L4 row | current L4 broad default; keep batch32 as ablation/sentinel |
| `sim8` | default; `sim16` not earning quality cost | sim16 may be affordable | keep sim16 sentinel only |
| `collector32` | clean baseline | C64/C96 improve throughput | consider C64 probe separately |
| browser-lines render | target semantics | CPU path is current reliable backend; batched GPU path is target architecture | use `cpu_oracle` now; use batched GPU only after it is wired and profiled |
| body-circles/fast render | historical CPU evidence only | ablation/control surface, not the renderer target | keep only when explicitly labeled ablation/control/fallback |

## Safe Speed Constraints

Optimizer changes are safe only if they preserve:

- observation tensor shape and meaning;
- render/downsample semantics;
- one-frame decision cadence;
- reward semantics;
- opponent mixture semantics;
- checkpoint/eval/tournament compatibility.

## Overnight Implications

For the next static overnight manifest:

- keep learning defaults from survival/leaderboard evidence;
- consider collector-width probes separately from quality probes;
- use `batch64` only for the current L4/C256 broad lane; do not generalize it to H100;
- do not confuse H100 compute probes with GPU renderer work;
- keep `cpu_oracle` until a batched GPU backend exists; do not use scalar
  `jax_gpu` for overnight training.

## Next Optimizer Checks Before Scale

- fresh full-loop profile with exact intended overnight settings;
- profile buckets: env step, render/observation, frozen opponent, MCTS/search,
  replay/sample, learner, checkpoints, eval/GIF, artifact I/O;
- confirm dirty-cache and bonus-render path are active;
- record render mode, bonus mode, death mode, trajectory length, collector count,
  sim count, and compute target.

## Non-Goals

- Do not redesign the model input from this lane.
- Do not treat CPU `body_circles_fast + simple_symbols` as the GPU renderer
  implementation target.
- Do not let speed-only settings override survival/leaderboard evidence.
