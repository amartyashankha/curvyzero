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
| `batch32` | better than `batch64` in matched learning/leaderboard evidence | no conflict | default |
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
- do not scale `batch64`;
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
