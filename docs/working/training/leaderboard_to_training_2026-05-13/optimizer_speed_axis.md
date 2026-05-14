# Optimizer Speed Axis

## Purpose

Optimizer recommendations answer a different question than survival or
leaderboard analysis:

```text
What can run fast enough at scale without changing the learning/evaluator contract?
```

This doc keeps speed advice separate from policy-quality advice.

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
two-seat adapter. `body_circles_fast` is different: it is the current stock-path
fast approximation. The 212-run matched evidence did not show a meaningful
learning gap versus `browser_lines`, so it is a legitimate speed/fidelity
candidate, not automatically a bad surface.

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
- Render remains important for long-survival/no-death regimes, but full-loop
  bottleneck is not only render.
- GPU render is promising research, not trusted training plumbing yet.
- CPU dirty/cache browser-lines path is the current trusted fidelity target.

## Interaction With Training Recommendations

| Setting | Learning/leaderboard read | Optimizer read | Current stance |
| --- | --- | --- | --- |
| `batch32` | better than `batch64` in matched learning/leaderboard evidence | no conflict | default |
| `sim8` | default; `sim16` not earning quality cost | sim16 may be affordable | keep sim16 sentinel only |
| `collector32` | clean baseline | C64/C96 improve throughput | consider C64 probe separately |
| browser render | same-checkpoint quality tied with fast | current trusted visual surface | promote if speed is now acceptable |
| body-circles/fast render | matched 212-run evidence roughly tied with browser | may be cheaper in some short regimes; not always faster after dirty-cache | valid approximation candidate; keep paired until Coach decides visual contract |

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
- do not switch visual surface away from CPU-reference `browser_lines` without
  a separate fidelity decision, but treat that decision as open rather than
  forbidden;
- if browser render is now fast enough, use browser more freely but keep matched
  fast controls where needed.

## Next Optimizer Checks Before Scale

- fresh full-loop profile with exact intended overnight settings;
- profile buckets: env step, render/observation, frozen opponent, MCTS/search,
  replay/sample, learner, checkpoints, eval/GIF, artifact I/O;
- confirm dirty-cache and bonus-render path are active;
- record render mode, bonus mode, death mode, trajectory length, collector count,
  sim count, and compute target.

## Non-Goals

- Do not redesign the model input from this lane.
- Do not use GPU render in production until parity and handoff are proven.
- Do not let speed-only settings override survival/leaderboard evidence.
