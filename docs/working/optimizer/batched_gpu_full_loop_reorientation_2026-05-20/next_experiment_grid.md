# Next Experiment Grid

Date: 2026-05-20

Status: planned, do not touch live training runs.

## Plain Goal

We need to stop guessing which wall is next. The direct GPU observation surface
is fast locally, but it is not a full-loop result. The next grid should answer
three questions:

1. What is the current stock full-loop baseline after recent repo changes?
2. Does the direct GPU vector facade stay fast after row/player stack and
   payload work?
3. If observation gets cheap, does the wall move to MCTS/search, RND, learner,
   host/process overhead, or payload/scalarization?

## Run Rules

- Do not modify or stop live training runs.
- Disable eval, GIF, tournament, and checkpoint clutter for profile rows.
- Keep real trainer/tournament defaults on `cpu_oracle` until promotion gates
  pass.
- Do not use scalar `jax_gpu` as the speed path.
- Surface/vector canaries must say `calls_train_muzero=false`.
- Full-loop rows must report matched env steps, MCTS roots, sim count, replay
  samples, learner calls, RND calls, and warmup policy.
- Use enough warmup to avoid JAX compile and first-step noise. For surface rows,
  use at least `warmup_steps=16` when measuring `steps=256`.
- Treat deltas below about `10%` as noisy unless repeated and interleaved.

## Prioritized Rows

2026-05-20 reorientation: the stock H100 C512/sim4 repeat is now a baseline
variance diagnostic, not the main workstream. Keep collecting it if already
launched, but do not let it delay the vector/full-loop bridge.

2026-05-20 zero-observation update: C64 and C128 zero-observation stock manager
rows passed. This changes the next experiment from "is the batched manager
structurally hopeless?" to "how close can real GPU-render observations get to
the zero-observation ceiling?"

2026-05-21 update: the C512 post-patch real GPU row reached about `1439.84`
steps/s. The existing C512 zero-observation ceiling is about `1805.22` steps/s.
So a perfect observation path would now be only about `1.25x` better at that
width. Continue renderer cleanup only where it is simple or directly reduces
manager/stack overhead; otherwise move the next big experiment toward
manager/policy/search architecture.

2026-05-21 later update: semantic attestation is now proven on fresh stock
profile smoke `opt-semantic-identity-smoke-20260521a`. The H100/L4 split
refresh repeated the earlier bottleneck read: model initial inference is much
faster than public LightZero collect-forward. The next experiment is therefore
not another broad renderer gate; it is a deeper split inside
`collect_mode.forward`.

2026-05-21 update: that deeper split has returned. On H100, MCTS collect
spent `35.36s` in collect-forward, model calls were only `1.81s`, MCTS search
was `10.97s`, raw ctree traverse/backpropagate was only `0.98s`, and
outside-MCTS wrapper/output residual was `24.40s`. Pure-policy collect reached
about `6286` roots/sec versus MCTS collect at about `2572` roots/sec. The new
priority is a profile-only replacement-ceiling toy, not more renderer work and
not another broad hardware sweep.

2026-05-21 latest update: the array-ceiling toy returned. H100
`policy_arrays` reached `9957.97` roots/sec; H100 `recurrent_toy` with 8 real
batched recurrent model calls reached `8681.01` roots/sec. This is about
`3.4x` faster than public MCTS collect on the same profile family. The result
does not prove real MCTS can run that fast, but it does prove the public MCTS
branch boundary has enough removable overhead to justify a real compact
arrays-in / arrays-out design. A second lane should split or remove the current
host-stack -> Torch H2D copy, because the H100 toy spent about `2.4s` in H2D
inside the array-ceiling bucket.

2026-05-21 direct update: the first real direct CTree arrays profile probe has
now returned and input modes are wired into it. Matched useful H100 rows:
stock facade `2419.81`, old direct host uint8 `3859.44`, current direct host
uint8 `5247.95`, current direct pinned uint8 `4678.23`, and resident reuse
ceiling `5820.96` roots/sec. This changes the next grid: do not build another
toy first. The next useful work is direct-vs-stock parity and deeper split of
the remaining direct path wall.

Longer input-mode repeats refined that read: host uint8 `4111.80`, pinned
uint8 `4513.15`, resident reuse `5537.40` roots/sec. So pinned input is a
modest stable win, resident reuse is still stale-input-only, and neither
displaces the main next gate: direct-vs-stock parity plus the remaining direct
path split.

| Priority | Row | Purpose | Trust Condition |
|---:|---|---|---|
| 1 | Direct CTree arrays fixed-seed parity tests | Decide whether the profile-only direct lane can ever become a trainer candidate | Must preserve legal masks, to_play, root noise, temperature, support/value/reward transforms, visit counts, decoded actions, and output schema |
| 2 | Direct path deeper split | Price remaining H100 wall after output fast path: MCTS search, root prep, H2D, observation, and model calls | Same B512/A16/sim8 shape, enough warmup, compare stock facade/direct/initial-only |
| 3 | Longer direct input-mode repeat | Decide whether pinned input is a real total-wall win or only an H2D bucket win | Same current image, longer warmup, host uint8 vs pinned uint8 vs resident ceiling |
| 4 | Matched stock full-loop A/B/C: CPU oracle, batched profile manager, zero observation | Reconnect boundary results to actual `train_muzero` profile speed | Fresh rows must pass `--require-attestation`; no eval/GIF; no live-run writes; include normal-death/autoreset smoke before launch advice |
| 5 | RND matched pair: no RND vs `rnd_meter_v0`, H100 C512, cadence `10` and `100` | Measure current training bells-and-whistles cost | Verify RND source counters and estimate/train calls |
| 6 | Targeted renderer/stack cleanup only if it also reduces manager/stack host work | Close part of remaining observation gap without a rabbit hole | Fresh Amdahl row must show render/stack is again a meaningful fraction |
| 7 | Payload/process probe: base vs subprocess at C32/C128/C512 | Decide whether host/process/IPC becomes the next lane | Pickle proxy is directional, not a perfect IPC timer |

## Current Recommendation

Do not launch a new training default from the direct GPU path yet. The profile
lane now shows real promise, and attested stock profile rows are working. The
direct CTree arrays result says the compact search-boundary replacement is worth
continuing, but it is still profile-only. Keep it out of Coach defaults until
fixed-seed MCTS parity gates pass, then reconnect to matched stock full-loop
A/B/C rows.

## Concrete Next Profile Grid

Use the profile-only boundary manifest preset for the next direct comparison:

```bash
uv run python scripts/build_curvytron_hybrid_observation_profile_grid.py \
  --experiment-id opt-direct-ctree-fixed-denom-20260522a \
  --next-direct-ctree-comparison-preset
```

This emits six rows: H100 and L4/T4, each at B512/A16/sim8 with 60 measured
steps and 15 warmup steps, comparing `stock_facade`, `direct_ctree_arrays`, and
`direct_ctree_gpu_latent` with `host_uint8` input and scalar materialization
off. To skip the cheap hardware sanity rows, add `--no-include-l4`.

This manifest is still profile-only: `calls_train_muzero=false`,
`touches_live_runs=false`, and each row carries
`comparison_group=mcts_arrays_boundary_fixed_denominator` plus the fixed
denominator fields. Dry-run or launch only this manifest; do not touch live
training runs or volumes.
