# MCTX Result Critique, 2026-05-23e

Scope: sidecar optimizer critique. Profile-only. Do not read this as Coach
training advice.

## Plain Verdict

The MCTX compact-slab result is a useful positive signal. It says that replacing
the LightZero CTree/list/control path with a compiled fixed-shape search body can
matter.

But the exact `2.49x-3.43x` win is not yet a hard architecture claim. It can be
inflated or distorted by denominator, warmup, toy-model, root-mask, and summary
math details.

Current safest wording:

```text
MCTX/JAX is a strong profile-only comparator. It beat direct_ctree_gpu_latent on
the tested compact slab rows, but it is not Coach-ready and the exact multiplier
needs a stricter same-manifest rerun.
```

## What Looks Real

- MCTX ran behind `CompactRolloutSlab`, not as a detached toy benchmark.
- It returned a checked `CompactSearchResultV1`.
- The slab applied selected actions into the next env step.
- Replay-index commit counts matched active roots in the four H100 rows.
- The runner summary now uses aggregate `compact_rollout_slab_sec`, not the old
  wrong "all roots divided by last service call" denominator.
- The rows are clearly labeled:
  `profile_only=true`, `not_lightzero_ctree=true`, `not_train_muzero=true`.

## Main Ways This Could Mislead Us

### 1. Direct Baseline And MCTX Were Not Same-Length Rows

The MCTX comparator rows used:

```text
steps=24, warmup_steps=4
```

The direct CTree baseline rows in `opt-compact-slab-h100-main-20260523d` used:

```text
steps=80, warmup_steps=20
```

That is not fatal, because both measurements exclude warmup from
`steps_per_sec`, but it is still a real denominator mismatch. Short MCTX rows can
overread a lucky steady slice, kernel scheduling variance, or post-compile cache
state.

Concrete check:

```text
Run direct_ctree_gpu_latent and MCTX in one manifest, same code state, same
steps/warmup, same B/A/sims/body/trail/render settings, same scalar-off slab
denominator.

Use at least:
  steps=80, warmup_steps=20
  B512/B1024
  sim16/sim32
  2-3 repeats or seeds
```

### 2. Warmup And JAX Compile Are Still Suspicious

The MCTX rows have huge warmup time compared with measured time. Example:

```text
B512/sim16:
  measured_sec = 1.51
  warmup_sec   = 8.10

B1024/sim32:
  measured_sec = 3.02
  warmup_sec   = 9.57
```

That probably includes JAX compile and first-use setup. Four warmup steps may be
enough for the current all-active static shape, but it is not proven enough for
all useful shapes.

Concrete checks:

- Repeat the exact rows with `warmup_steps=4/10/20/40`.
- Add or extract first measured step vs last measured step timing.
- Enable JAX compile logging on one row, or add a profile field that records
  whether `_backend_signature` changed during measured steps.
- Kill any claim if a measured row recompiles.

### 3. Toy JAX Model Is Not The LightZero Model

This is the largest semantic mismatch.

`MctxCompactSearchServiceV1` uses a small toy JAX model:

```text
visual conv -> hidden_dim=64 -> policy/value/reward toy heads
```

It is not the PyTorch LightZero MuZero model. It does not prove that the real
network, real support transforms, real value/reward heads, and real learner
objects can move through the same path at the same speed.

Concrete checks:

- Build a "realistic-cost JAX model" row with a closer conv tower and head sizes.
- Compare model FLOPs/parameter count with the current LightZero model.
- Keep a separate row that uses real LightZero initial/recurrent inference cost
  but MCTX-like fixed-shape search, if possible.
- Do not compare learning quality from this row.

### 4. Search Semantics Are Different

MCTX uses `gumbel_muzero_policy`. The direct baseline uses LightZero CTree MCTS.
Root noise is off for direct rows and effectively absent/different for MCTX.
Temperature, action selection, root priors, visit policy, and value extraction
are not guaranteed equivalent.

This does not invalidate the speed experiment, but it means the result answers:

```text
Can a compiled fixed-shape search backend be much faster?
```

It does not answer:

```text
Can we replace Coach search today without changing training behavior?
```

Concrete checks:

- Add a speed-only label in summaries: `search_semantics_match=false`.
- For promotion, require a separate policy/output parity gate on simple known
  roots, not just speed.

### 5. Root Mask Handling Is Too Easy Right Now

Current MCTX config defaults to:

```text
require_all_roots_active=true
```

The profile uses no-death rows, so all `B * P` roots are active. Real training
will have terminals, resets, maybe inactive players, and masks. If we handle
that by changing shape, JAX can recompile. If we compact active roots, ids and
replay ordering can get tricky.

Concrete checks:

- Forced inactive-root profile: 10%, 50%, 90% inactive.
- Natural-death profile with mixed active/inactive roots.
- Padded fixed-shape MCTX row where inactive roots stay in the array but are
  masked out.
- Non-prefix active root ids test: active roots like `[3, 7, 11, ...]`, not only
  a prefix.
- Check committed replay rows match exactly the previous active root ids.

### 6. Action And Replay Gates Are Necessary But Not Sufficient

Good signs:

```text
MCTX B512 rows:
  total_roots = 24576
  committed_index_rows = 24576

MCTX B1024 rows:
  total_roots = 49152
  committed_index_rows = 49152
```

But this is still a clean no-death, all-active, profile-only slab. It does not
prove the replay payload is trainer-complete.

Concrete checks:

- Random legal-mask stress: mask one or two actions per root and assert selected
  actions are always legal.
- Illegal-action hard fail test.
- Replay payload shape check:
  `visit_policy`, `root_value`, selected action, root id, env row, player id.
- Payload gate test with delayed/out-of-order replay attachment.
- Action checksum/collapse check so a broken search that always selects one
  action cannot look valid by shape alone.

### 7. Env Step Coupling Is Profile-Only

The slab loop does apply selected MCTX actions to the next env step. That is good.

But this profile deliberately excludes important Coach-loop costs:

```text
scalar materialization = off
train_muzero = false
RND = off
learner = absent
real replay buffer = absent
stock evaluator = absent
subprocess env manager = absent
```

So the MCTX result is not a full-loop speedup. It is a search/dataflow ceiling
inside the optimizer harness.

Concrete checks:

- Turn scalar materialization on in a small row and measure the tax.
- Run a profile row with RND collection/estimate enabled if that hook exists for
  this harness.
- Keep a separate stock/full-loop smoke before making any Coach recommendation.

### 8. Summary Math Needs An Artifact Linter

The current summary appears to use the right slab denominator:

```text
probe_roots_per_sec = compact_rollout_slab_total_roots / compact_rollout_slab_sec
steps_per_sec       = steps * batch_size * player_count / measured_sec
```

But we already had one denominator bug in this area, so every future speed claim
should be checked mechanically.

Concrete linter checks:

- `steps_per_sec == steps * batch_size * player_count / measured_sec`.
- `probe_roots_per_sec == compact_rollout_slab_total_roots / probe_total_sec`.
- `probe_total_sec == timings.compact_rollout_slab_sec` for slab rows.
- `compact_rollout_slab_committed_index_rows == compact_rollout_slab_total_roots`
  for no-death all-active rows after warmup.
- `compact_rollout_slab_roots_per_call == batch_size * player_count`.
- No fallback flags:
  `calls_train_muzero=false`, `touches_live_runs=false`, `profile_only=true`,
  `mctx backend=gpu`.
- Summary should expose `search_semantics_match=false` for MCTX.

## One More Caveat From The Newer Scaling Rows

There is already a later MCTX scaling artifact:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/
  opt-mctx-scaling-h100-l4-20260523e/
```

Those rows use different body/trail settings (`4096`) and longer rows
(`steps=40`, `warmup_steps=10`), so they are not apples-to-apples with the first
four-row MCTX result. But they show the exact multiplier is sensitive to row
shape. That supports the stricter same-manifest rerun above.

## Next Checks To Run

1. Matched H100 manifest:

```text
direct_ctree_gpu_latent vs MCTX
B512/B1024
sim16/sim32
steps=80 or 120
warmup_steps=20 or 40
trail_slots/body_capacity identical
materialize_scalar_timestep=false
compact_rollout_slab=true
root_noise_weight=0 for direct
```

2. Warmup sensitivity:

```text
MCTX B1024/sim32
warmup_steps=4,10,20,40
same measured steps
record first/last measured slab timings
```

3. Root-mask stress:

```text
all-active no-death
forced 50% inactive padded roots
natural terminal/reset row
non-prefix active root ids
```

4. Legality/replay stress:

```text
random legal masks
illegal selected-action rejection
committed replay ids equal previous root ids
selected-action checksum/collapse summary
```

5. Realism tax:

```text
toy JAX model
larger realistic JAX model
real LightZero model cost plus MCTX-like search if possible
scalar/RND toggles
```

## Recommendation

Keep the MCTX lane. Do not promote it.

The result is good enough to justify the next wave, but not clean enough to tell
Coach "use this." The next wave should be boring and strict: same manifest,
same denominator, longer warmup, root-mask stress, and an artifact linter.
