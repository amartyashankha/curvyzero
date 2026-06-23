# Real JAX MCTX Validation Critique, 2026-05-23i

Scope: validation/risk sidecar only. I inspected the JAX shadow parity wrapper,
its tests, the profile-only MCTX compact search service, and the compact
search/replay contract tests. I did not edit runtime code, touch Coach
defaults, start training, or touch live runs.

## Read Of Current Surface

The shadow model harness is a model-only bridge:

- `src/curvyzero/training/lightzero_jax_shadow_model_parity.py`
- `scripts/probe_lightzero_jax_shadow_model_parity.py`
- `src/curvyzero/infra/modal/lightzero_jax_shadow_model_parity.py`
- `tests/test_lightzero_jax_shadow_model_parity.py`

It correctly labels reports as profile-only, not `train_muzero`, not MCTX,
rejects mutable checkpoint refs, checks required inference-weight coverage, and
compares initial/recurrent outputs plus scalar support-transformed value/reward.
The important current fact is not "exact parity passed"; it is narrower: fresh
model parity passed, trained checkpoint parity showed root representation drift,
and recurrent dynamics/prediction look tight when fed the same PyTorch latent.

The MCTX compact service is currently a speed probe:

- `src/curvyzero/training/mctx_compact_search_service.py`
- `tests/test_mctx_compact_search_service.py`
- `tests/test_mctx_synthetic_benchmark_legality.py`
- `tests/test_compact_search_replay_contract.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- `src/curvyzero/training/compact_rollout_slab.py`

It is loudly profile-only, uses toy JAX parameters, calls
`mctx.gumbel_muzero_policy`, validates legal masks/root values/replay identity,
and currently assumes all roots active for fixed-shape profiling.

## Main Failure Modes

Accepting approximate parity is reasonable for a profile-only lane, but only if
we do not mistake it for a training-compatible replacement. The risks are:

- Representation drift can be amplified by search. A small root-latent error can
  reorder close policy logits, move scalar value enough to change backups, or
  interact with Gumbel noise. "Dynamics is tight from PyTorch latent" is useful,
  but real MCTX starts from the JAX root latent.
- Search semantics differ. MCTX/Gumbel MuZero is not LightZero CTree, so raw
  model parity does not imply action/visit parity versus stock collection.
- Input normalization can silently fork the policy. The toy MCTX path currently
  normalizes `uint8` observations. The real shadow model must pin the exact
  compact observation dtype/range/frame order that the PyTorch LightZero model
  receives.
- BatchNorm and backend precision can look like model bugs. Trained checkpoints
  can magnify PyTorch CUDA/JAX GPU numeric policy differences, especially through
  BN and residual blocks.
- Root contracts are easy to blur. Initial reward is zero-equivalent, recurrent
  reward is support logits, and search needs scalar value/reward semantics.
- Fixed-shape assumptions can hide production cases. The current service rejects
  inactive roots; real profile rows must prove terminal/no-legal rows either are
  excluded before search or remain checked through replay indexing.
- Root-value extraction is brittle if it depends on MCTX internals like
  `search_tree.node_values[:,0]`. A version bump could produce plausible actions
  with missing or shifted replay values.
- The speed signal may shrink with the real model. Toy recurrent cost is not the
  trained conv MuZero cost, and Torch-to-JAX conversion or H2D/D2H payloads can
  erase part of the search win. The strict MCTX rows already show Amdahl pressure
  from observation/env/handoff.
- Stale weight snapshots can become a hidden behavior change. Profile-only is
  fine, but every report must name the immutable checkpoint SHA and the JAX
  snapshot used. Never use `latest.pth.tar` or `ckpt_best.pth.tar`.

## Practical Gates

Do not require exact pixel or exact float parity. Require bounded behavioral
impact and loud containment:

1. Model bridge gate:
   - immutable checkpoint only;
   - all required inference keys consumed;
   - Torch model in eval mode;
   - report Torch/JAX backend, checkpoint SHA, support sizes, action count;
   - compare policy logits, scalar value/reward, root latent max-abs, and
     recurrent-from-JAX-latent plus recurrent-from-Torch-latent.

2. Search-impact gate:
   - fixed observations: zeros, ramp, checkerboard/edge, seeded random, and a
     few real compact roots;
   - fixed legal masks including one-action, two-action, and all-action rows;
   - report selected-action agreement versus a PyTorch-shadow or direct CTree
     comparator where feasible;
   - report visit-policy L1/TV distance, root scalar value delta, illegal mass,
     and top-1/top-2 policy agreement;
   - tolerate numeric drift only if action/visit/value deltas stay small on
     non-degenerate roots and are explained on near-tie roots.

3. Contract gate:
   - `profile_only=true`, `not_train_muzero=true`, trainer defaults unchanged;
   - no mutable checkpoint refs;
   - active-root identity, `policy_env_id`, env row/player, selected action, and
     delayed replay payload handle all round-trip through compact slab tests;
   - visit policy and raw counts assign zero mass to illegal actions;
   - root values are finite, scalar, and sourced from a documented MCTX output.

4. Performance gate:
   - run a tiny real-model MCTX profile before broad grids;
   - separately time representation, recurrent/search body, H2D, D2H, and
     checkpoint/JAX conversion;
   - compare against the matched direct CTree denominator. Kill the lane early if
     the real-model row loses the toy speed signal or only moves the bottleneck
     into conversion/materialization.

## Fast Falsifiers

Structural local guard, no live training:

```bash
uv run pytest \
  tests/test_lightzero_jax_shadow_model_parity.py \
  tests/test_mctx_compact_search_service.py \
  tests/test_compact_search_replay_contract.py \
  -q
```

This quickly falsifies safety labels, mutable-ref rejection, legal-mask
prechecks, compact replay identity, illegal visit mass, and slab telemetry
promotion. It does not prove real-model MCTX.

Read-only immutable checkpoint bridge probe:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_jax_shadow_model_parity \
  --compute l4 \
  --checkpoint-ref training/.../iteration_N.pth.tar \
  --batch-size 4 \
  --atol 5e-3 \
  --rtol 5e-3
```

This falsifies the lane if the current trained checkpoint cannot clear a
documented approximate model bridge with scalar/search-relevant outputs. It
should remain a profile-only read of an immutable checkpoint, not a Coach
training action.

## Recommendation

Wire the real JAX shadow model into MCTX only behind the existing profile-only
compact search service boundary. Treat approximate parity as acceptable if the
search-impact gate is quiet, not because raw tensors are close. Do not promote
to Coach defaults until a real-model compact MCTX smoke proves legal actions,
stable replay payloads, bounded scalar/action drift, immutable checkpoint
lineage, and a retained speed signal against the matched direct CTree baseline.
