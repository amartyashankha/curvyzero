# Real MCTX Shadow Bridge, 2026-05-23i

## Plain Goal

The toy MCTX/JAX profile rows were useful, but they did not use the real
CurvyTron policy. The next honest gate is:

```text
real immutable LightZero checkpoint
-> JAX shadow model
-> MCTX compact search service
-> CompactRolloutSlab
-> selected actions and compact replay rows
```

This is still profile-only. It is not a Coach training backend.

## What Changed

- `MctxCompactSearchServiceV1` can still run the toy JAX model.
- It can now also accept a `JaxMuZeroShadowModel`.
- The real-shadow backend converts LightZero categorical value/reward logits
  into scalar values before giving them to MCTX.
- The Modal hybrid profile wrapper can mount the runs volume read-only for an
  immutable checkpoint ref and build the JAX shadow model from it.
- The compact service contract did not change.

## Current Validation

Local gates passed:

```text
ruff:
  src/curvyzero/training/mctx_compact_search_service.py
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
  tests/test_mctx_compact_search_service.py

pytest:
  tests/test_mctx_compact_search_service.py
  tests/test_lightzero_jax_shadow_model_parity.py
  10 passed
```

A tiny Modal L4 real-checkpoint smoke passed:

```text
checkpoint:
  cz26a-r001 ... iteration_260000.pth.tar

shape:
  B2, actor1, steps2, warmup1, sim2

result:
  ok=true
  calls_train_muzero=false
  stock_lightzero_integrated=false
  touches_live_runs=false
  jax backend=gpu
  shadow coverage ok=true
  required inference keys consumed=123
```

H100 same-shape profile rows then passed:

| shape | scalar rows | MCTX real checkpoint | direct CTree GPU latent | speedup |
| --- | --- | ---: | ---: | ---: |
| B64/A4/sim8, 80 measured + 20 warmup | on | `3,817` steps/sec | `2,813` steps/sec | `1.36x` |
| B512/A8/sim8, 40 measured + 10 warmup | on | `10,037` steps/sec | `4,233` steps/sec | `2.37x` |
| B512/A8/sim8, 40 measured + 10 warmup | off | `14,257` steps/sec | `8,999` steps/sec | `1.58x` |
| B1024/A16/sim8, 30 measured + 10 warmup | off | `19,334` steps/sec | `8,792` steps/sec | `2.20x` |

Plain read:

```text
The real-checkpoint MCTX/JAX bridge is a real profile-only speed signal.
It gets more useful at larger root batches.

But it is still not a Coach backend:
  MCTX uses Gumbel MuZero semantics, not LightZero CTree.
  The shadow model has close-but-not-exact checkpoint parity.
  The profile shell does not call train_muzero.
```

The scalar-row result is important:

```text
Turning scalar LightZero timestep materialization off changed:
  direct CTree: 4,233 -> 8,999 steps/sec
  MCTX:        10,037 -> 14,257 steps/sec

This says the next architectural win is not only search. It is also avoiding
full scalar LightZero row materialization on the hot path and keeping replay
payloads compact until a real trainer boundary needs them.
```

## Current Risk

Checkpoint parity is close but not exact. The root representation latent drifts
by a few thousandths on trained checkpoints. Dynamics and prediction are much
tighter when fed the same latent. That means this bridge is good enough to test
profile speed and search behavior, but it is not yet Coach-facing.

## Same-Root Comparator Smoke, 2026-05-23j

A same-root comparator now exists in the profile wrapper:

```text
one CompactRootBatchV1
-> primary: real-checkpoint MCTX/JAX compact service
-> reference: same-checkpoint LightZero direct CTree compact service
-> return primary actions unchanged
-> record comparison telemetry
```

Tiny H100 smoke:

```text
shape: B2, actor1, steps2, warmup1, sim2
checkpoint: iteration_260000.pth.tar
primary: MCTX/JAX shadow model
reference: direct_ctree_gpu_latent
materialized_timestep_count=0
identity_match=true
```

Comparator telemetry:

```text
selected-action match: 2 / 4 roots = 0.50
visit L1 mean: 1.4623
visit L1 max:  1.9629
root value abs diff mean: 57.42
root value abs diff max:  65.06
```

Plain read:

```text
The comparator wiring is real: both searches saw the same roots and the same
checkpoint identity.

The semantic match is not cleared. Some of this may be expected because MCTX
uses Gumbel MuZero semantics and LightZero CTree uses its own CTree search and
action-selection rules. The root-value difference may also include value-scale
or summary-definition mismatch. Do not promote MCTX until a larger sim8 row
and a value-scale diagnostic explain this.
```

Larger H100 sim8 rows:

| reference | shape | identity | action match | visit L1 mean | root value abs diff mean | steps/sec |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` | B64/A4/steps20/warmup5/sim8 | true | `44/128 = 0.3438` | `1.2799` | `15.97` | `2033` |
| `direct_ctree_arrays` | B64/A4/steps20/warmup5/sim8 | true | `48/128 = 0.3750` | `1.2877` | `15.97` | `2261` |

Plain read:

```text
The larger sim8 rows did not clear the semantic concern. Both direct CTree
reference implementations disagree with MCTX in nearly the same way, so the
problem is probably not the direct-CTree adapter.

Next diagnostic: compare pre-search root model values and policy logits. If
those match, the mismatch is search semantics/value backup. If they do not, the
problem is input normalization, checkpoint translation, or model parity.
```

Pre-search diagnostic row:

```text
app: ap-9v6Qj9C1sptGIKx7chEG19
shape: B2, actor1, steps2, warmup1, sim8
reference: direct_ctree_gpu_latent
identity_match=true
predicted policy logits abs diff mean: 0.0000084
predicted policy logits abs diff max:  0.0000203
predicted value abs diff mean:         0.0090
predicted value abs diff max:          0.0219
searched root value abs diff mean:     17.67
visit L1 mean:                         1.34
selected action match:                 0.50
```

Plain read:

```text
The model/input bridge is close before search. The large post-search deltas are
therefore not explained by bad checkpoint loading or root order. They are
MCTX/Gumbel search semantics, value backup, or value summary differences versus
LightZero CTree.
```

## Next Gates

1. Finish search-impact parity: same roots through direct CTree and MCTX,
   compare selected actions, visit distributions, root values, and value-scale
   diagnostics on a larger sim8 row. Current read: pre-search model parity is
   close; the remaining mismatch is in search semantics/value backup.
2. Add terminal/inactive-root stress before claiming trainer-shaped semantics.
3. Add a compact replay/trainer boundary design that avoids scalar timestep
   materialization on the hot path.
4. Keep every row labeled profile-only until a separate Coach-facing design
   exists.
