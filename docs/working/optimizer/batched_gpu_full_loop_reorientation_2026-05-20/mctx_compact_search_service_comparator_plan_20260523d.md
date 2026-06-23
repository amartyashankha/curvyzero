# MCTX Compact Search Service Comparator, 2026-05-23d

Scope: optimizer profile-only. This does not touch live Coach runs, does not
call `train_muzero`, and is not a training recommendation.

## Plain Goal

We need to know whether the current wall is the LightZero CTree/list/control
search path.

The clean test is:

```text
CompactRootBatchV1
-> MCTX/JAX fixed-shape search
-> CompactSearchResultV1
-> CompactRolloutSlab selected actions
-> next env step
-> CompactReplayIndexRowsV1
```

If this beats `direct_ctree_gpu_latent` on the same compact slab denominator,
then a compiled/device-resident search body is worth deeper work.

If it does not beat it, stop treating MCTX as the next main speed lane.

## What Landed

Code:

- `src/curvyzero/training/mctx_compact_search_service.py`
  adds `MctxCompactSearchServiceV1`.
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  wires `--hybrid-mctx-compact-search-probe` into the hybrid profile app.
- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
  can emit MCTX compact slab rows.
- `src/curvyzero/training/compact_rollout_slab.py`
  now maps MCTX timing/byte telemetry into aggregate slab summaries.

Hard labels:

```text
profile_only=true
not_lightzero_ctree=true
not_train_muzero=true
calls_train_muzero=false
touches_live_runs=false
trainer_defaults_changed=false
```

## First Smoke

Command shape:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --hybrid-observation-canary \
  --compute gpu-h100 \
  --batch-size 4 \
  --actor-count 2 \
  --steps 2 \
  --warmup-steps 1 \
  --trail-slots 128 \
  --body-capacity 128 \
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --hybrid-batched-stack-probe-simulations 0 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-compact-rollout-slab-probe \
  --hybrid-mctx-compact-search-probe \
  --hybrid-mctx-num-simulations 2 \
  --hybrid-mctx-hidden-dim 16 \
  --hybrid-mctx-visual-channels 4
```

Result:

```text
ok=true
jax backend=gpu
LightZero=missing
torch=missing
compact_rollout_slab_calls=2
compact_rollout_slab_total_roots=16
compact_rollout_slab_committed_index_row_count=16
compact_rollout_slab_search_impl=mctx_compact_search_service_profile_only_v0
compact_rollout_slab_profile_only=true
calls_train_muzero=false
touches_live_runs=false
trainer_defaults_changed=false
```

Important detail: the tiny smoke is not a speed claim. Most time was renderer
warmup/persistent update, and the batch was tiny. It only proves wiring.

## Next Comparator Grid

Run profile-only H100 rows:

```text
B512/A16/sim16
B512/A16/sim32
B1024/A16/sim16
B1024/A16/sim32
```

Use:

```text
materialize_scalar_timestep=false
compact_rollout_slab_probe=true
mctx_compact_search_probe=true
hidden_dim=64
visual_channels=8
root noise: none, because this is not LightZero
```

Compare against existing `direct_ctree_gpu_latent` rows:

```text
B512/A16/sim16: 6522 steps/sec
B512/A16/sim32: 4177 steps/sec
B1024/A16/sim16: 6992 steps/sec
B1024/A16/sim32: 5314 steps/sec
```

## H100 Comparator Result

Artifact:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-mctx-compact-slab-h100-20260523d/
```

Rows:

| row | shape | MCTX steps/sec | direct CTree baseline | speedup |
| --- | --- | ---: | ---: | ---: |
| `001` | B512/A16/sim16 | `16,250` | `6,522` | `2.49x` |
| `002` | B512/A16/sim32 | `14,306` | `4,177` | `3.43x` |
| `003` | B1024/A16/sim16 | `20,557` | `6,992` | `2.94x` |
| `004` | B1024/A16/sim32 | `16,255` | `5,314` | `3.06x` |

All four rows kept the profile-only labels:

```text
calls_train_muzero=false
touches_live_runs=false
profile_only=true
promotion_eligible=false
compact_rollout_slab_search_impl=mctx_compact_search_service_profile_only_v0
```

Plain read:

```text
This is a real positive signal for compiled/device search on the same compact
slab denominator. It is not Coach-ready, because it uses a toy JAX model and
MCTX search semantics rather than the trusted LightZero PyTorch model plus
LightZero CTree.
```

## Keep / Kill Rule

Keep this lane only if the warmed H100 slab rows clearly beat
`direct_ctree_gpu_latent` while keeping replay-index commits and profile-only
labels.

2026-05-23 result: keep the lane for the next optimizer wave. The first real
grid beat the direct CTree rows by about `2.5x-3.4x`.

Kill or demote if:

- JAX recompiles during measured steps;
- legality or replay-index checks fail;
- H2D/render/env dominate so search speed is irrelevant;
- it only wins on tiny toy rows;
- it cannot beat direct CTree on B512/B1024 sim16 or sim32.

## Caveat

This uses a toy JAX model, not the LightZero PyTorch MuZero model. It answers a
speed architecture question, not a learning-quality question.
