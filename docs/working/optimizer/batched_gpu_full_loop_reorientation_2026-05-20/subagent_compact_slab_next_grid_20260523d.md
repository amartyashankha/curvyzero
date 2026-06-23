# Compact Slab Next Grid, 2026-05-23d

Scope: read-only review of compact slab docs, profile tooling, and the current
artifact
`artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-slab-direct-warm-telemetry-20260523`.
No code, live runs, checkpoints, evals, tournaments, GIFs, or volumes were
touched.

## Current Read

The slab now proves the important profile-only loop:

```text
compact batch -> compact search -> selected joint action -> next env step
-> compact replay index rows
```

The current H100 B64/A8/sim8 artifact is a wiring and warm timing proof, not a
final denominator. It shows:

| impl | steps/sec | probe roots/sec | probe sec | search sec | committed rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_arrays` | `1560` | `50665` | `0.0505` | `0.0417` | `2560` |
| `direct_ctree_gpu_latent` | `2204` | `83919` | `0.0305` | `0.0237` | `2560` |

Plain read: GPU-latent direct CTree is the real-search baseline to carry
forward. Direct arrays remains a control. The slab summary also still needs
caution on `h2d_sec`: nested telemetry shows H2D work, while the flattened
summary can report `0.0` for these rows. For the next grid, use the nested
profile telemetry as source of truth if the flat column looks wrong.

## Recommended Next Grid

Use H100 first. Keep `profile_only=true`, `calls_train_muzero=false`,
`touches_live_runs=false`, `materialize_scalar_timestep=false`, root noise
forced to `0.0`, renderer backend
`jax_gpu_persistent_policy_framebuffer_profile`, render surface `direct_gray64`,
input mode `host_uint8`, and slab enabled.

Main real-search grid:

| compute | batch | actors | sims | impl |
| --- | ---: | ---: | ---: | --- |
| H100 | 256 | 16 | 8,16,32 | `direct_ctree_gpu_latent` |
| H100 | 512 | 16 | 8,16,32 | `direct_ctree_gpu_latent` |
| H100 | 1024 | 16 | 8,16,32 | `direct_ctree_gpu_latent` |

Control rows:

| compute | batch | actors | sims | impl |
| --- | ---: | ---: | ---: | --- |
| H100 | 512 | 16 | 8,16,32 | `direct_ctree_arrays` |
| H100 | 1024 | 16 | 16 | `direct_ctree_arrays` |

Ceiling rows behind the same slab contract:

| compute | batch | actors | sims | array-ceiling mode |
| --- | ---: | ---: | ---: | --- |
| H100 | 512 | 16 | 16,32 | `service_tax_probe` |
| H100 | 512 | 16 | 16,32 | `mock_search_service` |
| H100 | 1024 | 16 | 16 | `service_tax_probe` |
| H100 | 1024 | 16 | 16 | `mock_search_service` |

Actor-count tax rows:

| compute | batch | actors | sims | impl |
| --- | ---: | ---: | ---: | --- |
| H100 | 512 | 8,16,32 | 16 | `direct_ctree_gpu_latent` |

L4 sanity rows, after H100 finishes:

| compute | batch | actors | sims | mode |
| --- | ---: | ---: | ---: | --- |
| L4/T4 | 512 | 16 | 16 | `direct_ctree_gpu_latent` |
| L4/T4 | 512 | 16 | 32 | `direct_ctree_gpu_latent` |
| L4/T4 | 512 | 16 | 16 | `service_tax_probe` |

This is about 24 rows. Run `80` measured steps and `20` warmup steps for H100
unless queue time gets silly; use at least `60/15` for L4. The important thing
is stable warm rows, not short smoke timing.

## Axes That Matter

- `simulations`: most important. It tells us whether MCTS/search or the rest of
  the compact loop is the wall.
- `batch_size`: second most important. B512 has been the cleanest denominator;
  B1024 tests whether batching still helps or just adds env/packaging cost.
- `actor_count`: useful but do not cross it with everything. Test only B512/sim16
  at A8/A16/A32 to find orchestration overhead.
- `compute`: H100 first, L4 only as a sanity/cost row. Do not run a full L4 grid
  until the H100 shape is clear.
- `backend`: `direct_ctree_gpu_latent` is the real baseline. `direct_ctree_arrays`
  is a control. `service_tax_probe` and `mock_search_service` are ceilings, not
  training claims.

## Redundant Rows

- `stock_facade` with slab: invalid for compact replay/search proof.
- `compact_service_replay_proof` together with slab: double-runs search and
  pollutes timing.
- `materialize_scalar_timestep=true`: not part of the compact slab question.
- `resident_torch_reuse`: stale-input ceiling, not a valid slab proof.
- Full L4 cross product: too much cost/noise before the H100 denominator is
  understood.
- Every render-mode/body-circle/browser-line variant: this grid is about the
  compact search/dataflow path, not visual fidelity.
- `direct_ctree_gpu_latent_precomputed_recurrent`: keep only as a separate
  synthetic falsifier; it is not a valid training-shaped row.

## Falsify This Lane If

- `direct_ctree_gpu_latent` slab fails to beat `direct_ctree_arrays` after warmup
  on B512/sim16 and B512/sim32.
- Service-tax/mock ceilings are only about `<=1.3x` faster than real search on
  the same denominator; then the remaining headroom is too small for this slice.
- B1024 is not faster than B512 or becomes unstable; then bigger batches are not
  the immediate answer.
- Actor A32 is worse than A16; then orchestration/fan-in is already a wall.
- Any row has nonzero fallbacks, illegal actions, mismatched committed index row
  counts, `python_rows_materialized > 0`, root observation copies, RND rows in
  the hot path, `calls_train_muzero=true`, or `touches_live_runs=true`.
- The timing summary cannot separate search/model/H2D/env buckets. If the flat
  columns disagree with nested telemetry, fix the summary before making a speed
  claim.

## One-Line Recommendation

Run a warm H100 slab grid centered on `direct_ctree_gpu_latent` at
B256/B512/B1024 and sim8/sim16/sim32, with direct-array controls plus
service-tax/mock ceilings; only then run a tiny L4 sanity grid on the winning
shape.
