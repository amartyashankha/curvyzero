# Compact Slab Implementation Plan, 2026-05-23c

Status: first profile-only implementation slice landed and validated. This lane
must not touch live Coach training runs.

## Plain Goal

Make the profile-only compact slab runnable from the normal hybrid profile
tooling.

The path should prove this loop:

```text
compact batch
-> CompactRootBatchV1
-> CompactSearchServiceV1
-> selected joint action
-> next env step
-> committed CompactReplayIndexRowsV1
```

## What Exists

- `CompactRolloutSlab` already owns selected-action feedback and commits the
  previous search result when the next env batch applies the staged action.
- `HybridBatchedObservationProfileManager` already accepts an optional slab.
- The Modal hybrid boundary runner already has search-service adapters for
  direct CTree and array-ceiling probes.

## Missing Piece, Now Fixed

The top-level profile runner and manifest builder do not expose the slab. That
means the strongest compact dataflow proof is currently local-only and hard to
profile on H100.

2026-05-23c update: the profile runner, Modal hybrid boundary entrypoint,
manifest builder, compact result filter, and manifest-runner summary now expose
the slab as `hybrid_compact_rollout_slab_probe` /
`--compact-rollout-slab-probe`.

## Patch

- Add `hybrid_compact_rollout_slab_probe` to the hybrid profile runner.
- Build a `CompactRolloutSlab` from the selected direct CTree or array-ceiling
  probe.
- Let the slab drive the next joint action.
- Record slab calls, active roots, committed replay rows, and last telemetry.
- Add manifest-builder support and guardrails.

## Guardrails

- Profile-only only.
- No `train_muzero`.
- No live runs.
- Do not run both old compact replay proof and slab proof in the same row; that
  would double-run the search and confuse timing.
- Require a real compact search source: direct CTree arrays or compact
  array-ceiling modes.

## Validation Run

Local focused tests first:

```text
tests/test_source_state_hybrid_observation_profile.py
tests/test_curvytron_hybrid_observation_profile_grid_builder.py
tests/test_source_state_batched_observation_boundary_profile.py
```

2026-05-23c local validation:

```text
ruff: passed for touched profile/slab/boundary/builder/runner/test files
pytest: 12 focused compact-slab/builder/boundary/runner tests passed
```

2026-05-23c Modal smoke:

```text
uv run modal run -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --batch-size 2 --actor-count 1 --steps 1 --warmup-steps 1 --max-ticks 8 \
  --compute gpu-l4-t4 --hybrid-observation-canary \
  --no-hybrid-materialize-scalar-timestep --hybrid-stack-storage-dtype uint8 \
  --hybrid-lightzero-array-ceiling-probe \
  --hybrid-lightzero-array-ceiling-mode mock_search_service \
  --hybrid-compact-rollout-slab-probe \
  --no-hybrid-lightzero-consumer-use-cuda \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --render-surface direct_gray64
```

Result: `ok=true`, `profile_only=true`, `compact_rollout_slab_enabled=true`,
`compact_rollout_slab_calls=1`, `compact_rollout_slab_total_roots=4`,
`compact_rollout_slab_committed_index_row_count=4`, and
`compact_rollout_slab_search_impl=mock_search_service`.

## Remaining Gates

- Run the same slab probe with a real direct CTree compact search source after
  the next same-denominator profile grid is built.
- Run enough warmup/steps for timing; the smoke above proves wiring, not speed.
- Keep this profile-only. It still does not call `train_muzero`, does not touch
  live Coach runs, and is not a Coach-facing training backend.

## 2026-05-23d Denominator Fix

The first warm H100 direct-CTree slab profile worked, but it exposed a tooling
bug:

```text
Bad summary:
  total roots / last slab search-call time

Correct summary:
  total roots / aggregate measured compact_rollout_slab_sec
```

That bug inflated the roots/sec column for slab rows. The manifest runner now
uses aggregate slab wall time as `probe_total_sec`, and keeps last-call search
service timings under explicit `compact_rollout_slab_last_*` fields. The profile
loop also records `compact_rollout_slab_telemetry_totals` for new rows, so
model/search/H2D totals can be read without confusing them with one last call.

Corrected warm H100 B64/A8/sim8 read:

| impl | full profile steps/sec | aggregate slab roots/sec | aggregate slab sec | last service sec |
| --- | ---: | ---: | ---: | ---: |
| `direct_ctree_arrays` | `1560` | `2270` | `1.128` | `0.0505` |
| `direct_ctree_gpu_latent` | `2204` | `3726` | `0.687` | `0.0305` |

Plain read: GPU-latent still wins on this smoke shape, but the win is much
smaller and more honest than the bad last-call denominator suggested. This is
still profile-only evidence.

Active wave launched after the fix:

```text
opt-compact-slab-h100-main-20260523d
opt-compact-slab-h100-direct-arrays-controls-20260523d
opt-compact-slab-h100-service-tax-20260523d
opt-compact-slab-h100-mock-ceiling-20260523d
opt-compact-slab-h100-actor-tax-a8-20260523d
opt-compact-slab-h100-actor-tax-a32-20260523d
```

These rows are profile-only, H100, scalar rows off, root noise off, slab on,
and do not touch live Coach training.
