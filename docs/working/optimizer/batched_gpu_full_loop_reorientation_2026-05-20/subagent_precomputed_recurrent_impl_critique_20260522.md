# Precomputed Recurrent Falsifier Implementation Critique - 2026-05-22

Scope reviewed:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
- focused tests in `tests/test_source_state_batched_observation_boundary_profile.py` and
  `tests/test_curvytron_hybrid_observation_profile_grid_builder.py`

I did not edit runtime code and did not touch live training runs.

## Verdict

The `direct_ctree_gpu_latent_precomputed_recurrent` mode is fit as a
profile-only falsifier, with one caveat: treat it as a lower-bound/shape probe
for "CTree + Python/list/control + mandatory CTree output D2H after recurrent is
removed", not as an exact cost decomposition of the real recurrent path.

The implementation really does bypass `model.recurrent_inference` inside the
measured search loop, keeps the LightZero CTree traverse/backprop structure, and
is kept out of the fixed default comparison preset. The focused CPU tests pass.

## Findings

1. Medium: `model_eval_count` is misleading for this mode.

   `model_eval_count` is still reported as `active_root_count * (1 + num_simulations)`
   even when recurrent calls are zero. The recurrent call telemetry is correct:
   `lightzero_consumer_model_recurrent_inference_calls == 0`, but summary tooling
   or humans may read `model_eval_count` as actual model work. Minimal follow-up:
   either add a separate `synthetic_recurrent_eval_count`/`actual_model_eval_count`
   field or document that `model_eval_count` is the logical MuZero search shape.

2. Medium: synthetic outputs change search dynamics, not just recurrent cost.

   Reward/value/policy are all zero tensors. That is shape-compatible and useful,
   but it changes priors, values, min/max stats, and likely tree expansion order
   versus real recurrent outputs. CTree call counts stay matched, but CTree work
   distribution may not be identical. Interpret deltas as "if recurrent vanished
   and outputs were trivial/uniform" rather than "exact recurrent launch + D2H
   subtraction."

3. Low/medium: the mode removes a little more than recurrent launch/D2H.

   The real branch inverse-transforms recurrent reward/value before D2H; the
   precomputed branch feeds already-plain zeros. Zero is semantically safe for
   LightZero scalar-transform expectations, but the scalar-transform cost is also
   skipped. That is probably fine for a falsifier, but worth naming in charts.

4. Low: it still pays some costs that a purer "precomputed outputs" loop would not.

   The loop still creates `last_actions_tensor`, synchronizes it, indexes/copies
   latents, and writes `latent_pool[simulation_index + 1]` even though
   `recurrent_inference` is not called. This is good if the goal is "same
   direct_ctree_gpu_latent skeleton minus recurrent", but it can overstate the
   irreducible CTree/list/control floor for a future device-native search.

5. Low: `precomputed_recurrent_payload_index_sec` is named stronger than what it measures.

   The payload is one resident constant reward/value/policy tensor reused every
   simulation; there is no per-simulation or per-leaf synthetic output pool
   indexing. The timer mostly measures assignment overhead. Rename only if this
   becomes a public metric; otherwise a doc note is enough.

## Shape / Dtype / Device Notes

- CPU path is covered by the new test and passes.
- CUDA path is not directly tested locally. The tensor construction is device-aware
  and should run on CUDA, but the real confidence gate should be a one-row H100
  canary asserting:
  - recurrent calls are zero;
  - `gpu_latent_precomputed_recurrent_enabled` is true;
  - `gpu_latent_search_output_d2h_bytes == roots * sims * (2 + ACTION_COUNT) * 4`;
  - no illegal actions;
  - CTree traverse/backprop calls equal `num_simulations`.
- Synthetic reward/value are plain float32 tensors, not scalar-support logits.
  That matches what CTree receives after inverse scalar transform. For zero,
  semantic parity is safe.

## Test Coverage Gaps

- Add a precomputed recurrent mixed-mask test. Existing skip-recurrent coverage
  uses all-actions-legal masks, so it does not exercise the slower legal-action
  decode path for this mode.
- Add a zero-active-root test for the new telemetry keys, or tolerate missing
  keys in consumers. The active-root path emits the precomputed flags; the
  no-active-root early return does not currently mirror all GPU-latent telemetry.
- Add a manifest/preset assertion that the fixed comparison preset excludes
  `direct_ctree_gpu_latent_precomputed_recurrent`. The current test effectively
  does this via exact rows; keep it.

## Profile-Only Boundary

Looks good:

- The mode lives in the Modal boundary profile sidecar.
- Grid rows are marked `profile_only: True`, `calls_train_muzero: False`,
  `touches_live_runs: False`.
- The impl is accepted as an explicit choice, but `NEXT_DIRECT_CTREE_COMPARISON_IMPLS`
  excludes it from the default fixed-denominator comparison grid.

## Minimal Patch Recommendation

No urgent code patch required before running the falsifier.

Best tiny follow-up patch:

- Add telemetry fields:
  - `lightzero_consumer_actual_model_eval_count`
  - `lightzero_consumer_synthetic_recurrent_eval_count`
  - mirrored `lightzero_mcts_arrays_boundary_*` versions
- Add one mixed-mask test for
  `direct_ctree_gpu_latent_precomputed_recurrent`.

## Verification

Ran:

```bash
uv run pytest \
  tests/test_source_state_batched_observation_boundary_profile.py::test_lightzero_mcts_arrays_boundary_precomputed_recurrent_skips_model_recurrent \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py::test_hybrid_profile_grid_can_emit_precomputed_recurrent_ctree_boundary_rows \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py::test_next_direct_ctree_comparison_preset_emits_fixed_denominator_grid
```

Result: `3 passed`, with only existing third-party `treevalue` deprecation
warnings.
