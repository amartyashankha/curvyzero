# Boundary Test Matrix - 2026-05-21

Purpose: keep the optimizer lane fast without letting it become fast and wrong.

## Plain Read

The current profile signal says render is no longer the biggest wall in the
real-consumer rows. That does not reduce the need for render/observation tests.
It makes the tests more important, because future optimization will cross more
boundaries: GPU observation, LightZero policy/search, RND, reset/death, and
summary tooling.

## P0 Boundaries

| Boundary | What can go wrong silently | Current status | Next proof |
| --- | --- | --- | --- |
| GPU observation surface | Fast path returns plausible `[4,64,64]` frames with wrong symbols, stale trails, wrong player view, or drift hidden by tolerant wording. | CPU-direct/JAX direct and divergence smokes exist; persistent path is explicitly approximate. | Add same-surface persistent-vs-stateless exact check and keep `direct_gray64` labeled as approximate versus CPU oracle. |
| Row/player order | `(row, player)` order gets transposed but shapes and speed still look fine. | Many local row-major tests exist; collect-forward fake test checks flattened root count/order metadata. | Add or confirm fake consumer sentinel that checks actual pixels received by the consumer in row-major order. |
| Stack FIFO/reset | Latest frame, previous frames, or selected-row reset shifts incorrectly. | Local stack push/reset tests exist. | Add mixed terminal/live row fixture so terminal rows do not pollute live roots. |
| LightZero handoff | Wrong `to_play`, zero masks forwarded, illegal actions ignored, scalar materialization hidden, or profile label overclaims GPU residency. | Collect-forward now uses `to_play=-1`, filters zero masks, and labels CPU-tree-inclusive timing. Initial-inference probe exists. | Add speed-row attestation and output-decode variants before using this as behavior loop proof. |
| RND | Predictor trains too rarely, novelty is only batch-relative, target model changes, or latest-frame source is wrong. | Stock-profile `rnd_meter_v0` gate has passed as an overhead/safety row: reward-model entrypoint ran, predictor changed, target stayed frozen, and target reward stayed unchanged. | Keep it as a separate axis; do not treat meter rows as positive-RND learning proof while batch-normalized novelty remains unresolved. |
| Normal death/autoreset | Profile no-death rows look fast but terminal/final observation behavior breaks in real training. | Stock `train_muzero` normal-death/autoreset gate passed after partial-render and complete-row-omission fixes. | Keep normal-death rows as semantic gates, not throughput anchors, because live root batches collapse after deaths. |
| Profile summaries | A fast row is summarized without saying surface, death mode, RND mode, scalar materialization, or consumer semantics. | Summary tooling now has profile-row attestation and can fail under-specified rows with `--require-attestation`. Fresh stock profile smoke `opt-semantic-identity-smoke-20260521a` now passes attestation after the runner JSON parser fix and stock Torch pin. | Keep using attestation before turning speed tables into Coach advice; old artifacts before `semantic_identity` remain historical only. |

## P0 Local Test Candidates

1. Fake consumer row/player/stack sentinel:
   a deterministic frame value per `(row, player, step)` and an assertion that
   flattened consumer input is `[row0p0, row0p1, row1p0, row1p1, ...]`.

2. Mixed terminal/live final-observation fixture:
   one row terminal and one row live in the same batch; terminal roots filtered,
   terminal final observation preserved, live row ids not compacted incorrectly.

3. Persistent cursor regression / row-selective reset:
   one row cursor regresses, another appends; stale trails must not survive in
   the regressed row.

   Status: added locally in
   `tests/test_source_state_batched_observation_boundary_profile.py` as
   `test_persistent_delta_state_row_selective_cursor_regression_resets_only_regressed_row`.

4. LightZero profile semantic attestation:
   a small artifact/summary validator that requires backend, surface, dtype,
   death mode, RND mode, `to_play`, mask filtering, scalar materialization, and
   consumer semantics.

   Status: guard now exists in
   `scripts/summarize_curvytron_optimizer_profile_results.py`. It requires the
   stock/profile identity, observation contract, renderer modes, death/RND
   mode, denominator source, MCTS/root counts, and core timers. It also fixes
   the old ambiguous table column by separating `render_mode` from
   `render_sec`. New compact outputs also carry
   `semantic_identity` with observation dtype, scalar materialization,
   `to_play`, zero-mask/action-mask, and consumer-semantics labels; full
   `--require-attestation` now requires that block.

5. RND stock-loop meter proof:
   use the reward-model entrypoint, prove predictor trains, target stays fixed,
   latest-frame feature source matches the policy stack, and reward target is
   unchanged for `rnd_meter_v0`.

   Status: passed as a profile overhead/safety gate in the C512 RND rows:
   predictor changed, target stayed frozen, and target reward stayed unchanged.
   It is not positive-RND learning proof.

## Current Subagent Wave

- Bohr the 2nd: completed coverage map:
  `subagent_test_coverage_map_20260521.md`.
- Parfit the 2nd: completed local GPU-observation test additions:
  `subagent_gpu_observation_test_impl_20260521.md`.
- Gauss the 2nd: completed LightZero boundary and profile-label critique:
  `subagent_lightzero_boundary_test_critique_20260521.md`.
- Mendel the 2nd: completed RND/reset/death critique:
  `subagent_rnd_reset_death_test_critique_20260521.md`.

## 2026-05-21 Local Additions

- Added `test_lightzero_initial_inference_stack_probe_filters_zero_mask_roots`.
  This keeps the model-only split honest: terminal/all-zero-mask roots are not
  silently fed to model initial inference while collect-forward filters them.
- Added
  `test_persistent_delta_state_row_selective_cursor_regression_resets_only_regressed_row`.
  This catches stale-trail leakage when one row regresses/resets while another
  row appends incrementally.
- Added profile summary attestation tests in
  `tests/test_summarize_curvytron_optimizer_profile_results.py`.
  Fresh stock profile smoke `opt-semantic-identity-smoke-20260521a` now passes
  `--require-attestation`. The failed pre-fix attempt exposed a runner bug: the
  local collector could parse a nested `semantic_identity` JSON object instead
  of the top-level compact result. That parser is now fixed and covered in
  `tests/test_curvytron_optimizer_profile_manifest_runner.py`.
- Added LightZero decode edge-case tests for string-keyed rows, list outputs,
  nested root wrappers, missing actions, and decoded illegal actions.
- Subagent-added tests also cover:
  persistent renderer row/player request order, exact parity label failures,
  renderer-backed `uint8` FIFO across consecutive steps, terminal autoreset
  stack reset, and action-mask order through the batched-stack probe/scalar
  timestep path.
- Validation:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py`
  returned `61 passed`; ruff returned clean.
- Combined validation after all local additions:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_mock_collector.py tests/test_exploration_bonus.py tests/test_vector_reset.py tests/test_vector_autoreset.py`
  returned `142 passed, 6 skipped`; matching ruff command returned clean.

## Remaining P0 Gates

1. Speed-row semantic attestation:
   current summary-side guard is in place, and new compact outputs carry a
   `semantic_identity` block for dtype, scalar materialization, `to_play`,
   mask filtering, and consumer semantics. Fresh Modal artifact
   `opt-semantic-identity-smoke-20260521a` verifies that the contract appears
   in a stock `train_muzero` profile row. Old local artifacts were written
   before this producer/runner fix and should not be treated as fully attested
   current evidence.

2. LightZero decode edge cases:
   string-keyed outputs, list outputs, nested outputs, missing actions, and
   illegal decoded actions are now covered locally. Noncompact original root
   ids remain intentionally out of scope because the current collect-forward
   probe decodes compact `ready_env_id` rows.

3. Deeper collect/search split:
   the initial-inference split says model root inference is fast; the next
   useful optimization evidence needs internal collect/search timers or a
   toy/direct batched search comparison. The H100/L4 refresh still points at
   public LightZero collect/search rather than rendering or root inference.

4. RND positive-reward semantics:
   `rnd_meter_v0` overhead/safety is okay, but positive RND remains blocked on
   a better novelty normalization and resume/checkpoint story.

## Rule

Do not use a speed number as Coach launch advice unless its row carries enough
semantic identity to answer: what observed the game, what consumed the batch,
what death/RND mode ran, and what work is excluded.
