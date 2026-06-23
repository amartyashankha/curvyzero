# Subagent insertion points, 2026-05-23

Status: local code/docs inspection only. No remote jobs and no live Coach training paths were touched.

## Short answer

Smallest clean insertion point: `HybridBatchedObservationProfileManager.step` in `src/curvyzero/training/source_state_hybrid_observation_profile.py`, after actor payload merge, observation stack update, terminal `final_observation` capture, and root sidecar construction, but before `materialize_lightzero_scalar_timestep`.

The existing profile-only shape is already the right skeleton:

`HybridCompactBatch -> CompactRootBatchV1 -> CompactSearchServiceV1 -> CompactSearchResultV1 -> CompactReplayIndexRowsV1`

A PufferLib-style compact slab/search-service should first take ownership of that profile path, not the live stock LightZero Coach lane. The first slice should turn the current manager-owned arrays into a named rollout slab and make search write selected actions back into a `[B, P]` action buffer for the next env tick, while leaving stock LightZero objects only at validation/debug/sample edges.

## Current hot dataflow

| Stage | Exact files/functions | Current data owner | Proposed new owner | Notes |
| --- | --- | --- | --- | --- |
| Live trusted env step | `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`: `CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.reset`, `step`, `_lightzero_observation`, `_update_stack`; registered wrapper `CurvyZeroSourceStateVisualSurvivalLightZeroEnv.step` | Stock single-ego LightZero env wrapper over `VectorMultiplayerEnv`; LightZero env manager owns `BaseEnvTimestep` flow | No first-slice change. Treat as semantic oracle only | This is the Coach lane. It scalarizes into LightZero/DI-engine objects and should not be used for the slab experiment. |
| Profile env mechanics | `src/curvyzero/training/source_state_hybrid_observation_profile.py`: `InProcessHybridCurvyTronActor.step_into`, `HybridBatchedObservationProfileManager.step` | `HybridBatchedObservationProfileManager` owns `_compact`; actors may write directly when `native_actor_buffer=True` | New `CompactRolloutSlab` owns compact state/action/result arrays; manager becomes adapter | This is the lowest-disruption attach point because CPU env stepping is already batched and array-shaped. |
| Observation stack and terminal capture | `HybridBatchedObservationProfileManager.step`, especially payload merge, `_update_observation`, final-observation-before-autoreset handling, and autoreset stack reset | Manager-owned `_zero_stack`, `_render_out`, `_compact.final_observation`, done/autoreset masks | Slab owns `obs_stack`, `final_observation`, `done`, `autoreset`, reward, env-row/player identity | Preserve the existing order: observation `k`, action `k`, transition `k+1`, row `k`, and terminal final observation before reset. |
| Root batch | `HybridBatchedObservationProfileManager.step` builds `HybridCompactBatch`; `src/curvyzero/training/compact_policy_row_bridge.py`: `build_compact_root_batch_v1` | Manager builds root sidecars; bridge reshapes/copies to `CompactRootBatchV1` | Slab exposes root-batch view/handle; bridge remains validator/materializer | Use existing `copy_observation=False` semantics where possible. The root view should carry `env_row`, `player`, `policy_env_id`, legal mask, to-play, target reward, and active-root mask. |
| Search | `src/curvyzero/training/compact_search_service.py`: `CompactSearchServiceV1.run`; `src/curvyzero/training/compact_torch_search_service.py`: `CompactTorchSearchServiceV1.run`; direct CTree profile probe path in the boundary tests | Profile probe/service owns search invocation and returns arrays | `CompactSearchLoop`/service owns device model/search state and returns selected action plus deferred replay payload | Keep `CompactSearchResultV1` as the first contract. Later, split selected-action readback from visit-policy/root-value replay payload readback. |
| Selected action feedback | Current tests construct `joint_action[env_row, player] = selected_action`; next tick calls `HybridBatchedObservationProfileManager.step(joint_action)` | Test/profile harness or stock LightZero collector, depending on lane | Search service writes a slab action buffer `[B, P]`; CPU actors consume it next tick | The only unavoidable sync in the target design should be selected actions for CPU env mechanics. |
| Replay rows | `src/curvyzero/training/compact_policy_row_bridge.py`: `build_compact_replay_index_rows_v1_from_search_result`, `materialize_compact_target_rows_from_index_rows_v1`; `src/curvyzero/training/multiplayer_source_state_target_rows.py`: `build_source_state_multiplayer_target_rows_v0`, `build_source_state_multiplayer_sample_batch_v0` | Compact bridge owns index rows; target-row builder owns validation/sample materialization | Slab/replay writer owns compact index ring; materialization stays at sample/debug edge | Do not make full observation copies in the hot path. Store identity and offsets, then materialize rows only when sampling or proving equivalence. |
| RND | `src/curvyzero/training/exploration_bonus.py`: `extract_policy_gray64_latest_for_rnd_from_compact_observation`, `CurvyRNDRewardModel.collect_data`, `train_with_data`, `estimate` | Stock LightZero reward model consumes `GameSegment.obs_segment`; compact helpers can extract latest frames from compact observations | Slab owns latest-frame view and identity sidecars; RND model consumes compact batches at flush/sample edge | RND should not force GameSegment materialization in the compact hot loop. Keep reward-model integration as a later learner-edge bridge. |

## Smallest clean change set

1. Add a profile-only slab owner, probably near `src/curvyzero/training/source_state_hybrid_observation_profile.py` or as `src/curvyzero/training/compact_rollout_slab.py`.

   First ownership transfer: move the conceptual ownership of `_compact`, `_zero_stack`, root sidecars, selected actions, and compact replay indexes from `HybridBatchedObservationProfileManager` into a named slab object. Keep the manager API intact for the existing profile tests.

2. Change the existing probe boundary, not the live env boundary.

   The current boundary is `_run_batched_stack_probe(self.batched_stack_probe, compact_batch)` in `HybridBatchedObservationProfileManager.step`. That should become the service attach point: slab/root view in, selected actions plus compact search result out.

3. Keep `CompactRootBatchV1` and `CompactSearchResultV1` as the canary contracts.

   They already validate legality, active roots, `env_row`, `player`, `policy_env_id`, visit policy shape, selected action legality, and root-value payloads. Do not invent a second contract until this one becomes a measurable bottleneck.

4. Write selected actions back to a compact `[B, P]` buffer.

   Tests already prove the pattern by constructing `joint_action` from `search_result.env_row`, `search_result.player`, and `search_result.selected_action`. Make that explicit in the slab/search-service handoff.

5. Leave scalar LightZero objects as validation edges.

   `materialize_lightzero_scalar_timestep`, `_ready_obs_by_env_id`, and `_split_timestep_by_env_id` in `src/curvyzero/training/source_state_batched_observation_mock_collector.py` are useful for proof/debug, but they should not be the target optimized path.

## First local canary

Primary canary:

```bash
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py -k 'real_direct_ctree_compact_service_drives_next_step_and_matches_rows'
```

Why this one: it starts at `HybridBatchedObservationProfileManager.step`, builds a compact root batch, runs the real direct CTree compact service when local LightZero hooks are available, feeds selected actions into the next env step, builds compact replay index rows, materializes target rows, and compares against the trusted immediate-row/sample path.

No-LightZero fallback canary:

```bash
uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py -k 'closed_compact_loop_index_rows_materialize_same_as_immediate_rows or compact_index_rows_materialized_sample_batch_matches_immediate_rows or deferred_search_payload_rows_match_immediate_rows'
```

Why this one: it proves the compact replay/search payload contract without requiring the direct CTree backend.

## Stale or misleading names

- `lightzero_curvyzero_stacked_debug_visual_survival_train.py`: despite the filename, the live variants now route through source-state wrappers; do not infer the current hot path from "stacked debug visual".
- `CurvyZeroSourceStateVisualSurvivalLightZeroEnv`: this is a single-ego LightZero env wrapper over multiplayer source state, not a compact `[B, P]` slab.
- `source_state_fixed_opponent`, `fixed_opponent`, and old `two_seat_self_play` names: these do not mean live simultaneous current-policy self-play in the Coach lane.
- `direct_ctree_gpu_latent`: currently a profile/backend experiment and semantic oracle, not the live Coach trainer path.
- `CompactTorchSearchServiceV1` and dense Torch MCTS names: useful profile candidates/falsifiers, but not trainer-ready replacement plumbing.
- `service_tax_probe` and mock search service names: these are ceiling/falsifier tools, not MCTS advice providers.
- `flat_a3` and vendor `lightzero_ctree_a3`: demoted fixed-action ABI experiments; do not build the new slab around them.
- `policy_observation_backend='jax_gpu'`: scalar JAX renderer experiment, not the resident slab/search-service target.
- `materialize_lightzero_scalar_timestep`, `_ready_obs_by_env_id`, `_split_timestep_by_env_id`: scalar compatibility/debug bridges, not the target optimized dataflow.
- `metadata-only replay` in `multiplayer_replay_v0.py`: public metadata replay, not the trainer/full-array replay path.
- "native" in bridge filenames can mean native LightZero `GameSegment`/MuZero buffer compatibility, not live Coach integration.

## Recommendation

Do the first slab insertion entirely in the profile path:

`HybridBatchedObservationProfileManager.step -> CompactRolloutSlab root view -> CompactSearchServiceV1.run -> selected-action slab -> next manager.step -> CompactReplayIndexRowsV1 ring -> materialize only for canary/sample`

This preserves the trusted Coach lane, reuses the strongest existing contract tests, and targets the dataflow boundary where scalar LightZero object churn can actually be avoided.
