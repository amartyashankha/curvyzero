# Replay Terminal and Seed Contract

Status: working memory
Date: 2026-05-09
Owner: REPLAY

This note is only about the local debug actor-loop replay chunk:
`curvyzero_debug_actor_loop_replay_chunk/v0`.

It is not a production replay shard format.

A separate file-level replay v0 contract now exists in
`src/curvyzero/training/replay_chunk_v0.py` for 1v1/no-bonus training chunks.
That contract includes observation, reward, action, action weights, root value,
done/terminated/truncated, `episode_id`, `reset_seed`, `reset_source`,
`final_observation`, `final_reward_map`, and compatibility hashes. The actor
bridge sample path can now write one replay-v0 `.npz` through
`--sample-only --sample-replay-v0-chunk PATH`, but that sample still carries
debug actor-bridge observations and rewards. Treat it as a shape/hash bridge
proof, not production replay integration.

`src/curvyzero/training/trainer_replay_v0_builder.py` now packs B=1 and B>1
`TrainerObservationBatch1v1` rows into replay-v0 arrays and metadata. The
helper packs trainer observation/reward payloads plus explicit actions, action
weights, root values, `episode_id`, `reset_seed`, and `reset_source`; it rejects
`done != terminated | truncated`, non-final terminal rows, and missing terminal
`final_reward_map`.

`src/curvyzero/training/vector_env_replay_recorder.py` records live
`VectorTrainerEnv1v1NoBonus` step batches into replay-v0. It preserves terminal
final arrays before autoreset, carries row-local episode/reset identity into
chunks, adds optional vector row metadata, and can build a small sidecar
manifest for strict replay/profile claims. The strict env info includes
`native_control_model_id`, `trainer_control_wrapper_id`, and `decision_ms`, and
those labels should travel with strict replay/profile artifacts. It still does
not record full RNG state/history/ref.

## Current Chunk

The debug chunk stores these arrays:

- `obs`: `float32[T,B,P,D]`
- `reward`: `float32[T,B,P]`
- `action`: `int8[T,B,P]`
- `action_weights`: `float32[T,B,P,A]`
- `root_value`: `float32[T,B,P]`
- `done`: `bool[T,B]`
- `ego_mask`: `bool[T,B,P]`

The chunk metadata validates schema, rules, observation, action-space, reward,
environment implementation, shapes, and dtypes.

## Explicit Missing Policy

The debug chunk now has to say these fields are absent:

- `episode_id_policy`: `absent_debug_sample_only`
- `reset_seed_policy`: `absent_debug_sample_only`
- `reset_source_policy`: `absent_debug_sample_only`
- `terminated_truncated_done_policy`:
  `done_debug_surface_only_absent_terminated_truncated`
- `final_observation_policy`: `absent_debug_sample_only_current_obs_only`

This means:

- no row-local episode id is stored;
- no row-local reset seed is stored;
- no reset source code is stored;
- no row-local `terminated` or `truncated` flags are stored;
- no final observation or reset observation policy is defined;
- `done` is only the current debug surface flag.

The reader and writer reject chunks that omit those policy fields or replace
them with a production replay claim.

The reader and writer also require these compatibility hashes:

- `replay_schema_hash`
- `rules_hash`
- `observation_schema_hash`
- `action_space_hash`
- `reward_schema_hash`

Those hashes only fence the current debug sample against obvious schema/rules
drift. They are not a manifest, lineage model, or production replay
compatibility contract.

## Blockers Before Production Replay

Production replay still needs broader integration around the v0 contract:

- full RNG state/history/ref, not only reset seed/source identity;
- source-faithful broader lifecycle rows beyond strict 1v1/no-bonus;
- public reset-to-terminal parity using the source fixture random tape and
  warmup policy;
- event/state/trace refs or ranges;
- manifest and compaction rules.

## Verification

Focused replay tests:

```bash
uv run pytest tests/test_debug_actor_loop_replay.py -q
```

Focused actor bridge sample replay test:

```bash
uv run pytest tests/test_benchmark_vector_actor_loop_bridge.py -q
```

That test now covers both sample-only debug replay chunks and sample-only
replay-v0 chunks. The replay-v0 chunk is still marked blocked for production
training when its payload comes from debug actor-bridge surfaces.

Focused trainer replay-v0 bridge test:

```bash
uv run pytest tests/test_trainer_replay_v0_builder.py -q
```

Touched-file lint:

```bash
uv run ruff check src/curvyzero/training/debug_actor_loop_replay.py tests/test_debug_actor_loop_replay.py
uv run ruff check src/curvyzero/training/trainer_replay_v0_builder.py tests/test_trainer_replay_v0_builder.py
```
