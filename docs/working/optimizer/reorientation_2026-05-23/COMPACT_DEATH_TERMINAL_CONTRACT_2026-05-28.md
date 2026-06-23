# Compact Death/Terminal Contract

Date: 2026-05-30

Status: guarded local normal-death contract mode landed; Coach gate remains
blocked until compact-owned lifecycle evidence is emitted and preserved.

## Decision

Do not treat the profile/no-death terminal N-step row as a promotable
death/terminal contract.

The current compact-owned code can honestly attest a narrow terminal-support
contract:

```text
contract=curvyzero_compact_death_terminal_contract/v1
death_mode=profile_no_death
terminal_target_mode=stock_terminal_no_bootstrap_return_discount_1.0
promotion_gate_satisfied=false
blocker=normal_collision_death_not_proven
```

This preserves the useful work: final observations before autoreset, final
reward maps, validity masks, post-terminal masking, resident final observations,
and no-bootstrap terminal value targets. It does not claim source-faithful
normal collision death.

The contract module now also knows the future promotable mode:

```text
mode=normal_collision_death_terminal_nstep_v1
death_mode=normal
promotion_gate_satisfied=true
```

That mode is deliberately unreachable from the default trainer/checkpoint path.
It only builds when supplied structured evidence for normal collision terminal
rows, zero truncation, death facts, final-observation/final-reward-before-
autoreset semantics, no-bootstrap terminal targets, resident/device replay
rows, compact learner done rows, and evidence refs. This proves the validator
can fail closed; it is not yet a durable proof that a compact-owned run emitted
those facts.

## Implementation

Added:

```text
src/curvyzero/training/compact_death_terminal_contract.py
tests/test_compact_death_terminal_contract.py
```

Threaded into:

```text
src/curvyzero/training/compact_owned_trainer.py
src/curvyzero/training/compact_trainer_checkpoint.py
tests/test_compact_owned_trainer.py
tests/test_compact_trainer_checkpoint.py
tests/test_compact_coach_compatibility.py
```

The metadata now carries the partial contract but leaves the required Coach
gate false. It rejects `death_mode=normal` until normal-death evidence exists,
and it only emits `death_terminal_contract` compatibility evidence when the
contract's promotion gate is actually satisfied.

Latest local implementation update:

```text
normal_collision_death_terminal_nstep_v1
curvyzero_compact_death_terminal_contract_evidence/v1
```

The normal mode requires structured evidence fields for terminal/terminated/
death counts, `truncated_row_count == 0`, opponent-trail or wall cause,
hit-owner presence, done semantics, final observation before autoreset, final
reward map use, no-bootstrap terminal value targets, validity masks, resident
terminal final observation use, device replay terminal rows, compact learner
done rows, and nonempty evidence refs. `CompactOwnedTrainerV1` and compact
checkpoints now copy status/booleans from the contract rather than hardcoding
profile/no-death values. Checkpoint `extra_metadata` rejects protected
Coach/death/reward compatibility overrides so a caller cannot spoof
`death_terminal_contract=true` or a compatibility gate.

Latest payload-derived evidence update:

```text
normal_collision_death_evidence_rows
build_normal_collision_death_evidence_from_profile_result_v1(...)
require_normal_death_terminal_contract=true
```

Hybrid profile results now emit bounded JSON-safe evidence rows with
`done`/`terminated`/`truncated`, terminal reason, death count/player/cause/hit
owner, winner/draw, reward/final reward map, final reward equality, and terminal
final-observation row proof. They also emit aggregate proof fields for
death-cause counts, normal collision causes, hit-owner presence, done semantics,
terminal final observations, and final reward maps. The Modal compact projection
and manifest runner preserve those fields. The contract helper now derives
`curvyzero_compact_death_terminal_contract_evidence/v1` from the emitted profile
payload plus compact sample/learner terminal telemetry, and the manifest runner
can require this derivation for opt-in rows. This removes the handbuilt-payload
gap; it still does not close the Coach gate until a compact-owned lifecycle
artifact/checkpoint consumes the evidence.

Follow-up groundwork now also threads real terminal/death sidecars through the
compact hybrid boundary:

```text
src/curvyzero/env/vector_multiplayer_env.py
src/curvyzero/training/source_state_hybrid_observation_profile.py
src/curvyzero/training/compact_rollout_slab.py
src/curvyzero/training/compact_policy_row_bridge.py
```

`HybridObservationProfileConfig` can carry `death_mode=normal`, compact actors
receive that mode, native and non-native payloads include `terminated`,
`truncated`, `terminal_reason`, `death_count`, `death_player`, `death_cause`,
`death_hit_owner`, `winner`, and `draw`, and replay/slab commits use real
terminated/truncated arrays instead of reconstructing them from `done`. Direct
synthetic `HybridCompactBatch(...)` construction remains backward-compatible by
defaulting the new sidecars from `done` and zero/no-death values.

The local proof fixture is now stronger than the older wall-death sketch:
`tests/test_source_state_hybrid_observation_profile.py` drives
`scenarios/environment/source_collision_head_head_reverse_order_single_death_step.json`
through `InProcessHybridCurvyTronActor` and through
`HybridBatchedObservationProfileManager(native_actor_buffer=True)`. It asserts
`death_mode=normal`, `done == terminated == true`, `truncated=false`,
`death_count=1`, `death_player=[0]`, `death_cause=opponent_trail`,
`death_hit_owner=1`, `winner=1`, terminal reward/final reward `[-1.0, 1.0]`,
`done_root=[true,true]`, terminal final observation presence, and inactive
compact roots after terminal.

The durable profile surface is no longer hardwired to no-death:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
scripts/build_curvytron_hybrid_observation_profile_grid.py
scripts/run_curvytron_hybrid_observation_profile_manifest.py
```

Rows now carry `death_mode`, commands include `--death-mode`, boundary config
validation passes it into `HybridObservationProfileConfig`, compact summaries
include death counters, and manifest preflight fails row/command/payload
mismatches. This makes a future durable normal-death row meaningful; it does not
by itself flip the Coach gate.

## Compatibility Refresh

Durable local report:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-death-terminal-contract-20260528/compatibility_report.json
```

Passed required gates:

- trainer entrypoint;
- checkpoint save/load;
- resume metadata;
- eval/GIF/tournament load;
- reward/RND contract.

Still missing:

- death/terminal contract;
- policy refresh handoff;
- training metrics lineage.

Selected next gate:

```text
death_terminal_contract
```

## Closing Evidence Needed

The next proof must be emitted by the compact-owned lifecycle, not handbuilt in
a unit test. It must use source-faithful `death_mode=normal` and bind:

- normal collision death facts such as death player/cause/hit owner or the
  equivalent source terminal cause fields;
- terminal final observations before autoreset;
- source final reward maps and shaped trainer reward separation where relevant;
- `done == terminated | truncated` masks, with truncation either proven or
  explicitly excluded by contract;
- compact replay/index/slab/sample/learner propagation;
- no-bootstrap terminal learner targets and post-terminal masks;
- checkpoint/resume metadata and compatibility evidence refs.
- a durable local or Modal artifact whose payload can be converted into
  `curvyzero_compact_death_terminal_contract_evidence/v1` without manual
  invention.

## 2026-05-30 Gate Closeout

The closing proof now exists and should be treated as the current durable
death/terminal evidence:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-normal-death-compact-owned-profile-20260530/row_001_result.json
artifacts/local/curvytron_compact_owned_normal_death_checkpoint_results/optimizer-compact-owned-normal-death-checkpoint-from-profile-20260530/normal_death_checkpoint_smoke_report.json
```

The profile row passed with `death_mode=normal`, 21 terminal/death rows, 0
truncations, wall and opponent-trail causes, hit-owner proof, terminal final
observations, source final-reward equality, B256 bounded learner sampling,
terminal sample rows `3`, terminal no-bootstrap value targets, compact MuZero
learner done count `3`, and
`normal_death_terminal_contract_promotion_gate_satisfied=true`.

The checkpoint smoke consumes that exact profile result via
`--profile-result`, saves and reloads a compact-owned checkpoint, attaches the
profile-result evidence ref, and sets:

```text
death_terminal_contract=true
compact_coach_compatibility_gate_death_terminal_contract=true
compact_coach_compatibility_promotion_eligible=false
```

This closes the death/terminal blocker for the compact-owned route. It does not
close promotion. That exact death-checkpoint smoke still lacks policy-refresh
handoff, training-metrics lineage, and refreshed eval/GIF/tournament-load
evidence because it predates those later artifacts.

Later 2026-05-30 local evidence closes lifecycle unification separately at:

```text
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json
```

Promotion remains blocked by explicit `coach_speed_row` evidence, not by the
death/terminal contract.

## Validation

```text
uv run ruff check src/curvyzero/training/compact_death_terminal_contract.py src/curvyzero/training/compact_reward_rnd_contract.py src/curvyzero/training/compact_coach_compatibility.py src/curvyzero/training/compact_owned_loop.py src/curvyzero/training/compact_owned_trainer.py src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/training/compact_stock_checkpoint_export.py tests/test_compact_death_terminal_contract.py tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_muzero_learner.py tests/test_compact_search_replay_contract.py
uv run pytest tests/test_compact_death_terminal_contract.py tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_muzero_learner.py tests/test_compact_search_replay_contract.py -q
uv run ruff check src/curvyzero/env/vector_multiplayer_env.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/compact_policy_row_bridge.py src/curvyzero/training/compact_trainer_checkpoint.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_search_replay_contract.py tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py
uv run pytest tests/test_compact_death_terminal_contract.py tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_muzero_learner.py tests/test_compact_search_replay_contract.py tests/test_source_state_hybrid_observation_profile.py -q
uv run ruff check src/curvyzero/env/vector_multiplayer_env.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/compact_policy_row_bridge.py src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py scripts/build_curvytron_hybrid_observation_profile_grid.py scripts/run_curvytron_hybrid_observation_profile_manifest.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_curvytron_hybrid_observation_profile_manifest_runner.py tests/test_compact_search_replay_contract.py tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py
uv run pytest tests/test_compact_death_terminal_contract.py tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_muzero_learner.py tests/test_compact_search_replay_contract.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_curvytron_hybrid_observation_profile_manifest_runner.py -q
uv run ruff check src/curvyzero/training/compact_death_terminal_contract.py src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/training/compact_owned_trainer.py tests/test_compact_death_terminal_contract.py tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py
uv run pytest tests/test_compact_death_terminal_contract.py tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py -q
uv run pytest tests/test_compact_death_terminal_contract.py tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_muzero_learner.py tests/test_compact_search_replay_contract.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_curvytron_hybrid_observation_profile_manifest_runner.py -q
uv run ruff check src/curvyzero/training/compact_death_terminal_contract.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py scripts/run_curvytron_hybrid_observation_profile_manifest.py tests/test_compact_death_terminal_contract.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_manifest_runner.py
uv run pytest tests/test_compact_death_terminal_contract.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_manifest_runner.py -q
uv run pytest tests/test_compact_death_terminal_contract.py tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_muzero_learner.py tests/test_compact_search_replay_contract.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_curvytron_hybrid_observation_profile_manifest_runner.py -q
```

Result:

```text
ruff passed
110 passed, 2 warnings
ruff passed
189 passed, 2 warnings
ruff passed
434 passed, 2 warnings
ruff passed
22 passed, 2 warnings
442 passed, 2 warnings
ruff passed
202 passed, 2 warnings
449 passed, 2 warnings
```

## Non-Claims

- no default checkpoint/trainer normal collision death claim;
- no promotion claim;
- no claim that every future checkpoint/trainer has normal collision death
  evidence by default; the positive claim is tied to the durable 2026-05-30
  profile-result evidence ref;
- no claim that the exact 2026-05-30 checkpoint smoke refreshed
  eval/GIF/tournament load, policy-refresh handoff, or training-metrics lineage;
- no truncation-bootstrap claim;
- no stock `train_muzero` call;
- no Coach speed claim;
- no live-run safety claim.
