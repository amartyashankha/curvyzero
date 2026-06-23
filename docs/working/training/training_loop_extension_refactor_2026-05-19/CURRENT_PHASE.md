# Current Phase

Last updated: 2026-05-19

## Plain Goal

Understand and then modularize the training loop so future research changes are easy to make without touching a huge trainer file, the env, tournament plumbing, and manifests all at once.

## Current Phase

Behavior-preserving extraction and local validation. We are not launching a new production run in this lane. Reward/support contracts are shared pure code, the LightZero visual-survival config builder now lives in a public training module, and the current job is to keep pinning those contracts with local tests before the next larger extraction.

## Current Truth

- The trainer already calls LightZero through a normal `train_muzero` entrypoint.
- The hard part is everything wrapped around that call: config construction, env registration, checkpoint publishing, resume hooks, opponent refresh, GIF/eval side work, and metadata.
- `src/curvyzero/training/reward_contracts.py` is now the shared source for reward variants, reward schema ids/hashes, reward perspectives, reward-space bounds, and LightZero support config.
- The source-state env, Modal trainer reward helpers, and eval model-target support patch now delegate to the reward contract module instead of carrying duplicate reward/support definitions.
- `src/curvyzero/training/lightzero_config_builder.py` now owns config path patching, checkpoint hook patch helpers, target support patching, env-variant specs, visual-survival config construction, and visual-survival surface extraction.
- The current config-builder boundary is typed: `VisualSurvivalConfigSpec -> build_visual_survival_config(...) -> VisualSurvivalConfigResult`.
- The typed config spec is grouped normalized config, not the experiment-facing knob list. Low-level fields still exist because LightZero/env construction needs them, but they should not all be exposed as top-level launch choices.
- `VisualSurvivalExperimentSpec` is the compact experiment-facing target surface. It expands `current_broad` defaults into grouped internal config and rejects low-level timing/render/template/LightZero fields as constructor kwargs.
- The broad `build_visual_survival_configs(**kwargs)` function is now a compatibility facade through the typed path, not the conceptual API.
- Grouped submit now requires only the minimal remote-call identity shell: `mode`, `seed`, `run_id`, and `attempt_id`.
- The submitter normalizes compact rows before validation/spawn: minimal `train_kwargs` get matching poller `run_id`/`attempt_id`/`seed`, and optional `experiment_spec` rows can expand reward/noise/current scale into flat trainer kwargs.
- `build_curvytron_tonight18_manifest.py` now emits compact-by-default `train_kwargs` and carries `train_kwargs_schema_id=curvyzero_tonight18_compact_train_kwargs/v0`. Its validator expands those compact rows from a local default table before checking the same assignment, slot, checkpoint, render, and learner-seat contracts.
- Non-migrated manifest builders may still emit large flat `train_kwargs` where row settings intentionally differ from trainer defaults. That remains compatibility and override shape, not the conceptual required experiment surface.
- The Modal trainer keeps `_build_visual_survival_configs(...)` only as a same-signature launch/test facade that delegates to the public builder.
- Eval imports public config-builder helpers, including `build_visual_survival_configs(...)` and `target_config_patches(...)`.
- Opponent assignment refresh readiness now verifies per-env split metadata, not just the assignment id/ref/hash.
- The env currently owns too many concepts at once: player perspective, opponent policy execution, reward components, telemetry, and visual observation.
- Batch terms are distinct: current defaults use `collector_env_num=256`, `n_episode=256`, and learner `batch_size=64`; 64-slot opponent recipes are collector-env split bags, not learner mini-batches.
- A side-network or exploration-bonus experiment will be risky until reward contracts, hook boundaries, and resume state are explicit.

## Decisions So Far

- This is a separate task lane with its own docs.
- Main thread owns docs and integration.
- Subagents are read-only scouts unless explicitly reassigned.
- First implementation should be behavior-preserving extraction, not a semantic experiment.
- Semantic changes, such as deterministic batch splits or new intrinsic reward, need their own task-board item and test gate.

## Active Lanes

| Lane | Status | Purpose |
| --- | --- | --- |
| Reward contracts | Partially extracted and wired | Shared module owns reward metadata/support; env/trainer/eval delegate to it; remaining work is to remove private-helper tests/import habits as later extractions land. |
| LightZero config builder | Compact experiment surface pinned | Public module owns patch/path/surface helpers, env specs, compact experiment spec, grouped visual-survival spec/result, and the broad compatibility facade. |
| Opponent assignment contract | First hardening landed | Ready-report proof now checks exact collector-env split metadata; next work is to move vocabulary/runtime helpers out of trainer-private shape. |
| Hook bundle and extension API | Planning | Isolate checkpoint/resume/opponent-refresh hooks and define a path for side-network experiments. |
| Env step modularity | Planning | Separate perspective, opponent execution, reward, observation, and telemetry helpers. |
| Batch construction | Mapped, no semantic change | Deterministic split controls collector-env assignment only; do not present it as learner mini-batch control. |
| Tests and migration | Active | Reward, grouped config-builder, grouped submitter, opponent split metadata, and no-Modal trainer/env/eval-ish gates pass locally so far. |

## Gates Before Next Larger Extraction

- Each extraction has a named source section, target module, and test plan.
- We know which behavior is supposed to stay identical.
- We know which stale defaults or hidden fallbacks should be removed rather than preserved.
- We have a rollback strategy that does not require reverting unrelated user work.

## Open Questions

- When can protected trainer/eval/test callers move off the broad `build_visual_survival_configs(**kwargs)` facade entirely?
- Which remaining manifest builders can safely compact by dropping default-equal keys, and which must stay explicit because their row defaults differ from current trainer defaults?
- Should deterministic collector-env split be exposed through that config-builder contract, or kept as manifest/assignment machinery?
- What is the cleanest policy-observation perspective contract across training and tournament eval?
- What hook surface is sufficient for a future side network without forking LightZero too deeply?
