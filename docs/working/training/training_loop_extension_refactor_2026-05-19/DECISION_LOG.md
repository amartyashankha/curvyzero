# Decision Log

Last updated: 2026-05-19

Record decisions here once they are real. Do not use this for open questions.

## D-001: Keep This Refactor In Its Own Task Folder

Status: accepted

Decision: planning, orchestration, task board, findings, source map, and design notes for this refactor live in `docs/working/training/training_loop_extension_refactor_2026-05-19/`.

Reason: the broader feedback-loop docs are already carrying tournament/training operations state. This refactor needs a clean local workspace.

## D-002: Scouts Are Read-Only First

Status: accepted

Decision: subagents for this phase are read-only scouts. They should return file/line references, risks, extraction candidates, tests, and follow-ups.

Reason: several boundaries overlap. Main thread should integrate findings before code edits so we do not create conflicting partial refactors.

## D-003: No Deterministic Batch Split Implementation Yet

Status: accepted

Decision: do not implement deterministic opponent split values until the batch construction path is mapped.

Reason: we need to distinguish collector env count, episode sampling, and learner batch size before pinning a contract.

## D-004: Reward Contract Owns `auto`

Status: accepted

Decision: `reward_contracts.py` owns trainer-facing `auto` reward normalization, but env construction still receives concrete env-facing reward variants.

Reason: `auto` is part of the public trainer/config contract. Keeping it trainer-private recreated the same duplicate reward/support logic this lane is removing.

## D-005: Config Helper Extraction Is Real But Partial

Status: superseded by D-007

Decision: `lightzero_config_builder.py` is now the public home for pure config patch/path helpers and visual-survival surface extraction. The full `_build_visual_survival_configs(...)` builder remains in the Modal trainer until the typed builder spec is ready.

Reason: moving the helper layer first reduces trainer-private imports without pretending the whole builder extraction is done.

## D-006: Opponent Assignment Readiness Must Prove Split Metadata

Status: accepted

Decision: assignment-refresh readiness is not complete unless each collector env reports the expected split unit, mode, plan hash, env index/count, entry name, and entry count.

Reason: proving only assignment id/ref/hash can let the pipeline look healthy while the per-env singleton split is wrong or stale.

## D-007: Public Config Builder Owns The Same-Signature Builder

Status: accepted

Decision: `lightzero_config_builder.py` owns env-variant specs, render/backend/seat/cadence validators, opponent relation helpers, and the same-signature `build_visual_survival_configs(...)`. The Modal trainer keeps `_build_visual_survival_configs(...)` only as an entrypoint facade, and eval imports the public builder directly.

Reason: the trainer was acting as a private config API. Moving the builder makes config construction locally testable without importing Modal as the source of truth. The broad signature is temporary; the next cleanup is a typed spec/result, not restoring builder logic to the trainer.

## D-008: Typed Visual-Survival Config Boundary Is Primary

Status: accepted

Decision: the durable config-builder boundary is `VisualSurvivalConfigSpec -> build_visual_survival_config(...) -> VisualSurvivalConfigResult`, with nested `FrozenOpponentConfig`. The broad `build_visual_survival_configs(**kwargs)` function remains only as a compatibility facade and must stay parity-tested against the typed path.

Reason: future work should reason about a named config contract, not a giant keyword bag. Trainer/eval still mutate the historical dict result today, so the facade stays ledgered until those protected callers move to the typed result deliberately.
