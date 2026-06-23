# Task Board

Last updated: 2026-05-19

| ID | Priority | Lane | Status | Next Action | Done When |
| --- | --- | --- | --- | --- | --- |
| TLR-DOC-001 | P0 | Docs | Done | Keep task-local docs current as findings land. | README, current phase, orchestration, operating patterns, task board, findings log, source map, decision log, glossary, and plan docs exist. |
| TLR-AGENT-001 | P0 | Orchestration | Done | Use completed scout findings to pick implementation order. | Each scout returned bounded findings with file/line references and risks. |
| TLR-REWARD-001 | P0 | Reward | Partially extracted | Finish migrating tests/imports away from private trainer reward helpers as adjacent config/eval extractions land. | `reward_contracts.py` owns reward/support contracts; env/trainer/eval delegate to it; focused tests pass. |
| TLR-CONFIG-001 | P0 | Config | Typed boundary landed | Keep typed/broad parity green; next shrink protected callers off broad facade where practical. | `lightzero_config_builder.py` owns visual-survival config construction, `VisualSurvivalConfigSpec`, `VisualSurvivalConfigResult`, env specs, patch/path/surface helpers, and tests; trainer/eval call the public builder. |
| TLR-EXPERIMENT-SURFACE-001 | P0 | Config | Tonight18 migrated | Keep submitter normalizer before side effects; finish or explicitly defer non-default-heavy builders; retire full `train_kwargs` as canonical artifact shape only after every intentional deviation has a named compact field. | Experiment-facing fields are documented/tested; grouped submit only requires mode/seed/run/attempt; compact/minimal rows are normalized before spawn; `tonight18` emits compact-by-default rows; low-level builder fields remain internal unless explicitly opened as ablations. |
| TLR-HOOK-001 | P0 | Hooks | Initial map done | Defer until reward/config are cleaner; add bundle-order test before moving hooks. | We know what can become a `HookBundle` and what must stay near LightZero. |
| TLR-ENV-001 | P1 | Env | Initial map done | Prefer reward helper or telemetry helper before moving opponent execution or render cache. | We have a safe extraction order for env helpers. |
| TLR-BATCH-001 | P1 | Batch | Initial map done | Rename/document units; do not implement deterministic split semantics yet. | We know deterministic split controls collector env assignment, not learner mini-batches. |
| TLR-TEST-001 | P0 | Tests | Active | Keep adding one focused public-module gate per extraction, plus one orchestration smoke per touched trainer path. | Reward/config-helper/grouped submitter/opponent split/no-Modal trainer spine/runtime reward gates pass locally. |
| TLR-CLEAN-001 | P1 | Cleanup | Active | Remove stale claims in this lane as soon as implementation moves. | Task docs reflect current truth and do not point agents at stale defaults. |
| TLR-WORKFLOW-001 | P0 | Workflow | Done | Keep hard gates active while editing. | Patch gate, launch gate, compatibility ledger, and subagent integration gate are documented. |
| TLR-COMPAT-001 | P0 | Compatibility | Started | Keep `COMPATIBILITY_LEDGER.md` current for any temporary facade/private import. | Each has keep/delete/expiry/test coverage. |
| TLR-API-001 | P0 | API | Started | Continue replacing private trainer helper imports with public pure modules; eval now imports config builder helpers publicly. | Tests/non-entrypoint modules import public pure modules where practical. |
| TLR-LAUNCH-001 | P0 | Launch | Pending | Define launch-readiness gate if this lane ever needs Modal. | Modal launch requires named question plus passing local reward/config/hook tests. |
| TLR-OPPONENT-CONTRACT-001 | P0 | Opponent | Started | Extract/document opponent assignment vocabulary and raw-vs-refresh semantics; keep split metadata proof pinned. | Config builder can consume a clear opponent assignment contract; ready-report proves exact split metadata. |
| TLR-METADATA-001 | P1 | Checkpoint | Pending | Extract pure checkpoint metadata sidecar payload construction after builder surface is stable. | Metadata serializes reward/opponent/observation/config surface without fallback shape drift. |
| TLR-PROGRESS-001 | P1 | Status | Pending | Extract pure progress/status payload contracts after metadata. | Progress/status builders consume stable reward/opponent/metadata surfaces. |
| TLR-TESTBOUNDARY-001 | P0 | Tests | Started | Keep moving pure helper tests to public modules; next full-builder work should type/narrow the public spec, not add trainer-private unit tests. | The Modal trainer is tested as orchestration, not as a helper library; real-builder-to-fake-train smoke passes. |

## First Candidate Extraction

First cut: reward contracts. The shared module exists and the active env/trainer/eval reward-support path is wired through it. The config cut moved patch/path/surface helpers, env-variant specs, and the visual-survival config builder into `lightzero_config_builder.py`; the typed config boundary is now primary, while the trainer and broad builder keep same-signature facades for launch/tests.

Current pure-contract extraction order:

1. Reward contracts.
2. Opponent assignment vocabulary and split metadata.
3. Continue shrinking compatibility facades around the typed config builder.
4. Checkpoint metadata.
5. Status/progress payloads.
6. Hook bundle/runtime publisher.

Next larger extraction should start with opponent-assignment vocabulary extraction, checkpoint metadata extraction, or moving protected callers off broad config facades; the public typed builder cut has direct tests plus trainer/eval smoke coverage.
