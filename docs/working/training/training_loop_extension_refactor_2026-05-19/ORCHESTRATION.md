# Orchestration

Last updated: 2026-05-19

This is the low-level coordination doc for the training-loop extension refactor. It is not the general operating-pattern doc and it is not the broader feedback-loop task board.

## Main Thread Contract

- Keep this folder current.
- Integrate subagent findings into `FINDINGS_LOG.md` and then into the relevant plan doc.
- Make final decisions explicitly in `CURRENT_PHASE.md`.
- Edit code only after a boundary and test gate are clear.
- Avoid reading every large file into the main context. Use targeted searches and subagent summaries.

## Current Main-Thread Work

| Item | Status | Notes |
| --- | --- | --- |
| Create task-local docs | Done | This folder is the active scratchpad for this refactor lane. |
| Dispatch read-only audits | Done | Scouts/critics mapped source boundaries, batch semantics, docs freshness, and test gates. |
| Integrate findings | Done through current reward/config-helper/opponent split cut | Key findings are in `FINDINGS_LOG.md`, plan docs, and the task board. |
| Pick first extraction | Done | Reward contracts went first. |
| Critique trainer bloat | Done | The central critique: the trainer is huge because it owns control-plane and pure contracts around a small stock `train_muzero` call. |
| Wire first extraction | Done | `reward_contracts.py` is wired into env/trainer/eval; focused reward/env/trainer gates pass. |
| Wire config builder extraction | Done | `lightzero_config_builder.py` owns pure patch/path/surface helpers, the public builder, and typed spec/result; trainer/eval use public facades. |
| Harden opponent split proof | Done | Assignment readiness now checks exact per-env split metadata and reports split slot counts/hash. |
| Refresh docs and gates | Done for current cut | Current truth, critique results, and passing local gates are recorded. |
| Shrink parameter surface | Tonight18 compact proof landed | Compact experiment spec is pinned; grouped submit requires only mode/seed/run/attempt; submitter normalizes compact/minimal rows before side effects; `tonight18` emits compact-by-default rows; legacy full kwargs remain accepted for non-migrated override-heavy builders. |

## Subagent Plan

| Agent | Status | Ask | Follow-up Planned |
| --- | --- | --- | --- |
| Reward Contract Scout: Carson `019e40dd-9018-7560-904c-e03f76c996f5` | Done | Mapped reward variants, reward component construction, support bounds, telemetry labels, and tests. | First extraction candidate: pure reward contract module, leaving scalar reward math in env initially. |
| Config Builder Scout: Zeno `019e40dd-8e83-7091-9914-749e80c4829f` | Done | Mapped `_build_visual_survival_configs` and related validation/default code. | Public typed builder extracted with compatibility facades. |
| Hook/Extension Scout: Kuhn `019e40dd-917f-7d33-a26f-1caf6243e555` | Done | Mapped checkpoint publisher, progress writer, resume hooks, opponent refresh, and learner-hook seams. | Defer hook extraction until reward/config/metadata are stable. |
| Env Step Scout: Averroes `019e40dd-9421-7483-acc8-16bf128f9b66` | Done | Mapped reset/step/reward/observation/opponent responsibilities in the source-state env. | Use reward helper or telemetry helper as lower-risk env cuts; defer opponent execution and observation cache movement. |
| Batch Construction Scout: Euclid `019e40dd-95ce-73b0-ae57-80febd819b7c` | Done | Mapped current batch size, opponent slot sampling, and where LightZero turns env config into rollout batches. | Document unit names; do not claim deterministic collector split controls learner mini-batch composition. |
| Test Gate Scout: Helmholtz `019e40dd-9764-7373-affa-2f3953c557d9` | Done | Mapped existing focused tests, missing gates, and staged migration order. | Use its staged gates as the default test plan unless contradicted by deeper scouts. |

## Trainer Bloat Critique Agents

| Agent | Status | Ask | Follow-up Planned |
| --- | --- | --- | --- |
| LightZero Integration Critic: Franklin `019e40ea-e9b9-7e90-8fc5-d2eb9d738c06` | Done | Explained what exists because stock LightZero lacks extension points and what should be pure wrapper code. | Folded into `TRAINER_BLOAT_CRITIQUE.md` and findings. |
| Modal Leakage Critic: Leibniz `019e40ea-eb8b-7b20-802b-dd5245e340a6` | Done | Split Modal control-plane responsibilities from algorithm/config logic. | Folded into findings and extraction sequence. |
| Hook Bloat Critic: Hilbert `019e40ea-ece0-7c81-9b99-b019aba7d8b9` | Done | Critiqued hook installers, monkey patches, install/restore order, and first hook extraction. | Updated hook-bundle plan. |
| Pure Contract Order Critic: Poincare `019e40eb-2095-7c53-868c-5aea546b69a4` | Done | Critiqued extraction order across reward/config/opponent/checkpoint/status contracts. | Revised sequence. |
| Test Structure Critic: Turing `019e40eb-2215-7601-8d32-273cd5bc1e27` | Done | Found tests that lock in private trainer helper shape and proposed better boundaries. | Updated test migration plan. |
| Workflow Critic: McClintock `019e40eb-23da-76a0-9c07-bc5d7116655a` | Done | Critiqued docs/subagents/patch order/compatibility/handoff patterns. | Updated operating patterns and task board. |

## Current Critique Follow-ups

| Agent | Status | Ask | Follow-up Planned |
| --- | --- | --- | --- |
| Trainer Boundary Critic: Kant `019e40f1-72b6-7843-ab0e-295e66e05bf2` | Done | Re-critiqued why the trainer is huge relative to the small LightZero call. | Keep extraction order: reward, config, checkpoint publishing, opponent assignment, background eval/GIF planning. |
| Docs Freshness Critic: Galileo `019e40f1-74bd-7b90-92e2-c45773f55527` | Done | Found stale planning/doc statuses after reward extraction began. | This update supersedes stale “planning only” wording. |
| Batch Contract Critic: Curie `019e40f1-7637-7f33-9528-d09524b31131` | Done | Mapped collector env count, learner batch size, slot bag, role perspective, and immortality runtime. | Do not implement deterministic learner-batch splits; if used, deterministic split is collector-env assignment. |
| Test Gate Critic: Lagrange `019e40f1-77e8-75b1-a5e6-b1ee34f64f9e` | Done | Identified high-signal local gates and bloated/low-value tests. | Use targeted nodeids, not entire giant trainer/tournament suites, for local refactor gates. |

## Current Validation/Critique Agents

| Agent | Status | Ask | Follow-up Planned |
| --- | --- | --- | --- |
| Config Helper Critic: Mill `019e415e-8b4b-7632-83ae-2c588dd39f03` | Done | Critiqued how much of config builder should move now. | Next larger cut should use a typed builder spec/result and parity cases. |
| Opponent Contract Critic: Aquinas `019e415e-8d15-7252-b583-61b31cea0574` | Done | Audited deterministic split/assignment readiness contract. | Ready-report now proves split metadata, not just assignment id/ref/hash. |
| Local E2E Critic: Ampere `019e415e-8e85-7843-b3ca-11148f0ad393` | Done | Ran/identified local no-Modal gates. | Keep reward/config-helper/opponent/trainer smoke gates green before any launch. |
| Docs Freshness Critic: Gibbs `019e415e-900f-7dd0-8802-78bf97025f7f` | Done | Found stale docs after implementation moved. | This doc refresh folds in the required updates. |
| Docs Stale Audit: Lovelace `019e4167-c27f-79c3-9c63-44c4a645e295` | Done | Re-audit docs after the latest helper/opponent changes. | Stale claims folded into current phase, task board, compatibility ledger, source map, findings, and test plan. |
| Config Contract Audit: Hubble `019e4167-ccad-7b60-b72b-c1251665f512` | Done | Critique helper tests and missing cases before next extraction. | Fixed eval private import escape; expanded helper tests; remaining full-builder parity gap recorded. |
| Local Gate Audit: Hume `019e4167-d587-7be3-a3c6-5c5a00f8a501` | Done | Identify smallest additional local E2E commands/gaps. | Passing commands folded into findings/test plan; real-builder-to-fake-train smoke added and passing. |
| Builder Extraction Critic: Hegel `019e4198-6957-7c03-ab48-e0e66e696633` | Done | Re-checked smallest safe builder extraction and parity tests. | Main thread proceeded with public builder, typed spec/result, trainer/eval facades, and public import/builder gates. |
| Batch Contract Critic: Bacon `019e4198-6afe-78f1-999c-d099e0521919` | Done | Re-checked 64-vs-256 semantics and deterministic collector-env split contract. | Folded into batch plan: `batch_size=64` is learner replay batch, `collector_env_num=256` is collector scale, 64-slot recipes are collector assignments only. |
| Local Gate Critic: Boole `019e4198-6c86-71d3-b3e2-dd258abd3356` | Done | Re-checked smallest final local gate set for this cut. | Use reward/config, eval inferred support, trainer real-builder smoke, opponent split, and env reward/opponent reset gates for final validation. |
| Typed Config Boundary Critic: Mendel `019e41c2-fb9a-7591-82ee-04baaad8a556` | Done | Critiqued the typed spec/result shape after public builder extraction. | Main thread made `VisualSurvivalConfigSpec -> build_visual_survival_config(...) -> VisualSurvivalConfigResult` primary and kept the broad builder as a ledgered facade. |
| Typed Config Test Critic: Nietzsche `019e41c2-fd4d-7c60-a9ae-b7f104d170b7` | Done | Audited missing tests for typed builder migration. | Added parity/signature/frozen-opponent coverage and adopted the focused local spine for final validation. |
| Docs Stale Critic: Planck `019e41c2-ff2a-7053-9b45-3e51977629cb` | Done | Re-audited docs after public builder extraction. | Updated this folder to reflect the typed builder boundary and ledgered broad facade. |
| Parameter Sprawl Critic: Hooke `019e41f5-87e1-7e12-b01f-bd3b4487cc72` | Done | Split true experiment knobs from internal defaults and identified the smallest safe grouped-submit cut. | Main thread shrank grouped-submit required keys and documented the remaining manifest-builder migration. |
| Compact Surface Test Critic: Goodall `019e41f5-9b9e-7f63-9824-6f89030e2e9b` | Done | Proposed contract tests preventing internal knobs from leaking into `VisualSurvivalExperimentSpec`. | Main thread removed internal fields from the compact spec and added rejection/field-list tests. |
| Manifest Compaction Critic: Kepler `019e4200-b55b-7661-8419-c35333101cba` | Done | Audited which trainer functions/builders are safe for compact kwargs. | Keep full payloads for non-default builders; compact by omission only where omitted values equal trainer defaults. |
| Submitter Test Critic: Locke `019e4200-c9eb-7100-8695-e5ae558ba98d` | Done | Found stale submitter tests and missing compact-row gates. | Fixed stale `decision_ms` required-key test; added legacy, compact, defaulting, poller-guard, and compact experiment normalization tests. |
| Docs Freshness Critic: Schrodinger `019e4200-ddad-7a10-9dcb-9a416a2ccbe3` | Done | Re-checked docs after parameter-surface shrink. | Added legacy full `train_kwargs` ledger item and clarified compact submitter vs active manifest payload shape. |
| Tonight18 Compaction Critic: Jason `019e420a-9c70-7a70-8df5-335dec6d1c80` | Done | Audited which `tonight18` kwargs can be omitted and which runtime semantics must remain explicit. | Main thread compacted default-equal trainer fields only and kept initial checkpoint, opponent source, refresh, reward/noise, background non-defaults, and CLI overrides explicit. |
| Compact Submitter Guard Critic: Hypatia `019e420a-aede-7b13-aed3-51fb80297af0` | Done | Critiqued compact-row precedence/conflict risks. | Added guards for mixed `train_kwargs`/`experiment_spec`, train identity overrides, noise-bundle overrides, runtime/top-level ref conflicts, and train/poller identity divergence. |
| Tonight18 Docs Critic: Euler `019e420a-c50d-7893-9738-1439f9e5b51f` | Done | Listed docs that become stale if `tonight18` emits compact rows. | Updated compact schema, current phase, task board, compatibility ledger, knob surface, orchestration, findings, and migration plan. |

## Follow-up Rhythm

- First pass: read-only map with file/line references and risks.
- Second pass: challenge the first extraction plan before code edits.
- Third pass: after implementation, targeted test review and missing-case audit.
- Blocking rule: each scout/critic batch must be integrated into findings, decisions, and task-board changes before expanding scope again.

## Checkpoint Questions

- Are we changing behavior or only moving code?
- If behavior changes, where is that decision documented?
- What test would fail if this extraction broke the trainer?
- Is this reducing future edit surface for exploration bonus or side-network work?
- Are we preserving old behavior because it is correct, or only because it is old?
