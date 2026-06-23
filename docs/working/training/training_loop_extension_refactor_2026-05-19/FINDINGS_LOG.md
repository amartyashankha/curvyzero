# Findings Log

Last updated: 2026-05-19

Append findings here before turning them into decisions or code.

## Result Index

| ID | Source | Topic | Status | Summary |
| --- | --- | --- | --- | --- |
| F-001 | Main thread | Existing doc pattern | Recorded | Use task-local current phase, orchestration, operating pattern, task board, findings log, and focused design docs. |
| F-002 | Main thread | Older LightZero refactor lane | Recorded | Keep the trusted train path stock-owned, add tests before moving code, and keep pure helpers separate from Modal side effects. |
| F-003 | Helmholtz | Test and migration gates | Recorded | Existing tests cover many hot paths; missing gates are pure reward/support, config-builder parity, hook-bundle restore, env helper parity, and batch split integration. |
| F-004 | Averroes | Env step modularity | Recorded | `step()` is highly coupled; reward helper and telemetry helper are lower-risk cuts than opponent execution or observation cache movement. |
| F-005 | Kuhn | Hook bundle and extension seams | Recorded | Hook install/restore order is the core risk; future side networks need explicit callbacks and state, not ad hoc hook patches. |
| F-006 | Zeno | LightZero config builder boundary | Recorded | Builder can move out of Modal, but direct-call validation gaps and hidden defaults need tests; keep train aliases initially. |
| F-007 | Carson | Reward contracts | Recorded | Reward metadata/support are split across contracts, env, trainer, eval/status; first cut should move pure reward contract/config logic only. |
| F-008 | Euclid | Batch construction and opponent slot units | Recorded | `64` learner batch, `256` collector envs/episodes, and `64` opponent-slot bag are different units; deterministic split controls collector env assignment, not replay mini-batches. |
| F-009 | McClintock | Workflow critique | Recorded | The failure was docs without hard gates; added trainer patch gate, launch gate, compatibility ledger, and subagent integration gate. |
| F-010 | Poincare | Pure contract extraction order | Recorded | Reward first remains correct; then opponent assignment vocabulary, config builder, checkpoint metadata, status/progress; hook bundle after pure contracts. |
| F-011 | Turing | Test structure critique | Recorded | Tests currently treat the Modal trainer as a private helper library; migrate private-helper tests to public extracted modules and keep only orchestration integration tests on the trainer. |
| F-012 | Leibniz | Modal leakage critique | Recorded | Modal leakage is not just decorators; pure config/reward/eval/test code imports a deployable Modal app to get ordinary training contracts. |
| F-013 | Hilbert | Hook bloat critique | Recorded | Hook bloat is manual install/restore choreography and global patch risk; first hook cut should be PatchRegistry plus checkpoint-save bundle after pure contracts. |
| F-014 | Franklin | LightZero integration critique | Recorded | The trainer is not big because CurvyZero needs custom MuZero; it is big because checkpoint/resume/opponent/metrics/eval/config concerns landed beside the stock call. |
| F-015 | Main thread | Reward extraction result | Recorded | `reward_contracts.py` is wired through env/trainer/eval; focused reward gates pass. |
| F-016 | Curie | Batch semantics correction | Recorded | 64-slot opponent recipes are collector-env assignment recipes, not learner replay batch contracts. |
| F-017 | Main thread / Mill / Hubble / Hegel | Config builder extraction result | Recorded | `lightzero_config_builder.py` owns patch/path/surface helpers, env specs, validators, opponent relation helpers, the typed builder, and the broad compatibility facade; trainer/eval use public implementations. |
| F-018 | Main thread / Aquinas | Opponent split proof hardening | Recorded | Ready report now verifies expected per-env split metadata and tests pin it. |
| F-019 | Hubble / Hume | Eval private import escape | Fixed | Eval inferred-support branch re-imported trainer `_target_config_patches`; removed and pinned with a public-import boundary test. |
| F-020 | Hume / Main thread | Local validation pass | Recorded | Compact contract, no-Modal spine, runtime reward, and opponent guardrail tests pass locally. |
| F-021 | Main thread / Mendel / Nietzsche / Planck | Typed config boundary result | Recorded | `VisualSurvivalConfigSpec -> build_visual_survival_config(...) -> VisualSurvivalConfigResult` is primary; broad builder remains a parity-tested facade. |
| F-022 | Dirac / Bohr | Recent experiment batch inventory | Recorded | Recent batches mostly fixed training scale and varied reward, opponent recipe, noise, initial policy/control mode, refresh/static mode, and immortality; `cz26-full` expanded this to a 136-run grid. |
| F-023 | Main thread / Anscombe / Huygens | Compact experiment surface | Recorded | `VisualSurvivalExperimentSpec` expands deliberate experiment knobs into grouped internal config; tests pin current broad defaults, trainer facade parity, and observation perspective surface. |
| F-024 | Arendt | Policy observation perspective audit | Recorded | Training randomizes learner seat, tournament balances seats, policy observations are controlled-player-relative; standalone eval is random-by-seed, not explicitly balanced. |
| F-025 | Main thread / Hooke / Goodall | Parameter surface shrink | Recorded | Compact experiment spec rejects internal knobs; grouped submit only requires mode/seed/run/attempt and treats full `train_kwargs` as legacy overrides. |

## F-001: Existing Doc Pattern

Source docs reviewed from prior lanes used a consistent pattern:

- A local README explains the task lane and links the docs.
- A current-phase/current-truth doc states the goal, known truth, active lanes, and gates.
- An orchestration doc tracks main-thread work, subagents, follow-ups, and integration.
- A task board tracks work with status and done criteria.
- A separate operating-pattern doc captures the durable way to work, not the current plan.
- Findings and experiments are append-only until promoted into decisions.

Decision: this refactor gets its own task-local version of that pattern.

## F-002: Older LightZero Refactor Lane

Source docs reviewed:

- `docs/working/training/lightzero_train_refactor_2026-05-13/README.md`
- `docs/working/training/lightzero_train_refactor_2026-05-13/current_source_of_truth.md`
- `docs/working/training/lightzero_train_refactor_2026-05-13/operating_patterns.md`
- `docs/working/training/lightzero_train_refactor_2026-05-13/refactor_sequence.md`

Useful patterns to preserve:

- The trusted training lane should call stock LightZero; CurvyZero supplies env, config, checkpoint/artifact plumbing, and observability.
- Tests come before broad refactors.
- Pure helper contracts should move first; Modal volume reloads, sleeps, remote calls, and live status writes should stay near Modal wrappers until the pure part is stable.
- Main thread should plan, delegate, integrate findings, and decide the next move rather than disappearing into a giant edit.
- Exact file/function refs matter more than broad claims.

Decision: this lane will use the same discipline, but focused on extension boundaries: reward contracts, config builder, hook bundle, env helpers, batch construction, and test gates.

## F-003: Test And Migration Gate Scout

Source: Helmholtz `019e40dd-9764-7373-affa-2f3953c557d9`.

Existing coverage:

- Reward/support code is currently in the Modal trainer and source-state env:
  - `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:650`
  - `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:781`
  - `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7539`
  - `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1698`
- Reward behavior has tests around survival-plus-bonus, same-step bonus, terminal exclusion, plus-outcome, and scaled terminal reward in `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`.
- Config construction tests already cover mocked stock train entrypoint, registered env smoke, and several config knobs in `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Learner-seat and policy-perspective behavior has focused coverage in `tests/test_source_state_visual_survival_learner_seat_regression.py`.
- Opponent mixture and immortality have tests in `tests/test_opponent_mixture.py` and env-level immortal tests in `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`.
- Hook/checkpoint behavior has existing coverage for progress writer, sidecar, save hook, live publisher, trigger scheduling, and retry gates in `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.

Missing or weak gates:

Historical scout list; several items are now covered by F-015 through F-020. Keep this section as original finding, not current task state.

- Pure reward/support tests for an extracted module, including `auto` normalization, invalid alpha, invalid support cap, and invalid `td_steps`.
- Builder parity tests comparing the current `_build_visual_survival_configs` output to a new builder for fixed, proactive, frozen, mixture, assignment-context, and learner-seat cases.
- Hook-bundle install/restore tests that prove original method preservation and install order.
- Metadata tests proving builder surface fields feed checkpoint metadata sidecars.
- Env helper parity over a short fixed seed/action sequence: observation tensor, reward, done, and key info fields.
- Batch split tests tying deterministic slot plans to actual collector reset params and rejecting incompatible `collector_env_num` at config-build time.

Recommended fast commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_opponent_mixture.py tests/test_lightzero_checkpoint_opponent_provider.py tests/test_source_state_visual_survival_learner_seat_regression.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -k 'survival_plus_bonus or opponent_immortal or opponent_mixture or player_perspective or blank_canvas_noop'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_curvytron_live_checkpoint_eval_plumbing.py -k 'survival_plus_bonus or target_support or modal_config or checkpoint_progress_writer or save_ckpt_hook or live_checkpoint_publisher or fresh_resume_hooks or opponent_assignment_refresh or stock_source_state_mixture_config'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_trainer_contract.py -k 'reward or perspective or terminal'
```

Default staged gates:

1. Reward extraction: add pure tests, then run reward/support and env reward commands.
2. Config builder extraction: old call site delegates to new builder; run config command and builder parity.
3. Env step modularity: extract reset/step/reward/observation helpers; run source env, learner-seat, and trainer-contract commands.
4. Opponent slot migration: run `tests/test_opponent_mixture.py` plus assignment-refresh tests.
5. Hook bundle extraction: run hook/publisher/resume tests; require original calls preserved and no Modal spawn in local tests.
6. Final local smoke: registered env instantiation plus mocked `train_muzero` entrypoint only. Modal launch stays out of this refactor lane.

## F-004: Env Step Modularity Scout

Source: Averroes `019e40dd-9421-7483-acc8-16bf128f9b66`.

Important source refs:

- Learner-seat config/init: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:454`, `503`, `3076`.
- Reset flow: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:805`.
- Seat selection: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:874`, `883`.
- Episode opponent selection: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:904`.
- Step knot: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:939`.
- Opponent action execution: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1462`.
- Observation stack/render: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1335`, `1344`.
- Reward components: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1670`, `1698`.
- Step info and base info: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1841`, `1972`.
- Telemetry writer: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:2472`.

Key findings:

- `step()` mixes control noise, opponent action, vector stepping, action repeat, reward accumulation, observation render, final observation, info payload, and telemetry.
- Opponent mixture is selected once per reset.
- `opponent_immortal` and source bonus `invincible` are not the same thing. Immortality is a death-immunity mask. Bonus invincibility currently blocks body/trail collision death, but not wall death.
- Blank-canvas noop is not just a view filter. It mutates disabled masks, source state, and render masks.
- `random_per_episode` seat variation depends on `_reset_index`; fixed-seed env use may not rotate seats if `_next_seed` does not advance reset index.
- `_validate_policy_action_repeat_config` and `_default_control_noise_profile_id` appear twice in the env class. Later definitions win.

Lower-risk helper candidates:

- `resolve_learner_seat(mode, reset_seed, reset_index) -> ego_index`.
- `build_joint_action(ego_index, opponent_index, executed_ego_action, opponent_action)`.
- `compute_reward_components(variant, source_reward, alive, catch_count, done, accumulated_return, alpha)`.
- `build_step_info(base_info, step_result, reward_summary, final_maps)`.
- `build_env_step_telemetry_row(info, timestep_reward, sampled_stride, opponent_metadata)`.

Higher-risk areas to defer:

- Opponent execution, especially frozen checkpoint action selection.
- Observation cache mutation and renderer state.
- Blank-canvas noop semantics.

Follow-ups:

- Decide fixed-seed `random_per_episode` semantics before extracting seat selection.
- Add parity tests for reward helper across variants and terminal/action-repeat cutoff.
- Add telemetry schema/key parity before moving telemetry.
- Audit whether `PLAYER_PERSPECTIVE_SCHEMA_ID` is stale versus emitted policy-observation contract metadata.

## F-005: Hook Bundle And Extension Scout

Source: Kuhn `019e40dd-917f-7d33-a26f-1caf6243e555`.

Important source refs:

- Current install order in `_run_visual_survival_train`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5003`.
- Restore order pieces: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5155`.
- Live checkpoint publisher: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2000`.
- Progress writer: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2574`.
- Own-checkpoint opponent publisher: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2287`, `2430`.
- Learner metrics: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2115`, `2636`.
- Full resume hooks: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2699`.
- Opponent refresh: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5867`, `6064`, `6224`, `6259`.
- Target audit: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3922`, `4043`.

Key findings:

- Multiple installers patch the same methods, especially `BaseLearner.save_checkpoint`, `BaseLearner.train`, `Collector.collect`, and `GameBuffer.sample`.
- Current behavior depends on install order and restore order.
- Hook discovery is LightZero-version fragile because it uses `train_muzero.__globals__`, fallback imports, and owner lookup.
- Resume is not a true full replay resume. Raw replay segments are metadata-only.
- Opponent refresh can partially reset envs before proof failure. It raises before collect, but there is no rollback.
- Learner metrics sees train result, not learner batch. Target audit sees replay samples only compactly. A side network needs explicit batch/state callbacks.

Proposed shape:

- `LightZeroHookBundle.install(context, extensions=()) -> HookBundleHandle`.
- `HookBundleHandle` owns ordered installed hooks, `restore()`, learner metrics recorder, target audit, and events.
- Use one shared patch registry with explicit order instead of ad hoc patch state in each installer.
- Minimal `TrainingExtension` callbacks:
  - `on_before_run`
  - `on_replay_sample`
  - `before_learner_train`
  - `after_learner_train`
  - `on_checkpoint`
  - `state_dict`
  - `load_state_dict`
  - `progress_payload`
- First extension mode should be passive/no-op only. Mutation of batches, losses, rewards, or collector policy should require explicit capability flags.

Tests to add before hook extraction:

- Bundle-order test proving nested `save_checkpoint` wrappers fire in current order and restore cleanly.
- No-op extension test proving callbacks observe train/sample/checkpoint without changing return values.
- Extension resume roundtrip test for model and optimizer state in a sidecar.
- Failure-policy tests for extension checkpoint failure, replay-sample callback exception, and partial opponent-refresh proof failure.

## F-006: LightZero Config Builder Boundary Scout

Source: Zeno `019e40dd-8e83-7091-9914-749e80c4829f`.

2026-05-19 update: reward contracts, public config-builder extraction, and typed config-builder boundary have landed. Surface omissions for learner seat, policy-observation backend, natural-bonus spawn, and opponent assignment context are fixed. The remaining gap is shrinking protected callers off the broad compatibility facade, not moving builder logic out of the trainer again.

Important source refs:

- Primary typed builder: `src/curvyzero/training/lightzero_config_builder.py::build_visual_survival_config`.
- Typed input/result: `VisualSurvivalConfigSpec`, `FrozenOpponentConfig`, `VisualSurvivalConfigResult`.
- Broad compatibility facade: `src/curvyzero/training/lightzero_config_builder.py::build_visual_survival_configs`.
- Trainer launch facade: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py::_build_visual_survival_configs`.
- Eval imports the public builder and `target_config_patches` from `curvyzero.training.lightzero_config_builder`.
- Builder result fields remain `template_module`, `main_config`, `create_config`, `surface`, `patches`.

Input groups:

- Runtime paths and flags: seed, exp name, telemetry path, cuda, multi-GPU, env manager type.
- LightZero knobs: collector/evaluator env counts, episode counts, simulations, batch size, eval frequency, train/env limits, checkpoint cadence.
- Env/reward/observation knobs: env variant, reward variant, reward outcome alpha, render modes, policy observation backend, learner-seat mode, action repeat, death/runtime modes.
- Target support: model support cap and `td_steps`.
- Opponent data: policy kind, use-cuda flag, resolved checkpoint dict, snapshot/state key, mixture, assignment context.

Hidden defaults and blind spots:

- `reward_variant="auto"` normalizes per env; current default fixed-opponent becomes sparse outcome.
- `DEFAULT_MODEL_SUPPORT_CAP=None` is not really uncapped for fixed-opponent; effective default cap is 300.
- `DEFAULT_TD_STEPS=None` preserves template default for fixed-opponent, but joint-action uses `source_max_steps`.
- `lightzero_eval_freq=0` becomes `max_train_iter + 1`.
- Builder hardcodes the Atari MuZero template path, model type `conv`, image channel 4, frame stack 1, SSL loss on, grayscale env config, dynamic seed, and `max_ticks=source_max_steps`.
- Frozen opponent `opponent_snapshot_ref` silently defaults to `curvytron_visual_survival_frozen_opponent`.
- `extract_visual_survival_surface` now records learner-seat mode and policy-observation backend defaults/supports.
- Surface validation coverage has moved into public builder/surface tests; add more only when those paths change.
- Env has stricter JAX GPU backend compatibility than the builder validates.

Import-cycle and migration guidance:

- Moving config builder into `curvyzero.training.lightzero_config_builder` removes Modal from direct config tests.
- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py` imports the public config-builder helpers; tests guard against reintroducing trainer-private builder/target-patch imports.
- Keep compatibility aliases in the Modal trainer first because tests and tournament code import private train symbols.
- Avoid importing large experimental trainer modules just for defaults.

First extraction plan:

1. Create `src/curvyzero/training/lightzero_config_builder.py` with copied behavior and no Modal imports.
2. Add typed dataclasses only around the public boundary: `VisualSurvivalConfigSpec`, `FrozenOpponentConfig`, `VisualSurvivalConfigResult`.
3. Keep `LightZeroConfigBuildResult` only as a compatibility alias while tests/callers settle.
4. Replace Modal trainer private function bodies with imports/aliases from the new module.
5. Update eval module imports to the new module while keeping train aliases.
6. Keep command construction in the Modal trainer for the first pass.
7. Add direct builder tests that do not import the Modal trainer.

Decision for now: do not start config-builder extraction before reward contracts, because reward/support logic is the least config-shaped part of the builder and has a cleaner first cut.

## F-007: Reward Contracts Scout

Source: Carson `019e40dd-9018-7560-904c-e03f76c996f5`.

Important source refs:

- Core reward variant strings: `src/curvyzero/contracts/curvytron.py:123`.
- Env allowed variants and diagnostic variants: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:284`.
- Trainer-only `auto` and default: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:450`.
- Reward normalization: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:590`.
- Trainer reward policy metadata: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:650`.
- LightZero support config: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:781`.
- Target config patches: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7539`.
- Scalar reward math: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1698`.
- Reward space bounds: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1809`.
- Env reward telemetry fields: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1925`, `2472`.
- Eval reward components: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:909`, `954`.
- Status reward rollup: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:483`.

Key findings:

- Reward variant names, allowed variant sets, schema metadata, scalar reward math, reward bounds, support config, and telemetry fields are split across several files.
- `auto` is trainer-only and must be normalized before env construction.
- Env reward bounds account for `policy_action_repeat_max`; trainer support config does not visibly take repeat max as input.
- Eval/status reward components omit `sparse_outcome` and `terminal_outcome`, even though plus-outcome telemetry exposes terminal fields.
- Effective default support cap is 300, but public default is `None`.
- There are historical/local pseudo-variants, such as `survival_only`, that need an explicit decision before being treated as canonical.

Proposed first extraction:

- Create `src/curvyzero/training/reward_contracts.py`.
- Move pure contract/config logic first:
  - variant constants or re-exports;
  - allowed-variant sets;
  - `REWARD_VARIANT_AUTO`;
  - `normalize_reward_variant_for_env`;
  - `normalize_reward_outcome_alpha`;
  - reward schema ids/hashes and bonus constants;
  - `reward_policy_for_variant`;
  - `lightzero_target_config_for_reward`;
  - reward-space bounds helper.
- Leave `_reward_components_for_player` in the env for the first patch, but let it read contract metadata from the new module later.
- Make trainer and eval import the new module instead of private reward helpers from the Modal trainer.

Focused tests:

- Direct unit tests for reward contract matrix: allowed variants, schema ids/hashes, policy metadata, invalid variant, invalid alpha.
- Preserve existing support expectations in `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Preserve env reward behavior tests around survival/bonus/outcome in `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`.
- Add a small bounds-vs-support test for `policy_action_repeat_max > 1`; decide whether it documents current behavior or drives a later semantic fix.

Open questions:

- Should `auto` live in reward contracts or stay trainer-only? Current leaning: move it into reward contracts because it is part of trainer-facing reward contract validation, but keep env-facing values normalized.
- Should `policy_action_repeat_max` affect LightZero reward support now? Current leaning: document current behavior first; semantic fix is separate.
- Should canonical `reward_components` include `sparse_outcome` and `terminal_outcome`? Current leaning: yes eventually, but not in the first extraction.
- Should multiplayer trainer surface switch to the same reward contract immediately? Current leaning: no, leave it out of first cut unless import cycles force a small alias.

## F-008: Batch Construction And Opponent Slot Units

Source: Euclid `019e40dd-95ce-73b0-ae57-80febd819b7c`.

Definitions:

- `64` is currently the learner replay mini-batch size: `CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE = 64` in `src/curvyzero/contracts/curvytron.py:103`.
- That value is wired to `policy.batch_size` in `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6879`.
- `256` is the default collector env count and episodes-per-collect: `src/curvyzero/contracts/curvytron.py:101`.
- The trainer exposes those as `collector_env_num` and `n_episode` around `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:11942`.
- Each LightZero env instance wraps a single-row game env: `VectorMultiplayerEnv(batch_size=1)` at `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1144`.
- The tonight18 64-slot bag is a collector-opponent slot total, not the learner batch size, even though both use the number 64.

Sampling flow:

- Raw opponent mixture sampling happens per env reset, effectively per episode.
- `select_opponent_mixture_entry(...)` is deterministic from mixture seed, episode seed, and reset index.
- With default `collector_env_num=256` and `n_episode=256`, one collect call is normally one episode per collector env.
- Learner batches are sampled from replay transitions. A learner update’s opponent mix depends on replay sampling, episode lengths, priority, buffer history, and timing.

Deterministic split status:

- `deterministic_collector_env_mixture_plan` already exists in `src/curvyzero/training/opponent_mixture.py:145`.
- It turns integer mixture weights into shuffled collector-env assignments.
- It requires slot total to be a power of two, no larger than `env_num`, and dividing `env_num`.
- For 64 slots over 256 envs, the repeat count is 4.
- Assignment refresh applies the plan by replacing each env’s next-reset mixture with a singleton mixture.
- Direct trainer default refresh interval is `0`, while the tonight18 manifest uses assignment refresh interval `2000`.

Risks:

- The overloaded `64` is a footgun: learner batch size, opponent slot count, and background eval batch size can all be 64 but mean different things.
- Deterministic collector-env split does not guarantee deterministic learner mini-batch composition.
- Raw mixture mode and assignment-refresh mode have different semantics.
- The power-of-two rule is a local split-contract rule, not a LightZero requirement.
- Periodic assignment refresh resets collector envs before collect, which is a real behavioral event and needs metadata/tests.

Follow-ups:

- Rename or document units as `learner_batch_size`, `collector_env_num`, `episodes_per_collect`, and `opponent_slot_count_total`.
- Add a contract note that deterministic split controls collector env assignment only, not replay learner batch composition.
- Keep or add a fake-collector test proving raw `opponent_mixture_spec` stays episode-reset weighted sampling while assignment refresh uses singleton env slots.
- Decide later whether the power-of-two rule is a long-term research contract or just tonight18 convenience.

## F-009: Workflow Critique

Source: McClintock `019e40eb-23da-76a0-9c07-bc5d7116655a`.

Core critique:

- Trainer bloat persisted because guidance was documented but not made operationally binding.
- Urgent runtime fixes still had an easy path into the giant trainer.
- Private helper imports became informal APIs.
- Compatibility aliases and hidden fallbacks let old behavior survive without re-justification.
- Launches were sometimes treated as discovery instead of confirmation.

Corrections applied:

- Added hard gates to `README.md`.
- Added patch order, trainer patch gate, compatibility ledger, and subagent integration gate to `OPERATING_PATTERNS.md`.
- Made scout/critic batch integration blocking in `ORCHESTRATION.md`.
- Added an acceptance checklist to `TRAINER_BLOAT_CRITIQUE.md`.
- Added task-board items:
  - `TLR-WORKFLOW-001`
  - `TLR-COMPAT-001`
  - `TLR-API-001`
  - `TLR-LAUNCH-001`

Decision: these gates are binding for this refactor lane. Trainer edits must reduce responsibility, move a contract out, preserve a real Modal side effect with a reason, or be a ledgered temporary shim.

## F-010: Pure Contract Extraction Order

Source: Poincare `019e40eb-2095-7c53-868c-5aea546b69a4`.

Verdict:

- Reward contracts should leave first.
- After reward, the current planned order should change. It should follow data-contract dependencies, not implementation areas.

Accepted order:

1. Reward contract:
   - variant normalization;
   - alpha validation;
   - reward policy metadata;
   - schema ids/hashes;
   - support config;
   - reward-space bounds.
   - Do not move scalar env reward math yet.
2. Opponent assignment contract:
   - unit names: `learner_batch_size`, `collector_env_num`, `episodes_per_collect`, `opponent_slot_count_total`;
   - raw mixture versus assignment refresh semantics;
   - deterministic split means collector assignment, not replay mini-batch composition.
3. Config builder contract:
   - typed inputs/results;
   - consume reward and opponent contracts instead of preserving trainer-shaped dict soup.
4. Checkpoint metadata contract:
   - pure sidecar payload construction after builder surface is stable.
5. Status/progress payload contracts:
   - pure payload builders after reward/opponent/metadata surfaces stabilize.
6. Hook bundle/runtime publisher:
   - after pure contracts are stable, because hook install/restore is higher risk.

Decision: update task board and test plan to make this the current extraction sequence.

## F-011: Test Structure Critique

Source: Turing `019e40eb-2215-7601-8d32-273cd5bc1e27`.

Core critique:

- The tests are useful, but many treat `lightzero_curvyzero_stacked_debug_visual_survival_train.py` as a private helper library.
- That turns the giant trainer file into an accidental API.
- The test suite should lock down public contracts, not private trainer structure.

Highest-risk couplings:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py` imports the giant trainer and exercises many `_private` helpers.
- `tests/test_opponent_mixture.py` starts with public opponent-mixture tests, then reaches into trainer-private resolution helpers.
- `tests/test_lightzero_timestamped_checkpoint_discovery.py` starts with `curvyzero.training.lightzero_checkpoints`, then falls back to trainer-private progress/resume/poller helpers.
- `tests/test_curvytron_training_candidate_controller_local.py` imports the trainer for assignment resolution, refresh application, and lineage.
- `tests/test_lightzero_phase_profiler.py` imports phase-profiler hook internals from the trainer.
- `tests/test_curvytron_shared_contracts.py` imports the trainer to assert constants.

Recommended migration:

- Reward/support tests should import `curvyzero.training.reward_contracts`.
- Config-builder tests should move to `tests/test_lightzero_config_builder.py` and import a public builder, not Modal.
- Opponent assignment/runtime tests should target a public `opponent_assignment_runtime` module.
- Hook tests should target a public `lightzero_hooks` module.
- Checkpoint/resume/publishing tests should target `lightzero_checkpoints`, `checkpoint_publishers`, or a resume module.
- Env/readiness/frame helper tests should move out of trainer-private tests when those helpers become public.

What should remain trainer integration:

- `_run_visual_survival_train` calls stock `train_muzero` with built configs.
- Trainer-level refresh hook installation/failure behavior.
- Poller/background eval orchestration with fake Modal functions.
- Tournament/controller to trainer visibility chain, through public assignment APIs.

Decision: add public-module tests first, keep trainer-private tests temporarily as parity, flip trainer to delegate, then delete or shrink private-helper tests once integration smoke passes.

## F-012: Modal Leakage Critique

Source: Leibniz `019e40ea-eb8b-7b20-802b-dd5245e340a6`.

Core critique:

- The leakage is not just Modal decorators. The deeper problem is that pure contracts live in a Modal module.
- Config, reward, eval, tests, and tournament code import a deployable Modal app to get ordinary training logic.

Should stay in the Modal trainer:

- Modal image, volumes, app, and mount names.
- `@app.function` wrappers and resource choices.
- Local entrypoint routing, `.spawn()`, `.remote()`, function call IDs.
- Volume reload/commit/ref adapters.
- Thin final orchestration around `train_muzero`: resolve refs/paths, build config, install hook bundle, call `train_muzero`, write final status.

Should move:

- Reward/config contract logic.
- Env/config builder surface.
- LightZero monkeypatch/hook bundle.
- Checkpoint metadata/progress payload builders.
- Background eval/GIF planning, while Modal `.spawn()` stays.

Historical leak evidence:

- `lightzero_curvytron_visual_survival_eval.py` used to import private builder/helpers from the Modal trainer; current eval imports public config helpers and boundary tests guard this.
- Tests import the Modal trainer for config/opponent/checkpoint helpers.
- Scripts encode the Modal module path as the training API.

Suggested extraction sequence:

1. `reward_contracts.py`.
2. `lightzero_config_builder.py`.
3. `checkpoint_contracts.py`.
4. Modal volume IO boundary.
5. `lightzero_hooks.py`.
6. `opponent_assignment_runtime.py`.
7. `background_eval_plan.py`.
8. Thin `_run_visual_survival_train` adapter.

Decision: this supports the current first move, but we will keep Poincare's dependency-aware order for pure contracts: reward, opponent assignment vocabulary, config builder, checkpoint metadata, status/progress, then hooks/runtime publishers.

## F-013: Hook Bloat Critique

Source: Hilbert `019e40ea-ece0-7c81-9b99-b019aba7d8b9`.

Core critique:

- Hook bloat is the clearest example of the trainer becoming a control plane.
- The trainer owns profiling, checkpoint progress, live eval spawning, resume sidecars, initial policy loading, target audit, learner metrics, and opponent refresh as local monkey patches.
- The file is large partly because behavior is encoded in manual install/restore choreography.

Mostly necessary:

- Native LightZero config hook knobs, such as save/load checkpoint config mutation.
- Checkpoint progress writer.
- Live checkpoint publisher when background eval launch kind is hook.
- Resume sidecar hooks, with the caveat that this is best-effort, not true full replay resume.
- Opponent assignment refresh.
- Initial policy checkpoint load, though it is misnamed as audit while mutating load behavior.

Mostly optional diagnostics or structural debt:

- Phase profiler: useful but huge and invasive.
- Learner metrics recorder: should become an extension callback around learner train.
- Target audit: useful read-only audit, but optional and overlapping with profiler hooks.

Order risks:

- `BaseLearner.save_checkpoint` is wrapped by progress writer, live publisher, and phase profiler.
- `Collector.collect` is wrapped by opponent refresh, target audit, and phase profiler.
- `torch.load` is wrapped by initial-policy load audit and resume trusted-load retry.
- `BaseLearner.call_hook` is wrapped by resume and possibly profiler.
- Restore order must be correct or global patches can remain wrong.

Smaller bundle shape:

- `LightZeroSymbols`: resolves LightZero classes/functions once.
- `PatchRegistry`: one place for `setattr`, global replacement, `torch.load` patching, and reverse restore.
- `HookBundleConfig`: enables checkpoint progress, live eval, resume, opponent refresh, metrics, target audit, profile.
- `HookBundleHandle`: owns `restore()`, installed hooks, events, metrics recorder, target audit, and profiler summary.
- Ordered callbacks: `before_collect`, `after_collect`, `before_train`, `after_train`, `after_checkpoint`, `before_run`, `state_dict`, `load_state_dict`.

First safe hook extraction:

- Add a stacked-order test for stock save -> progress write -> live spawn -> clean restore.
- Extract `PatchRegistry` plus checkpoint-save bundle for `_install_checkpoint_progress_writer` and `_install_live_checkpoint_publisher`.
- Defer resume, opponent refresh, and initial-policy loading.

Decision: do not start hook extraction until reward/opponent/config/metadata/progress pure contracts are stable.

## F-014: LightZero Integration Critique

Source: Franklin `019e40ea-e9b9-7e90-8fc5-d2eb9d738c06`.

Bottom line:

- The actual integration is small: stock `train_muzero` is called at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5118`.
- The file is not large because CurvyZero needs custom MuZero.
- It is large because checkpoint publishing, resume, opponent refresh, metrics, target audit, Modal volumes, eval/GIF side work, and config/reward contracts all landed beside that stock call.

Only because LightZero lacks hook points:

- Checkpoint side effects around `BaseLearner.save_checkpoint`.
- Full resume hooks around LightZero internals.
- Opponent assignment refresh around `Collector.collect`.
- Target audit and learner metrics as observer callbacks.
- Initial policy loading via global `torch.load` and load-state patches.

Should be pure wrappers:

- Config construction.
- Reward/support math.
- Opponent mixture and deterministic collector plan.
- Train launch argument payload builder.

Should disappear from this file:

- Custom two-seat branch from the stock trainer lane.
- Private trainer imports from eval, tournament, GIF subscriber, and tests.
- Hidden defaults such as frozen opponent snapshot fallback and implicit effective support cap.

Decision:

- Finish deleting reward/support duplication by wiring the trainer and eval through `reward_contracts.py`.
- Keep the custom two-seat cleanup as a later separate task; do not mix it into the reward first cut.

## F-015: Reward Contract Extraction Result

Source: main thread implementation, after Kant/Galileo/Curie/Lagrange follow-up critiques.

What changed:

- Added `src/curvyzero/training/reward_contracts.py` as the shared source for reward variant constants, reward schema ids/hashes, reward policy metadata, reward perspective labels, reward-space bounds, reward alpha validation, and LightZero support config.
- Added `tests/test_reward_contracts.py` for reward normalization, alpha validation, support cap validation, `td_steps`, and reward-space/support behavior.
- Wired the source-state env reward metadata helpers to delegate to `reward_contracts.py`; the env still owns runtime scalar reward computation.
- Wired the Modal trainer reward helper surface to delegate to `reward_contracts.py`; trainer reward constants are now facade imports from the shared module rather than duplicate definitions.
- Wired eval model-target support config to use `reward_contracts.py`; target support patching later moved to public `lightzero_config_builder.py`.

Validation run locally:

- `uv run pytest -q -p no:cacheprovider tests/test_reward_contracts.py` -> 6 passed.
- Focused source-state env reward/metadata tests -> 6 passed.
- Focused trainer plumbing tests for stock `train_muzero` and source-state opponent mixture config -> 2 passed.

Decision:

- Reward contracts are partially extracted and wired.
- Continue extractions only with the compatibility ledger current and helper/import habits pinned with public-module tests.

## F-016: Current Batch-Semantics Correction

Source: Curie `019e40f1-7637-7f33-9528-d09524b31131`.

Plain finding:

- Current defaults use `collector_env_num=256` and `n_episode=256`.
- Current learner update `batch_size=64`.
- A 64-slot opponent recipe is a collector-env assignment recipe. It is not the learner replay mini-batch.
- Deterministic opponent splitting, when used, assigns opponent singleton mixtures per collector env. The learner replay batch can still skew based on transition history, episode length, priority, and replay sampling.

Decision:

- Do not implement “deterministic learner batch splits” from slot recipes. If we expose deterministic slots, name them as collector-env split contracts.

## F-017: Config Builder Extraction Result

Source: main thread implementation after Mill, Hubble, and Hegel critiques.

What changed:

- Added `src/curvyzero/training/lightzero_config_builder.py`.
- Added `tests/test_lightzero_config_builder.py`.
- Public helper module now owns `to_plain`, `set_or_add_path`, `get_path`, `target_config_patches`, `set_save_ckpt_after_iter`, `set_load_ckpt_before_run`, `env_variant_spec`, render/backend/seat/cadence validators, opponent relation helpers, `build_visual_survival_configs`, and `extract_visual_survival_surface`.
- The Modal trainer delegates those helper/builder functions instead of keeping local duplicate logic.
- Eval imports public `build_visual_survival_configs` and `target_config_patches` from the helper module.
- The visual-survival surface now records learner-seat mode, policy-observation backend defaults, supported policy-observation backends, natural-bonus spawn, and opponent assignment context.
- The primary builder boundary is now `VisualSurvivalConfigSpec -> build_visual_survival_config(...) -> VisualSurvivalConfigResult`.
- The broad `build_visual_survival_configs(**kwargs)` function is a compatibility facade through the typed path.

What is not done:

- Protected trainer/eval/test callers still use the broad historical facade; move them only when it makes the surrounding code smaller and clearer.
- Trainer facade wrappers remain until launch/tests stop importing the trainer as a helper module.
- Frozen checkpoint snapshot fallback is still only inventoried; decide keep/delete during checkpoint metadata extraction.

Validation:

- `uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py` -> 16 passed after typed spec/result parity and frozen-opponent coverage.
- `uv run pytest -q -p no:cacheprovider tests/test_reward_contracts.py tests/test_lightzero_config_builder.py` -> pending rerun after typed cleanup in this local turn.

## F-018: Opponent Split Proof Hardening

Source: main thread implementation after Aquinas critique.

What changed:

- `_opponent_assignment_refresh_ready_report(...)` now builds the expected reset-param split plan and compares each env's reported split metadata.
- Required proof fields now include split unit, split mode, split plan hash, env index, env count, entry name, and entry count.
- Slot-count reporting prefers `opponent_split_entry_name`, falling back only for older info payloads.
- Added/updated tests so missing split metadata fails the ready report.

Validation:

- Current compact spine after the `tonight18` migration -> 35 passed.
- Opponent guardrail command -> 63 passed.

## F-019: Eval Private Import Escape Fixed

Source: Hubble and Hume critiques.

Problem:

- Eval had a module-level public import for `target_config_patches`, but the checkpoint-inferred-support branch re-imported trainer-private `_target_config_patches` inside `_make_policy_and_env(...)`.
- That meant the cleanup was incomplete exactly in the branch that patches checkpoint head-size support.

Fix:

- Removed the inner trainer-private import.
- Added a boundary test that parses the eval source and fails if `_target_config_patches` is imported from the Modal trainer again.
- Added a real-builder-to-fake-stock-train smoke proving `_run_visual_survival_train(...)` can feed the actual builder output into a fake `train_muzero` and instantiate/step the registered source-state env.

Validation:

- `uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py` -> 8 passed.
- No-Modal eval/trainer spine command -> 9 passed, 92 deselected.

## F-020: Current Local Validation Pass

Source: Hume and main thread.

2026-05-19 current update after the `tonight18` compact-manifest migration:

Passing commands:

```bash
uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py tests/test_reward_contracts.py tests/test_opponent_mixture.py::test_singleton_mixture_preserves_entry_refs_and_only_reweights tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_refresh_ready_report_requires_all_envs tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_real_builder_config_reaches_fake_lightzero_entrypoint
```

Result: 35 passed.

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvytron_live_checkpoint_eval_plumbing.py -k 'stock_train_mode_calls_lightzero_train_muzero_entrypoint or stock_train_mode_real_builder_config_reaches_fake_lightzero_entrypoint or stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action or survival_plus_bonus_no_outcome_uses_capped_separate_supports or survival_plus_bonus_plus_outcome_alpha_threads_to_env_policy_and_supports or fixed_opponent_target_support_and_td_steps_can_be_overridden or eval_episode_and_tables_preserve_reward_components or background_eval_inspection_and_gif_can_be_explicitly_enabled or test_live_checkpoint_trigger_spawns_eval_and_selfplay_gif_without_volume_commit or checkpoint_eval_poller_completes_eval_inspection_and_selfplay_gif_jobs or eval_infers_model_support_from_checkpoint_head_shapes or make_policy_and_env_applies_checkpoint_inferred_support_with_public_patches'
```

Result: 12 passed, 93 deselected.

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -k 'survival_plus_bonus_no_outcome or survival_plus_bonus_plus_outcome or opponent_mixture_refresh_applies_on_reset_and_records_assignment_context'
```

Result: 8 passed, 41 deselected.

```bash
uv run pytest -q -p no:cacheprovider tests/test_opponent_mixture.py tests/test_opponent_registry.py tests/test_opponent_leaderboard.py tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_opponent_mixture_selects_once_per_reset_not_per_step tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_opponent_mixture_refresh_applies_on_reset_and_records_assignment_context tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_context_is_passed_to_env_config tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_refresh_reset_param_uses_exact_collector_slot_split
```

Result: 63 passed.

The previously identified real-builder-to-fake-train local gap is now covered.

## F-021: Typed Config Boundary Result

Source: main thread after Mendel, Nietzsche, and Planck critiques.

What changed:

- Added `VisualSurvivalConfigSpec.from_builder_kwargs(...)` so the old broad keyword bag is converted explicitly.
- Added `build_visual_survival_config(spec)` as the primary builder API.
- Grouped `VisualSurvivalConfigSpec` into run/runtime, training scale, timing, observation, behavior, reward/target, and opponent sections.
- Renamed the typed result to `VisualSurvivalConfigResult`, with `env_config` and `lightzero_target_config` properties.
- Kept `LightZeroConfigBuildResult` as a compatibility alias only.
- Made `build_visual_survival_configs(**kwargs)` delegate through the typed spec/result path and return the historical dict shape.
- Added tests for broad/typed parity, broad signature vs typed keyword conversion, grouped knob placement, unknown-kwarg rejection, opponent mixture/context parity, and frozen checkpoint env/surface fields.

Validation so far:

- `uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py` -> 23 passed.

Remaining risk:

- Trainer and eval still intentionally consume the historical dict shape because those paths mutate `main_config`/`surface`. Do not return typed results there until a focused eval/trainer gate proves the mutation semantics are preserved or deleted.
- This is still normalized config, not a launch UX. A smaller experiment-facing preset/knob surface should be designed separately from the builder internals.

## F-022: Recent Experiment Batch Inventory

Source: Dirac and Bohr read-only side investigations.

Plain finding:

- Recent 18-row batches mostly fixed training scale and varied `3 reward variants x 3 opponent recipes x 2 noise modes`.
- Main fixed scale in the 18-row manifests: `collector_env_num=256`, `n_episode=256`, `batch_size=32`, `num_simulations=8`, `max_env_step=30000000`, `max_train_iter=300000`, checkpoint every 10000.
- Larger `cz26-full-20260517a` fixed `collector_env_num=256`, `n_episode=256`, `batch_size=64`, `num_simulations=8`, `max_env_step=30000000`, `max_train_iter=300000`, checkpoint every 10000, refresh every 2000, random learner seat, and shared R18 rank-1 `iteration_180000` initialization.
- `cz26-full` varied recipe, reward alpha, noise, and leaderboard immortality across 136 submitted/spawned runs.
- Main observability caveat: opponent slot counts are collector-env or episode-reset assignment controls; they are not guaranteed learner mini-batch composition.

Details live in `EXPERIMENT_BATCH_INVENTORY.md`.

## F-023: Compact Experiment Surface

Source: main thread after Anscombe and Huygens critiques.

Plain finding:

- The giant flat launch payload is the real parameter-sprawl problem.
- `VisualSurvivalConfigSpec` is normalized internal config; it still needs many fields because LightZero/env construction needs a complete config.
- Added `VisualSurvivalExperimentSpec` as a compact experiment-facing target surface.
- `current_broad` expands to current CZ26 broad defaults: 256 collector envs, 256 episodes per collect, learner batch 64, 8 simulations, 30M env steps, 300k train iterations, checkpoint cadence 10000, source max steps 1048576, current policy observation surface.
- Added tests for compact expansion, current broad config build, unknown scale rejection, trainer facade parity, and policy-observation perspective in builder surface.

Validation:

- `uv run pytest -q -p no:cacheprovider tests/test_reward_contracts.py tests/test_lightzero_config_builder.py` -> 29 passed.

Remaining risk:

- Grouped submit accepts compact/minimal train kwargs, and `tonight18` is now compact-by-default. Non-migrated builders and the final Modal spawn still use flat kwargs. Migrate remaining builders only when omitted fields are known to equal deployed trainer defaults or are represented by explicit compact overrides.

Details live in `EXPERIMENT_KNOB_SURFACE.md`.

## F-024: Policy Observation Perspective Audit

Source: Arendt read-only audit.

Current truth:

- Training defaults to `learner_seat_mode=random_per_episode`.
- Tournament ratings use balanced random seat order.
- Policy observations are controlled-player-relative: seat N receives its own controlled-player view and controls player N.
- Standalone checkpoint eval uses normal env config and random learner seat by seed; it is not currently an explicitly balanced paired-seat eval protocol.

Recommended next pin:

- Add a focused standalone-eval contract test proving result rows expose learner seat mode, ego player index, and observation perspective player id, or introduce a balanced standalone eval mode deliberately.

## F-025: Parameter Surface Shrink

Source: main thread after Hooke and Goodall critiques.

Plain finding:

- The many parameters come from confusing two different layers: a compact experiment decision surface and the fully normalized LightZero/env config needed at runtime.
- `VisualSurvivalExperimentSpec` is now actually compact. It accepts only seed/paths, reward choice/alpha, opponent source, action-noise probability, and scale preset.
- It rejects internal launch knobs such as `collector_env_num`, `batch_size`, render modes, policy-observation backend, source cadence, `td_steps`, and model support caps.
- Grouped submit's required train kwargs are now only `mode`, `seed`, `run_id`, and `attempt_id`; the deployed trainer fills current defaults for the broad Modal signature.
- Missing policy render fields in grouped submit now validate against current defaults instead of requiring every row to carry them.
- The submitter normalizes minimal compact rows before side effects so the poller gets the correct run identity.
- Optional compact `experiment_spec` rows can expand current reward/noise/current-scale fields into flat trainer kwargs.
- Train-only/policy/config fields remain rejected from `poller_kwargs`.

Validation:

- `uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py` -> 25 passed.
- `uv run pytest -q -p no:cacheprovider tests/test_curvytron_shared_contracts.py tests/test_curvytron_survivaldiag_submitter.py tests/test_curvytron_tonight18_manifest.py::test_grouped_submit_accepts_compact_train_kwargs_with_current_defaults` -> 19 passed.

Remaining risk:

- Non-migrated manifest builders still emit large `train_kwargs` dictionaries. That is now legacy payload shape, not the required contract. Next cleanup should compact only fields that match current trainer defaults; survivaldiag/opponent-mixture rows have intentional non-defaults and must stay explicit until represented in compact overrides.

## F-026: Tonight18 Compact Manifest Rows

Source: main thread after Jason, Hypatia, and Euler critiques.

Plain finding:

- `build_curvytron_tonight18_manifest.py` can safely omit default-equal trainer kwargs for the current CPU40 trainer functions because those deployed wrappers fill visual-survival defaults before calling the real trainer.
- The migrated rows now carry `train_kwargs_schema_id=curvyzero_tonight18_compact_train_kwargs/v0`.
- The builder keeps explicit row semantics and non-default runtime fields: identity, reward choice/alpha, non-clean action noise, non-scratch initial checkpoint/load mode, exactly one opponent source, assignment refresh refs/intervals, own-checkpoint refresh, background eval/GIF non-defaults, and any CLI override that differs from the current default.
- The builder validator expands compact train kwargs from a local default table before checking assignment, slot-count, pressure, checkpoint, render, learner-seat, and refresh contracts.
- The submitter now rejects ambiguous compact rows: mixed `train_kwargs` plus `experiment_spec`, train identity overrides, action-noise bundle overrides when `action_noise_probability` owns the bundle, runtime/top-level ref conflicts, and train/poller identity divergence.

Validation:

- `python -m py_compile scripts/build_curvytron_tonight18_manifest.py scripts/submit_curvytron_survivaldiag_manifest.py` -> passed.
- `uv run pytest -q -p no:cacheprovider tests/test_curvytron_tonight18_manifest.py tests/test_curvytron_survivaldiag_submitter.py` -> 32 passed.
- `uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py tests/test_curvytron_shared_contracts.py tests/test_curvytron_survivaldiag_submitter.py tests/test_curvytron_tonight18_manifest.py tests/test_curvytron_survivaldiag_manifest.py tests/test_curvytron_opponent_mixture_manifest.py` -> 87 passed.
- `uv run pytest -q -p no:cacheprovider tests/test_curvytron_launch_manifest_ref_audit.py tests/test_curvytron_next_batch_manifest.py tests/test_feedback_loop_lineage.py tests/test_curvytron_training_candidate_controller_local.py` -> 30 passed.
- Wider no-Modal spine after this migration: config/reward compact spine -> 35 passed; trainer/eval spine -> 12 passed, 93 deselected; runtime reward/opponent slice -> 8 passed, 41 deselected; opponent split gate -> 63 passed.
- Real script dry-run: `build_curvytron_tonight18_manifest.py --scratch-bootstrap` wrote 18 compact rows; `submit_curvytron_survivaldiag_manifest.py` accepted that manifest in dry-run with 18 selected rows, 3 assignment records, and 3 refresh-pointer records.

Remaining risk:

- This does not migrate `survivaldiag`, `next_batch`, or `opponent_mixture` builders. They still need explicit payloads or named compact override fields because they carry more intentional non-defaults.
