# Architecture Boundary Audit - LightZero Train Refactor - 2026-05-13

This is a read-only, doc-only audit of the CurvyZero LightZero training launcher
surface. It is scoped to training code only: trainer launcher/scaffolding,
checkpoint discovery, resume/status, poller, eval/GIF triggering, manifests,
Modal wrappers, and tests. Environment mechanics, reward semantics, and
opponent behavior are explicitly out of scope.

The trusted lane is:

`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`

That lane should stay close to stock `lzero.entry.train_muzero`: build configs,
install the smallest necessary observability/resume hooks, call
`train_muzero([main_config, create_config], ...)`, then publish artifacts.

This audit was assembled from the main repository read plus three parallel
read-only subagent passes:

- Agent A: trainer anatomy, function grouping, LightZero hook boundaries.
- Agent B: checkpoint/resume/status/poller/eval/GIF discovery paths.
- Agent C: adjacent scripts/modules/tests, extraction risk, anti-goals.

## Executive Boundary

The first safe extraction is not the whole trainer. The first safe extraction is
plain checkpoint and resume artifact discovery:

- Parse `iteration_N.pth.tar` and `iteration_N.pkl`.
- Discover both fixed `lightzero_exp` and timestamped `lightzero_exp_*`
  LightZero experiment directories.
- Select latest/specific/all candidates deterministically.
- Return plain candidate records with refs, paths, iterations, mtimes, and source
  kind.

Keep Modal `.spawn`, `runs_volume.reload/commit`, LightZero monkeypatch hooks,
and environment config mechanics in place until those plain helpers are covered
by tests and consumed by the current callers.

The main current architectural hazard is that several training-adjacent readers
still assume:

`attempts/<attempt>/train/lightzero_exp/ckpt`

DI-engine/LightZero can also create timestamped sibling experiment directories:

`attempts/<attempt>/train/lightzero_exp_YYMMDD_HHMMSS/ckpt`

That means progress, resume, poller, and status can miss fresher checkpoints even
when training is producing valid artifacts.

## Current Responsibility Map

### 1. Modal App, Image, Volume, And Runtime Shell

Primary file:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

Responsibilities:

- Define Modal app/image/volume/runtime constants.
- Own remote function wrappers for CPU/GPU/H100 variants.
- Own local entrypoint argument surface.

Refs:

- `lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8741`
- `lightzero_curvytron_visual_survival_checkpoint_selfplay_gif`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8798`
- `lightzero_curvytron_visual_survival_checkpoint_eval_poller`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8854`
- `lightzero_curvytron_visual_survival_cpu`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8941`
- `lightzero_curvytron_visual_survival_cpu64`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9151`
- `lightzero_curvytron_visual_survival_gpu`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9164`
- `lightzero_curvytron_visual_survival_gpu_cpu40`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9305`
- `lightzero_curvytron_visual_survival_h100_cpu40`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9318`
- `lightzero_curvytron_visual_survival_h100x2_cpu40`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9331`
- `lightzero_curvytron_visual_survival_opponent_smoke`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9337`
- `main`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9495`

Boundary classification:

- Modal-bound. Do not extract first.
- These functions should become thinner only after plain helper modules exist.

### 2. Trusted Train Orchestrator

Primary function:

- `_run_visual_survival_train`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2974`

Responsibilities:

- Validate mode/compute/env/reward arguments.
- Build command/config metadata.
- Create run and attempt refs.
- Build LightZero configs.
- Prepare auto-resume.
- Patch `load_ckpt_before_run` when needed.
- Validate the config surface.
- Compile config summary.
- Install observability/resume/target-audit hooks.
- Call stock `lzero.entry.train_muzero`.
- Scan/mirror/publish artifacts.
- Write manifests, summaries, action traces, target audit, stdout/stderr, latest
  attempt, and heartbeat state.

Boundary classification:

- Modal-bound orchestration with a stock LightZero core call.
- Should remain the trusted lane.
- Extraction should remove helper clutter around it, not reimplement
  `train_muzero` semantics.

### 3. Config, Reward, And Env Surface Construction

Primary functions:

- `_normalize_opponent_policy_kind_for_env`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:484`
- `_normalize_reward_variant_for_env`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:499`
- `_validate_source_state_trail_render_mode`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:509`
- `_reward_policy_for_variant`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:519`
- `_reward_schema_id_for_variant`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:607`
- `_lightzero_target_config_for_reward`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:615`
- `_source_state_fixed_opponent_wrapper_env_spec_fields`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4125`
- `_env_variant_spec`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4168`
- `_build_visual_survival_configs`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4382`
- `_extract_surface`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4640`
- `_validate_visual_survival_surface`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4757`
- `_source_state_fixed_opponent_training_readiness_gate`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4906`
- `_compile_config_summary`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4998`
- `_target_config_patches`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5076`
- `_set_save_ckpt_after_iter`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5101`
- `_set_load_ckpt_before_run`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5117`

Responsibilities:

- Convert CLI flags into env and LightZero config fields.
- Encode the reward variant into config support shape.
- Preserve the training/eval surface contract used by the eval harness.
- Validate that the command's intended surface survived config compilation.

Boundary classification:

- Mostly pure or pure-ish, but tightly coupled to CurvyTron env definitions and
  LightZero config shape.
- Do not make this the first extraction. It is not the source of the timestamped
  checkpoint discovery risk.

### 4. Opponent Checkpoint And Mixture Resolution

Primary functions:

- `_resolve_opponent_checkpoint_for_env`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4020`
- `_resolve_opponent_mixture_for_env`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4068`
- `_reject_mutable_frozen_opponent_checkpoint_ref`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8198`

Adjacent module:

- `src/curvyzero/training/opponent_mixture.py`

Responsibilities:

- Resolve opponent checkpoint refs into mounted paths.
- Validate immutable checkpoint refs where the training lane needs them.
- Parse/normalize opponent mixture specs, then pass them into env config.

Boundary classification:

- Mixed. Parsing belongs to training-domain pure code; ref resolution is
  Modal/volume IO.
- Leave this alone in the first extraction unless checkpoint discovery helpers
  need shared immutable-ref labeling.

### 5. LightZero Hooks, Profiling, And Target Audit

Primary functions/classes:

- `_PhaseTimer`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:778`
- `_LightZeroPhaseProfiler`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:808`
- `_install_lightzero_phase_profile`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1003`
- `_install_live_checkpoint_publisher`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1742`
- `_install_checkpoint_progress_writer`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1868`
- `_install_lightzero_full_resume_state_hooks`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1921`
- `_install_lightzero_target_audit`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2860`

Responsibilities:

- Monkeypatch LightZero/DI-engine methods to publish checkpoints and progress.
- Capture profile timings.
- Save sidecar resume state.
- Restore collector/evaluator/replay/policy/RNG state.
- Audit target samples.

Boundary classification:

- LightZero hook-bound and brittle by nature.
- These hooks can call extracted pure helpers later, but should not be moved
  until checkpoint/resume behavior is covered by tests.

### 6. Checkpoint Parsing, Discovery, Mirroring, And Publishing

Primary functions:

- `_safe_int_or_none`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1797`
- `_latest_lightzero_iteration_checkpoint`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804`
- `_write_checkpoint_progress_latest`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1829`
- `_lightzero_iteration_checkpoint_name`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2460`
- `_lightzero_resume_state_name`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2464`
- `_lightzero_iteration_from_checkpoint_name`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5356`
- `_lightzero_iteration_from_resume_state_name`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5367`
- `_scan_lightzero_artifacts`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5432`
- `_mirror_lightzero_checkpoints`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5471`
- `_publish_live_lightzero_checkpoints`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5516`
- `_checkpoint_source_ref`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5738`
- `_checkpoint_ref_sort_key`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5784`
- `_checkpoint_label_from_ref`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5791`
- `_live_eval_checkpoint_name`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6078`

Responsibilities:

- Parse LightZero checkpoint and resume-state iteration names.
- Find current/latest checkpoint under the expected exp dir.
- Mirror current checkpoints into stable run-level checkpoint refs.
- Publish progress and live checkpoint index metadata.
- Convert volume paths back into refs for eval/GIF workers.

Boundary classification:

- Best first extraction target.
- Split into pure selection/parsing helpers plus thin IO wrappers.
- Mirroring/copying stays Modal/volume-bound.

Current risk:

- `_latest_lightzero_iteration_checkpoint` scans only `exp_name / "ckpt"`.
- `_scan_lightzero_artifacts` recursively scans only the single `exp_name` root
  it receives.
- Callers pass the fixed `lightzero_exp` root, so timestamped sibling roots can
  be invisible.

### 7. Resume Selection And Resume Sidecar State

Primary functions:

- `_save_lightzero_resume_sidecar_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2085`
- `_build_lightzero_resume_sidecar_payload`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2127`
- `_load_lightzero_resume_sidecar_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2168`
- `_lightzero_collector_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2202`
- `_restore_lightzero_collector_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2213`
- `_lightzero_evaluator_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2226`
- `_lightzero_replay_buffer_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2244`
- `_restore_lightzero_replay_buffer_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2271`
- `_lightzero_policy_extra_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2295`
- `_restore_lightzero_policy_extra_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2311`
- `_install_trusted_run_torch_load_retry`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2363`
- `_prepare_lightzero_auto_resume`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5133`
- `_find_lightzero_resume_sidecar`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5256`

Responsibilities:

- Choose a checkpoint for auto-resume.
- Choose a matching sidecar resume-state file.
- Save/restore extra LightZero runtime state not captured by the normal
  checkpoint.
- Patch torch load behavior for trusted run-local checkpoint paths.

Boundary classification:

- Selection is a good second extraction after checkpoint discovery.
- State capture/restore and torch-load retry are LightZero hook-bound; leave in
  the trainer until later.

Current risk:

- `_prepare_lightzero_auto_resume` checks current fixed `exp_name_ref / "ckpt"`,
  prior fixed `train/lightzero_exp/ckpt`, and stable mirrors. It does not inspect
  sibling `lightzero_exp_*` dirs.
- `_find_lightzero_resume_sidecar` checks current/prior fixed
  `lightzero_resume_state` dirs and stable state mirrors. It does not inspect
  sibling `lightzero_exp_*` dirs.

### 8. Background Eval And GIF Triggering

Primary functions:

- `_background_eval_config_from_command`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5548`
- `_background_gif_config_from_command`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5616`
- `_stable_seed_mix`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5683`
- `_mix_seed`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5689`
- `_spawn_checkpoint_eval_triggers`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5693`
- `_schedule_live_checkpoint_background_eval`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5746`
- `_copied_now_checkpoints`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5772`
- `_schedule_one_checkpoint_background_eval`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5811`
- `_spawn_one_checkpoint_background_eval`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5827`
- `_spawn_one_checkpoint_background_gif`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5943`
- `_checkpoint_eval_poller_status_path`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6082`
- `_write_checkpoint_eval_poller_status`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6089`
- `_checkpoint_eval_poller_train_done`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6100`
- `_checkpoint_eval_poller_command`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6117`
- `_run_checkpoint_eval_poller`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6193`

Responsibilities:

- Convert train command metadata into eval/GIF worker config.
- Spawn eval and GIF jobs for newly visible checkpoints.
- Poll for checkpoint stability and train completion.
- Write poller status for browsers/status endpoints.

Boundary classification:

- Config and scheduling decision payloads can become pure helpers.
- `.spawn`, `.remote`, status writes, sleeps, reloads, and volume commits remain
  Modal-bound.

Current risk:

- `lightzero_curvytron_visual_survival_checkpoint_eval_poller` defaults
  `exp_name_ref` to fixed `attempt_train_ref(...) / "lightzero_exp"` at
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8892`.
- `main` builds the same fixed `exp_name_ref` before spawning the poller at
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9927`.
- `_run_checkpoint_eval_poller` scans the single `exp_name_ref` it receives, so
  it inherits the fixed-root assumption.

### 9. Eval/Inspect And GIF Workers

Primary functions:

- `_run_checkpoint_eval_and_inspect`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6527`
- `_capture_checkpoint_selfplay_gif_variant`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7189`
- `_run_checkpoint_selfplay_gif`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7452`

Adjacent eval harness:

- `_checkpoint_summary`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:203`
- `_find_state_dict`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:225`
- `_parse_eval_seeds`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:337`
- `_checkpoint_label`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:381`
- `_checkpoint_ref_for_iteration`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:392`
- `_selected_checkpoint_refs`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:412`
- `_make_policy_and_env`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:686`
- `_run_survival_episode`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:845`
- `_eval_checkpoint`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:965`
- `_row_from_result`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:1036`
- `_survival_aggregate_table`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:1207`
- `_run_eval`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:1341`

Responsibilities:

- Inspect/checkpoint-load policy state.
- Run evaluation episodes.
- Render GIF/selfplay artifacts.
- Write eval manifests and summaries.

Boundary classification:

- Observability/scoring path, not training semantics.
- It imports trainer config helpers today, so trainer extraction should avoid
  breaking eval imports.

### 10. Run Manifests, Attempt State, Heartbeats, And Status Readers

Primary trainer writers:

- `_write_run_manifest_once`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7921`
- `_write_gif_browser_run_marker`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8025`
- `_write_attempt_state`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8045`
- `_write_latest_attempt`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8073`
- `_write_train_status_heartbeat`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8098`
- `_write_text`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8134`

Shared run-management helpers:

- `clean_id`: `src/curvyzero/infra/modal/run_management.py:57`
- `run_root_ref`: `src/curvyzero/infra/modal/run_management.py:73`
- `attempt_train_ref`: `src/curvyzero/infra/modal/run_management.py:101`
- `checkpoints_root_ref`: `src/curvyzero/infra/modal/run_management.py:116`
- `require_relative_ref`: `src/curvyzero/infra/modal/run_management.py:158`
- `volume_path`: `src/curvyzero/infra/modal/run_management.py:178`
- `write_json`: `src/curvyzero/infra/modal/run_management.py:264`
- `run_manifest`: `src/curvyzero/infra/modal/run_management.py:279`
- `attempt_manifest`: `src/curvyzero/infra/modal/run_management.py:310`
- `latest_attempt_pointer`: `src/curvyzero/infra/modal/run_management.py:340`
- `checkpoint_pointer`: `src/curvyzero/infra/modal/run_management.py:368`
- `best_checkpoint_pointer`: `src/curvyzero/infra/modal/run_management.py:396`

Status reader:

- `_checkpoint_summary`: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:809`
- `_run_status`: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:863`
- `_progress_curve`: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:946`
- `_eval_curve_status`: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:1007`

Responsibilities:

- Normalize run refs.
- Write run/attempt/checkpoint JSON state.
- Read progress, poller status, eval manifests, GIF summaries, and checkpoints
  into status output.

Boundary classification:

- `run_management.py` should stay boring: path/ref/JSON plumbing only.
- Training-specific checkpoint discovery should not be dumped into
  `run_management.py`.
- `lightzero_curvytron_run_status.py` should consume shared checkpoint discovery
  once it exists.

Current risk:

- `_checkpoint_summary` first picks stable mirror
  `checkpoints/lightzero`, then fixed attempt `train/lightzero_exp/ckpt` at
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:820`.
- It does not scan timestamped `lightzero_exp_*` attempt dirs.

### 11. Adjacent Manifests And Scripts

Opponent mixture manifest:

- `_poller_kwargs`: `scripts/build_curvytron_opponent_mixture_manifest.py:613`
- Fixed `lightzero_exp` path construction inside `_poller_kwargs`:
  `scripts/build_curvytron_opponent_mixture_manifest.py:621`

Stock train manifest:

- `Row`: `scripts/build_curvytron_stock_train_manifest.py:63`
- `_command_for_row`: `scripts/build_curvytron_stock_train_manifest.py:733`
- `_manifest_row`: `scripts/build_curvytron_stock_train_manifest.py:826`
- `build_manifest`: `scripts/build_curvytron_stock_train_manifest.py:928`

Older live eval queue:

- `_default_checkpoint_dir`: `scripts/lightzero_live_eval_queue.py:37`

Responsibilities:

- Build dry-run launch manifests.
- Encode expected refs/status outputs.
- Provide queueing helpers for older LightZero live eval flows.

Boundary classification:

- Adjacent, not the first extraction target.
- Update only after the central helper contract is established.
- `scripts/lightzero_live_eval_queue.py` appears Pong/older-lane specific; do
  not let it drive the CurvyTron train refactor.

### 12. Historical Two-Seat Lane

Primary functions:

- `_two_seat_checkpoint_ref`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8194`
- `_two_seat_background_eval_config`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8225`
- `_spawn_two_seat_checkpoint_poller`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8331`
- `_run_two_seat_selfplay_payload`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8402`
- `lightzero_curvytron_two_seat_selfplay_cpu`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8688`
- `lightzero_curvytron_two_seat_selfplay_gpu`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8704`
- `lightzero_curvytron_two_seat_selfplay_h100`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8720`

Responsibilities:

- Historical/experimental two-seat selfplay adapter.
- Owns a different training payload and checkpoint shape.

Boundary classification:

- Anti-goal for this lane.
- Do not use two-seat architecture as the model for trusted stock
  `train_muzero` refactor.

## Pure Vs Modal-Bound Boundaries

### Pure Candidates

These can be extracted first into a training-domain module such as
`src/curvyzero/training/lightzero_checkpoints.py` once tests are in place:

- `parse_iteration_checkpoint_name(name: str) -> int | None`
  - Source today: `_lightzero_iteration_from_checkpoint_name` at
    `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5356`.
- `parse_resume_state_name(name: str) -> int | None`
  - Source today: `_lightzero_iteration_from_resume_state_name` at
    `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5367`.
- `lightzero_exp_dirs(attempt_train_root: Path) -> list[Path]`
  - Should include fixed `lightzero_exp` and timestamped `lightzero_exp_*`.
  - Should use deterministic ordering and not assume mtimes alone.
- `checkpoint_candidates_from_exp_dirs(...) -> list[CheckpointCandidate]`
  - Should record iteration, path, relative ref when known, size, mtime, source
    kind, and exp dir name.
- `select_latest_checkpoint(candidates) -> CheckpointCandidate | None`
  - Should tie-break deterministically by iteration, mtime, and path/ref.
- `select_checkpoint_for_iteration(candidates, iteration: int)`.
- `select_all_checkpoints(candidates)`.
- Label/sort helpers derived from:
  - `_checkpoint_ref_sort_key`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5784`
  - `_checkpoint_label_from_ref`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5791`

These helpers should not import `modal`, trainer app/image objects, or CurvyTron
environment modules.

### Pure-ish But Not First

These are good later candidates, but only after the checkpoint/resume discovery
bug surface is covered:

- Eval/GIF command payload builders:
  - `_background_eval_config_from_command`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5548`
  - `_background_gif_config_from_command`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5616`
  - `_checkpoint_eval_poller_command`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6117`
- Progress/status payload builders:
  - `_write_checkpoint_progress_latest`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1829`
  - `_write_train_status_heartbeat`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8098`
  - `_checkpoint_summary`: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:809`
- Config surface helpers:
  - `_build_visual_survival_configs`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4382`
  - `_extract_surface`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4640`
  - `_validate_visual_survival_surface`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4757`

### Modal-Bound / IO-Bound

Keep these near Modal wrappers and orchestration:

- `modal.App`, `modal.Image`, `modal.Volume`, and function decorators.
- `.spawn`, `.remote`, `.get`, object id extraction.
- `runs_volume.reload()` and `runs_volume.commit()`.
- Mounted path resolution through `runs.volume_path(...)` when operating on the
  live Modal volume.
- File copying/mirroring through `shutil.copy2(...)`.
- Status JSON writes through `runs.write_json(...)`.
- Poller sleeps, timeout loops, and train-done detection.

Refs:

- `_mirror_lightzero_checkpoints`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5471`
- `_publish_live_lightzero_checkpoints`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5516`
- `_spawn_one_checkpoint_background_eval`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5827`
- `_spawn_one_checkpoint_background_gif`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5943`
- `_run_checkpoint_eval_poller`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6193`
- `_commit_runs_volume_with_backoff`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6487`

### LightZero Hook-Bound

Keep these in the trainer until the helper modules are stable:

- Monkeypatches around `BaseLearner`, checkpoint save hooks, collector/evaluator
  state, replay buffer state, policy wrapper state, RNG state, and target audit.
- Import-time calls into `lzero`/`ding`.
- Trusted-run torch-load retry behavior.

Refs:

- `_install_lightzero_phase_profile`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1003`
- `_install_live_checkpoint_publisher`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1742`
- `_install_checkpoint_progress_writer`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1868`
- `_install_lightzero_full_resume_state_hooks`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1921`
- `_install_trusted_run_torch_load_retry`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2363`
- `_install_lightzero_target_audit`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2860`

## Smallest Extraction Cuts After Tests

### Cut 0 - Tests First, No Behavior Move

Before moving code, add tests that reproduce the fixed-root blind spot:

- Progress writer sees a fresher checkpoint under
  `train/lightzero_exp_YYMMDD_HHMMSS/ckpt`.
- Auto-resume chooses that fresher checkpoint when the fixed root has an older
  checkpoint.
- Resume sidecar lookup can find the matching timestamped exp dir sidecar.
- Poller discovers and schedules stable checkpoints in timestamped exp dirs.
- Run status reports timestamped attempt checkpoints when stable mirror is absent
  or stale.
- Existing fixed-root behavior still passes.

Useful existing test anchors:

- `test_checkpoint_progress_writer_updates_browser_speed_file`: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:61`
- `test_save_ckpt_hook_updates_browser_speed_file`: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:105`
- `test_live_checkpoint_trigger_spawns_eval_and_selfplay_gif_without_volume_commit`: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:1898`
- `test_save_hook_trigger_uses_source_checkpoint_ref_not_future_mirror`: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:2085`
- `test_hook_trigger_retries_after_spawn_failure_and_skips_mutable_best_checkpoint`: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:2160`
- `test_checkpoint_eval_poller_completes_eval_inspection_and_selfplay_gif_jobs`: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:2281`
- `test_local_launcher_passes_gif_config_to_poller_and_prints_enabled`: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:2454`
- `test_status_rolls_up_missing_reason_train_actions_poller_and_gifs`: `tests/test_curvytron_run_status.py:205`
- `test_checkpoint_summary_includes_checkpoint_mtimes`: `tests/test_curvytron_run_status.py:305`

Existing timestamped-discovery test patterns are already present in the
tournament lane:

- `test_checkpoint_discovery_finds_latest_real_lightzero_weight`: `tests/test_curvytron_checkpoint_tournament.py:3115`
- `test_checkpoint_discovery_scans_timestamped_lightzero_exp_dirs`: `tests/test_curvytron_checkpoint_tournament.py:3162`
- `test_checkpoint_discovery_iteration_filter_scans_timestamped_dirs`: `tests/test_curvytron_checkpoint_tournament.py:3200`
- `test_checkpoint_discovery_all_returns_all_nonempty_weight_files`: `tests/test_curvytron_checkpoint_tournament.py:3237`

Use those as test shape inspiration, not as a reason to couple training to the
tournament module.

### Cut 1 - Minimal Bugfix In Place

After tests fail for the timestamped-root bug, make the smallest in-place change:

- Keep the trainer file intact.
- Teach the existing checkpoint scan call sites to consider fixed
  `lightzero_exp` plus sibling `lightzero_exp_*` roots.
- Preserve existing stable mirror behavior and immutable checkpoint refs.
- Preserve the poller command surface and Modal wrappers.

Affected first-pass call sites:

- `_latest_lightzero_iteration_checkpoint`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804`
- `_prepare_lightzero_auto_resume`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5133`
- `_find_lightzero_resume_sidecar`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5256`
- `_scan_lightzero_artifacts`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5432`
- `_run_checkpoint_eval_poller`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6193`
- `_checkpoint_summary`: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:809`

This keeps risk low: tests prove behavior, then extraction can be mechanical.

### Cut 2 - Extract Pure Checkpoint Discovery

Create `src/curvyzero/training/lightzero_checkpoints.py` after Cut 1 is green.

Initial contents:

- Iteration name parsers.
- Exp-dir discovery for fixed and timestamped LightZero roots.
- Candidate dataclass or plain typed dict.
- Latest/specific/all selectors.
- Stable sort keys and labels.

Update callers to use the module:

- `_latest_lightzero_iteration_checkpoint`
- `_scan_lightzero_artifacts`
- `_run_checkpoint_eval_poller`
- `lightzero_curvytron_run_status._checkpoint_summary`

Keep file copying and JSON writes in the existing Modal modules.

### Cut 3 - Extract Resume Selection, Not Resume Mechanics

Create or extend a training-domain helper for resume selection only:

- Select auto-resume checkpoint candidates.
- Select matching sidecar candidate by iteration.
- Prefer explicit `checkpoint_ref` over auto-discovery exactly as today.
- Preserve stable mirror fallback.

Do not move yet:

- `_save_lightzero_resume_sidecar_state`
- `_load_lightzero_resume_sidecar_state`
- `_restore_lightzero_*`
- `_install_lightzero_full_resume_state_hooks`
- `_install_trusted_run_torch_load_retry`

Those are LightZero hook-bound and should remain close to the trusted lane.

### Cut 4 - Extract Progress And Status Payload Builders

Once checkpoint discovery is shared, extract pure payload assembly:

- Progress latest payload shape from `_write_checkpoint_progress_latest`.
- Heartbeat payload shape from `_write_train_status_heartbeat`.
- Checkpoint status summary shape from
  `lightzero_curvytron_run_status._checkpoint_summary`.

Keep actual writes/readers local:

- `runs.write_json`
- `runs.file_summary`
- mounted path resolution
- status endpoint/Modal wrapper code

### Cut 5 - Extract Poller Scheduling Decisions

Extract the deterministic parts of poller behavior:

- Stable checkpoint candidate state.
- Deduplication of scheduled checkpoint refs.
- Eval/GIF job request payload shape.
- Seed derivation helpers.

Keep Modal behavior in the trainer:

- `runs_volume.reload()`
- `time.sleep`
- timeout loop
- `.spawn`
- status writes
- train-done detection

This makes `_run_checkpoint_eval_poller` shorter without changing how jobs are
launched.

### Cut 6 - Extract Background Eval/GIF Config Builders

Move only plain builders after tests cover the option matrix:

- `_background_eval_config_from_command`
- `_background_gif_config_from_command`
- `_checkpoint_eval_poller_command`

Consumers:

- trainer poller spawn
- eval/GIF worker wrappers
- manifest builders

This helps scripts and Modal wrappers share one payload contract.

### Cut 7 - Extract Config Surface Helpers Last

Only after the checkpoint/resume/poller work is stable, consider moving config
surface helpers into a training-domain module.

Candidate later module:

- `src/curvyzero/training/lightzero_visual_survival_config.py`

Do this last because `_build_visual_survival_configs` is shared with eval and is
closer to CurvyTron env mechanics than the checkpoint bug surface is.

## Anti-Goals

- Do not redesign environment mechanics, observation rendering, reward
  semantics, source-state wrappers, opponent runtime behavior, or two-seat
  mechanics.
- Do not reimplement or wrap stock `lzero.entry.train_muzero` beyond the current
  launcher/hook responsibilities.
- Do not move collector, replay buffer, learner, MCTS, target-network, or search
  logic into CurvyZero training helpers.
- Do not perform a broad split of the 10k-line trainer before tests isolate the
  checkpoint/resume/poller behavior.
- Do not move LightZero monkeypatch hooks first. Let them call extracted pure
  helpers later.
- Do not make `src/curvyzero/infra/modal/run_management.py` responsible for
  training semantics. It should stay ref/path/JSON plumbing.
- Do not couple trusted training discovery to the tournament module. Reuse the
  test idea, not the module boundary.
- Do not let the historical two-seat lane drive the stock `train_muzero` lane.
- Do not change manifest refs from immutable `iteration_N.pth.tar` to mutable
  latest/best refs.
- Do not launch Modal runs or background jobs as part of the refactor tests.
- Do not optimize GIF rendering or eval episode mechanics in this architecture
  pass.
- Do not remove existing fixed `lightzero_exp` support. Add timestamped sibling
  discovery alongside it.

## Exact Bug-Surface Refs To Watch

These are the refs most likely to need tests and then small code changes:

- Progress writer latest checkpoint:
  - `_latest_lightzero_iteration_checkpoint`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804`
  - `_write_checkpoint_progress_latest`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1829`
- Auto-resume:
  - `_prepare_lightzero_auto_resume`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5133`
  - `_find_lightzero_resume_sidecar`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5256`
- Artifact scan/mirror/publish:
  - `_scan_lightzero_artifacts`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5432`
  - `_mirror_lightzero_checkpoints`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5471`
  - `_publish_live_lightzero_checkpoints`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5516`
- Poller:
  - `_run_checkpoint_eval_poller`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6193`
  - `lightzero_curvytron_visual_survival_checkpoint_eval_poller`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8854`
  - default fixed `exp_name_ref`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8892`
  - local entrypoint fixed `exp_name_ref`: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9927`
- Status:
  - `_checkpoint_summary`: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:809`
  - fixed attempt checkpoint fallback: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:820`
- Manifest/script surfaces:
  - `_poller_kwargs`: `scripts/build_curvytron_opponent_mixture_manifest.py:613`
  - fixed `lightzero_exp` poller ref construction:
    `scripts/build_curvytron_opponent_mixture_manifest.py:621`
  - `_default_checkpoint_dir`: `scripts/lightzero_live_eval_queue.py:37`

## Suggested End State

The end state should be a boring dependency graph:

`curvyzero.training.lightzero_checkpoints`

- Pure parsing/discovery/selection.
- No Modal imports.
- No CurvyTron env imports.
- Unit tested directly.

`curvyzero.training.lightzero_resume`

- Pure resume candidate matching only.
- No hook/state mutation.
- May reuse checkpoint candidates.

`curvyzero.training.lightzero_progress`

- Pure progress/status payload builders.
- No file writes.

`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

- Modal app/image/volume.
- Trusted train launcher around stock `train_muzero`.
- LightZero hooks.
- Mounted path resolution, copying, JSON writes, `.spawn`, volume reload/commit.

`src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`

- Status endpoint/read model.
- Consumes shared checkpoint discovery.
- Does not own training discovery semantics.

This keeps the trusted training lane close to stock LightZero while making the
artifact discovery behavior testable, reusable, and less fragile.
