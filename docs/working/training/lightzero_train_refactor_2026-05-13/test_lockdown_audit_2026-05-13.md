# Test Lockdown Audit - 2026-05-13

## Scope

This audit covers Coach/training refactor tests only: trainer launcher/scaffolding, checkpoint discovery, resume/status, poller, eval/GIF triggering, manifests, Modal wrappers, and the tests that pin those contracts.

The trusted training lane is `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`, which should remain close to stock `lzero.entry.train_muzero`. The old `--mode two-seat-selfplay` lane is treated as historical/untrusted for learning claims. Environment code is treated as an interface contract, not as a redesign target.

## Commands And Results

- Read the required planning docs:
  - `docs/working/training/lightzero_train_refactor_2026-05-13/README.md`
  - `docs/working/training/lightzero_train_refactor_2026-05-13/current_source_of_truth.md`
  - `docs/working/training/lightzero_train_refactor_2026-05-13/regression_test_lockdown_plan.md`
  - `docs/working/training/lightzero_train_refactor_2026-05-13/test_inventory.md`
  - `docs/working/training/lightzero_train_refactor_2026-05-13/bug_registry.md`
- Ran parallel read-only subagent audits with `codex exec --ephemeral --sandbox read-only -C /Users/shankha/curvy`. All subagents exited. One subagent nevertheless attempted test/doc edits; the accidental `tests/test_curvytron_live_checkpoint_eval_plumbing.py` diff was restored before this audit was written.
- `git status --short` showed a pre-existing dirty worktree with many unrelated modified/untracked files. `tests/test_curvytron_live_checkpoint_eval_plumbing.py` is clean after restoration.
- `test -e docs/working/training/lightzero_train_refactor_2026-05-13/test_lockdown_audit_2026-05-13.md` returned exit `1` before this file was created.
- Used `rg -n` to locate exact function and test anchors. No pytest was run; this is a read-only/doc-only audit.

## Facts

### Known Checkpoint Bug

DI-engine can write checkpoints under timestamped experiment directories such as:

```text
train/lightzero_exp_YYMMDD_HHMMSS/ckpt/iteration_180000.pth.tar
```

Several CurvyZero training helpers currently assume the fixed directory:

```text
train/lightzero_exp/ckpt
```

The regression fake tree from the plan is:

```text
train/lightzero_exp/ckpt/iteration_0.pth.tar
train/lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar
```

The expected broad-discovery result is `iteration_180000.pth.tar` from `lightzero_exp_260513_123802`.

### Current Source Anchors

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804` defines `_latest_lightzero_iteration_checkpoint(exp_name)`. It scans only `exp_name / "ckpt"` for `iteration_*.pth.tar`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1829` defines `_write_checkpoint_progress_latest(...)`. It calls `_latest_lightzero_iteration_checkpoint(exp_name)`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2974` defines `_run_visual_survival_train(...)`. It sets `exp_name_ref` to the fixed `attempts/<attempt>/train/lightzero_exp` path before configuring training, hooks, resume, final scan, and poller setup.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4382` defines `_build_visual_survival_configs(...)`, and it writes `main_config.exp_name` from the supplied `exp_name`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5133` defines `_prepare_lightzero_auto_resume(...)`. It checks the stable mirror and the fixed `attempt_dir / "train" / "lightzero_exp" / "ckpt"` path, but not timestamped `lightzero_exp_*` siblings.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5256` defines `_find_lightzero_resume_sidecar(...)`. It checks the fixed `lightzero_exp/lightzero_resume_state` path and prior fixed attempt paths, but not timestamped `lightzero_exp_*` state dirs.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5432` defines `_scan_lightzero_artifacts(exp_name)`. It scans recursively under only the supplied `exp_name`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5516` defines `_publish_live_lightzero_checkpoints(...)`, which depends on `_scan_lightzero_artifacts`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5693` defines `_spawn_checkpoint_eval_triggers(...)`, which depends on `_scan_lightzero_artifacts` and `_live_eval_checkpoint_name`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6193` defines `_run_checkpoint_eval_poller(...)`, which resolves `exp_name_ref` and repeatedly calls `_scan_lightzero_artifacts(str(exp_name))`.
- `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:809` defines `_checkpoint_summary(...)`. It checks the stable mirror first, then fixed `attempts/<attempt>/train/lightzero_exp/ckpt`, but not timestamped siblings.
- `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:863` defines `_run_status(...)`, which consumes `_checkpoint_summary`.
- Tournament discovery already has a broad glob contract: `src/curvyzero/tournament/curvytron/contracts.py:28` uses `train/lightzero_exp*/ckpt/iteration_*.pth.tar`; `:29` uses `lightzero_exp*/ckpt`.
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:254` defines `_checkpoint_candidate_rows_for_run(...)`, which scans `lightzero_exp*/ckpt`.
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:295` defines `_discover_checkpoint_refs(...)`.
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:669` defines `_intake_manifest_from_discovery(...)`.
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:6159` and `:6290` define checkpoint intake seed/tick Modal wrappers.
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:59` defines `normalize_checkpoint_spec(...)`; explicit refs are normalized as supplied and are not discovery.

### Existing Tests Observed

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:46` pins progress writer plumbing for fixed `lightzero_exp`.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:90` pins SaveCkptHook progress writer plumbing for fixed `lightzero_exp`.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:149` pins eval row reward component preservation.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:303` pins the default training env variant contract.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:848` pins background eval/GIF enablement.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:1883`, `:2070`, `:2145`, `:2232`, and `:2266` pin live checkpoint eval/GIF trigger behavior, source refs, retry behavior, mutable checkpoint skipping, joint-action guards, and poller completion.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:2439` pins launcher-to-poller GIF config plumbing and the printed command payload.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py:966` and `:2511` are examples of old two-seat selfplay launcher/GIF tests.
- `tests/test_curvytron_checkpoint_tournament.py:3115`, `:3162`, `:3200`, `:3237`, and `:3308` already cover broad timestamped checkpoint discovery for tournament discovery.
- `tests/test_curvytron_checkpoint_tournament.py:1720` and `:1751` cover intake manifest queue partitioning and explicit checkpoint refs.
- `tests/test_curvytron_run_status.py:205`, `:280`, and `:305` cover status rollups, partial progress handling, and checkpoint mtimes for the fixed checkpoint path.
- `tests/test_opponent_mixture.py:19`, `:128`, `:147`, `:203`, `:238`, and `:270` cover opponent mixture parsing, immutable checkpoint ref validation, background eval/GIF propagation, and selected mixture metadata.
- `tests/test_multiplayer_source_state_trainer_surface.py:277`, `:300`, `:339`, `:394`, `:523`, `:532`, and `:552` cover environment/trainer interface contracts.

## Recommendations

### Existing Tests Worth Keeping

- Keep the fixed-path progress and SaveCkptHook tests at `tests/test_curvytron_live_checkpoint_eval_plumbing.py:46` and `:90`. They remain useful for the hook contract, but they do not cover DI-engine timestamped directories.
- Keep the training-lane config and launcher tests around `tests/test_curvytron_live_checkpoint_eval_plumbing.py:303`, `:848`, and `:2439`. They pin that `--mode train` remains the trusted path and that launcher payloads reach the poller.
- Keep the live eval/GIF trigger tests at `tests/test_curvytron_live_checkpoint_eval_plumbing.py:1883`, `:2070`, `:2145`, `:2232`, and `:2266`. They are the best existing coverage for checkpoint-driven background work.
- Keep the tournament discovery tests at `tests/test_curvytron_checkpoint_tournament.py:3115`, `:3162`, `:3200`, `:3237`, and `:3308`. They show the desired broad-discovery behavior already exists in the tournament lane.
- Keep `tests/test_curvytron_run_status.py:205`, `:280`, and `:305`, then add timestamped sibling coverage beside them.
- Keep `tests/test_opponent_mixture.py` coverage for immutable refs and mixture metadata. It is relevant to manifests and eval/GIF metadata, but it is not a checkpoint discovery test.
- Keep only the environment surface tests that define the trainer interface contract, such as reset/step observation shape, action mask, terminal final observation, and explicit approximate render labels.

### Stale Or Likely Low-Relevance Tests After Refactor

- Treat old `--mode two-seat-selfplay` tests as historical compatibility only. Examples: `tests/test_curvytron_live_checkpoint_eval_plumbing.py:966`, `:2511`, and nearby two-seat GIF/launcher tests. They should not be used to prove learning, checkpoint discovery, resume, or trusted training correctness.
- Avoid expanding tests that inspect two-seat collect knobs, two-seat browser markers, or two-seat render modes unless the refactor explicitly preserves the historical lane.
- Avoid using environment fidelity breadth tests as evidence for Coach correctness. They can stay as interface guards, but the training refactor should not branch into environment mechanics.

### Missing Regression Tests

#### `test_latest_lightzero_checkpoint_scans_timestamped_exp_dirs`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup:
  - `run_id = "run-timestamped"`
  - `attempt_id = "attempt-a"`
  - `train_root = train_mod.runs.volume_path(tmp_path, train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id))`
  - Write nonempty `train_root / "lightzero_exp" / "ckpt" / "iteration_0.pth.tar"`.
  - Write nonempty `train_root / "lightzero_exp_260513_123802" / "ckpt" / "iteration_180000.pth.tar"`.
  - Also write ignored files: `ckpt_best.pth.tar`, `iteration_notnum.pth.tar`, and optionally a zero-byte higher iteration if the chosen helper filters empty files.
- Pins: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804` `_latest_lightzero_iteration_checkpoint(...)`, or a new central helper if extracted.
- Expected: the selected checkpoint is `iteration_180000.pth.tar` from `lightzero_exp_260513_123802`, not the fixed-dir `iteration_0.pth.tar`.

#### `test_progress_latest_uses_broad_lightzero_checkpoint_discovery`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup: same fake tree as above.
- Call `_write_checkpoint_progress_latest(...)` with `attempt_train_root=train_root`, `exp_name=train_root / "lightzero_exp"`, and a fake learner with `train_iter = 180000`.
- Pins: `_write_checkpoint_progress_latest(...)` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1829` and `_latest_lightzero_iteration_checkpoint(...)` at `:1804`.
- Expected: `progress_latest.json` reports `iteration == 180000`, `checkpoint_name == "iteration_180000.pth.tar"`, and a `checkpoint_ref` containing `lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar`.

#### `test_auto_resume_selects_latest_checkpoint_from_timestamped_exp_dir`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup:
  - Monkeypatch `train_mod.RUNS_MOUNT = tmp_path`.
  - Monkeypatch `train_mod.runs_volume` to a fake object with `reload()`.
  - Build the same stale fixed-dir and fresh timestamped checkpoint tree under the attempt train root.
  - Use `exp_name_ref = train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id) / "lightzero_exp"`.
- Pins: `_prepare_lightzero_auto_resume(...)` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5133`.
- Expected: `found is True`, `iteration == 180000`, and `checkpoint_ref` contains the timestamped exp dir.

#### `test_resume_sidecar_matches_timestamped_checkpoint_iteration`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup:
  - Same `RUNS_MOUNT`, `run_id`, `attempt_id`, and `exp_name_ref` pattern.
  - Write `train_root / "lightzero_exp" / "lightzero_resume_state" / "iteration_0.resume_state.pkl"`.
  - Write `train_root / "lightzero_exp_260513_123802" / "lightzero_resume_state" / "iteration_180000.resume_state.pkl"`.
- Pins: `_find_lightzero_resume_sidecar(...)` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5256`.
- Expected: the sidecar for `iteration=180000` is found under `lightzero_exp_260513_123802`, with `resume_state_found is True`.

#### `test_scan_lightzero_artifacts_includes_lightzero_exp_siblings`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup: same stale/fresh checkpoint tree under one attempt train root, plus a lightweight non-checkpoint artifact under the timestamped exp dir if final summary needs artifact coverage.
- Call `_scan_lightzero_artifacts(str(train_root / "lightzero_exp"))`.
- Pins: `_scan_lightzero_artifacts(...)` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5432`.
- Expected: `checkpoint_files` includes both fixed and timestamped checkpoint files and preserves enough path data for downstream `_checkpoint_source_ref(...)`.

#### `test_publish_live_lightzero_checkpoints_mirrors_timestamped_latest`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup: same fake tree, with `RUNS_MOUNT = tmp_path` and a fake or local mirror destination.
- Call `_publish_live_lightzero_checkpoints(...)` with `exp_name=train_root / "lightzero_exp"` and `attempt_train_root=train_root`.
- Pins: `_publish_live_lightzero_checkpoints(...)` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5516`, `_scan_lightzero_artifacts(...)` at `:5432`, and `_mirror_lightzero_checkpoints(...)` at `:5471`.
- Expected: the stable mirror receives the timestamped `iteration_180000.pth.tar` and does not mistake the stale fixed-dir checkpoint as latest.

#### `test_checkpoint_eval_hook_trigger_scans_timestamped_lightzero_exp_dirs`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup: adapt the source-ref assertions from `test_save_hook_trigger_uses_source_checkpoint_ref_not_future_mirror` at `tests/test_curvytron_live_checkpoint_eval_plumbing.py:2070`, but place the fresh checkpoint under `lightzero_exp_260513_123802`.
- Pins: `_spawn_checkpoint_eval_triggers(...)` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5693`, `_scan_lightzero_artifacts(...)` at `:5432`, and `_checkpoint_source_ref(...)` at `:5738`.
- Expected: spawned eval/GIF payloads use a checkpoint ref from the timestamped source path, not a future stable mirror and not the fixed stale checkpoint.

#### `test_checkpoint_eval_poller_discovers_timestamped_exp_dir_checkpoints`

- Target file: `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Fake setup:
  - Monkeypatch `train_mod.RUNS_MOUNT = tmp_path`.
  - Fake `train_mod.runs_volume.reload`.
  - Same stale/fresh fake tree.
  - Write `train_root / "summary.json"` so the poller can stop cleanly.
  - Stub `lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect.spawn` and, if enabled, GIF spawn functions.
  - Build the poller command with `_checkpoint_eval_poller_command(...)` and `background_eval_enabled=True`.
  - Call `_run_checkpoint_eval_poller(..., exp_name_ref="<attempt>/train/lightzero_exp", stable_polls=0, max_runtime_sec=1.0, idle_after_train_done_sec=0.0)`.
- Pins: `_run_checkpoint_eval_poller(...)` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6193`, `_scan_lightzero_artifacts(...)` at `:5432`, `_spawn_one_checkpoint_background_eval(...)` at `:5827`, and `_spawn_one_checkpoint_background_gif(...)` at `:5943` if GIF is enabled.
- Expected: scheduled checkpoint refs include `lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar`.

#### `test_run_status_checkpoint_summary_scans_lightzero_exp_siblings`

- Target file: `tests/test_curvytron_run_status.py`.
- Fake setup:
  - Monkeypatch `status_mod.RUNS_MOUNT = tmp_path`.
  - Build attempt train root with `status_mod.runs.volume_path(tmp_path, status_mod.runs.attempt_train_ref(status_mod.TASK_ID, run_id, attempt_id))`.
  - Write fixed stale `iteration_0.pth.tar` and timestamped fresh `iteration_180000.pth.tar`.
- Pins: `_checkpoint_summary(...)` at `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:809` and, through one higher-level assertion if cheap, `_run_status(...)` at `:863`.
- Expected: `latest_checkpoint == "iteration_180000"` and the count/list includes both checkpoints.

#### `test_intake_manifest_checkpoint_selection_uses_broad_discovery`

- Target file: `tests/test_curvytron_checkpoint_tournament.py`.
- Fake setup:
  - Use the same run/attempt fake tree, but through the tournament discovery helpers.
  - Call `_discover_checkpoint_refs(...)` or the wrapper that existing tests use for `tests/test_curvytron_checkpoint_tournament.py:3162`.
  - Feed that discovery into `_intake_manifest_from_discovery(...)`.
- Pins: `_discover_checkpoint_refs(...)` at `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:295` and `_intake_manifest_from_discovery(...)` at `:669`.
- Expected: the manifest queue contains the timestamped checkpoint ref selected by broad discovery, not a hand-written fixed-path ref.

#### `test_checkpoint_intake_tick_enqueues_new_timestamped_checkpoint_refs`

- Target file: `tests/test_curvytron_checkpoint_tournament.py`, if the Modal wrapper can be exercised with local fakes.
- Fake setup:
  - Seed intake state with the fixed-dir checkpoint already seen.
  - Add `lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar`.
  - Run the intake tick path with local fake state storage.
- Pins: `curvytron_checkpoint_intake_tick(...)` at `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:6290` plus `_discover_checkpoint_refs_from_scan_spec(...)` at `:498`.
- Expected: the tick enqueues the timestamped checkpoint as new work and leaves already-seen fixed refs untouched.

## Minimal First Sequence For The Checkpoint-Discovery Bug

1. Add a shared test helper in `tests/test_curvytron_live_checkpoint_eval_plumbing.py` that creates:

   ```text
   train/lightzero_exp/ckpt/iteration_0.pth.tar
   train/lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar
   ```

   Use nonempty checkpoint files and derive `train_root` from `runs.attempt_train_ref(...)` plus `runs.volume_path(...)`.

2. Implement `test_progress_latest_uses_broad_lightzero_checkpoint_discovery` first. It is small, fails directly against `_latest_lightzero_iteration_checkpoint(...)`, and proves the write path stops reporting stale fixed-dir progress.

3. Implement `test_auto_resume_selects_latest_checkpoint_from_timestamped_exp_dir` second. This pins the most dangerous learning-continuity behavior before refactoring resume code.

4. Implement `test_checkpoint_eval_poller_discovers_timestamped_exp_dir_checkpoints` third. This pins live background eval/GIF scheduling against real DI-engine directory shape.

5. Implement `test_run_status_checkpoint_summary_scans_lightzero_exp_siblings` fourth. This keeps operator status aligned with the same checkpoint discovery contract.

6. After those fail for the known reason, add the broader publish/sidecar/manifest tests. They are important, but the first four give the smallest end-to-end lockdown for the bug.
