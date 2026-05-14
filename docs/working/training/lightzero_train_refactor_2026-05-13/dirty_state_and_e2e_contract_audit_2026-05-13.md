# Dirty State And E2E Contract Audit

Date: 2026-05-13

Scope: read-only/doc-only audit for the stock LightZero training refactor. No
source or test files were edited. Three read-only Codex subagents were launched
in parallel for dirty-state classification, source/test contract mapping, and
first-test recommendations; this doc is the synthesis.

## Commands Run

Required read-first docs:

```bash
sed -n '1,220p' docs/working/training/lightzero_train_refactor_2026-05-13/README.md
sed -n '1,260p' docs/working/training/lightzero_train_refactor_2026-05-13/current_source_of_truth.md
sed -n '1,260p' docs/working/training/lightzero_train_refactor_2026-05-13/source_file_inventory.md
sed -n '1,260p' docs/working/training/lightzero_train_refactor_2026-05-13/regression_test_lockdown_plan.md
```

Dirty-state and file facts:

```bash
git status --short
git status --porcelain=v1 -uall
git status --porcelain=v1 -uall -- docs/working/training/lightzero_train_refactor_2026-05-13 tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py src/curvyzero/infra/modal/lightzero_curvytron_run_status.py scripts/build_curvytron_opponent_mixture_manifest.py tests/test_opponent_mixture.py
git diff --name-only
git ls-files --stage -- docs/working/training/lightzero_train_refactor_2026-05-13
```

Source/test trace commands:

```bash
rg -n "^def .*checkpoint|^def .*poller|^def .*resume|^def .*gif|^def .*eval|^def _prepare_lightzero_auto_resume|^def _checkpoint_source_ref|^def _spawn_checkpoint_eval_triggers|^def _poll_checkpoint|^def _wait_for_visible_checkpoint|^def _run_checkpoint_selfplay_gif|^def lightzero_curvytron_visual_survival_checkpoint_eval_poller" src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
rg -n "curvytron_run_status|_checkpoint_summary|_run_status|lightzero_curvytron_run_status|status_heartbeat|progress_latest" tests src/curvyzero/infra/modal/lightzero_curvytron_run_status.py
rg -n "CHECKPOINT_EXP_CKPT_DIR_GLOB|CHECKPOINT_SCAN_GLOB|CHECKPOINT_WEIGHT_FILENAME_GLOB|lightzero_exp" src/curvyzero/tournament src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py
```

Parallel read-only subagent pattern:

```bash
codex exec --ephemeral --sandbox read-only --cd /Users/shankha/curvy "<read the four required docs; inspect one audit lane; report only; no edits>"
```

## Dirty-State Classification

Relevant to this refactor:

- `?? docs/working/training/lightzero_train_refactor_2026-05-13/`
  is the untracked working-doc directory for this lane. The four required docs
  and neighboring planning docs are in this directory.
- No tracked primary LightZero trainer source/test file was dirty in the scoped
  status query.

Safe files for first tests and first patch, after this doc-only audit:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`
- `tests/test_curvytron_run_status.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`

Clean but lower-priority relevant files:

- `scripts/build_curvytron_opponent_mixture_manifest.py`
- `tests/test_opponent_mixture.py`

Adjacent evidence, avoid touching in the first training patch because these are
dirty or separate lanes:

- `docs/working/training/checkpoint_tournament_*`
- `docs/working/training/curvytron_architecture_research_2026-05-12/*`
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
- `src/curvyzero/tournament/curvytron/`
- `tests/test_curvytron_checkpoint_tournament.py`

Unrelated or out-of-scope lanes, avoid:

- `docs/working/environment/*`
- `docs/working/optimizer/*`
- `scripts/profile_curvytron_render_trajectory_lengths.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_replay.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`
- `tests/test_curvytron_two_seat_render_mode.py`
- `tests/test_multiplayer_presence_leave_fidelity.py`
- `tests/test_multiplayer_bonus_breadth_fidelity.py`
- `tests/test_multiplayer_collision_breadth_fidelity.py`
- `tests/test_multiplayer_source_state_trainer_surface.py`

Recommendation: use dirty tournament code only as evidence for the desired
`lightzero_exp*` discovery shape. Do not import tournament helpers into the
trainer tests or patch.

## Current Fixed-Path Facts

Observed trainer/status functions with fixed-path risk:

- `_latest_lightzero_iteration_checkpoint(exp_name)` scans only
  `exp_name / "ckpt"`.
- `_write_checkpoint_progress_latest(...)` depends on that fixed `exp_name`
  lookup.
- `_prepare_lightzero_auto_resume(...)` scans current/prior attempts at
  `train/lightzero_exp/ckpt`, plus the stable mirror.
- `_find_lightzero_resume_sidecar(...)` scans current/prior attempts at
  `train/lightzero_exp/<resume_state_dir>`, plus the stable mirror.
- `_scan_lightzero_artifacts(exp_name)` recursively scans one supplied
  experiment path.
- `_spawn_checkpoint_eval_triggers(...)`, `_publish_live_lightzero_checkpoints(...)`,
  and `_run_checkpoint_eval_poller(...)` use the single-exp scan.
- `lightzero_curvytron_run_status._checkpoint_summary(...)` picks the first
  existing checkpoint dir from stable mirror or fixed
  `train/lightzero_exp/ckpt`, then reports only checkpoint labels, iteration,
  mtime, and size.

Relevant adjacent model:

- Tournament discovery now uses `train/lightzero_exp*/ckpt/iteration_*.pth.tar`
  and records `exp_dir_name`.
- Existing tournament tests around
  `test_checkpoint_discovery_scans_timestamped_lightzero_exp_dirs` are good
  examples, but should stay decoupled from trainer tests.

Manifest note:

- `scripts/build_curvytron_opponent_mixture_manifest.py` has hard-coded default
  refs under `train/lightzero_exp/ckpt/iteration_*.pth.tar`, and its poller
  kwargs build an `exp_name_ref` ending in fixed `lightzero_exp`.
- Leave this out of the first gate unless the first patch starts resolving
  "recent", "mid", or "latest" dynamically.

## Local E2E Contract

Use a temp run root with both fixed and timestamped experiment dirs:

```text
tmp_path/
  training/lightzero-curvytron-visual-survival/run-e2e/
    checkpoints/lightzero/iteration_170000.pth.tar
    attempts/attempt-e2e/train/
      lightzero_exp/ckpt/iteration_0.pth.tar
      lightzero_exp/ckpt/ckpt_best.pth.tar
      lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar
      lightzero_exp_260513_123802/lightzero_resume_state/iteration_180000.resume_state.pkl
      summary.json
```

Test setup:

- Patch both `train_mod.RUNS_MOUNT` and `status_mod.RUNS_MOUNT` to `tmp_path`.
- Use `runs.attempt_train_ref(TASK_ID, run_id, attempt_id)` to build refs.
- Keep `exp_name_ref` pointing at fixed
  `.../train/lightzero_exp`; this is the stale closure that must still discover
  the timestamped sibling.
- Make checkpoint files non-empty. Add invalid and mutable names only to prove
  they are ignored.
- Stub eval and GIF Modal functions with fake `.spawn(**kwargs)` recorders whose
  returned calls implement `.get()`.
- Add `summary.json` so `_run_checkpoint_eval_poller` exits quickly with
  `idle_after_train_done_sec=0.0`.

Expected single source of truth:

```text
latest_checkpoint_name = iteration_180000.pth.tar
latest_checkpoint_label = iteration_180000
latest_checkpoint_ref contains lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar
```

Assertions:

- `_write_checkpoint_progress_latest(...)` writes `progress_latest.json` with
  `iteration == 180000`, `checkpoint_name == "iteration_180000.pth.tar"`, and
  a timestamped `checkpoint_ref`.
- `_prepare_lightzero_auto_resume(...)` returns `found is True`,
  `iteration == 180000`, and the same timestamped checkpoint ref/path.
- `_find_lightzero_resume_sidecar(..., iteration=180000)` returns
  `resume_state_found is True` and a timestamped sidecar ref/path.
- `status_mod._checkpoint_summary(run_id, attempt_id)` reports
  `latest_checkpoint == "iteration_180000"` and should include or expose enough
  data to verify the same timestamped ref.
- `_run_checkpoint_eval_poller(...)` schedules at least one eval and one GIF job
  whose `checkpoint_ref` is the timestamped ref, with
  `eval_id == "live_checkpoint_iteration_180000"` and
  `checkpoint_label == "iteration_180000"`.
- If poller semantics remain "schedule every unseen checkpoint", the test should
  assert the timestamped checkpoint is included and that its eval/GIF metadata
  is coherent. If the intended behavior becomes "schedule only latest", make
  that policy explicit before asserting `scheduled_count == 1`.

## First Test Bundle

Put these in `tests/test_curvytron_live_checkpoint_eval_plumbing.py`:

1. `test_progress_latest_uses_timestamped_lightzero_exp_checkpoint`
2. `test_auto_resume_selects_timestamped_lightzero_exp_checkpoint`
3. `test_resume_sidecar_scans_timestamped_lightzero_exp_state_dir`
4. `test_save_hook_trigger_scans_timestamped_lightzero_exp_dirs`
5. `test_checkpoint_eval_poller_scans_timestamped_lightzero_exp_dirs`

Put this in `tests/test_curvytron_run_status.py`:

1. `test_run_status_checkpoint_summary_scans_timestamped_lightzero_exp_dirs`

Then add one coherence test only after the smaller red tests exist:

```text
test_lightzero_timestamped_checkpoint_contract_is_coherent
```

That test should reuse the same temp tree and assert progress, status, resume,
poller, eval request, and GIF request all name the timestamped checkpoint.

## Verification Commands For The Next Coding Pass

After adding only the red tests:

```bash
uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py -k "timestamped_lightzero_exp or checkpoint_summary_scans"
```

After the minimal source patch:

```bash
uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py
```

Useful status check before editing:

```bash
git status --porcelain=v1 -uall -- tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py src/curvyzero/infra/modal/lightzero_curvytron_run_status.py
```

## Recommendations

- Start with tests only. The first patch should be a small fixed-path discovery
  fix, not a module extraction.
- Keep the helper contract plain: `iteration`, `name`, `path`, `ref`,
  `source_kind`, `exp_dir_name`, `size_bytes`, and `mtime_ns`.
- Select latest by iteration, then mtime, then size, then ref/path string.
- Ignore invalid names, non-files, zero-byte files, resume sidecars in weight
  discovery, and mutable `ckpt_best` unless a caller explicitly asks for it.
- Defer `src/curvyzero/training/lightzero_checkpoints.py` until the red tests
  pass and the behavior is pinned.
- Do not redesign environment mechanics, reward, opponent semantics, or the
  historical two-seat selfplay path in this lane.
