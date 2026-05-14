# Naming And Test Cleanup Critique

Date: 2026-05-13

Scope: read-only critique for the CurvyTron stock LightZero training refactor.
No source or test edits were made.

## Summary

- FACT: The docs already set the right boundary: keep
  `--mode train` close to stock `lzero.entry.train_muzero`, and treat
  `--mode two-seat-selfplay` as historical.
- FACT: The main confusion risk is not one bad name. It is several near-synonyms
  around checkpoints, progress, status, resume, pollers, and manifests.
- OPINION: The refactor should first make the checkpoint language boring and
  shared, then move code in small cuts. A broad "helper cleanup" before that
  will probably move the bug around.

## Unclear Names And Terms

- FACT: The docs use `Trusted Training Lane`, `Trusted Lane`,
  `trusted learning lane`, and `trusted training path` for the same
  executable path.
- OPINION: Standardize on `stock LightZero train path` in prose. If a shorter
  noun is needed, use `stock train path`.

- FACT: `lane` is used for the executable path, the refactor effort,
  historical self-play, and subagent workstreams.
- OPINION: Use `training path` for code, `refactor scope` for this doc set,
  and `agent workstream` for parallel audits.

- FACT: `CurvyTron` and `CurvyZero` appear together without a local glossary
  distinction.
- OPINION: Add one glossary rule: `CurvyTron` is the game/task;
  `CurvyZero` is the package/codebase.

- FACT: `stacked_debug_visual_survival_train.py` is the live stock trainer file,
  despite its legacy/debug-heavy name.
- OPINION: Do not rename the file in the first pass. If a later thin entrypoint
  is created, prefer `curvytron_stock_lightzero_train_modal.py` or
  `curvytron_lightzero_stock_train_modal.py`.

- FACT: Candidate module names conflict across docs:
  `training/checkpoints.py` in one place and `training/lightzero_checkpoints.py`
  elsewhere.
- OPINION: Prefer explicit LightZero-prefixed helper modules:
  `lightzero_checkpoints.py`, `lightzero_progress.py`,
  `lightzero_resume.py`, and `lightzero_background_jobs.py`.

- FACT: `checkpoint discovery` currently covers parsing names, finding
  `lightzero_exp*` dirs, listing candidates, selecting latest/all/specific
  checkpoints, matching resume sidecars, and freezing manifest refs.
- OPINION: Split the vocabulary:
  `checkpoint name parsing`, `lightzero_exp* discovery`,
  `checkpoint candidate listing`, `checkpoint selection`,
  `resume-state sidecar lookup`, and `checkpoint ref freezing`.

- FACT: `broad discovery` is used as shorthand but is not defined.
- OPINION: Use `lightzero_exp* discovery` or
  `LightZero experiment sibling discovery` instead.

- FACT: `status`, `progress`, `heartbeat`, `latest`, and `display cache` are
  close enough to confuse readers.
- OPINION: Use `progress artifact` for machine-readable persisted progress,
  `status heartbeat` for trainer liveness/stage, and `status view` for CLI/web
  summaries.

- FACT: `full resume` overclaims current behavior. The sidecar text says replay
  GameSegments and env-manager internals are not fully restored.
- OPINION: Prefer `resume-state sidecar` and
  `_install_lightzero_resume_sidecar_hooks`.

- FACT: `External Observability` includes eval/GIF artifacts and manifests, but
  manifests can feed later checkpoint refs.
- OPINION: Split this into `artifact jobs` for eval/GIF and
  `checkpoint ref consumers` for manifests/tournaments/status readers.

## Better Simple Names If Extracted

- OPINION: First extracted module should be
  `src/curvyzero/training/lightzero_checkpoints.py`, after tests and the small
  in-place bugfix are green.
- OPINION: Good first functions:
  `parse_lightzero_iteration_checkpoint_name`,
  `parse_lightzero_resume_state_name`,
  `list_lightzero_exp_checkpoint_dirs`,
  `list_lightzero_checkpoint_candidates`,
  `select_latest_lightzero_checkpoint`,
  `select_lightzero_checkpoint`,
  and `find_matching_resume_state_sidecar`.

- OPINION: Keep progress/status separate:
  `build_train_progress_latest_payload`,
  `write_train_progress_latest`,
  `build_checkpoint_status_summary`,
  and `build_train_status_view`.

- OPINION: Use `lightzero_background_jobs.py` or
  `live_checkpoint_artifacts.py` for poller/eval/GIF request builders. Avoid
  generic names like `checkpoint_utils.py` or `background_eval.py` if the module
  owns both eval and GIF triggering.

- OPINION: Function rename candidates, when wrappers can be preserved:
  `_latest_lightzero_iteration_checkpoint` ->
  `select_latest_checkpoint_in_lightzero_exp_dir` before broadening, then
  `select_latest_lightzero_checkpoint`;
  `_scan_lightzero_artifacts` -> `scan_lightzero_exp_artifacts`;
  `_prepare_lightzero_auto_resume` -> `select_auto_resume_checkpoint`;
  `_find_lightzero_resume_sidecar` -> `find_matching_resume_state_sidecar`;
  `_write_checkpoint_progress_latest` -> `write_train_progress_latest`;
  `_run_checkpoint_eval_poller` -> `run_live_checkpoint_artifact_poller`;
  `_spawn_one_checkpoint_background_eval` ->
  `spawn_live_checkpoint_eval_and_gif_jobs`;
  `_build_visual_survival_configs` ->
  `build_stock_muzero_visual_survival_configs`;
  `_run_visual_survival_train` -> `run_stock_lightzero_curvytron_train`.

- OPINION: Better new test names:
  `test_latest_lightzero_checkpoint_scans_lightzero_exp_siblings`,
  `test_progress_latest_uses_lightzero_exp_sibling_discovery`,
  `test_auto_resume_selects_latest_checkpoint_from_lightzero_exp_siblings`,
  `test_resume_sidecar_matches_selected_checkpoint_iteration`,
  `test_checkpoint_eval_poller_discovers_lightzero_exp_sibling_checkpoints`,
  `test_run_status_checkpoint_summary_scans_lightzero_exp_siblings`,
  and `test_manifest_checkpoint_refs_are_immutable_iteration_refs`.

## Tests That Look Stale

- FACT: `tests/test_curvytron_run_status.py::test_checkpoint_summary_includes_checkpoint_mtimes`
  only covers `train/lightzero_exp/ckpt`.
- OPINION: Strengthen it with a timestamped `lightzero_exp_*` sibling before
  treating it as reliable status coverage.

- FACT: The first progress-writer tests in
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py` build fixed
  `lightzero_exp/ckpt` paths and assert refs ending in that fixed path.
- OPINION: They are stale for discovery, but should be preserved until new
  progress tests prove timestamped sibling behavior.

- FACT: `test_checkpoint_eval_poller_completes_eval_inspection_and_selfplay_gif_jobs`
  in the same file creates one fixed `lightzero_exp/ckpt` checkpoint.
- OPINION: Preserve its job-spawn assertions, then add a sibling-dir poller
  test rather than rewriting it wholesale.

- FACT: `tests/test_curvytron_survivaldiag_manifest.py` asserts poller kwargs
  end with `/train/lightzero_exp`.
- OPINION: That assertion becomes stale if the poller API moves to a train root
  or discovery root. Until then, it documents the current launcher contract.

- FACT: No direct test was found for `_prepare_lightzero_auto_resume` plus
  `_find_lightzero_resume_sidecar` across `lightzero_exp*` siblings.
- OPINION: This is the highest-risk missing test area.

- FACT: Two-seat tests in `tests/test_curvytron_live_checkpoint_eval_plumbing.py`
  protect the historical branch while it still exists.
- OPINION: Do not use those tests as evidence for learning quality. Later,
  consider moving or labeling them as legacy two-seat coverage.

## Tests To Preserve Until Replacements Exist

- FACT: `tests/test_curvytron_checkpoint_tournament.py` already tests broad
  timestamped LightZero discovery, including latest, iteration-filtered, and
  all-checkpoint selection.
- OPINION: Keep these as evidence and examples, but do not couple the trainer
  refactor to tournament internals.

- FACT: `tests/test_opponent_mixture.py` and
  `tests/test_curvytron_opponent_mixture_manifest.py` protect immutable frozen
  checkpoint refs and eval/GIF mixture metadata.
- OPINION: Preserve them even if manifest checkpoint selection changes.

- FACT: `tests/test_curvytron_gif_browser.py`, `tests/test_eval_curves.py`, and
  `tests/test_curvytron_run_status.py` protect readout behavior outside raw
  checkpoint discovery.
- OPINION: Keep them green while adding narrower checkpoint-discovery tests.

- FACT: Environment/trainer surface tests such as
  `tests/test_multiplayer_source_state_trainer_surface.py` are mostly outside
  this coach scope.
- OPINION: Preserve them as interface guardrails, but do not edit them for a
  training-scaffolding refactor unless a trainer contract mismatch is proven.

- FACT: `tests/test_lightzero_phase_profiler.py` protects LightZero monkeypatch
  hook behavior.
- OPINION: Re-run or extend it if hook installation is moved; hook leaks would
  be painful to debug.

## Boundary And Abstraction Risks

- FACT: Resume selection, status display, live eval/GIF polling, GIF browser
  listing, and tournament intake all consume checkpoint refs, but they have
  different freshness and trust rules.
- OPINION: Share low-level candidate listing and parsing, not one giant
  "checkpoint discovery" policy for every consumer.

- FACT: The Modal trainer monkeypatches LightZero/DI-engine hooks around
  `train_muzero`.
- OPINION: Keep hook installation scoped to the stock train call. Do not move
  hook mutation into generic helpers used by eval/GIF/status code.

- FACT: `run_management.py` is already narrow and says it does not implement
  resume or Modal orchestration.
- OPINION: Keep it that way. It should own refs and JSON shapes, not training
  semantics.

- FACT: Stable mirrored checkpoints can exist even when the active LightZero
  experiment directory is stale or timestamped.
- OPINION: Treat the mirror as a consumer/fallback, not as proof that fixed-path
  discovery is correct.

- FACT: `_build_visual_survival_configs` mixes LightZero config construction
  with many environment/reward/opponent fields.
- OPINION: Do not extract this early. It is a tempting module boundary but a
  bad first move because it can accidentally change environment semantics.

- FACT: `--mode two-seat-selfplay` is a peer mode in the current CLI branch.
- OPINION: Do not let its names, tests, or helpers shape the stock LightZero
  train refactor. If separated later, make the legacy status explicit.

## Concrete Recommendations

1. FACT: The immediate bug is fixed-path checkpoint discovery.
   OPINION: Add the `lightzero_exp*` regression tests before any extraction.

2. FACT: Existing fixed-path tests still protect useful behavior.
   OPINION: Mark them as `strengthen` or `replace after coverage`, not
   deletion candidates.

3. FACT: The docs currently disagree on helper module names.
   OPINION: Pick `lightzero_checkpoints.py` as the first helper name and defer
   the other modules until after the bugfix.

4. FACT: The largest source file contains stock training, historical two-seat,
   eval/GIF jobs, status artifacts, and config building.
   OPINION: Extract only one responsibility at a time, preserving old wrapper
   function names through the first pass.

5. FACT: Future agents will likely search by human questions.
   OPINION: Name helpers around questions: "which checkpoints exist?", "which
   checkpoint resumes?", "what progress artifact do we write?", "which artifact
   jobs should launch?", and "which refs are immutable?"
