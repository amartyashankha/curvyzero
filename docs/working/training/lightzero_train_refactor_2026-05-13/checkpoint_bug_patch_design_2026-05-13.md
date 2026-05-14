# Checkpoint Bug Patch Design - 2026-05-13

Scope: read-only/doc-only audit of the trusted stock LightZero training lane.
No source or test patch is included here. The bug is that DI-engine can renew
`cfg.exp_name` from `train/lightzero_exp` to
`train/lightzero_exp_YYMMDD_HHMMSS`, while CurvyZero trainer/status readers keep
scanning only the original directory.

## Root Cause Anchors

- The trainer computes a fixed intended exp dir before calling LightZero:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3187`
  to `:3193`.
- `_build_visual_survival_configs` writes that fixed path into
  `main_config.exp_name` at
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4458`
  to `:4460`.
- `_compile_config_summary` calls DI-engine `compile_config` with
  `save_cfg=False`, so it does not exercise the directory-renewal branch that
  real `train_muzero` uses with config saving:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4998`
  to `:5009`.
- The runtime wrappers are installed before `train_muzero` and close over the
  original `exp_name`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3539`
  to `:3567`.
- The actual call to stock LightZero is at
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3596`
  to `:3601`; inside that call, DI-engine may renew the actual checkpoint
  directory.

## Affected Trainer Paths

### Progress Latest

- `_latest_lightzero_iteration_checkpoint(exp_name)` scans only
  `exp_name / "ckpt"`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804`
  to `:1826`.
- `_write_checkpoint_progress_latest` depends on that fixed-path helper:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1829`
  to `:1865`.
- Callers are the `BaseLearner.save_checkpoint` wrapper at
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1895`
  to `:1905`, and the `SaveCkptHook.__call__` wrapper at
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2045`
  to `:2065`.

Current behavior: if fixed `lightzero_exp/ckpt` has only `iteration_0` and a
timestamped sibling has `iteration_180000`, progress can keep reporting
`iteration_0` while `learner_train_iter` advances.

Desired behavior: select the latest non-empty `iteration_*.pth.tar` across all
`attempt/train/lightzero_exp*/ckpt` roots, then write that exact source
`checkpoint_ref`, `checkpoint_path`, `checkpoint_name`, and `exp_dir_name`.

### Resume And Sidecars

- `_save_lightzero_resume_sidecar_state` looks for the matching checkpoint only
  at `Path(exp_name) / "ckpt" / iteration_N.pth.tar`, then writes sidecar state
  under the fixed `Path(exp_name) / lightzero_resume_state`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2085`
  to `:2124`.
- `_prepare_lightzero_auto_resume` scans current fixed `lightzero_exp/ckpt`,
  prior fixed `lightzero_exp/ckpt`, then the stable mirror:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5133`
  to `:5253`.
- `_find_lightzero_resume_sidecar` scans fixed state dirs plus the stable state
  mirror:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5256`
  to `:5345`.

Current behavior: sidecar save can return
`matching_iteration_checkpoint_not_found` even when the matching checkpoint
exists in a timestamped exp dir. Auto-resume can choose a stale fixed-path or
mirror checkpoint if the true latest checkpoint never got mirrored.

Desired behavior: discover checkpoints broadly before resume selection; for a
selected checkpoint, prefer the sidecar in the same exp dir, then other
matching broad state dirs in deterministic order, then the run-level mirror.
Sidecar save should write next to the actual checkpoint exp dir when that
checkpoint is in a timestamped directory, and still mirror to the run-level
resume-state root.

### Artifact Scan, Mirror, And Live Publication

- `_scan_lightzero_artifacts(exp_name)` recursively scans only the supplied root:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5432`
  to `:5468`.
- `_mirror_lightzero_checkpoints` mirrors only `checkpoint_files` from that
  stale summary:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5471`
  to `:5513`.
- `_publish_live_lightzero_checkpoints` uses `_scan_lightzero_artifacts(str(exp_name))`
  and writes `live_checkpoint_publish.json`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5516`
  to `:5545`. No active call site was found, but the helper is still affected if
  reused.
- Final train summary and `lightzero_artifacts_manifest.json` are built from the
  same scan:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3672`
  to `:3676`, `:3728` to `:3729`, and `:3808` to `:3822`.

Current behavior: final summaries and mirrors can say "no LightZero checkpoint
artifacts" or mirror only stale fixed-path checkpoints while timestamped
checkpoints exist.

Desired behavior: artifact scanning for an exp root inside an attempt train root
should scan sibling `lightzero_exp*` roots, include each source root in the
manifest, and mirror all non-empty iteration checkpoints with deterministic
latest ordering.

### Eval/GIF Scheduling

- `_install_live_checkpoint_publisher` calls `_spawn_checkpoint_eval_triggers`
  after save:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1742`
  to `:1794`.
- `_spawn_checkpoint_eval_triggers` scans the fixed exp root:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5693`
  to `:5735`.
- The Modal poller default also resolves to fixed `lightzero_exp`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8854`
  to `:8935`.
- The launcher passes that fixed ref to the spawned poller:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9927`
  to `:9975`.
- `_run_checkpoint_eval_poller` repeatedly scans only that fixed path:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6193`
  to `:6381`, with the scan at `:6274`.

Current behavior: `seen_count`, scheduled evals, and GIFs can stay at zero or
`iteration_0` even while timestamped checkpoints are being written.

Desired behavior: both hook-trigger and poller modes should discover candidate
checkpoints from the attempt train root via `lightzero_exp*/ckpt`. The
checkpoint ref passed to eval/GIF should be the actual source timestamped ref,
not the future mirror and not the stale fixed ref. `ckpt_best` remains excluded.

### Heartbeats And Manifests

- `_write_train_status_heartbeat` records the fixed `exp_name_ref` and run-level
  `checkpoint_root_ref`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8098`
  to `:8131`.
- `_write_run_manifest_once`, `_write_attempt_state`, and `_write_latest_attempt`
  do not discover checkpoints directly:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7921`
  to `:7928` and `:8045` to `:8095`.

Current behavior: these manifests are not direct readers, but they preserve
config/status payloads that point humans and tooling at the fixed exp ref.

Desired behavior: keep the fixed `exp_name_ref` as the configured path, but add
or propagate a `checkpoint_scan_glob`/`discovered_exp_dirs` summary from the
central helper where status payloads need discovery evidence.

## Affected Status Paths

- `lightzero_curvytron_run_status._checkpoint_summary` chooses the first existing
  dir from run mirror or fixed `attempt/train/lightzero_exp/ckpt`, then reads
  only that one dir:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:809` to `:860`.
- `_run_status` calls `_checkpoint_summary`:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:863` to `:878`.
- `_progress_curve` includes `_checkpoint_summary`:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:946` to `:1004`.
- `_eval_curve_status` includes `_checkpoint_summary`:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:1007` to `:1032`.
- The printed tables surface `checkpoint_count` and `latest_checkpoint` from
  that summary:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py:1074` to `:1103`,
  `:1200` to `:1325`, and `:1330` to `:1443`.

Current behavior: if the stable mirror exists but is stale, it wins before the
attempt dir is considered. If no mirror exists, fixed `lightzero_exp/ckpt` can
hide a newer timestamped checkpoint. If only a timestamped dir exists, status
can report no checkpoints.

Desired behavior: aggregate candidates across the run mirror and all
`attempt/train/lightzero_exp*/ckpt` dirs, then choose latest by the shared
selection rule. Include checkpoint refs/source kinds in the status payload so
callers can tell whether the latest came from the mirror or attempt source.

## Manifest Builder Risk

The stock training launcher's run/attempt manifest helpers are not the selector
bug. The adjacent manifest-builder risk is freezing refs that were chosen from a
fixed path:

- `scripts/build_curvytron_opponent_mixture_manifest.py` has default frozen
  opponent refs under `train/lightzero_exp/ckpt`:
  `scripts/build_curvytron_opponent_mixture_manifest.py:58` to `:75`.
- `_checkpoint_refs` only rejects mutable or non-iteration names:
  `scripts/build_curvytron_opponent_mixture_manifest.py:317` to `:328`.
- `build_manifest` freezes those refs into `checkpoint_refs`:
  `scripts/build_curvytron_opponent_mixture_manifest.py:1037` to `:1044` and
  `:1280`.

Current behavior: exact fixed refs may be valid historical choices, but they are
not safe as a "recent/mid/old" discovery pattern.

Desired behavior: either keep these as explicit user-provided immutable refs, or
resolve any generated recent/mid/old defaults through the same broad discovery
contract before freezing them.

The tournament helper already has the right scan shape:
`src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:254` to `:292`
uses `lightzero_exp*/ckpt`, with regression coverage in
`tests/test_curvytron_checkpoint_tournament.py:3162` to `:3292`.

## Central Helper Contract

Create one pure discovery contract, then make the existing functions delegate to
it. The existing `checkpoint_helper_contract.md` already has the right high
level shape; this is the concrete patch contract.

Candidate fields:

```text
iteration
checkpoint_name
checkpoint_path
checkpoint_ref
source_kind
run_id
attempt_id
exp_dir_name
size_bytes
mtime_ns
```

Attempt discovery:

```text
discover_lightzero_attempt_checkpoints(
    attempt_train_root: Path,
    *,
    mount: Path,
    source_kind_prefix: str = "attempt_lightzero_exp",
    iteration: int | None = None,
) -> list[dict[str, Any]]
```

It scans exactly:

```text
attempt_train_root/lightzero_exp*/ckpt/iteration_*.pth.tar
```

Run resume discovery:

```text
discover_lightzero_run_checkpoints(
    run_root: Path,
    *,
    current_attempt_id: str,
    mount: Path,
    include_prior_attempts: bool = True,
    include_mirror: bool = True,
) -> list[dict[str, Any]]
```

Sidecar discovery:

```text
discover_lightzero_resume_sidecars(
    run_root: Path,
    *,
    current_attempt_id: str,
    iteration: int,
    preferred_exp_dir_name: str | None,
    mount: Path,
) -> list[dict[str, Any]]
```

Selection rule for latest checkpoint:

```text
max(iteration, mtime_ns, size_bytes, checkpoint_ref)
```

Shared rules:

- Ignore invalid names, directories, missing files, and zero-byte files.
- Ignore `ckpt_best` and other mutable names for progress, resume, status, and
  eval/GIF scheduling.
- Return plain data only. No policy load, no torch load, no Modal remote call.
- Preserve exact source refs. The stable mirror is a candidate source, not a
  replacement for source attempt refs.
- Derive `attempt_train_root` from an existing `exp_name`/`exp_name_ref` by
  taking the parent when the name starts with `lightzero_exp`, so old call sites
  can be patched without changing public signatures first.

## Minimal Patch Order After Tests

1. Add failing regression tests first for progress, auto-resume, sidecar,
   poller, artifact scan/mirror, status, and manifest ref freezing.
2. Add the pure helper in the smallest local location, preferably near the
   existing trainer checkpoint helpers for the first patch. Extract to
   `src/curvyzero/training/lightzero_checkpoints.py` only after behavior is
   pinned.
3. Make `_latest_lightzero_iteration_checkpoint` and
   `_write_checkpoint_progress_latest` delegate to broad attempt discovery.
4. Patch `_scan_lightzero_artifacts`, `_mirror_lightzero_checkpoints`,
   `_spawn_checkpoint_eval_triggers`, and `_run_checkpoint_eval_poller` to use
   broad attempt roots. Keep existing public arguments for compatibility.
5. Patch `_prepare_lightzero_auto_resume`, `_find_lightzero_resume_sidecar`, and
   `_save_lightzero_resume_sidecar_state` so resume and sidecar state match the
   selected actual checkpoint.
6. Patch `lightzero_curvytron_run_status._checkpoint_summary` to aggregate
   mirror plus broad attempt candidates through the same selector.
7. Only after the trainer/status readers pass, update manifest-builder defaults
   or add guardrails for generated fixed-path refs.

## Regression Tests To Prove The Patch

- `test_progress_latest_uses_timestamped_lightzero_exp_checkpoint`: fixed
  `iteration_0` plus timestamped `iteration_180000`; assert
  `progress_latest.json` chooses timestamped.
- `test_auto_resume_selects_timestamped_lightzero_exp_checkpoint`: same tree;
  assert `_prepare_lightzero_auto_resume` chooses timestamped and records
  source kind/exp dir.
- `test_resume_sidecar_scans_timestamped_lightzero_exp_state_dir`: sidecar under
  timestamped exp dir; assert `_find_lightzero_resume_sidecar` finds it.
- `test_save_sidecar_writes_next_to_actual_timestamped_checkpoint`: fake learner
  at `180000`; checkpoint exists only in timestamped dir; assert sidecar is
  saved beside it and mirrored.
- `test_scan_lightzero_artifacts_includes_timestamped_exp_dirs`: assert manifest
  contains files from both fixed and timestamped roots and records root names.
- `test_checkpoint_mirror_copies_timestamped_iteration_checkpoint`: assert the
  run-level mirror receives the timestamped checkpoint, not only fixed
  `iteration_0`.
- `test_hook_trigger_scans_timestamped_lightzero_exp_dirs`: stub eval function;
  assert `_spawn_checkpoint_eval_triggers` passes timestamped source ref and
  skips `ckpt_best`.
- `test_checkpoint_eval_poller_scans_timestamped_lightzero_exp_dirs`: existing
  poller stub test shape, but place the checkpoint only under
  `lightzero_exp_260513_123802/ckpt`.
- `test_run_status_checkpoint_summary_scans_timestamped_lightzero_exp_dirs`:
  assert `_checkpoint_summary` reports `iteration_180000` and includes source
  ref/exp dir even if fixed `iteration_0` exists.
- `test_checkpoint_selection_tie_break_is_deterministic`: same iteration in
  fixed and timestamped dirs; assert mtime, size, then ref tie-break.
- `test_manifest_checkpoint_selection_uses_broad_discovery`: for any generated
  recent/mid/old manifest mode, assert selected refs came from broad discovery
  metadata, or explicit fixed refs are marked as explicit inputs rather than
  discovered defaults.

Do not use Modal for these tests. Temp directories under `tmp_path` are enough.
