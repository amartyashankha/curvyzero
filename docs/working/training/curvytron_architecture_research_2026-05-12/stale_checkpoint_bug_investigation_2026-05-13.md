# Stale Checkpoint Bug Investigation

Date: 2026-05-13

Scope: read-only investigation. No source code was changed. Modal runs were not
killed or modified during this pass.

## Short Read

The scary symptom was real, but the first interpretation was incomplete.

Some runs looked stuck at `iteration_0` because CurvyZero was scanning this
fixed path:

```text
.../train/lightzero_exp/ckpt
```

After restarts, LightZero/DI-engine often wrote checkpoints here instead:

```text
.../train/lightzero_exp_260513_HHMMSS/ckpt
```

So the learner could keep training and saving, while our status page, checkpoint
poller, progress writer, resume sidecar writer, and eval/GIF pipeline looked in
the wrong directory.

Plainly: this is mostly an observability/path-tracking bug, not proof that
LightZero stopped learning or stopped checkpointing.

## Why This Happens

Upstream LightZero calls DI-engine `compile_config(...)` inside
`lzero.entry.train_muzero`.

Source checked:

- LightZero v0.2.0 `train_muzero`:
  `https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/entry/train_muzero.py#L72-L127`
- DI-engine v0.5.3 `compile_config`:
  `https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/config/config.py#L369-L471`
- DI-engine v0.5.3 `BaseLearner`:
  `https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/worker/learner/base_learner.py#L75-L104`
- DI-engine v0.5.3 `SaveCkptHook`:
  `https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/worker/learner/learner_hook.py#L127-L169`

In DI-engine `compile_config`, the default is `renew_dir=True`. If
`cfg.exp_name` already exists, DI-engine appends a timestamp:

```text
if os.path.exists(cfg.exp_name) and renew_dir:
    cfg.exp_name += datetime.datetime.now().strftime("_%y%m%d_%H%M%S")
```

LightZero then creates the learner, collector, evaluator, and tensorboard writer
with the mutated `cfg.exp_name`.

CurvyZero, however, computes this fixed path before calling `train_muzero`:

```text
exp_name_ref = attempt_train_ref / "lightzero_exp"
exp_name = Path(exp_name_ref.as_posix())
```

Several CurvyZero hooks close over that original `exp_name`. They do not know
that DI-engine later changed the actual runtime experiment directory.

## Code Paths Affected

These CurvyZero paths use the fixed original `lightzero_exp` path:

- `progress_latest.json`: `_write_checkpoint_progress_latest(...)` scans
  `exp_name/ckpt`.
- resume sidecars: `_save_lightzero_resume_sidecar_state(...)` looks for
  `exp_name/ckpt/iteration_<learner.train_iter>.pth.tar`.
- auto-resume: `_prepare_lightzero_auto_resume(...)` scans the current attempt
  `lightzero_exp/ckpt`, older attempts' `lightzero_exp/ckpt`, and the canonical
  mirror. It does not scan `lightzero_exp_*` timestamped directories.
- checkpoint eval/GIF poller: `_run_checkpoint_eval_poller(...)` scans the fixed
  `exp_name_ref`.
- final mirror: `_mirror_lightzero_checkpoints(...)` scans the fixed
  `artifact_summary` root after `train_muzero` returns.

The actual LightZero checkpoint save path uses the compiled `cfg.exp_name`:

- `BaseLearner(..., exp_name=cfg.exp_name)`
- `SaveCkptHook.__call__`
- save directory `./{engine.exp_name}/ckpt`

So after DI-engine renames the experiment dir, training and CurvyZero
observability disagree about where checkpoints live.

## Evidence Table

These checks used `modal volume ls` against `curvyzero-runs`.

| Run | Fixed `lightzero_exp/ckpt` | Timestamped checkpoint dir | What this means |
| --- | --- | --- | --- |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | only `iteration_0.pth.tar` and `ckpt_best.pth.tar` | `lightzero_exp_260513_123802/ckpt` has `iteration_0` through `iteration_180000` | Status said stuck at zero, but real checkpoints exist elsewhere. |
| same run, older restart dir | `lightzero_exp_260513_121133/ckpt` has only `iteration_0` | later restart dir has many checkpoints | The run restarted more than once. |
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | sampled fixed path looked stale | `lightzero_exp_260513_175026/ckpt` has `iteration_0` through `iteration_40000` | Same pattern. |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | sampled fixed path looked stale | `lightzero_exp_260513_172427/ckpt` has `iteration_0` through `iteration_70000` | Same pattern. |
| `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | sampled fixed path looked stale | `lightzero_exp_260513_152011/ckpt` has `iteration_0` through `iteration_100000` | Same pattern. |
| `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011` | sampled fixed path looked stale | `lightzero_exp_260513_152601/ckpt` has `iteration_0` through `iteration_110000` | Same pattern. |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s16-c32-l32-repM-k10-c1-s2306011` | sampled fixed path looked stale | `lightzero_exp_260513_151652/ckpt` has through `iteration_40000`; `lightzero_exp_260513_170247/ckpt` has through `iteration_10000`; `lightzero_exp_260513_172955/ckpt` has through `iteration_50000` | Same pattern, with multiple restart fragments. |

This strongly explains why `progress_latest.json` could show a high
`learner_train_iter` while still reporting `checkpoint_name=iteration_0.pth.tar`:
the progress writer scanned the old directory, while the learner saved in the
timestamped directory.

This covers all six rows that the fixed-path status snapshot showed at
`iteration_0`. The fixed-path health table was therefore misleading for those
rows; it did not scan the true set of LightZero experiment directories.

The Modal tournament discovery helper was also checked directly against these
six rows with `--mode discover --checkpoint-selection latest --run-ids ...`.
It found all six and returned timestamped checkpoint refs:

| Run | Broad discovery latest checkpoint |
| --- | --- |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | `lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar` |
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | `lightzero_exp_260513_121430/ckpt/iteration_110000.pth.tar` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | `lightzero_exp_260513_172427/ckpt/iteration_80000.pth.tar` |
| `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | `lightzero_exp_260513_152011/ckpt/iteration_110000.pth.tar` |
| `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011` | `lightzero_exp_260513_152601/ckpt/iteration_110000.pth.tar` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s16-c32-l32-repM-k10-c1-s2306011` | `lightzero_exp_260513_172955/ckpt/iteration_50000.pth.tar` |

## What The Earlier Evidence Still Means

The earlier subagent findings were useful, but should now be reinterpreted.

`progress_latest.source="SaveCkptHook.__call__"` means the hook wrapper ran. It
does not by itself prove that DI-engine saved a checkpoint in the fixed
directory.

`train_status=running` is weak. It means a status file says running. It is not a
hard liveness or checkpoint-health proof.

The missing sidecars are explained by the path mismatch. The sidecar writer
looks for a matching checkpoint in fixed `lightzero_exp/ckpt`; if the checkpoint
is in `lightzero_exp_260513_HHMMSS/ckpt`, sidecar save returns
`matching_iteration_checkpoint_not_found`.

Eval/GIF did not advance because the poller scanned fixed `lightzero_exp/ckpt`.
It did not see the timestamped checkpoints.

Auto-resume is probably worse than observability. If a Modal preemption happens,
CurvyZero may resume from the old fixed directory or mirror rather than from the
newest timestamped checkpoint. That can lose hours of training progress or fork
another timestamped directory.

## Other Real Problems Found

These are still real, but they are not the main explanation for the
`iteration_0` rows.

1. Modal preemption happened many times in the app.
2. Eval/GIF volume commits hit many `DataLossError: failed to publish commit to
   server` retries.
3. Some GIF/eval workers tried to load checkpoints while they were corrupt,
   incomplete, or modified, producing PyTorch zip central-directory errors.
4. Modal Volume docs warn that v1 Volumes are meant for write-once/read-many
   work, that too many concurrent commits cause contention, and that concurrent
   modifications have last-write-wins behavior:
   `https://frontend.modal.com/docs/guide/volumes`

Those problems can make artifact publication and eval/GIF flaky. But the
timestamped `exp_name` discovery explains the stale-zero checkpoint read more
directly.

Latest quick log pass, 2026-05-13 16:23 EDT, against app
`ap-2mvquK3ZZJvDqleyHS088M` over the previous 6 hours:

- `Runner interrupted`: about 362 log lines.
- `failed to publish commit`: about 1210 log lines.
- `PytorchStreamReader failed reading zip archive`: about 201 log lines.
- `matching_iteration_checkpoint_not_found`: no matching log lines in this
  app-log search.

These are rough log-line counts, not deduplicated event counts. They support
the same split: preemption and eval/GIF volume pressure are real, but the
fixed-path `iteration_0` symptom has a clearer directory-discovery cause.

## Current Best Diagnosis

Highest-confidence diagnosis:

```text
DI-engine changes cfg.exp_name on restart because the configured exp directory
already exists. CurvyZero continues tracking the original exp path.
```

Consequences:

- Training checkpoints can exist but be invisible to our status tooling.
- Eval/GIF can stop after `iteration_0` even while training produces later
  checkpoints.
- Resume sidecars can stop being written.
- Auto-resume can resume from stale checkpoints.
- Current run-health summaries probably undercount progress for restarted rows.

Important tournament nuance: the current Modal tournament discovery helper in
`src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` already scans
`train_root.glob("lightzero_exp*/ckpt")` when it is asked to discover from run
roots. That broad scan is the right shape. The footgun remains for any caller
or side path that hands tournament code only a fixed
`train/lightzero_exp/ckpt/...` ref, or relies on CurvyZero's stable mirror,
progress file, status reader, or poller output.

Adjacent manifest nuance: `scripts/build_curvytron_opponent_mixture_manifest.py`
has default opponent checkpoint refs under `train/lightzero_exp/ckpt`. Those
were historical fixed refs, not a safe discovery pattern for new runs. New
opponent/tournament manifests should freeze refs only after a broad
`lightzero_exp*/ckpt` scan.

## What Is Not Yet Proven

These need more checks or instrumentation:

- Whether every fixed-path stale row outside the six known `iteration_0` rows is
  explained by timestamped `exp_name` directories. The six fixed-path
  `iteration_0` rows all fit, but the full preserved run set was not
  exhaustively scanned for slower/stale nonzero rows in this pass.
- Whether any rows also have true checkpoint-save failures.
- Whether checkpoint corruption comes from eval/GIF reading while a checkpoint
  is still being written, Modal Volume visibility timing, concurrent commits, or
  another race.
- Whether canonical mirrors are stale only because final mirroring waits for
  `train_muzero` to return, or also because live mirroring scans the wrong root.

## Read-Only Checks Run

Representative commands:

```bash
uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/train

uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/train/lightzero_exp/ckpt

uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/train/lightzero_exp_260513_123802/ckpt

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search DataLossError --since 12h

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search PytorchStreamReader --since 12h
```

## Subagent Notes Produced

- `modal_log_timeline_2026-05-13.md`
- `checkpoint_progress_semantics_2026-05-13.md`
- `iteration0_vs_healthy_checkpoint_evidence_2026-05-13.md`
- `stale_checkpoint_hypothesis_skeptical_review_2026-05-13.md`
- Earlier related notes:
  - `checkpoint_save_path_critique_agent_2026-05-13.md`
  - `artifact_liveness_audit_agent_2026-05-13.md`
  - `stale_config_analysis_newton_2026-05-13.md`
  - `log_investigation_newton_2026-05-13.md`

## Next Checks Before Any Patch

1. Full scan: for every preserved run, list all `train/lightzero_exp*` dirs and
   compute the true highest checkpoint across all of them.
2. Compare true highest checkpoint vs status-reader highest checkpoint.
3. Check whether the latest timestamped directory is still receiving new
   checkpoints.
4. Check whether eval/GIF summaries only exist for fixed-path checkpoints.
5. Check whether auto-resume selected a stale checkpoint after preemption.
6. Decide whether the clean fix is:
   - prevent DI-engine from renaming the directory, or
   - teach all CurvyZero observers/resume logic to follow the compiled
     `cfg.exp_name`, or
   - both.

## Patch Direction To Consider Later

No source patch was made in this pass, but the likely fix shape is now clearer:

1. In live hooks that run after a save, prefer the actual LightZero learner path
   (`learner.exp_name` or `engine.exp_name`) over the closed-over original
   `exp_name`.
2. In poller, resume, status, and mirror code, scan all sibling
   `train/lightzero_exp*/ckpt` directories and select by checkpoint iteration,
   mtime, and size.
3. Keep the exact selected checkpoint ref and source directory in status,
   tournament inputs, and eval/GIF summaries. Do not collapse everything to the
   basename alone when multiple timestamped directories exist.
4. If possible, prevent DI-engine from silently renewing the directory on
   intended resume. But do not rely on that alone; broad discovery is still the
   safer reader-side behavior for old artifacts and restarted jobs.

No code change was made in this investigation.
