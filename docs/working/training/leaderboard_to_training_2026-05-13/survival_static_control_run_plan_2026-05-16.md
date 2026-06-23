# Survival Static Control Run Plan, 2026-05-16

Status: docs-only plan. No build, submit, launch, or intake command was run in
this inspection.

## Goal

Run a tiny control slice that keeps the current trainer/submitter path but
removes tournament-refresh feedback. The control should answer whether late
policy collapse still happens when opponents are static.

Current r18fresh context: best survival improved in `18/18`, latest survival
improved in only `9/18`, and many rows later collapse. The live feedback lane
uses assignment refs plus mutable control refresh pointers; this plan removes
that pointer path while leaving checkpoint saving/eval/GIF artifacts on.

## Code Path Evidence

- Current builder: `scripts/build_curvytron_tonight18_manifest.py`.
  - It uses the current v2 trainer app via `curvytron_train_app_name()`.
  - It supports `--opponent-source mixture` and `--opponent-source assignment`.
  - It supports explicit refs through `--checkpoint-refs-file`.
  - It supports the refresh-disable knob
    `--assignment-refresh-interval-train-iter 0`.
- Current submitter: `scripts/submit_curvytron_survivaldiag_manifest.py`.
  - Without `--allow-launch`, it is a dry-run validator.
  - With row filters, it launches only selected rows.
  - It validates exact `iteration_N.pth.tar` initial checkpoints and rejects
    mutable `latest`/`ckpt_best` refs.
- Current v2 constants: `src/curvyzero/contracts/curvytron.py`.
  - Runs volume: `curvyzero-runs-v2`.
  - Trainer app: `curvyzero-lightzero-curvytron-visual-survival-train-v2`.
  - `source_max_steps=1048576`.
  - `save_ckpt_after_iter=10000`.
  - `commit_on_checkpoint=true`.
  - Current default feedback lane refresh interval is `2000`, which is exactly
    what this control disables at manifest level.

The builder validator expects the full `3 reward x 3 recipe x 2 noise = 18`
matrix. So the minimal control should build an 18-row manifest but submit only a
2-3 row slice.

## Recommended Control Slice

Use static inline mixtures, not assignment refs. That avoids both immutable
assignment publication and mutable refresh pointers.

Selected rows after build:

| Row | Purpose |
| --- | --- |
| `r007` | `survival_plus_bonus_no_outcome`, recipe `blank10-wall10-rank2_25-rank1_55`, clean noise |
| `r009` | `survival_plus_bonus_no_outcome`, recipe `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5`, clean noise |
| `r011` | `survival_plus_bonus_no_outcome`, recipe `blank20-wall5-rank1_70-rank1imm5`, clean noise |

Why these three: they keep reward fixed to the survival objective under
investigation, remove the stochastic/noise axis, and vary only the opponent
recipe. If only two rows are affordable, use `r007` and `r011`.

## Immutable Seed Source

Use the already prepared nonzero checkpoint refs file:

```text
artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt
```

The first ref in that file becomes the learner initial checkpoint for every
non-scratch row:

```text
training/lightzero-curvytron-visual-survival/curvy-n18conn-sparse-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-clean-s822650214/attempts/try-n18conn-sparse-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-clean-s822650214/train/lightzero_exp/ckpt/iteration_240000.pth.tar
```

This is an exact immutable `iteration_240000.pth.tar` ref, not `latest` or
`ckpt_best`. Existing audit artifact
`artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/source-refs-v2-target-after-copy-audit.json`
reports `ok=true`, `ref_count=96`, `missing_ref_count=0`,
`bad_ref_count=0`, `existence_checked=true`, and
`runs_volume_name=curvyzero-runs-v2`.

The first four refs in `refs.txt` become static rank slots for the inline
opponent mixtures. The builder records them in `top_checkpoint_source`.

## Exact Commands

Set names:

```bash
MATRIX=curvy-r18staticctl-seeded-norefresh-20260516a
RUN_PREFIX=curvy-r18staticctl
ATTEMPT_PREFIX=try-r18staticctl
REFS=artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt
MAN=artifacts/local/curvytron_tonight18_manifests/$MATRIX/$MATRIX.json
```

Build the review manifest:

```bash
uv run python scripts/build_curvytron_tonight18_manifest.py \
  --checkpoint-refs-file "$REFS" \
  --opponent-source mixture \
  --assignment-refresh-interval-train-iter 0 \
  --matrix-name "$MATRIX" \
  --run-prefix "$RUN_PREFIX" \
  --attempt-prefix "$ATTEMPT_PREFIX"
```

Verify the manifest proves no refresh feedback:

```bash
jq '{
  fixed_refresh: .fixed_knobs.assignment_refresh_interval_train_iter,
  assignment_bank: .assignment_bank,
  selected_rows: [
    .rows[]
    | select(["r007","r009","r011"] | index(.row_id))
    | {
        row_id,
        label,
        opponent_source,
        opponent_mixture_enabled,
        opponent_assignment_ref,
        opponent_assignment_refresh_ref,
        train_assignment_ref: .train_kwargs.opponent_assignment_ref,
        poller_assignment_ref: .poller_kwargs.opponent_assignment_ref,
        train_has_refresh_ref: (.train_kwargs | has("opponent_assignment_refresh_ref")),
        train_has_refresh_interval: (.train_kwargs | has("opponent_assignment_refresh_interval_train_iter")),
        initial_policy_checkpoint_ref: .train_kwargs.initial_policy_checkpoint_ref,
        initial_policy_checkpoint_load_mode: .train_kwargs.initial_policy_checkpoint_load_mode,
        save_ckpt_after_iter: .train_kwargs.save_ckpt_after_iter,
        commit_on_checkpoint: .train_kwargs.commit_on_checkpoint
      }
  ]
}' "$MAN"
```

Expected proof fields:

- `.fixed_knobs.assignment_refresh_interval_train_iter == 0`
- `.assignment_bank == null`
- selected rows have `opponent_source == "mixture"`
- selected rows have `opponent_mixture_enabled == true`
- selected rows have `opponent_assignment_ref == null`
- selected rows have `opponent_assignment_refresh_ref == null`
- selected rows have `.train_kwargs.opponent_assignment_ref == null`
- selected rows have `.poller_kwargs.opponent_assignment_ref == null`
- selected rows do not have
  `.train_kwargs.opponent_assignment_refresh_ref`
- selected rows do not have
  `.train_kwargs.opponent_assignment_refresh_interval_train_iter`
- selected rows have exact `initial_policy_checkpoint_ref` ending in
  `iteration_240000.pth.tar`
- selected rows have `initial_policy_checkpoint_load_mode == "matching_shape"`
- selected rows have `save_ckpt_after_iter == 10000`
- selected rows have `commit_on_checkpoint == true`

Dry-run the selected submit only:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py "$MAN" \
  --row-id r007 \
  --row-id r009 \
  --row-id r011 \
  --output "artifacts/local/curvytron_tonight18_manifests/$MATRIX/$MATRIX.selected3.dry_submit.json"
```

Expected dry-run output:

- `dry_run=true`
- `selected_row_count=3`
- `row_count=3`
- `assignment_write_count=0`
- `refresh_pointer_write_count=0`

Operator launch command, not run during this inspection:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py "$MAN" \
  --row-id r007 \
  --row-id r009 \
  --row-id r011 \
  --allow-launch \
  --output "artifacts/local/curvytron_tonight18_manifests/$MATRIX/$MATRIX.selected3.launch.json"
```

## Assignment Variant

If an immutable assignment/audit artifact is preferred over inline mixture JSON,
use the same command but replace `--opponent-source mixture` with
`--opponent-source assignment` and keep
`--assignment-refresh-interval-train-iter 0`.

Expected static-assignment proof fields then change to:

- `.fixed_knobs.assignment_refresh_interval_train_iter == 0`
- `.assignment_bank.assignments` exists
- `.assignment_bank.refresh_pointers` is absent/null
- selected rows have `opponent_source == "assignment"`
- selected rows have `opponent_assignment_ref` under `control:.../assignments/...`
- selected rows have `opponent_assignment_refresh_ref == null`
- selected rows do not have
  `.train_kwargs.opponent_assignment_refresh_ref`

For the most minimal feedback-free control, prefer the inline mixture plan.

## Tournament Intake

Strictest isolation: do not attach the control to any tournament intake. The
training rows still write checkpoints, background evals, GIF status, and
survival curves.

If ratings are useful, attach a separate diagnostic tournament after the rows
are launched. Do not use the current r18fresh feedback tournament and do not run
`training-candidate-refresh` or `training-candidate-auto-refresh` for this
control. That keeps ratings visible without writing any control refresh pointer
back to trainers.

Optional diagnostic intake command, not run during this inspection:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode intake-seed \
  --tournament-id curvy-r18staticctl-live-20260516a \
  --rating-run-id elo-r18staticctl-live-20260516a \
  --run-id-prefix "$RUN_PREFIX-" \
  --checkpoint-selection all \
  --round-count 1 \
  --continue-from-latest \
  --pair-selection adaptive_v0 \
  --pairs-per-round 300 \
  --games-per-pair 21 \
  --games-per-shard 21 \
  --reuse-policies-per-shard \
  --active-pool-limit 100 \
  --decision-source-frames 1 \
  --decision-ms 16.666666666666668 \
  --source-physics-step-ms 16.666666666666668 \
  --policy-mode eval \
  --max-steps 1048576 \
  --num-simulations 8 \
  --no-save-gif \
  --intake-spawn-rating
```

Do not pass any `--training-candidate-refresh-pointers` for this tournament.
Do not point the scheduled controller at this tournament. The control manifest
itself has no pointer field for the controller to update.

## Artifacts To Monitor

Training volume refs per selected row:

- `training/lightzero-curvytron-visual-survival/<run_id>/run.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/latest_attempt.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/attempt.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/status_heartbeat.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/progress_latest.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/summary.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/checkpoint_eval_poller.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/background_gif_jobs.json`
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/env_steps.jsonl`
- `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero/iteration_N.pth.tar`
- `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero/iteration_N.pth.tar.metadata.json`

No-refresh runtime proof:

- `summary.json.opponent_assignment_refresh.enabled == false`
- `summary.json.opponent_assignment_refresh.pending_assignment_ref == null`
- `summary.json.opponent_assignment_refresh.event_count == 0`
- no meaningful
  `train/opponent_assignment_refresh_events.jsonl` should be created

Status commands:

```bash
RUN_IDS=$(jq -r '[.rows[] | select(["r007","r009","r011"] | index(.row_id)) | .run_id] | join(",")' "$MAN")

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$RUN_IDS" \
  --output table

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$RUN_IDS" \
  --output eval-summary

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$RUN_IDS" \
  --output curve-summary
```

If optional intake is attached, monitor:

- `tournaments/curvytron/curvy-r18staticctl-live-20260516a/intake/elo-r18staticctl-live-20260516a/config.json`
- `tournaments/curvytron/curvy-r18staticctl-live-20260516a/intake/elo-r18staticctl-live-20260516a/progress.json`
- `tournaments/curvytron/curvy-r18staticctl-live-20260516a/ratings/elo-r18staticctl-live-20260516a/config.json`
- `tournaments/curvytron/curvy-r18staticctl-live-20260516a/ratings/elo-r18staticctl-live-20260516a/progress.json`
- `tournaments/curvytron/curvy-r18staticctl-live-20260516a/ratings/elo-r18staticctl-live-20260516a/latest.json`
- `tournaments/curvytron/curvy-r18staticctl-live-20260516a/ratings/elo-r18staticctl-live-20260516a/rounds/round-000000/input.json`

Use `scripts/curvytron_tournament_debug_bundle.py` after fetching those JSON
artifacts locally if the intake/rating state becomes confusing.

## Readout

Primary comparison:

- Does `latest_mean_survival` still drop far below `best_mean_survival` in the
  three static rows?
- Do action-collapse flags still appear?
- Is `best_iteration` mid-run while final/latest regresses?
- Does the static control produce a cleaner latest-vs-best gap than r18fresh?

If collapse persists under static mixtures and no refresh pointer, the leading
cause is probably not tournament-refresh feedback. If collapse weakens or
disappears, refresh-induced nonstationarity becomes the prime suspect.
