# No-Tournament-Feedback Control Run Plan, 2026-05-16

Status: docs-only plan. I inspected the current manifest builder, grouped
submitter, trainer args, and live working notes. I did not build, submit, or
launch anything.

## Goal

Run a tiny control slice where training sees static opponents and blanks but
does not consume tournament feedback while learning. The checkpoints may still
be discovered by a separate diagnostic tournament, but that tournament must not
write assignment refresh pointers back to the trainers.

Hypothesis under test: late survival regression is caused mainly by
tournament-fed opponent nonstationarity, not by the stock trainer/reward loop
itself.

## Current Code Truth

- `scripts/build_curvytron_tonight18_manifest.py` can already express the
  control:
  - `--opponent-source mixture` embeds static opponent mixtures directly in each
    row.
  - `--checkpoint-refs-file` supplies exact frozen `iteration_N.pth.tar` refs
    for rank slots and the shared initial policy.
  - `--assignment-refresh-interval-train-iter 0` prevents refresh pointer fields
    from being emitted.
  - The builder still emits the full `18`-row matrix, so submit only selected
    rows.
- `scripts/submit_curvytron_survivaldiag_manifest.py` is safe as a dry-run
  validator unless `--allow-launch` is passed. For mixture rows it writes `0`
  assignments and `0` refresh pointers.
- The trainer default refresh interval is `0`, and the refresh hook is installed
  only when `opponent_assignment_refresh_interval_train_iter > 0` and a refresh
  ref exists. The current live feedback manifest overrides this to `2000`; this
  control does not.
- The trainer functions already accept `opponent_mixture_spec`,
  `opponent_assignment_ref`, `initial_policy_checkpoint_ref`,
  `save_ckpt_after_iter`, and `commit_on_checkpoint`.

## Code Change Required?

No code change is required for a static frozen/blanks control. Use the existing
manifest builder with `--opponent-source mixture` and refresh interval `0`.

A code change would be required only for true "own latest" or current-policy
self-play opponents. The current live source-state path records
`current_policy_self_play=false` and trains against weighted fixed opponent
mixtures or assignments; it does not automatically make the learner's latest
checkpoint the opponent during the same run.

## Source Refs

Use post-purge exact refs only. Do not reuse pre-purge manifests as launch truth.
For this control, derive the four static rank slots from the current post-purge
bounded live rating snapshot, accepting that it is a source of exact frozen refs
rather than a production-quality leaderboard.

```bash
SRC_DIR=artifacts/local/curvytron_no_tournament_control_20260516/source
mkdir -p "$SRC_DIR"

uv run --extra modal modal volume get --force \
  curvyzero-curvytron-tournaments-v2 \
  tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b/latest.json \
  "$SRC_DIR/r18fresh_bounded_latest.json"

jq -r '
  (.ratings // .rows)
  | map(select(
      ((.status // "active") == "active")
      and (((.checkpoint_ref // "") | test("iteration_[0-9]+\\.pth\\.tar$")))
      and ((((.checkpoint_ref // "") | test("iteration_0\\.pth\\.tar$"))) | not)
    ))
  | sort_by(.rank // 999999)
  | .[:4][]
  | .checkpoint_ref
' "$SRC_DIR/r18fresh_bounded_latest.json" > "$SRC_DIR/static_top4_nonzero_refs.txt"

test "$(wc -l < "$SRC_DIR/static_top4_nonzero_refs.txt")" -ge 4
```

If the last check fails, stop and do not launch the control. The fallback is a
separately named scratch/static-placeholder control, but that tests a weaker
question because it has no frozen checkpoint opponents.

## Manifest Commands

```bash
MATRIX=curvy-r18nofb-staticmix-20260516a
RUN_PREFIX=curvy-r18nofb
ATTEMPT_PREFIX=try-r18nofb
REFS=artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.txt
MAN=artifacts/local/curvytron_tonight18_manifests/$MATRIX/$MATRIX.json

uv run python scripts/build_curvytron_tonight18_manifest.py \
  --checkpoint-refs-file "$REFS" \
  --opponent-source mixture \
  --assignment-refresh-interval-train-iter 0 \
  --matrix-name "$MATRIX" \
  --run-prefix "$RUN_PREFIX" \
  --attempt-prefix "$ATTEMPT_PREFIX"
```

Verify the manifest is feedback-free:

```bash
jq '{
  matrix_name,
  opponent_source,
  fixed_refresh: .fixed_knobs.assignment_refresh_interval_train_iter,
  assignment_bank,
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

Required proof fields:

- `opponent_source == "mixture"`
- `fixed_refresh == 0`
- `assignment_bank == null`
- selected rows have `opponent_mixture_enabled == true`
- selected rows have no `opponent_assignment_ref` and no
  `opponent_assignment_refresh_ref`
- selected rows do not carry refresh keys in `train_kwargs`
- `initial_policy_checkpoint_ref` is exact and ends in `iteration_N.pth.tar`
- `save_ckpt_after_iter == 10000`
- `commit_on_checkpoint == true`

Dry-run grouped submission:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py "$MAN" \
  --row-id r007 \
  --row-id r009 \
  --row-id r011 \
  --output "artifacts/local/curvytron_tonight18_manifests/$MATRIX/$MATRIX.selected3.dry_submit.json"
```

Expected dry-run fields: `dry_run=true`, `selected_row_count=3`,
`row_count=3`, `assignment_write_count=0`, and
`refresh_pointer_write_count=0`.

Operator launch command, not run during this planning pass:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py "$MAN" \
  --row-id r007 \
  --row-id r009 \
  --row-id r011 \
  --allow-launch \
  --output "artifacts/local/curvytron_tonight18_manifests/$MATRIX/$MATRIX.selected3.launch.json"
```

## First Rows

Launch these first:

| Row | Why |
| --- | --- |
| `r007` | `survival_plus_bonus_no_outcome`, clean, first recipe. Directly mirrors one weak live slice without noise. |
| `r011` | Same reward/noise, rank1-heavy recipe. Tests whether the easier static recipe still regresses. |
| `r009` | Same reward/noise, wider rank4/rank3/rank2/rank1 recipe. Add if capacity allows a 3-row slice. |

If capacity is tight, launch only `r007` and `r011`. Keep `r009` as the first
extension; do not switch rewards before the static-vs-feedback question has a
clean read.

## Optional Tournament Pickup

Strict isolation means no tournament at all. If ratings are useful, use a fresh
diagnostic tournament that watches this run prefix but does not run any
training-candidate refresh or pointer publishing.

Optional command, not part of the trainer launch:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode intake-seed \
  --tournament-id curvy-r18nofb-staticmix-live-20260516a \
  --rating-run-id elo-r18nofb-staticmix-live-20260516a \
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

Do not run `training-candidate-refresh`,
`training-candidate-auto-refresh`, or any controller command that rewrites
control assignment refresh pointers for this control.

## Runtime Isolation Proof

For every launched row, the completed `summary.json` should show:

- `opponent_assignment_refresh.enabled == false`
- `opponent_assignment_refresh.pending_assignment_ref == null`
- `opponent_assignment_refresh.event_count == 0`
- no meaningful `train/opponent_assignment_refresh_events.jsonl`

Status commands after launch:

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

## Signal

Evidence that supports the hypothesis:

- The static rows still improve over iteration 0, but latest checkpoints stay
  near their own best: at least `2/3` rows have latest survival within `10%` of
  best survival.
- Mean best-to-latest drop for the selected static rows is less than half the
  matched live-feedback rows, roughly below `35` steps given the current
  `r18fresh` aggregate.
- Assignment refresh proof remains zero-event for all selected rows.
- Action-collapse flags are absent or materially rarer than the matched live
  rows.

Evidence that falsifies or weakens the hypothesis:

- Static rows show the same shape as the live lane: mid-run best improves, but
  latest regresses by around `50+` steps or more in `2/3` rows.
- Latest is not better than iteration 0 in most static rows.
- Action-collapse appears at a similar rate even though no refresh events fired.

If falsified, the next suspect is not tournament feedback; look at reward/value
support scaling, replay/update ratio, action collapse dynamics, and reward
component ownership.
