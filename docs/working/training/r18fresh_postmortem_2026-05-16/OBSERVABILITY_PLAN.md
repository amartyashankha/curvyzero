# Observability Plan

Last updated: 2026-05-16 14:29 EDT.

This is the r18fresh case-study observability plan. The general contract now
lives in `docs/working/training/curvytron_feedback_loop/OBSERVABILITY_CONTRACT.md`.
The goal is not more logs. The goal is fewer ambiguous questions.

## The System We Need To See

The feedback loop has seven steps:

1. Trainer writes a checkpoint.
2. Subscriber/intake accepts the checkpoint.
3. Tournament schedules and plays games with it.
4. Tournament rating updates rank/status.
5. Trainer-facing export selects frozen opponents.
6. Trainers apply the export and load those exact policies.
7. Learning metrics move after the new opponents are actually used.

If any step cannot be checked from a durable artifact, the loop is not
observable enough.

## Required Artifacts

### Lineage Event Log

The smallest durable implementation is one append-only JSONL stream:
`lineage_events.jsonl`.

Emit one row per boundary transition, not one row per game or env step.

Common fields:

- `event_id`, `observed_at`, `stage`, `status`, `reason`.
- `run_id`, `attempt_id`, `trainer_modal_task_id`.
- `checkpoint_ref`, `checkpoint_id`, `iteration`, `checkpoint_size_bytes`,
  `checkpoint_mtime_ns`.
- `tournament_id`, `rating_run_id`, `round_id`, `round_index`.
- `leaderboard_id`, `snapshot_id`, `generation`.
- `assignment_ref`, `assignment_sha256`, `pointer_ref`,
  `pointer_generation`.
- `source_ref`, `source_sha256`, `output_ref`, `output_sha256`.
- `queue_partition`, `queue_len_before`, `event_count`, `call_id`.
- `counts_json`, `error_type`, `error`.

Closed stage set:

- `checkpoint_written`
- `checkpoint_intake_seen`
- `checkpoint_intake_enqueued`
- `rating_spawn_claimed`
- `rating_round_started`
- `rating_round_reduced`
- `rating_latest_written`
- `leaderboard_published`
- `training_candidate_assignment_written`
- `assignment_pointer_rewritten`
- `trainer_assignment_loaded`
- `trainer_assignment_applied`

Keep large payloads in their existing artifacts. The lineage row stores refs,
hashes, counts, and top-level status only.

## Derived Tables

### Checkpoint Lineage Table

Derived from `lineage_events.jsonl` plus eval/rating artifacts. One row per
checkpoint.

| Field Group | Required Fields | Question It Answers |
| --- | --- | --- |
| identity | run id, attempt id, recipe, reward variant, seed, learner seat mode | Which experiment produced this? |
| checkpoint | ref, iteration, write time, metadata sidecar, observation surface/backend, commit status | Was the checkpoint valid and self-describing? |
| eval | survival mean/p50/p90, own reward, terminal outcome, bonus count/reward, action histogram | Did this checkpoint improve, regress, or collapse? |
| intake | first-seen tick, queued event, accepted/rejected, rejection reason | Did the subscriber pick it up? |
| tournament | first rating round, current status, rank, rating, games, failed games, distinct opponents, seat split | Did it actually play enough fair games? |
| export | export generation ids, selected rank/slot, assignment refs, assignment sha | Did it become trainer-visible? |
| trainer use | runs applied, first iteration applied, provider ok/fail counts, opponent slot/immortal flag | Did trainers actually load it? |

### Tournament Round Card

One row per rating round.

Required fields: round id, input checkpoint count, latest checkpoint count
before/after, raw row count, active/provisional/retired counts, new checkpoint
count, planned pairs/games, completed/failed games, max delta, stable flag,
published latest pointer, export generation, and age of each pointer.

This would have made "71 rows on the website" versus "newer checkpoints are in
a running round but not latest yet" obvious instead of mysterious.

### Export And Assignment Ledger

One row per trainer-facing export generation.

Required fields: source tournament id, source rating id, source round, source
snapshot hash, selected checkpoint refs, selection policy, assignment shas,
pointer old/new refs, expected trainer run count, trainers applied, first
iteration applied, provider-load failures, and lag from rating write to trainer
apply.

This is the proof that the tournament winners came back into frozen employment
for training.

### Operator Health Readout

One CLI command or dashboard header should show:

- current tournament id and rating id.
- current run prefix/category.
- latest completed round and current running round.
- discovered, accepted, queued, rating-input, latest, active, and exported
  checkpoint counts.
- max checkpoint iteration at each stage.
- newest checkpoint age.
- active row count and active cap.
- min/median/max games and distinct opponents for active rows.
- latest trainer export generation and age.
- trainer runs expected/applied and provider-load failures.

This should be the first thing a human sees before debugging individual rows.

## Hook Points

These are the first places to emit lineage events if code work resumes:

- Trainer checkpoint write:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  `_write_checkpoint_progress_latest`,
  `_write_checkpoint_policy_metadata_sidecar`,
  `_write_own_checkpoint_opponent_refresh`.
- Subscriber/intake:
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
  `_intake_manifest_from_discovery`,
  `curvytron_checkpoint_intake_submit`,
  `curvytron_checkpoint_intake_tick`,
  `curvytron_checkpoint_intake_drain`.
- Tournament/rating:
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
  `curvytron_rating_round`,
  `curvytron_rating_loop`;
  `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
  `rating_snapshot_from_pair_results`.
- Leaderboard/export:
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
  `curvytron_opponent_leaderboard_publish`,
  `curvytron_training_candidate_refresh`;
  `scripts/materialize_curvytron_leaderboard_assignment.py`.
- Trainer consumption:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  `_resolve_opponent_assignment_for_env`,
  `_install_lightzero_opponent_assignment_refresh_hook`,
  `_opponent_assignment_refresh_ready_report`.

Do not make lineage writes fatal on the hot path except in explicit canary
mode. Emit best-effort rows and surface lineage failures in run summaries.

## Fair Analysis Rules

- Use matched checkpoint iteration for cross-run comparisons.
- Report first, best-so-far, latest, latest/best retention, and area under the
  survival curve.
- Compare own reward only within a reward variant.
- Compare survival and tournament rank separately; tournament rank is relative
  head-to-head strength, not raw duration.
- Dedupe tournament seeds by run or setting unless the goal is explicitly a
  champion-heavy curriculum.
- Treat latest-only analysis as operational health, not learning quality.

## Future-Launch Acceptance Criteria

The `cz26-full-20260517a` launch already happened, so these are now
future-launch gates and current post-launch validation gaps. Before another
large launch:

1. Preflight prints current app, Volume, Dict, Queue, tournament, rating, run
   prefix, manifest rows, recipe ids, reward variants, learner seat mode,
   checkpoint cadence, assignment refresh interval, and confirms all active
   storage is v2.
2. A canary on the same code path proves:
   checkpoint written -> intake accepted -> tournament rated -> export written
   -> assignment applied -> provider-load rows show the exact checkpoint.
3. Every planned trainer row emits learner metrics, checkpoint metadata, eval
   survival/reward/action rows, assignment SHA, learner seat counts, and
   provider-load rows.
4. Every tournament internal game batch emits the game-batch card above.
5. Every export emits the export ledger above.
6. The dashboard or CLI health readout can explain row-count differences,
   stale/latest game-batch lag, and whether new checkpoints are waiting,
   playing, active, retired, or missing.

If these are not true, launch only a deliberately scoped canary/control run.

## What r18fresh Could Not Answer In Real Time

- Whether every checkpoint moved through the whole loop without hand-joining
  artifacts.
- Whether recipe labels matched actual opponents after scratch bootstrap and
  assignment refresh.
- Why trainer/public leaderboard generation lagged rating rounds.
- Whether the top run dominated because it was genuinely stronger, duplicated
  many sibling checkpoints, received better matchup exposure, or all three.
- Whether retention failure came from tournament feedback, reward scaling,
  optimizer instability, opponent refresh, or ordinary self-play instability.
- Whether dashboard row counts represented raw rating history, active trainer
  rows, current game-batch input, or published exports.

These are now explicit observability requirements, not mysteries to rediscover.

## Dashboard Traps To Avoid

- Raw rating row count is not the trainer-facing top-N export.
- "Current" is meaningless unless the page names the current tournament,
  rating, run prefix, export generation, and selected URL filters.
- Rating round and export generation are different clocks.
- GIF presence does not prove a checkpoint was rated, active, or exported.
- Latest checkpoint, best-so-far checkpoint, and tournament-selected checkpoint
  are different objects.
- Dashboards explain Volume artifacts; they do not replace refs, hashes, and
  immutable snapshots as truth.
