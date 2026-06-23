# Observability Contract

Last updated: 2026-05-17.

This is the general observability contract for all future CurvyTron batches.
The r18fresh postmortem is one case study that motivated it.

## Required Artifact

Emit a best-effort append-only `lineage_events.jsonl` with one row per boundary
transition. Do not turn this into per-game or per-env-step logging.

The helper lives in `src/curvyzero/observability/feedback_loop_lineage.py`.
As of the current local code, every required stage below has a call site. A
focused local lifecycle proof now creates synthetic checkpoints and proves the
chain through intake, rating, leaderboard publish, trainer assignment refresh,
trainer assignment load, and trainer assignment apply with matching refs and
assignment SHA. The remaining proof is deployed: the same chain must be observed
in current Modal Volume/Dict state after a canary run.

Required stages:

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

Required common fields:

- event id, timestamp, stage, status, reason.
- run id, attempt id, trainer task id.
- checkpoint ref/id/iteration/size/mtime.
- tournament id, rating id, internal game-batch id/index.
- leaderboard id, snapshot id, generation.
- assignment ref, assignment SHA, pointer ref, pointer generation.
- source/output refs and hashes.
- queue partition, queue length, event count, call id.
- compact counts, error type, error.

## Derived Readouts

- Checkpoint lifecycle table: one row per checkpoint showing written, intake,
  tournament, export, assignment, trainer-load, and learning metrics.
- Tournament game-batch card: one row per internal game batch with raw/active/provisional/retired
  rows, pairs/games, stable flag, max delta, latest pointer, and export age.
- Tournament scheduling card: requested pairs, coverage floor pairs, actual
  scheduled pairs, budget expansion factor, schedule reasons, distinct-opponent
  percentiles, outside-run opponent percentiles, anchor concentration,
  zero-appearance rows, repeat share, graph components, and top-100 boundary
  churn.
- Export ledger: one row per trainer-facing export generation with selected
  refs, assignment SHAs, trainer apply count, provider-load OK/fail, and lag.
- Operator health readout: one command or dashboard header naming the current
  arena/rating/run prefix and showing row counts, newest checkpoint stage,
  latest round, latest export, stale flags, and trainer consumption.

## Tournament Progress Readout Rule

The operator readout must not mix up three different ideas:

- Liveness sample: a small scan that proves some game output exists. Label this
  as sampled/liveness only.
- Root game-batch summary: the aggregate fields written after reduce. Before
  reduce, these can be zero or absent even when per-game outputs exist.
- Total or sampled completion progress: explicit counts of expected games,
  observed game summaries, completed pairs, partial pairs, newest output time,
  and whether the count is exhaustive or sampled.

For the live cz26 batch on 2026-05-17, `probe_completed_game_count=21` only
meant "one narrow liveness probe found outputs." It was not total completed
games. Future status output should encode that directly so operators do not
have to infer it from raw Modal logs or Volume files.

## Launch Gate

The local synthetic proof is now present. The deployed canary
`curvy-e2e-current-contract-live-20260516a` is in progress: two current-code
trainer checkpoints are in intake and rating `round-000000` has been spawned.
No large batch should launch until this canary shows one fresh checkpoint moving
through Modal Volume/Dict state from write to trainer provider load, with the
same assignment SHA visible at export and consumption.
