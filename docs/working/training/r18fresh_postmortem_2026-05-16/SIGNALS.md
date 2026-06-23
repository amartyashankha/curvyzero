# Signal Inventory

Use this file to decide what to measure and where to get it.

General feedback-loop observability requirements now live in
`docs/working/training/curvytron_feedback_loop/OBSERVABILITY_CONTRACT.md`.
This file keeps r18fresh-specific commands, paths, and measurements.

## Retrospective Question

The last batch was hard to interpret because we had many nearby signals but no
single stitched readout. The next batch should answer these questions without a
human manually chasing artifacts:

- Did each trainer produce checkpoints on schedule?
- Did each checkpoint enter intake and get scheduled by the tournament?
- Did the tournament actually run enough games for that checkpoint?
- Did the checkpoint improve in tournament rank/Elo/head-to-head survival?
- Did the trainer-facing export include the tournament winners?
- Did running trainers apply that export and load those exact frozen opponents?
- Did learning improve by survival, own reward, reward components, and action
  distribution, or did it only find a mid-run peak and regress?

If a readout cannot answer those seven questions, it is not enough.

## Training And Eval Signals

- Eval survival at matched checkpoint iterations.
- Eval reward within reward variant.
- Eval reward breakdown: `mean_training_reward`, `mean_reward_components`,
  `mean_bonus_reward`, `mean_bonus_pickup_count`, and `outcome_histogram`.
- Best-so-far eval survival.
- Retention from best-so-far to latest/common endpoint.
- Checkpoint count and latest checkpoint iteration.
- Learner metrics, if present: reward/value losses, support saturation,
  entropy/action mix, and reward components.

Required per-run/per-checkpoint columns for next batch:

- run id, attempt id, recipe id, reward variant, seed, learner seat mode.
- checkpoint ref, iteration, write time, sidecar metadata presence, policy
  observation surface, and checkpoint publish/commit status.
- eval survival mean/median/p10/p90, own training reward, terminal outcome
  rate, bonus pickup count, bonus reward, and action histogram.
- best-so-far survival, latest survival, retention ratio
  `latest / best_so_far`, and best checkpoint ref.
- opponent assignment sha used during the eval window, assignment generation,
  opponent mixture entry name, opponent immortal flag, provider-load-ok count,
  and provider-load-fail count.
- learner seat counts for player 0 and player 1, so perspective randomization
  is visible rather than assumed.

Primary commands:

```bash
R18_RUN_IDS=$(jq -r '.rows[] | .run_id' artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json | paste -sd,)

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$R18_RUN_IDS" \
  --output eval-summary
```

Raw eval JSON:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$R18_RUN_IDS" \
  --output eval-json
```

## Tournament Signals

- Internal rating rows: all historical rated checkpoints, active and retired.
- Active rows: top 100 by current tournament status.
- Top 10 or top 100 trainer-facing bootstrap candidates.
- Rank/Elo history by checkpoint.
- Games and distinct opponents per row.
- Tournament game duration from battle summaries.
- Stable flag and max delta.

Required per-checkpoint tournament columns for next batch:

- checkpoint id/ref/run/iteration and first-seen round.
- latest status: active, provisional, retired, or missing.
- current rank, rating, rating delta from first entry, games, valid games,
  failed games, distinct opponents, and seat split.
- tournament survival/duration mean by game and by opponent band.
- last scheduled round, last completed round, and whether the checkpoint is
  currently starved, pending, or fully rated.
- export generation(s) where this checkpoint appeared in trainer-facing
  assignments.

Required per-round tournament columns:

- round id, roster size, active/provisional/retired counts, new checkpoint
  count, pair count, game count, completed games, failed games, max delta,
  stable flag, and published latest pointer.
- for adaptive rounds: active pool limit, pairs per round, checkpoint coverage
  percent, and top-N distinct-opponent coverage.

Live operator readout must also include `pool_status`:

- intake checkpoint count;
- latest published rating checkpoint count;
- active game-batch checkpoint count;
- active game-batch embedded rating-spec checkpoint count;
- active game-batch roster count;
- whether the active batch is larger than latest rating;
- whether the active batch appears to cover only the already-rated pool.

This is the guard against mistaking "games are running" for "new checkpoints
are being rated."

Current truth paths:

```text
tournaments/curvytron/cz26-live-20260517a/ratings/elo-cz26-live-20260517a/latest.json
tournaments/curvytron/cz26-live-20260517a/ratings/elo-cz26-live-20260517a/rounds/<game_batch_id>/ratings.json
tournaments/curvytron/leaderboards/cz26-live-20260517a-elo-cz26-live-20260517a-training/latest.json
```

First-line current status command:

```bash
uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 64
```

The smaller `12` lookahead missed a far-ahead blocking game-batch artifact.

Do not use `modal run ... --mode loop-status` for live control. It creates a
temporary scheduled Modal app. Use the deployed-function script above.

First-line trainer-consumption proof command:

```bash
uv run --extra modal python scripts/curvytron_live_loop_control.py --action trainer-proof --activity-probe-pairs 0 --run-limit 0
```

By default this prints summary counts only. Add
`--assignment-proof-row-limit -1` only when the full per-run rows are needed.

Latest proof on 2026-05-17 05:35 EDT:

- `136` runs scanned.
- `24` generation-4 target assignment SHAs.
- `48` runs had a target SHA as latest applied.
- `43723` target provider-ok env rows.
- `0` target provider-false env rows.

This proves partial trainer consumption. It does not prove every trainer has
refreshed yet.

## Full-Loop Signals

The full loop is only proven when the same lineage can be followed through:

1. Trainer writes a checkpoint.
2. Subscriber/intake accepts it.
3. Tournament schedules and rates it.
4. Leaderboard publisher writes a trainer-facing snapshot.
5. Training-candidate controller writes assignments and pointers.
6. Trainer applies the assignment SHA.
7. Env telemetry shows provider loads from that assignment.

Next-batch full-loop table should have one row per export generation:

- source tournament/rating round id and latest snapshot hash.
- export generation id, assignment refs, assignment shas, recipe ids, and top
  checkpoint refs selected.
- trainer run count expected, run count still alive, run count applied, first
  train iteration applied, and post-apply env-row count.
- frozen checkpoint provider-ok rows and provider-load-fail rows.
- lag from tournament latest write to trainer assignment application.

Historical proof command shape:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$R18_RUN_IDS" \
  --output assignment-proof-json \
  --assignment-proof-tail-bytes 134217728
```

For the current CZ26 lane, prefer the deployed-function script above so we do
not create temporary scheduled Modal apps.

## Dashboard Signals

Dashboards are useful for quick inspection, but not primary truth:

- Tournament website can show hundreds of raw rating rows.
- GIF browser can show many archived runs unless filtered.
- Public/trainer leaderboard active rows are selected by status/rank, not by
  website page length.

If a dashboard disagrees with deployed status/control tooling, trust the
deployed operator tooling first. Use Modal Volume artifacts as backing evidence
when debugging the disagreement.

Dashboard readouts that would have prevented past confusion:

- Tournament page header: current tournament id, rating id, latest game-batch id,
  raw rated row count, active trainer-facing cap, active row count, max
  checkpoint iteration, newest checkpoint age, and export generation age.
- Per-row badges: `active/provisional/retired`, games, distinct opponents,
  first-seen round, last-played round, and whether the row has been exported to
  trainer assignments.
- A "pipeline health" strip: discovered refs, queued events, scheduled games,
  completed games, latest snapshot round, export generation, and trainers that
  applied the latest shas.
- GIF/browser pages must show which run-prefix/category is selected in the URL
  and header; hidden filters should be impossible.

## Minimum Next-Batch Preflight

Before launching a large batch, run or build a command that prints:

- current app/Volume/Dict/Queue names and confirms all are v2.
- current tournament/rating ids and whether their artifact roots are clean.
- manifest rows, reward variants, recipes, learner seat mode, checkpoint cadence,
  and assignment refresh interval.
- expected observability files for trainer metrics, eval summaries,
  checkpoint sidecars, tournament latest, export latest, and assignment proof.
- whether a small canary has closed the full loop with the same code path.

## Critique Lanes To Run Before Next Launch

Use parallel critiques for different failure modes, then integrate the results
into `FINDINGS.md` and `TODO.md`:

- Trainer observability: can we explain reward, survival, action mix, learner
  seat, opponent assignment, and provider-load changes at each checkpoint?
- Tournament observability: can we explain for every checkpoint whether it is
  missing, pending, provisional, active, retired, starved for games, or fully
  rated?
- Experiment design: are comparisons matched by reward axis, opponent recipe,
  noise mode, seed/checkpoint count, and wall-clock maturity?
- Operator surface: can a human open the dashboard or run one CLI command and
  know which arena is current, whether the subscriber is moving, and whether
  trainers have consumed the latest export?
- Cleanup/runtime inventory: can we list what is current, preserved, archived,
  and safe to kill without relying on memory?
