# Modal Feedback Loop Debugging Follow-Up, 2026-05-16

Scope: focused research pass only. No production code edits.

Question: how do we prove the Modal feedback loop is autonomous, complete, and
non-stalling?

Target loop:

```text
trainer checkpoint
-> subscriber/intake manifest
-> tournament rating round
-> leaderboard/training-candidate snapshot
-> assignment/control pointer
-> same running trainer refresh + provider-ok env rows
```

## Official Modal Caveats

- Deployed apps persist until stopped; non-detached `modal run` creates an
  ephemeral app, and ephemeral apps stop when the local caller exits unless run
  with `--detach`. Use deployed services for long autonomous work; use detached
  ephemeral apps only as explicit proof/dev lanes.
  Source: <https://modal.com/docs/guide/apps>
- `modal app logs` can be filtered by app, time window, search text, function
  id, function-call id, source, and container id. This is the first log query
  surface for "did the autonomous service wake up after I walked away?"
  Source: <https://modal.com/docs/reference/cli/app>
- Background work should be submitted with `.spawn()` when the caller should not
  block. Persist the returned `FunctionCall` id. `FunctionCall.get(timeout=0)`
  polls immediately; results are available for a limited post-completion window.
  Sources: <https://modal.com/docs/guide/job-queue>,
  <https://modal.com/docs/reference/modal.FunctionCall>
- Modal Queues are good coordination/wakeup channels, not durable truth. The
  default partition lifetime is 24h after the last put; queue items and
  partitions have limits. The durable record must be Volume artifacts.
  Sources: <https://modal.com/docs/guide/queues>,
  <https://modal.com/docs/reference/modal.Queue>
- Modal Dict is useful for pointers, active keys, claims, and operator intent.
  Do not make it the only historical record; keep reconstructable Volume files.
  Source: <https://modal.com/docs/reference/modal.Dict>
- Volume writes become visible to other containers after commit/reload
  boundaries. Concurrent writes to the same file are last-writer-wins, and too
  many concurrent commits contend. Avoid shared hot JSON writers; write
  immutable files first and update one small pointer last.
  Source: <https://modal.com/docs/guide/volumes>
- Volume reload can fail when files are open, and during reload the initiating
  container sees the Volume as empty. Readers and websites should report stale
  data with timestamps on reload failure, not mutate progress state.
  Source: <https://modal.com/docs/guide/volumes>
- Deployed app container crashes are retried indefinitely with crash-loop
  backoff. That is useful for service continuity but can hide repeated bugs
  unless logs/dashboards are checked.
  Source: <https://modal.com/docs/guide/retries>

## Current Repo Surfaces

Active app/storage contract, from local docs:

- Tournament app: `curvyzero-checkpoint-tournament-v2`
- Trainer app: `curvyzero-lightzero-curvytron-visual-survival-train-v2`
- Runs Volume: `curvyzero-runs-v2`
- Tournament Volume: `curvyzero-curvytron-tournaments-v2`
- Control Volume: `curvyzero-curvytron-control-v2`
- Intake Dict: `curvyzero-curvytron-checkpoint-intake-v2`
- Intake Queue: `curvyzero-curvytron-checkpoint-events-v2`
- Public leaderboard Dict: `curvyzero-curvytron-opponent-leaderboard-live-v2`

Current proven lane to use as a template:

- `FULL_LOOP_PROOF.md` records live-batch proof through scheduled generation 12:
  checkpoints reached tournament, scheduled controller wrote fresh assignments,
  and every still-running trainer consumed the new assignment shas with
  `opponent_provider_load_ok=true`.
- `TOURNAMENT_DEBUG.md` records a separate dirty/stress caveat: stale
  `latest.json` can lag a running round, and root progress from polluted lanes
  is not enough proof. Persisted round input and final `latest.json`/ratings are
  the truth.
- `survival_stagnation_investigation_2026-05-16.md` records the learning-health
  split: wiring is mostly proven, while survival retention is mixed.

## Concrete Observability Commands

Set ids first:

```bash
T=curvy-r18fresh-live-bounded-dsf1-20260516b
R=elo-r18fresh-live-bounded-dsf1-20260516b
TVOL=curvyzero-curvytron-tournaments-v2
RVOL=curvyzero-runs-v2
CVOL=curvyzero-curvytron-control-v2
INTAKE_DICT=curvyzero-curvytron-checkpoint-intake-v2
INTAKE_QUEUE=curvyzero-curvytron-checkpoint-events-v2
TOURN_APP=curvyzero-checkpoint-tournament-v2
TRAIN_APP=curvyzero-lightzero-curvytron-visual-survival-train-v2
OUT=/tmp/curvy-loop-debug
mkdir -p "$OUT"
```

Check app autonomy and logs:

```bash
uv run --extra modal modal app list --json
uv run --extra modal modal app history "$TOURN_APP" --json
uv run --extra modal modal app logs "$TOURN_APP" --since 2h \
  --timestamps --show-function-call-id --show-container-id
uv run --extra modal modal app logs "$TRAIN_APP" --since 2h \
  --timestamps --show-function-call-id --show-container-id
uv run --extra modal modal app logs "$TOURN_APP" --since 2h \
  --search training_candidate --timestamps --show-function-call-id
uv run --extra modal modal app logs "$TOURN_APP" --since 2h \
  --search intake --timestamps --show-function-call-id
```

Check tournament durable truth:

```bash
BASE=tournaments/curvytron/$T
uv run --extra modal modal volume get "$TVOL" \
  "$BASE/intake/$R/config.json" "$OUT/intake_config.json"
uv run --extra modal modal volume get "$TVOL" \
  "$BASE/intake/$R/progress.json" "$OUT/intake_progress.json"
uv run --extra modal modal volume get "$TVOL" \
  "$BASE/ratings/$R/config.json" "$OUT/rating_config.json"
uv run --extra modal modal volume get "$TVOL" \
  "$BASE/ratings/$R/latest.json" "$OUT/rating_latest.json"
uv run --extra modal modal volume get "$TVOL" \
  "$BASE/ratings/$R/progress.json" "$OUT/rating_progress.json"
ROUND=$(jq -r '.round_id // .latest_round_id // empty' "$OUT/rating_latest.json")
uv run --extra modal modal volume get "$TVOL" \
  "$BASE/ratings/$R/rounds/$ROUND/input.json" "$OUT/round_input.json"
uv run --extra modal modal volume get "$TVOL" \
  "$BASE/ratings/$R/rounds/$ROUND/ratings.json" "$OUT/round_ratings.json"
```

Check intake Dict/Queue. `scripts/curvytron_tournament_debug_bundle.py` can
derive exact keys and emits command hints; use it whenever possible:

```bash
uv run python scripts/curvytron_tournament_debug_bundle.py \
  --tournament-id "$T" --rating-run-id "$R" \
  --include-modal-hints
```

The common direct shape is:

```bash
MANIFEST_KEY="$T:$R"
QUEUE_PARTITION="$(python - <<'PY'
from scripts.curvytron_tournament_debug_bundle import intake_queue_partition
print(intake_queue_partition("curvy-r18fresh-live-bounded-dsf1-20260516b", "elo-r18fresh-live-bounded-dsf1-20260516b"))
PY
)"
uv run --extra modal modal dict get "$INTAKE_DICT" "$MANIFEST_KEY"
uv run --extra modal modal queue len "$INTAKE_QUEUE" --partition "$QUEUE_PARTITION"
uv run --extra modal modal queue peek "$INTAKE_QUEUE" --partition "$QUEUE_PARTITION" --n 5
```

Check trainer consumption of fresh assignments:

```bash
RUN_IDS=comma,separated,current,run,ids
ATTEMPT_IDS=comma,separated,current,attempt,ids
SHAS=comma,separated,assignment,shas
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$RUN_IDS" --attempt-ids "$ATTEMPT_IDS" \
  --output assignment-proof-json --target-assignment-shas "$SHAS" \
  > "$OUT/assignment_proof.json"
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$RUN_IDS" --attempt-ids "$ATTEMPT_IDS" \
  --output eval-summary
```

Minimum full-loop proof fields:

- trainer: exact nonzero checkpoint ref exists in `curvyzero-runs-v2`;
- intake: durable manifest count increased and includes that ref;
- queue: queue may be empty, but manifest/pool must still include desired refs;
- tournament: latest or current round input includes that ref;
- rating: completed round wrote `ratings.json`/`latest.json`;
- leaderboard/controller: new training-candidate snapshot and assignment shas;
- trainer refresh: `decision=applied` event count increased;
- env telemetry: rows with target assignment sha and
  `opponent_provider_load_ok=true`;
- learning: survival/action metrics are separate, not implied by wiring proof.

## Failure Modes To Check First

| Symptom | Likely failure | Fast proof |
| --- | --- | --- |
| App looks submitted but nothing advances | Non-detached ephemeral parent died, or child work was only scheduled | `modal app logs`; check completed game summaries and `latest.json`, not round input alone |
| Website row count is stale | Web container cache/reload issue or latest snapshot lags running round | Direct `modal volume get` of round input/progress/latest |
| Queue length is zero but refs are missing | Queue event was lost/expired or bookkeeping is wrong | Rebuild from durable intake manifest; Queue is not the source of truth |
| Old partial pool blocks larger pool | Claim key too broad or stale claim not expired/repaired | Inspect claim Dict key; verify claim identity includes pool hash/mode and stale timeout |
| Trainers do not pick up new opponents | Controller did not rewrite pointers, refresh disabled, or refresh boundary not reached | Inspect control pointer, assignment sha, refresh events, env telemetry |
| Provider load false | Assignment points at missing checkpoint ref or wrong Volume prefix | Audit exact refs in `curvyzero-runs-v2`; require immutable `iteration_N.pth.tar` |
| Ratings exist but do not cover the intended pool | Reducer ran on a smaller/older round | Compare intake manifest refs, round input refs, latest roster refs, and pool hashes |
| Progress file is malformed/half-written | Direct overwrite of hot JSON or concurrent writer | Prefer atomic temp+replace for future writes; treat read error as stale observation |
| Volume reload error stalls UI/subscriber | Reload attempted with open files or during active IO | Record reload error visibly; keep serving stale cached data; retry later |
| Survival falls while loop is healthy | Learning/curriculum issue, not control-plane issue | Run fixed-grid eval, no-refresh control, target/support audit, action-collapse audit |

## Suggested Toy Experiment Shape

Use two toy lanes in parallel: one real-tiny lane and one fake-fast control.

1. Real-tiny autonomous lane:
   - Deploy current trainer/tournament apps.
   - Start 2-3 tiny trainers with frequent checkpoints and assignment refresh
     enabled.
   - Start an intake watch by run id/prefix, not only exact refs.
   - Let scheduled subscriber/drain/controller handle it; do not manually feed
     every checkpoint.
   - Sleep long enough for at least one new checkpoint cadence.
   - Prove: new checkpoint ref -> manifest -> round input/latest -> assignment
     sha -> same trainer env row.

2. Fake-fast control lane:
   - Use a tiny writer function that writes small fake checkpoint marker files
     plus sidecars to a test Volume path every few seconds.
   - Point the subscriber/intake scanner at that prefix if it can accept marker
     refs, or run a minimal tournament over 3-5 real tiny checkpoints if the
     policy loader requires real checkpoints.
   - Inject failure cases: duplicate checkpoint submit, missing Queue event,
     stale claim, killed subscriber container, and one bad checkpoint ref.
   - Expected result: bad refs are skipped or quarantined, good refs continue,
     claims expire/repair, and the manifest never shrinks.

3. Control/no-tournament learning lane:
   - Keep the currently launched static/no-feedback rows as the learning-health
     comparison.
   - This lane is not a loop proof. It answers whether survival regression is
     caused by live tournament nonstationarity or by local MuZero/reward/cadence
     settings.

Proof rule for all toy lanes: the local shell may disappear. A lane is
autonomous only if deployed/scheduled functions or detached calls continue
advancing durable artifacts after the local caller exits.

## Practical Design Rules

- Do not make the operator part of the steady-state loop. Manual calls are
  allowed for smoke tests, but the production proof is scheduled subscriber,
  scheduled/intake drain, scheduled controller, and trainer refresh telemetry.
- Every stage should be idempotent. Duplicate checkpoint events should not
  create duplicate pool rows; missing Queue events should be rebuildable from
  the manifest; failed games should not block unrelated pairs forever.
- Use immutable files for truth:
  `iteration_N.pth.tar`, round input/results/ratings, leaderboard snapshot,
  assignment JSON, refresh event JSONL. Use latest pointers only after the
  immutable file is written and committed.
- Scope claims by exact work identity: tournament id, rating id, mode, pool
  hash, round id, shard id, and generation where relevant. A claim for 10 refs
  must not block a later claim for 100 refs.
- Record join keys everywhere: run id, attempt id, checkpoint ref, tournament
  id, rating run id, round id, pool hash, assignment sha, FunctionCall id,
  generation, and artifact refs.
- Treat survival improvement as a separate proof. The control plane can be
  correct while the policy regresses.

## Immediate Follow-Up Checklist

- Run the command set above against the current bounded lane and archive the
  pulled JSON under `/tmp/curvy-loop-debug`.
- Check scheduled controller logs after at least one checkpoint interval; prove
  a generation advanced without an operator-triggered controller call.
- For one fresh checkpoint ref, write a one-row trace across all six stages.
- Keep the static/no-feedback control running until at least two nonzero eval
  points per row; compare latest-vs-best retention to the live-feedback rows.
- If any stage stalls, repair the smallest contract: rebuild from Volume truth,
  expire only the stale claim, quarantine only bad refs, and keep the rest of
  the pipeline moving.
