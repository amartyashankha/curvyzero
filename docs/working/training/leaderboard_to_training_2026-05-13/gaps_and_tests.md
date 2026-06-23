# Gaps And Tests

For plain-language definitions of the repair, continuation, validation,
assignment, and refresh terms used here, see `README.md`.

## Highest Priority Gaps

| Gap | Why it matters | First test |
| --- | --- | --- |
| Live Dict pointer repair | Publisher writes pointer, but Modal Dict is not durable truth. | Local repair command repopulates missing/stale Dict from validated immutable Volume snapshot; tiny remote smoke passed; minimal operator runbook added. |
| Assignment command path | Assignment writer works, but we need one clear command path. | Command stores `assignment.json` and `audit.json` under a training attempt and records the returned refs. |
| Run-scoped slot recipe control | Operators should be able to change a run's desired slots after launch without changing trainer code. | Dict recipe validator plus materializer smoke writes a new immutable assignment and audit for one run id. |
| Run-scoped reward recipe control | Operators should be able to choose survival, bonus, and final-outcome reward weights per run. | Pure validator maps named profiles to current reward variants and stores a frozen reward recipe/hash in launch or new-attempt artifacts. |
| Online tournament continuation on Modal | New checkpoints need to enter an existing tournament without resetting evidence. | Small bounded Modal proof passed: old 3 -> new 3, continued to round 1, wrote 6 rating rows. Next proof should use the deployed service shape and a larger active pool. |
| Queue/claim repair on Modal | Queue and claim state can be missing or stale. | Small bounded Modal proof passed: empty Queue was rebuilt from manifest and a forced stale claim was replaced. Next proof should avoid forced takeover and show normal expiry/retry. |
| Checkpoint id stability | Old rating evidence is keyed by checkpoint id. If ids change when new refs arrive, old ratings and pair history can detach from the same checkpoint. | Local regression now preserves old ids from `latest.json` when explicit continuation specs add new checkpoint refs. |
| Dict manifest loss | Modal Dict can forget coordination state; Volume manifest is the durable truth. | Local regression now rebuilds the Dict manifest from the Volume manifest during drain when a tournament/rating id is supplied. |
| Partial-pool claim blocking | A claim for a small player pool must not block a later larger player pool. | Local regression now keys rating claims by desired checkpoint-pool hash and forces live run watches into continuation mode. |
| Modal launch lifetime | Background tournament game/rating workers can die when a non-detached `modal run` parent exits. | Remote proof must use `modal run --detach`, a deployed scheduled function that keeps work alive, or a parent command that waits for children; then verify `latest.json` and completed game summaries. |
| Website reload busy errors | A failed Volume reload must not make the website think it is fresh. | Local regression now retries after a busy/error reload instead of throttling the next attempt. |
| One-frame tournament parity | New leaderboard must match current train cadence. | Rating spec with `decision_source_frames=1` is recorded, hashed, and used in game summaries; tiny two-checkpoint remote rating/publish smoke passed. |
| Stable-slot recency smoke | `recent_strong` only means recent if rating rows carry run/attempt/iteration/latest metadata. | Rerun a bounded multi-checkpoint smoke and verify `recent_strong` selects the latest checkpoint for the watched run, not only the best remaining row. |
| Larger bounded closed-loop smoke | Canary-scale proof exists, but scale, repair behavior, and non-diagnostic gates still need evidence. | Run a bounded multi-checkpoint smoke and verify publish -> assignment -> same-trainer refresh -> provider-ok env rows still works after metadata repair. |
| Running trainer assignment refresh | Proven at canary scale; production runs still need monitoring. | Keep fake-LightZero collector-boundary and env reset-mixture tests, and require same-trainer `decision=applied` plus provider-ok env rows for every production proof. |
| Seeded non-checkpoint players | Scripted/hand-coded baselines need roster identity if included. | Normalize scripted player specs without fake checkpoint refs. |
| Fractional invincibility overlay | Some percentage of episodes may need invincible opponents regardless of base policy. | Deterministic selection applies public `opponent_immortal=true` as an overlay and records telemetry/audit; env-bound `opponent_death_mode=immortal` is derived runtime plumbing. |

## Existing Tests To Reuse

- `tests/test_opponent_registry.py`
- `tests/test_curvytron_checkpoint_tournament.py`
- `tests/test_curvytron_tournament_scheduler_guardrails.py`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`

## Testing Strategy

Keep the tests layered and honest:

1. Pure contract tests for snapshot, assignment, scheduler, and intake helpers.
2. Local Modal-function tests for publisher, intake, drain, assignment writer,
   website readers, and trainer/poller command wiring.
3. Tiny remote smokes for cross-Volume handoff.
4. Bounded larger smokes only after repair and idempotency paths exist.

Scheduler fairness does not mean uniform pairing. For the Elo service, fairness
means:

- new or provisional checkpoints get enough placement games and distinct
  opponents before top-policy bias dominates;
- higher-rated policies are sampled more often, but lower-ranked or uncertain
  policies still appear;
- a batch of new strong checkpoints should not be judged only against the same
  established players and then mostly dropped because of random noise;
- when a large new batch arrives, some new-vs-new games are useful after each
  entrant has enough established-opponent placement signal;
- repeated comfortable pairs are de-prioritized when fresh useful pairs exist;
- duplicate pairs are blocked within a round;
- some cross-band games remain so scalar Elo does not hide non-transitive
  matchups.

Do not treat local tests or tiny Modal smokes as enough. The useful proof is:
add candidates to a real Modal-backed tournament, see games start in parallel,
watch ratings continue from old evidence, and recover from broken Queue/claim
state.

The first small Modal proof passed on 2026-05-14, but it is not the final shape:
it used manual `modal run` commands and forced stale-claim takeover. The next
proof should run from the deployed app path and should not need manual repair
except for the specific failure mode being tested.

Launch-lifetime rule: do not call a round healthy just because input/progress
files were written. If a local `modal run` returns before child game workers
finish, use `modal run --detach` or wait for the children. Success means
`latest.json` advanced and completed game summaries exist.

## Current Online Tournament Proof Target

Plain desired behavior:

- Keep an active pool of about 100 mature checkpoints.
- When a new checkpoint arrives, schedule games against many active checkpoints.
- If 10 new checkpoints arrive together, schedule all of their games together.
- The games should be separate Modal calls as much as possible.
- The next rating snapshot must include old rows, old pair history, and the new
  evidence.

Local proof already added:

- 2 new checkpoints against 100 established checkpoints get 20 placement
  opponents each in one generated round.
- 10 new checkpoints against 50 established checkpoints get established
  opponents first; no new-vs-new placement pairs are needed for the first
  placement burst.

Bounded first proof:

- Use a small active pool first, such as 10 existing checkpoints.
- Add 2 new checkpoints.
- Ask for 20 distinct opponents per new checkpoint if possible.
- Use an odd `games_per_pair`, such as 3 or 21 depending on speed.
- Check the generated pair list before trusting the run.
- Then run the same path on Modal and inspect progress artifacts.

Small proof that passed:

- 3 existing checkpoint refs were rated first.
- 3 new checkpoint refs were submitted through intake.
- Queue-loss repair rebuilt the missing events from the manifest.
- Stale-claim repair allowed the drain to take over.
- The continued rating wrote round 1, not another round 0.
- The final snapshot kept the old rows and added the new rows.

What that proof does not prove:

- no overlap between scheduled drains and manual drains;
- no non-detached `modal run` parent killing child workers after scheduling;
- no stale website/provisional snapshot issue;
- no production-scale fanout;
- no deployed-service lifecycle;
- no checkpoint-id reorder case on Modal. That case is now covered locally and
  should be included in the next remote proof.

Follow-up local repairs now added:

- Missing Dict manifest can be rebuilt from the Volume manifest for explicit
  tournament/rating ids.
- Rating claims use the desired checkpoint-pool hash, so a stale small-pool
  claim does not block an expanded-pool continuation. Claim keys also include
  fresh-vs-continuation mode, so a fresh-run claim does not block an explicit
  continuation.
- Live run watches force continuation during drain and spawn, even if the
  original rating defaults did not say `continue_from_latest=true`.
- Website reload errors no longer update the success throttle.

Terms:

- **Queue loss**: the manifest says a checkpoint should be processed, but the
  Queue has no event. The drain should recreate the event.
- **Stale claim**: a previous drain claimed the right to start a rating job but
  did not finish. A later drain should take over after the timeout.
- **One-frame validation**: rating games record `decision_source_frames=1`, so
  the tournament is not using an old slow timing setup.

## New Test Groups

### Scheduler Selection

- done: explicit `all_pairs` enumerates unordered non-self pairs exactly once;
- done: `random` selection is seeded, budgeted, and duplicate-free within a
  round;
- done: `adaptive_v0` covers new checkpoints, undercovered checkpoints,
  opponent-id evidence, placement-before-top-band ordering, and top-band bias
  without starving the lower half in a synthetic round;
- done: multi-round synthetic test proving placement floors are reached before
  top-band repeats dominate;
- done: pair-history freshness test proving a useful unplayed pair is preferred
  over a repeatedly played near-rating pair;
- done: active-pool limit defaults to top 100, mature tail rows are marked
  `retired`, retired rows are excluded from adaptive scheduling, and unplaced
  tail rows stay `provisional` instead of being retired early;
- done: new checkpoint batches prefer established active opponents in placement
  before spending placement budget on new-vs-new pairs;
- next: seat-balance policy for repeated canonical pairs;

### Public Leaderboard Contract

- done for pure builder in `tests/test_opponent_leaderboard.py`;
- done for tournament-side publisher local coverage and remote smoke;
- done: training-facing publish rejects snapshots with no active rows;
- done: explicitly allowed no-active snapshots are diagnostic-only and do not
  move `latest.json` or the live `current:` pointer;
- done: repair/fallback command for stale or missing Dict pointer rebuilds from
  immutable snapshot files and refuses unproven `latest.json`;
- done: reject legacy or non-one-frame rating contexts for the public training
  leaderboard unless explicitly labeled legacy.
- done: pointer compact summary separates `active_count`, `provisional_count`,
  and `retired_count` so retired rows are not misreported as provisional.

### Assignment Materializer

- done for pure top-slots v0 in `tests/test_opponent_leaderboard.py`;
- done for writing assignment/audit artifacts under a training attempt in the
  smoke path;
- decision: purge `slot_rules_v0` as the production direction;
- done: implement and test `stable_slots_v1` as the Coach-owned materializer;
- done: verify `stable_slots_v1` uses nested `recency.latest_for_run`;
- done: tournament/intake rating rows preserve run id, attempt id, iteration,
  mtime, and `latest_for_run` so `recent_strong` has real metadata;
- done: explicit checkpoint refs and rating continuation preserve that metadata
  instead of thinning old checkpoints back to bare refs;
- done: dedupe checkpoint entries by checkpoint id and checkpoint ref;
- done: test blank and wall-avoidant immortal sentinels as normal assignment
  entries;
- done: test context hash mismatch, provisional row gating, audit source snapshot
  hash, per-slot evidence, and parser compatibility;
- done: tournament rating rows preserve checkpoint `run_id`, `attempt_id`,
  `iteration`, `checkpoint_mtime_ns`, and per-run `latest_for_run` metadata;
- next: remote-smoke `stable_slots_v1` after this metadata repair and inspect
  the concrete assignment entries;
- next: document the assignment command path.
- next: implement run-scoped slot recipe Dict control and materialize it into
  assignment/audit artifacts.

### Trainer Wiring

- assignment ref resolver helper reads local/Volume assignment and resolves the
  existing mixture contract;
- local train/poller command plumbing records assignment ref/metadata;
- tiny remote train smokes consumed assignment refs;
- next: run-scoped reward recipes should map to existing reward variants first:
  `sparse_outcome`, `dense_survival_plus_outcome`, and
  `survival_plus_bonus_no_outcome`;
- next: launch/attempt metadata should record reward profile, reward schema id,
  reward schema hash, and reward recipe hash;
- next: changing reward recipe during one attempt should fail unless a specific
  replay policy exists;
- resume reuses assignment by default;
- explicit refresh creates new assignment id;
- eval/GIF receives same assignment metadata as training;
- no LightZero collector/search/learner code imports tournament modules.
- direct in-run assignment refresh now exists for the trainer:
  `opponent_assignment_refresh_interval_train_iter` plus
  `opponent_assignment_refresh_ref`.
- done locally: coarse refresh helpers build per-env reset params, verify all
  envs report the new assignment before collect, keep the old assignment on
  pre-reset pending-assignment failures, keep/retry when the pending assignment
  is missing `assignment_sha256`, block collect on reset-proof failure, fail
  before training if the hook cannot be installed, and write refresh JSONL plus
  summary data.
- done locally: fake-LightZero tests cover bucket behavior around train
  iterations 49/50/51/100 and prove refresh applies before the future collection
  batch.
- done locally: env tests prove a refreshed mixture clears the old selected
  opponent and old loaded frozen opponent object when a slot name is reused.
- done remotely: tiny Modal train proof `refresh-e2e-smoke-20260514/train-refresh-d`
  completed with initial assignment A, pending assignment B, one applied refresh
  event, all-env ready proof, and telemetry rows under both A then B.
- next: wire the direct path to a run-control pending-assignment pointer with
  `ready.json` verification.
- next: add failure tests for missing Dict, wrong run id, wrong attempt id,
  stale generation, stale refresh index, missing assignment file, hash mismatch,
  mutable checkpoint refs, partial write, and env-manager update failure.

### Intake And Online Elo

- local: scheduled subscriber discovers new broad checkpoints;
- local: duplicate checkpoint events are harmless;
- local: submit rejects scheduler knobs and only accepts candidate refs/run IDs;
- local: drain uses manifest policy unless overrides are explicitly allowed;
- local: live watches are seeded with run IDs or run prefixes;
- local: explicit checkpoint refs are frozen seeds and do not discover future
  checkpoints;
- local: discovery scans `train/lightzero_exp*/ckpt/iteration_*.pth.tar`, including
  timestamped DI-engine experiment folders;
- local: drain cannot spawn into an existing rating run unless continuation is
  explicit with `continue_from_latest=True`;
- local: continuation uses the full `seen_checkpoint_refs` pool, loads previous
  `latest.json`, increments round index, and preserves pair history.
- local: continuation preserves retired checkpoint history in the rating spec;
  the scheduler filters retired rows at pair selection time instead of deleting
  them from durable rating state.
- local: explicit continuation preserves old checkpoint ids from `latest.json`
  when new checkpoint refs are inserted before old refs.
- local: queue loss can be repaired from manifest queued refs when drain finds
  an empty queue;
- local: stale claims expire or repair without consuming events twice;
- local: a missing Dict manifest can be repaired from the durable Volume
  manifest when drain/status/tick/submit are called with a tournament/rating id;
- local: a fresh claim for a smaller desired pool does not block a larger pool;
- local: fresh-run and continue-from-latest claims are separate claim modes;
- local: live run-id/prefix watches force continuation and use the full known
  `seen_checkpoint_refs` pool during drain/spawn;
- next: prove the live-watch continuation repair remotely on Modal;
- next: replaying the same rating round output is idempotent and does not
  double-count pair history or scheduler reason counts.

### One-Frame Evaluator

- official one-frame rating spec has `decision_source_frames=1`;
- game summary records `decision_source_frames=1`;
- rating context hash changes if cadence changes;
- old 12-frame tournaments are labeled legacy and never mixed into one-frame leaderboard.

### Seeded Roster / Scripted Players

- checkpoint-only tournament remains valid as the first clean leaderboard;
- general player specs are required before scripted policies can appear as
  leaderboard rows;
- scripted player identity must include kind, version, params hash, and
  rating/evaluator context;
- website/review routes must tolerate rows without checkpoint refs if scripted
  players become leaderboard members;
- resume-from-latest must preserve scripted rows, not drop them.

### Invincibility / Death-Immunity Designs

- duplicated mixture entries with mortal/immortal variants select at configured
  weights;
- telemetry records selected entry, public `opponent_immortal`, derived
  `opponent_death_mode`, and runtime mode;
- assignment audit records whether immortality is a per-entry property or a
  global overlay;
- frozen-policy cache behavior is tested if mortal/immortal variants reuse the
  same checkpoint at scale;
- tournament contexts label passive/immortal rows as diagnostic unless promoted
  intentionally.

### Existing Training Opponent Modes

- `blank_canvas_noop` requires `fixed_straight`;
- frozen checkpoint entries require exact immutable refs and normal runtime;
- `proactive_wall_avoidant` uses scripted wall-avoidant logic and safe margin;
- Public `opponent_immortal=true` is death immunity. Env-bound
  `opponent_death_mode=immortal` is derived runtime plumbing, not source bonus
  invincibility.

## Blockers Before Overnight Leaderboard-Fed Training

1. Online tournament continuation and queue/claim repair have local contract
   tests, but production-scale proof is still needed.
2. One-frame tournament/leaderboard run has a tiny remote smoke, but is not yet
   validated at current public-source scale.
3. A larger bounded closed-loop smoke is still needed, and it should explicitly
   verify the `recent_strong` slot.
4. Assignment writer flow needs one clear command path.
5. Direct periodic refresh exists, but leaderboard promotion still needs the
   run-control/`ready.json` handoff before it can safely change a running
   trainer without explicit `opponent_assignment_refresh_ref`.
6. Reward recipe control is designed but not implemented; first version should
   be launch/new-attempt only.

## Mutable Refresh Validation Gates

Plain target:

```text
Coach writes immutable assignment
-> Coach writes pending assignment pointer to run-control
-> trainer checks that pointer every about 50 learner iterations
-> trainer applies it before a future collection batch
-> trainer records exactly what happened
```

Do before code is trusted:

1. Pure contract tests for direct assignment hash/checkpoint resolution.
   **Local tests added for the direct path.**
2. Fake-LightZero tests for `Collector.collect` wrapping and learner-iteration
   cadence. **Local helper/fake tests added.**
3. Env-level tests for applying a new opponent mixture at reset and clearing old
   loaded opponent objects. **Local tests added.**
4. Subprocess-env smoke proving all collector envs use the same assignment after
   a forced reset.
5. Tiny train with initial assignment A and pending assignment B, checking
   refresh JSONL, summary, env telemetry, and checkpoint/GIF metadata.
   **Remote base-env proof added for refresh JSONL, summary, and env telemetry.**

Telemetry that must exist:

- refresh decision: `planned`, `applied`, `kept_previous`, or `failed`;
- run id and attempt id;
- Dict generation and refresh index;
- old and new assignment id/ref/hash;
- learner train iteration at check and apply time;
- collection batch index or collect call index;
- timing for Dict read, assignment resolve, env reset/apply, and frozen
  checkpoint load if it happened;
- failure reason if the old assignment was kept.

Artifacts to extend:

- `opponent_assignment_refresh_events.jsonl`: every check/apply/failure event;
- `summary.json`: initial assignment, latest applied assignment, refresh count,
  and failed refresh count;
- `progress_latest.json`: active assignment at checkpoint time;
- `env_steps.jsonl`: assignment id/ref/hash/refresh index for each row, not only
  mixture entry name;
- eval/GIF command metadata: assignment active when the checkpoint was saved,
  not whichever assignment is latest later.

Race conditions to keep testing:

- Dict points at a file before the file is visible on Volume;
- a newer pending assignment appears while the trainer is resolving an older
  one;
- two attempts for one run id both try to apply or acknowledge a refresh;
- one subprocess env updates and another does not;
- refresh happens after `Collector.collect` has started;
- a replay batch contains transitions from mixed assignment ids;
- a reused slot name keeps an old frozen checkpoint object alive;
- a checkpoint load stalls the first env action instead of the boundary hook.

Fallback behavior to test:

- Bad pending assignment before env reset: keep old assignment, log a visible
  `kept_previous` or `failed` event, continue.
- No pending assignment: no-op, continue.
- Assignment verified but one env reset fails: do not collect a possibly mixed
  batch; rebuild/restart or end the attempt cleanly.
- A refresh that fails should be easy to see in `refresh_events.jsonl`,
  `summary.json`, and the run website/status bundle.

## Minimal End-To-End Test Plan

1. Fixture rating snapshot with five eligible rows. **Done for pure path.**
2. Build public leaderboard snapshot. **Done for pure path.**
3. Materialize 3-5 `stable_slots_v1` assignment entries. **Done locally.**
4. Parse materialized assignment with existing trainer parser. **Done locally.**
5. Launch tiny dry/train smoke using the assignment ref. **Done manually.**
6. Emit one checkpoint. **Done manually.**
7. Intake discovers checkpoint. **Done manually.**
8. Rating updates. **Done manually.**
9. New `stable_slots_v1` assignment generated at explicit refresh boundary.
   **Done mechanically once; rerun after checkpoint-recency metadata repair.**

## 2026-05-14 Online Intake Proof

Passed:

- `inspector-detached-online-proof-20260514b` /
  `elo-oneframe-detached-proof` proved the small online path on Modal.
- Baseline: three checkpoints, one round, 3/3 games complete.
- Intake: three new checkpoints accepted.
- Continuation: round 1 ran with six checkpoints, 9/9 games complete, and
  `latest.json` advanced to round 1.

The proof specifically checks the footgun we hit earlier:

- `input.json` alone is not enough.
- `progress.json` saying games are running is not enough.
- Success means finished game summaries exist and `latest.json` points at the
  new round.

Still missing:

- Larger proof with a real current-source checkpoint pool.
- Public leaderboard publish at production thresholds.
- Website responsiveness proof against this completed run.
- Long-running subscriber/drain proof without manual operator steps.

Done for the small proof:

- Published `inspector-detached-proof-leaderboard-20260514b` from
  `inspector-detached-online-proof-20260514b` /
  `elo-oneframe-detached-proof`.
- Dict pointer published successfully.
- Tiny proof thresholds were used, so this validates the write path only.

## Non-Blockers For Plain Overnight Training

- Public leaderboard integration is not required if the next run uses static
  manifest-defined opponents.
- Optimizer speed recommendations can be applied as independent throughput
  settings if they do not alter observation/evaluator contract.
