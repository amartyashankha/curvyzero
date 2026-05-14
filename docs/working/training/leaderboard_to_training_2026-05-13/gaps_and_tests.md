# Gaps And Tests

## Highest Priority Gaps

| Gap | Why it matters | First test |
| --- | --- | --- |
| Live Dict pointer repair | Publisher writes pointer, but Modal Dict is not durable truth. | Local repair command repopulates missing/stale Dict from validated immutable Volume snapshot; tiny remote smoke passed; production runbook still needed. |
| Production assignment runbook | Assignment writer works, but the operator flow needs one documented safe path. | Runbook command stores `assignment.json` and `audit.json` under attempt path and records the returned refs. |
| Run-scoped slot recipe control | Operators should be able to change a run's desired slots after launch without changing trainer code. | Dict recipe validator plus materializer smoke writes a new immutable assignment and audit for one run id. |
| Intake continuation remote proof | Online Elo needs to add new checkpoints without resetting evidence. Local tests cover the contract; remote/prod proof is still needed. | Existing `latest.json` starts next round and preserves pair history in a bounded remote smoke. |
| Queue/dedupe repair remote proof | Queue events are not durable enough alone. Local tests cover repair; remote/prod proof is still needed. | Duplicate/lost event repaired by periodic scan in a bounded remote smoke. |
| One-frame tournament parity | New leaderboard must match current train cadence. | Rating spec with `decision_source_frames=1` is recorded, hashed, and used in game summaries; tiny two-checkpoint remote rating/publish smoke passed. |
| Stable-slot recency smoke | `recent_strong` only means recent if rating rows carry run/attempt/iteration/latest metadata. | Rerun a bounded multi-checkpoint smoke and verify `recent_strong` selects the latest checkpoint for the watched run, not only the best remaining row. |
| Larger bounded closed-loop smoke | Tiny manual smoke proves plumbing, not scale or repair behavior. | Run a bounded multi-checkpoint smoke and verify publish -> assignment -> train still works after metadata repair. |
| Seeded non-checkpoint players | Scripted/hand-coded baselines need roster identity if included. | Normalize scripted player specs without fake checkpoint refs. |
| Fractional invincibility overlay | Some percentage of episodes may need invincible opponents regardless of base policy. | Deterministic selection applies `opponent_death_mode=immortal` as an overlay and records telemetry/audit. |

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
- repeated comfortable pairs are de-prioritized when fresh useful pairs exist;
- duplicate pairs are blocked within a round;
- some cross-band games remain so scalar Elo does not hide non-transitive
  matchups.

Do not treat local contract coverage or tiny remote smokes as production-scale
proof. Production gates still need remote queue/stale-claim proof, a larger
current-source one-frame leaderboard run, and a larger closed-loop smoke.

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
- done: dedupe checkpoint entries by checkpoint id and checkpoint ref;
- done: test blank and wall-avoidant immortal sentinels as normal assignment
  entries;
- done: test context hash mismatch, provisional row gating, audit source snapshot
  hash, per-slot evidence, and parser compatibility;
- done: tournament rating rows preserve checkpoint `run_id`, `attempt_id`,
  `iteration`, `checkpoint_mtime_ns`, and per-run `latest_for_run` metadata;
- next: remote-smoke `stable_slots_v1` after this metadata repair and inspect
  the concrete assignment entries;
- next: document the production operator flow.
- next: implement run-scoped slot recipe Dict control and materialize it into
  assignment/audit artifacts.

### Trainer Wiring

- assignment ref resolver helper reads local/Volume assignment and resolves the
  existing mixture contract;
- local train/poller command plumbing records assignment ref/metadata;
- tiny remote train smokes consumed assignment refs;
- resume reuses assignment by default;
- explicit refresh creates new assignment id;
- eval/GIF receives same assignment metadata as training;
- no LightZero collector/search/learner code imports tournament modules.

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
- local: queue loss can be repaired from manifest queued refs when drain finds
  an empty queue;
- local: stale claims expire or repair without consuming events twice;
- next: missing Dict manifest can be rebuilt from a Volume manifest;
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
- telemetry records selected entry, `opponent_death_mode`, and runtime mode;
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
- `opponent_death_mode=immortal` is death immunity, not source bonus
  invincibility.

## Blockers Before Overnight Leaderboard-Fed Training

1. Modal Dict pointer repair/fallback has local tests and a tiny remote smoke,
   but still needs production runbook coverage.
2. Assignment writer/operator flow needs a production runbook.
3. Periodic safe refresh semantics are absent.
4. Online Elo continuation and queue/stale-claim repair have local contract
   tests, but production-scale proof is still needed.
5. One-frame tournament/leaderboard run has a tiny remote smoke, but is not yet
   validated at current public-source scale.
6. A larger bounded closed-loop smoke is still needed, and it should explicitly
   verify the `recent_strong` slot.

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

## Non-Blockers For Plain Overnight Training

- Public leaderboard integration is not required if the next run uses static
  manifest-defined opponents.
- Optimizer speed recommendations can be applied as independent throughput
  settings if they do not alter observation/evaluator contract.
