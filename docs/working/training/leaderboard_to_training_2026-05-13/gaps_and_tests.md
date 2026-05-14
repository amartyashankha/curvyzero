# Gaps And Tests

## Highest Priority Gaps

| Gap | Why it matters | First test |
| --- | --- | --- |
| Live Dict pointer repair | Publisher writes pointer, but no explicit repair path. | Missing/stale Dict falls back to Volume snapshot and can be republished. |
| Production assignment runbook | Assignment writer works, but the operator flow needs one documented safe path. | Runbook command stores `assignment.json` and `audit.json` under attempt path and records the returned refs. |
| Intake continuation | Online Elo needs to add new checkpoints without resetting evidence. | Existing `latest.json` starts next round and preserves pair history. |
| Queue/dedupe repair | Queue events are not durable enough alone. | Duplicate/lost event repaired by periodic scan. |
| One-frame tournament parity | New leaderboard must match current train cadence. | Rating spec with `decision_source_frames=1` is recorded, hashed, and used in game summaries. |
| Larger bounded closed-loop smoke | Tiny manual smoke proves plumbing, not scale or repair behavior. | Run a bounded multi-checkpoint smoke and verify publish -> assignment -> train still works. |
| Seeded non-checkpoint players | Scripted/hand-coded baselines need roster identity if included. | Normalize scripted player specs without fake checkpoint refs. |
| Fractional invincibility overlay | Some percentage of episodes may need invincible opponents regardless of base policy. | Deterministic selection applies `opponent_death_mode=immortal` as an overlay and records telemetry/audit. |

## Existing Tests To Reuse

- `tests/test_opponent_registry.py`
- `tests/test_curvytron_checkpoint_tournament.py`
- `tests/test_curvytron_tournament_scheduler_guardrails.py`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`

## New Test Groups

### Public Leaderboard Contract

- done for pure builder in `tests/test_opponent_leaderboard.py`;
- done for tournament-side publisher local coverage and remote smoke;
- next: repair/fallback command for stale or missing Dict pointer.

### Assignment Selector

- done for pure top-slots v0 in `tests/test_opponent_leaderboard.py`;
- done for writing assignment/audit artifacts under a training attempt in the
  smoke path;
- next: improve diversity/lineage logic and document the production operator
  flow.

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

- scheduled subscriber discovers new broad checkpoints;
- duplicate checkpoint events are harmless;
- queue loss can be repaired from manifest state;
- drain cannot spawn into existing rating run unless continuation is explicit;
- continuation loads previous `latest.json`, increments round index, and preserves pair history.

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

1. Modal Dict pointer repair/fallback is absent.
2. Assignment writer/operator flow needs a production runbook.
3. Periodic safe refresh semantics are absent.
4. Online Elo continuation and queue/dedupe repair are absent.
5. One-frame tournament/leaderboard run is not yet validated as the current public source.
6. A larger bounded closed-loop smoke is still needed.

## Minimal End-To-End Test Plan

1. Fixture rating snapshot with five eligible rows. **Done for pure path.**
2. Build public leaderboard snapshot. **Done for pure path.**
3. Select five assignment slots. **Done for pure path.**
4. Parse assignment with existing trainer parser. **Done for pure path.**
5. Launch tiny dry/train smoke using the assignment ref. **Done manually.**
6. Emit one checkpoint. **Done manually.**
7. Intake discovers checkpoint. **Done manually.**
8. Rating updates. **Done manually.**
9. New assignment generated at explicit refresh boundary. **Done manually.**

## Non-Blockers For Plain Overnight Training

- Public leaderboard integration is not required if the next run uses static
  manifest-defined opponents.
- Optimizer speed recommendations can be applied as independent throughput
  settings if they do not alter observation/evaluator contract.
