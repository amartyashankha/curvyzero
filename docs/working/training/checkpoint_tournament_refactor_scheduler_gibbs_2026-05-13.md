# Checkpoint Tournament Refactor Scheduler Critique, 2026-05-13

Scope: adaptive Elo for every useful checkpoint from every CurvyTron run. We
cannot run all-pairs. This is a critique lane, not an implementation patch.

## Bottom Line

Adaptive Elo is fine as the first map, but it must not pretend to be a neutral
round robin. The scheduler becomes part of the measurement instrument. If the
coach sees only rank and rating, adaptive selection can mislead.

The V0 architecture should keep batch-wave Elo, but add stronger guardrails:
rating context compatibility, lineage caps, explicit uncertainty labels,
mandatory bridge games, seat-bias audits, and held-out random audits.

## Current Architecture Pressure Points

- `adaptive_v0` is moving in the right direction: bounded rounds, schedule
  reasons, pair history, and static Modal work.
- The state hash looks roster-based. That is dangerous for an online pool:
  adding one new checkpoint changes the hash and can make old scheduler state or
  pair history look incompatible. We need a separate `rating_context_hash` for
  evaluator semantics and a `roster_hash` for the current player list.
- Pair history is keyed by unordered checkpoint pair. Good for rematches, but it
  can hide seat-order evidence unless seat-specific counts are stored and shown.
- Current active/provisional labels are simple floors. Good start, but not
  enough to express uncertainty, lineage diversity, or opponent strength.
- Odd games per battle are useful operationally, but they do not solve seat
  bias, draw-heavy matches, or close-pair noise.

## V0 Design

Keep V0 boring:

1. Static batch waves.
   - Scheduler writes a round manifest.
   - Modal runs independent pair shards.
   - Reducer updates Elo once from the full wave.
   - No worker updates rating state.

2. Split hashes.
   - `rating_context_hash`: env variant, reward variant, evaluator version,
     score rule, max steps, decision timing, policy mode, num simulations,
     action semantics, observation/render contract.
   - `roster_hash`: current checkpoint ids/refs.
   - Pair history may carry forward when the roster changes, but must not carry
     across context changes.

3. Keep a fixed schedule mix.
   - 50-60% near-rating pairs.
   - 15-20% provisional/uncertain pairs.
   - 10% anchors.
   - 10-15% random bridge pairs.
   - 5-10% replay/audit pairs.

4. Add same-lineage caps.
   - Direct lineage comparisons are useful, especially predecessor vs child.
   - But same-run or same-family pairs must not dominate a checkpoint's evidence.
   - Active status should require outside-lineage opponents.

5. Make uncertainty visible.
   - `provisional`, `active`, `audit_needed`, or `stale`.
   - Show games, distinct opponents, outside-lineage opponents, anchor games,
     last-round delta, failure rate, draw/timeout rate, and opponent strength.

6. Treat bridges as required evidence.
   - Each rating band needs random cross-band bridges.
   - Each lineage family needs outside-family bridges.
   - Top checkpoints need periodic bridges to anchors and nearby non-lineage
     rivals.

7. Keep 21 games as routing evidence, not final proof.
   - Use 21 by default.
   - Top up close/rank-critical pairs to 51 or 101.
   - If timeouts/draws are frequent, do not silently accept 0.5 as equality.

## V1 Design

V1 should still keep Modal embarrassingly parallel, but improve the model:

- Add explicit uncertainty, Glicko-style RD, or a simple posterior interval.
- Add seat-aware rating or at least a global seat-bias correction report.
- Add a Bradley-Terry or logistic recompute from all pair summaries as an audit
  against wave-by-wave Elo drift.
- Add nontransitivity diagnostics: cycles, matchup clusters, and top-K stability.
- Add active-evaluation mode for "find the best checkpoint" separately from
  "rank every checkpoint."
- Add holdout audit sets that the scheduler cannot optimize against.
- Add context migration tools: new evaluator context starts a new pool, with
  explicit bridge jobs only when comparison is needed.

## Risks That Could Mislead The Coach

Selection bias:

- Near-rating schedules over-sample interesting boundaries and under-sample
  boring regions.
- Mitigation: mandatory random bridge quota and held-out random audit rounds.

Sparse graph confidence:

- A checkpoint can look settled after many games against too few opponents.
- Mitigation: active status needs games, distinct opponents, outside-lineage
  opponents, and anchor coverage.

Nontransitivity:

- A scalar Elo can hide A beats B, B beats C, C beats A.
- Mitigation: cycle finder and cluster report beside the leaderboard.

Seat bias:

- Odd battle counts do not fix seat advantage.
- Mitigation: seat-swapped audits, seat-specific win rates, and optional
  seat-aware Elo correction.

Draws/timeouts:

- Timeout draws can mean equal skill, weak play, bad max-step settings, or a
  broken policy.
- Mitigation: publish draw and timeout rates separately; replay high-timeout
  pairs with review/GIF sampling.

Same-lineage overconfidence:

- Many checkpoints from one run can mostly play each other and create fake
  precision.
- Mitigation: cap same-lineage pair share and require outside-family evidence.

Roster-vs-context confusion:

- If adding checkpoints invalidates history, online scheduling becomes clumsy.
  If incompatible evaluator contexts share history, ratings become wrong.
- Mitigation: separate `rating_context_hash` and `roster_hash`.

Provisional labels too weak:

- `active` can sound like "trusted" when evidence is narrow.
- Mitigation: use status plus reason fields: `low_games`,
  `low_outside_lineage`, `high_delta`, `high_timeout`, `stale`, `audit_needed`.

Odd games per battle:

- 21 games avoids pure W/L split ties, but close pairs remain noisy and draws can
  still tie.
- Mitigation: rank-critical top-ups and bootstrap intervals.

Batch-wave lag:

- Batch updates are order-stable and Modal-friendly, but the scheduler cannot
  react inside a long wave.
- Mitigation: keep waves modest; use more waves rather than one giant wave.
  Batch-wave updates are enough for V0 if each wave includes bridges and audits.

## Tests And Simulations Needed

Scheduler determinism:

- Same spec, prior snapshot, pair history, and seed produce identical pair specs.

Context compatibility:

- Roster expansion preserves valid pair history.
- Evaluator/context change rejects old history unless explicitly bridged.

Same-lineage caps:

- Synthetic pool with one long run and many short runs does not spend most pairs
  inside the long run.
- Active status fails without enough outside-lineage opponents.

Bridge coverage:

- Every rating decile and lineage family receives required bridge games over N
  rounds.
- Removing bridge quota creates detectable graph islands in the test.

Seat bias:

- Inject a seat-0 advantage in simulation and verify the report detects it.
- Seat-swapped pair results should estimate the bias and avoid false promotion.

Draws/timeouts:

- Inject high timeout rates and verify rows become `audit_needed`, not simply
  stable.
- Replay/top-up logic should target high-timeout pairs.

Uncertainty/provisional:

- New strong checkpoint, new weak checkpoint, and collapsed lineage checkpoint
  all remain provisional until enough outside evidence exists.
- Coach-facing snapshots must show status and reason next to rating.

Nontransitivity:

- Simulate rock-paper-scissors policies and verify Elo alone is flagged as
  insufficient.

Selection bias:

- Compare adaptive ratings against a held-out random audit set.
- Alert when top rankings are not supported by audit games.

Batch-wave adequacy:

- Compare one giant wave, many small waves, and online game-by-game updates on
  the same synthetic truth.
- V0 passes if batch waves find the same top group and produce stable enough
  ranks with lower Modal complexity.

Coach UI/report smoke:

- No leaderboard row should show rating and rank without status, games,
  distinct opponents, outside-lineage opponents, draw/timeout rate, and
  last-round delta.

## V0 Recommendation

Use batch-wave adaptive Elo, but make it honest:

- fixed budget per round;
- mandatory bridges;
- lineage caps;
- explicit context hash;
- visible uncertainty;
- seat/draw/timeout diagnostics;
- periodic held-out audit rounds.

Do not optimize the scheduler only for pretty rating convergence. Optimize it
for coach trust: enough evidence, clear caveats, and fast detection when Elo is
the wrong summary.

## Follow-Up: V0 Rating Context Hash Fields

Goal: separate "who is in the roster" from "what measurement system produced
the games." Old pair evidence can survive roster expansion, but must not survive
evaluator/rating-context drift.

### `rating_context_hash(spec)` V0

Hash a canonical JSON object with exactly these fields:

- `context_schema`: fixed string, `curvytron_rating_context/v0`.
- `game`: fixed string, `curvytron`.
- `score_rule`: fixed string or spec value for first-death win, simultaneous
  death draw, timeout draw.
- `rating_formula_version`: current batch Elo formula version.
- `draw_score`.
- `min_valid_fraction`.
- `max_steps`.
- `decision_ms`.
- `decision_source_frames`.
- `source_physics_step_ms`.
- `policy_mode`.
- `collect_temperature`.
- `collect_epsilon`.
- `num_simulations`.
- `natural_bonus_spawn`.
- `evaluation_env_variant`.
- `evaluation_reward_variant`.
- `observation_contract`: stable fields that affect policy inputs, especially
  visual/source-state mode and `policy_trail_render_mode`.
- `action_space_contract`: stable id for left/straight/right action semantics.
- `seat_policy`: unordered single battle, ordered seats, or required seat-swap
  protocol.
- `timeout_policy`: timeout is draw in V0.

Use normalized values: explicit nulls, sorted keys, no timestamps, no paths.

### `roster_hash(spec)` V0

Hash only roster identity:

- `roster_schema`: fixed string, `curvytron_rating_roster/v0`.
- Sorted rows of `checkpoint_id`, `checkpoint_ref`, and optional immutable
  checkpoint state key.

Roster hash changes when checkpoints are added or removed. Rating context hash
does not.

### What Not To Include In `rating_context_hash`

Do not include:

- `checkpoint_id`, `checkpoint_ref`, labels, run ids, lineage ids, or roster
  size.
- `tournament_id`, `rating_run_id`, round id, timestamps, Modal call ids, file
  refs, or Volume paths.
- `seed`, game index, pair index, or battle id.
- `pairs_per_round`, pair selection mode, schedule reason mix, anchors, replay
  policy, or scheduler budget.
- `games_per_pair` or `games_per_shard`.
- `save_gif`, GIF sampling, frame saving, frame size, GIF FPS, or browser
  rendering details used only for review artifacts.
- `policy_batch_size` if it is only a performance batching knob. If a later
  policy implementation can change actions when batch size changes, promote it
  into the context hash and treat old evidence as incompatible.
- Provisional thresholds, active thresholds, or coach UI display settings.

### Tests Needed

Context drift rejection:

- Changing `max_steps` changes `rating_context_hash` and old pair history is
  rejected unless an explicit bridge/import mode is used.
- Changing `policy_mode` from eval to collect changes the hash.
- Changing `num_simulations` changes the hash.
- Changing `draw_score` or `min_valid_fraction` changes the hash.
- Changing eval env/reward variant changes the hash.
- Changing observation contract or `policy_trail_render_mode` changes the hash.
- Changing timeout policy changes the hash.

Roster expansion reuse:

- Start with roster A/B and pair history for A-vs-B.
- Expand roster to A/B/C with the same rating context.
- `rating_context_hash` stays equal.
- `roster_hash` changes.
- A-vs-B pair history remains loadable and usable.
- C starts with no pair history and provisional status.

Non-context changes:

- Changing `tournament_id`, `rating_run_id`, round index, seed, or Modal refs
  does not change `rating_context_hash`.
- Changing `pairs_per_round`, scheduler mix, or pair selection mode does not
  change `rating_context_hash`.
- Changing `games_per_pair` or `games_per_shard` does not change
  `rating_context_hash`; it changes evidence quantity, not game semantics.
- Changing GIF settings does not change `rating_context_hash`.

Safety test:

- If old pair history has matching `rating_context_hash` but missing players
  from the current roster, keep only rows whose two checkpoint ids are present
  in the current roster.
