# Aggressive Reconstruction Burst Plan - 2026-05-09

## Current Truth

- 16 lifecycle fixtures are green, including the focused 4P all-present
  all-dead warmdown/next-round proof.
- Forced `BonusGameClear` is green through `source_bonus_game_clear_immediate_step.json`.
- Full environment reproduction is not done. Do not imply otherwise.
- The next work is not more status writing. The next work is source-backed fixtures, Python parity, and fast-runtime promotion in parallel.

## Why This Has Taken Too Long

- We have been treating "full env" like one giant finish line instead of a queue of small source facts.
- We keep spending time reconciling docs, matrices, and wording while the executable surface stays narrow.
- Too many patches touch shared runner files at once, which creates merge drag and review stalls.
- Lifecycle, bonus, movement traces, long rollouts, and vector promotion have been serialized even when their fixtures can be produced independently.
- Some work has stopped at "source read" instead of producing a named scenario, JS oracle proof, Python parity check, and promotion decision.
- We have allowed broad phrases like "full lifecycle" or "bonus support" to mask the exact missing rule.
- We have not consistently separated oracle generation from source-env implementation from vector/runtime promotion.

## Stop Doing

- Stop editing existing environment docs for progress narration. This burst owns this file only.
- Stop claiming broader lifecycle coverage from the 16 green fixtures.
- Stop using fixture count language that lags the repo state.
- Stop bundling multiple bonus effects into one "bonus patch".
- Stop adding source-env behavior without a fixture name and a test name.
- Stop touching merge-lock files casually. Announce the file, land the smallest patch, then get out.
- Stop making vector/full-runtime claims until scalar source-env parity is green for that chunk.
- Stop waiting for the perfect full-env proof before landing isolated missing rules.

## Parallel Work Now

Run these lanes concurrently. Each lane has its own fixture files and tests. Only the merge captain touches shared registries after the lane proves local facts.

| Lane | Goal | Primary Output | Owner Rules |
| --- | --- | --- | --- |
| A | 4P survivor next-round | `source_lifecycle_survivor_score_4p_next_round.json` | One lifecycle owner only touches lifecycle runner/test registry. |
| B | Present/absent survivor scoring | `source_lifecycle_present_absent_3p_survivor_score_round_end.json` and next-round companion if needed | Reuse present/absent setup; do not widen unrelated spawn assertions. |
| C | 3P mid-round leave continuation | `source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json` | Source-env owner owns leave semantics; lifecycle owner only consumes fixture once stable. |
| D | Longer JS/Python rollouts | `source_lifecycle_long_3p_no_bonus_wall_round_done.json` and `source_lifecycle_long_3p_no_bonus_next_round.json` | Reuse persistent JS worker; no runtime/vector claims yet. |
| E | Live movement event trace | `source_live_movement_event_trace_2p_no_bonus_multistep` | Promoted; keep as the live movement event-order guard. |
| F | Bonus defaults/caps/natural clear | `source_bonus_default_weights_type_rng_step.json`, `source_bonus_spawn_cap_twenty_step.json`, `source_bonus_game_clear_natural_type_rng_step.json` | One bonus rule per fixture; keep forced `BonusGameClear` as the baseline. |
| G | Merge captain | Registry/test glue only | Serializes shared files after lane fixtures are ready. |

## Merge Locks

Only one person should edit each of these at a time:

- `src/curvyzero/fidelity/source_runners.py`: lifecycle scenario allow-lists, trace payload shape, runner copy.
- `src/curvyzero/env/source_env.py`: executable source-env semantics, timer behavior, bonus behavior, leave behavior.
- `tests/test_source_lifecycle_runner.py`: lifecycle parity assertions and scenario constants.
- `tests/test_lifecycle_oracle.py`: JS oracle fixture wiring for lifecycle.
- `tests/test_source_env.py`: scalar source-env behavioral assertions.
- `tests/test_env_scenarios.py`: JS scenario assertions, especially bonus and event-order assertions.
- `tests/test_compare_vector_arrays_to_fidelity.py`: vector acceptance gates; touch only after scalar parity.
- `scenarios/environment/*_batch.json`: batch membership is a shared registry, not a scratchpad.
- `src/curvyzero/env/config.py`: default bonus list and config contract.

Non-lock files that can be created in parallel:

- New `scenarios/environment/source_lifecycle_*.json` files.
- New `scenarios/environment/source_bonus_*.json` files.
- Narrow one-off tests that do not edit central fixture registries.
- Local probe scripts under `scripts/` when they are temporary and named for the lane.

## Implementable Chunks

### A1 - 4P Survivor Next-Round

Fixture: `source_lifecycle_survivor_score_4p_next_round.json`

Implement:

- Start from `source_lifecycle_spawn_rng_4p_next_round.json`.
- Force three wall deaths and leave one survivor.
- Assert `round:end`, survivor score increment, `game:stop`, next `round:new`, next spawn RNG labels, and reverse PrintManager stop/start order.
- Add JS oracle assertion in `tests/test_lifecycle_oracle.py`.
- Add Python runner assertion in `tests/test_source_lifecycle_runner.py`.
- Only then add runner allow-list glue in `source_runners.py`.

Done when:

- JS oracle and Python lifecycle runner match event order, random calls, snapshots, and score state.

### B1 - Present/Absent Survivor Round-End

Fixture: `source_lifecycle_present_absent_3p_survivor_score_round_end.json`

Implement:

- Start from `source_lifecycle_present_absent_3p_round_new.json`.
- Keep one avatar absent before `newRound`.
- Force one present avatar to die while the other present avatar survives.
- Assert absent avatar does not steal score, does not get live movement/PrintManager claims, and survivor scoring counts present/alive source semantics.
- Add focused source-env assertions before vector promotion.

Done when:

- Round-end scoring proves present/absent survivor behavior without next-round complexity.

### B2 - Present/Absent Survivor Next-Round

Fixture: `source_lifecycle_present_absent_3p_survivor_score_next_round.json`

Implement:

- Extend B1 through warmdown and next `round:new`.
- Assert next-round spawn stream excludes non-present avatars where source excludes them and preserves delayed PrintManager behavior where source still schedules it.
- Keep this separate from B1 so failures are diagnosable.

Done when:

- Python runner matches JS oracle through next-round spawn and timer events.

### C1 - 3P Mid-Round Leave Continues

Fixture: `source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json`

Implement:

- Start from existing `source_lifecycle_mid_round_remove_avatar_2p.json`.
- Use three players.
- Remove one avatar mid-round.
- Keep two present/active enough that the round continues after `player:leave`.
- Then force terminal death and assert round-end scoring uses remaining source state.
- Add `CurvyTronSourceEnv.remove_avatar` or equivalent parity tests first if source-env behavior is missing.

Done when:

- The leave event does not prematurely end the round and final scoring matches source.

### C2 - 3P Mid-Round Leave Then Next-Round

Fixture: `source_lifecycle_mid_round_remove_avatar_3p_continue_next_round.json`

Implement:

- Extend C1 through warmdown.
- Assert removed avatar is not respawned.
- Assert timer drain handles already-scheduled PrintManager timers for removed/non-present avatars exactly like source.

Done when:

- Source-env and lifecycle runner agree on leave, stop/start timers, next `round:new`, and score carryover.

### D1 - Longer Natural 3P Round-Done Rollout

Fixture: `source_lifecycle_long_3p_no_bonus_wall_round_done.json`

Implement:

- Start from `source_lifecycle_long_1v1_no_bonus_wall_round_done.json`.
- Use persistent JS worker.
- Disable bonuses.
- Run deterministic movement until natural wall terminal `round:end`.
- Compare terminal tick, final scores, deaths, world body counts, and event order.
- Keep the assertion narrow: round done, not warmdown.

Done when:

- JS worker and Python source-env produce the same terminal summary for the pinned rollout.

### D2 - Longer Natural 3P Next-Round Rollout

Fixture: `source_lifecycle_long_3p_no_bonus_next_round.json`

Implement:

- Extend D1 through warmdown and next `round:new`.
- Assert timer drain, PrintManager stop/start order, next spawn RNG, and persistent score.
- Add timeout guard so the test fails fast if the rollout misses terminal state.

Done when:

- Python source-env survives a longer natural JS comparison without hand-forced deaths.

### E1 - Live Movement Event Trace

Fixture: `source_live_movement_event_trace_2p_no_bonus_multistep`

Implement:

- Capture a live, non-terminal update after `game:start`.
- Assert movement event ordering relative to PrintManager point insertion, body/world updates, and zero-elapsed position events.
- Include per-avatar event labels and source timestamp/order facts.
- Do not mix in bonus, wall death, or round-end.

Done when:

- The trace proves live movement events, not just snapshots after lifecycle edges.

### F1 - Bonus Default Weights

Fixture: `source_bonus_default_weights_type_rng_step.json`

Implement:

- Use default bonus list from config/source.
- Pin random values that select at least `BonusSelfSmall`, `BonusGameClear`, and one non-small/non-clear type across separate cases or subfixtures.
- Assert RNG labels and selected type only.
- Do not implement every effect in this chunk.

Done when:

- Source-env can reproduce default weighted type selection without pretending all selected effects are implemented.

### F2 - Bonus Cap

Fixture: `source_bonus_spawn_cap_twenty_step.json`

Implement:

- Seed or naturally create 20 active bonuses.
- Trigger `popBonus`.
- Assert no new bonus is added and no position/type RNG is consumed past the source cap branch.
- Keep this independent from weighted type selection.

Done when:

- Cap behavior is source-backed and prevents accidental over-spawn.

### F3 - Natural BonusGameClear Selection

Fixture: `source_bonus_game_clear_natural_type_rng_step.json`

Implement:

- Start from `source_bonus_game_clear_immediate_step.json` for clear semantics.
- Add natural spawn/type selection that chooses `BonusGameClear`.
- Assert default-weight random path, `bonus:pop`, catch, `bonus:clear`, `clear`, and final world state.
- Keep forced clear fixture unchanged as the baseline.

Done when:

- `BonusGameClear` is proven both as forced catch behavior and as natural selectable bonus behavior.

## Verification Commands

Run the smallest command per lane before touching locks:

- Lifecycle JS: `uv run pytest tests/test_lifecycle_oracle.py -q`
- Lifecycle Python: `uv run pytest tests/test_source_lifecycle_runner.py -q`
- Source-env scalar: `uv run pytest tests/test_source_env.py -q`
- Bonus JS/Python: `uv run pytest tests/test_env_scenarios.py tests/test_source_env.py -q`
- Long JS reuse: `uv run pytest tests/test_js_reuse_probe.py -q`
- Vector promotion only after scalar parity: `uv run pytest tests/test_compare_vector_arrays_to_fidelity.py -q`

## Burst Rule

Every patch must land one of these:

- A new fixture.
- A JS oracle assertion for that fixture.
- A Python source-env or lifecycle runner assertion for that fixture.
- A minimal runner/config change needed for that fixture.
- A vector/runtime promotion for an already green scalar fixture.

Anything else waits.
