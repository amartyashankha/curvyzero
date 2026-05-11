# Multiplayer Fast Env Push

Status: working memory; reset/warmup, narrow warmdown, and metadata env landed
Date: 2026-05-11

Scope: no-bonus 3P/4P fast runtime parity. Do not pull in bonuses,
visuals, LightZero, or broad 1v1 refactors.

One runtime is under hardening: `VectorMultiplayerEnv`.
`CurvyTronSourceEnv` and the source JS oracle are proof tools, and strict
`VectorTrainerEnv1v1NoBonus` is proof/profiling only.
No-bonus or strict wrapper restrictions in this note are temporary explicit
profile configs. They are not the reconstruction path and must not replace
source-default CurvyTron behavior.

Priority gap list:
[multiplayer_env_gap_targets_2026-05-10.md](multiplayer_env_gap_targets_2026-05-10.md).

Parallel execution now: body/trail public tests, 2P metadata replay, and the
bonus/borderless plan are active outside this note's landed-slice summary.

## 2P Status

See [active_lanes.md](active_lanes.md#2p-status) for the live 2P status. This
landed-slice note should not carry a copied status paragraph.

## What Landed In This Turn

- Comparator/direct-runtime 3P/4P wall fixtures are green:
  - `source_normal_wall_3p_two_die_one_survivor_step`
  - `source_normal_wall_4p_ordered_deaths_survivor_score`
  - `source_normal_wall_4p_two_prior_then_same_frame_terminal_draw`
- No-bonus N-player reset and warmup helpers are green for the focused 2P/3P/4P
  lifecycle slice.
- No-bonus 3P warmdown/next-round helper is green for the all-dead
  `source_lifecycle_spawn_rng_3p_next_round` slice.
- No-bonus 4P warmdown/next-round helper is green for the all-dead
  `source_lifecycle_spawn_rng_4p_next_round` slice.
- No-bonus 3P survivor warmdown-death/next-round proof is green for
  `source_lifecycle_survivor_score_3p_next_round`. It protects the rule that a
  survivor can die during warmdown without scoring the completed round again.
- No-bonus 4P survivor-score/next-round proof is green for
  `source_lifecycle_survivor_score_4p_next_round`, including source-style
  `game:stop` PrintManager cleanup before the next spawn RNG draws.
- No-bonus 3P present/absent draw warmdown/next-round proof is green for
  `source_lifecycle_present_absent_3p_next_round`. It protects the source rule
  that `game:stop` resizes the next arena to the present-player count, next
  `round:new` skips the absent avatar for spawn RNG, and absent delayed
  PrintManager state survives because source clears only present avatars.
- `VectorMultiplayerEnv` now exposes a metadata-only public 2P/3P/4P
  no-bonus surface. It is intentionally debug metadata only and makes no
  learned/trainer observation claim.
- Public seeded 4P wall canaries now run through `VectorMultiplayerEnv`
  for survivor-score and all-dead draw metadata.
- Public 4P fixture-tape reset/spawn now runs through `VectorMultiplayerEnv`
  for `source_lifecycle_spawn_rng_order_4p`, including scheduled warmup metadata.
- Public 3P fixture-tape present/absent reset now runs through
  `VectorMultiplayerEnv` for `source_lifecycle_present_absent_3p_round_new`.
  The fix moved reset/warmup round-local clearing before spawn so the source-style
  absent-player death list is not erased after spawn.
- Public 3P present/absent survivor scoring now runs through
  `VectorMultiplayerEnv` for
  `source_lifecycle_present_absent_3p_survivor_score_round_end`. Warmup now
  schedules PrintManager starts for every avatar, matching the source absent
  avatar fixture.
- Public 3P present/absent next-round continuation now runs through the
  metadata-only `VectorMultiplayerEnv.advance_warmdown(...)` bridge for
  `source_lifecycle_present_absent_3p_next_round`. It proves map-size shrink,
  absent death-list preservation, next-round action mask, and random cursor
  metadata. It is not a trainer-ready natural lifecycle/autoreset API.
- Multiplayer replay contract guards landed. They do not write 3P/4P replay
  yet; they reject missing metadata and block 3P/4P from claiming the strict
  1v1 replay/observation schema.
- Helper-level 3P match/tie/multi-round coverage landed in lifecycle tests.
  Unique max-score leader ends the match, tied max-score leaders continue to
  next round, and the multi-round fixture continues once before ending later.
- Public metadata-only 3P match/tie/multi-round coverage now runs through
  `VectorMultiplayerEnv` for the same focused fixtures. It reports
  warmdown `match_done`/`match_winner` for a unique max-score leader, continues
  tied max-score leaders into the next round, and reaches match end on the
  second warmdown in the multi-round fixture.
- Focused lifecycle/runtime/multiplayer guard, ruff, and doc guard are green on
  the touched set.
- This is still not full public 3P/4P env parity. Natural public reset,
  source-backed public warmdown/replay, learned observations, public survivor
  warmdown, match-mode episode policy, autoreset, and final-observation policy
  remain separate work.
- CurvyTron visual stacked-frame input from our own renderer remains an
  intended LightZero training target. The current scalar/ray single-ego rows are
  a practical bridge, and multiplayer replay will need visual frame provenance
  plus full wrapper action logs, opponent policy ids, player ids, present/alive
  masks, death order, score vectors, and reset/RNG metadata.

## Promoted Source Fixtures To Carry Forward

Lifecycle 3P:

- `source_lifecycle_spawn_rng_order_3p`
- `source_lifecycle_spawn_rng_warmup_print_start_3p`
- `source_lifecycle_spawn_rng_3p_next_round`
- `source_lifecycle_survivor_score_3p_round_end`
- `source_lifecycle_survivor_score_3p_next_round`
- `source_lifecycle_present_absent_3p_round_new`
- `source_lifecycle_present_absent_3p_survivor_score_round_end`
- `source_lifecycle_present_absent_3p_next_round`
- `source_lifecycle_match_end_at_max_score_3p`
- `source_lifecycle_tie_at_max_score_3p`
- `source_lifecycle_multi_round_match_end_3p`

Lifecycle 4P:

- `source_lifecycle_spawn_rng_order_4p`
- `source_lifecycle_spawn_rng_4p_next_round`
- `source_lifecycle_survivor_score_4p_next_round`

Normal-wall multiplayer:

- `source_normal_wall_3p_two_die_one_survivor_step`
- `source_normal_wall_4p_ordered_deaths_survivor_score`
- `source_normal_wall_4p_two_prior_then_same_frame_terminal_draw`

## Already In Fast Code

- `src/curvyzero/env/vector_spawn.py`
  - Supports `player_count` 3 and 4 for reverse natural spawn order.
  - Supports row-local random tape calls for spawn x, y, and angle retries.
  - Supports present/absent spawn skip and absent death list writes.
- `src/curvyzero/env/vector_lifecycle.py`
  - Has no-bonus reset/spawn/warmup rows for dynamic 2P/3P/4P player counts.
  - Has focused 3P/4P warmdown/next-round helper proofs, including survivor
    continuation and the 3P present/absent next-round arena-resize rule.
- `src/curvyzero/env/vector_runtime.py`
  - `step_many` accepts `player_count` greater than 2.
  - Reverse player update order is already in the main loop.
  - Frame-start death score and survivor score use `player_count`.
  - Optional row terminal metadata is already written when terminal arrays
    exist.
  - Has no-bonus warmup timer advancement for dynamic 2P/3P/4P player counts.
  - Has narrow no-bonus 3P and 4P warmdown/next-round helper proofs for
    `source_lifecycle_spawn_rng_3p_next_round` and
    `source_lifecycle_spawn_rng_4p_next_round`.
- `src/curvyzero/env/vector_multiplayer_env.py`
  - Exposes `VectorMultiplayerEnv` for metadata-only 2P/3P/4P stepping.
  - Adds metadata-only `advance_warmdown(...)` for narrow fixture-backed
    lifecycle proofs.
  - Accepts `[B,P]` public actions and returns debug transition metadata.
  - Does not claim natural public reset fidelity, replay fidelity, or learned
    trainer observations.
- `tests/test_vector_spawn.py`
  - Already pins 3P spawn order, 3P present/absent spawn, and 4P spawn order.
- `tests/test_vector_runtime.py`
  - Pins direct 3P and 4P no-bonus wall scoring against promoted fixtures.
  - Pins 3P and 4P all-dead warmdown/next-round spawn proofs.
  - Pins 3P survivor warmdown death without round re-scoring, then next-round
    spawn.
  - Pins 4P survivor-score next-round, including live survivor PrintManager
    stop at `game:stop`.
- `tests/test_vector_lifecycle.py`
  - Pins the no-bonus N-player reset/warmup helper behavior.
  - Pins 3P present/absent warmdown/next-round behavior for map-size resize,
    absent death-list preservation, spawn RNG cursor, and absent PrintManager
    state.
  - Pins helper-level 3P match/tie/multi-round behavior.
- `tests/test_vector_multiplayer_env.py`
  - Pins metadata-only public 3P and 4P seeded wall canaries.
  - Pins one metadata-only public 4P fixture-tape reset/spawn/scheduled-warmup
    proof.
  - Pins one metadata-only public 3P fixture-tape present/absent reset proof.
  - Pins one metadata-only public 3P present/absent survivor-scoring proof.
  - Pins one metadata-only public 3P present/absent warmdown/next-round proof.
  - Pins metadata-only public 3P match/tie/multi-round warmdown behavior.
- `src/curvyzero/training/multiplayer_replay_contract.py`
  - Defines the required future multiplayer replay metadata fields and rejects
    incomplete 3P/4P metadata.
- `tests/test_multiplayer_replay_contract.py`
  - Guards against accidentally routing 3P/4P through strict replay-v0 or the
    1v1 ray observation schema.

## Missing Fast Pieces

1. Broader warmdown, next round, and match end.
   - Focused helper proofs now cover 3P/4P all-dead next round, 3P/4P survivor
     next round, and one 3P present/absent next round.
   - Public metadata now covers one present/absent next-round continuation.
   - Helper-level and public metadata match-end/tie/multi-round behavior is
     green for focused 3P fixtures; broader present/non-present variants remain.
   - Preserve tie-at-max behavior: tied leaders continue, single leader ends.

2. Broader death list bookkeeping.
   - Optional death-list append exists in the fast runtime.
   - Still test more public metadata surfaces and broader lifecycle cases.
   - Keep score calculation based on frame-start deaths, not the just-mutated
     death list.

3. Public no-bonus multiplayer surface.
   - A dynamic-player metadata surface has landed on the shared
     `VectorMultiplayerEnv` runtime.
   - Keep this work out of the strict `VectorTrainerEnv1v1NoBonus` proof class.
   - Required inputs: `player_count`, `present`, `max_score`, `reset_seed`,
     `decision_ms`, capacities.
   - Required action shape: `[B,P]`; absent/dead players must not require live
     actions.
   - Required returned metadata: `player_count`, `present`, `winner`,
     `draw`, `round_done`, `match_done`, `death_player`, `death_count`,
     `score`, `round_score`, `terminal_reason`, reset seed/source.
   - Current status: metadata-only public stepping exists. Natural public reset,
     warmdown/replay integration, learned observations, and broad lifecycle
     parity are still missing.

4. Public observation/reward bridge.
   - Edit `src/curvyzero/env/vector_trainer_observation.py`.
   - Do not force the 1v1 ego/opponent schema onto 3P/4P.
   - First safe path: expose debug/public transition metadata for 3P/4P and
     keep trainer observations out of scope.
   - If a trainer-facing observation is required, make a new schema id for
     dynamic no-bonus multiplayer and raycast against all other players.

5. Autoreset and replay metadata.
   - Edit `src/curvyzero/env/vector_autoreset.py` only after the source
     terminal contract is split into round terminal versus match terminal.
   - Edit `src/curvyzero/training/vector_env_replay_recorder.py` after public
     env metadata exists.
   - Replay must carry `player_count`, `present`, death order, score vector,
     round/match terminal flags, reset seed/source, and RNG cursor/history.

## Next Patch Set

Patch A: broaden warmdown and next round.

- First target landed:
  `source_lifecycle_spawn_rng_3p_next_round`.
- Second target landed:
  `source_lifecycle_spawn_rng_4p_next_round`.
- Survivor targets landed:
  `source_lifecycle_survivor_score_4p_next_round`.
- Next targets:
  present/absent lifecycle and match/tie/multi-round metadata.

Patch B: public runtime metadata.

- Keep hardening the no-bonus `VectorMultiplayerEnv` metadata surface with no
  trainer observation claim.
- Public 4P fixture canaries landed for seeded wall metadata:
  ordered-death survivor and terminal draw. They assert public metadata and
  final debug observation only.
- Natural public reset, replay integration, and learned/trainer observations
  stay missing until they have their own source-backed claims.
- The 4P fixture-tape reset proof does not claim general seed reset parity or
  PrintManager start; it only proves source tape spawn and scheduled warmup.
- The 3P present/absent reset proof does not claim survivor scoring or next-round
  continuation by itself. Survivor scoring is now covered separately; next-round
  continuation is still open.

Patch C: match/presence edges.

- Add tests for:
  `source_lifecycle_present_absent_3p_round_new`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`,
  `source_lifecycle_present_absent_3p_next_round`,
  `source_lifecycle_match_end_at_max_score_3p`,
  `source_lifecycle_tie_at_max_score_3p`,
  `source_lifecycle_multi_round_match_end_3p`.

## Do Not Do In This Batch

- No bonuses.
- No visual renderer.
- No LightZero adapter.
- No broad rewrite of strict 1v1 public env.
- No training-speed claims until reset, runtime, public transition metadata,
  final rows, and replay are source-backed for 3P/4P.
