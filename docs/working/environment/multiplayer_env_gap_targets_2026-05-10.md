# Multiplayer Environment Gap Targets

Status: blunt implementation target list
Date: 2026-05-10

Goal: one fast, source-faithful CurvyTron runtime for multiplayer self-play:
`VectorMultiplayerEnv` under hardening. Old no-bonus public-env naming was stale
current-contract wording, not a separate second product implementation.
Cleanup decision: keep one fast runtime path and consolidate source-backed
behavior there, rather than treating historical public-env naming as a second
implementation.
`CurvyTronSourceEnv` and the JS oracle are proof/oracle tools, not the product.
The strict 1v1/no-bonus trainer path is a proven proof/profiling boundary, not
the target scope. Proven source behavior should move into
`VectorMultiplayerEnv`, not into parallel product environments.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

Source-native transition model: CurvyTron holds per-avatar control state and
advances it over elapsed-ms server frames. `joint_action`, `step`, fixed
decision cadence, and action sidecars are CurvyZero wrapper/replay words only. Do
not use those words as if they are native source semantics.
Wrapper restrictions are temporary explicit profile/adapter configs; they are
not the reconstruction path and must not replace source-default CurvyTron
behavior.

Current green boundary:

- Strict `VectorTrainerEnv1v1NoBonus` still guards the proof-wrapper version of
  the long 1v1/no-bonus fixture, and `VectorMultiplayerEnv` now runs the
  same long source rollout from reset to terminal as the intended runtime guard.
- Direct public body/trail/collision canary tests now run through
  `VectorMultiplayerEnv`.
- Focused 2P public warmdown/match-end checks and the 2P metadata replay bridge
  are green.
- The source-state LightZero wrapper fixed-opponent sidecar proof is green, and
  the fixed-opponent route has a tiny CPU Modal smoke. This is route evidence
  only, not the next priority, two-seat self-play, browser pixel truth, or
  environment fidelity.
- Latest validation reported on 2026-05-11: focused environment validation
  reported `282 passed`, source bonus validation reported `33 passed`, ruff
  passed, and the environment doc guard passed.
- Direct seeded 3P/4P no-bonus wall scoring/order canaries are green.
- N-player no-bonus reset/warmup helpers are green.
- One narrow no-bonus 3P all-dead warmdown/next-round helper is green for
  `source_lifecycle_spawn_rng_3p_next_round`.
- One narrow no-bonus 4P all-dead warmdown/next-round helper is green for
  `source_lifecycle_spawn_rng_4p_next_round`.
- No-bonus survivor continuation proofs are green for
  `source_lifecycle_survivor_score_3p_next_round` and
  `source_lifecycle_survivor_score_4p_next_round`. The 3P proof covers survivor
  death during warmdown without scoring the completed round a second time. The
  4P proof covers separate survivor-score deaths, `game:stop` cleanup of the
  live survivor's PrintManager, and next-round spawn RNG.
- `VectorMultiplayerEnv` exists as a metadata-only public 2P/3P/4P
  no-bonus surface. It uses `curvyzero_debug_metadata_only/v0` and explicitly
  makes no trainer-observation claim.
- `src/curvyzero/env/vector_multiplayer_observation.py` adds
  `curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0`, a narrow
  learned-observation schema/projection over `VectorMultiplayerEnv.state`.
  It emits `float32[R,27]` present+alive ego rows. It is not a second env, not
  trainer-ready env support, not visual/pixel support, and not
  source-fidelity completion. A separate scalar replay-shaped artifact can now
  package these rows with public metadata; that artifact is still not trainer
  replay or policy/search/value target data.
- Public 4P seeded wall canaries are green through `VectorMultiplayerEnv`.
  They prove seeded public fixture metadata only, not natural public reset,
  replay, or trainer observation parity.
- Public 4P fixture-tape reset/spawn is green through `VectorMultiplayerEnv`
  for `source_lifecycle_spawn_rng_order_4p`. It proves source-tape spawn
  positions/headings and scheduled warmup metadata only, not general seed reset
  parity or PrintManager start. Public reset metadata now labels whether the
  random tape came from `source_fixture_random_tape_values` or
  `seed_generated_source_random_history`, carries `random_tape_length`,
  `rng_impl_id`, and `random_tape_history_ref`, and can carry a source fixture
  reference when the reset call supplies one. The seed-generated path uses
  `curvyzero_seeded_source_math_random_history/v0`: deterministic row-local
  `float64` values in `[0,1)` from `reset_seed`, not V8 `Math.random` bit
  parity. A focused public reset test feeds that generated row history into
  `CurvyTronSourceEnv` for 2P/3P/4P and proves reset spawn plus warmup
  `print_manager.start_distance` random-call order. That is the exact scope of
  `natural_multiplayer_reset_claim=true` for generated reset rows.
- Public 3P fixture-tape present/absent reset is green for
  `source_lifecycle_present_absent_3p_round_new`: the absent player is not
  spawned, has no action mask, and remains in the source-style death list.
- Public 3P present/absent survivor scoring is green for
  `source_lifecycle_present_absent_3p_survivor_score_round_end`: the absent
  player stays out of the action mask/reward losers, remains in death order, and
  source-style warmup starts PrintManagers for every avatar before the present
  death is scored.
- Helper-level 3P present/absent next-round continuation is green for
  `source_lifecycle_present_absent_3p_next_round`: first round uses the
  3-player arena, warmdown `game:stop` shrinks the next spawn arena to the
  2-present-player size, next `round:new` skips the absent avatar for spawn RNG,
  re-adds it to the death list, and preserves its delayed PrintManager state.
- Public metadata coverage for the same present/absent next-round continuation
  is green through the narrow metadata-only `advance_warmdown(...)` bridge. It
  proves public metadata and masks after the next spawn; it is still not a
  trainer-ready natural lifecycle/autoreset API.
- Helper-level match/tie/multi-round checks are green for the focused 3P source
  fixtures: one unique leader at max score ends the match, tied leaders at max
  score continue to the next round, and the multi-round fixture continues once
  before ending on the later warmdown.
- The 4P unique-leader source fixture
  `source_lifecycle_match_end_at_max_score_4p.json` is promoted as JS oracle
  plus Python source-runner under `source-lifecycle-v25`. It also has one
  focused public metadata proof. The 4P all-present multi-round fixture
  `source_lifecycle_multi_round_match_end_4p.json` is source-promoted under the
  same lifecycle hash and has one focused public metadata proof. The three 4P
  present/absent reset/survivor/next-round source fixtures are also promoted
  under that hash and have focused public metadata proofs. The 3P and 4P
  present/absent tie-at-max source fixtures are promoted under
  `source-lifecycle-v25` and have focused public metadata proofs.
- 3P and 4P mid-round `removeAvatar` continuation coverage is green through JS
  oracle fixtures and Python `CurvyTronSourceEnv` parity. They prove the leaver
  emits `player:leave`, becomes non-present/non-alive, is not added to
  current-round `deaths`, the round continues while enough avatars remain
  alive, and a later terminal death scores from source avatar-collection
  sizing.
- `source_lifecycle_remove_avatar_to_single_present_3p.json` is promoted under
  `source-lifecycle-v25` as JS oracle plus Python source-runner. It proves an
  active 3P no-bonus leave edge: avatar 3 dies first and enters `deaths=[3]`;
  removing live avatar 2 emits `die` then `player:leave`, sets avatar 2
  `present=false/alive=false`, does not add avatar 2 to current deaths,
  immediately ends the round because only avatar 1 remains alive, gives avatar
  1 `roundScore=2` using total avatar count, does not emit `end` at warmdown
  because avatar 3 is still present, and starts the next round at the
  two-present-player size with avatar 2 in next-round deaths. Focused public
  metadata parity is green.
- Public metadata coverage for active-round leave exists only for narrow paths
  through `VectorMultiplayerEnv.remove_player(...)`: 3P/4P continuation,
  2P immediate round end, and one 4P source-rule canary for survivor scoring
  after already-dead players. It is metadata-only, uses zero-based public ids
  with source ids equal to public id plus one, marks the leaver present/alive
  false, and keeps the leaver out of `death_player`. It rejects warmdown,
  terminal, absent-player, dead-player, and bad-shape calls. This is not broad
  public leave, broad public warmdown leave, replay, trainer, visual, or bonus
  support.
- Focused 3P staged match-mode warmdown leave metadata is green. It does not
  claim broad public warmdown leave or broader leave edge variants.
- Public metadata coverage for the same 3P match/tie/multi-round fixtures is
  green through `VectorMultiplayerEnv`: unique max-score leader reports
  `match_done`/`match_winner` after warmdown, tied max-score leaders continue
  to the next round, and the multi-round fixture continues once before ending
  on the second warmdown. `episode_end_mode=match` is now explicit in the same
  metadata-only env: round end reports `round_done` without public `done` or
  `needs_reset`, and match-end warmdown reports `done`/`match_done` with final
  debug metadata before reset. This remains outside trainer-ready match-mode
  episodes.
- Public lifecycle identity metadata now separates reset episode identity from
  source round identity. `round_id` starts at `1` after reset and increments
  only when `advance_warmdown(...)` spawns a next round; `reset_episode_id`
  remains tied to the explicit reset episode. Public info and metadata replay
  also carry `source_round_id`, `reset_episode_id_policy`,
  `source_round_id_policy`, and a versioned `lifecycle_policy_id` so callers
  cannot read a multi-round match as multiple reset episodes.
- Public lifecycle facts are now owned by state arrays from reset, not only
  stitched into later metadata bridge rows: `round_done`, `warmdown_pending`,
  `match_done`, `round_winner`, and `match_winner`. This is guarded by
  `test_public_lifecycle_metadata_arrays_exist_from_reset` plus focused round
  and match warmdown tests. It is still not a natural public reset or full
  lifecycle parity claim. The stale lifecycle metadata overlay plumbing has
  been removed from `vector_multiplayer_env.py`.
- Public final-row metadata is green for the metadata-only env: terminal
  `step(...)` batches label final observation/reward rows separately from the
  versioned `final_observation_policy`, next-round warmdown reports no new
  final rows and clears `needs_reset`, and the focused 3P match-end warmdown
  returns a debug-metadata final observation plus a zero reward map for the
  `match_done` row. Reset/public metadata preserves seed/source/cursor/draw
  count plus random tape source, length, and RNG implementation id; seed-generated
  tape is deterministic local RNG metadata and is not a source reset parity
  claim. This is not hidden autoreset, production replay, or learned-observation
  support.
- Public 2P/3P/4P metadata env output can be packaged into a metadata-only
  multiplayer replay-v0 record/chunk, and a narrow sequence recorder now turns
  public `VectorMultiplayerEnv.step(...)` batches into validated
  metadata records. The recorder uses the terminal barrier policy: any
  terminal/final row closes the current chunk, and later appends require
  explicit recorder reset or a new recorder. Records carry player ids,
  present/alive masks, action mask, full wrapper action map / `joint_action`,
  reward vector,
  done/terminated/truncated, round/match terminal facts, winner/draw, score
  vectors, death order, reset seed/source, random tape cursor/draw count,
  optional RNG provenance ref, action sidecar, observation schema id,
  final-observation policy, and an optional opponent policy sidecar with policy
  id/version/seed/actions. It remains `metadata_only=true` and
  `trainer_observation_claim=false`.
- The 3P/4P scalar projection can now be packaged into a replay-shaped scalar
  row artifact, `curvyzero_multiplayer_scalar_observation_replay_shape/v0`.
  It stores `observation`, scalar `action_mask`, `lightzero_action_mask`,
  `env_row_id`, `ego_player_id`, `row_mask`, and `source_shape`, and each active
  scalar row carries a validated public metadata replay record. That nested
  public record is the source of truth for reset/episode/round/step trace
  fields, so `round_id` keeps the public rule: reset starts at `1`, next-round
  warmdown increments it, and match-end rows keep the final round id. This is
  not full trainer replay, not visual replay, not source-fidelity completion,
  and not policy/search/value targets. The dead legacy scalar replay builder
  has been removed from `multiplayer_replay_v0.py`.
- Public masked reset after terminal rows is explicit for
  `VectorMultiplayerEnv`: `autoreset_done_rows()` defaults to
  `_needs_reset`, rejects row masks that select nonterminal rows, resets only
  the selected rows with reset source `autoreset`, reports a versioned
  `public_reset_policy`, preserves the vector reset pre-reset terminal
  snapshot, and leaves the previous terminal batch's final-row metadata
  available through `last_step_info`. This is still an explicit API call, not
  hidden autoreset or trainer-ready match episodes.
- `source_lifecycle_tie_at_max_score_4p.json` now has JS oracle, Python source
  runner, and focused public metadata parity for tied 4P leaders continuing to
  the next round. Keep broader 4P match lifecycle separate.

Current blockers remain simple:

- Broader natural reset/warmup beyond the seed-generated source-history
  reset/spawn/warmup call-order proof. Warmup frame movement, full lifecycle,
  replay, and V8 RNG bit parity remain out of claim.
- Broader native lifecycle ownership beyond the current reset-owned public
  arrays and narrow bridge helpers.
- Broad public warmdown movement/death beyond the one explicit 3P match-mode
  metadata bridge.
- Broader leave variants: broad public leave, broad public warmdown leave beyond
  the focused 3P staged metadata proof, and leave edge cases. Immediate
  round-end public leave is only 2P fixture-backed plus one 4P canary. The
  3P single-present leave edge has focused public metadata parity, but broader
  leave variants remain open.
- Masks and rewards for every public state: live, dead, absent, terminal,
  warmdown, timeout, overflow, draw, survivor, and match-end.
- Full public replay/final observations after the public source contract is
  stable.
- Seeded/natural public bonus fixture support has landed for promoted slices,
  including `BonusSelfMaster` and `BonusAllColor`. Focused public natural
  source-default catch/effect coverage now exists for self, enemy, game, and
  all-target effects. Same-frame natural bonus plus PrintManager random-order
  accounting is protected. The low-level natural bonus spawn helper in
  `vector_runtime.py` has type/position/retry/cap tests. `BonusSelfMaster`
  wall/body parity and `BonusAllColor` reverse target event order plus
  older-wins overlap behavior are fixed. Full replay/final state,
  manual/direct stack guard documentation, possible fully-blocked generated-map
  policy, broader stack/death stress, and broad natural bonus replay support
  remain open.
- Seeded public bonus replay metadata now preserves audit fields only. Forced
  optional-array `BonusGameBorderless` catch is runtime-tested. The tiny CPU
  Modal fixed-opponent smoke is route evidence only; the spawn helper is not
  public natural spawn support.
- Old toy/debug paths are quarantined as historical smoke/interface evidence
  only. Do not cite them as product-runtime, replay, bonus, or fidelity proof.

## Priority 1: Source-Native Mechanics Missing From The Fast Env

These are the first runtime targets because they still require source behavior
to move out of proof tools and into the fast path.

| Target | Current proof | Fast-env target |
| --- | --- | --- |
| 4P next-round warmdown | Source fixtures exist for all-dead and survivor 4P continuation. Helper proofs are green for `source_lifecycle_spawn_rng_4p_next_round` and `source_lifecycle_survivor_score_4p_next_round`. | Add public metadata coverage only if this behavior needs to be exposed through `VectorMultiplayerEnv` before the broader lifecycle API. |
| Survivor warmdown movement/death | Source proves a 3P survivor can keep moving after `round:end`, die during warmdown, then continue to next round. The 3P no-rescore proof is green. Public metadata coverage now exists for one focused 3P match-mode row through explicit `VectorMultiplayerEnv.advance_warmdown_frame(...)`: ordinary `step()` still raises during warmdown, the explicit 1150 ms warmdown frame moves the source-fixture survivor from `x=18.991` to the source death point `x=0.591`, records death order `[2, 1, 0]`, keeps score `[2, 0, 0]`, leaves round score `[2, 0, 0]`, and then `advance_warmdown(3850.0)` reaches next round. Source also proves 4P survivor scoring and next-round continuation; the fast proof is green. | Keep this as fixture-backed metadata-only match-mode coverage. It is not hidden autoreset, not ordinary public `step()` during warmdown, not replay/visual/trainer support, and not broad source frame-loop lifecycle. |
| Round terminal versus match terminal | Source fixtures distinguish round end, match end, and tie-at-max continuation. | Split fast metadata into `round_done`, `match_done`, `terminal_reason`, `winner`, `draw`, score, and round score without using `done` as the only lifecycle fact. |
| Present/absent lifecycle | Source has focused 3P present/absent first-round, survivor scoring, and continuation fixtures. Public fixture-tape first-round reset, survivor scoring, and next-round continuation are green. | Broaden only after a new source fixture isolates a new present/non-present rule. |
| Mid-round and warmdown leave lifecycle | Source has focused 2P immediate round-end leave coverage, 3P and 4P continuation-through-round-end fixtures checked by the JS oracle and `CurvyTronSourceEnv`, and the new `source_lifecycle_remove_avatar_to_single_present_3p.json` single-present edge under `source-lifecycle-v25`. Public metadata now has narrow active-round support through `VectorMultiplayerEnv.remove_player(...)`: 3P/4P continuation, 2P immediate round end, the 3P single-present leave edge, and one 4P source-rule canary. Focused 3P staged match-mode warmdown leave metadata is also green. | Keep the bridge metadata-only and fixture/source-rule-backed: zero-based public ids, source ids equal to public id plus one, leaver present/alive false, and no leaver entry in `death_player`. Do not claim broad public leave, broad public warmdown leave, replay, trainer, visual, or bonus support. |
| Match-end and multi-round behavior | Source has 2P max-score, 3P max-score, 4P max-score, 3P tie-at-max, 4P tie-at-max, 3P/4P present/absent tie-at-max, and 3P/4P all-present multi-round match-end fixtures. Helper-level and public metadata-only 3P unique-leader, tied-leader, multi-round, and explicit round-mode/match-mode episode-policy checks are green. Public 4P tie-at-max, unique-leader max-score, and all-present multi-round metadata are focused and green. | Keep match/tie/multi-round public coverage fixture-backed. Next chunks are replay metadata and broader lifecycle claims, not a second env or hidden reset policy. |
| Row-local RNG provenance | Vector rows carry seed/source plus tape cursor/draw count. Public reset/env metadata now also carries random tape source, tape length, RNG implementation id, `random_tape_history_ref`, and an optional source fixture ref when supplied. Seed-generated rows use `seed_generated_source_random_history` with `curvyzero_seeded_source_math_random_history/v0`; one parameterized 2P/3P/4P test proves the generated history drives public reset spawn/warmup random-call order identically when fed into `CurvyTronSourceEnv`. Metadata replay preserves the older fields and allows an optional external RNG provenance ref. | Preserve seed/source/cursor/draw count plus source/length/history metadata for fidelity and replay/debug provenance. Keep the claim scoped to generated source-history reset/spawn/warmup call order; do not claim V8 bit parity, broad warmup frame movement, replay parity, or a heavyweight RNG subsystem. |
| Bonuses | Source has narrow bonus proofs plus broad Python stack/effect support for self/enemy/all bonuses, table-backed optional-array runtime slices for promoted effects including `BonusSelfMaster`, `BonusAllColor`, and `BonusEnemyStraightAngle`, focused public natural source-default catch/effect coverage, seeded public bonus replay audit metadata, same-frame natural bonus plus PrintManager random-order accounting, and a low-level natural spawn helper with type/position/retry/cap tests. `BonusSelfMaster` wall/body parity and `BonusAllColor` reverse target event order plus older-wins overlap behavior are fixed. Seed-generated public random tape auto-extends deterministically, generated natural bonus position retry is no longer capped by `natural_bonus_position_attempt_capacity`, and public natural bonus timer advancement has no artificial callback cap; fixture/direct finite tapes and `vector_runtime` finite helpers remain strict. | Keep public bonus claims narrow until stack semantics beyond the guarded cases, full replay/final state, manual/direct stack guard documentation, possible fully-blocked generated-map policy, and broad natural replay support are promoted. Natural catch/effect tests do not imply replay/final-state coverage. |

## Priority 2: 2P/3P/4P No-Bonus Lifecycle Gaps

Use these as concrete implementation chunks. Each target should get source
fixture parity first, then vector/runtime parity, then public metadata coverage.
The source-only fixture gap checklist is
[no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md](no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md).

1. Done narrowly: promote match/tie/multi-round behavior through public
   metadata for `source_lifecycle_match_end_at_max_score_3p`,
   `source_lifecycle_tie_at_max_score_3p`, and
   `source_lifecycle_multi_round_match_end_3p`.
2. Done narrowly: promote `source_lifecycle_match_end_at_max_score_4p`
   under `source-lifecycle-v25` with JS oracle plus Python source-runner, plus
   focused public metadata parity for unique-leader match end.
3. Done narrowly: promote `source_lifecycle_multi_round_match_end_4p`
   under `source-lifecycle-v25` with JS oracle plus Python source-runner, plus
   focused public metadata parity for that 4P all-present multi-round path.
4. Done narrowly: promote the three `source_lifecycle_present_absent_4p_*`
   fixtures under `source-lifecycle-v25` with JS oracle plus Python
   source-runner, plus focused public metadata parity for reset/spawn,
   survivor scoring, and next-round continuation.
5. Done narrowly: promote `source_lifecycle_remove_avatar_during_warmdown_3p`
   under `source-lifecycle-v25`, plus focused 3P staged match-mode warmdown
   leave metadata parity.
6. Done narrowly: promote `source_lifecycle_remove_avatar_to_single_present_3p`
   under `source-lifecycle-v25`, plus focused public metadata parity for this
   single-present active leave edge.
7. Add remaining source fixtures only before making broader lifecycle claims:
   broader warmdown leave variants and broader present/absent match variants.
   The 3P/4P present/absent tie-at-max source fixtures and the 4P all-present
   tied max-score continuation are now covered narrowly; do not widen them into
   broader present/non-present lifecycle.
8. Done narrowly: add 3P and 4P mid-round `removeAvatar` source-oracle and
   `CurvyTronSourceEnv` parity for continuation through later round end. Public
   metadata support exists for 3P/4P continuation, 2P immediate round end, and
   one 4P source-rule immediate survivor canary. It must not be widened without
   new fixture-backed tests.

The direct 3P/4P wall canaries are not enough for lifecycle claims. They protect
death order and scoring on seeded runtime rows only.

Public canary target landed:

- 4P public metadata canaries now run through `VectorMultiplayerEnv` for the
  promoted seeded wall fixtures:
  `source_normal_wall_4p_ordered_deaths_survivor_score` and
  `source_normal_wall_4p_two_prior_then_same_frame_terminal_draw`.
- They assert `player_count`, `present`, `alive`, `score`,
  `round_score`, `death_player`, `death_count`, `winner`, `draw`,
  `terminal_reason`, `reward`, `final_observation`, `action_sidecar`, and
  `trainer_observation_claim=false`.
- Do not bundle natural reset, warmdown, replay, or learned observation into
  this canary patch.

## Priority 3: Public Env, Replay, And Observation Gaps

`VectorMultiplayerEnv` is useful now, but it is deliberately not a
trainer-ready env. Its no-bonus name is the current intended public runtime
name, not a second product path.

| Surface | Exists now | Environment gaps |
| --- | --- | --- |
| Public env | Metadata-only 2P/3P/4P env, explicit round and match episode modes, action shape `[B,P]`, debug observation, action sidecar, direct public body/trail/collision canaries, terminal metadata for seeded wall rows, the long 2P reset-to-terminal source rollout, focused 2P warmdown/match-end checks, one fixture-tape 4P reset/spawn/scheduled-warmup proof, focused 3P/4P present/absent reset and survivor-scoring proofs, focused 3P/4P present/absent `advance_warmdown(...)` next-round proofs, one focused 3P explicit `advance_warmdown_frame(...)` survivor warmdown movement/death/no-rescore proof, focused 3P match/tie/multi-round public metadata checks, focused 4P unique-leader/tie/all-present multi-round public metadata checks, focused 3P staged match-mode warmdown leave metadata, reset-owned lifecycle arrays (`round_done`, `warmdown_pending`, `match_done`, `round_winner`, `match_winner`), reset-vs-round identity metadata (`reset_episode_id`, one-based `source_round_id`, and lifecycle policy ids), seed-generated source-history reset metadata with one parameterized 2P/3P/4P `CurvyTronSourceEnv` call-order proof, narrow final-row metadata for terminal step plus 3P match-end warmdown, metadata-only replay packaging, 2P metadata replay bridge, and explicit `autoreset_done_rows()` for terminal rows preserving prior final-row metadata while leaving live rows untouched. | Broader warmup/frame lifecycle parity, broader warmdown/next-round public tests, replay metadata hardening for match-mode rows, hidden autoreset policy beyond the explicit done-row reset API if one is later needed, and minimal seed/source/cursor provenance for replay/fidelity. |
| Replay and final observations | Strict 1v1/no-bonus replay-v0 live-step path exists. A narrow multiplayer metadata replay-v0 packager now accepts public 2P/3P/4P metadata env rows and validates player count, present/alive masks, action mask, full wrapper action map / `joint_action`, reward vector, done/terminated/truncated, round/match terminal facts, winner/draw, score vectors, death order, reset episode id, source round id, lifecycle policy id, reset seed/source, random tape cursor/draw count, optional RNG ref, action sidecar, observation schema id, final-observation policy, and seeded public bonus audit metadata when present. A metadata-only sequence recorder closes on terminal/final rows and may attach an optional opponent policy sidecar with policy id/version/seed/actions. A separate scalar row artifact can package the 3P/4P scalar observations and masks with nested public metadata records. | Broaden into full public replay/final-observation rows only after lifecycle/reset policy and bonus audit facts are stable. Keep current surfaces out of broad 3P/4P replay claims. |
| Observation surface | 1v1 `curvyzero_egocentric_rays/v0` exists for the strict no-bonus slice. `curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0` exists as a pure `float32[R,27]` 3P/4P no-bonus state projection for present+alive ego rows, and those rows now have a replay-shaped packaging artifact. | Full public observation/final-observation support is still missing. Do not reuse the 1v1 ray schema for 3P/4P. Do not treat the scalar artifact as a production replay writer, visual/pixel path, source-fidelity completion, or policy/search/value target source. |
| Rewards and masks | Sparse round-outcome policy is drafted and used narrowly. | Per-player reward vector and legal mask tests for live, dead, absent, terminal, timeout, draw, survivor win, and truncation cases. |
| Bonus public metadata | Seeded public bonus audit metadata exists narrowly, seeded `BonusGameBorderless` catch/expiry support has landed, focused public natural source-default type selection exists, same-frame natural bonus plus PrintManager random-order accounting is protected, and source/runtime/public borderless duration/expiry passes. | Borderless stack semantics, fuller bonus replay audit, full replay/final state, and broader runtime/public effects remain open. |
| Browser/source pixels | Not implemented for multiplayer. The landed 3P/4P scalar projection is non-visual. | Pixel parity is later, after source state, public replay/final observations, and lifecycle rows are stable. |

## Priority 4: Do Not Claim Yet

- Do not claim full environment fidelity.
- Do not claim trainer-ready 3P/4P observation support. The scalar 3P/4P
  learned-observation schema/projection exists, but it is only a pure no-bonus
  state projection.
- Do not claim production or trainer-ready 3P/4P replay. Current multiplayer
  replay is metadata-only public-env packaging plus scalar observation row
  packaging, without trainer targets or a shard/manifest writer.
- Do not claim full natural public multiplayer reset. Seed-generated rows now
  use `seed_generated_source_random_history` and set
  `natural_multiplayer_reset_claim=true` only for the tested
  `seeded_source_history_reset_spawn_warmup_call_order/v0` scope: the generated
  row history feeds public reset and `CurvyTronSourceEnv` with matching spawn
  plus warmup `Math.random` call order. Fixture-tape and direct-state reset
  paths still keep the claim false. This is not V8 RNG bit parity, broad warmup
  frame movement, replay parity, hidden autoreset, or trainer-ready lifecycle.
- Do not claim broad public active-round leave. The current
  `remove_player(...)` support is only the metadata-only 3P/4P continuation,
  2P immediate round-end, and one 4P source-rule canary
  bridge described above.
- Do not claim hidden public autoreset. The current public reset behavior is
  explicit `autoreset_done_rows()` after terminal rows, with final metadata
  preserved for callers to inspect.
- Do not claim trainer-ready match-mode public multiplayer episodes. Public
  metadata now reports focused 3P round/match split facts, narrow final-row
  policy, explicit round/match episode policy in `VectorMultiplayerEnv`,
  one explicit fixture-backed warmdown-frame survivor death/no-rescore proof,
  reset-owned lifecycle arrays, explicit reset-vs-round identity metadata,
  metadata-only replay records, and explicit terminal-row autoreset, but hidden
  autoreset, production replay, and full lifecycle parity are still missing.
  Ordinary `step()` remains blocked while `warmdown_pending=true`; use
  `advance_warmdown_frame(...)` only as a metadata-only bridge for the named
  survivor warmdown proof.
- Do not turn RNG provenance into a training-stochasticity subsystem. Training
  should use source-like randomness; controlled/replayable seed/source/cursor
  metadata is for fidelity checks, replay audit, and debugging.
- Keep future training stochasticity or domain randomization behind explicit
  config, and keep it disabled for source-fidelity tests unless a fixture
  explicitly opts in.
- Do not claim bonuses, visuals, browser pixel fidelity, LightZero integration,
  or whole-loop self-play speed.
- Seeded public bonus support is narrow: fixture-seeded `BonusSelfSmall`,
  `BonusGameClear`, and `BonusGameBorderless` through public tests. Do not
  widen that to natural bonuses, replay, or broad effects. Focused public
  natural `BonusSelfSmall` spawn is partial only. The low-level spawn helper is
  landed, but public natural spawn timer ownership/scheduling/random
  accounting, borderless stack semantics, remaining runtime/public effects, and
  full replay/final state remain gaps. Source/runtime/public borderless
  duration/expiry itself is no longer missing.
- Do not claim the Modal/source-state LightZero path is two-seat self-play or
  browser pixel truth. It is a fixed-opponent source-state visual wrapper backed
  by `VectorMultiplayerEnv`.
- Do not treat strict 1v1/no-bonus as the destination. It is only the first
  well-proven trainer boundary.
- Do not treat direct seeded runtime canaries as natural reset/warmup,
  warmdown, replay, or observation parity.
- Do not treat `step`, `joint_action`, or fixed decision cadence as native
  CurvyTron semantics. Source-native state is held real-time control state
  advanced by elapsed-ms server frames; `step` and `joint_action` are
  CurvyZero wrapper/replay abstractions.

## Stale Language To Fix When Touching Nearby Docs

- Current wording to use: metadata-only public 2P/3P/4P env exists;
  source-backed lifecycle, replay, and trainer observation parity are missing.
- Continue `VectorMultiplayerEnv` in
  `src/curvyzero/env/vector_multiplayer_env.py`; add trainer wrappers above it
  only after metadata parity is stable.
- Strict 1v1/no-bonus is a proof boundary; multiplayer remains the target.
- Keep `joint_action` wording tied to wrapper/replay sidecars, not native source
  objects.
