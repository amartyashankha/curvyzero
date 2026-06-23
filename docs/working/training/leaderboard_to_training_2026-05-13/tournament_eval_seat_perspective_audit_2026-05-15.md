# Tournament eval seat/perspective audit - 2026-05-15

## Plain-language verdict

Tournament evaluation is using the right per-seat visual perspective: seat 0 gets the player-0 slice of `SourceStateGray64Stack4`, and seat 1 gets the player-1 slice. That stack is explicitly player-perspective: each controlled player is rendered as SELF and the other player as OTHER. The tournament now keeps LightZero `to_play=-1`, validates the seat's legal action mask, and executes a real joint action `[seat0_action, seat1_action]`.

The ratings are therefore not broken because tournament gives seat 1 a player-1 POV. That is the correct eval contract for a true two-seat policy.

The historical real18/v2 risk was training/eval distribution mismatch. If a
checkpoint was trained only as ego/player 0, then its seat-1 tournament games
are out-of-distribution. Fresh restart checkpoints should use
`learner_seat_mode=random_per_episode`, so tournament balanced seating becomes
the correct deployment/eval contract rather than an invalid distribution shift.

## Evidence

- Tournament loads two policies, creates one 2-player `VectorMultiplayerEnv`, and creates `SourceStateGray64Stack4(batch_size=1, player_count=2)` surfaces keyed by each checkpoint's requested render modes. See `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3106`, `:3226`, and `:3237`.
- In the game loop, tournament reads `observation[0, seat]`, `batch.action_mask[0, seat]`, sets `"to_play": -1`, calls the matching seat policy, validates legality, and writes `actions[0, seat]`. The seat is encoded by the selected observation slice and action mask, not by LightZero `to_play`.
- `SourceStateGray64Stack4` stores shape `[B, P, 4, 64, 64]`, shifts every player stack together, and renders raw frames for each controlled player. The two-seat path calls `render_source_state_canvas_gray64_player_perspectives(...)` with `controlled_player=player` palettes before writing `stack[env_row, player, -1]`. See `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py:347`, `:425`, and `:477`.
- Historical two-seat smoke code still contains player-id `to_play` paths. Do not use those as the current `source_state_fixed_opponent` contract. The current fixed-opponent LightZero env reports `to_play=-1`; seat identity lives in controlled-player observation/action-mask metadata.
- Current source-state visual survival training no longer takes
  `ego_player_index` config. It uses `learner_seat_mode`; fresh runs default to
  `random_per_episode`, while `fixed_player_0` and `fixed_player_1` are explicit
  diagnostics. Its LightZero observation still has `to_play=-1`, so the seat
  assignment must be recorded through reset/step metadata rather than inferred
  from `to_play`.
- Frozen checkpoint opponent inference is visually seat-aware because the wrapper builds both controlled-player stacks and selects the opponent slot. The old drift risk around LightZero `to_play` has been repaired: `LightZeroCheckpointOpponentProvider` ignores physical `player_id` for LightZero `to_play` and forwards `to_play=[-1]` for the current non-board-game contract. Seat identity lives in the selected observation/action-mask slice.
- New checkpoint metadata hardening is in progress locally: fresh checkpoints now write a small `iteration_N.pth.tar.metadata.json` sidecar with policy observation backend, trail render mode, bonus render mode, observation contract id, runtime timing, model env/reward variants, and learner seat mode. Tournament checkpoint discovery and policy loading read that sidecar before falling back to run/attempt metadata or defaults.

## Action Semantics

The action space is shared and seat-independent: `0=left`, `1=straight`, `2=right`, mapped to source moves `(-1, 0, 1)`. For live players, the legal mask is all three actions. The tournament does not use training's policy no-op skip machinery; each live seat must provide a legal action every tournament step. Action `1` is both the straight action and the project's padding/no-op convention for inactive or skipped slots, but in tournament it is a normal legal "go straight" choice.

This makes it unsafe to "fix" eval by feeding both policies the literal player-0 observation. If seat 1 received player 0's visual slice, it would see the wrong body as SELF while its action would still be applied to seat 1's physical heading. That would mix observation ownership and actuator ownership. The safer fix is to train the policy on the same distribution eval uses: player-perspective observations for both seats, or randomized/balanced seat assignment during training. If a diagnostic needs player-0-only strength, create a separate player-0-only eval lane and do not merge it into the tournament rating.

## Greedy vs noisy eval

Tournament defaults to `policy_mode="eval"` (`DEFAULT_COLLECT_TEMPERATURE=1.0` and `DEFAULT_COLLECT_EPSILON=0.25` only matter if `policy_mode="collect"`). In eval mode the tournament calls `eval_mod._policy_eval_action(...)`; in collect mode it calls `policy.collect_mode.forward(..., temperature, epsilon)`. See `src/curvyzero/tournament/curvytron/contracts.py:44` and `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2928`.

For ratings, use eval/greedy MCTS with fixed search budget, no root exploration noise, no epsilon, and no sampling. That matches AlphaZero/MuZero practice: training self-play uses search distributions and exploration, while match/evaluation play is deterministic/greedy from search under a fixed thinking budget. AlphaZero's paper describes search output as proportional or greedy by visit counts and reports tournament evaluation games with fixed time controls; LightZero's MuZero eval path documents `prepare_no_noise` and deterministic argmax during evaluation, while collect mode adds Dirichlet noise and samples. Sources: AlphaZero arXiv paper lines 26-28 and tournament table lines 53-62 at https://arxiv.org/abs/1712.01815; LightZero MuZero docs lines 716-750 and 820-884 at https://www.aidoczh.com/lightzero/_modules/lzero/policy/muzero.html.

Collect/noisy tournament games can be useful as diagnostics, but they should not share the same rating context as deterministic eval. The existing `rating_context_hash` includes `policy_mode`, temperature, epsilon, timing, max steps, simulations, and render modes, which is the right direction.

## Runtime and render parity

Tournament runtime is mostly guarded. It reads checkpoint runtime settings, requires consistent `decision_source_frames`, `decision_ms`, and `source_physics_step_ms`, and converts tournament `max_steps` into `source_max_ticks = max_steps * decision_source_frames`. See `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2970` and `:2997`.

The default tournament max step count is extremely high (`1_048_576`), so normal games should end by death, not timeout. That is fine for ratings if all checkpoints use the same runtime contract, but parity tests should assert training/eval max-tick meaning explicitly because the older envs use names like `source_max_steps` and `max_ticks`.

Policy observations and GIFs are intentionally different surfaces. The policy sees per-seat `SourceStateGray64Stack4` with each checkpoint's policy render modes. The GIF sees a full 704 RGB browser-lines canvas for humans. That is acceptable, but tests must prevent a future GIF/render refactor from accidentally becoming the policy surface.

## Recommendation

1. Treat current tournament eval as correct for two-seat checkpoints, but do not use it to claim clean seat-invariant strength for checkpoints trained only as player 0.
2. Make the training side match eval: train on both player-perspective seats, or randomize/balance controlled seat assignment. This is safer than forcing tournament eval into player-0 POV because action semantics are already actor-relative and the visual SELF/OTHER ownership would be wrong.
3. For public ratings, keep `policy_mode="eval"` and record it prominently. No exploration noise, no epsilon, no action sampling.
4. Add a seat-balance guard to ratings: either schedule both ordered seat assignments for every unordered matchup or aggregate paired games where each checkpoint plays both seats. This protects against accidental seat-specialization and roster-order bias.

## Tests to add

- `SourceStateGray64Stack4` parity test: from a seeded two-player state, assert `stack[0,0]` marks player 0 as SELF and player 1 as OTHER, while `stack[0,1]` marks player 1 as SELF and player 0 as OTHER.
- Tournament policy-input spy: with fake policies for both seats, assert seat 0 receives `observation[0,0]`, `to_play=-1`, and `action_mask[0,0]`; seat 1 receives `observation[0,1]`, `to_play=-1`, and `action_mask[0,1]`.
- Seat-swap rating test: for two deterministic fake checkpoints, build/evaluate both A-vs-B and B-vs-A seat assignments and assert the rating aggregator can either require or preserve both directions.
- Training replay parity test: current source-state fixed-opponent training must keep LightZero `to_play=-1` while recording the selected controlled player in metadata.
- Frozen opponent provider test: when the frozen slot is player 1 or player 0, assert provider receives that slot's observation slice and still forwards `to_play=[-1]`.
- Eval mode test: tournament default must call eval mode, not collect mode; collect mode must be opt-in and produce a different rating context hash.
- Noise test: rating eval must reject or clearly isolate nonzero collect `epsilon`, stochastic collect sampling, action no-op probability, and policy action repeat/no-op skip.
- Action semantics test: assert `0/1/2 -> left/straight/right`, live masks are all true, and action `1` is not removed from tournament legal actions.
- Runtime parity test: checkpoint runtime metadata mismatch for `decision_source_frames`, `decision_ms`, or `source_physics_step_ms` must fail fast; `source_max_ticks` must equal `max_steps * decision_source_frames`.
- Render surface test: assert policy render modes come from checkpoint/pair policy metadata and GIF render mode remains human-only `browser_lines`.
