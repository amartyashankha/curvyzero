# Observation And Reward Design

Status: Proposed
Date: 2026-05-08

Scope: first simulator and baseline learning gates for `curvyzero-v0`.

## Short Answer

Use one fixed ego-perspective action and reward contract across random,
heuristic, PPO, and later MuZero work:

- Action space: fixed `A = 3` ordered as `left`, `straight`, `right`; use masks
  for rulesets that disable an action instead of silently changing model shape.
- v0 learned observation: ego-relative ray features plus small scalar features
  and `legal_action_mask`. This is the fastest way to debug PPO and imitation.
- v1 learned observation: ego-centered, heading-aligned local occupancy raster
  plus the same scalar and mask fields. This is the better default for CNN
  policies and MuZero representation/dynamics work.
- Reward: sparse terminal ego payoff first. In 1v1, win `+1`, loss `-1`,
  same-tick tie `0`, timeout as `truncated` with no terminal payoff.
- Replay: always store schema hashes, ego player id, perspective transform,
  legal mask, joint actions, reward-after-action alignment, death/tie metadata,
  and opponent policy/checkpoint ids.

The current flat global state vector in `CurvyTronEnv._observations()` is useful
for deterministic tests and privileged debugging, but it should not be the first
learned-policy contract. It omits trail topology, exposes absolute coordinates,
and does not match the ego-perspective shared-policy plan.

## Constraints From Current Docs

- v0 is 1v1, one round per episode, no bonuses, deterministic seeds,
  wrapper-level simultaneous decisions, fixed action repeat, and terminal
  outcome reward. Source CurvyTron itself stores held player controls and
  advances elapsed-ms server frames; it does not expose this trainer wrapper.
- The training plan requires random stress, heuristic-vs-random, then PPO before
  serious MuZero work.
- The multiplayer decision record chooses shared ego-perspective policy/value
  learning, with scalar ego payoff and checkpoint/opponent metadata in replay.
- MuZero notes require fixed observation/action shapes, stable legal masks,
  reward alignment tests, and replay records that identify rules, observation,
  reward, search, model, and perspective versions.
- Performance notes favor occupancy-grid collision for v0 and warn that
  observation generation can dominate stepping if local transforms are too
  expensive.

## Observation Options

| Option | Strengths | Risks | Verdict |
| --- | --- | --- | --- |
| Global state vector | Cheap, easy to inspect, good for deterministic tests and privileged oracle heuristics. | Leaks seat/absolute position, is not translation/rotation invariant, currently omits occupancy/trail topology, and can produce policies that overfit spawn layout. | Keep as `oracle_global_debug`, not as the default learned observation. |
| Egocentric rays | Small, normalized, MLP-friendly, fast enough for PPO, directly supports wall/trail clearance heuristics, and is easy to unit test. | Can miss topology behind the ego, narrow corridors, and curved future collisions unless ray count/channels are chosen carefully. | Recommended v0 learned observation. |
| Local raster | Preserves nearby geometry, works naturally with CNNs and MuZero spatial latents, shares data structure with occupancy collision, and is easy to render/debug. | Crop rotation and interpolation can be costly; crop size may hide long-horizon walls or traps; schema is larger than rays. | Recommended v1 and MuZero observation. |
| Full occupancy raster | Complete arena information and simple debugging for small maps. | Memory-heavy, encourages absolute-coordinate overfit unless transformed, scales poorly for vector envs/replay, and increases model/search cost before proving benefit. | Defer. Use for debug visualizations or an oracle ablation only. |

## v0 Observation Schema: Rays

Original proposed schema id: `curvyzero-observe-v0-rays`. The pinned working
contract now uses canonical id `curvyzero_egocentric_rays/v0`; keep the older
name as a legacy alias only.

Use this for the first heuristic/imitation/PPO gates. The core simulator can
return a structured observation; wrappers may flatten it for SB3/CleanRL.

```text
ObservationV0Rays:
  rays          float32[24, 4]
  scalars       float32[10]
  action_mask   bool[3]
```

Coordinate convention:

- Ego head is the origin.
- Ego heading points along local `+x`.
- Local `+y` is ego-left.
- Distances are clipped and normalized to `[0, 1]`; no hit is `1.0`,
  immediate contact is `0.0`.

Ray field:

- 24 fixed angles around the ego, initially uniform over 360 degrees.
- Channels: wall, own trail, opponent trail, opponent head.
- Ray casting should use the same occupancy/collision grid semantics as the
  simulator, not rendered pixels.
- If the first PPO fails while privileged heuristics succeed, try 32 rays with
  extra forward/side density before adding reward shaping.

Scalar field:

```text
0  ego_alive
1  opponent_alive
2  tick_fraction
3  opponent_rel_x_clipped
4  opponent_rel_y_clipped
5  opponent_heading_sin_relative
6  opponent_heading_cos_relative
7  speed_norm
8  turn_rate_norm
9  trail_radius_norm
```

Notes:

- Do not include stable color, player index, or uncanonicalized seat id.
- Avoid absolute `x/y` as a default learned feature. Rays already expose wall
  distance in ego coordinates.
- Constant scalars such as speed and turn rate are still useful because schema
  compatibility can survive curriculum sweeps later.
- Include `tick_fraction` primarily to diagnose timeout behavior. If policies
  overfit to late-episode clocks, remove it in a new schema version.

## v1 Observation Schema: Local Raster

Schema id: `curvyzero-observe-v1-local-raster`.

Use this when moving from vector PPO to CNN policies or MuZero.

```text
ObservationV1Raster:
  planes        float32[5, 48, 48]
  scalars       float32[10]
  action_mask   bool[3]
```

Raster convention:

- Crop is centered on the ego head and rotated so ego heading points toward
  decreasing raster row. Ego-left maps toward decreasing raster column.
- Pixel scale maps one simulator occupancy cell to one raster cell for the
  first version. If resampling is needed later, that is a new schema.
- Out-of-arena cells are encoded explicitly in the wall channel.

Channels:

```text
0 wall_or_out_of_bounds
1 own_trail
2 opponent_trail
3 ego_head
4 opponent_head
```

Add recent trail age, candidate swept cells, or union occupancy only in a new
schema version after measuring a concrete failure.

MuZero-specific notes:

- The model does not need to reconstruct the next raster. It needs the raster
  to support policy, value, and reward predictions through the latent dynamics.
- Keep `action_mask` outside the raster planes and feed it to search/policy loss.
- Start with one frame. Add a short history stack only if trail holes, delayed
  effects, or action-repeat artifacts make the current frame insufficient.

## Action Masks

Use the mask even when every live-player action is currently legal:

```text
action_id 0 = left
action_id 1 = straight
action_id 2 = right
```

Mask rules:

- Alive player in normal v0: `[true, true, true]`.
- Strict left/right ruleset: `[true, false, true]`.
- Dead or terminal padded replay row: `[false, false, false]`, with losses
  masked out for policy/value targets after terminal.
- If a wrapper requires a no-op after death, keep that as wrapper machinery and
  do not train the policy on dead-player action targets.

PPO needs:

- If using a library without masked action distributions, sample only from legal
  actions in the wrapper and still log the mask for debugging.
- If using MaskablePPO or a custom distribution, invalid actions must receive
  zero probability and must be excluded from entropy/action-logprob terms.

MuZero needs:

- Mask invalid actions at the root and recurrent nodes.
- Store `root_action_weights[3]` with invalid actions set to `0`.
- Normalize search policy targets over valid actions only.
- Keep the action count static across rulesets or record a different
  `action_schema_hash`.

## Reward Schemes

| Scheme | Strengths | Risks | Verdict |
| --- | --- | --- | --- |
| 1v1 terminal win/loss/tie | Clean, easy to test, compatible with PPO and MuZero, avoids hidden incentives. | Sparse for PPO and value learning. | v0 default. |
| Centered rank payoff | Extends to 3+ players, handles tied death ranks, keeps ego value scalar. | Requires careful tie metadata and no sign-flip shortcuts in 3+ player games. | v1 multiplayer default. |
| Survival tick bonus | Can reduce sparse-reward pain. | Rewards stalling, wall-hugging, and timeouts unless very carefully bounded. | Not default. Add only after diagnosing PPO failure. |
| Collision penalty | Intuitive and often redundant with terminal loss. | Can double-scale terminal loss and obscure outcome reward. | Avoid in v0 unless terminal loss is not used. |
| Clearance/progress shaping | Encourages safe driving and helps imitation-like behavior. | Bakes the heuristic into the reward, can punish tactical traps, and may fight winning. | Keep in heuristic scoring, not environment reward. |
| Turn penalty | Can smooth motion. | Teaches under-turning in a game where sharp turns are often necessary. | Use only as a heuristic tie-breaker. |
| Kill/pressure reward | May help multiplayer credit assignment. | Ambiguous in simultaneous collisions and can encourage kingmaking or griefing. | Defer until after 1v1 and rank payoff work. |

## v0 Reward Schema

Schema id: `curvyzero_sparse_round_outcome/v0`.

```text
episode_unit: one_round
perspective: ego_player
reward_alignment: reward_{t+1}_after_wrapper_joint_action_t
terminal_payoff:
  ego_survives_opponent_dies: +1.0
  ego_dies_opponent_survives: -1.0
  both_die_same_terminal_tick: 0.0
  both_alive_at_time_limit: no terminal reward, truncated=true
per_decision_nonterminal_reward: 0.0
discount: trainer/search config, not hidden in env reward
shaping_terms: []
```

Timeouts must not be silently treated as normal draws. Log them separately and
count a high timeout rate as an environment or curriculum failure until a later
ruleset explicitly accepts timeout scoring.

## v1 Reward Schema

Schema id: `curvyzero-reward-v1-centered-rank`.

For `N` players, assign each player a rank score in `[0, N - 1]`:

- First death gets the lowest score.
- Later death gets a higher score.
- Final survivor gets `N - 1`.
- Players dying on the same tick share the average of the score slots they
  occupy.
- If all remaining players die on the same terminal tick, they tie over the
  remaining score slots.

Convert to ego payoff:

```text
payoff_i = 2 * rank_score_i / (N - 1) - 1
```

For 1v1 this gives `+1`, `-1`, and `0` for exact ties. For 3+ players, do not
derive one player's reward by sign-flipping another player's scalar. Store each
ego payoff explicitly.

## Tie Handling

Tie policy must be deterministic and order-independent:

- Movement and collision detection happen in phases for all alive players.
- Same-tick collisions are resolved from the pre-resolution collision set, not
  by iterating players and mutating occupancy in seat order.
- 1v1 same-tick death is a draw with reward `0` for both players.
- Head-head or crossed-segment conflicts should be tested separately from old
  trail collisions, even if v0 approximates them through occupancy cells.
- Replay records should store `death_tick`, `death_cause`, `terminal_rank_score`,
  and `tie_group_id` or equivalent fields.

This matches the spirit of the source-derived scoring note that simultaneous
deaths in one server frame receive the same captured round score, while keeping
the v0 environment simpler than the browser game's full match scoring.

## Shaping Risks

Dense shaping should be treated as an experiment, not as a quiet fix.

Common failure patterns:

- Positive survival reward can make timeout farming better than decisive wins.
- Per-step negative reward can make fast suicide attractive if losing is likely.
- Clearance reward can prevent useful trapping behavior near trails.
- Turn penalties can hide action-mapping bugs by making one action dominate.
- Opponent-distance rewards can create policies that chase instead of survive.
- Scaling a shaping term by physics tick instead of decision step changes the
  effective objective when `action_repeat` changes.

If shaping is added, create a new reward schema id and store:

```text
term_name
weight
units: per_physics_tick or per_decision
clip_range
terminal_scale_ratio
enabled_rulesets
reason_added
removal_gate
```

Keep total shaping small relative to terminal payoff, and evaluate the shaped
policy under the unshaped terminal metric.

## PPO Versus MuZero Requirements

PPO baseline:

- Needs fast observation generation over many CPU envs.
- Prefers normalized vector features first, making rays a better v0 fit than
  rasters.
- Needs correct time-limit handling: `terminated` for game outcome,
  `truncated` for max tick, and terminal observation recorded by wrappers.
- Can use sparse reward for the gate, but if it fails, diagnose observation,
  action mapping, entropy, wrapper autoreset, and curriculum before shaping.
- Should log action histograms, value predictions, logprobs, entropy, masks, and
  terminal causes.

MuZero:

- Needs fixed action count, fixed observation shape, legal masks, and exact
  reward-after-action alignment.
- Benefits from local raster observations because CNN representation and action
  planes fit the learned dynamics path.
- Stores root search policy and root value for every ego timestep.
- Requires joint actions or opponent action metadata because real environment
  steps are simultaneous even if search controls only the ego action.
- Should mask root reward loss at unroll depth `0` and train recurrent rewards
  against `reward_{t+k}` for `k > 0`.

Both:

- Need ego-perspective canonicalization.
- Need schema hashes in replay and checkpoints.
- Need fixed and held-out seed evaluation with swapped seat assignments.
- Need observation and reward tests before learning curves are trusted.

## Tests To Add

Observation tests:

- Shape, dtype, range, and key stability for every schema.
- Same seed/state/action trace produces byte-stable observations.
- Swapping seats and applying the ego transform produces mirrored/canonical
  observations with no stable seat leak.
- Known wall/trail fixtures produce expected ray distances.
- Known occupancy fixtures produce expected local raster planes after rotation.
- `observe_many` matches repeated single-env observation generation.
- Observations do not mutate simulator state.
- Schema hash changes when ray count, channels, crop size, scalar order, action
  order, or normalization changes.

Mask tests:

- Normal v0 live mask is all true.
- Strict left/right ruleset masks straight without reordering left/right.
- Dead/terminal padded rows are excluded from policy losses.
- Search/PPO sampling never emits a masked action.
- Stored search policy has zero mass on invalid actions and sums to one over
  valid actions.

Reward and tie tests:

- Ego win, loss, same-tick tie, wall death, self-trail death, opponent-trail
  death, head-head death, and timeout truncation.
- Simultaneous deaths are independent of player iteration order.
- Reward is emitted on `t+1` after wrapper `joint_action_t`.
- Action repeat emits at most one decision-level terminal payoff.
- N-player average tied ranks produce centered payoffs.
- PPO/Gym wrapper distinguishes `terminated` from `truncated` and preserves
  terminal observation.

Replay/target tests:

- Deterministic replay reconstructs an episode from seed, config, and joint
  action trace.
- MuZero target construction catches reward off-by-one errors.
- Post-terminal padding masks policy, value, and reward losses correctly.
- Replay readers reject mismatched rules, observation, reward, action, or search
  schema hashes.

## Replay Metadata

Chunk-level metadata:

```text
run_id
chunk_id
created_at
code_version_or_commit
ruleset
rules_hash
collision_backend
observation_schema_id
observation_schema_hash
reward_schema_id
reward_schema_hash
action_schema_id
action_schema_hash
action_repeat
player_count
max_ticks
seed_range_or_episode_seeds
library_versions
```

Episode-level metadata:

```text
episode_id
reset_seed
rng_state_or_spawn_record
spawn_positions
spawn_headings
opponent_policy_assignments
timeout_tick
terminal_tick
outcome_by_player
death_tick_by_player
death_cause_by_player
rank_score_by_player
```

Per ego timestep:

```text
tick_before_action
ego_player_id
perspective_transform_id
observation_or_observation_ref
legal_action_mask
ego_action
joint_action
opponent_actions
reward_next
terminated_next
truncated_next
info_death_cause_next
terminal_rank_score_next
```

PPO fields when relevant:

```text
policy_checkpoint_id
opponent_checkpoint_id
action_logprob
policy_entropy
value_prediction
episode_return_so_far
```

MuZero/search fields when relevant:

```text
model_step_used_for_search
search_config_hash
root_action_weights
root_value
raw_network_value
search_temperature
root_exploration_noise_seed
num_simulations
opponent_policy_versions
```

Store complete joint actions even for ego-vs-scripted PPO. Without joint action
history, simultaneous collision bugs and deterministic replay failures are much
harder to diagnose.

## Recommendation

For the first simulator/baselines:

1. Keep the current global vector only as a debug/oracle observation.
2. Add `curvyzero_egocentric_rays/v0` for heuristic imitation and PPO
   (`curvyzero-observe-v0-rays` is the legacy research alias).
3. Add `curvyzero_sparse_round_outcome/v0` with no shaping.
4. Add action masks immediately, even though v0 actions are usually all legal.
5. Add observation/reward/replay metadata tests before comparing agents.
6. Promote `curvyzero-observe-v1-local-raster` when starting CNN policy or
   MuZero work.

The main practical bet is that a ray-based PPO gate will expose simulator,
wrapper, action, and reward bugs faster than a raster/CNN stack, while the v1
local raster keeps the path to MuZero clean once the baseline gate passes.

## Sources

- `docs/design/deterministic_environment.md`
- `docs/design/training_architecture.md`
- `docs/design/rulesets.md`
- `docs/decisions/0003-multiplayer-selfplay-v0-formulation.md`
- `docs/research/baseline_learnability.md`
- `docs/research/multiplayer_selfplay_muzero.md`
- `docs/research/muzero_architecture_deep_dive.md`
- `docs/research/performance_vectorization.md`
- `docs/research/curvytron_reference_notes.md`
- `curvytron_muzero_modal_handoff.md`
