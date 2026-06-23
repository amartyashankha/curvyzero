# Survival Mechanics And Reward Audit - 2026-05-16

Scope: focused read of the current working tree around survival rewards, bonus ownership, policy observations/metadata, tournament policy loading/action selection, and seat handling. I did not do a repo-wide audit. The worktree was already dirty, so these findings are about the code currently checked out in `/Users/shankha/curvy`.

## Findings

### 1. Likely training-risk bug: source-state dense rewards are much larger than the LightZero support cap

The active curvytron contract uses `CURVYTRON_SOURCE_MAX_STEPS = 1_048_576` and `CURVYTRON_DECISION_SOURCE_FRAMES = 1` (`src/curvyzero/contracts/curvytron.py:74-76`). For source-state fixed-opponent rewards, `_lightzero_target_config_for_reward` computes support scales from `source_max_steps`, but then caps both reward and value support at `SOURCE_STATE_FIXED_OPPONENT_MAX_MODEL_SUPPORT_SCALE = 300` (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:435`, `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:749-807`).

Concrete impact:

- `survival_plus_bonus_no_outcome`: immediate reward scale is small, but requested value scale is `source_max_steps * (1 + bonus_reward)`, so about `2,097,152` for r18fresh. It is capped to `300`.
- `survival_plus_bonus_plus_outcome`: requested reward scale is about `1,048,578`, because terminal outcome is step-count scaled; requested value scale is about `3,145,728`. Both are capped to `300`.

This does not prove mechanics are wrong, but it is a high-signal explanation for mixed survival training: the reward stream can be million-scale while the model target support is deliberately compressed to `[-300, 301)`. If LightZero target projection saturates at the support edge, terminal/value magnitudes are not represented faithfully.

### 2. Contract footgun: plus-outcome terminal reward says source steps, implementation uses wrapper physical-step index

The schema says the plus-outcome terminal term is `"terminal_sparse_outcome_scaled_by_episode_source_step_count"` and `"sparse_round_outcome * episode_source_step_count"` (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:374-391`). The implementation uses `self._physical_step_index` when `batch.done[0]` and sparse outcome is nonzero (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1760-1779`), incremented once per wrapper/vector step (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1003-1019`).

For r18fresh this is exonerated by contract because `CURVYTRON_DECISION_SOURCE_FRAMES = 1` (`src/curvyzero/contracts/curvytron.py:74-75`). If a future run uses multiple source frames per decision/action-repeat, the name/schema and value can drift.

### 3. Exonerated: bonus pickup does not appear to apply to the wrong player

The source reference path targets the catcher for self bonuses, alive non-catchers for enemy bonuses, and all alive players for all-player bonuses. `_test_bonus_catch` removes the caught bonus, then branches by type: `BonusSelf*` applies to `[avatar]`, `BonusEnemy*` applies to alive targets with `target.id != avatar.id`, and `BonusAll*` applies to all alive avatars (`src/curvyzero/env/source_env.py:1482-1510`). `_apply_bonus_to_avatars` records `target_ids`, appends the bonus to each target's active stack, and resolves that target's effects (`src/curvyzero/env/source_env.py:1527-1545`). The effect table itself matches the expected semantics: self small/slow/fast/master, enemy slow/fast/big/inverse/straight-angle, all color (`src/curvyzero/env/source_env.py:2061-2083`), and application mutates the passed avatar (`src/curvyzero/env/source_env.py:1690-1727`).

The vector runtime mirrors this. The effect table marks self bonuses as `BONUS_TARGET_SELF`, enemy bonuses as `BONUS_TARGET_ENEMY`, all-color as `BONUS_TARGET_ALL`, and game bonuses separately (`src/curvyzero/env/vector_runtime.py:247-330`). `_catch_bonus_batched` checks collision for the current `player`, increments `state["bonus_catch_count_step"][row_int, player]`, removes the active bonus, computes target players from the target group, then applies stack/effects to each `target_player` (`src/curvyzero/env/vector_runtime.py:2847-3054`). `_bonus_target_players` returns `(catcher,)` for self and excludes the catcher for enemy (`src/curvyzero/env/vector_runtime.py:3057-3080`).

Step-level accounting also lines up: the public step clears `bonus_catch_count_step` once before stepping (`src/curvyzero/env/vector_multiplayer_env.py:1026-1030`), accumulates through the decision source-frame loop (`src/curvyzero/env/vector_multiplayer_env.py:1167-1202`), and exposes the `[batch, player]` count in public info (`src/curvyzero/env/vector_multiplayer_env.py:3181-3184`).

### 4. Exonerated: reward ownership uses ego player/catcher count

The source-state reward schema explicitly defines the bonus pickup source as `bonus_catch_count_step[0, ego_player_index]` (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:340-353`, `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:374-387`). During `step`, the wrapper writes the learner action into `joint_action[0, self.ego_player_index]`, the opponent action into the opponent index, and computes reward components for `player_index=self.ego_player_index` (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:980-1021`).

The reward helper validates `bonus_catch_count_step` shape and reads `[0, player_index]` (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1702-1716`). Sparse outcome is also read from `batch.reward[0, player_index]`, and dense survival checks `self._env.state["alive"][0, player_index]` (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1698-1700`, `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1730-1779`). Step info records `ego_player_index`, `controlled_player_id`, `reward_player_id`, and the ego bonus/reward breakdown (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1928-1989`).

### 5. Exonerated: model observations are player-perspective and sidecar metadata is rich enough

The training wrapper returns a LightZero observation containing the FIFO stack, action mask, `to_play=-1`, and timestep (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1349-1356`). `_update_stack` renders player-perspective frames for both seats and copies the ego frame into the learner stack; the GPU path uses `render_player_perspectives`, the CPU dirty-cache path renders both controlled-player palettes, and the fallback renders ego/opponent separately with `controlled_player=self.ego_player_index` / opponent (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1358-1459`).

The wrapper's base info includes the policy observation contract, backend, perspective schema/owner, ego/opponent perspective indices, trail/bonus render modes, reward schema, reward variant, and learner seat mode (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:2017-2225`). Checkpoint sidecars preserve the same policy surface plus model env/reward variants and runtime cadence (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2105-2153`) and are written next to each checkpoint as `<checkpoint>.metadata.json` (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2168-2188`).

Caveat: `raw_observation(player_perspective=True)` currently ignores the flag and returns the latest raw RGB canvas (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1116-1131`). That looks like a debug/render artifact caveat, not a model-input bug, because the policy stack path above is player-perspective.

### 6. Exonerated: tournament eval loads policy contracts and selects actions from the correct seat observation

Tournament checkpoint refs first look for the durable checkpoint metadata sidecar, then attempt/run metadata (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:176-209`). The loader extracts policy trail mode, bonus mode, backend, runtime settings, and model env/reward variant from checkpoint payload/config/metadata/sidecars (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2783-2998`). `_load_policy_from_checkpoint` passes the recovered `model_env_variant` and `model_reward_variant` into eval `_make_policy_and_env` (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3013-3135`).

Inside `_make_policy_and_env`, the model variant/reward variant are used only to rebuild the checkpoint model target shape. If they differ from the runtime eval env/reward, it calls `_lightzero_target_config_for_reward` for the model contract and patches the compiled config before instantiating/loading the MuZero policy (`src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:688-851`). It also infers support sizes from the state dict and patches those if present (`src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:649-685`, `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:781-795`).

Action selection uses the observation/action mask/to_play expected by LightZero. Eval mode wraps `observation["observation"]` with a batch dim, passes `action_mask`, `to_play`, and `ready_env_id`, then extracts the action (`src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:607-631`). Tournament collect mode does the same with `policy.collect_mode.forward`, including temperature and epsilon (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3243-3292`).

During games, tournament builds one `SourceStateGray64Stack4` per required policy surface (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3569-3588`). Each step, seat `N` receives `observation[0, N]`, the corresponding `batch.action_mask[0, N]`, and controls `actions[0, N]` (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3615-3654`). The summary records the invariant: `"seat N receives observation[0,N] and controls player N"` (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3768-3815`). The stack renderer itself stores frames as `[batch, player, stack...]` and renders each player with `controlled_player=player` (`src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py:341-425`, `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py:451-484`).

### 7. Exonerated: seat balancing and preloaded policy ordering handle swaps

Balanced/random seat order creates a shuffled list of swaps, and `seat_order_for_game` maps physical seats to logical players (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:616-652`). `build_game_specs_for_pair` rewrites each game's `players` so physical seat `N` gets the checkpoint for the logical player assigned to that seat (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:685-710`).

If policies are preloaded, `_preloaded_policy_entries_for_players` reorders entries by checkpoint ref and state key to match the current game's seat-ordered player list (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3138-3170`). Raw checkpoint specs get default policy surfaces before loading, but once loaded, checkpoint metadata wins for those raw specs (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:243-265`, `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3173-3183`).

## Focused Verification Run

Passed:

```text
uv run pytest \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_blank_canvas_noop_steps_without_player_1_artifacts_or_terminal \
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_policy_metadata_sidecar_overrides_run_metadata \
  tests/test_curvytron_checkpoint_tournament.py::test_loaded_policy_metadata_wins_for_raw_checkpoint_specs \
  tests/test_curvytron_checkpoint_tournament.py::test_policy_loader_recovers_model_contract_from_checkpoint_metadata \
  tests/test_curvytron_checkpoint_tournament.py::test_preloaded_policy_entries_reorder_to_actual_game_seats \
  -q

5 passed in 0.34s
```

## Bottom Line

I did not find evidence that bonus pickup is credited to the wrong player, that source-state reward components are read from the wrong seat, or that tournament eval feeds a policy the wrong seat/perspective observation. The main actionable risk is reward/value support scaling: the active survival/bonus reward regimes can produce million-scale returns, while source-state fixed-opponent LightZero targets are capped at support scale `300`. The plus-outcome step-count implementation is also only clean under the current `decision_source_frames=1` contract.
