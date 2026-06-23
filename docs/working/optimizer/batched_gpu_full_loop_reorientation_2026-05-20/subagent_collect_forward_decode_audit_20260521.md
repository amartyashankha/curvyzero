# collect_mode.forward input/decode audit - 2026-05-21

Scope: audited `policy.collect_mode.forward` call sites under `src/curvyzero` plus the CurvyTron/Pong LightZero action decode helpers. No live training runs were touched.

## Direct CurvyTron collect call sites

### Background GIF collect path

File refs:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9850`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9879`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9888`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9896`

Input format:

- Source observation dict is single-root/per-seat: `observation["observation"]` is wrapped as `np.asarray([observation["observation"]])`, then `torch.as_tensor(..., dtype=torch.float32, device=eval_mod._policy_model_device(policy))`.
- Effective `obs_tensor` shape is `[1, 4, 64, 64]` for the current stacked visual surface. The stack contract is `(4, 64, 64)` in `src/curvyzero/training/curvytron_visual_observation.py:27`.
- `action_mask = np.asarray([observation["action_mask"]], dtype=np.float32)`, effective shape `[1, 3]`.
- `to_play = [int(np.asarray(observation.get("to_play", -1)).reshape(-1)[0])]`, a Python list of length 1.
- `ready_env_id = np.asarray([0])`, NumPy integer array, shape `[1]`.
- `temperature=float(collect_temperature)` after requiring `collect_temperature > 0.0`.
- `epsilon=float(collect_epsilon)` after requiring `0.0 <= collect_epsilon <= 1.0`.

Output/decode:

- Calls `policy.collect_mode.forward(obs_tensor, action_mask=..., temperature=..., to_play=..., epsilon=..., ready_env_id=...)`.
- Returns `action = eval_mod._extract_eval_action(output)` and `compact_output = eval_mod._compact_mcts_output(output)`.
- This path is single-root only; the imported compact helper unwraps key `0`/`"0"` and would only summarize root 0 for batched output.

### Tournament per-seat collect path

File refs:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3520`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3544`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3553`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3896`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3899`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3911`

Input format:

- The tournament environment is `VectorMultiplayerEnv(batch_size=1, player_count=2, ...)`, and visual policy stacks are `SourceStateGray64Stack4(batch_size=1, player_count=2, ...)`.
- For each seat, the caller builds:
  - `observation`: `np.asarray(observation[0, seat], dtype=np.float32)`, shape `[4, 64, 64]`.
  - `action_mask`: `np.asarray(batch.action_mask[0, seat], dtype=np.float32)`, shape `[3]`.
  - `to_play`: literal `-1`.
- `_policy_action` wraps the row exactly like the GIF path:
  - `obs_tensor`: torch float32 `[1, 4, 64, 64]`.
  - `action_mask`: NumPy float32 `[1, 3]`.
  - `to_play`: `[-1]`.
  - `ready_env_id`: `np.asarray([0])`.
  - `temperature`/`epsilon`: floats.

Output/decode:

- Calls `policy.collect_mode.forward(...)`.
- Decodes with `eval_mod._extract_eval_action(output)` and `eval_mod._compact_mcts_output(output)`.
- After decode, the tournament validates `0 <= action < ACTION_COUNT` and `obs["action_mask"][action]` before stepping.
- Like the GIF path, decode/compact is single-root-biased.

### Two-seat current-policy training smoke, batched collect

File refs:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2046`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2072`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2082`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2113`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2117`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2696`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2731`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2748`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2758`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2874`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2907`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2922`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2937`

Input format:

- Upstream policy observation is the full two-seat stack `[B, 2, 4, 64, 64]`; validation expects that shape and float32 in `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py:1126`.
- Upstream env mask is `[B, 2, 3]`; validation expects that shape in `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py:1132`.
- `build_policy_row_mapping(...)` flattens active/current-policy seats into row form.
- `fresh_observations = active_observations[fresh_active_indices]`, shape `[R, 4, 64, 64]`, where `R <= B * 2` and can be less due to terminal rows, frozen opponent rows, or no-op skip/repeat rows.
- `fresh_legal = active_legal[fresh_active_indices]`, shape `[R, 3]`.
- `fresh_player_id = active_player_id[fresh_active_indices]`, shape `[R]`, values are seat/player ids `0` or `1`.
- `_policy_actions_batch` prepares:
  - `obs_array = np.asarray(observations, dtype=np.float32)`, shape `[R, 4, 64, 64]`.
  - `action_mask = np.asarray(legal_action_mask, dtype=np.float32)`, shape `[R, 3]`.
  - `players = np.asarray(player_id, dtype=np.int64).reshape(-1)`, shape `[R]`.
  - `obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=_policy_model_device(policy))`.
  - `ready_env_id = np.arange(obs_array.shape[0])`, shape `[R]`.
  - `to_play = [int(item) for item in players]`, Python list length `R`.
  - `temperature=float(temperature)`, `epsilon=float(epsilon)`.

Output/decode:

- Calls `policy.collect_mode.forward(...)` once for all `R` fresh active rows.
- Converts output to plain Python/NumPy-ish values with `_to_plain(output)`.
- For each row, `_policy_output_row_from_plain(plain_output, row)` supports three shapes:
  - mapping keyed by `row` or `str(row)`;
  - mapping of batched fields, where any array-like value with leading dimension larger than `row` is indexed at `row`;
  - list output, indexed at `row`.
- `_extract_eval_action_from_plain(row_output)` checks `"action"`, then key `0`/`"0"`, then `"actions"`, `"selected_action"`, `"selected_actions"`, then list-first.
- `_action_weights_from_policy_output(row_output, action)` reads `visit_count_distribution` or `visit_count_distributions`, normalizes if size is `ACTION_COUNT`, otherwise returns one-hot on the chosen action.
- `_root_value_from_policy_output(row_output)` reads `searched_value`, `predicted_value`, or `value`, otherwise returns `0.0`.
- Per-row record includes `action`, `action_weights`, `root_value`, full batch shape, row shape, row mask shape, `to_play=[to_play[row]]`, `ready_env_id=[ready_env_id[row]]`, `batch_size=R`, temperature, epsilon.

### Two-seat smoke, row fallback collect

File refs:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2158`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2162`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2970`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2991`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:3000`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:3008`

Input format:

- Fallback loops each fresh row and calls `_policy_action`.
- Single row observation is wrapped to torch float32 `[1, 4, 64, 64]`.
- Single row action mask is wrapped to NumPy float32 `[1, 3]`.
- `to_play` is a single-element list containing the actual player id, not `-1`.
- `ready_env_id = np.asarray([0])`.
- `temperature`/`epsilon` are floats.

Output/decode:

- Calls `policy.collect_mode.forward(...)`.
- Decodes action with the imported profile helper `_extract_eval_action(output)`.
- Compacts with `_compact_mcts_output(output)`.
- This fallback is single-root only.

## Requested profile file

File refs:

- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:482`
- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:486`
- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:495`
- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:501`

Important gotcha: this file does not call `collect_mode.forward`; it calls `policy.eval_mode.forward`.

Input format is still relevant:

- `obs_tensor = torch.as_tensor(np.asarray([observation["observation"]]), dtype=torch.float32, device=_policy_model_device(policy))`, effective shape `[1, 4, 64, 64]`.
- `action_mask = np.asarray([observation["action_mask"]], dtype=np.float32)`, shape `[1, 3]`.
- `to_play` is a one-element Python list from observation or `-1`.
- `ready_env_id = np.asarray([0])`.
- No `temperature`; no `epsilon`.

Output/decode helpers from this file:

- `_extract_eval_action` at `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:1474` supports `"action"`, root key `0`/`"0"`, `"actions"`, `"selected_action"`, `"selected_actions"`, and list-first.
- `_compact_mcts_output` at `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:1490` unwraps root `0`/`"0"` and keeps `action`, visit distributions/counts/entropy, policy logits, predicted/searched/value, and `output_keys`.
- `_action_weights` at `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:1370` normalizes compact visit distributions of size 3, otherwise one-hot.
- `_root_value` at `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:1384` chooses searched, predicted, then plain value.

## Shared CurvyTron/Pong decode helpers

CurvyTron eval module used by GIF/tournament:

- `_root_output`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:559`
- `_compact_mcts_output`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:568`
- `_extract_eval_action`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:589`
- `_policy_eval_action`: `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:614`

Checkpoint-opponent provider helper:

- `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:554` calls `policy.eval_mode.forward`, not collect.
- It validates observations as `(4, 64, 64)` and masks as `(3,)` at `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:618` and `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:628`.
- It extracts only the first/root output via `_first_policy_output` and requires `"action"` in `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:580`.

Pong note:

- `rg "collect_mode.forward" src/curvyzero` found no Pong source call site. Current Pong LightZero paths are eval-mode only.
- Pong eval helpers use the same general output convention: root key `0`/`"0"` or list-first, then `"action"`.
- Relevant refs:
  - `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:731`
  - `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:908`
  - `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:933`
  - `src/curvyzero/infra/modal/lightzero_pong_muzero_agent96_eval.py:432`
  - `src/curvyzero/infra/modal/lightzero_pong_github_upstream_eval.py:488`
  - `src/curvyzero/training/lightzero_dummy_pong_policy.py:450`

## Gotchas for an `N = B * 2` real-consumer canary

- Do not feed `[B, 2, 4, 64, 64]` directly to the existing collect call. The only batched collect consumer flattens active seats to `[R, 4, 64, 64]`, with `R <= B * 2`.
- The matching mask shape is `[R, 3]`, not `[B, 2, 3]`.
- Existing collect calls pass `action_mask` as NumPy `float32`, not a torch tensor. Masks may originate as bool/int8, but the call boundary uses float32.
- For two-seat self-play semantics, `to_play` is the player/seat id list `[0, 1, ...]` with length `R`. Tournament/GIF single-root collect uses `[-1]`.
- Use `ready_env_id = np.arange(R)` for batched roots. The row decoder assumes output row order/keys line up with row indexes because ready ids are contiguous.
- For batched output, use the two-seat row splitter pattern, not `eval_mod._compact_mcts_output` directly. The shared compact helpers unwrap only root `0`/`"0"` and will silently drop the other roots in a batched output.
- Be ready for either `{0: {...}, 1: {...}}` style output, list output, or a dict of batched arrays. `_policy_output_row_from_plain` is the most robust existing pattern.
- The action space is 3: `("left", "straight", "right")` in `src/curvyzero/env/trainer_contract.py:59`; `ACTION_COUNT = len(ACTION_NAMES)` in `src/curvyzero/env/vector_multiplayer_env.py:190`.
