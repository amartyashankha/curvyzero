# Stock LightZero `train_muzero` Dataflow

Purpose: factual map of the stock LightZero MuZero runtime as exercised by the CurvyTron launcher, and the exact architectural seams where `--mode two-seat-selfplay` diverges.

Local repo basis:

- Stock launcher: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- Stock single-ego env wrapper: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- Custom two-seat loop: `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- Native bridge experiment: `src/curvyzero/training/two_seat_native_replay_bridge.py`

Upstream LightZero basis: LightZero `v0.2.0` `lzero.entry.train_muzero`, `lzero.mcts.buffer.game_segment.GameSegment`, and MuZero game-buffer code from the LightZero package installed by the Modal runtime.

## Stock CurvyTron Entry

The stock CurvyTron train/profile path builds patched LightZero configs, imports `lzero.entry.train_muzero`, and calls:

```python
train_muzero(
    [patched["main_config"], patched["create_config"]],
    seed=seed,
    max_train_iter=max_train_iter,
    max_env_step=max_env_step,
)
```

The launcher records this as `trainer_entrypoint: lzero.entry.train_muzero` and sets `called_train_muzero = True` only around that call.

Config facts on this path:

- `create_config.policy.type` is `muzero`.
- `create_config.policy.import_names` includes `lzero.policy.muzero`.
- `create_config.env_manager.type` is configurable and defaults to `subprocess`.
- The default stock CurvyTron env variant is `source_state_fixed_opponent`.
- The stock source-state env is single-ego: LightZero chooses one scalar ego action and the wrapper supplies the opponent action internally.
- The launcher blocks `source_state_turn_commit` in stock train mode because pending and commit scalar steps would both be stored as ordinary LightZero transitions.

## Stock LightZero Runtime Objects

Plain-language dataflow:

1. `train_muzero` compiles config.
2. It creates collector and evaluator vector env managers from the DI-engine env config.
3. It creates one MuZero policy with `learn`, `collect`, and `eval` modes.
4. It wraps `policy.learn_mode` in `BaseLearner`.
5. It creates a MuZero replay buffer.
6. It creates `MuZeroCollector` for collection and `MuZeroEvaluator` for eval.
7. The loop alternates: collect game segments, push into buffer, sample training batches, learner train step, optional priority update, eval/checkpoint hooks.

Runtime object chain:

```text
env config
  -> create_env_manager(...)
  -> collector_env / evaluator_env
  -> MuZeroCollector(policy.collect_mode)
  -> MuZero search inside policy collect/eval forward
  -> GameSegment objects
  -> MuZeroGameBuffer.push_game_segments(...)
  -> MuZeroGameBuffer.sample(...)
  -> BaseLearner.train(...)
  -> policy.learn_mode
  -> learner/evaluator checkpoint hooks
```

## Env Manager Seam

Stock path:

- `train_muzero` calls LightZero/DI-engine env factory helpers.
- The collector and evaluator envs are vector env managers.
- The CurvyTron source-state wrapper inherits the LightZero/DI-engine `BaseEnv` shape and returns `BaseEnvTimestep`.
- `reset()` returns the first LightZero observation.
- `step(action)` accepts one scalar ego action, obtains or computes an opponent action inside the env wrapper, executes a two-player source action, and returns one ego reward/done/info stream.

`two-seat-selfplay` path:

- The main launcher dispatches to the two-seat Modal function instead of the stock train kwargs block.
- The loop directly constructs `VectorMultiplayerEnv` and `SourceStateGray64Stack4`.
- There is no DI-engine env manager object in the active two-seat loop.
- Reset, active-seat selection, autoreset bookkeeping, and player/environment row mapping are handled by repo code.

## Collector Seam

Stock path:

- `MuZeroCollector.collect(...)` is the object that drives environment stepping.
- The collector asks `policy.collect_mode` for actions.
- The collector assembles LightZero `GameSegment` instances and returns them with metadata.
- The repo profiler patches `Collector.collect` as a stock runtime phase.

`two-seat-selfplay` path:

- `_collect_current_policy_iteration(...)` is the collector equivalent.
- It builds active policy rows for both player seats.
- It calls `_policy_actions_batch(...)` to get actions for currently active rows.
- It maps per-seat actions into a joint env action and steps the multiplayer env directly.
- It emits repo-owned replay row dictionaries, not LightZero `GameSegment` objects.

## Policy/Search Seam

Stock path:

- `create_policy(..., enable_field=["learn", "collect", "eval"])` creates one MuZero policy with three modes.
- Collection and eval call `policy.collect_mode` / `policy.eval_mode`.
- MuZero search is inside the LightZero policy forward path; the repo profiler patches MuZero MCTS search methods separately.
- Collector policy kwargs include visit-count temperature and, depending on config, epsilon.

`two-seat-selfplay` path:

- One live installed `MuZeroPolicy` object is used for both seats unless a frozen opponent is configured.
- `_policy_actions_batch(...)` directly calls:

```python
policy.collect_mode.forward(
    obs_tensor,
    action_mask=...,
    temperature=...,
    to_play=players,
    epsilon=...,
    ready_env_id=...,
)
```

- Action records are converted into per-seat replay rows by repo code.
- If a frozen opponent is configured, selected opponent-seat slots are replaced by the frozen policy output before the joint action is stepped.

## GameSegment Seam

Stock path:

- LightZero `GameSegment` is the collector-to-buffer trajectory unit.
- It stores stacked observations, scalar actions, rewards, action masks, visit distributions, root values, `to_play`, and timestep fields.
- `reset(init_observations)` seeds the initial frame stack.
- Each transition stores search stats, then action/next observation/reward/action mask/to-play/timestep.
- `game_segment_to_array()` converts the lists into numpy arrays before replay-buffer use.

Stock CurvyTron source-state env contract:

- The stock single-ego env projects a two-player physical world into one LightZero player stream.
- `action_space_size` is the ego scalar action count for fixed-opponent variants.
- The joint-action env variant encodes both players into one scalar action, but its config marks it as centralized joint-action control, not true competitive self-play.

`two-seat-selfplay` path:

- The active two-seat loop does not build LightZero `GameSegment` objects.
- Replay rows are per current-policy seat decision and include fields such as `iteration`, `env_row_id`, `player_id`, `to_play`, observation, next observation, action, action mask, action weights, root value, reward, done, and return context.
- Repo code explicitly refuses non-current-policy rows into learner replay.
- `two_seat_native_replay_bridge.py` can project a two-seat physical tick trace into two LightZero `GameSegment` objects, but this is a separate bridge/helper and is not the main `--mode two-seat-selfplay` training loop.

## MuZeroGameBuffer Seam

Stock path:

- For `create_config.policy.type == "muzero"`, `train_muzero` uses `MuZeroGameBuffer`.
- `push_game_segments((data, meta))` receives collected segments and metadata.
- Buffer metadata controls valid replay length: completed segments use full data length; incomplete segments subtract `num_unroll_steps + td_steps`.
- The buffer tracks segment/position lookup, priorities, and transition count.
- The train loop samples batches by calling `replay_buffer.sample(batch_size, policy)`.
- If priority is enabled, the learner output feeds `replay_buffer.update_priority(...)`.
- Old segments are removed when the transition count exceeds replay-buffer size.

`two-seat-selfplay` path:

- The main loop uses an in-memory list of repo replay row dictionaries.
- `_sample_replay_batch(...)` samples rows from that list and builds arrays for observations, next observations, actions, rewards, done flags, policy targets, and return context.
- No `MuZeroGameBuffer.push_game_segments`, `sample`, `remove_oldest_data_to_fit`, or priority update call is part of the active two-seat loop.
- The native replay bridge has a helper that can instantiate `MuZeroGameBuffer` and push bridged segments, but that helper is separate from the active two-seat training path.

## Learner Seam

Stock path:

- `BaseLearner` owns the stock training call boundary.
- The train loop calls `learner.train(train_data, collector.envstep)`.
- The learner calls into `policy.learn_mode` and owns standard learner hooks.
- The repo profiler patches `BaseLearner.train` and `policy._forward_learn` as stock phases.

`two-seat-selfplay` path:

- There is no `BaseLearner.train(...)` call in the active loop.
- `_learn_mode_forward_update(...)` builds LightZero-shaped current/target batches from sampled replay rows.
- It calls `policy.learn_mode.forward([current_batch, target_batch])` directly.
- Its result records `trainer_entrypoint_called: False` and `api: MuZeroPolicy.learn_mode.forward`.
- A loss-only variant calls the same learn-mode path without allowing an optimizer step.

## Checkpoint Seam

Stock path:

- Checkpoints are owned by LightZero learner/evaluator hooks.
- `train_muzero` passes `learner.save_checkpoint` into evaluator eval.
- The repo launcher later scans artifact directories and mirrors checkpoint files.
- A successful stock train summary expects checkpoint artifacts when train mode ran without earlier failure.

`two-seat-selfplay` path:

- Checkpointing is manual repo code.
- `_save_lightzero_policy_checkpoint(...)` writes payloads containing model state, metadata, and optionally target-model and optimizer state.
- It writes iteration checkpoint files and copies `ckpt_best.pth.tar` / `latest.pth.tar`.
- Checkpoint cadence is controlled by the two-seat payload fields such as `checkpoint_every` and `save_initial_checkpoint`.

## LightZero Runtime Assumptions Exposed Here

- LightZero stock MuZero assumes a vector env manager drives collection/eval.
- Stock MuZero collection assumes one policy action stream per env step as seen by the collector.
- Search statistics are collected before replay insertion and travel with each `GameSegment`.
- Replay is segment-position based, not row-dictionary based.
- `num_unroll_steps` and `td_steps` affect which tail positions in an incomplete segment are valid for sampling.
- Stock learner entry is `BaseLearner.train(...)`, not direct `policy.learn_mode.forward(...)`.
- Stock checkpointing is hooked through learner/evaluator objects, not a repo-local checkpoint writer.
- CurvyTron stock fixed-opponent mode fits the stock assumption by hiding the second player's action inside the env wrapper.
- CurvyTron two-seat mode keeps the two-player action topology outside the stock collector/buffer/learner chain.
