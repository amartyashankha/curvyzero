# Pong Official Vs Custom Source Map - 2026-05-09

Scope: source/config comparison only. No training and no pytest were run.

## Reconciled Decision

Atari-style means LightZero-compatible visual env shape, not literal ALE.
Official Atari Pong is the ALE-backed stock ROM control lane.

Official Atari Pong is the primary LightZero reproduction/control lane.
Custom dummy Pong is a bridge/debug lane only. It is useful for wrapper,
telemetry, target-replay, opponent, and scorecard work, but it is not a rival
quality lane and its scores should not be compared to official Atari Pong
scores.

See `docs/working/pong_lane_reconciliation_2026-05-10.md` for the plain-language
readout.

## Source Paths Compared

Official Atari Pong path:

- Repo wrapper: `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- Dry segment config wrapper: `src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py`
- Eval/checkpoint wrappers:
  `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` and
  `src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py`
- LightZero config imported by the trainer wrapper:
  `zoo.atari.config.atari_muzero_config`
- Local LightZero source snapshot checked:
  `/tmp/lightzero-src/zoo/atari/config/atari_muzero_config.py`
- Official env class:
  `/tmp/lightzero-src/zoo/atari/envs/atari_lightzero_env.py`

Custom dummy Pong path:

- Train wrapper: `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- Scaled train wrapper: `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- Config builder: `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- Env adapter: `src/curvyzero/training/lightzero_dummy_pong_env.py`
- Toy env: `src/curvyzero/training/dummy_pong.py`
- Feature encoder: `src/curvyzero/training/lightzero_dummy_pong_features.py`
- Independent MCTS scorecard:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
  and `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py`

## Concrete Differences

| Area | Official Atari Pong | Custom dummy Pong |
| --- | --- | --- |
| Env API | Uses LightZero `atari_lightzero` env type and `AtariEnvLightZero`. Single-agent Atari action is passed directly to `step(action)`. `reset()`/`observe()` return the LightZero dict: `observation`, `action_mask`, `to_play`, `timestep`. | Registers `dummy_pong_lightzero` with `DummyPongLightZeroEnv`. LightZero controls one ego paddle only; wrapper supplies opponent action, records joint action, and returns the same LightZero dict shape. |
| Observation shape | Official `atari_muzero_config` source uses visual `observation_shape=(4, 64, 64)` and `frame_stack_num=4`. Eval smoke has extra frame-stack handling because the env can expose one current frame before the policy stack is assembled. | Default `feature_mode=tabular_ego` is `10` float features. `raster_flat` is `15 * 9 = 135` flat values. No custom conv image path is used by the LightZero trainer config. |
| Model type | Conv MuZero: `policy.model.model_type='conv'`. | MLP MuZero patched from CartPole config: `policy.model.model_type='mlp'`. |
| Frame/history handling | Official config has stacked Atari frame history via `frame_stack_num=4`; default `game_segment_length=400`, with repo tiny train caps it to `16`. Atari wrappers also use frame skip/warp/grayscale behavior in the env. | No frame stack in the LightZero dummy adapter. `tabular_ego` includes ball velocity and step fraction; `raster_flat` is a single current grid. Default custom `game_segment_length=50`, with some runs overriding it. |
| Reward | Atari env collector cfg sets `clip_rewards=True`; evaluator cfg sets `clip_rewards=False`. The env reports raw `eval_episode_return` at done. | Toy env reward is sparse: `+1/-1` on score, `0` otherwise and on timeout. `shaped_loss_delay_return` is telemetry in `info`, not the reward returned to LightZero. |
| Terminal/truncation | Official `step()` returns one `done` from wrapped Gym/ALE. Collector/evaluator max episode steps are set through `collect_max_episode_steps` and `eval_max_episode_steps`; collector uses `episode_life=True`, evaluator uses `episode_life=False`. | `PongEnv.step()` separates `terminated` from `truncated`; adapter returns `done = terminated or truncated` and stores both booleans in `info["curvyzero_pong"]`. Target replay rows do not carry per-transition truncation; the source note says truncation stays in `episodes.jsonl`. |
| Action space | `action_space_size` comes from `atari_env_action_space_map`; Pong is patched/validated as `6`. Action mask is all ones over Atari actions. | `ACTION_LABELS=("up", "stay", "down")`; action space is `3`; action mask is always `[1, 1, 1]`. |
| MCTS sims | Upstream official Atari config defaults to `num_simulations=50`. Repo dry/tiny official smokes patch this down to `2` for infrastructure checks, with train-smoke validation allowing small capped values only. | Custom default is also `2` in config/train wrappers. Scaled experiments can pass `8` or `16`. MCTS checkpoint opponent/eval defaults are also `2` unless overridden. |
| Support scale / support range | Official Atari MuZero config source does not set reward/value support ranges in the repo wrapper path; it relies on LightZero model/policy defaults. | Custom config accepts optional `reward_support_*` and `value_support_*` args and records requested ranges in target replay metadata. The current source does not explicitly patch/log a compiled `policy.model.support_scale`; existing notes treat that as a separate verification gap. |
| Replay target logging | Official stock train wrapper parses stdout/stderr for metric names such as `target_reward_avg`, checkpoint saves, and final rewards. It does not mirror GameSegment target rows. | Custom train wrapper monkeypatches `MuZeroCollector.collect` during training and writes `target_replay_steps.jsonl` with `action_segment`, `child_visit_segment`, reward, root value, and config snapshot. |
| Modal volume path | Uses Volume `curvyzero-runs` mounted at `/runs`, `TASK_ID="lightzero-official-visual-pong"`. Canonical refs are under `training/lightzero-official-visual-pong/<run_id>/...`. | Uses the same Volume and mount, but `TASK_ID="lightzero-dummy-pong"`. Canonical refs are under `training/lightzero-dummy-pong/<run_id>/...`. |
| Evaluator / scorecard | Official path has LightZero's evaluator plus repo checkpoint/eval smokes. It can load a checkpoint and step ALE Pong, but there is no rich project scorecard table in the stock train wrapper. | Custom path writes env-side `episodes.jsonl`, summarizes a `pong_scorecard`, and has an independent MCTS eval-mode scoreboard with checkpoint rows, baselines, paired seats, wins/losses, truncations, survival stats, returns, and action histograms. |

## Short Readout

Official Atari Pong is the visual/conv/frame-stack path through ALE and
LightZero's built-in `atari_lightzero` config. Custom dummy Pong is a
single-ego adapter around a two-player toy env, patched from CartPole into an
MLP MuZero config with tabular or flat-raster features.

The largest source-level differences are not hidden action mapping issues.
They are: visual stacked frames vs small MLP features, official-scale search
defaults vs tiny custom/default sims, official evaluator/logging vs custom
sidecar scorecards, and the custom path's extra target-replay telemetry.
