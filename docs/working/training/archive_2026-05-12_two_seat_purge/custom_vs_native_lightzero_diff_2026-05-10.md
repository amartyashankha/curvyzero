# Custom vs Native LightZero Diff - 2026-05-10

## Scope

This compares the custom CurvyTron two-seat bounded trainer with the native
LightZero `train_muzero` path, now that CurvyTron should be treated as
Pong-like for the main training path.

Custom bounded path:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py`

Native path:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `docs/working/training/curvytron_native_train_muzero_probe_2026-05-10.md`

Run evidence:

- `docs/working/training/curvytron_accumulated_replay_run_2026-05-10.md`

No pytest was run for this note.

## Short Answer

The native path is the Pong-like path: LightZero emits one ego action, the env
owns the rest of the game transition, and `train_muzero` owns collection,
replay, sampling, learning, checkpointing, and evaluator cadence.

The custom bounded path is a two-seat local adapter: one live `MuZeroPolicy`
chooses both players' actions, local code maps those policy rows back to a
joint action, local replay rows are built by hand, and local code calls
`MuZeroPolicy.learn_mode.forward` directly. It is useful as a mechanical bridge
and diagnosis tool, but it stayed flat and can diverge from LightZero internals.
It is diagnostic only, not the main scaling path.

## Component Diff

| Component | Native LightZero behavior | Custom bounded behavior | Likely learning impact | Evidence |
| --- | --- | --- | --- | --- |
| Training entrypoint | Calls `lzero.entry.train_muzero([main_config, create_config], seed=..., max_train_iter=..., max_env_step=...)`. | Does not call `train_muzero`; directly builds one installed `MuZeroPolicy` and calls eval/learn modes. | Native gets the tested LightZero training loop; custom risks mismatching upstream assumptions. | Native `_run_visual_survival_train` imports `lzero.entry.train_muzero` and sets `called_train_muzero`. Custom result sets `called_train_muzero: False` and `trainer_entrypoint_called: False`. |
| Game model | Pong-like single-ego: policy emits one scalar action and env owns opponent/world transition. | Simultaneous two-seat: policy is queried once per active player row, then rows are folded into `joint_action [B,2]`. | Native matches Atari/Pong MuZero structure; custom teaches a different interface than stock LightZero expects. | Native probe says CurvyTron fits by hiding the opponent inside `env.step(action)`. Custom `_collect_current_policy_iteration` builds policy row mapping and calls `policy_rows_to_joint_action`. |
| Opponent handling | Opponent is env-internal: fixed straight or frozen checkpoint. Current-policy two-seat self-play is explicitly false. | Same live policy object chooses both seats during collection. | Native is simpler and stable for Pong-like training; custom can test self-play mechanics but bypasses LightZero collector semantics. | Native command records `opponent_policy_kind`, `opponent_training_relation`, and `current_policy_self_play: false`. Custom records `same_policy_object_for_both_seats: True` and `policy_object: shared_live_lightzero_policy`. |
| Collector | Uses LightZero `Collector.collect` through `train_muzero`; profile hooks count collector calls and env steps. | Local Python loop steps `VectorMultiplayerEnv` for a bounded number of iterations/steps. | Native gets LightZero collection cadence and trajectory packaging; custom can underfit or bias collection by its small bounded loop. | Native profiler patches `Collector.collect`. Custom loops over `outer_iterations` and `collect_steps_per_iteration`. |
| Replay buffer | Uses LightZero GameBuffer classes via `train_muzero`; profile hooks observe push/sample/update-priority calls. | Stores plain local replay rows with observation, next observation, player metadata, action mask, action weights, root value, reward, and done. | Native gets upstream sampling/priority/segment behavior; custom replay may be mechanically valid but not equivalent. | Native profiler patches `MuZeroGameBuffer`-style methods. Custom `_sample_replay_batch` selects rows from a local list. |
| Learning call | Uses LightZero `BaseLearner.train` inside `train_muzero`. | Calls `MuZeroPolicy.learn_mode.forward([current_batch, target_batch])` directly, optionally allowing an optimizer step. | Native exercises the actual learner lifecycle; custom can update weights but skips learner hooks, buffer priority updates, and normal train iteration control. | Native profiler patches `BaseLearner.train`, `save_checkpoint`, and hooks. Custom `_learn_mode_forward_update` calls `policy.learn_mode.forward`. |
| Target construction | Native LightZero builds MuZero training batches/targets from its game segments and replay path. | Local adapter builds `target_reward`, `target_value`, and `target_policy`; value can be discounted survival return grouped by iteration/env/player/decision metadata. | Custom target code is bespoke and therefore a major source of behavior drift. Native should be preferred unless custom target semantics are the experiment. | Custom `_learn_mode_batches` and `_target_value_batch` build discounted survival value targets. Native path delegates target construction inside `train_muzero`. |
| Observation surface | Registered env emits single-agent `[4,64,64]` debug visual survival observations; model is patched to conv, `image_channel=4`, `frame_stack_num=1`. | Local two-seat observations are `[B,2,4,64,64]`; each active player row is fed to policy as `[4,64,64]`. | Native surface aligns with Atari/Pong config expectations; custom has an extra player dimension outside LightZero's native env API. | Native `_build_visual_survival_configs` sets observation shape `[4,64,64]`. Custom `_validate_visual_batch` expects `[B,2,4,64,64]`. |
| Action surface | One scalar ego action in action space size `3`. | Two-seat joint action `[B,2]`, with legal masks `[B,2,3]`; no-op fills inactive rows. | Native is simpler and closer to Pong; custom introduces row mapping and legality validation that can affect data distribution. | Native config sets `action_space_size: 3`. Custom validates action mask shape `[B,2,3]` and maps rows to joint action. |
| Reward | Survival-time reward in env: survival only, no terminal outcome bonus/penalty. | Per-player survival reward: `1.0` if that player is alive after the step else `0.0`. | Reward idea is similar, but custom assigns it per seat and then hand-builds targets. | Native summary records `reward_schema_id: curvyzero_survival_time/v0` and `survival_only: True`. Custom `_survival_reward` reads `alive_after[B,P]`. |
| Checkpoints/artifacts | LightZero writes normal experiment artifacts; wrapper mirrors LightZero checkpoints into `curvyzero-runs`. | Local code saves `iteration_N.pth.tar`, `ckpt_best.pth.tar`, and `latest.pth.tar` from policy/model/optimizer state. | Native artifacts are closer to downstream LightZero expectations. Custom checkpoints load mechanically but come from a nonstandard loop. | Native `_scan_lightzero_artifacts` and `_mirror_lightzero_checkpoints`. Custom `_save_lightzero_policy_checkpoint`. |
| Scaling signal so far | Main recommended direction in the native probe; native profile should prove collector/replay/MCTS/learner calls. | Accumulated replay run completed mechanically but produced a weak/flat survival curve. | Do not scale the custom loop further as the main path; use native first for CurvyTron-as-Pong-like. | Accumulated run: `ok: true`, 33 checkpoints, 4096 replay rows, but mean steps only `191.688 -> 201.844` then flat through `iteration_32`. |

## Top Differences

1. Native calls `train_muzero`; custom does not.
2. Native treats CurvyTron as Pong-like single-ego; custom exposes both seats.
3. Native uses LightZero collector/replay/learner; custom hand-rolls bounded collection, local replay, and direct `learn_mode.forward`.
4. Native hides opponent control inside the env; custom lets the same live policy choose both players.
5. Native should be the main scaling path; the custom accumulated-replay run was mechanically clean but did not show a convincing learning curve.

## Recommendation

Use the native `train_muzero` path for the next real CurvyTron-as-Pong-like
training runs. Judge it by survival-time checkpoint curves over reproducible
random start panels run in parallel, not seed stories or raw win/loss. Keep the
custom two-seat bounded path for narrow diagnostics: joint-action mapping,
future current-policy two-seat experiments, and learner-adapter target probes.
