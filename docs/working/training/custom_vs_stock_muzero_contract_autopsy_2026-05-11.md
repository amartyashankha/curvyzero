# Custom Vs Stock MuZero Contract Autopsy - 2026-05-11

Purpose: make the stock `train_muzero` contract explicit before using any
custom CurvyTron path as learning evidence.

No pytest was run. No code was edited.

## Short Verdict

Stock LightZero Pong is not just "policy forward plus optimizer step." It is a
full loop: collector, MCTS/search, `GameSegment`, `MuZeroGameBuffer`, sampled
targets, `BaseLearner.train`, priority updates, checkpoints, and evaluator.

The custom two-seat path is useful, but it is a local actor/learner adapter. It
asks one live `MuZeroPolicy` for both seats, writes local replay rows, builds
targets by hand, and calls `learn_mode.forward` directly. That can change
weights, but it is not the same training contract as stock `train_muzero`.

Before CurvyTron, either keep the Pong-like single-ego `train_muzero` path, or
make the custom simultaneous path feed native-compatible `GameSegment` /
`MuZeroGameBuffer` data and prove the target semantics match.

## Concrete Discrepancy Table

| Contract piece | Stock `train_muzero` does X | Custom two-seat path does Y | Why this could break or hide learning | Fix or validate before CurvyTron |
| --- | --- | --- | --- | --- |
| Entrypoint | Calls `lzero.entry.train_muzero([main_config, create_config], seed=..., max_env_step=...)`. The exact Pong wrapper only patches the experiment name in stock mode. | Does not call `train_muzero`. It builds one installed `MuZeroPolicy` and runs local collect, replay, learn, and checkpoint code. | A direct policy call can update weights while skipping the tested LightZero training lifecycle. A flat curve may be a broken adapter, not a MuZero result. | Label this as an adapter smoke unless `called_train_muzero=true`. For a serious run, use stock `train_muzero` or prove an explicit replacement for every skipped lifecycle step. |
| Loop ownership | Stock loop does: maybe eval, `collector.collect`, push segments to replay, compute update count, sample replay, `learner.train`, maybe update priorities, stop on env/train budget. | Local loop does: collect bounded steps, sample local rows, call `policy.learn_mode.forward`, maybe save a local checkpoint. | The custom path may miss ordering assumptions. For example, checkpoint, priority, learner counters, and evaluator cadence no longer mean the same thing. | Add a contract summary that names who owns collect, replay, targets, learner, checkpoint, and eval. Do not compare runs unless these owners match. |
| Collector | Uses LightZero `MuZeroCollector` through env managers. Stock Pong uses `8` collector envs and `n_episode=8`. | Uses `VectorMultiplayerEnv` directly and collects `collect_steps_per_iteration` in a Python loop. | Bounded local collection can produce too little data, different reset timing, and different episode boundaries. It can look like training while starving replay. | Report completed episodes, rows collected, rows sampled, reset/autoreset policy, and policy version. For native LightZero reuse, convert complete per-seat trajectories to `GameSegment`. |
| Env action API | Stock Atari/Pong policy emits one scalar action for one env row; the env owns the rest of the world. | Builds one policy row per active player, then folds selected actions into `joint_action[B,2]` before stepping the env. | This is not stock single-agent Pong and not joint-action MCTS. Independent searches can fight each other or create targets that do not describe the actual joint transition. | Choose one: hide opponent/world action inside `env.step(action)` for a Pong-like run, or label the algorithm as independent-seat search. If using simultaneous play, record joint action as audit data and train on scalar ego rows. |
| Self-play meaning | Stock Atari Pong is single-agent ALE training. It is not two live policies controlling two seats. | The same live policy object chooses both seats in the same physical step. | Calling this "self-play" can hide a major algorithm change. It is same-policy independent-seat control, not a stock self-play game buffer. | Every run must say: ALE single-agent, fixed opponent, frozen opponent, same-policy independent-seat search, or joint-action MCTS. |
| Search target | Stock MuZero stores MCTS visit distributions in game segments; target policy comes from search visits, not just the executed action. | Stores `action_weights` from compact MCTS output when available, else falls back to one-hot on the selected action. | If visit distributions are missing or malformed, policy learning can silently become imitation of a noisy action. That hides bad search targets behind nonzero loss. | Audit every batch: selected action, legal mask, visit distribution sum, root value, and target policy. Fail if too many rows use one-hot fallback without being an explicit ablation. |
| Post-search action noise | Stock exploration belongs inside LightZero collector/search settings. | Custom path can apply no-op action noise after search. The replay `action` can differ from `policy_selected_action`, while `action_weights` are based on the policy-selected action. | The dynamics unroll may train on one action while the policy target describes another. With noise enabled, targets can become internally inconsistent. | Keep `action_noop_probability=0` for learning claims until audited. When enabled, report mismatch count between `action` and `policy_selected_action`, and decide whether target policy should follow executed action or search action. |
| Replay type | Uses LightZero `GameSegment` and `MuZeroGameBuffer`, including segment trimming, sampling, target construction, and priority support. | Stores plain local dict rows with observation, next observation, action mask, action, action weights, root value, reward, done, and metadata. | Local rows may be mechanically valid but not semantically equivalent. The most dangerous bug is target construction that has the right shape but wrong meaning. | Spike the optimizer-doc plan: group rows by `(episode_id, env_row_id, player_id)`, build one native-compatible `GameSegment` per seat, push to `MuZeroGameBuffer`, then sample through native code. |
| Replay scope | Stock Pong has large replay capacity (`1,000,000`) and samples from the growing game buffer. | Default replay scope is `current_iteration`; accumulated replay is optional and capped by `max_replay_rows` by default. | Current-iteration replay can train repeatedly on a tiny, fresh slice. Accumulated replay can still be too small or too young. Either can hide real learning behind variance. | For learning runs, default to accumulated replay and report replay age, episode ids, row count, sample count, and replacement/no-replacement sampling. |
| Update count | Stock Pong uses `update_per_collect=None` with `replay_ratio=0.25`, so LightZero computes learner updates from collected data. | Uses manual `updates_per_iteration` / `learner_updates` around local collection. | Too few updates underfit. Too many updates overfit stale tiny replay. Either can make CurvyTron look unlearnable when only the update/data ratio is wrong. | Record actual optimizer steps per env step and per replay row. Match stock auto replay-ratio behavior when testing parity, or sweep this as a named knob. |
| Learner call | Uses DI-engine `BaseLearner.train`, which wraps policy learning with trainer counters, hooks, checkpoint behavior, and logging. | Calls `MuZeroPolicy.learn_mode.forward([current_batch, target_batch])` directly. | Direct forward can run a loss and even an optimizer step, but it skips normal learner bookkeeping. Priority updates and checkpoint hooks are not stock. | If direct learning stays, add a repo-owned learner contract. Otherwise, route through native replay sampling and the normal learner path. |
| Optimizer step | Stock learner owns optimizer, scheduler, target model update cadence, and train iteration accounting. Pong stock surface uses `learning_rate=0.2` and `target_update_freq=100`. | Default path blocks/restores optimizer steps when `allow_optimizer_step=false`. With `allow_optimizer_step=true`, it allows direct `learn_mode.forward` updates and checks model hash changed. | "Weights changed" is weaker than "stock learner trained." Target-model and scheduler cadence may not match stock, and no-op smokes can be mistaken for real training. | For every run, report optimizer step count, scheduler step count, target update count, `last_iter`/train counter if available, and whether the no-op patch was active. |
| Batch and target shape | Stock `MuZeroGameBuffer.sample` builds the full MuZero current/target batch for the configured unroll, td steps, masks, weights, and priorities. | `_learn_mode_batches` builds arrays by hand: repeats actions/rewards over unroll steps, repeats policy targets, sets masks/weights to ones, and builds discounted survival value targets from local metadata when present. | Shape-compatible targets can still be wrong. Repeated reward/policy targets may not match native bootstrapping, padding, or priority assumptions. | Compare one custom sampled batch against one native `MuZeroGameBuffer.sample` batch on a tiny known trajectory. Check reward, value, policy, mask, index, and weight arrays. |
| `to_play` / player semantics | Stock Atari is effectively single-agent / not-board-game. CurvyTron-as-Pong should usually use scalar ego rows and avoid board-game player semantics unless proven. | Passes `to_play=player_id` for each active seat. Optimizer docs suggest native-compatible CurvyTron segments should likely use scalar ego rows with `to_play=-1`. | Passing player ids may trigger assumptions meant for turn-based board games or value sign handling. This can silently corrupt value targets. | Make a one-page `to_play` decision. For LightZero native-compatible replay, validate `to_play=-1` versus `player_id` on identical rows before any scale run. |
| Observation contract | Stock Pong uses env-owned four-frame grayscale stack `[4,64,64]`, `frame_stack_num=4`, action space `6`. | Custom path has outer `[B,2,4,64,64]`, then feeds each active player row as `[4,64,64]`; the frames are source-state gray64/player-perspective debug visuals, action space `3`. | Same tensor shape does not mean same information. Four Atari frames and four source-state channels/frames can have different time and perspective meaning. | Every artifact must record observation shape, dtype, value range, frame-stack owner, player perspective, renderer/schema id, and terminal-frame policy. |
| Reward | Stock exact Pong uses unshaped Atari Pong reward by default; survival shaping is a separate labeled ablation. | Custom path trains per-player survival reward: alive after step is `1.0`, dead is `0.0`. | Survival reward can learn a different skill than Pong score. Comparing stock score curves to custom survival curves can hide progress or invent it. | Name both training reward schema and eval metric schema. Judge CurvyTron survival against same-run `iteration_0`, not stock Pong score. |
| Support/value scale | Stock config support values are tuned with its reward/value range and Atari recipe. | Custom survival returns can grow with episode length and discount. | If value targets exceed support assumptions, the learner may saturate or flatten while losses still move. | Report reward range, value target min/max/mean, discount, support scale, and percent of targets near support edges. |
| Checkpoint lifecycle | Stock `BaseLearner` writes LightZero experiment checkpoints and `ckpt_best` through normal hooks; wrapper mirrors them. | Local code saves `iteration_N.pth.tar`, then copies the same file to `ckpt_best.pth.tar` and `latest.pth.tar`. | `ckpt_best` in the custom path may only mean "last saved," not best evaluator result. Downstream eval can load a valid file but read the wrong semantics. | Include checkpoint kind, selected state key, strict-load result, model surface, optimizer state presence, and whether `ckpt_best` is metric-best or copy-of-latest. |
| Eval contract | Stock serious read uses strict checkpoint load, no fallback, stock `MuZeroEvaluator`/stock-only eval where available, seed panels, and same-run `iteration_0`. | Custom path mostly reports local survival/iteration summaries and checkpoints; it does not automatically run the stock evaluator contract. | A trainer-side curve can be collection noise, not policy improvement. Strict eval can also hide learning if caps/seeds differ from training notes. | Before claiming learning, run same-run `iteration_0` and later checkpoints on the same held-out seed panel, strict load, no fallback, matching caps, and recorded opponent/eval mode. |
| Budget and horizon names | Stock exact wrapper treats `max_env_step` as trainer budget and does not set episode caps in exact mode. | Custom path has `outer_iterations`, `collect_steps_per_iteration`, `env_max_ticks`, and checkpoint cadence. | A longer training budget, episode horizon, and eval cap can be accidentally conflated. That can manufacture or erase survival movement. | Use separate names in summaries: `train_max_env_steps`, `collect_steps_per_iteration`, `episode_max_ticks`, and `eval_cap_steps`. |
| Architecture scale | Stock LightZero is still a single-container trainer, but it gives a coherent collector/replay/learner/evaluator system. Optimizer docs say real MuZero scale is actor/search/replay/learner/checkpoint/eval. | Custom path is one local actor/learner smoke. It is not a distributed searched self-play data factory. | Speeding this loop up is not the same as building the system that can produce enough searched experience. A no-learning result may be throughput architecture, not algorithm failure. | Treat the custom path as actor prototype plus learner smoke. Next useful bridge is actor chunks with explicit policy version, native-compatible replay chunks, learner merge, checkpoint publish, and replay-age metrics. |

## Minimum Validation Before A CurvyTron Learning Claim

Use this as the gate before scaling a custom path:

1. `called_train_muzero` or `custom_contract_id` is explicit.
2. Each replay row has: episode id, env row, player id, decision index, scalar ego action, optional joint action, legal mask, visit distribution, root value, reward, done, and policy/checkpoint id.
3. Visit distributions are present and normalized; one-hot fallback rate is reported.
4. Target batch arrays are dumped for one tiny known trajectory and compared to native `MuZeroGameBuffer.sample` or to a written repo-owned formula.
5. Optimizer, scheduler, and target-network update counts are reported.
6. Replay age, replay size, rows sampled, and updates per env step are reported.
7. Observation and reward schemas are written into the checkpoint metadata.
8. Checkpoints strict-load with no fallback and state which key was loaded.
9. Eval uses same-run `iteration_0` and later checkpoints on the same seed panel.
10. The run name says whether it is single-ego, fixed-opponent, frozen-opponent, same-policy independent-seat, or joint-action MCTS.

## Plain Next Step

Do not scale the current two-seat direct learner as the main CurvyTron proof.
The clean bridge is:

1. keep the custom simultaneous collector if needed;
2. group each seat perspective into a complete scalar-action trajectory;
3. convert those groups into native-compatible `GameSegment` objects;
4. push them into `MuZeroGameBuffer`;
5. sample with LightZero's native replay code;
6. call the normal learner path or, at minimum, compare direct
   `learn_mode.forward` inputs against native sampled inputs.

That keeps CurvyTron's simultaneous physics without quietly replacing MuZero's
training contract.

## Evidence Read

- `docs/working/training/muzero_training_footguns_2026-05-11.md`
- `docs/working/training/custom_vs_native_lightzero_diff_2026-05-10.md`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py`
- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py`
- `docs/working/optimizer/lightzero_modal_loop_2026-05-09.md`
- `docs/working/optimizer/full_training_loop_worldview_2026-05-11.md`
- `docs/working/optimizer/distributed_muzero_architecture_research_2026-05-11.md`
- `docs/working/lightzero_official_atari_settings_audit_2026-05-09.md`
