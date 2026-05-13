# LightZero Stock Dataflow For CurvyTron Frozen Opponent

Date: 2026-05-12

Scope: stock LightZero `train_muzero` as used by CurvyTron
`env_variant=source_state_fixed_opponent`, especially
`opponent_policy_kind=frozen_lightzero_checkpoint`.

No live training was touched. Local `uv run` does not currently import
`lzero`/`ding`; the Modal image in the launcher installs `LightZero==0.2.0`.
LightZero internals below are therefore based on the local launcher/env code
and upstream LightZero source docs plus the raw v0.2.0 Atari MuZero config.

## Plain Read

There is no separate actor fleet in the trusted stock lane. One Modal training
function starts one stock LightZero process. Inside that process, LightZero owns
the policy, collector, evaluator, replay buffer, learner, and checkpoint hooks.

Parallelism is narrow:

- the collector can drive many envs through a vector env manager;
- `env_manager_type=subprocess` can put env steps in worker processes;
- policy/search calls batch over ready env roots;
- learner updates run after collection, not at the same time as collection.

The learner/search model can use GPU when `compute=gpu-*` sets
`policy.cuda=true`. The env, render/observation packing, replay bookkeeping,
MCTS tree bookkeeping, subprocess IPC, artifact I/O, and default frozen
opponent all remain CPU-side.

## One Stock Iteration

The CurvyTron launcher builds LightZero configs from
`zoo.atari.config.atari_muzero_config`, patches them, then calls:

```python
train_muzero(
    [patched["main_config"], patched["create_config"]],
    seed=seed,
    max_train_iter=max_train_iter,
    max_env_step=max_env_step,
)
```

LightZero `train_muzero` then does this:

1. Compile config and set `cfg.policy.device` from `policy.cuda`.
2. Build collector and evaluator vector env managers from `cfg.env`.
3. Create one MuZero policy with `learn`, `collect`, and `eval` modes.
4. Wrap learn mode in `BaseLearner`.
5. Create `MuZeroGameBuffer` for replay.
6. Create `MuZeroCollector(policy.collect_mode)` and
   `MuZeroEvaluator(policy.eval_mode)`.
7. Run learner `before_run` hook.
8. Optionally random collect. Current CurvyTron stock path inherits
   `random_collect_episode_num=0`.
9. Run an initial evaluator pass unless the Optimizer profile hook skips it.
10. Enter the train loop.

Each loop body is synchronous:

```text
maybe evaluate
  -> collector.collect(...)
  -> calculate update_per_collect
  -> replay_buffer.push_game_segments(...)
  -> replay_buffer.remove_oldest_data_to_fit()
  -> repeat learner updates:
       replay_buffer.sample(batch_size, policy)
       learner.train(train_data, collector.envstep)
       optional replay_buffer.update_priority(...)
  -> stop if max_env_step or max_train_iter reached
```

With CurvyTron fixed/frozen opponent, one collected LightZero transition means:

```text
collector sees ego observation/action mask
  -> policy.collect_mode chooses one scalar ego action using MuZero MCTS
  -> env.step(action) computes the opponent action internally
  -> VectorMultiplayerEnv advances a two-player joint action
  -> env returns one ego reward/done/info stream
  -> collector records the result in a LightZero GameSegment
```

The hidden opponent is why this path fits stock LightZero: LightZero sees a
single-agent fixed-action-space env, while the wrapper handles player 1.

## Where Things Live

Actor/collector:

- There is no independent long-running actor service.
- `MuZeroCollector` lives inside `lzero.entry.train_muzero`.
- It calls `policy.collect_mode.forward(...)` in the parent training process.
- It returns LightZero `GameSegment` objects plus metadata.

Env manager:

- Configured by `create_config.env_manager.type`.
- CurvyTron default is `subprocess`; `base` is available.
- `base` keeps env work in the train process and gives better profiler
  attribution.
- `subprocess` puts env instances in worker processes; env/render/opponent time
  is mostly folded into collector wall time unless worker telemetry is enabled.

MCTS/search:

- Owned by `MuZeroPolicy.collect_mode` and `eval_mode`.
- Collection does model initial inference, creates MCTS roots for ready envs,
  runs `self._mcts_collect.search(...)`, gets visit distributions/root values,
  then samples or chooses actions.
- MCTS is synchronous inside collector policy forward. It is batched by the
  number of ready roots, not by a separate search service.

Replay buffer:

- `MuZeroGameBuffer` lives in the same train process.
- Collector output is pushed via `push_game_segments`.
- Sampling builds `[current_batch, target_batch]` for `policy.learn_mode`.
- With current inherited Atari defaults, `reanalyze_ratio=0`, so normal samples
  mainly use stored search targets rather than running fresh replay reanalysis.

Learner:

- `BaseLearner` lives in the same train process.
- It calls `policy.learn_mode` through `learner.train(...)`.
- No learner update runs concurrently with collection in the stock entry.

Checkpoints/eval:

- Stock LightZero writes under the configured `exp_name`, which CurvyTron sets
  inside the run attempt's `lightzero_exp` directory under `/runs`.
- Learner hooks write LightZero checkpoints under `lightzero_exp/ckpt`.
- The launcher later scans artifacts and mirrors LightZero checkpoints into the
  CurvyZero run checkpoint root under `checkpoints/lightzero`.
- Stock evaluator is `MuZeroEvaluator` in the train process and blocks the main
  loop when it runs.
- Background checkpoint eval/GIF is CurvyZero launcher infrastructure. In the
  default train CLI path, the poller can be spawned as a separate Modal function
  to watch published checkpoints; it is outside the stock LightZero loop.

## CurvyTron Env Step

The registered env is
`CurvyZeroSourceStateVisualSurvivalLightZeroEnv`, backed by
`CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv`.

On reset:

- create a one-row, two-player `VectorMultiplayerEnv`;
- reset wrapper-owned FIFO stacks;
- render/source-pack the first `[4, 64, 64]` normalized gray observation;
- return LightZero obs dict with `observation`, `action_mask`, `to_play=-1`,
  and `timestep`.

On step:

- validate scalar ego action;
- optionally apply profile/control override;
- compute opponent action:
  - fixed-straight: action id `1`;
  - frozen checkpoint: call snapshot-backed LightZero checkpoint opponent;
- advance `VectorMultiplayerEnv` with joint action `[ego, opponent]`;
- optionally repeat the same policy action for multiple source ticks;
- compute sparse outcome or dense-survival-plus-outcome reward;
- render/pack next source-state gray64 stack;
- write telemetry row;
- return `BaseEnvTimestep`.

For the frozen checkpoint opponent, the env loads a separate MuZero policy from
the checkpoint on first use inside that env instance. The current safe default
is `opponent_use_cuda=false`, so subprocess env workers do not each try to own
CUDA.

## Synchronous Vs Parallel

Synchronous:

- `train_muzero` main loop;
- stock evaluator calls;
- one `collector.collect` call as a blocking phase;
- replay push/remove/sample;
- each learner update;
- stock checkpoint hook work in the learner process;
- post-train artifact scan/mirror in the launcher.

Parallel:

- subprocess env workers can step multiple CurvyTron envs at once;
- policy/search batches ready env roots in one forward/search call;
- CUDA kernels may overlap internally unless profile mode forces CUDA sync;
- background checkpoint poller/eval/GIF can run in separate Modal functions if
  enabled.

Not parallel in stock `train_muzero`:

- collection and learner updates;
- replay and learner;
- collector actors across machines;
- centralized batched frozen-opponent inference across env workers.

## GPU And CPU Boundary

Can use GPU when `compute=gpu-*`:

- live MuZero model initial inference for collect/eval;
- live MuZero recurrent inference used during search;
- learner forward/backward and optimizer step;
- target model operations during replay reanalysis, if reanalysis is enabled;
- frozen opponent model/search only if `opponent_use_cuda=true`.

Still CPU-side today:

- `VectorMultiplayerEnv` physics/runtime;
- source-state RGB render, gray64 conversion, FIFO stack updates;
- action masks, reward calculation, info/telemetry assembly;
- env manager IPC and subprocess scheduling;
- MCTS tree/root bookkeeping and action selection around model calls;
- root preparation and visit distributions use CPU/numpy objects after the
  policy converts initial inference outputs back from tensors;
- default frozen checkpoint opponent inference/search
  (`opponent_use_cuda=false`);
- replay buffer storage, segment-position bookkeeping, target assembly for
  non-reanalyze samples;
- checkpoint file I/O, artifact scanning, mirroring, JSON summaries.

Plain consequence: bigger GPUs help the live model/learner/search-model calls,
but they do not remove the long-survival render/observation/opponent CPU costs.

## Knobs That Control Width, Batch, And Search

Direct launcher knobs:

- `--env-manager-type`: `base` or `subprocess`.
- `--collector-env-num`: number of collector env instances/workers.
- `--n-episode`: complete episodes collected per `collector.collect`.
- `--evaluator-env-num`: evaluator env instances.
- `--n-evaluator-episode`: evaluator episode count.
- `--num-simulations`: MCTS simulations per root for the live policy; also
  copied into frozen-opponent config as `opponent_num_simulations`.
- `--batch-size`: learner minibatch size; also copied into frozen-opponent
  config as `opponent_batch_size`.
- `--max-env-step`: stop cap on collected env steps.
- `--max-train-iter`: stop cap on learner iterations.
- `--source-max-steps`: per-episode source tick cap and CurvyTron-patched
  `td_steps`.
- `--decision-ms`: source time advanced per wrapper source decision.
- `--source-state-trail-render-mode`: render implementation, currently the big
  long-survival CPU lever.
- `--lightzero-eval-freq`: stock evaluator cadence. `0` is patched to
  `max_train_iter + 1`, but the initial eval still exists unless profile hooks
  skip it.
- `--skip-lightzero-eval-in-profile`: Optimizer profile-only eval skip.
- `--save-ckpt-after-iter`: learner checkpoint cadence.
- `--background-eval-*` and `--background-gif-*`: CurvyZero checkpoint
  evaluation/GIF side work.
- `--opponent-policy-kind`: fixed-straight vs frozen checkpoint.
- `--opponent-use-cuda`: device for the frozen checkpoint opponent, separate
  from learner CUDA.
- `--lightzero-multi-gpu`: passes through to LightZero policy config, but does
  not create actor/search/replay parallelism.

Inherited LightZero/Atari-template knobs not currently exposed as first-class
CurvyTron CLI controls:

- `update_per_collect=None`;
- `replay_ratio=0.25`, used by stock `train_muzero` to derive learner updates
  from collected transition count when `update_per_collect` is `None`;
- `game_segment_length=400`;
- `num_unroll_steps=5`;
- `random_collect_episode_num=0`;
- `reanalyze_ratio=0`;
- `replay_buffer_size=1_000_000`;
- `use_priority=False`;
- `target_update_freq=100`;
- `mcts_ctree=True`;
- augmentation/SSL/model details inherited from the Atari MuZero config except
  where the launcher patches model shape/channel/action support.

These inherited knobs matter because they shape learner work per collect,
segment validity, target construction, and replay memory pressure even though
the current Optimizer command surface mostly talks about collectors, batch, and
sims.

## Biggest Architecture Limits

1. Stock LightZero here is a single synchronous training process, not an
   AlphaZero-style actor fleet. Collection, replay sampling, learning, and
   stock eval take turns.
2. Subprocess env workers parallelize env stepping, but search/model action
   selection is still called by the parent collector as one blocking policy
   phase over ready roots.
3. The frozen checkpoint opponent is inside each env instance. With subprocess
   envs, that means per-worker CPU policy/search work by default, not one shared
   batched opponent service.
4. Long-survival source-state `browser_lines` profiles are dominated by
   observation/render/stack CPU work. GPU upgrades cannot fix that alone.
5. `train_muzero` collects complete episodes. Long episodes or disabled-death
   profiles can make collection the wall-clock gate. Segment collection is a
   different stock entry and is not the current trusted lane.
6. Replay is in-process memory, not a service. There is no natural way for many
   remote actors to push segments into this stock buffer without building a
   wrapper architecture.
7. Multi-GPU is not the same as distributed self-play. It may help learner
   compute once batches are large, but it does not scale env/render/search
   actors.
8. Checkpoint/eval/GIF work can be useful observability, but frequent cadence
   can add blocking or side-load if the goal is pure throughput measurement.
9. This lane is a trusted proof/control lane, not true current-policy
   simultaneous two-seat self-play. It learns against a fixed or frozen
   opponent hidden behind the env wrapper.

## Optimizer Implications

- Use `base` manager for fine attribution; use `subprocess` for throughput
  ladders.
- Keep `opponent_use_cuda=false` unless explicitly testing GPU contention and
  worker behavior.
- Tie all throughput numbers to collector count, `n_episode`, render mode,
  `num_simulations`, `batch_size`, eval/checkpoint settings, and death mode.
- Treat `collector_env_num` as root-width pressure, not a guarantee of full GPU
  saturation. Ready roots depend on env availability and `n_episode`.
- The large architectural speedup path is likely actor/search fanout plus
  batched inference/replay handoff, or a render/observation contract speedup,
  not just a bigger learner GPU.

## Source Basis

Local:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py`
- `docs/working/optimizer/stock_frozen_optimizer_pivot_2026-05-12.md`
- `docs/working/training/curvytron_architecture_research_2026-05-12/stock_lightzero_dataflow.md`

Upstream LightZero source/docs:

- `lzero.entry.train_muzero`
  https://opendilab.github.io/LightZero/_modules/lzero/entry/train_muzero.html
- `lzero.worker.muzero_collector`
  https://www.aidoczh.com/lightzero/_modules/lzero/worker/muzero_collector.html
- `lzero.policy.muzero`
  https://www.aidoczh.com/lightzero/_modules/lzero/policy/muzero.html
- `lzero.mcts.buffer.game_buffer_muzero`
  https://www.aidoczh.com/lightzero/_modules/lzero/mcts/buffer/game_buffer_muzero.html
- LightZero v0.2.0 Atari MuZero config
  https://raw.githubusercontent.com/opendilab/LightZero/v0.2.0/zoo/atari/config/atari_muzero_config.py
