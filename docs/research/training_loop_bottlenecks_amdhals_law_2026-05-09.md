# Training Loop Bottlenecks And Amdahl's Law

Date: 2026-05-09

Status: research and architecture note. No code changes.

## Top Summary

Speed work only helps in proportion to the part of the training loop it speeds
up. A 10x environment speedup is huge if environment stepping is 70% of wall
time. It is small if MCTS/model inference is 80% of wall time and environment
stepping is only 10%.

Practical priority order:

1. Measure and optimize the whole actor loop first, not isolated microbenchmarks.
2. Optimize env step next only if it remains the largest production-relevant
   bucket after debug event output is removed and real model/search timing is
   included.
3. Optimize model/search/MCTS as soon as calibrated timing exists, because real
   search can dominate the loop even when today's synthetic stand-in is cheap.
4. Keep observation packing boring and fixed-shape; optimize it when it is a
   visible wall-time bucket or creates extra copies across the CPU/GPU boundary.
5. Keep replay as chunky array staging and measure write throughput; optimize it
   when serialization or learner handoff shows up, not while in-memory staging is
   around 1% to 2% in debug scouts.
6. Make reset/autoreset correct and measurable before making it clever. Optimize
   it when terminal rows stall batches or corrupt action latency.

Latency and throughput answer different questions. Throughput asks, "How many
useful games, env rows, or ego decisions do we finish per minute?" Latency asks,
"How long does one ready observation wait before its action is available?"
Bigger batches can improve throughput while hurting p95/p99 action latency or
policy freshness. The right speedup improves useful games/minute without making
actions stale.

Current debug/local numbers are hints, not the optimization plan. The toy
object-env serial bridge reported about `19,937` wall env steps/sec,
`p95=1.391ms`, `p99=1.678ms`, and `env_step=94.5%`, while threads hurt tail
latency. The fixture-seeded vector bridge at `B=32`, no-event,
`simulations=1`, reported `P2_K4 env_step=46.7%`, synthetic policy/search
`7.9%`, replay `1.7%`; and `P3_K4 env_step=66.2%`, synthetic policy/search
`9.8%`, replay `1.8%`. Treat these as debug/local evidence only: fake
policy/search, no real MCTS, no GPU, no learner, and no production replay
writer.

For this repo, the safest near-term plan is:

- keep the real environment on CPU first;
- batch many environment and ego rows;
- keep model inference, MCTS, and any learner update in the same process or
  container;
- measure wall-clock shares before rewriting the environment for GPU, JAX,
  C++, Rust, EnvPool, or distributed actors;
- keep Modal at whole-job boundaries, not per action, step, MCTS node, or
  replay row.

The current environment-speed work is useful, but it is not yet proof of
training speed. It should now connect environment rows, observation packing,
policy/search timing, replay staging, reset/autoreset, and policy staleness into
one measured actor-loop report.

The first optimization target should be the largest measured bucket that is both
real and on the production path. Today's best answer is therefore: finish the
CPU vector actor-loop report, plug in calibrated model/search timing, then
optimize whichever bucket still dominates useful games/minute and p95/p99 action
latency.

## Sources Checked

External sources:

- MuZero paper: [arXiv:1911.08265](https://arxiv.org/abs/1911.08265).
- MuZero pseudocode from the paper's supplemental material, checked through an
  online mirror of `https://arxiv.org/src/1911.08265v1/anc/pseudocode.py`:
  [mirror gist](https://gist.github.com/crizCraig/ec806f0d606c9e727512dce886bdc966).
- Mctx README and examples: [google-deepmind/mctx](https://github.com/google-deepmind/mctx).
- LightZero repo, docs, collector source, and configs:
  [repo](https://github.com/opendilab/LightZero),
  [worker docs](https://opendilab.github.io/LightZero/api_doc/worker/index.html),
  [MuZeroCollector source docs](https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html),
  [config guide](https://opendilab.github.io/LightZero/tutorials/config/config.html),
  [CartPole MuZero config](https://github.com/opendilab/LightZero/blob/main/zoo/classic_control/cartpole/config/cartpole_muzero_config.py).
- Gymnasium vector env docs:
  [VectorEnv](https://gymnasium.farama.org/main/api/vector/),
  [AsyncVectorEnv](https://gymnasium.farama.org/main/api/vector/async_vector_env/),
  [vector A2C tutorial](https://gymnasium.farama.org/main/tutorials/training_agents/vector_a2c/).
- EnvPool:
  [docs](https://envpool.readthedocs.io/en/latest/),
  [NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html),
  [arXiv:2206.10558](https://arxiv.org/abs/2206.10558).
- IMPALA:
  [PMLR paper page](https://proceedings.mlr.press/v80/espeholt18a.html),
  [Google Research page](https://research.google/pubs/impala-scalable-distributed-deep-rl-with-importance-weighted-actor-learner-architectures/),
  [arXiv:1802.01561](https://arxiv.org/abs/1802.01561).
- SEED RL:
  [Google Research page](https://research.google/pubs/seed-rl-scalable-and-efficient-deep-rl-with-accelerated-central-inference/),
  [arXiv:1910.06591](https://arxiv.org/abs/1910.06591),
  [repo README](https://github.com/google-research/seed_rl),
  [run_local.sh](https://github.com/google-research/seed_rl/blob/master/run_local.sh),
  [common_flags.py](https://github.com/google-research/seed_rl/blob/master/common/common_flags.py).

Local sources:

- [MuZero on Modal architecture](../design/muzero_modal_architecture.md)
- [MuZero architecture deep dive](muzero_architecture_deep_dive.md)
- [JAX/Mctx integration plan](mctx_integration.md)
- [LightZero integration critique](lightzero_integration.md)
- [Multiplayer self-play and MuZero-style search](multiplayer_selfplay_muzero.md)
- [Environment performance and vectorization plan](environment/performance_vectorization_plan.md)
- [Simulator performance and vectorization](performance_vectorization.md)
- [Training architecture](../design/training_architecture.md)
- [Training eval rules](../design/training_eval_protocol.md)
- [Modal patterns](modal_patterns.md)
- [Toy-v0 performance scout](../experiments/environment/2026-05-09-toy-v0-performance-scout.md)
- [Vector batch-row timing](../experiments/environment/2026-05-09-vector-batch-row-timing.md)
- [Vector debug obs/reward packing](../experiments/environment/2026-05-09-vector-obs-reward-packing.md)
- `scripts/benchmark_policy_search_batch_standin.py`
- `scripts/benchmark_vector_actor_loop_bridge.py`
- `tests/test_benchmark_vector_actor_loop_bridge.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `docs/working/training_coach_self_critique_2026-05-09.md`
- `docs/working/pong_selfplay_training_plan_2026-05-09.md`

## Full MuZero-Style Loop

MuZero has two loops that run at the same time:

1. Self-play actors create experience.
2. The learner trains from replay and publishes newer checkpoints.

The MuZero paper's supplemental pseudocode makes that split explicit: many
independent self-play jobs read the latest network from shared storage, play a
game with MCTS, save the finished game to replay, and repeat. The learner
samples from replay, updates the network, and periodically saves a new network.

For this project, the full loop should be understood like this:

```text
latest checkpoint / network version
  -> self-play actor batch loads or refreshes policy
  -> reset many real environments
  -> real env observations [B_env, players, ...]
  -> pack live ego rows [B_ego, obs_shape]
  -> model initial inference:
       representation(obs) -> hidden
       prediction(hidden) -> policy logits, value
  -> MCTS/Mctx over learned dynamics:
       recurrent dynamics(hidden, action) -> reward, next_hidden, discount
       prediction(next_hidden) -> logits, value
  -> choose one action per ego row
  -> map ego actions back to joint env actions
  -> real env step on CPU
  -> rewards, dones, truncations, next observations, info
  -> reset/autoreset finished rows after final transition is staged
  -> replay row or chunk:
       obs, action, joint_action, reward, done, search policy,
       root value, raw value, legal mask, ego id, env id,
       model version, rules/obs/reward/action/search hashes
  -> learner samples replay sequences
  -> learner builds n-step/unroll targets
  -> learner runs initial and recurrent model unrolls
  -> learner computes policy, value, reward losses
  -> optimizer update
  -> checkpoint and latest pointer
  -> actors refresh policy; policy staleness is recorded
  -> eval jobs score checkpoints on fixed splits
```

The important distinction: MCTS does not call the real environment in standard
MuZero. It searches over the learned model. The real environment is used to
produce data, rewards, terminal states, and evaluation.

For CurvyTron-style simultaneous play, the simplest v0 compromise is one shared
ego-perspective model. Each live player becomes one ego row. Opponent actions
come from the same policy, a checkpoint, random, or a heuristic. Full joint
action search grows as `A ** players`, so it is a later research experiment.

## How Self-Play Is Parallelized Elsewhere

MuZero paper / pseudocode:

- Self-play jobs are independent actors.
- Each actor repeatedly grabs a network snapshot, plays one game with MCTS, and
  writes the completed game to replay.
- The learner is separate and periodically saves checkpoints.
- The paper pseudocode uses large scale settings: board-game configs use 3000
  actors and 800 simulations per move; Atari uses 350 actors and 50
  simulations per decision. Those are not starter numbers for this repo.

Mctx:

- Mctx is not a trainer. It provides JAX-native, JIT-compatible, batched search.
- It parallelizes across root decisions in a batch.
- A `RootFnOutput` carries root logits, value, and embedding. A `recurrent_fn`
  carries learned dynamics and prediction. `PolicyOutput.action` gives actions,
  and `PolicyOutput.action_weights` gives policy targets.
- It does not provide actors, replay, learner updates, checkpoint format, env
  wrappers, or Modal orchestration.

LightZero:

- LightZero is a PyTorch MCTS+RL toolkit with MuZero, Gumbel MuZero,
  EfficientZero, Stochastic MuZero, and related variants.
- It has model, policy, and MCTS modules. Its MCTS can use Python or C++ tree
  implementations.
- Its `MuZeroCollector` is episode based. It works with a vectorized
  environment manager, tracks ready env ids, asks the policy for actions, steps
  the env manager, stores search stats, appends transitions to game segments,
  and emits completed segments to replay.
- Its CartPole MuZero config uses `collector_env_num = 8`,
  `n_episode = 8`, `num_simulations = 25`, `batch_size = 256`, and
  `update_per_collect = 100`. This is a useful small-framework baseline, not a
  claim about our optimal CurvyTron shape.
- LightZero's public adapter shape is closer to single-agent Gym-style tasks
  and alternating board games than simultaneous multiplayer CurvyTron. A
  CurvyTron adapter probably has to hide opponents inside an ego wrapper first.

Gymnasium vector envs:

- Vector envs run multiple independent copies of an environment at once.
- `SyncVectorEnv` batches in one process. `AsyncVectorEnv` uses
  multiprocessing and pipes.
- `reset` and `step` return batched observations, rewards, terminations,
  truncations, and info.
- Autoreset mode matters. A wrong autoreset contract can corrupt replay because
  the final transition must be staged before the row is reset.

EnvPool:

- EnvPool is a C++ batched environment pool with a thread pool.
- It supports gym/dm_env style APIs, sync and async execution, and single or
  multiplayer envs.
- The paper and docs frame parallel environment execution as a common RL
  bottleneck and report very high Atari/MuJoCo FPS. This is evidence that a CPU
  C++/threadpool path can matter when env stepping is truly the bottleneck.
- It is not evidence that this project should build an EnvPool-style backend
  before measuring the full loop.

IMPALA and SEED RL:

- IMPALA decouples actors from the learner. Actors generate trajectories while a
  learner trains from batches. V-trace corrects off-policy drift.
- SEED RL moves inference to the learner/accelerator and leaves actors as
  environment runners. It batches inference requests centrally. Its repo also
  supports multiple envs per actor through `env_batch_size`.
- SEED's local examples include `4` actors with `4` envs each for Atari/DMLab
  and `4` actors with `32` envs each for MuJoCo PPO. Its README also mentions a
  HalfCheetah setup with `8 * 32 = 256` parallel environments.
- The lesson is not "copy SEED now." The lesson is that central inference helps
  only when batching wins more than actor/learner latency costs.

## How Many Games Or Envs At Once?

There is no universal number. The right number is the one that keeps the
slowest useful component busy without making latency, memory, or staleness
unacceptable.

Useful anchors:

| System | Typical parallelism from checked sources | What it means for us |
| --- | ---: | --- |
| MuZero paper pseudocode | 350 Atari actors, 3000 board-game actors | Research-scale self-play. Not a v0 target. |
| LightZero CartPole MuZero config | 8 collector envs, 8 episodes, 25 simulations | Small contained framework run. Good smoke scale. |
| Gymnasium vector docs | User-chosen `num_envs`; examples show 2 or 3 | API pattern, not a performance prescription. |
| SEED local examples | 4x4, 4x1, 4x32; HalfCheetah 8x32 | Multiple envs per actor are normal when actors wait on inference. |
| Current repo actor bridge | Batch sizes 32 and 128 in local scouts | Useful local shape work, not final throughput. |
| Current repo Mctx synthetic | L4 run recorded `B=64`, 16 simulations | Useful GPU search shape, not full training. |

The bottleneck decides the number:

- If CPU env stepping is slow, add env workers or optimize env stepping.
- If GPU MCTS/model is slow, raise root batch size until GPU is busy, lower
  simulations, or shrink hidden state.
- If transfer latency is high, batch bigger or keep more of the loop on one
  side of the CPU/GPU boundary.
- If replay/learner is slow, actors will fill replay and then create stale or
  wasted data.
- If p95/p99 action latency grows, bigger batches may hurt control even when
  average throughput improves.

## Bottleneck Cases

### Env Stepping Is The Bottleneck

This happens when:

- policy is cheap or pure-policy;
- MCTS has few simulations;
- the real environment has expensive collisions, trail writes, observation
  generation, wrapper dicts, reset/autoreset, or debug/event logging;
- actors spend most wall time inside env stepping and observation packing;
- GPU inference/search is idle waiting for env rows.

Current repo evidence points in this direction for some environment-only scouts:
toy-v0 profiling found segment/trail rasterization and observation copies
visible; vector batch-row timing found debug event output can be around half of
the measured narrow step bucket; obs/reward packing is currently cheap relative
to the narrow env step.

### MCTS Or Model Inference Is The Bottleneck

This happens when:

- `num_simulations` is large;
- hidden state is large;
- root batch is too small to amortize overhead;
- tree storage is memory heavy;
- recurrent inference dominates;
- actors are idle waiting for actions;
- p50/p95/p99 action latency tracks search time, not env step time.

Mctx makes this easier to batch, but the tree stores per-node data and
embeddings. Hidden size and simulations are first-order memory costs. The local
Mctx note already warns that a spatial hidden tree can become large quickly.

### Replay Or Learner Is The Bottleneck

This happens when:

- actors produce data faster than the learner consumes it;
- replay serialization/compression/write time is high;
- replay sampling or target construction is slow;
- GPU training is saturated while actors keep generating stale data;
- latest checkpoint age grows;
- policy version in replay lags far behind learner step.

This repo does not yet have a full MuZero learner, so this is mostly a design
constraint. The current actor bridge stages replay in memory only. It does not
measure disk/object-store writes, sampling, target construction, optimizer
updates, or learner idle.

### CPU/GPU Transfer Is The Bottleneck

This happens when:

- observations are built on CPU but every action requires a small GPU
  `device_put`;
- search outputs move back to CPU one tiny batch at a time;
- legal masks, action weights, root values, and replay targets are copied too
  often;
- GPU utilization is low while CPU waits for transfer or synchronization;
- host-device transfer time is a large share of action latency.

The cure is not automatically a GPU environment. Often the cure is bigger
batches, fewer transfers, in-process central inference, or keeping replay target
construction on the side where the data already lives.

## Amdahl's Law In Plain Terms

Amdahl's law says total speedup is limited by the fraction of wall time you
actually improve.

Formula:

```text
new_total_time = unchanged_time + improved_time / speedup
total_speedup = old_total_time / new_total_time
```

Small examples:

| Wall-time share improved | Part sped up | Local speedup | Total speedup |
| ---: | --- | ---: | ---: |
| 10% | env stepping | 10x | 1.10x |
| 20% | env stepping | 10x | 1.22x |
| 50% | env stepping | 10x | 1.82x |
| 80% | MCTS/model | 2x | 1.67x |
| 80% | MCTS/model | 10x | 3.57x |
| 5% | obs packing | 10x | 1.05x |
| 90% | combined hot loop | 2x | 1.82x |

Concrete warning: if MCTS/model is 80% of wall time, and env stepping is only
10%, then a 10x env speedup changes total time from:

```text
80% MCTS + 10% env + 10% other
to
80% MCTS + 1% env + 10% other = 91% of old time
```

That is only about `1.10x` total speedup. It is nice, but it will not change the
training loop by itself.

## What We Should Measure

The benchmark worker should measure wall-clock shares, rates, tails, and
staleness in one actor-loop report. Mean steps/sec is not enough.

Required timing buckets:

| Metric | Meaning |
| --- | --- |
| p50/p95/p99 action latency | Time from env observation ready to action ready for env step. Report tails, not only mean. |
| completed games/min | End-to-end self-play throughput. This catches reset, long episodes, and terminal handling. |
| env step time | Real environment transition, including physics, collision, trail writes, rewards, done/truncated, and info. |
| observation packing time | Convert env state to `[B_env, P, ...]`, then live ego rows `[B_ego, ...]`, masks, ids, legal actions. |
| policy initial inference time | Representation/prediction at root before search. |
| MCTS/search time | Tree search plus recurrent model calls. Split compile/first-run from steady state. |
| action unmap/encode time | Convert ego decisions back to joint env actions. |
| CPU to GPU transfer time | Host observations/masks to device. Include synchronization. |
| GPU to CPU transfer time | Actions, action weights, root values, diagnostics back to host if needed. |
| replay stage time | Copy transition data into in-memory chunk arrays. |
| replay write time | Serialize/compress/write chunk to disk/Volume/bucket, separate from in-memory stage. |
| reset/autoreset time | Reset finished rows after final transition has been staged. |
| learner sample time | Replay sample and target construction. |
| learner update time | Forward/backward/optimizer/checkpoint update. |
| learner idle time | Learner waiting for replay or batches. |
| actor idle time | Actors waiting for inference, search, learner, replay write, or reset. |
| GPU idle/utilization | Device wait, memory, and utilization metrics when available. |
| policy staleness | `latest_learner_step - actor_model_step_used_for_search`, per replay row or chunk. |

Also record:

- `B_env`, `players`, live ego rows, padded ego rows;
- action count, observation schema, hidden shape, simulations, max depth;
- action repeat;
- event/debug mode;
- chunk size and bytes/chunk;
- episode length distribution;
- terminal causes and truncation reasons;
- policy version used for each replay chunk.

## Architecture Implications

### CPU Env Plus GPU MCTS/Model

This is the best first serious architecture if we use Mctx or a GPU learner.

Shape:

```text
CPU vector env batch
  -> CPU obs/reward/mask pack
  -> one batched host-to-device copy
  -> GPU representation/prediction/MCTS
  -> actions back to CPU
  -> CPU env step
  -> replay chunks
```

Keep it in one process or one tightly coupled process group first. Use fixed
batch profiles. Pad tail rows. Measure host-device transfer and action latency.

### GPU Env Only If The Loop Stays On Device

A GPU environment is worth considering only if:

- env stepping is a measured bottleneck;
- the environment state, observation packing, policy/search, replay targets, and
  learner can stay on device or transfer rarely;
- GPU stepping does not steal memory bandwidth from MCTS/model/learner;
- correctness fixtures still pass.

If the loop becomes CPU env -> GPU env -> CPU replay -> GPU search every tick,
the transfer cost can erase the benefit.

### CPU Actor Pools

CPU actor pools make sense after one-container measurement proves CPU env work
is starving the GPU or learner.

Good shape:

```text
many CPU actor processes
  each owns many envs
  write chunky replay or request batched inference
one GPU process
  batches inference/search and/or trains
```

Bad shape:

```text
one remote call per action
one queue item per env step
one tiny replay file per transition
```

### Async Actors

Async actors improve throughput when actors and learner do not need to wait for
each other. They also introduce staleness.

Use them only with:

- policy version in every replay row/chunk;
- maximum staleness limits;
- replay age sampling policy;
- actor idle and learner idle metrics;
- fixed eval that can detect stale-policy regressions.

### Central Inference

Central inference helps when many CPU actors need one accelerator and local
model inference on each actor is wasteful. SEED RL is the example: actors run
environments, learner/accelerator performs batched inference.

For this repo, try central inference inside one container first: threads or
processes submit observations to one GPU search/inference worker. A network RPC
or Modal boundary should wait until local batching proves the need.

### Bigger Sync Batches

Bigger synchronous batches are the simplest first lever:

- more roots per Mctx call;
- fewer transfers per decision;
- better GPU utilization;
- easier replay chunking.

The risk is tail latency. If env rows wait for a slow reset or if action
latency p99 grows, games/min and learning may get worse. Measure p50/p95/p99,
not only average throughput.

## Critique

High-level self-critique:

The biggest risk in this note is that it assumes MuZero-style search is worth
optimizing before the project has proven a strong simple learner. The local
training critiques are right: current Pong and dummy scaffolds are not MuZero
quality evidence. A policy-gradient, CEM, PPO, or other simpler baseline may be
the better near-term path if it gives clearer learning pressure per wall-clock
hour.

Where this advice can become a rabbit hole:

- Optimizing the environment because env microbenchmarks are visible, while the
  future full loop is actually dominated by MCTS/model or learner updates.
- Building a GPU/JAX environment because Mctx is JAX, even though standard
  MuZero searches learned dynamics, not the real simulator.
- Adapting LightZero deeply enough that we are debugging a framework instead of
  learning whether CurvyTron is learnable.
- Increasing batch size until throughput looks good while p99 action latency,
  stale policies, or replay age quietly break learning.
- Treating debug event/logging speed as production training speed.
- Treating synthetic Mctx throughput as real self-play throughput.

Assumptions that may not apply to CurvyTron:

| Recommendation | Assumption | What might be wrong | Measurement that disproves it |
| --- | --- | --- | --- |
| CPU env first | CPU env can feed GPU search/learner after batching. | Source-faithful collision and observation work may be too slow. | GPU/search idle is high while env+obs+reset is most wall time after simple CPU batching. |
| Mctx fallback/comparison | Ego-only MuZero search is a useful approximation. | Simultaneous opponent actions may make single-action recurrent dynamics misleading. | MCTS improves policy targets in search but eval does not improve over pure policy at equal wall-clock; value calibration fails by opponent slice. |
| LightZero-first custom smoke | LightZero can preserve our env, telemetry, and artifacts cheaply. | Its collector/API may not fit simultaneous ego rows or our metadata. | Wrapper-only and pure-policy LightZero collection are much slower than direct runner, or required metadata/search stats cannot be recovered without invasive patches. |
| Bigger sync batches first | Larger batches improve throughput without hurting control. | CurvyTron may need low-latency actions or short episodes may create tail stalls. | p95/p99 action latency rises and games/min or eval quality falls while average roots/sec improves. |
| Central inference later | In-process batching is enough for v0. | Many CPU actors may be needed earlier than expected. | GPU utilization remains low with one-container batching while CPU actors are saturated and actor idle is dominated by local inference/search queueing. |
| Delay GPU env | Transfer and GPU contention will dominate unless everything stays on device. | A compact JAX/PyTorch env might remove a larger CPU packing/transfer bottleneck. | A correctness-matched all-device spike beats CPU-env+GPU-search in full action latency and games/min, including replay target handling. |
| Delay C++/Rust/EnvPool-like backend | NumPy/Numba-style CPU batching will be enough for early loops. | Source-faithful trail/collision scans may hit a hard CPU ceiling. | Env step plus obs generation remains the largest wall share after vectorization and targeted Numba, and adding actor CPUs does not scale. |

What I might be missing:

- Learning quality can dominate systems speed. A slower loop with better targets
  can beat a faster loop with bad reward, stale opponents, or impossible eval
  gates.
- Evaluation cost can become real. Fixed checkpoint scoreboards, heldout seeds,
  and opponent pools may need their own parallel architecture.
- Replay quality matters as much as replay throughput. Old, stale, or
  one-policy replay can make actors look productive while hurting learning.
- CurvyTron may need action repeat or lower decision frequency. That could
  reduce MCTS cost more than any simulator optimization, but it changes control
  semantics and must be benchmarked as a ruleset choice.
- MCTS may be the wrong first planning tool for simultaneous real-time steering.
  Short policy-only rollouts, CEM, or heuristic safety layers may produce better
  early pressure.

## Implications For Current Environment-Speed Work

Worth doing now:

- Keep the source-fidelity guardrails and fixed-shape array work.
- Add split timers for movement, point/body insertion, collision, observation,
  wrapper/dict output, reset/autoreset, replay staging, and replay writing.
- Extend the actor-loop bridge so it reports action latency percentiles,
  games/min, env rows/sec, ego rows/sec, replay rows/sec, and wall-time share by
  bucket.
- Keep comparing debug-event and no-event modes, but label debug-event speed as
  debug speed.
- Measure host-device transfer in the Mctx `curvytron_actor_bridge_sample` path.
- Keep the Mctx synthetic benchmark as search/runtime evidence only.
- Keep LightZero custom-env work small: config/import smoke first, tiny trainer
  smoke second, reject if telemetry or overhead is bad.

Should wait:

- GPU-resident environment rewrite.
- JAX-native environment rewrite.
- C++/Rust simulator.
- EnvPool-style engine.
- Distributed Modal actor fleet.
- Central model server over network.
- Full joint-action MCTS.
- League/Elo/population training.
- Heavy replay/object-store architecture.

Likely rabbit holes:

- Optimizing the current toy-v0 env as if it were production CurvyTron.
- Moving debug event rows into the production hot path because the benchmark
  currently emits them.
- Treating `steps/sec` without policy/search/learner as training speed.
- Calling Modal Queues or Functions inside the action loop.
- Making Mctx drive the real simulator inside search.
- Increasing simulations before proving search improves eval at equal
  wall-clock budget.

2026-05-09 Worker D status note:

- The fixture-seeded vector actor bridge already reports action-latency
  percentiles, staged transition rates, a terminal-row games/min proxy, bucket
  shares, and debug-event versus no-event comparisons.
- A small timing-output helper was added to the toy object-env parallel bridge
  to rank bucket shares and print the largest timed bucket beside p95/p99
  actor-step latency.
- The cheap local follow-up still points at env-step cost for the current debug
  fixture slice, but this is not a GPU, LightZero, real-MCTS, or production
  self-play claim. The biggest unknown remains calibrated model/search timing
  at the env-step versus policy/search boundary.

## Concrete Benchmark Handoff

Give the benchmark worker these near-term tasks:

1. Extend `scripts/benchmark_vector_actor_loop_bridge.py` with per-action
   latency percentiles: p50, p95, p99, max. Measure observation-ready to
   action-ready.
2. Add completed games/min or completed episodes/min to the actor bridge. If the
   current fixture bridge cannot represent full games, add a clear
   `not_measured` field and a follow-up full-episode benchmark.
3. In the actor bridge, report wall-time share for reset copy, env step,
   obs/reward pack, policy root, policy search, action select, action encode,
   replay stage, autoreset, and loop overhead.
4. Add replay serialization timing after in-memory staging: `.npz` or chosen
   chunk format write time, bytes/chunk, rows/chunk, and write throughput.
5. Add reset/autoreset correctness and timing for finished rows after replay
   staging. Keep final transition and reset observation separate.
6. Add policy version fields to synthetic actor bridge chunks:
   `model_step_used_for_search`, `latest_model_step_seen`,
   `policy_staleness_steps`.
7. Add actor idle and learner idle placeholders even before the real learner
   exists. For now, define actor idle as waiting for policy/search; define
   learner idle as `not_applicable_no_learner`.
8. Run a local matrix over `B_env in {32, 128}`, `rollout_steps in {1, 4, 16}`,
   `event_mode in {debug-event, no-event}`, and synthetic simulations in
   `{1, 4, 16}`.
9. Run the Modal Mctx synthetic benchmark in `curvytron_actor_bridge_sample`
   mode with a small matrix: `B_env in {16, 64}`, `num_simulations in {4, 16}`,
   and report host setup, host-to-device transfer, compile/first run, steady
   search, output transfer, and root batch size.
10. Add one end-to-end summary table that ranks bucket shares from largest to
    smallest. The first optimization target should be the largest measured
    bucket that is production-relevant.

## Files Read And Created

Created:

- `docs/research/training_loop_bottlenecks_amdhals_law_2026-05-09.md`

Opened/read locally:

- `docs/design/muzero_modal_architecture.md`
- `docs/research/muzero_architecture_deep_dive.md`
- `docs/research/mctx_integration.md`
- `docs/research/lightzero_integration.md`
- `docs/research/multiplayer_selfplay_muzero.md`
- `docs/research/environment/performance_vectorization_plan.md`
- `docs/research/performance_vectorization.md`
- `docs/design/training_architecture.md`
- `docs/design/training_eval_protocol.md`
- `docs/research/modal_patterns.md`
- `docs/experiments/environment/2026-05-09-vector-batch-row-timing.md`
- `docs/experiments/environment/2026-05-09-vector-obs-reward-packing.md`
- `docs/experiments/environment/2026-05-09-toy-v0-performance-scout.md`
- `docs/working/training_coach_self_critique_2026-05-09.md`
- `docs/working/pong_selfplay_training_plan_2026-05-09.md`
- `scripts/benchmark_policy_search_batch_standin.py`
- `scripts/benchmark_vector_actor_loop_bridge.py`
- `tests/test_benchmark_vector_actor_loop_bridge.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`

Also ran broad `rg` searches across `docs`, `src`, `tests`, `README*`, and
`pyproject.toml` to find relevant MuZero/Mctx/LightZero/Modal/vectorization
context.
