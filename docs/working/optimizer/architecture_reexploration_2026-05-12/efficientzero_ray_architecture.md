# EfficientZero / Ray Architecture Research

Date: 2026-05-12

Status: architecture evidence note only. This does not recommend migrating
CurvyTron to EfficientZero.

## Sources Inspected

- Existing optimizer notes:
  - `docs/working/optimizer/framework_reassessment_2026-05-11.md`
  - `docs/working/optimizer/distributed_muzero_architecture_research_2026-05-11.md`
- Upstream repo cloned in `/private/tmp/EfficientZero`.
- Upstream commit inspected: `468bb0309f6d5a632a53da9c7d329f88fc9ebf8e`
  from `https://github.com/YeWR/EfficientZero`.
- Key upstream source links:
  - README usage and architecture note:
    https://github.com/YeWR/EfficientZero/blob/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/README.md#L20-L56
  - Ray worker construction:
    https://github.com/YeWR/EfficientZero/blob/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/core/train.py#L445-L468
  - Learner/checkpoint loop:
    https://github.com/YeWR/EfficientZero/blob/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/core/train.py#L360-L423
  - Self-play worker:
    https://github.com/YeWR/EfficientZero/blob/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/core/selfplay_worker.py
  - Replay buffer:
    https://github.com/YeWR/EfficientZero/blob/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/core/replay_buffer.py
  - Reanalysis workers:
    https://github.com/YeWR/EfficientZero/blob/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/core/reanalyze_worker.py
  - C++/Cython tree search:
    https://github.com/YeWR/EfficientZero/tree/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/core/ctree

## Plain Read

EfficientZero is useful as a full-loop Ray reference: shared weights, replay,
self-play, target reanalysis, learner, checkpoint save, and eval are separated
into explicit roles. It is not just a local `collect -> train` loop.

The important nuance is that this codebase is still Atari-shaped. The public
Atari config uses `num_actors=1`; most parallelism comes from `p_mcts_num`
parallel envs inside one self-play `DataWorker`, plus many CPU/GPU reanalysis
workers. The code can instantiate more `DataWorker`s if a config changes
`num_actors`, but the CLI exposes `cpu_actor`, `gpu_actor`, and `p_mcts_num`,
not `num_actors`.

## Process Topology

`main.py` initializes Ray with user-specified CPUs, GPUs, and object-store
memory, then calls `train(config, ...)`. The default training shell uses four
GPUs, ninety-six CPUs, fourteen CPU batch workers, twenty GPU batch workers,
and four parallel MCTS envs per self-play worker.

`train()` builds:

- `SharedStorage.remote(model, target_model)` for current self-play weights,
  delayed target/reanalysis weights, step counter, start signal, and logs.
- `ReplayBuffer.remote(config=config)` for completed trajectory blocks and
  prioritized transition sampling.
- `QueueStorage(18, 25)` as an MCTS/reanalysis context queue.
- `QueueStorage(15, 20)` as a ready learner-batch queue.
- `BatchWorker_CPU` actors for replay sampling and CPU-side batch context
  preparation.
- `BatchWorker_GPU` actors for target/reanalysis inference and MCTS.
- `DataWorker` actors for self-play collection.
- One Ray eval worker.
- The learner loop in the main process, not as a Ray actor.

Evidence: `/private/tmp/EfficientZero/core/train.py:445`.

## Learner And Shared Storage

The learner waits until replay has at least `start_transitions`, then signals
workers to start producing learner batches. It pops complete batches from
`batch_storage`; if the queue is empty it sleeps and warns.

The learner publishes current self-play weights every `checkpoint_interval`
and publishes target/reanalysis weights every `target_model_interval`. In the
Atari config those are `100` and `200` learner steps. Disk checkpoints are
saved every `save_ckpt_interval`.

Shared storage is an in-memory Ray actor. It stores actual model objects and
serves `get_weights()` / `set_weights()` calls. It is a checkpoint publisher
and log aggregator, but not a durable artifact service or versioned checkpoint
registry.

Evidence:

- `/private/tmp/EfficientZero/core/train.py:360`
- `/private/tmp/EfficientZero/core/storage.py:33`
- `/private/tmp/EfficientZero/config/atari/__init__.py:19`

## Self-Play Workers

`DataWorker` is a Ray actor with `num_gpus=0.125`. Each worker owns a local
network copy and `p_mcts_num` live envs. At each env step it:

1. Builds a batch of root observations across its envs.
2. Runs `model.initial_inference` once on that batch.
3. Creates `cytree.Roots(env_nums, action_space_size, num_simulations)`.
4. Runs MCTS against those roots.
5. Samples one action per env from visit counts.
6. Steps each single-agent Atari env with one scalar action.
7. Stores action, observation, reward, root value, and visit distribution into
   `GameHistory`.
8. Writes completed or padded history blocks to replay.

It refreshes its local model when `trained_steps // checkpoint_interval`
advances. It also throttles when self-play has generated more of its assigned
transition quota than the learner progress ratio.

Evidence:

- `/private/tmp/EfficientZero/core/selfplay_worker.py:15`
- `/private/tmp/EfficientZero/core/selfplay_worker.py:106`
- `/private/tmp/EfficientZero/core/selfplay_worker.py:185`
- `/private/tmp/EfficientZero/core/selfplay_worker.py:262`

## Replay Buffer And Trajectory Shape

Replay is a Ray actor. It stores `GameHistory` blocks, a flat transition
priority array, and `game_look_up` entries mapping sampled transition indices
back to `(game_id, step_pos)`. It samples transition indices by priority and
returns game objects plus positions to the CPU reanalysis workers.

`GameHistory` is a single-agent trajectory block. It stores:

- `actions`: one scalar action per step
- `rewards`: one scalar reward per step
- `child_visits`: one root visit distribution per step
- `root_values`: one scalar root value per step
- `obs_history`: frame-stack source observations

For visual memory, observations are optionally JPEG-encoded strings, and
`game_over()` puts the observation array into the Ray object store to reduce
copying.

Evidence:

- `/private/tmp/EfficientZero/core/replay_buffer.py:7`
- `/private/tmp/EfficientZero/core/replay_buffer.py:79`
- `/private/tmp/EfficientZero/core/game.py:32`
- `/private/tmp/EfficientZero/core/game.py:168`
- `/private/tmp/EfficientZero/core/utils.py:322`

## Reanalysis And Batch Making

EfficientZero has a distinct middle tier between replay and the learner.

`BatchWorker_CPU` samples from replay and prepares CPU-heavy context:
input observations, unroll action masks, off-policy TD-step shortening,
reward/value bootstrap context, and a split between reanalyzed policy targets
and old stored policy targets. It pushes this context into `mcts_storage`.

`BatchWorker_GPU` pops that context, loads new target weights when provided,
then prepares final targets:

- value/value-prefix targets from target-network inference, optionally using
  MCTS root values if `--use_root_value` is enabled;
- reanalyzed policy targets by running MCTS on stored observations for some
  fraction of the batch;
- non-reanalyzed policy targets from stored `child_visits`.

The GPU worker slices inference by `mini_infer_size` to fit many GPU actors.
The final `[inputs_batch, targets_batch]` is pushed to `batch_storage`, and the
learner only consumes already-built batches.

Evidence:

- `/private/tmp/EfficientZero/core/reanalyze_worker.py:14`
- `/private/tmp/EfficientZero/core/reanalyze_worker.py:161`
- `/private/tmp/EfficientZero/core/reanalyze_worker.py:266`
- `/private/tmp/EfficientZero/core/reanalyze_worker.py:394`

## Search Implementation

Tree bookkeeping is C++ exposed through Cython. Python owns the MCTS loop;
C++ performs batched traversal and backpropagation over root trees, while
PyTorch performs batched recurrent inference for selected leaves.

For each simulation:

- C++ `batch_traverse` walks one path per root and returns leaf hidden-state
  indices plus selected last actions.
- Python gathers those hidden states into a tensor batch.
- PyTorch runs `model.recurrent_inference` over that batch.
- C++ `batch_back_propagate` expands leaves and backs values up through each
  path.

This is batched over root count, but not a fully accelerator-resident MCTS.
The loop crosses Python/C++/PyTorch every simulation.

Evidence:

- `/private/tmp/EfficientZero/core/mcts.py:13`
- `/private/tmp/EfficientZero/core/mcts.py:44`
- `/private/tmp/EfficientZero/core/ctree/cytree.pyx:39`
- `/private/tmp/EfficientZero/core/ctree/cnode.cpp:380`

## CPU / GPU Split

Observed split:

- Env stepping, replay sampling, priority bookkeeping, observation decoding,
  and CPU batch-context assembly are CPU/Ray-object-store work.
- Self-play model inference and MCTS leaf inference run inside fractional-GPU
  `DataWorker` actors.
- Reanalysis model inference and reanalysis MCTS run inside fractional-GPU
  `BatchWorker_GPU` actors.
- Learner forward/backward runs on `config.device` in the main process and can
  use torch AMP.
- C++ tree selection/backprop removes the slowest pure-Python tree operations,
  but inference is still PyTorch GPU work.

This split is relevant to CurvyTron because it separates env/search throughput
from learner throughput and makes queue starvation visible.

## Patterns That Could Help CurvyTron

- Keep explicit system roles: actor/search workers, replay, batch builder,
  learner, checkpoint/shared-weight publisher, eval.
- Treat searched self-play as a data factory, not as a synchronous subroutine
  inside one trainer.
- Use bounded queues with watermarks between CPU context preparation, GPU
  reanalysis/search, and learner batches.
- Record and control policy freshness through checkpoint intervals and actor
  refresh cadence.
- Batch roots across independent envs; for CurvyTron this suggests batching
  across `B` envs and possibly `P` seat-root searches as `[B * P]` roots before
  committing one physical simultaneous tick.
- Store trajectory blocks with enough future padding for unroll and TD targets.
- Use object-store handles or compact encodings for visual observations when
  replay movement dominates.
- Split CPU target-context work from GPU reanalysis work so the learner sees
  ready batches.
- Use C++/Cython tree kernels or another compiled/batched search primitive for
  selection/backprop while keeping neural inference batched.

## Hard To Adapt To CurvyTron

- The environment contract is one scalar action -> one observation, one reward,
  one done. CurvyTron needs simultaneous `[P]` actions and per-player or
  game-level terminal/reward semantics.
- `GameHistory` stores one action, one policy distribution, and one scalar
  value/reward per step. It has no player id, seat perspective, joint action,
  pending-action privacy, or per-seat target fields.
- The C++ tree has a `to_play` field, but the inspected code expands and
  backpropagates with `0`; there is no active alternating-player value sign
  handling, simultaneous-move matrix logic, or multiplayer payoff vector.
- Reanalysis assumes one policy target per sampled transition. CurvyTron would
  need either per-seat reanalysis, joint-action reanalysis, or a clearly labeled
  frozen-opponent/focal-player target contract.
- The model dynamics takes one scalar action and appends a single action plane
  normalized by `action_space_size`. Joint actions could be flattened for
  two-player `3x3`, but that changes semantics to centralized control; true
  independent-seat search would require different target/data plumbing.
- The Atari visual path is `96x96`, frame-skipped, frame-stacked, and optionally
  JPEG encoded. CurvyTron can borrow the storage idea, but not the ALE-specific
  wrappers or episode-life assumptions.
- The Ray dependency set is old (`ray==1.0.0`, old Gym/Atari stack). Reusing the
  code directly would carry dependency and maintenance risk separate from the
  algorithmic work.
- Multiplayer scaling is unresolved. Joint actions grow as `3^P`; independent
  per-seat search grows root count as `B * P` and needs clean hidden-action and
  reward-perspective semantics.

## Evidence-Bound Takeaway

EfficientZero is strongest here as an architecture reference for a continuous
Ray self-play/replay/reanalysis/learner system. Its reusable lesson is the
middle-tier pipeline: CPU workers sample and shape replay contexts, GPU workers
reanalyze with batched search/inference, and the learner consumes ready batches
while self-play continues.

The hard boundary is semantics, not just engineering. EfficientZero's concrete
data model, env wrapper, model dynamics action input, and tree search are
single-agent Atari-shaped. Any CurvyTron use would need a new trajectory/target
contract for simultaneous two-player play before the Ray topology could be
trusted.
