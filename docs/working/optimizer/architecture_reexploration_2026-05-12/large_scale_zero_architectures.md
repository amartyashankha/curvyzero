# Large-Scale Zero-Style Training Architectures

Date: 2026-05-12

Status: focused optimizer research note. This is architecture guidance, not a
CurvyTron learning-quality claim.

## Plain Answer

Large AlphaZero/MuZero systems separate the hot loop:

```text
checkpoint K
  -> actors run env + search
  -> searched games/chunks go to replay
  -> learner samples replay and updates weights
  -> checkpoint/eval jobs publish or judge K+1
```

CurvyTron should copy this separation, not the giant hardware scale. The next
step should be a synchronous collect-only fanout from one frozen checkpoint,
then merge/import, then learner work. That directly tests whether searched data
generation is the wall-clock bottleneck without changing MuZero semantics.

## 1. Pieces That Are Usually Separated

**Actors.** Actors own environment interaction and produce games or trajectory
chunks. In Zero-style training they normally run search at each decision and
store search-improved policy targets, not just raw policy-head actions.
AlphaZero describes self-play games where moves are selected by MCTS and the
network is trained to match both game outcome and search probabilities.
[AlphaZero paper](https://arxiv.org/abs/1712.01815),
[HTML text](https://ar5iv.labs.arxiv.org/html/1712.01815).

**Inference/search.** Search is often the expensive actor-side loop. It may be
local to actors, centralized, or split into batched accelerator-backed inference
workers. OpenSpiel's AlphaZero docs are a good small-system reference: actors
play games and send trajectories to the learner; the Python path does not batch
inference, while the C++ path uses threads, shared cache, batched inference, and
GPUs. [OpenSpiel AlphaZero docs](https://openspiel.readthedocs.io/en/latest/alpha_zero.html).
MCTX is a search-kernel reference: its JAX searches operate on batches in
parallel to use accelerators, but it is not a replay/learner system.
[MCTX](https://github.com/google-deepmind/mctx).

**Replay/storage.** Replay stores completed games or chunks with enough metadata
to reconstruct targets and know data age. OpenSpiel's learner pulls actor
trajectories into FIFO replay, samples once enough data exists, updates, saves a
checkpoint, and refreshes actor models.
[OpenSpiel AlphaZero docs](https://openspiel.readthedocs.io/en/latest/alpha_zero.html).
MuZero's released pseudocode explicitly splits network training from self-play:
the two communicate through latest checkpoints and finished games, and
`num_actors` jobs write games to replay. [MuZero pseudocode copy](https://gist.github.com/Mononofu/6c2d27ea1b3a9b3c1a293ebabed062ed);
Nature notes the official `pseudocode.py` is in the supplementary information.
[Nature MuZero](https://www.nature.com/articles/s41586-020-03051-4).

**Learner.** The learner builds batches from replay, trains the model, and
publishes weights. AlphaZero's reported scale shows the intended split: 5,000
first-generation TPUs generated self-play, while 64 second-generation TPUs
trained the network for 700,000 minibatch steps of size 4,096.
[AlphaZero HTML text](https://ar5iv.labs.arxiv.org/html/1712.01815).

**Checkpoint/eval.** Checkpoint publishing and evaluation are separate from hot
collection. OpenSpiel runs evaluator processes/threads separately. Lc0 and
KataGo show the open-source distributed pattern: clients generate self-play
games, upload data, training servers produce new networks, and clients refresh.
[Lc0 overview](https://lczero.org/dev/overview/),
[Lc0 getting started](https://lczero.org/dev/wiki/getting-started/),
[KataGo distributed training](https://katagotraining.org/),
[KataGo repo](https://github.com/lightvector/katago).

## 2. Why Massive Self-Play Helps

Massive self-play helps when searched data is the slow part. Each actor spends
compute to turn positions into stronger policy/value targets through search.
More actors can produce more searched decisions per hour and keep the learner
from waiting.

The tradeoff is actor throughput versus learner freshness:

- If actors are slow and the learner is idle, more actors help.
- If the learner is slow, more actors just build a queue.
- If one checkpoint produces too much data, the next policy update is delayed.
- If actors are too similar, extra games can be correlated rather than useful.
- If replay write/merge is slow, fanout moves the bottleneck rather than fixing
  it.

SEED RL is not MuZero, but it is relevant architecture evidence. It moves model
inference to the learner side, batches actor requests, and keeps actors as small
environment loops to reduce CPU inference waste and parameter-transfer
bandwidth. It also reports cases where higher throughput or larger batches
improved wall-clock but hurt sample complexity. Treat that last point as an
uncertain CurvyTron risk, not a proven CurvyTron result.
[SEED RL](https://arxiv.org/abs/1910.06591),
[HTML text](https://ar5iv.labs.arxiv.org/html/1910.06591).

## 3. Does Policy Staleness Matter?

In a synchronous batch:

```text
freeze K -> collect K data -> train K+1
```

There is no hidden actor-fleet staleness. Every chunk is intentionally from
checkpoint `K`. The only question is whether the batch is too large: did we
collect more old-policy data than the learner can use before a better `K+1`
should exist?

In a continuous actor fleet:

```text
actors collect while learner trains and publishes newer checkpoints
```

Staleness is real but normal. OpenSpiel explicitly warns that too many actors
for the hardware can make games finish slower, so data can be more out of date
relative to current weights. The usual controls are checkpoint ids, actor
refresh cadence, replay age windows, and eval gates.

SEED-style central inference changes the shape again: actors can receive actions
from very recent model weights, while one trajectory may include actions from
several nearby policy versions. Useful later; unnecessary for CurvyTron's first
fanout experiment.

## 4. Which Bottleneck Moves First?

There is no universal order. The likely progression is:

```text
few actors:
  env/render, reset/setup, search/tree bookkeeping, model inference

moderate fanout:
  shared GPU inference/search batching, CPU saturation, process/IPC overhead

larger fanout:
  replay chunk write, merge/import, learner input building, checkpoint IO

after collection is fast:
  learner forward/backward and update throughput
```

For CurvyTron, the first limiter is uncertain. Current local notes suggest these
are the candidates to measure:

- source-state render and visual stack CPU time;
- frozen opponent inference in subprocess workers;
- live MuZero inference and MCTS batching;
- Python/C++ tree coordination and env-manager IPC;
- replay chunk write/merge;
- learner batch construction and GPU utilization.

This is exactly why the first scale test should be a ladder instead of a giant
service build.

## 5. CurvyTron Experiments To Run

Run a collect-only synchronous fanout:

```text
For one frozen checkpoint K:
  N = 1, 2, 4, 8 collect-only jobs
  env_variant = source_state_fixed_opponent
  same frozen opponent checkpoint
  same render mode
  same MCTS simulations
  same episode/death settings
  write searched chunks
  merge/import chunks
  run fixed learner work
```

Every chunk should carry checkpoint id, env variant, render mode, action schema,
MCTS settings, seed, model hash, opponent checkpoint, software version, and
completed-game/decision counts.

Measure:

- games/sec, decisions/sec, MCTS sims/sec;
- average neural inference batch size;
- CPU utilization and GPU utilization;
- chunk bytes/sec, write time, merge/import time;
- learner sample/build/update time;
- checkpoint publish/read time;
- replay age and checkpoint id distribution.

Interpretation:

- If `N=1/2/4/8` scales and merge/learner stay small, expand actor fanout.
- If GPU inference saturates first, test batched inference/search.
- If env/render saturates first, optimize observation/render or allocate more
  CPU per actor.
- If replay merge dominates, fix chunk format and merge path before services.
- If learner dominates after fanout, consider learner batching or multi-GPU, with
  Coach checking sample-efficiency effects.

Do not copy yet:

- a permanent async actor service before chunk semantics are proven;
- policy-head-only collection as a replacement for searched MuZero targets;
- multi-GPU learner work before the learner is actually the bottleneck;
- a new search backend before stock LightZero timing says search/inference is
  the limiting component.

## Bottom Line

CurvyTron should copy the actor/search/replay/learner/checkpoint separation and
prove it with `N=1/2/4/8` collect-only fanout. In synchronous fanout, policy
staleness is controlled by construction; the real question is where the
bottleneck moves and whether the learner can turn searched chunks into useful
updates quickly enough.
