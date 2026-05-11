# Distributed MuZero Architecture Research

Date: 2026-05-11

Status: optimizer correction note. This replaces the weaker assumption that the
main CurvyTron speed path is mostly single-loop micro-optimization.

Related framework comparison and simultaneous-action modeling note:
`docs/working/optimizer/framework_reassessment_2026-05-11.md`.

Related component checklist for the full training loop:
`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md`.

## Plain Verdict

The current native CurvyTron LightZero loop is useful as the first
coach-facing trainer surface, but it is not structurally close to the training
systems that made AlphaZero/MuZero fast.

Important correction: the current stock `train_muzero` path is synchronous
inside one trainer container. It collects, pushes to local replay, samples, and
trains in one ordered loop. There is no distributed actor-policy staleness
problem in that current loop.

Those systems are actor/search/replay/learner systems. They generate searched
self-play data at high throughput, keep neural inference batched, write complete
trajectories into replay, and train from replay while actors continue working.

LightZero gives useful MuZero pieces: policy, model, learner, MCTS, C++ tree
search support, vector env collection knobs, and Atari-style configs. The
current CurvyTron source-state trainer uses those pieces through
`train_muzero`. It does not give CurvyTron a ready-made production actor fleet
with shared replay, checkpoint serving, policy-version tracking, and true
simultaneous current-policy `[B,P]` collection.

## Primary Evidence

- AlphaZero used large self-play hardware: the accepted preprint reports
  `5,000` first-generation TPUs for self-play and `64` second-generation TPUs
  for training, with minibatches of `4,096` for `700,000` training steps:
  https://arxiv.org/pdf/1712.01815
- The MuZero pseudocode explicitly splits training into network training and
  self-play generation, connected only by latest checkpoints and finished games.
  It launches `config.num_actors` self-play jobs, with example settings of
  `3000` board-game actors and `350` Atari actors:
  https://gist.github.com/tkukurin/45b3a4cdccf2c99ad7aa013798183fb9
- The MuZero Nature page says the official pseudocode is in the supplementary
  material and that MuZero trains only from self-generated data:
  https://www.nature.com/articles/s41586-020-03051-4
- LightZero advertises efficiency from mixed heterogeneous computing for the
  expensive MCTS parts and has Python/C++ tree search implementations, but its
  public worker docs describe `MuZeroCollector` as a serial collector over an
  env manager:
  https://github.com/opendilab/LightZero
  https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html
- LightZero config supports `collector_env_num`, env managers, CUDA,
  multi-GPU, and `mcts_ctree`, but this is not the same as a distributed actor
  fleet and replay service:
  https://opendilab.github.io/LightZero/tutorials/config/config.html
- OpenSpiel AlphaZero is a strong reference for the missing architecture:
  actors generate self-play games, a learner pulls trajectories into FIFO
  replay, checkpoints are published, and evaluators run separately. Its docs
  also call out that Python has no inference batching and CPU-only inference,
  while the C++ path uses threads, shared cache, batched inference, and GPUs:
  https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- EfficientZero's open implementation is closer to the system shape: Ray,
  CPU/GPU actors, C++/Cython tree search, parallel self-play/reanalysis, and
  explicit worker counts:
  https://github.com/YeWR/EfficientZero
- MCTX is an important alternative inner-loop primitive: JAX-native batched MCTS
  for AlphaZero/MuZero/Gumbel MuZero. It is not a full trainer, but it shows the
  right search shape: operate on batches of roots in parallel on accelerators:
  https://github.com/google-deepmind/mctx
- KataGo and Leela-style projects confirm the production pattern in open source:
  distributed self-play clients generate games, a central pipeline trains and
  publishes networks, and self-play compute dominates training compute:
  https://katagotraining.org/
  https://github.com/lightvector/katago
  https://lczero.org/dev/overview/

## Correction To Optimizer Plan

The first serious 10x target is not "make the current local loop 10x faster."
The target is to move from one synchronous collector/learner loop toward a data
factory:

```text
many actors/search workers
  -> searched trajectory chunks with policy_version metadata
  -> replay/shuffle/storage
  -> learner samples sequences and updates network
  -> checkpoint publisher
  -> actors refresh checkpoint on cadence
```

Do not read `policy_version` / policy lag as a reason not to parallelize. It is
metadata that keeps a parallel system honest. Massive self-play is useful when
the learner can turn searched trajectories into updates fast enough and the
replay window remains representative. If actors outrun the learner by too much,
the issue is usually sample efficiency and replay balance, not an immediate
off-policy correctness failure.

There are three separate regimes:

```text
1. Stock LightZero train_muzero:
   one process/container, ordered collect -> replay -> sample -> learn.
   No actor-fleet staleness. Parallelism comes from env_manager workers and
   collector_env_num/n_episode.

2. Coarse synchronous Modal fanout:
   freeze checkpoint K, launch N collect-only actors, merge chunks, train to K+1.
   No hidden policy lag within the batch. The risk is wasting time/data if N is
   so large that the learner waits too long before improving the policy.

3. Continuous actor/replay/learner:
   actors collect while learner trains and publishes checkpoints.
   This is the AlphaZero/MuZero-style production shape. Track policy version,
   replay age, and checkpoint refresh cadence; do not treat them as blockers.
```

The current native LightZero path should be treated as the baseline trainer and
profile surface. A future distributed actor loop may reuse the same env/config
surface, but should be designed as actor chunks plus replay/learner/checkpoint
handoff rather than endless single-process polish.

## CurvyTron-Specific Issues

- CurvyTron is visual, non-ALE, simultaneous two-player self-play.
- Most AlphaZero systems assume alternating turns. A naive joint-action MCTS
  would multiply action space by player count. For `3` actions and `2` players,
  joint action is only `9`, so this may be tractable for 2-player CurvyTron, but
  the semantics must be explicit.
- The current native trainer is fixed-opponent single-ego, not true
  current-policy self-play. It is useful because it stays close to the working
  LightZero/Pong pattern.
- `source_state_turn_commit` is smoke/profile only. Target audit showed fake
  pending rows and bad reward credit, so train mode is blocked.
- Turing recommendation, candidate/control only until tested: a 9-action
  centralized joint-action wrapper. One scalar action -> `(p0,p1)`, one real
  CurvyTron tick, one reward, `to_play=-1`, `action_space_size=9`. Loud caveat:
  centralized control, not true competitive self-play.
- A future simultaneous self-play collector must label its semantics: staged
  frozen-opponent, independent seat search, turn-commit adapter, or deliberate
  joint-action search. Turn-commit also carries the pending-step reward-credit
  caveat. Do not change the algorithm silently.

## What To Measure Next

Every profile should report:

- env transitions/sec and full games/sec
- MCTS simulations/sec
- neural inference calls/sec
- average neural inference batch size
- GPU utilization and CPU utilization
- replay rows or trajectory chunks/sec
- learner updates/sec
- actor policy version lag if using continuous actors
- replay age distribution
- checkpoint publish/read cost
- time split between env/render/reset, search/tree bookkeeping, model
  inference, replay write/read, learner, eval, and artifact IO

## Near-Term Plan

1. Keep the current CurvyTron native LightZero loop as the baseline trainer and
   profile surface.
2. Build a collect-only actor chunk function that loads a frozen checkpoint,
   collects complete two-seat visual trajectories, and writes compact replay
   chunks with explicit `policy_version`.
3. Build a learner step that reads N chunks, samples sequences, calls
   LightZero-compatible learner code, and publishes the next checkpoint.
4. Run N actor chunks in parallel on Modal. First target is coarse
   synchronous generations, not fully async services. This directly tests
   whether searched CurvyTron collection can scale beyond one `train_muzero`
   process without changing MuZero semantics.
5. Separately research whether MiniZero, EfficientZero, OpenSpiel C++ AlphaZero,
   or MCTX should become a stronger reference or replacement for the inner loop.

## Open Package Verdict

- LightZero: keep as current algorithm reference and MuZero machinery, but do
  not expect it to provide the full distributed CurvyTron system.
- EfficientZero: useful reference for Ray actor/reanalysis architecture and
  sample-efficiency ideas, but adapting it to simultaneous CurvyTron is work.
- muzero-general: useful educational reference for `SharedStorage`,
  `ReplayBuffer`, `SelfPlay`, and `Trainer` roles; not production-grade.
- OpenSpiel AlphaZero: useful architecture reference for actors/learner/replay
  and batched C++ inference; its game model is not a natural visual CurvyTron
  fit.
- MiniZero: likely worth deeper follow-up because it appears closest to a full
  open AlphaZero/MuZero distributed system.
- MCTX: worth deeper follow-up if we are willing to move the search core to JAX.
  It provides batched accelerator-friendly search, not the replay/actor system.

## Coach Handoff Summary

The optimizer correction is: current CurvyTron runs are useful smokes, but they
are not yet architecturally comparable to fast AlphaZero/MuZero. The next speed
lane should be a distributed searched self-play data factory: actor chunks,
explicit policy versions, replay chunks, learner merge, checkpoint publish, and
freshness metrics. In the current synchronous `train_muzero` loop there is no
actor-fleet staleness issue. In a future continuous actor/replay design,
freshness is a metric to log and control, not a reason to avoid parallel
self-play. Keep MuZero search on decisions; do not use policy-head-only
collection as a training replacement.
