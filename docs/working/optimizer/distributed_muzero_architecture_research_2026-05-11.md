# Distributed MuZero Architecture Research

Date: 2026-05-11

Status: historical optimizer correction note. Some launcher facts below were
true on 2026-05-11 and are now stale. For the current optimizer architecture
map, start with:
`docs/working/optimizer/architecture_reexploration_2026-05-12/README.md`.

Current launcher truth, 2026-05-12: the trusted proof/profile lane is stock
LightZero `train_muzero` with `source_state_fixed_opponent` and a frozen
checkpoint opponent. The old custom `two-seat-selfplay` path is historical
until native replay and target semantics are proven.

Related framework comparison and simultaneous-action modeling note:
`docs/working/optimizer/framework_reassessment_2026-05-11.md`.

Related component checklist for the full training loop:
`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md`.

## Plain Verdict

The current native CurvyTron LightZero stock loop is useful as
fixed/frozen-opponent control/profile evidence. It is still not structurally
close to the training systems that made AlphaZero/MuZero fast.

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

Current fact: stock `train_muzero` is one synchronous collector/learner loop.
The next scale experiment should be coarse synchronous fanout: run multiple
frozen-checkpoint collection jobs, merge their searched trajectory chunks, then
train the next checkpoint.

```text
checkpoint K
  -> N collect-only actors/search workers use checkpoint K
  -> searched trajectory chunks with checkpoint id, schema, seed, and search settings
  -> merge/import chunks
  -> learner samples sequences and updates to K+1
  -> checkpoint publisher writes K+1
```

In the coarse fanout experiment, all actors use the same frozen checkpoint, so
there is no live actor/learner lag inside that batch. If we later build a
continuous actor/replay system, record checkpoint ids, sample age, and policy
version as normal run metadata. Massive self-play is useful when the learner
can turn searched trajectories into updates fast enough and the replay window
remains representative.

## Massive Self-Play Question

Running far more self-play at the same checkpoint is policy-valid in a
synchronous generation: freeze checkpoint `K`, launch many actors, collect
searched games, merge them, then train `K+1`. The danger is not off-policy
correctness. The danger is wasting wall-clock and storage on too much data from
one old policy before the learner has a chance to improve.

A million parallel games is only useful if these are true:

- actors/search workers are the bottleneck and scale nearly linearly;
- replay merge/write is not the new bottleneck;
- learner can consume enough samples before the policy should refresh;
- the next checkpoint is better per wall-clock after Coach's quality check,
  not just raw games/sec;
- chunks carry checkpoint id, env/reward/action/obs schema, search settings,
  seed, and completed-game counts.

The tradeoff is simple: bigger self-play batches give more searched positions
per learner update, but delay the next policy update. That is fine when search
is the bottleneck and the learner can digest the batch. It is wasteful when the
batch is so large that actors keep generating checkpoint-`K` data after a
smaller batch would already have produced a better `K+1`. Diversity also
matters: many identical actors can produce a large but correlated replay slice.

So the first test should be small and explicit: `N={1,2,4,8}` collect-only
actors from one frozen checkpoint, same CurvyTron config, same MCTS sims, each
writing compact searched chunks. Measure games/sec, decisions/sec, MCTS
sims/sec, chunk bytes/sec, write/merge time, learner update time after merge,
and checkpoint-quality change per wall-clock. If N=8 scales cleanly and learner/merge are
small, then scale N aggressively. If learner/merge dominate, more actors alone
will not speed training.

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

The current native stock LightZero path should be treated as a control/profile
surface. The Coach baseline is the two-seat self-play launcher. A future
distributed actor loop may reuse the same env/config surface, but should be
designed as actor chunks plus replay/learner/checkpoint handoff rather than
endless single-process stock-loop polish.

## CurvyTron-Specific Issues

- CurvyTron is visual, non-ALE, simultaneous two-player self-play.
- Most AlphaZero systems assume alternating turns. A naive joint-action MCTS
  would multiply action space by player count. For `3` actions and `2` players,
  joint action is only `9`, so this may be tractable for 2-player CurvyTron, but
  the semantics must be explicit.
- The current native stock trainer is fixed-opponent single-ego, not true
  current-policy self-play and not the Coach canonical launcher. It is useful
  because it stays close to the working LightZero/Pong control pattern.
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

1. Keep the two-seat self-play launcher as the Coach baseline, and keep the
   current CurvyTron native stock LightZero loop as a control/profile surface.
2. Build a collect-only actor chunk function that loads one frozen checkpoint,
   collects searched source-state visual chunks on the current fixed-opponent
   path, and writes compact chunks with checkpoint id, schema, seed, and search
   settings.
3. Build a learner step that reads N chunks, samples sequences, calls
   LightZero-compatible learner code, and publishes the next checkpoint.
4. Run N actor chunks in parallel on Modal. First target is coarse
   synchronous generations, not an async service. This directly tests
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
lane should be coarse synchronous searched-trajectory fanout: actor chunks from
one frozen checkpoint, replay chunks, learner merge, checkpoint publish, and
checkpoint ids/sample age only if actors and learner are decoupled. In the
current synchronous `train_muzero` loop there is no actor-fleet staleness issue.
In a future continuous actor/replay design, freshness is a metric to log and
control, not a reason to avoid parallel self-play. Keep MuZero search on
decisions; do not use policy-head-only collection as a training replacement.
