# CurvyTron Optimizer System Map

Date: 2026-05-12

Status: optimizer working memory. This is a systems map, not a learning claim.

## Plain Goal

We need to make CurvyTron training faster without changing the MuZero-style
algorithm. The main question is not only "can one render be faster?" It is:
where does wall-clock time go across self-play, search, replay, learner, and
checkpoint handoff, and which parts can actually scale?

## Current Trusted Lane

The trusted proof/profile lane is:

```text
stock LightZero train_muzero
  + CurvyTron source_state_fixed_opponent env
  + frozen checkpoint opponent
```

This is trusted because it calls LightZero's real training loop. It proves
plumbing and gives honest timing. It is not yet true simultaneous current-policy
self-play.

The old custom `two-seat-selfplay` trainer is historical until its replay and
target semantics are proven. Do not use old failures from that path as evidence
against CurvyTron or LightZero.

## Full Loop Shape

The system we eventually want looks like this:

```text
checkpoint K
  -> self-play/search workers collect searched trajectories
  -> replay stores complete games or trajectory chunks
  -> learner samples replay and updates the network
  -> checkpoint K+1 is published
  -> eval/inspection/GIF jobs run off the hot path
```

The current stock LightZero lane compresses most of this into one trainer
process/container:

```text
collect -> push to replay -> sample -> learn -> checkpoint
```

That is good for correctness and profiling. It is not the same as the large
actor fleet used by AlphaZero/MuZero-style production systems.

## Current CPU / GPU Split

Known or likely CPU-side today:

- CurvyTron physics step.
- Source-state render and downsample.
- Visual stack update.
- Subprocess env worker coordination.
- Frozen opponent inference in subprocess workers, because CUDA in forked env
  workers is unsafe in the current setup.
- Replay bookkeeping, artifact metadata, checkpoint file movement, eval/GIF
  side jobs.

Known or likely GPU-side today:

- Live LightZero policy/model inference.
- Learner forward/backward.
- Search-time neural model calls when the LightZero policy is on CUDA.

Mixed or still needs a sharper timing split:

- MCTS tree bookkeeping: LightZero has C++ tree support, but the full loop still
  crosses Python, env workers, and model calls.
- Search batching: root batches help the GPU, but env/render/opponent CPU work
  can still starve it.

## What The Profiles Currently Suggest

Short bad-policy profiles are not mainly render-bound. Episodes end quickly, so
reset/setup, collector width, MCTS, and learner overhead matter more.

Long no-death profiles make rendering obvious. With `browser_lines`, render and
visual stack work dominate worker CPU time. `body_circles_fast` is about a 2x
end-to-end speed lens in long no-death profiles, but it is not automatically the
trusted final visual surface.

The frozen opponent is also visible in worker timing. Keeping it CPU-safe in
subprocess envs avoids CUDA fork failures, but it means not everything is on the
GPU.

Bigger GPUs alone are not expected to help much while CPU env/render/opponent
time is the limiter. Bigger batches or more collector workers can help only if
they keep the GPU busy and do not simply move the bottleneck to CPU workers,
replay merge, or learner input building.

## Amdahl Read

If long-run wall time is mostly render/observation, optimizing MCTS will not
move total time much.

If search becomes dominant after render is fixed, then batching roots and using
faster search backends becomes high leverage.

If replay merge or learner input building becomes dominant after actor fanout,
then adding more self-play workers will only make a bigger queue.

So every architecture experiment needs a component breakdown, not just
steps/sec.

## Why Massive Self-Play Is Not Automatically Free

In a synchronous generation, it is valid to freeze checkpoint `K`, launch many
self-play/search jobs, collect data, then train `K+1`. There is no hidden
off-policy problem inside that frozen batch.

The caveat is waste. If we collect far more checkpoint-`K` data than the learner
can use before a better `K+1` should be produced, we burn time and storage on
old data. More actors are useful only when searched data is the bottleneck and
the learner can digest it.

The first real scale test should therefore be a ladder, not a giant leap:

```text
N = 1, 2, 4, 8 collect-only jobs from checkpoint K
  -> write searched chunks with checkpoint id and schema
  -> merge/import
  -> run learner work
  -> compare throughput and Coach-owned quality per wall-clock
```

Measure: games/sec, decisions/sec, simulations/sec, chunk bytes/sec, write time,
merge time, learner update time, checkpoint publish time, and data age.

## Framework Lessons So Far

LightZero is still the best near-term path because it is already wired to our
Modal and CurvyTron work, and the stock `train_muzero` lane has passed canaries.
But stock LightZero does not hand us a production CurvyTron actor fleet.

EfficientZero is useful as a Ray architecture reference. It separates self-play,
shared weights, replay, reanalysis/batch builders, and the learner. Its code is
Atari-shaped and old, so copying it directly is risky.

MiniZero is a serious full-system Zero framework candidate. It needs a focused
read before any migration claim.

MCTX/JAX is a serious future search primitive. It could make batched search much
faster, but it is not a replay/learner/checkpoint system. A real MCTX path means
owning a JAX model/learner or maintaining a shadow model.

OpenSpiel AlphaZero is a clean architecture reference, especially for actor,
learner, replay, evaluator, and batched inference roles. It is not a natural
visual CurvyTron environment fit.

After the framework read, the practical verdict is unchanged: do not migrate
now. Copy the system shape, not the whole framework. LightZero remains the
trusted control/proof path; EfficientZero and MiniZero are references for how to
split actors/replay/learner; MCTX is a search benchmark candidate.

## Best Next Experiments

1. Stock LightZero iteration anatomy.
   Run one trusted stock frozen-opponent profile with worker telemetry and
   enough warm-up. Report collect, search, render/obs, opponent, replay/sample,
   learner, checkpoint, and artifact timing.

2. Collector width ladder.
   Run `collector_env_num` / worker width sweep with the same checkpoint,
   render mode, simulations, and death setting. Goal: find where CPU/process
   scaling flattens.

3. Coarse synchronous fanout prototype.
   Start with collect-only jobs from one checkpoint. Do not build a big service
   yet. First prove that searched chunks can be produced, written, merged, and
   consumed without changing MuZero semantics.

4. MCTX visual-root benchmark.
   Benchmark `float32[B,2,4,64,64] -> tiny CNN -> mctx.gumbel_muzero_policy`.
   This tests whether a GPU-native batched search path is worth future work.

5. MiniZero architecture spike.
   Read its worker/server/optimizer/storage path and decide whether it is a
   migration candidate or just a reference.

## Optimizer Boundary

Optimizer owns setup, profiles, bottleneck maps, throughput experiments, and
speed recommendations.

Coach owns learning claims, eval gates, checkpoint quality, and whether a run is
worth scaling.

Environment Reconstruction owns source-fidelity claims. Optimizer can test
faster render surfaces, but cannot promote them as faithful without review.
