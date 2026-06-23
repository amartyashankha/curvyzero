# Architecture Research Read

Date: 2026-05-21

Status: active optimizer working memory. This is not a Coach launch
recommendation.

## Plain Read

The current GPU observation work is real, but bounded.

In the latest profile shape, batched GPU observations beat the CPU-oracle
profile anchor by about `1.5x`. That is significant. It is not a `10x` result
because the profile is no longer dominated by only rendering. After the render
gets cheaper, stock LightZero collect/search, scalar timestep materialization,
stack movement, and manager work are visible.

The important distinction:

```text
faster render != fully batched accelerator RL
```

The systems that scale hard usually keep a large batch alive through many
stages:

```text
many actors/envs
-> batched observation/inference/search
-> replay/chunk storage
-> learner update
-> refreshed actor weights/checkpoints
```

## Sources Checked

- OpenSpiel AlphaZero docs: actors generate games with MCTS, learner consumes a
  FIFO replay buffer, C++ path uses threads, shared cache, batched inference,
  and GPU inference/training.
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- DeepMind Mctx: JAX-native AlphaZero/MuZero/Gumbel MuZero search; search
  operates on batches and supports JIT compilation for accelerators.
  <https://github.com/google-deepmind/mctx>
- NVIDIA CuLE: Atari envs and rendering directly on GPU; scales by running
  thousands of games and avoiding CPU/GPU frame traffic.
  <https://research.nvidia.com/publication/2020-12_accelerating-reinforcement-learning-through-gpu-atari-emulation>
- EnvPool: compiled/threaded CPU environment execution; addresses env
  execution as a common RL bottleneck.
  <https://arxiv.org/abs/2206.10558>
- Brax: JAX accelerator-resident environment plus learning on the same device.
  <https://arxiv.org/abs/2106.13281>
- LightZero paper: useful benchmark/toolkit for MCTS+RL, but not evidence that
  our current CurvyTron path is fully accelerator-resident.
  <https://arxiv.org/abs/2310.08348>

## What This Means For CurvyTron

Near-term production truth should stay on the stock LightZero lane until a
profile lane proves it can be promoted safely.

The next optimizer target is not another isolated render kernel tweak. The next
target is a batched boundary that preserves batch shape longer:

```text
VectorMultiplayerEnv[B,2]
-> compact state
-> batched observation stack
-> batched policy/search roots
-> replay rows/chunks
```

The clean research prototype is a profile-only "resident chunk" canary:

```text
compact CurvyTron batch for T steps
-> GPU observation stack
-> batched policy/value call over B*2 roots
-> synthetic or real search pressure
-> materialize LightZero-like rows only once per chunk
```

Success criteria:

- beats subprocess CPU-oracle C64/C256, not just scalar base;
- reports host/device bytes, sync count, root count, and batch fill latency;
- keeps row/player/action-mask/final-observation/RND-latest-frame gates green;
- does not change trainer, tournament, or live-run defaults.

## Current Working Hypothesis

The next `5x-10x` class path, if it exists, comes from one of these:

1. actor-parallel env stepping plus central batched observation/inference/search;
2. deeper MCTS/root batching, possibly learning from Mctx;
3. accelerator-resident CurvyTron toy env as a research lane;
4. compiled CPU env/manager path if Python scalarization stays dominant.

The current batched GPU observation path remains valuable because it is the
first proof that CurvyTron observations can be formed as a real batch. But the
product is the batch, not a single frame.

## 2026-05-21 Research And Profile Update

The newest resident canary makes the external research more concrete.

At B512/A16 with a persistent direct64 GPU renderer, `uint8` stack, and a
synthetic batched GPU consumer:

```text
H100 scalar edge off: ~13.8k scalar roots/sec
H100 scalar edge on:  ~6.5k scalar roots/sec
L4 scalar edge off:   ~9.0k scalar roots/sec
L4 scalar edge on:    ~4.2k scalar roots/sec
```

So the next high-value question is not "can the GPU draw CurvyTron?" It can.
The question is:

```text
Can the real self-play/search/replay loop keep this batch alive long enough to
benefit from it?
```

Research synthesis:

- CuLE, Brax/MJX, Isaac Gym, Pgx, and Mctx all win by keeping large batches in
  compiled/device-owned loops, not by making one frame faster.
- OpenSpiel's own docs call out the difference between a Python AlphaZero path
  with no inference batching and a C++ path with threads, shared cache, batched
  inference, and GPU inference/training.
- EnvPool is a reminder that CPU actor parallelism can still be the right
  answer when the environment is CPU-bound, but it does not solve scattered
  policy/search calls by itself.
- LightZero gives useful stock MuZero plumbing and some batched/C++ tree-search
  pieces. It does not automatically make CurvyTron env state, observation
  stacks, replay chunks, and search control device-resident.

Current plain recommendation:

```text
Keep production/Coach defaults on the trusted stock path.
Use profile-only work to find where the batch dies.
Do not promote a GPU observation default until a real policy/search/replay
boundary can use it without immediately scalarizing away the win.
```

Next toy experiments, in priority order:

1. Real LightZero/search-boundary audit: identify exactly where observations
   become scalar `BaseEnvTimestep` objects and where model/search already
   batches roots today.
2. Resident chunk plus real-consumer prototype: keep `uint8 [B,2,4,64,64]`
   through a model/search-shaped workload and materialize scalar rows only as a
   measured edge cost.
3. RND/normal-death guardrail rows: after the no-RND/no-death canary wins, add
   one RND meter row and one normal-death/autoreset semantic row. Do not mix
   those into the primary speed ratio.

## 2026-05-21 Resident Probe Update

The profile-only resident chunk probe is now implemented and smoke-tested. It
does not touch Coach training, but it gives a clearer shape for the next
architecture test:

```text
compact CurvyTron rows
-> renderer-backed uint8 [B,2,4,64,64]
-> GPU replay-like ring write/sample
-> GPU policy/search-shaped synthetic work
-> optional scalar LightZero materialization edge
```

Medium rows, B512/A16/sim8:

| compute | scalar off | scalar on | read |
| --- | ---: | ---: | --- |
| H100 | `10980.47` | `7620.76` | strong profile-only resident batch headroom; scalar edge keeps `~69%` |
| L4/T4 | `5839.67` | `4133.28` | useful, but much slower under GPU-shaped pressure; scalar edge keeps `~71%` |

Next research question:

```text
Can a real LightZero model/search/replay-shaped consumer use this batch before
we pay scalar timestep materialization?
```

If yes, resident batching is the next serious optimizer lane. If no, keep
production on the trusted stock path and focus on CPU actor parallelism plus
central batched inference/search.

Stock-boundary audit update:

- The trusted stock env path returns scalar NumPy observation dicts per env.
- LightZero collector rebuilds batches from scalar env-id `GameSegment`
  objects.
- MuZero collect forward then detaches latent roots/logits to CPU NumPy before
  MCTS in the audited LightZero version.

This means the next canary must treat policy/search as part of the boundary.
Only preserving the observation batch is not enough.

## 2026-05-21 Next Architecture Choices

The next profile-only canary should call the real public LightZero collect
policy boundary from the pre-scalar batch:

```text
uint8 [B,2,4,64,64]
-> flatten roots
-> torch policy tensor
-> MuZeroPolicy.collect_mode.forward(...)
-> label known CPU MCTS/tree boundary explicitly
```

This is not a full trainer and not a device-resident MCTS proof. It is the
smallest honest way to test whether the real LightZero policy/search consumer
can use the batch before scalar timesteps are materialized.

The JAX/MCTX lane is worth a tiny timeboxed scratch spike, but only after this
LightZero real-consumer canary plan is either implemented or blocked. It tests
the alternate premise: fixed-shape CurvyTron-like state plus batched JAX search
without stock LightZero's scalar/CPU MCTS boundary.

Current batch-size read:

- B512 is the best default profile shape.
- B1024 H100 did not improve throughput and made scalar materialization worse.
- Bigger batches should be revisited only when the consumer is real policy/search
  work rather than the current synthetic probe.
