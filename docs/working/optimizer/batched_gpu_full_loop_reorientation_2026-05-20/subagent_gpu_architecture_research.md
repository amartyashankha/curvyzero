# Subagent GPU Architecture Research

Date: 2026-05-20

Status: research note only. Do not touch live runs or trainer defaults.

## Question

Are we missing a higher-level GPU env/render/self-play architecture pattern for
CurvyTron, beyond the current batched GPU observation renderer work?

Short answer: yes, but the missing pattern is probably not another small render
kernel. The outside systems that go much faster usually do one of two things:

1. keep env state, observation, reward/reset, policy/search, and learning
   tensors resident on accelerator; or
2. keep the env on CPU, but use a serious actor/inference/replay architecture
   with shared-memory or compact-buffer movement and batched model/search work.

CurvyTron is currently in the middle: it can form a large batched GPU
observation surface, but the stock LightZero loop still wants scalar
Python/NumPy timesteps at the boundary. That scalar boundary is now the main
architecture question.

## Local Context To Preserve

Current local source of truth:

- `README.md`: the safe production path is still stock LightZero with
  `source_state_fixed_opponent` and CPU policy observations. The promising GPU
  path is the profile-only vector facade:
  `VectorMultiplayerEnv[B,2] -> compact state -> batched GPU render ->
  [B,2,4,64,64] -> scalar LightZero timesteps`.
- `world_model.md`: the latest Amdahl read says the current batched GPU manager
  is not end-to-end GPU RL. It still returns scalar LightZero timesteps.
- `next_experiment_grid.md`: C512 real batched GPU post-patch reached about
  `1439.84 steps/s`; the C512 zero-observation ceiling is about
  `1805.22 steps/s`, so perfecting only the current observation path is bounded
  to about `1.25x` at that width.
- `host_overhead_map.md`: after direct render/copy cleanup, payload shape,
  stack/pack layout, subprocess/process effects, search, and RND are the next
  likely walls.
- `subagent_rnd_batched_gate_critique.md` and related RND docs: meter-only RND
  costs roughly `10-12%` in realistic cadence rows, and aggressive RND cadence
  can dominate. Keep RND as its own axis.

Non-negotiable gates before promotion:

- normal-death autoreset and terminal `final_observation` must be correct;
- row/player perspective and stack order must be correct for both seats;
- no hidden CPU fallback in a "GPU" row;
- RND latest-frame extraction must not confuse player axis with unroll axis;
- full-loop rows must call stock `train_muzero` and report matched env steps,
  MCTS roots, learner calls, replay samples, and RND calls.

## External Patterns

### End-to-end GPU env stacks

Isaac Gym is the cleanest example of the pattern: simulation, observation,
reward, action buffers, and policy tensors stay on GPU, avoiding CPU/GPU
transfer. NVIDIA's writeup says observation/reward calculations run on GPU and
"observation, reward, and action buffers can stay on the GPU" through learning:
https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/

Brax is the same architectural answer in JAX form. The Brax paper describes
environment and learning code compiling together so "the learning algorithm and
the environment processing" occur on the same device:
https://arxiv.org/abs/2106.13281

PixelBrax extends that lesson to pixel observations: Brax plus a pure JAX
renderer runs end-to-end on GPU, renders thousands of parallel envs, and reports
two orders of magnitude over CPU-rendered pixel benchmarks:
https://arxiv.org/abs/2502.00021

CuLE is the Atari version. Its key lesson is not "GPU render is nice"; it is
"run thousands of games and render frames directly on GPU to avoid CPU/GPU
bandwidth limits." NeurIPS reports up to 155M frames/hour on one GPU:
https://papers.nips.cc/paper/2020/hash/e4d78a6b4d93e1d79241f7b282fa3413-Abstract.html

What applies to CurvyTron:

- A true end-to-end JAX/Torch env would be the clean architecture for a large
  win: compact state, physics step, active trail update, direct gray64 render,
  reward, reset, action mask, and maybe policy inference all device-resident.
- This path likely requires owning more than a renderer. It pushes toward a
  repo-owned actor/search/replay/learner stack or a major LightZero boundary
  rewrite.

What does not directly apply:

- CurvyTron is source-faithful, simultaneous-action, two-seat, and currently
  LightZero/PyTorch-shaped. Isaac/Brax-style speed assumes the env API itself is
  accelerator-native, not a GPU renderer bolted onto scalar Python envs.

### High-throughput CPU/vector env stacks

EnvPool is the opposite useful pattern: keep simulation on CPU/C++, but make it
batched, asynchronous, and low-overhead. The repo describes a C++ batched env
pool with sync/async APIs, multiplayer support, batched RGB output, and about
1M Atari FPS / 3M MuJoCo steps/s on a large CPU box:
https://github.com/sail-sg/envpool

Sample Factory separates rollout workers, inference workers, batcher, and
learner. Its architecture docs explicitly avoid serializing observations across
processes and instead pass shared-memory buffer IDs:
https://www.samplefactory.dev/06-architecture/overview/

Sample Factory also notes that for GPU-vectorized environments like Isaac Gym,
async mode matters less because work is on the same device; sync mode can be
better for sample efficiency:
https://www.samplefactory.dev/07-advanced-topics/sync-async/

PufferLib's vectorization docs echo the same CPU-side lesson: multiple envs per
worker, one shared memory buffer, shared flags instead of pipe/queue chatter,
and zero-copy batching:
https://pufferai.github.io/dev/build/html/rst/landing.html

What applies to CurvyTron:

- If stock LightZero keeps scalar timestep boundaries, preserve CPU actor
  parallelism and reduce movement: compact state buffers, shared memory or
  contiguous buffers, and a batched render service can beat a one-process
  vector facade once C512 saturates.
- The current subprocess CPU-oracle wins are consistent with this literature:
  process/actor parallelism is a real lever, not just a workaround.

What does not directly apply:

- EnvPool/Puffer-style vectorization is easiest when the env is a native C/C++
  binding or a simple Gym contract. CurvyTron's two-player simultaneous action,
  player-perspective observation, final-observation, and RND hooks need explicit
  contract guards.

### Zero-style self-play systems

OpenSpiel's AlphaZero docs are a compact reference split: actors generate
self-play data with MCTS, a learner updates the network, evaluators gauge
progress, and checkpoints/logs are separate. The docs also contrast the Python
path with the C++ path: the C++ implementation uses threads, a shared cache,
batched inference, and GPU inference/training:
https://openspiel.readthedocs.io/en/latest/alpha_zero.html

SEED RL is not MuZero, but it is an important actor/inference architecture:
centralized inference plus an optimized communication layer, with millions of
frames/sec and lower experiment cost:
https://arxiv.org/abs/1910.06591

Podracer argues for accelerator-friendly RL architecture design at scale, with
the useful meta-lesson that RL systems need different topology than ordinary
supervised learning pipelines:
https://arxiv.org/abs/2104.06272

LightZero itself is a benchmark/system decomposition for MCTS/MuZero-style
agents across many domains, but CurvyTron is currently using its stock scalar
collection assumptions:
https://arxiv.org/abs/2310.08348

MCTX is the relevant search-kernel reference if we ever move search into JAX.
Its README says MCTS algorithms are JIT-compatible and operate on batches of
inputs in parallel:
https://github.com/google-deepmind/mctx

What applies to CurvyTron:

- Separate hot roles: env actors, batched observation/render, batched
  policy/search inference, replay/chunk storage, learner, checkpoint/eval.
- Batch across roots, not just frames. For CurvyTron that probably means
  flattening active seat rows as `[B * P]` policy/search roots, then scattering
  actions back to simultaneous `[B, P]` env steps.
- Keep checkpoint id/model hash in chunks so actor freshness is measurable.

What does not directly apply:

- MCTX is a search primitive, not a training system. A real MCTX path implies
  JAX model ownership or shadow conversion from PyTorch, plus new replay and
  checkpoint semantics.
- Zero-style actor fanout does not remove render cost; it only stops render,
  search, learner, and evaluation from blocking each other in one loop.

## Architecture Read For CurvyTron

The current C512 gap says renderer-only headroom is now finite:

```text
real batched GPU C512:          ~1439.84 steps/s
zero-observation C512 ceiling:  ~1805.22 steps/s
renderer/observation-only max:  ~1.25x
```

So the next big win is unlikely to be "make direct_gray64 2x faster" unless it
also removes stack/pack/scalar materialization. The architectural question is:

```text
Can we keep large batches alive across env step -> observation -> policy/search
without collapsing back into scalar LightZero env workers?
```

Three plausible routes:

1. **Stock-compatible vector facade:** keep one batched manager in-process,
   return scalar LightZero-shaped timesteps only at the outer boundary, and
   squeeze stack/pack/payload overhead until real rows approach the
   zero-observation ceiling.
2. **Hybrid actor plus render service:** subprocess actors step compact
   CurvyTron state and submit render requests to one central batched GPU
   renderer. This preserves CPU actor parallelism while still forming GPU
   render batches.
3. **Device-resident env/search rewrite:** rewrite CurvyTron state transition,
   observation, reward/reset, and search root construction in JAX/Torch so the
   loop becomes Isaac/Brax/CuLE-like. This is the biggest potential win and the
   biggest ownership change.

Practical implication: continue the current batched manager gates, but set a
stop condition. If C512 real rows stabilize within roughly `10-15%` of
zero-observation, stop render work and move to search/manager/RND. If they stay
farther away, focus on stack/pack/payload and the hybrid render-service sketch.

## Concrete Next Experiments

1. **Semantic gate bundle for the batched manager.**
   Add/verify tests for normal-death autoreset, partial row reset,
   terminal `final_observation`, row/player order, stack order, registry
   restore, and RND latest-frame extraction. Promotion should fail closed on
   exact backend identity and no hidden CPU fallback.

2. **C512/C768 real-vs-zero repeat with matched counters.**
   Repeat no-RND sim2 rows for real batched GPU and zero-observation at C512
   and C768. Require matched env steps, MCTS roots, learner calls, replay
   samples, and warmup. This decides whether C512 is already near the local
   observation ceiling or whether stack/pack/render still deserves work.

3. **Search/root batching sweep.**
   Run H100 C256/C512/C768 by sim2/sim4/sim8 with CPU-oracle and batched-GPU
   anchors. Record policy/search timer, average inference/root batch size,
   manager wait, and GPU utilization. Goal: separate MCTS/search from
   observation after direct render.

4. **Hybrid render-service prototype, profile-only.**
   Build the smallest non-training prototype where N CPU actor processes step
   compact source state and a single GPU renderer consumes a queue of compact
   states, returns row-major `[B,2,4,64,64]` frames through shared memory or
   pinned host buffers, and reports queue wait/render/pack/copy separately.
   This should not call live training or change defaults.

5. **RND matched cadence pair.**
   Compare no-RND versus `rnd_meter_v0` at C512 with cadence `10` and `100`,
   using the same observation backend and matched workload counters. Report
   RND collect/train/estimate time and source counters separately from render.

## Follow-up: smallest useful prototype

The new C768 pair changes the practical prototype target. C768 real render was
basically flat versus C512 (`1420.45` vs `1439.84 steps/s`), and C768
zero-observation was slower than real render (`1191.71 steps/s`). Treat that as
a topology/scheduling warning, not a renderer result. To plausibly move beyond
the stable `~1.8k steps/s` zero-observation ceiling, the smallest useful
prototype is the hybrid actor-parallel env-step plus central batched observation
profile, not a wider one-process manager row.

Touch only profile-only code:

- Add a new script next to `scripts/profile_batched_observation_mock_collector.py`,
  e.g. `scripts/profile_hybrid_batched_observation_manager.py`.
- Add a new module next to
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py`,
  e.g. `src/curvyzero/training/source_state_hybrid_observation_profile.py`.
- Reuse `VectorMultiplayerEnv` for actor-owned CPU source-state stepping.
- Reuse `SourceStateBatchedRenderRequest`,
  `SourceStateBatchedObservationRenderer`, and the existing direct/zero
  observation seams rather than adding a trainer backend.
- Add focused tests beside `tests/test_source_state_batched_observation_mock_collector.py`
  for action routing, row/player ids, terminal handoff, and "no CUDA/JAX in
  actor subprocess" metadata.

Prototype shape:

```text
parent harness
  -> N actor subprocesses step compact CurvyTron rows only
  -> parent gathers compact state/reward/done packets
  -> parent batches zero-observation first, then direct_gray64 if zero passes
  -> parent materializes scalar LightZero-shaped ready_obs/timesteps
```

Measure, per parent step and per physical row:

- actor env-step time and actor idle/wait time;
- actor-to-parent serialize/send/receive time;
- compact payload bytes and pickle time;
- parent gather/merge time;
- central observation time split into zero, pack, render, readback, stack;
- scalar timestep/`ready_obs` materialization time;
- missing/extra action rejection cost is not important, but correctness is;
- total throughput versus one-process zero observation and current C512 real.

Must not touch:

- stock `train_muzero`, LightZero config defaults, Coach launch defaults, live
  runs, checkpoint/tournament/eval/GIF paths, or positive-RND reward plumbing;
- scalar `jax_gpu` trainer backend promotion;
- DI-engine env-manager registry until the profile harness beats the current
  ceiling;
- rendered `[4,64,64]` frame transfer from actor subprocesses to parent.

Success criteria:

- `calls_train_muzero=false`, `profile_only=true`, and output metadata names a
  new backend such as `hybrid_zero_observation_profile`.
- Zero-observation hybrid at comparable total width exceeds the one-process
  C512 zero ceiling by at least `20%` after warmup, or clearly trends upward
  with actor count while one-process C768 does not.
- Compact actor payload is at least `10x` smaller than shipping both players'
  rendered stack frames.
- Env-step counts, row/player ids, action masks, rewards, done flags, and reset
  generations match a one-process oracle on a small deterministic seed.
- Normal-death terminal `final_observation` is rendered from terminal compact
  state before reset in a small semantic test, even if first speed rows run
  no-death.

Failure criteria:

- Hybrid zero-observation cannot match one-process zero-observation at C512, or
  actor IPC/gather dominates before real render is enabled.
- Throughput only improves by increasing total env rows while per-row latency
  and policy/search-facing batch shape get worse.
- The first real-render row is attempted before zero-observation proves the
  actor topology itself can beat the current ceiling.
- Implementing the profile requires trainer/default changes. That means the
  prototype is too large.

A device-resident toy loop is the backup falsifier, not the first prototype:
one JAX/Torch function with fake actions, compact state tensors, no LightZero,
zero/direct observation, and no replay. It is useful only to estimate the upper
bound of "env step + observation without scalar Python timesteps"; it should
not be used as evidence for stock-training speed.

## Do Not Do Yet

- Do not promote `direct_gray64` or the batched GPU manager to trainer defaults
  before normal-death and RND gates pass.
- Do not spend a full phase on renderer-only micro-optimizations unless a
  matched real-vs-zero row shows observation remains the dominant gap.
- Do not start a full JAX/MCTX rewrite as a speed patch. Treat it as a separate
  architecture lane after stock-compatible batching and hybrid render service
  have been measured.
- Do not mix positive-RND learning claims with RND meter throughput claims.
