# External GPU RL Full-Loop Patterns

Date: 2026-05-21

Scope: research memo only. No production code, trainer defaults, tournament
paths, or live runs changed.

## Short Read

The established fast RL/game systems do not treat "GPU rendering" as a separate
island. They make the batch the unit of work and keep as much of the loop as
possible in one resident tensor world:

```text
batched state -> batched step/render/obs/reward -> batched policy/search
-> batched replay/learner tensors -> coarse logging/checkpoint readback
```

When they cannot keep the environment on the accelerator, they still avoid
scalar inference by using central batched inference/search workers and chunky
replay handoff. The recurring failure mode for CurvyTron + stock LightZero is
therefore not "the GPU renderer is too slow"; it is "the fast batched renderer
is forced back through scalar `env.step`, per-row NumPy payloads, and per-root
search calls before the learner sees it."

## System Patterns

### CuLE / GPU Atari

Primary sources: [CuLE paper](https://arxiv.org/abs/1907.08467),
[NVLabs CuLE repo](https://github.com/NVlabs/cule).

What stays batched/resident:

- Thousands of Atari emulators run in parallel on CUDA.
- Frame rendering happens directly on GPU.
- Training examples are consumed in explicit minibatch/update schedules; the
  repo example uses `--num-ales 1200`, `--num-steps 20`, and minibatches so one
  simulated CuLE step can feed one GPU update.

Where CPU sync is avoided:

- The paper's central claim is avoiding the CPU emulator -> GPU network
  bandwidth bottleneck by rendering on GPU and batching enough emulator states
  to saturate the accelerator.
- CPU is orchestration/build/eval plumbing, not a per-frame pixel owner.

CurvyTron tradeoff:

- This is the closest analogy for a 2D line/sprite game: tiny rules, cheap
  rendering, huge payoff only if many rows are stepped/rendered at once.
- CuLE also shows the uncomfortable part: the environment itself is ported, not
  just the observation function. A CurvyTron "CuLE-like" win would require
  device-owned positions/headings/trails/resets/rewards, not scalar CPU source
  rows feeding a GPU render call one row at a time.
- For LightZero, CuLE's pattern only survives if replay rows are produced in
  chunks and MCTS roots are batched. A renderer-only adapter that returns
  `[4,64,64]` NumPy frames per `BaseEnvTimestep` is the anti-pattern.

### Isaac Gym / Isaac Lab

Primary sources: [Isaac Gym paper](https://arxiv.org/abs/2108.10470),
[Isaac Lab task workflow docs](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html).

What stays batched/resident:

- Isaac Gym keeps physics simulation and policy training on GPU, passing data
  from physics buffers to PyTorch tensors without CPU copies.
- Observation, reward, reset, and action buffers are task tensors over many
  environments.
- Isaac Lab preserves the same shape at the task level. Its direct workflow is
  one environment class that owns observations, actions, rewards, and resets;
  docs call out direct implementations as better for performance and for large
  chunks of optimized PyTorch JIT/Warp logic.

Where CPU sync is avoided:

- The expensive loop is GPU sim buffer -> PyTorch tensor -> policy action tensor
  -> GPU sim. The CPU configures assets/tasks, logs, and launches training.
- The important design detail is that rewards and resets also stay tensorized.
  It is not just "physics on GPU"; rollout bookkeeping avoids per-env Python.

CurvyTron tradeoff:

- CurvyTron has simpler dynamics than robotics. That helps: fixed-size
  `[B,2]` or `[B,2,S]` line/trail state is much easier than contact physics.
- But LightZero's stock collector wants Python timestep objects and NumPy ->
  Torch transfers. Isaac-style residency would mean a new internal collector
  boundary: device obs/action/reward/done stacks for `T` steps, materialized to
  CPU only per chunk or for diagnostics.
- If we stay stock-LightZero compatible, borrow the direct-workflow idea:
  implement a large internal vector facade and keep scalar timesteps only at
  the outside edge.

### Brax / MJX / PixelBrax

Primary sources: [Brax paper](https://arxiv.org/abs/2106.13281),
[Brax repo](https://github.com/google/brax),
[MJX docs](https://mujoco.readthedocs.io/en/latest/mjx.html),
[PixelBrax paper](https://arxiv.org/abs/2502.00021).

What stays batched/resident:

- Brax is JAX-native and compiles RL algorithms alongside environments so env
  processing and learning occur on the same accelerator.
- MJX is explicit about its sweet spot: big batches of identical scenes on
  SIMD/XLA hardware. It may be slower than CPU MuJoCo for a single scene, but
  works best with thousands or tens of thousands of scenes.
- PixelBrax adds a pure JAX renderer so pixel observations over thousands of
  parallel environments stay end-to-end on GPU.

Where CPU sync is avoided:

- JAX `jit`/`vmap`/`scan` turn per-step Python into compiled loops. State,
  stochastic keys, observations, and learner batches remain JAX arrays until
  checkpoint/log readback.
- PixelBrax specifically avoids CPU rendering, which is the same broad problem
  as CurvyTron's visual observation path.

CurvyTron tradeoff:

- The lesson is not "rewrite in JAX tomorrow"; it is "single-scene GPU is often
  bad, large-batch GPU is the product." CurvyTron should use fixed static
  shapes, padded rows, and compiled batch loops if we test a JAX/CUDA path.
- Dynamic trail lists and source-fidelity details can fight accelerator
  branchlessness. Use fixed trail capacity / occupancy tensors for a spike.
- PyTorch LightZero creates an impedance mismatch. MCTX/JAX can own search, or
  Torch LightZero can own training, but bouncing every step between JAX env and
  Torch MCTS would recreate the host-sync problem.

### Pgx + MCTX

Primary sources: [Pgx paper](https://arxiv.org/abs/2303.17503),
[Pgx repo](https://github.com/sotetsuk/pgx),
[MCTX repo](https://github.com/google-deepmind/mctx).

What stays batched/resident:

- Pgx game states are JAX-native and JIT-able; the README's example uses
  `jax.jit(jax.vmap(env.init))` and `jax.jit(jax.vmap(env.step))` with
  `batch_size = 1024`.
- Pgx exposes observations, legal-action masks, current player, rewards, and
  terminal flags as batched state fields.
- MCTX search is JAX-native, JIT-compatible, and defined over batches of root
  inputs in parallel. It provides MuZero and Gumbel MuZero policy helpers.

Where CPU sync is avoided:

- Env step, legal action mask, current-player logic, and search can all be
  compiled into array programs. Host readback is only needed for logging,
  checkpointing, or external visualization.
- Pgx v2's explicit PRNG key for stochastic envs is a useful reproducibility
  pattern: randomness should be an input tensor, not hidden Python state.

CurvyTron tradeoff:

- Pgx is board-game/sequential-turn shaped. CurvyTron is simultaneous-action,
  continuous-ish line movement with visual history. We should borrow the array
  contract, not the game model.
- MCTX maps well to "batch roots first": flatten active seats to `[B*2]`, run
  representation/prediction/search in one call, then scatter actions back to
  simultaneous `[B,2]` env steps.
- This is a bigger departure from stock LightZero. It is most useful as a
  search/inference benchmark or alternate lane if LightZero's collector keeps
  collapsing batches.

### AlphaZero / MuZero Distributed Self-Play

Primary sources: [MuZero paper](https://arxiv.org/abs/1911.08265),
[AlphaZero paper](https://arxiv.org/abs/1712.01815),
[OpenSpiel AlphaZero docs](https://openspiel.readthedocs.io/en/latest/alpha_zero.html),
[SEED RL paper](https://arxiv.org/abs/1910.06591),
[SEED RL repo](https://github.com/google-research/seed_rl).

What stays batched/resident:

- AlphaZero/MuZero separate self-play actors, replay, learner, and evaluation.
  The algorithmic unit stored in replay is a trajectory/game segment with
  observations, actions, rewards, search policies/visit counts, values, and
  metadata, not isolated scalar rows.
- OpenSpiel's docs make the systems lesson unusually explicit: the Python
  AlphaZero path lacks inference batching and is CPU-only; the C++ path uses
  threads, a shared cache, batched inference, and GPU inference/training.
- SEED RL generalizes the "many actors, one accelerator" shape: actors run
  environments, while training and inference are centralized on the learner.

Where CPU sync is avoided:

- In distributed CPU-env systems, the env can remain CPU, but neural inference
  does not happen per actor in tiny calls. Actors send requests to a central
  inference/learner service, which batches accelerator work.
- Replay is chunky and versioned enough that learner/actor asynchrony can be
  measured and bounded.

CurvyTron tradeoff:

- This is the most practical bridge for LightZero if full device-resident envs
  are too invasive: keep CPU/source-state actors, but centralize batched
  render/inference/search and write replay chunks with policy version, row/seat,
  observation backend, and final-observation metadata.
- For a small 2D game, remote/distributed plumbing can be overkill. Try the
  architecture inside one process/container first: many env rows submit to one
  batched GPU worker.
- Staleness matters in self-play. Central batching improves throughput, but
  replay must carry policy version and age or the learner will train on an
  untracked mixture of policies.

### LightZero Current Fit

Primary sources: [LightZero repo](https://github.com/opendilab/LightZero),
[LightZero worker docs](https://opendilab.github.io/LightZero/api_doc/worker/index.html),
[MuZeroCollector source docs](https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html).

Relevant observed shape:

- LightZero is a PyTorch MCTS+RL toolkit with MuZero, EfficientZero, Gumbel
  MuZero, Stochastic MuZero, and related algorithms.
- Its worker docs describe `MuZeroCollector` as episode-based and serial.
- The collector source gathers `ready_obs`, converts stacked observations via
  NumPy preparation and `torch.from_numpy(...).to(device)`, calls
  `policy.forward`, then unpacks per-env outputs and steps the env manager.

CurvyTron read:

- LightZero can batch "ready envs" for policy forward, which is good.
- But its public collector boundary is still CPU/Python env-manager shaped. A
  fast CurvyTron GPU surface will only matter if we keep a large internal batch
  alive before the collector sees row objects, or if we replace/extend the
  collector with a resident chunk path.
- The least invasive test is not a production backend flag. It is a measured
  mock/full-loop profile that reports where the batch collapses: env step,
  stack update, host-device copy, policy/search, replay packing, learner, RND.

## Borrowable Implementation Patterns

| Pattern | Borrow for CurvyTron | Why it matters |
| --- | --- | --- |
| Fixed batch axes | Treat rows, seats, MCTS roots, and rollout time as explicit axes: `[B,2,T,C,H,W]` or flattened `[B*2]` roots. | Prevents accidental scalar loops around fast kernels. |
| Resident rollout chunks | Keep obs/action/reward/done/final_obs on device for `T` steps, then materialize one replay chunk. | Avoids per-step Python object construction and tiny CPU/GPU transfers. |
| Central batched inference/search | Many env rows request inference from one GPU worker; search operates over a root batch. | Useful if LightZero cannot own a GPU-resident env but can consume batched observations. |
| Explicit PRNG tensors | Pass reset/randomness keys or seeds as batch tensors. | Reproducibility, replayability, and no hidden host state. |
| Chunk metadata | Store policy version, backend, row id, seat id, stack order, final-observation-before-reset flag, and replay schema. | Makes async/staleness and observation-surface drift inspectable. |
| Measure sync count | Count `.cpu()`, `.numpy()`, `device_get`, `torch.cuda.synchronize`, tiny H2D/D2H copies, and per-row object materialization. | These are usually where the theoretical GPU win leaks out. |

## Three Practical Recommendations

1. **Promote a resident-chunk profile before any production backend flag.**
   Build/measure `B x T` chunks where CurvyTron state, direct `64x64` obs,
   actions, rewards, dones, and final observations stay batched until chunk
   close. Compare against the stock LightZero path with explicit sync/copy and
   row-materialization counters.

2. **Batch MCTS roots/inference as a first-class contract.**
   Flatten acting seats to `[B*2]`, run policy/value/search in one batch, then
   scatter actions to simultaneous `[B,2]`. Even if the first implementation is
   Torch + LightZero-compatible, measure scalar per-root calls versus batched
   root calls; this is where MuZero-style systems spend real wall time.

3. **Keep one-container central batching as the fallback architecture.**
   If stock LightZero collector APIs block device residency, prototype a local
   central render/inference/search worker fed by CPU vector actors through
   chunky shared buffers. Add policy-version and replay-age metadata from day
   one; defer distributed/Modal boundaries until local batching proves the need.

## Three Traps To Avoid

1. **Renderer-only victory laps.**
   A fast isolated GPU renderer is not a full-loop win if frames come back to
   CPU every step or LightZero immediately scalarizes replay rows.

2. **Single-scene GPU intuition.**
   MJX/Brax/CuLE-style systems win at thousands of parallel states. A tiny
   CurvyTron GPU call per player/frame can be slower than a CPU vector path
   because launch, sync, and transfer overhead dominate.

3. **Unversioned async replay.**
   Central inference and actor pools can increase throughput while quietly
   training on stale or mixed-policy data. Replay chunks need policy version,
   observation backend/schema, seat perspective, final-observation timing, and
   chunk age metrics before they are trusted.

## Source Index

- CuLE paper: https://arxiv.org/abs/1907.08467
- CuLE repo: https://github.com/NVlabs/cule
- Isaac Gym paper: https://arxiv.org/abs/2108.10470
- Isaac Lab task workflow docs: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html
- Brax paper: https://arxiv.org/abs/2106.13281
- Brax repo: https://github.com/google/brax
- MJX docs: https://mujoco.readthedocs.io/en/latest/mjx.html
- PixelBrax paper: https://arxiv.org/abs/2502.00021
- Pgx paper: https://arxiv.org/abs/2303.17503
- Pgx repo: https://github.com/sotetsuk/pgx
- MCTX repo: https://github.com/google-deepmind/mctx
- AlphaZero paper: https://arxiv.org/abs/1712.01815
- MuZero paper: https://arxiv.org/abs/1911.08265
- OpenSpiel AlphaZero docs: https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- SEED RL paper: https://arxiv.org/abs/1910.06591
- SEED RL repo: https://github.com/google-research/seed_rl
- LightZero repo: https://github.com/opendilab/LightZero
- LightZero worker docs: https://opendilab.github.io/LightZero/api_doc/worker/index.html
- LightZero MuZeroCollector source docs: https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html
