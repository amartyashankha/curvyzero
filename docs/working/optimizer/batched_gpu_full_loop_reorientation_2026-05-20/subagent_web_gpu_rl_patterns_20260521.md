# Web note: GPU RL/rendering architecture patterns

Date: 2026-05-21

Scope: CuLE, JAXAtari, Isaac Gym, Brax/PixelBrax, Sample Factory, EnvPool, MCTX.

## Short read

The repeated pattern is: keep environment state, rendering/observation construction, policy inference, and learner batches in one batched tensor world for as long as possible. The fastest GPU-native systems are not just "render on GPU"; they remove the step-by-step Python/CPU boundary from the hot loop. CPU involvement becomes orchestration, logging, occasional checkpointing, and coarse batch handoff.

For CurvyTron LightZero, the useful mental model is:

```text
many source states/actions -> batched GPU env/render/obs tensors -> policy/search/learner tensors
```

not:

```text
one env step -> CPU state -> one render -> CPU/PIL/numpy -> torch copy -> policy
```

The main transfer budget should be paid only for coarse outputs: metrics, replay/checkpoints, and possibly compact source-state snapshots. Per-agent/per-step pixels crossing CPU/GPU is the thing these systems avoid.

## Patterns by system

### CuLE / GPU Atari

CuLE ports Atari emulation to CUDA, runs thousands of games simultaneously, and renders frames directly on GPU. The paper frames the bottleneck in classic RL as CPU environments feeding GPU DNNs through limited CPU-GPU bandwidth; CuLE removes that by keeping emulation/rendering on the accelerator and batching enough envs to saturate it. Reported throughput is 40M to 190M frames/hour on a single GPU.

Lesson for CurvyTron: a GPU render backend only pays off if it accepts and returns large batches. The "source state -> pixels" conversion should become a wide kernel/tensor operation, not a Python loop around many small GPU calls.

Source: [CuLE paper, NVIDIA Research PDF](https://research.nvidia.com/sites/default/files/pubs/2019-07_GPU-Accelerated-Atari-Emulation/CuLE.pdf)

### JAXAtari

JAXAtari reimplements Atari-like game logic natively in JAX, with JIT compilation and vectorized GPU execution. Its docs emphasize GPU-accelerated, object-centric environments, pixel/object/combined wrappers, and massive parallelization.

Lesson for CurvyTron: if the environment rules can be represented as pure array transformations, the cleanest architecture is a functional batched state transition plus batched observation function. That aligns better with JIT/vectorization than object-heavy Gym-style `step()` calls.

Source: [JAXAtari docs](https://jaxatari.readthedocs.io/en/latest/)

### Isaac Gym

Isaac Gym is the clearest "full loop on GPU" example. Physics simulation, observation calculation, reward calculation, policy tensors, action tensors, and rollout buffers can live on GPU. NVIDIA explicitly calls out eliminating costly CPU/GPU transfers and directly feeding action tensors back into the physics system.

Lesson for CurvyTron: the ideal target is not only GPU rendering, but GPU-resident observation/reward/action buffers. LightZero integration should try to preserve tensor residency between collector/search/model steps; CPU should coordinate batch boundaries, not own the inner loop.

Source: [NVIDIA Isaac Gym blog](https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/)

### Brax and PixelBrax

Brax uses JAX for accelerator-parallel physics and includes RL algorithms that compile alongside the environments, letting environment processing and learning occur on the same device. PixelBrax extends the point to visual RL: Brax physics plus a pure JAX renderer enables pixel observations over thousands of parallel envs and claims two orders of magnitude speedup over CPU-rendered benchmarks.

Lesson for CurvyTron: pixel observations are not inherently doomed, but CPU rendering is. If visual CurvyTron remains the contract, the render path should be a pure batched tensor function where stochasticity, resets, and distractors/augmentations are explicit tensor inputs.

Sources: [Brax arXiv](https://arxiv.org/abs/2106.13281), [PixelBrax arXiv](https://arxiv.org/abs/2502.00021)

### Sample Factory

Sample Factory is the high-throughput "pipeline the CPU/GPU pieces carefully" counterexample. Its architecture splits rollout workers, inference workers, batcher, and learner, with shared-memory buffers and lightweight signals instead of serializing observations between processes. Its docs also note that for GPU-accelerated envs like Isaac Gym, async mode gives little extra speed because everything is already on the same device; sync mode is recommended for sample efficiency.

Lesson for CurvyTron: if we cannot make the entire loop GPU-native immediately, use Sample Factory's discipline: coarse shared buffers, explicit components, double-buffered sampling, and measured handoff points. But once env/render/search are on one GPU, extra async plumbing may add staleness/complexity without much speed.

Sources: [Sample Factory architecture](https://www.samplefactory.dev/06-architecture/overview/), [Sample Factory sync/async note](https://www.samplefactory.dev/07-advanced-topics/sync-async/)

### EnvPool

EnvPool attacks the same bottleneck from the CPU side: a C++ batched environment pool using pybind11 and thread pools, with sync/async APIs and broad Gym/dm_env compatibility. It reports about 1M Atari FPS and 3M MuJoCo steps/sec on DGX-A100-class hardware, and describes itself as a general environment parallelization solution compared with GPU-specific systems like Brax and Isaac Gym.

Lesson for CurvyTron: EnvPool is a reminder that if source-state simulation remains CPU-owned, the fix is still batching and removing Python overhead. A compiled/batched CPU pool can be useful as an intermediate stage, but it still leaves visual tensor construction and model/search transfer as the next bottleneck.

Sources: [EnvPool docs](https://envpool.readthedocs.io/en/latest/), [EnvPool arXiv](https://arxiv.org/abs/2206.10558)

### MCTX / batched search

MCTX implements AlphaZero/MuZero/Gumbel MuZero-style tree search in JAX. Its README states that the search algorithms are JIT-compatible and operate on batches of inputs in parallel, which is what makes accelerator use effective with learned models.

Lesson for CurvyTron LightZero: MCTS/search should see a batch dimension as a first-class axis. If search repeatedly calls back into scalar env/render code, GPU observation work will not matter much. The handoff should be batched roots/source states -> batched model/search expansion -> batched actions/targets.

Source: [MCTX GitHub](https://github.com/google-deepmind/mctx)

## Transfer map

| Pattern | CPU owns | GPU owns | Crosses CPU/GPU in hot loop |
| --- | --- | --- | --- |
| Classic Gym + PyTorch | env step, render, reward, reset | policy/learner | observations/actions every step, often many small copies |
| CuLE/JAXAtari/Brax/PixelBrax | orchestration, logging, maybe checkpointing | env state, step, render/obs, policy/learner | ideally nothing per step; only coarse summaries |
| Isaac Gym | orchestration/assets plus task setup | physics, obs, rewards, actions, rollout buffers, policy | ideally no rollout data readback |
| Sample Factory | component scheduling, CPU envs unless GPU env configured | inference/learner, optionally env | shared-memory CPU movement for CPU envs; much less serialization |
| EnvPool | compiled CPU env pool | learner/inference unless paired with JAX/XLA | batched observations/actions, fewer Python crossings |
| MCTX | orchestration | batched search and model evaluation | avoid scalar search callbacks |

## CurvyTron-specific guidance

1. Treat the batch dimension as the product: `num_envs * players/seats * search_roots * rollout_steps` where possible. Small GPU kernels per snake/player/frame will lose to overhead.
2. Keep source-state tensors compact and device-resident through render/observation. Pulling rendered frames back to CPU before LightZero policy/search is the anti-pattern.
3. Prefer one synchronous GPU batch loop once env/render/search are co-located. Async workers help most when CPU simulators and GPU learners are separate bottlenecks.
4. If LightZero APIs force CPU/Gym boundaries, build a narrow adapter around a larger internal batch. The adapter can look scalar; the implementation should not be scalar.
5. Measure transfer volume explicitly: bytes of source state in, bytes of pixels/features out, and number of host-device synchronizations per train step. The sources above repeatedly point to bandwidth/synchronization, not raw shader/math, as the common failure mode.

Bottom line: the most relevant external pattern is Isaac Gym/Brax/PixelBrax for the destination, with Sample Factory/EnvPool as fallback architecture if LightZero cannot immediately consume a fully GPU-native vectorized env. For CurvyTron, the next useful design pressure is to make GPU rendering part of a batched full-loop contract, not a standalone acceleration island.
