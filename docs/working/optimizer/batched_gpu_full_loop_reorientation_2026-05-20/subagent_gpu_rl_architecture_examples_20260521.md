# Subagent GPU RL Architecture Examples

Date: 2026-05-21

Status: external research note only. Do not edit live training runs, trainer
defaults, tournament defaults, or active Modal runs.

## Short Read

The external systems that get material GPU RL speedups usually do more than
accelerate rendering. They preserve a large batch across environment state,
observation construction, policy/search inference, reward/reset, replay, and
learning. The less invasive alternative is a CPU actor system with shared
buffers and centralized batched inference.

For CurvyTron, the current profile-only hybrid scaffold is pointed in the right
direction:

```text
VectorMultiplayerEnv[B,2]
-> compact state
-> batched direct_gray64 observation
-> [B,2,4,64,64]
-> stock LightZero-shaped boundary
```

The architectural question is whether that batch survives contact with
LightZero collection/search/replay, or whether it collapses back into many
scalar Python timesteps.

## Pattern Table

| Example | Concrete pattern | Optimizes | CurvyTron changes required | Expected speedup class | Main risks | Next toy experiment |
| --- | --- | --- | --- | --- | --- | --- |
| CuLE | CUDA Atari emulator runs thousands of games and renders frames on GPU. | CPU env bottleneck, CPU/GPU bandwidth, tiny per-env dispatch overhead. | Move more than observation rendering: compact source state, simultaneous step, trail update, reset/reward, and direct observation would need a batched device-owned representation. | Large if full env+obs is device-resident; small to medium if only render is moved and frames return to CPU each step. | Source-faithful CurvyTron step/reset semantics are easier to break than Atari frame emulation; GPU kernels can become trail-capacity dominated. | Implement a tiny fixed-size two-snake device step kernel with no LightZero: `B=1024`, actions `[B,2]`, output compact state + terminal flags + direct gray64, compare against CPU oracle on row/player/reset parity. |
| JAXAtari | Game logic is rewritten as pure JAX, JIT compiled and vectorized, with pixel/object wrappers. | Python object `step()` overhead and host-device crossings. | A functional CurvyTron state transition in JAX: arrays for positions, headings, alive flags, trail occupancy, RNG, rewards, terminal/final observation bookkeeping. | Large for simulator/observation throughput if the full loop can remain JAX-native; likely not a drop-in LightZero win. | JAX rewrite cost, dynamic trails/branching, and PyTorch LightZero model/search mismatch. | Build a minimal JAX CurvyLine toy with fixed grid and one-step terminal collision only; use `vmap`/`scan` to measure random-policy env steps/sec and verify both player perspectives. |
| Brax / PixelBrax | Environment processing, renderer, and RL algorithms compile together on accelerator; PixelBrax adds pure JAX pixel rendering. | End-to-end pixel RL where CPU rendering would dominate. | Direct `64x64` observation should be expressed as a pure batched tensor function, and policy/search should consume the tensor without readback. | Medium to large only if search/inference also stays batched; renderer-only headroom is bounded by current zero-observation ceiling. | Pixel parity is the wrong target, but semantic drift is dangerous: missing trails, wrong draw order, wrong latest-frame/RND extraction. | Extend the existing profile-only direct surface benchmark with a no-readback path: render `[B,2,4,64,64]` then immediately run a dummy Torch/JAX policy batch on GPU, reporting sync count and bytes read back. |
| Isaac Gym | Physics, observations, rewards, actions, rollout buffers, and policy tensors stay on GPU through learning. | Rollout data readback and CPU orchestration in the inner loop. | CurvyTron would need GPU-resident action buffers, reward/reset buffers, stack buffers, and possibly replay chunks; CPU would only coordinate coarse batches and logging. | Large when the loop is genuinely resident; limited if LightZero still requires CPU `BaseEnvTimestep` objects every row. | Biggest ownership change; stock LightZero contracts, replay format, RND cadence, checkpoints, and eval/tournament metadata become integration hazards. | Prototype a "resident rollout buffer" profile: keep observations/actions/rewards/dones as device tensors for `T` steps, materialize CPU LightZero-like timesteps only once at the end, and measure against per-step materialization. |
| JaxMARL | Multi-agent envs and algorithms are written in JAX with `vmap`/`scan`, enabling GPU-vectorized MARL and self-play. | Multi-agent batch axes, agent axes, and rollout-time axes are first-class arrays. | CurvyTron seat axis should be explicit: env rows `[B]`, players `[2]`, active roots `[B*2]`, simultaneous actions `[B,2]`. | Medium to large for multi-agent self-play mechanics if rewritten; medium as a design guide for the existing vector facade. | Easy axis bugs: player view, row order, terminal observation before autoreset, and treating player axis as RND unroll axis. | Add a profile-only axis stress canary with asymmetric player states and deterministic actions, asserting `[B,2] -> [B*2] -> [B,2]` gather/scatter round trips before any trainer wiring. |
| EnvPool | C++ batched CPU environment pool with sync/async APIs and large thread pools. | Python env overhead, slow subprocess vectorization, and straggler effects. | If source-state simulation stays CPU-owned, CurvyTron needs compact contiguous state buffers and an API that can step many envs with actions in bulk, not many Python `step()` calls. | Medium; can be large versus pure Python env managers, but still pays CPU/GPU observation or inference transfer unless paired with batched GPU inference/render. | Rewriting the env pool in C++/Rust is nontrivial; simultaneous two-player semantics and final observations need careful ABI tests. | Build a Python-free toy step loop in NumPy/Numba/C++ for just compact state transition, no render, and compare to current vector manager step cost at C512/C1024. |
| Sample Factory | Rollout workers, inference workers, batcher, and learner communicate through shared-memory buffers and centralized inference. | Serialization, per-process inference calls, and learner/actor coupling. | Keep CPU actors if they are useful, but replace per-row observation/model calls with shared compact buffers plus a central batched renderer/inference service. | Medium to large if current stock loop is actor/search orchestration limited; less useful once env/render/search are all on one GPU. | Policy staleness, queue latency, harder reproducibility, and more moving pieces around RND/replay/checkpoint metadata. | Profile a toy central inference service: N subprocess actors submit compact fake observations into shared memory, one process batches Torch policy calls, actors receive actions; report batch fill latency and actions/sec. |
| Batched MCTS / MCTX | MCTS is JAX-native, JIT-compatible, and operates over batches of roots in parallel. | Scalar search expansion and many small model calls. | Flatten active player roots to `[B*2]`, run batched representation/dynamics/prediction, then scatter selected actions back to simultaneous `[B,2]` env steps. A real MCTX path likely means JAX model/search ownership or a shadow model. | Medium to large in search-heavy rows; especially relevant when GPU utilization is low and MCTS/model calls dominate. | PyTorch LightZero mismatch, different MuZero semantics, checkpoint/replay incompatibility, and debug difficulty for tree state. | Before changing algorithms, add a batched-root inference-only microbench in current Torch: collect `B*2` observations, run one policy/value batch, then compare against scalar per-root calls and report utilization. |

## Source Notes

- CuLE: CUDA Atari emulation removes CPU-based emulator bottlenecks and reports
  tens to hundreds of millions of frames/hour on one GPU.
  Source: https://mgarland.org/papers/2019/cule/
- JAXAtari: JAX-based Atari-style environments emphasize JIT/vectorized GPU
  execution and object/pixel wrappers.
  Source: https://jaxatari.readthedocs.io/en/latest/
- Isaac Gym: NVIDIA describes observation/reward calculations running on GPU,
  with observation, reward, action, and rollout buffers staying on GPU through
  learning.
  Source: https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/
- Brax: the paper describes JAX RL algorithms compiling alongside environments
  so environment processing and learning occur on the same device.
  Source: https://arxiv.org/abs/2106.13281
- PixelBrax: combines Brax with a pure JAX renderer for end-to-end GPU pixel RL
  and reports two-orders-of-magnitude speedups over CPU-rendered benchmarks.
  Source: https://arxiv.org/abs/2502.00021
- JaxMARL: pure JAX multi-agent environments and algorithms report about 14x
  wall-clock improvement over existing approaches, and much larger gains when
  vectorizing multiple runs.
  Source: https://arxiv.org/abs/2311.10090
- EnvPool: C++ batched environment pool with sync/async APIs, Gym/dm_env
  compatibility, multiplayer support, and reported million-FPS-class Atari
  throughput on large CPU hardware.
  Sources: https://github.com/sail-sg/envpool and https://arxiv.org/abs/2206.10558
- Sample Factory: architecture uses shared-memory buffers rather than
  serializing observations between components; batched mode is intended for
  massively vectorized envs like Isaac Gym or EnvPool.
  Sources: https://www.samplefactory.dev/06-architecture/overview/ and
  https://www.samplefactory.dev/07-advanced-topics/batched-non-batched/
- MCTX: JAX-native AlphaZero/MuZero/Gumbel MuZero search supports JIT
  compilation and parallel batches of inputs.
  Source: https://github.com/google-deepmind/mctx

## CurvyTron Read

The likely speed classes are:

1. **Renderer-only tuning:** bounded, because the current direct surface has
   already removed much of the dense render wall in profile-only rows.
2. **Stock-compatible vector facade:** medium, if stack/pack/timestep overhead
   can approach the zero-observation ceiling while preserving LightZero
   contracts.
3. **CPU actor pool plus batched render/inference:** medium to large, if
   subprocess collection remains the practical win and model/search calls can
   be centralized.
4. **Device-resident env/search rewrite:** largest potential, but also a new
   RL system boundary rather than a trainer backend flag.

My read: keep the current profile-only hybrid scaffold, but use external
systems as pressure to measure where the batch dies. If the batch dies at
`BaseEnvTimestep` materialization, pursue the stock-compatible vector facade. If
it dies at actor/model calls, prototype Sample-Factory-style central batched
inference. If both become walls, only then consider a JAX/CUDA CurvyTron env
rewrite.

## Recommended Next Toy Experiments

1. **Batched-root Torch inference microbench:** use existing direct-surface
   observations, flatten `[B,2]` to `[B*2]`, run one policy/value batch, and
   compare against scalar per-root inference. This is the cheapest MCTS-adjacent
   signal without leaving LightZero/PyTorch.
2. **Resident rollout-buffer profile:** keep `obs/action/reward/done` tensors on
   GPU for `T` profile steps and materialize CPU timesteps once per chunk. This
   directly tests whether per-step host materialization is the current batch
   killer.
3. **Axis stress canary:** deterministic asymmetric two-player rows that verify
   gather/scatter, stack order, terminal `final_observation`, and RND latest
   frame extraction before any trainer-visible integration.
4. **Central inference service toy:** CPU actors submit compact state or fake
   obs to shared memory; one service batches render/inference; actors receive
   actions. Measure batch fill latency, actions/sec, and staleness.

None of these require touching live runs, trainer defaults, or tournament
defaults. They are profile-only architecture probes.
