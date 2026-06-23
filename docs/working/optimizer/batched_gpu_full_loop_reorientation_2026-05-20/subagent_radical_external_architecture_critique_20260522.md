# Radical External Architecture Critique

Date: 2026-05-22

Scope: external architecture critique for the CurvyTron optimizer lane. No
live Coach runs, trainer defaults, checkpoint jobs, eval jobs, GIF jobs,
tournament paths, or production code were touched.

## Short Answer

The user worry is correct.

The current `direct_ctree_gpu_latent + output-fast` work is useful, but it is
not shaped like a 5-10x architecture move. It is a local repair inside the
stock LightZero collection/search topology.

Current matched full-loop speedups:

```text
no-RND:       stock 433.17 steps/sec -> direct 566.19, about 1.31x
rnd_meter:   stock 351.02 steps/sec -> direct 448.52, about 1.28x
```

The durable profile-only ceiling wave now says:

```text
mock_search_service sim16:       11648.29 roots/sec
direct_ctree_gpu_latent sim16:    5303.97 roots/sec
recurrent_toy sim16:              8512.57 roots/sec
```

Treat that as strong evidence that compact arrays are promising. Do not treat
it as Coach-facing speed or as proof that search alone can deliver 5-10x.

Plain read:

```text
Small patches are working at the scale small patches should work.
They are not failing mysteriously.
They are leaving the main topology intact.
Compact search-service shape adds about 2.2x profile-only headroom, but the
missing multiplier probably also lives in the env/collector/replay object
boundary.
```

The big move for 5-10x is:

```text
many active CurvyTron row/seat roots
-> compact batched observation and legal-mask arrays
-> long-lived batched search/inference service
-> compact action / visit / value arrays
-> replay rows materialized only at a coarse edge
```

That is a MiniZero/KataGo/OpenSpiel-C++/Sample-Factory-style shape, not a
single LightZero hook.

## What The External Systems Actually Do

### AlphaZero And MuZero

AlphaZero and MuZero were not made fast by a tiny scalar Python loop. They used
large separate self-play/search and training pools. The AlphaZero paper reports
large TPU pools for self-play generation and network training, and the MuZero
paper reports separate self-play and training TPU resources for board games and
Atari:

- AlphaZero paper: <https://arxiv.org/abs/1712.01815>
- MuZero paper: <https://arxiv.org/abs/1911.08265>

CurvyTron implication:

```text
The scaling unit is actors/roots/games, not one collect call.
```

Do not expect an H100 to save a loop that feeds it tiny dynamic pieces through
Python/CPU/list boundaries.

### OpenSpiel AlphaZero

OpenSpiel describes actors, neural evaluators, learners, and evaluators. Its
docs explicitly distinguish the slow Python version from the C++ path: the C++
path uses threads, a shared cache, batched inference, and GPU support:

<https://openspiel.readthedocs.io/en/latest/alpha_zero.html>

CurvyTron implication:

```text
The useful pattern is shared batched inference/search.
C++ is useful when it removes the scalar boundary.
```

### MiniZero

MiniZero is the closest MuZero-like systems match. Its repo describes a server,
self-play workers, an optimization worker, and data storage. Each self-play
worker maintains multiple MCTS instances and batches GPU inference:

<https://github.com/rlglab/minizero>

CurvyTron implication:

```text
Keep multiple row/seat roots alive.
Collect leaves across roots.
Run recurrent inference as a real batch.
Return compact records.
```

This is the clearest shape for a CurvyTron search service.

### KataGo

KataGo's analysis engine gets speed from many positions in flight, GPU batch
fill, an asynchronous interface, and an NN cache:

- Analysis engine: <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>
- Example analysis config: <https://github.com/lightvector/KataGo/blob/master/cpp/configs/analysis_example.cfg>
- Self-play acceleration paper: <https://arxiv.org/abs/1902.10565>

CurvyTron implication:

```text
Many positions/roots feeding one search engine is the point.
Cache may matter less for CurvyTron than Go, but the service scheduler matters.
```

### LightZero

LightZero already has C++ CTree internals. Its docs say MuZero CTree has C++
`batch_traverse` and `batch_backpropagate`, and the README describes Python
`ptree` and C++ `ctree` implementations:

- LightZero repo: <https://github.com/opendilab/LightZero>
- LightZero tree-search docs: <https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>

CurvyTron implication:

```text
"Move CTree to C++" is not the missing 10x by itself.
The remaining wall is the boundary around CTree:
Python control, list APIs, model-output D2H, and scalar object packaging.
```

Array-native fixed-`A=3` CTree may still be useful, but it is probably a
1.5-2.5x compatibility bridge, not the whole radical architecture.

### MCTX

MCTX is JAX-native, batch-first, and JIT-compatible:

<https://github.com/google-deepmind/mctx>

CurvyTron implication:

```text
The clean endpoint is array-native search state visible to the compiler.
But MCTX is not a drop-in patch for PyTorch/LightZero.
```

Use it as a scratch/reference lane only if the model/search/replay boundary is
allowed to become JAX-shaped. Calling PyTorch from JAX would recreate the host
boundary.

### Sample Factory

Sample Factory splits high-throughput RL into rollout workers, inference
workers, a batcher, and a learner, communicating through shared memory rather
than serialized objects:

<https://www.samplefactory.dev/06-architecture/overview/>

CurvyTron implication:

```text
If actors stay CPU-owned, do not send scalar observations through Python.
Use compact shared buffers and central batched inference/search.
```

This is useful if we cannot or should not put the whole CurvyTron env on GPU
yet.

### PufferLib

PufferLib should be treated as a first-class target because its design is very
close to the kind of boundary cleanup CurvyTron keeps circling.

Relevant sources:

- Current docs: <https://puffer.ai/docs.html>
- Repo: <https://github.com/pufferai/pufferlib>
- Vectorization notes: <https://pufferai.github.io/dev/build/html/rst/landing.html>
- Performance tuning: <https://pufferai-pufferlib.mintlify.app/advanced/performance-tuning>
- Emulation/native-env docs: <https://pufferai-pufferlib.mintlify.app/concepts/emulation>
- Paper: <https://arxiv.org/abs/2406.12905>

PufferLib's reported high-throughput path is built around a few blunt
contracts:

```text
flat Box-like observations
simple action spaces
preallocated buffers
shared memory / zero-copy vector batches
multiple envs per worker
async send/recv ready batches
native multiagent handling
static memory for CUDA graph / compile-friendly training
```

The current PufferLib 4.0 docs push the same idea harder than the older
vectorization docs. The fast path is not just multiprocessing. It is static
memory plus native environment buffers: tensors are registered up front,
allocated as large contiguous blocks, and not recreated during the run. Repeated
GPU work is made graph-friendly with warmup and CUDA graph replay. Environment
instances are chunked into buffers owned by rollout workers, each worker can
queue pinned async transfers to the GPU, and the native Ocean-style envs write
observations, actions, rewards, and terminals into contiguous memory across
many environment instances.

The older vectorization docs and public source shape are still useful because
they expose the CPU-side contract:

- the async API is `async_reset/send/recv`, not only synchronous `step`;
- the vectorizer may simulate more envs than needed and return whichever
  contiguous ready batch is available;
- multiple envs per worker are used when individual envs are fast;
- workers write into shared observation/reward/done/mask/action buffers;
- shared flags and busy-waiting avoid per-step queue/pipe traffic;
- infos are kept out of the hot path and communicated at coarse cadence;
- native multi-agent support uses an agent axis and active-agent masks instead
  of a separate wrapper layer.

The docs also distinguish emulated environments from native Puffer
environments. Emulation can flatten Gym/PettingZoo spaces and can be convenient
for compatibility, but native environments with preallocated buffers are the
speed path. That distinction matters for CurvyTron: a Gym/PettingZoo wrapper
around our existing scalar LightZero env would mostly prove compatibility, not
the high-throughput architecture.

CurvyTron implication:

```text
PufferLib is not a drop-in MuZero/MCTS replacement.
PufferLib is a very strong hint about the collector/env boundary we should
build: flat arrays, no scalar timesteps, no dict fanout, no per-step allocation.
```

What applies directly:

- use a flat `[N,4,64,64]` uint8 observation buffer;
- use flat `[N,3]` legal masks;
- use flat `[N]` reward, done, row id, player id, and episode metadata;
- use a flat action writeback buffer, not an env-id keyed action dict;
- let workers write directly into preallocated contiguous buffers;
- receive ready contiguous subsets instead of one Python object per env;
- keep the player/agent axis native instead of wrapping multiagent as an
  afterthought;
- use async ready-batch scheduling when env step times vary;
- keep infos, diagnostics, and aggregate metrics out of the per-step buffer;
- use pinned host memory only where it actually reduces transfer wall, and
  avoid CPU offload when the observation/history batch fits on device;
- treat static memory as a design constraint, not an implementation detail.

What does not apply directly:

- PufferLib's main trainer path is PPO/V-Trace-style, not stock LightZero
  MuZero with MCTS/replay targets.
- Using PufferLib's trainer would mean rewriting the learner/collector
  algorithm, not optimizing the current MuZero path.
- Emulating CurvyTron as a normal Gym/PettingZoo env may be useful for a quick
  vectorization smoke, but the maximum-speed path would require a native
  PufferEnv-like CurvyTron boundary with preallocated buffers.

Concrete CurvyTron read:

```text
PufferLib strengthens the search-service thesis.
The first service should look more like a Puffer native vector env feeding a
batched search service than a LightZero scalar env manager.
```

#### PufferLib Pattern Versus Current CurvyTron Wall

The local bottleneck model now says the renderer is not the main wall. The
current profile-only direct hook improved the collect/search path, but the
matched full-loop gain is still about `1.28x-1.31x` because the loop keeps
paying scalar/object boundaries around a partially accelerated search.

Puffer-style architecture maps to that model like this:

| Puffer-style pattern | Current CurvyTron/LightZero state | Useful test |
| --- | --- | --- |
| Native vector env writes observations/rewards/dones/masks into preallocated buffers | The hybrid profile can keep `[B,2,4,64,64]` compact for a while, but stock LightZero still materializes env-id keyed timesteps and action dicts at the edge. | Measure flat-buffer consumer versus scalar timestep materialization on the same actor batch. |
| Async ready-batch `send/recv` with more envs than the consumer batch | Current stock/profile manager scheduling is mostly synchronous and scalar-env shaped. Actor/env scheduling was still visible in the local Puffer-style zero-observation probe. | Build an async in-process ready-batch mock before real multiprocessing. |
| Shared memory and shared flags instead of per-step serialized objects | Current stock env manager uses Python dicts, `BaseEnvTimestep` objects, info rows, copied masks, and per-env outputs. | Price object fanout by bypassing `_split_timestep_by_env_id` and action-dict assembly. |
| Static memory and CUDA graph-friendly repeated shapes | Direct CTree still returns model outputs to CPU/list form each simulation. Shapes are batch-stable, but the tree/list API prevents a clean graph. | Preallocate all search/replay arrays for fixed `B`, `A=3`, `sim`, then compare current list CTree versus flat-array loops. |
| Pinned async GPU transfer only where it moves the wall | Longer direct rows showed pinned input can help modestly, but input transfer was not the phase-changing bottleneck. | Keep pinned transfer as a measured option, not the architecture center. |
| Native multi-agent axis with active masks | CurvyTron has row/player axes, fixed-opponent `to_play=-1`, death/autoreset, and RND latest-frame semantics. Stock LightZero sees mostly scalar rows. | Treat `[row, player]` as the native root axis and test death/autoreset/RND separately before promotion. |
| Infos and diagnostics out of the hot path | Current RND/profile/summary paths can force CPU conversions, hashes, deep copies, and synchronized metrics. | Keep RND/metrics as separate profile axes; do not mix them into search-service claims. |

The concrete read is narrow:

```text
PufferLib supports the "compact buffers first" thesis.
It does not prove that environment vectorization alone is the missing 5-10x.
Our current evidence says search/replay/RND boundaries must also stop
reconstructing scalar Python objects.
```

#### Puffer-Style Falsifiers, Ordered By Value/Effort

1. **Compact actor batch plus mock search service, no scalar edge.**

   Reuse the existing profile-only hybrid batch shape:

   ```text
   obs_uint8[B,2,4,64,64]
   legal_mask[B,2,3]
   reward/done/row_id/player_id arrays
   -> mock_search_service or initial_inference-only consumer
   -> action/visit/value arrays
   ```

   Compare against the same row with scalar LightZero timestep materialization
   enabled. This is the cheapest way to test whether the Puffer-style flat
   boundary beats our object edge by enough to justify a larger rewrite.

   Kill condition:

   ```text
   If flat-buffer no-scalar is not at least 2x faster than the scalar edge on
   the same denominator, Puffer-style collection is not the next big lever by
   itself.
   ```

2. **Async ready-batch scheduler mock, still in one process.**

   Implement a profile-only scheduler that owns more actor rows than the
   consumer batch and returns ready contiguous chunks. It should write actions
   back through a flat array and report ready wait time, skipped/inactive rows,
   and batch fill. Do not use Python multiprocessing until this shape is worth
   preserving.

   Kill condition:

   ```text
   If ready-batch scheduling does not improve actor/search overlap or makes
   row ordering/reset semantics hard to preserve, keep the synchronous batch
   until search/replay are fixed.
   ```

3. **Flat-array CTree/list-boundary microbench.**

   Build a no-model microbench with fixed `B`, `A=3`, and `sim` that feeds
   synthetic reward/value/policy arrays through the current LightZero CTree
   list API, then through a flat-array prototype or shim. This directly tests
   the boundary PufferLib warns about: object/list churn around otherwise
   regular buffers.

   Kill condition:

   ```text
   If the current CTree list API is not a large fraction of the no-model
   microbench wall, the next service core should focus on recurrent launch,
   model batching, or replay, not CTree API surgery.
   ```

4. **Compact replay edge writer.**

   Take compact arrays from the mock or direct service and materialize exactly
   the fields stock LightZero replay/targets need at a coarse edge. Compare
   target fields against stock on forced-mask and clear-preference cases. The
   goal is not to change training behavior; it is to price whether replay
   immediately reintroduces the object wall.

   Kill condition:

   ```text
   If replay materialization consumes most of the search-service win, the
   architecture must own compact replay chunks before actor scaling.
   ```

5. **RND latest-frame resident path ablation.**

   Run matched profile-only RND rows where latest-frame extraction and metrics
   hashing are taken from a pre-extracted compact tensor/cache rather than from
   CPU object paths. Keep reward semantics explicitly non-production until
   parity is tested.

   Kill condition:

   ```text
   If RND stays flat, keep it as a guardrail axis. If it moves materially, a
   Puffer-style resident buffer is required before any Coach-facing RND lane.
   ```

### EnvPool, CuLE, Brax, Isaac Gym

These systems attack environment and host-transfer overhead at the environment
level:

- EnvPool: C++ batched env pool with sync/async APIs and million-FPS-class
  Atari throughput on large CPU hardware: <https://github.com/sail-sg/envpool>,
  paper <https://arxiv.org/abs/2206.10558>.
- CuLE: GPU Atari emulation and batching for RL:
  <https://arxiv.org/abs/1907.08467>.
- Brax: JAX environment and learning code compile together on accelerator:
  <https://arxiv.org/abs/2106.13281>.
- Isaac Gym: observations, rewards, actions, and rollout buffers can stay on
  GPU through learning:
  <https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/>.

CurvyTron implication:

```text
Full device-resident env + obs + policy + replay is a real endpoint.
But it is a new system, not the smallest next fix.
```

The immediate bottleneck evidence points first at search/collection topology.
If search becomes cheap, env/replay/RND may become the next walls, and these
systems become more directly relevant.

## What Is Actually Needed For 5-10x

The smallest credible 5-10x design is not:

```text
stock LightZero collector
plus one faster search hook
plus scalar env-output dicts
```

The credible design is:

```text
CurvyTron actor batch:
  obs_uint8[N,4,64,64]
  legal_mask[N,3]
  row_id[N]
  player_id[N]
  reset/final-observation metadata

Search service:
  owns root/tree/search state
  batches initial/recurrent inference
  keeps search state compact
  avoids per-simulation Python/list handoff
  returns action[N], visits[N,3], root_value[N]

Replay edge:
  writes compact rows/chunks
  materializes stock LightZero-shaped objects only for debug/compatibility
```

For fixed-opponent first, keep the scope even smaller:

```text
only learner-seat roots need real search;
the opponent can remain the current fixed/checkpoint policy path;
the first service does not need tournament, league, or true self-play support.
```

The service boundary matters more than the first search core. Behind the same
contract we can try:

- mock search-service ceiling;
- fixed-shape dense Torch/Triton search;
- array-native fixed-`A=3` CTree;
- later MCTX/JAX or custom CUDA.

PufferLib adds one sharper requirement:

```text
Do not design the service around Python env objects.
Design it around fixed buffers that an env worker can fill and a search service
can consume without copying.
```

## What Is Unrealistic

### "Just Use A Bigger GPU"

Unrealistic while the loop does this:

```text
GPU model tensors
-> CPU NumPy/list
-> Python CTree wrapper
-> GPU recurrent inference
-> CPU arrays/lists again
```

Bigger GPUs help only if they get bigger, steadier batches.

### "Just Add More CPUs"

Usually a distraction here. More CPUs can help CPU env pools, but the current
wall is not simply raw CPU math. It is boundary shape: scalar Python objects,
list conversions, and CPU/GPU synchronization points.

### "Just Rewrite CTree In C++"

LightZero already has C++ CTree. The missing part is array-native ownership and
service shape around the tree.

### "Use MCTX Directly"

Not a small patch. It is sensible only if CurvyTron is willing to own a JAX
model/search/replay path. Otherwise it just moves data across frameworks and
keeps the same bad boundary.

### "GPU-Resident Whole Env Immediately"

This may be a strong endpoint, especially after search is fixed, but it is too
broad as the next proof. It risks rewriting the environment before proving that
search-service ownership gives the needed multiplier.

### "Treat Mock 12k Roots/S As Training Speed"

Do not do this. The mock row removes real search and is profile-only. It is a
ceiling probe for architecture headroom, not an algorithm or Coach run.

### "Just Use PufferLib"

Also unrealistic as stated. PufferLib is fast because its env/trainer boundary
is different. If we put stock LightZero `BaseEnvTimestep` objects and MCTS
dict outputs on top of it, we keep the exact boundary that is hurting us.

The useful version is:

```text
steal PufferLib's buffer/vectorization contract;
do not blindly swap out the MuZero trainer.
```

## Smallest Falsifier

The smallest useful falsifier is already the right one:

```text
profile-only mock_search_service
same real batched observations
same real legal masks
same real scratch MuZero initial_inference
no CTree/search/recurrent rollout
compact action/visit/value arrays out
```

But the next pass must capture comparators durably.

Required first-wave rows on the same denominator:

| row | meaning |
| --- | --- |
| `mock_search_service` sim16 label | ceiling for perfect compact search boundary |
| `direct_ctree_gpu_latent` sim16 | current practical real-search boundary |
| `recurrent_toy` sim16 | real recurrent model pressure without CTree |
| optional `stock_facade` sim16 | public LightZero wrapper denominator |

Use the same H100 shape:

```text
B512 / actor_count 16 / sim16 label
60 measured / 15 warmup
uint8 stack
direct_gray64
persistent GPU render backend
scalar materialization off
root_noise 0.0
```

Decision rule:

```text
If mock_search_service is >=3x direct_ctree_gpu_latent:
  the search-service architecture thesis is strongly alive.

If mock_search_service is only <=1.5x direct_ctree_gpu_latent:
  search service alone cannot deliver 5-10x; move to replay/RND/env topology.

If mock is high but recurrent_toy is low:
  model/recurrent launch or dynamics inference is the next service-core wall.

If mock and recurrent_toy are high but direct is low:
  CTree/list/Python/search boundary is the wall.
```

This is better than immediately building a large service because it can kill
the thesis cheaply.

## Next Architecture Ladder

### Step 1: Durable Capture For The Ceiling Wave

Do not rely on PTY scrollback. Record:

- Modal app id;
- exact CLI;
- compact JSON result;
- roots/sec;
- measured wall;
- core timers;
- semantic identity;
- `touches_live_runs=false`;
- `calls_train_muzero=false`.

Until this is durable, do not make a strong decision from the `~12k` mock row.

### Step 2: Define The Search-Service Contract

Write the contract before implementing a real service:

```text
input:
  obs_uint8[N,4,64,64]
  legal_mask[N,3]
  to_play[N]
  row_id[N]
  player_id[N]
  reset/final-observation metadata

output:
  action[N]
  visit_policy[N,3]
  root_value[N]
  predicted_value[N]
  policy_logits[N,3]
  telemetry
```

The contract should say whether it is:

- fixed-opponent only;
- no-death profile only;
- normal-death/autoreset compatible;
- RND-compatible;
- stock replay compatible.

### Step 3: Put Multiple Search Cores Behind The Contract

Start with mock output, then plug in one real core at a time:

1. fixed-shape dense Torch/Triton core;
2. array-native fixed-`A=3` CTree core;
3. later JAX/MCTX or CUDA core.

Do not let each core invent a new training integration path.

### Step 4: Compact Replay Edge

If the service returns compact arrays but the next line immediately builds
per-env Python objects, the speedup will collapse. The first replay edge can be
compatibility-oriented, but it must be measured:

```text
compact service output -> stock LightZero row materialization
```

If that edge is expensive, build compact replay chunks before going wider.

### Step 5: Puffer-Style Vector Boundary Mock

Before a full actor farm, run one profile-only Puffer-style boundary mock:

```text
CurvyTron workers or in-process actor shards
-> preallocated shared/contiguous buffers:
     obs_uint8[N,4,64,64]
     legal_mask[N,3]
     reward[N]
     done[N]
     row_id[N]
     player_id[N]
-> central consumer reads a ready batch
-> mock_search_service or initial_inference consumes the batch
-> actions[N] written back as a flat array
```

Start in-process if that is faster to implement. The first test does not need
real Python multiprocessing. It needs the same contract PufferLib optimizes:
flat buffers and no scalar object fanout.

Metrics:

- env/actor step wall;
- bytes written to buffers;
- copy count;
- ready batch size and wait time;
- consumer roots/sec;
- action writeback time;
- optional cost to materialize stock LightZero rows at the edge.

Decision rule:

```text
If flat-buffer collection plus mock_search_service is much faster than the
current LightZero-shaped sidecar, Puffer-style collection is worth pursuing.

If it is close to the current sidecar, the next wall is search/replay/RND, not
the vectorization layer.
```

This is the smallest experiment that tests the PufferLib pattern without
rewriting the trainer.

### Step 6: Actor Scale Only After One Actor Batch Works

Do not start with 200 actors. First prove one batch/service path. Then scale:

```text
one B512 actor batch -> service
many B512 actor batches -> service queue
multiple services -> learner/replay
```

Scaling early only hides the bottleneck.

## What Not To Waste Time On

- More renderer-only profiling as a main lane. Keep renderer tests as
  validation/background.
- More output-fast polishing unless a fresh timer shows output is hot again.
- More CPU count sweeps before a compact search-service row is captured.
- More H100/L4 comparisons without first ensuring the GPU sees enough batch.
- A PufferLib trainer port before proving a Puffer-style flat-buffer collector
  helps the current CurvyTron/MuZero denominator.
- Exact neutral/tie-heavy CTree visit parity. Use forced cases and statistical
  gates.
- Full self-play league/tournament integration before the fixed-opponent
  service boundary is proven.
- RND optimization mixed into search-service speed claims. Keep RND as a
  separate axis.
- A full JAX/CUDA env rewrite before the search-service falsifier says the
  multiplier exists.

## Concrete Recommendation

The next real optimizer move should be:

```text
Finish the profile-only mock_search_service ceiling wave with durable capture.
Then write the fixed-opponent search-service contract.
Then implement exactly one real search core behind that contract.
```

My current bet:

```text
Current direct hook:     useful 1.3x tactical bridge.
Array-native CTree:      maybe 1.5-2.5x compatibility bridge.
Search-service boundary: first credible 5-10x lane, but only if the mock
                         ceiling stays far above direct when captured cleanly.
Puffer-style buffers:    best concrete collector boundary pattern to combine
                         with the search service.
Full GPU env/replay:     later endpoint if search service moves the wall.
```

The fastest way to stop spinning is not to pick the final architecture today.
It is to make the mock service comparison impossible to argue with.
