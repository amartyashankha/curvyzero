# Architecture Design Critiques - 2026-05-22

Status: docs-only optimizer subagent note. I did not touch live Coach training,
trainer defaults, checkpoints, evals, GIFs, tournaments, Modal volumes, or
source code.

Scope: CurvyTron compact visual + MCTX/search/replay loop. The goal is to find
architecture changes that can beat the current `1.3x-1.8x` incremental class.

## Starting Read

The current profile-only fast path already fixed the obvious mistake:

```text
MCTX root values now come from direct root-node fields.
Replay row construction is small on the latest replay-valid rows.
Raw GPU drawing is small.
```

The remaining wall is not one clean function. It is a chain:

```text
GPU search picks action
-> action copied to CPU
-> CPU CurvyTron state advances
-> CPU compact/render/search-input state is rebuilt or packed
-> small/medium deltas go to GPU
-> GPU stack/root input is updated
-> MCTX/search runs
-> compact replay fields are read/validated/materialized
```

That means the big wins probably require changing ownership, not shaving one
timer. A good design removes a boundary. A bad design only moves the wait to a
new timer name.

Speedup classes used below:

| Class | Meaning |
| --- | --- |
| `<1.2x` | Micro win. Worth keeping only if simple and safe. |
| `1.5x` | Useful boundary cleanup, not a new system. |
| `2-4x` | Structural win on one large subsystem. |
| `5-10x` | Requires env/search/replay ownership change together. |

## Design Matrix

| # | Design | What It Removes | CPU/GPU Residency | Possible Speedup | Risk | Validation Gate | Why It Might Fail |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Direct visual-delta renderer path | Production-to-compact visual conversion and much of CPU delta packing. | CPU env remains owner; GPU owns persistent framebuffer/stack; CPU sends only changed visual deltas. | `<1.2x` to `1.5x` | Medium | Same frames/checksums as current persistent renderer on seeded cold, incremental, reset, bonus, and trail-break cases; matched H100 loop improves at least `5-10%`. | Current data says raw draw is tiny and H2D/delta is only part of `env_step_sec`; this is probably not a 10x lane. |
| 2 | PufferLib-style contiguous env/search buffers | Scalar env timesteps, per-root dicts, Python payload merge, repeated small object materialization. | CPU owns static SoA buffers for obs/reward/done/mask/meta; GPU consumes obs/masks; replay writer reads arrays. | `1.5x` to `2-4x` | Medium | A compact consumer can run one full collect/search/replay-profile loop without `BaseEnvTimestep` objects and produce identical replay rows. | If MCTS/search dominates after scalar objects are removed, env buffer cleanup alone caps out near `1.5x`. |
| 3 | Numba/SoA CPU vectorized CurvyTron env | Python row loops in mechanics, collision, trail updates, and render-state packing. | CPU stays authoritative, but hot arrays are typed SoA NumPy/Numba buffers; GPU still handles render/search. | `1.5x` to `2-4x` | Medium | `step_many` parity against scalar CurvyTron fixtures over long no-death, collision, reset, bonus, and stochasticity cases; env bucket falls by `2x+`. | Branchy collision/trail logic may not vectorize cleanly; search/input handoff may still dominate. |
| 4 | C++/Rust native env plus delta-pack extension | Python/NumPy overhead in mechanics and visual delta construction. | CPU native extension owns env state and emits compact deltas/replay sidecars; GPU still owns render stack/search. | `2-4x` on env/handoff, maybe `5-10x` only if env is the wall | High | Native and Python scalar states match over seeded adversarial traces; Modal build is stable; no Python objects in the hot loop; full loop improves `2x+`. | Expensive to build and easy to make semantically wrong; if search is already the wall at sim32+, native env work will not deliver 10x. |
| 5 | Resident JAX env state | CPU env step, CPU render-state packing, H2D visual deltas, and selected-action CPU dependency. | JAX/GPU owns positions, trails, bonuses, masks, observation stack, and maybe rewards/done; CPU sees summaries only. | `5-10x` if it covers env + obs + search | Very high | A jitted fixed-shape no-death loop runs many steps without host sync and matches scalar fixtures within defined tolerance; terminal/reset support has exact gates. | Dynamic collision/trail/bonus semantics may cause huge JAX complexity, recompiles, or memory blowup; PyTorch/LightZero interop may erase the win. |
| 6 | Full JAX/MCTX compact loop | Framework boundary between env/render/search; Python/CPU CTree-style control; repeated host/device ownership changes. | JAX owns obs/root/search/tree arrays and recurrent model; CPU only validates/export chunks. | `5-10x` search-boundary ceiling | Very high | Fixed `B,P,A,sim` MCTX loop with real compact observations beats current replay-valid profile by `3x+` and maps to `CompactSearchResultV1` exactly. | The current JAX model is toy; porting the production model/learner/replay path may become a new project. |
| 7 | MiniZero/KataGo-style batched search service | Tight synchronous env/search interleave and underfilled GPU search batches. | Many CPU env actors feed a GPU search/model service; compact results return as action/visit/value arrays; replay materializes at chunk edge. | `2-4x` first, `5-10x` if batching stays full | High | Mock service ceiling first: replace real search with legal synthetic results and prove the rest of loop can run far faster; then plug MCTX/dense search behind same API. | Queue latency, policy staleness, replay ordering, or env-side bottlenecks can eat the win; if one batch already saturates GPU, service adds overhead. |
| 8 | Array-native fixed-`A=3` CTree/search API | Python lists and dynamic legal-action objects at the CTree boundary. | CPU C++ CTree remains owner; inputs/outputs are dense arrays; Torch/JAX model may stay on GPU. | `<1.2x` to `1.5x` | Medium | Same legal masks, visit targets, root values, and action choices as LightZero CTree; no-model benchmark improves and matched full-loop improves `>=10%`. | Prior flat-A3 evidence suggests list ABI cleanup is real but not enough; CPU traversal and recurrent sync remain. |
| 9 | Dense Torch/Triton fixed-shape GPU tree | CPU CTree traversal/backprop, Python loop control, CPU tree memory layout. | GPU tensors own tree edges, priors, visits, values, latent slots; CPU reads final action/visit/value. | `2-4x`, possible `5-10x` if search dominates | High | Forced-mask and clear-preference tests exact; sim16/sim32 same-denominator row beats current MCTX/CTree path materially; replay rows match current compact contract. | MCTS has sequential dependence inside each simulation; naive GPU tensor code can be slower than C++ CTree unless roots are huge and kernels are fused. |
| 10 | Replay materialization deferral / compact replay owner | Per-step scalar replay row construction, strict validation in the hot path, and premature CPU materialization of visit/value payloads. | Search outputs remain compact arrays, preferably device or pinned host; replay chunks materialize only at safe flush/sample boundaries. | `<1.2x` to `1.5x` alone; `2-4x` as part of service architecture | Medium | Multi-record parity canary proves actions, visits, root values, rewards, done, final obs flags, row ids, and player ids match immediate replay-valid path. | Latest replay-index rows are already small; deferral alone just moves cost unless paired with a bigger search/env service. |

## Design Notes

### 1. Direct Visual-Delta Renderer Path

This is the cleanest near-term renderer change, but it is not the main 10x bet.
The current persistent renderer still does:

```text
production state -> compact visual state -> delta state -> H2D -> GPU update
```

A direct path would read `visual_trail_*`, head positions, alive/present flags,
and bonus fields directly from the production state and emit the delta/compose
payload. It should keep the current compact path as the oracle.

Ruthless read: test it because it is small and clarifies the wall. Do not keep
hammering it if whole-loop speed moves less than `5-10%`.

### 2. PufferLib-Style Contiguous Buffers

This is the most practical systems cleanup. The current code has already found
the right attach point: the compact batch before scalar LightZero materializing.

Target shape:

```text
obs[B,P,4,64,64] uint8
legal_mask[B,P,3] bool
reward[B,P] float32
done[B] bool
row/player ids
final observation masks
```

The optimized consumer should read these arrays directly. Stock LightZero
objects should be compatibility/debug output, not the hot path.

Ruthless read: this is not glamorous, but it is how fast RL systems avoid
death-by-small-Python-object. It is probably a prerequisite for every larger
search-service design.

### 3. Numba/SoA CPU Vectorized Env

This keeps the architecture familiar while attacking Python mechanics. It is
less risky than JAX/C++ but more serious than local renderer flags.

Good first target:

```text
step_many(action[B,P]) -> compact sidecars + visual deltas
```

Keep rendering/search unchanged. If this does not cut the env bucket, the CPU
env is not the current wall. If it does, it gives a clean route before a native
extension.

Ruthless read: start with no-death long trajectories and fixed stochasticity.
Do not implement the whole game before proving a kernel-shaped step is faster.

### 4. C++/Rust Native Env Extension

This is the brute-force version of design 3. It can be the right answer if
Python mechanics and visual-delta packing remain the wall after search cleanup.

The native extension should not render browser pixels. It should own:

```text
positions, heading, alive, trail write cursors, collision facts, bonus facts,
legal masks, rewards, done, compact visual deltas
```

Ruthless read: good endpoint, bad first move unless profiling clearly says env
mechanics/packing are dominant. Packaging and semantic drift are real costs.

### 5. Resident JAX Env State

This is the pure GPU dream: the env state never leaves device, actions are
selected on device, and MCTX consumes the next root immediately.

Best-case loop:

```text
jax env state
-> mctx action
-> jax env step
-> jax render/stack update
-> next mctx root
```

Ruthless read: this is the only env-side design that can remove the selected
action CPU sync. It is also the easiest way to build a fast thing that no
longer matches CurvyTron. Treat it as a scratch ceiling, not a trainer patch.

### 6. Full JAX/MCTX Compact Loop

MCTX is the cleanest all-device search reference. The current profile-only lane
already proves the shape is promising, but it uses a toy model and is not stock
LightZero training.

The useful next test is not "can MCTX run?" It can. The useful test is:

```text
Can a fixed-shape compact loop keep env/search/replay summaries device-first
and still produce the exact compact replay contract?
```

Ruthless read: high upside, high integration risk. Use it to learn the ceiling
and the contracts before deciding whether to port production training.

### 7. Batched Search Service

This is the most credible architecture for large speedups without requiring the
entire game to become JAX immediately.

Shape:

```text
many env rows/actors
-> queue compact roots
-> one GPU search/model service batches roots
-> return action, visit policy, root value
-> compact replay writer
```

The service can start with a mock search core. If the mock service is not much
faster, a real MCTS service cannot save the full loop.

Ruthless read: this should be tested before deeper C++ or JAX rewrites. It
answers whether the current synchronous topology is the real ceiling.

### 8. Array-Native Fixed-`A=3` CTree

This is a compatibility bridge, not the big win. It respects LightZero-style
search while deleting nested Python lists.

Inputs/outputs should look like:

```text
reward[N] float32
value[N] float32
policy[N,3] float32
legal_mask[N,3] bool
action[N] int16
visits[N,3] float32
```

Ruthless read: prior evidence says this alone is too small. Keep it as a lower
risk fallback if full GPU search is too expensive.

### 9. Dense Torch/Triton Fixed-Shape GPU Tree

This keeps the production framework closer to Torch and avoids a full JAX port.
It only makes sense if the tree is fixed-shape and preallocated:

```text
N roots, A=3, fixed simulations, fixed max nodes
```

Ruthless read: naive GPU MCTS can be slower because tree search has sequential
parts. It needs many roots, fixed shapes, and fused kernels to win.

### 10. Compact Replay Owner / Deferred Materialization

Deferring replay materialization is not magic. If we copy and validate the same
payload later on the same thread, total wall comes back.

It becomes useful when paired with a chunked service:

```text
search emits compact arrays
-> replay chunk stores compact arrays
-> scalar/debug rows are materialized only for parity or sampling boundaries
```

Ruthless read: do not sell this as a standalone win. Do build it as a contract
piece for search-service and Puffer-style buffer designs.

## Cross-Cutting Validation Gates

Any serious design needs these gates before becoming trainer-facing:

1. **Replay parity.** Same selected action, visit policy, root value, reward,
   done, final-observation flag, row id, and player id as the current compact
   replay path on seeded traces.
2. **Observation parity or documented tolerance.** Same stack ordering, legal
   masks, player perspective, reset behavior, bonus encoding, and terminal
   final observation.
3. **Long no-death profile.** Short trajectories hide env/render costs. Use
   loop lengths like `96+` and stable warmup.
4. **Search stress.** Run sim16 and sim32 at least. If sim32 is search-heavy,
   env-only work cannot be the full answer.
5. **Host-sync accounting.** Report selected-action D2H, mask H2D, stack/root
   ownership, search, replay materialization, and unattributed wall separately.
6. **Mock ceiling first for rewrites.** Before building a real search service,
   replace search with legal synthetic outputs and measure the maximum possible
   loop gain.

## Top 3 Recommendations To Test First

### 1. Mock Batched Search Service Ceiling

Why first: it tells us whether changing the search/replay topology can even
produce a `5-10x` loop. It is the fastest way to kill or justify the big
architecture thesis.

Test:

```text
current compact env/root producer
-> mock legal action/visit/value service
-> compact replay writer
```

Pass signal: full replay-shaped compact loop is several times faster than the
current MCTX/CTree loop. Fail signal: env/obs/replay remain too slow even with
search removed.

### 2. Puffer-Style Contiguous Compact Buffer Consumer

Why second: every big design needs the same array-native contract. It removes
scalar LightZero objects from the hot path and makes search-service, RND, and
replay ownership testable.

Test:

```text
Hybrid manager writes static row/player arrays
-> compact consumer reads them directly
-> no BaseEnvTimestep or per-env dicts in the measured hot path
```

Pass signal: scalar/object boundary falls out of the profile and compact replay
rows still match.

### 3. Fixed-Shape GPU Search Core Behind The Same Service API

Why third: after the mock service proves the ceiling, plug in a real search
core. Prefer the already-promising MCTX/JAX compact-root lane for the first
ceiling test, and keep dense Torch/Triton as the Torch-native alternative.

Test:

```text
input: obs/root latent + legal_mask[N,3]
output: action[N], visits[N,3], root_value[N]
```

Pass signal: sim16 and sim32 beat the current replay-valid loop by a structural
margin, not another `5%` timer reshuffle.

## Bottom Line

The next real speedup is probably not hidden inside one renderer function.

The main optimizer should test whether the hot loop can become:

```text
contiguous compact env buffers
-> batched search service
-> compact replay chunks
```

If that mock ceiling is high, build the GPU search core. If it is low, stop
guessing and move the attack to whatever bucket remains after search is mocked
away.
