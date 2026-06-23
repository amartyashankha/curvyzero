# Batched GPU Observation Trainer Integration Plan

Date: 2026-05-15

Status: plan-only / no live runs. This document is a planning note only. It
does not change trainer defaults, runtime code, launchers, checkpoints,
tournaments, Modal Volumes, or active jobs.

## Current Reality

The fast observation candidate is a profile-only batched GPU boundary, not the
current trainer backend named `policy_observation_backend=jax_gpu`.

What the profile lane currently proves:

- It targets the policy surface we care about:
  `browser_lines + simple_symbols -> [4,64,64]`.
- It renders both player views.
- It uses owner-ordered compact trail input so the renderer sees active visual
  trail records in the right draw order.
- It treats `geometry_dtype=float32` as the aggressive default candidate.
- It keeps `geometry_dtype=float64` as the exact CPU-oracle reference/debug
  mode.
- It can model stack update, row-major player ordering, reset frames, and
  terminal `final_observation` capture in the sidecar profile.

What the trainer currently has:

- The wired scalar `policy_observation_backend=jax_gpu` path is slow in the
  full trainer.
- That scalar path is useful as a diagnostic canary only.
- It should not become the default training backend.

Plain read: the batched boundary is promising because it measures the shape we
actually want. It is not yet the backend the trainer calls. The scalar
`jax_gpu` trainer flag is not the speed path.

## Why It Is Not Just A Flag Flip

The flag that exists today selects a scalar backend. It does not select the
profile-only batched boundary.

The difference matters:

- Scalar `jax_gpu` renders one small observation boundary at a time, so launch,
  transfer, readback, and static trail capacity costs dominate.
- The batched sidecar renders many rows and both player views together. That is
  the whole reason it can amortize GPU overhead.
- Stock LightZero expects observations to arrive through its env-manager and
  timestep contracts. The profile sidecar owns a cleaner boundary than the
  current trainer path.
- LightZero `env.step` expects an observation immediately. A production batched
  GPU path therefore has to live at an env-manager/vector-env boundary, not as a
  late collector-side decoration after timesteps already exist.
- The renderer returns fused/batched frames, but the trainer consumes
  per-row/per-player stacked observations with reset and terminal semantics.
- The sidecar still reads frames back to host before stack update. A production
  version may need a different memory boundary to keep the win.
- The batched profile has parity and timing evidence, but not a full trainer
  canary with policy forward, replay write, subprocess behavior, and terminal
  row reuse.

So the answer to "why not just use batched?" is: because "batched" is currently
a measured boundary shape, not a trainer integration. We need to decide where
that boundary lives in the training loop, prove it preserves the observation
contract, and only then expose it as a selectable backend.

## Why This Is Suspicious But Real

It is suspicious because the wired GPU flag is slower than CPU. That result is
real, but it measures the wrong shape: one row at a time, with launch/copy
overhead paid every step.

It is also real that the sidecar can be faster. It batches many env rows and
both player views, then updates the same `[B,2,4,64,64]` observation contract
the trainer needs. The catch is that the sidecar owns the clean boundary today;
the trainer does not. The next proof is not "GPU render is fast." The next proof
is "the env-manager can return correct LightZero timesteps through the batched
boundary without losing the timing win."

## External Pattern Check

Other systems do batch the expensive parts. The suspicious part is not the idea
of batching; the suspicious part is that our current trainer flag does not reach
the batching boundary.

- MCTX is the cleanest search reference. It is JAX-native, JIT-friendly, and its
  search algorithms operate on batches of inputs in parallel:
  <https://github.com/google-deepmind/mctx>.
- OpenSpiel's AlphaZero docs make the same systems split explicit: the simple
  Python implementation is CPU/non-batched, while the C++ path uses threads, a
  shared cache, batched inference, and GPU inference/training:
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>.
- DI-engine/LightZero's env-manager contract explains our local blocker:
  `ready_obs` feeds policy input, and `step(actions)` returns timestep objects
  with observation, reward, done, and info. That means observation construction
  is part of the env-manager contract, not a harmless afterthought:
  <https://di-engine-docs.readthedocs.io/en/latest/05_api_doc/env.html> and
  <https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html>.
- EnvPool shows the high-throughput env pattern: batched env APIs, C++ thread
  pools, batched RGB output, sync/async modes, and very high raw FPS:
  <https://github.com/sail-sg/envpool>.
- RLlib exposes the same axes at a larger systems level: many EnvRunner actors,
  vectorized sub-envs per runner, and learner actors. Its docs also point out
  that default vector envs may still step/reset sub-envs sequentially while
  batching model inference:
  <https://docs.ray.io/en/latest/rllib/scaling-guide.html>.
- Sample Factory separates rollout workers, inference workers, a batcher, and a
  learner, and uses double-buffered sampling to keep CPU env work from idling
  while GPU inference runs:
  <https://www.samplefactory.dev/06-architecture/overview/> and
  <https://www.samplefactory.dev/07-advanced-topics/double-buffered/>.
- EfficientZero-style systems use data/self-play workers, replay, CPU context
  workers, GPU batch workers, reanalysis, and a learner in parallel. The lesson
  is not "copy the whole framework"; it is "put batchable work at a real batch
  boundary": <https://github.com/YeWR/EfficientZero> and
  <https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material>.

Plain conclusion: yes, batching is a known solution. No, it is not normally
implemented by hiding a GPU batcher inside a scalar `env.step`. The closest
pattern to copy is an env-manager/vector-env owner that batches state-to-image
work before returning `ready_obs` and `BaseEnvTimestep` objects.

## Shape Correction

The profile sidecar and the stock LightZero env do not expose the same shape.

- The sidecar thinks in full CurvyTron rows: `[B,2]` players, both player views,
  then `[B,2,4,64,64]` stacks.
- The stock LightZero env thinks in scalar ego observations: one
  `BaseEnvTimestep` per env id, one `observation` stack, one action mask.
- The current wrapper creates an underlying `VectorMultiplayerEnv` with
  `batch_size=1`, then chooses one ego view before returning to LightZero.

That means the first real integration should not pretend `[B,2,4,64,64]` drops
directly into stock `train_muzero`. The safe path is:

1. First prove a repo-owned vector facade that owns `[B,2]` Curvy rows and
   produces correct stacks, masks, resets, and terminal final observations.
2. Then materialize scalar-compatible LightZero timestep objects from that
   facade, with one explicit mapping from env id to row/player view.
3. Only after that, try a DI-engine env-manager canary.

## Candidate Integration Shapes

1. Keep scalar `jax_gpu` as diagnostic only

   Leave `policy_observation_backend=jax_gpu` out of the default path. Use it
   only to debug renderer semantics, capacity sensitivity, and CUDA/JAX setup.
   This is not the speedup path.

2. Parent-side batched observation adapter

   Keep CPU env stepping mostly as-is. After a batch of env rows steps, pack
   compact source state in the parent process, call the batched GPU renderer for
   both player views, convert view-major output to row-major, update stacks, and
   hand trainer-shaped observations back to the collector.

   This is useful only if it sits on the env-manager/vector-env side before
   `BaseEnvTimestep` objects are handed onward. A later adapter is too late
   because `env.step` has already promised an observation.

3. Vector-env facade owns observation stacks

   Move the sidecar profile contract closer to a real vector env: env rows,
   compact state, GPU render, row-major player views, stack FIFO, partial reset,
   autoreset, and terminal `final_observation` all live behind one facade.

   This better matches the desired batched boundary, but touches more trainer
   plumbing and must be gated carefully.

4. GPU observation owner process/service

   Keep fixed device buffers and submit render jobs from the collector or env
   manager. This can reduce repeated setup and may eventually feed device
   tensors to policy forward without unnecessary host churn.

   This is the likely long-game shape, but it is too large for the first slice.

5. Device-resident policy observation pipeline

   Render, stack, normalize, and policy-forward on device. Host receives only
   actions, metadata, and replay payloads that truly need to cross the boundary.

   This is the end-state worth designing toward, not the first trainer patch.

## Recommended First Slice

Build the smallest vector-facade canary around the existing batched boundary,
without changing defaults. Do not start by claiming stock `train_muzero`
integration.

Scope:

- Add a new explicit backend name for the batched candidate, separate from the
  scalar `jax_gpu` name.
- Keep `cpu_oracle` as the production default.
- Start with a repo-owned vector facade: step CPU source-state env rows, pack
  the batch, render both views on GPU, update stacks, and expose a precise
  row/player-view API.
- Then add a scalar timestep materialization layer that LightZero can consume:
  env id -> row id -> controlled player -> observation stack/action mask.
- Preserve the observed sidecar contract:
  `browser_lines`, `simple_symbols`, both player views,
  owner-ordered compact input, `float32` aggressive mode, `float64` exact mode.
- Produce the trainer-facing shape without ambiguity:
  raw frames `[B,2,1,64,64]`, stacks `[B,2,4,64,64]`, row-major player order.
- Carry terminal `final_observation` before reset/autoreset.
- Keep a kill switch that falls back to `cpu_oracle`.

Minimal implementation slice:

- One opt-in vector-env facade, behind a new batched backend name.
- CPU stepping stays unchanged; only observation construction moves behind the
  batched boundary.
- Compare it against the existing CPU `SourceStateGray64Stack4` path on reset,
  normal step, terminal step, both seats, and partial row autoreset.
- Only after the facade passes, wrap it in a single-process env-manager canary.
- Subprocess is a separate later proof that tests CUDA/JAX initialization, IPC
  payload size, and row identity.
- Timings must include pack, H2D, render, readback, row-major conversion, stack,
  reset/final observation, timestep construction, IPC, policy forward, and
  replay write.

First slice success is not "training score improved." First slice success is:
one isolated vector-facade canary can step batches with the batched observation
path, match CPU-oracle observation semantics on checked rows, and report honest
timing for the whole boundary. The next success is a single-process
LightZero-shaped canary that consumes that facade.

## Risks/Gates

Hard gates before promotion:

- CPU oracle parity for reset, normal step, both player views, and stack FIFO.
- GPU parity against CPU oracle on real rollouts and adversarial fixtures.
- Controlled-player perspective must be per row, never hidden in a global
  renderer closure.
- Owner ordering must remain compact and semantic, not adjacent-slot trail
  connection.
- `final_observation` must be captured before reset/autoreset row reuse.
- Action masks, observation dtype, stack order, and player-view order must match
  what checkpoint policies already expect.
- No hidden CPU renderer fallback in the candidate timing path.
- No trainer default change until a full-loop canary clears correctness and
  timing gates.

Main risks:

- Seat/perspective drift: each row must get the correct controlled-player view
  and action mask; no global renderer closure may choose the player.
- Shape drift: the internal facade can be `[B,2,4,64,64]`, but stock LightZero
  still needs one ego stack per env id. The mapping must be explicit.
- Terminal/autoreset drift: `final_observation` must be copied before row reuse,
  and reset stacks must match LightZero's timestep contract.
- Subprocess CUDA/IPC cost or failure: JAX/CUDA initialization, device buffers,
  process start method, and serialized timestep payloads can erase the win.
- Metadata/tournament identity drift: backend, surface, seat, and checkpoint
  identity must remain explicit so lab GPU rows do not leak into ratings.
- Host readback/synchronization: copying frames back to NumPy may dominate until
  policy/replay boundaries can consume a better memory shape.
- The boundary win disappears once policy forward, replay write, and remaining
  collector work are included.
- Float32 geometry can produce tiny luma differences. That is acceptable only
  for the aggressive learned-policy candidate, not for exact parity gates.
- Long trail capacity can make render time dominate again unless active-prefix
  compaction, bucketing, or kernel work improves.
- Reset/autoreset behavior can look correct in a sidecar and still drift under
  trainer row reuse.
- A new backend flag could be mistaken for permission to flip defaults.

## Tests/Profiles

Plan-only: no tests or profiles were run for this document.

Needed before integration:

- CPU facade tests for reset, one step, stack FIFO, terminal
  `final_observation`, partial reset, and both-player row-major ordering.
- GPU parity tests on fixed adversarial fixtures:
  interleaved owners, inactive holes, visual cursor wrap, `break_before`, color
  ownership, bonuses, near-edge lines, and terminal rows.
- Float64 exact-reference profile rows for parity diagnosis.
- Float32 aggressive-candidate profile rows for the likely fast default.
- Trainer canary profile with separate timings for packing, host-to-device,
  device render, device-to-host, row-major conversion, stack update, reset,
  final observation, policy forward, and replay write.
- Full-loop comparison against `cpu_oracle` with identical run shape and no
  default change.

Promotion profile readout should report median and p95. It should keep compile
and warmup separate from steady-state timings.

## Non-goals

- Do not edit runtime code as part of this planning note.
- Do not run live training, tournaments, eval, Modal jobs, or launchers.
- Do not make scalar `policy_observation_backend=jax_gpu` the default.
- Do not treat the profile-only sidecar as already integrated.
- Do not change the policy observation contract.
- Do not switch from `browser_lines + simple_symbols`.
- Do not accept body-circle, browser-sprite, adjacent-slot, or wrong-player-view
  fallbacks.
- Do not optimize training score in the first slice; first prove the boundary.
