# Current Hot Path Bottleneck Map, 2026-05-22

Scope: current stock LightZero train/profile path and the promoted `collect_search_backend=direct_ctree_gpu_latent` path, including env manager, observation render/stack, policy initial inference, LightZero CTree/recurrent search, replay, learner, and RND.

This is a boundary critique, not a change plan for the existing live run. No live runs were touched for this audit.

## Executive Read

The current `direct_ctree_gpu_latent` work removed a real boundary, but not the dominant topology. It keeps LightZero's stock collector/search contract: Python env ids, dict observations, `BaseEnvTimestep` objects, CTree root objects, Python list-shaped traverse/backprop APIs, per-simulation Python control, CPU policy/value/reward arrays, replay segments, learner batches, and optional RND CPU extraction/hashing.

That makes the observed `~1.28-1.31x` full-loop speedup plausible rather than surprising. The direct path improved the search slice, and the output fast path removed most action-output assembly overhead, but the denominator still includes:

- scalar/object env-manager materialization,
- host observation stacks,
- initial/root CPU preparation,
- CTree CPU/list APIs,
- recurrent-output GPU-to-CPU transfers every simulation,
- replay/learner/RND stages,
- and profiler/summary synchronization when enabled.

The boundary that is still hot after `direct_ctree_gpu_latent` is therefore not "rendering" and not only "latent copy". It is the LightZero collect/search topology boundary:

```text
GPU model tensors
  -> CPU NumPy/list root + CTree contracts
  -> Python simulation loop
  -> GPU recurrent inference
  -> CPU NumPy/list reward/value/policy backprop
  -> Python dict action output
  -> scalar env manager/replay objects
```

A 5-10x move needs to remove or bypass that contract, not polish it.

## Measured Denominator Context

Current matched no-RND full loop:

- Stock C64/sim16/3 learner: `433.17 steps/sec`, wall `37.82s`, collect `26.02s`, policy collect `17.10s`, MCTS `12.09s`, learner `4.17s`.
- Direct output-fast: `566.19 steps/sec`, wall `28.94s`, collect `19.41s`, policy collect `10.31s`, MCTS `8.06s`, learner `1.03s`, direct D2H `2.47s`, direct output assembly `0.077s`, fast-path calls `256`, fallback `0`.
- Full-loop gain: `~1.31x`.

Current matched RND hash-fixed full loop:

- Stock: `351.02 steps/sec`, wall `46.68s`, policy collect `23.62s`, MCTS `17.30s`, RND train `0.590s`, RND hash `0.131s`, RND estimate `0.086s`.
- Direct: `448.52 steps/sec`, wall `36.53s`, policy collect `13.32s`, MCTS `10.66s`, RND train `0.603s`, RND hash `0.140s`, RND estimate `0.093s`.
- Full-loop gain: `~1.28x`.

Older observation-denominator checks also point away from rendering as the current main wall: the C512 real-observation row was `~1439.84 steps/s`, while zero observation was `~1805.22 steps/s`, only `~1.25x` from deleting observation work in that setup. Profile-only resident uint8 stack/search measurements can reach much higher synthetic rates, but the stock train path must still pay the LightZero object and CPU boundary.

## Current Stock/Profile Dataflow

The train entrypoint is `_run_visual_survival_train`, which builds LightZero configs and calls `lzero.entry.train_muzero` with patched instrumentation and optional backend hooks. See `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py` around:

- config and entrypoint setup: lines `5126`, `5802-5855`, `6189-6200`,
- collect-search hook install: lines `1977-2260`,
- batched profile env-manager hook install: lines `2265-2768`,
- final summary/semantic attestation: lines `13159-13503`.

The full path is:

```text
_run_visual_survival_train
  -> build_lightzero_config / build create_config
  -> train_muzero([main_config, create_config], ...)
  -> create_env_manager(...)
  -> Collector.collect(...)
  -> policy.collect_mode.forward / _forward_collect(...)
  -> model.initial_inference(...)
  -> roots.prepare(...)
  -> MCTS.search or direct_ctree_gpu_latent hook
  -> recurrent_inference for each simulation batch
  -> selected action dict per env_id
  -> env_manager.step(action_dict)
  -> trajectory/game segment materialization
  -> replay buffer push/sample
  -> learner.train(...)
  -> optional RND collect_data/train_with_data/estimate
```

The LightZero config keeps the stock policy contract alive: CUDA policy is enabled, collector env count and simulations are configured, observations are `4 x 64 x 64`, and action space size is `3`. See `src/curvyzero/training/lightzero_config_builder.py` lines `1220-1360`, especially env manager selection, `collector_env_num`, `num_simulations`, `batch_size`, `image_channel=4`, `observation_shape`, `action_space_size`, and observation backend/contract fields.

## Stage-by-Stage Boundary Map

| Stage | Code references | Host/device syncs | Python/object fanout | Boundary critique |
| --- | --- | --- | --- | --- |
| Profile/timer wrappers | `lightzero_curvyzero_stacked_debug_visual_survival_train.py` lines `816-900`, `1041-1811` | `_PhaseTimer` and `_LightZeroPhaseProfiler` can call `torch.cuda.synchronize()` for CUDA-timed model, policy, MCTS, replay, learner, and RND sections. | Monkey-patched wrappers preserve stock LightZero call topology. | These syncs are measurement overhead when enabled, not the production algorithm, but they expose where async GPU work is forced visible. Treat profile timings as synchronized slice timings. |
| Stock scalar env | `curvyzero_source_state_visual_survival_lightzero_env.py` lines `690-947`, `1227-1372`, `2678-2698` | CPU path. GPU/JAX render backend can force device-to-host inside observation update. | One env object per scalar LightZero env; observation dict per step; `BaseEnvTimestep`; copied action masks; per-step info dict/list; optional pickling for profile bytes. | Even with a batched substrate, the stock surface must be re-scalarized for LightZero. This is a durable Python denominator. |
| Scalar observation stack | `curvyzero_source_state_visual_survival_lightzero_env.py` lines `631-632`, `1256-1356` | If scalar JAX GPU renderer is used, render output returns to host; stack lives as `np.float32`. | `stack.copy()`, `action_mask.copy()`, stack roll/copy, opponent stack, frozen-opponent observation object. | Rendering was once hot, but the active full-loop ceiling is now mostly after this point. Stack is still host float32 and copied into LightZero. |
| Batched profile surface | `multiplayer_source_state_trainer_surface.py` lines `349-423`, `529-629` | Renderer-backed stack may synchronize before host policy observation exists. `_surface_step` itself is NumPy/host materialization. | Copies legal masks, done flags, reward arrays, final observations, info rows; creates `MultiplayerTrainerStepV0`. | The profile manager batches env physics/render, then immediately has to expose stock-shaped host arrays and row/player metadata. |
| Batched scalar bridge | `source_state_batched_observation_mock_collector.py` lines `220-370`, `819-919`, `930-1023`, `1069-1139` | CPU path after surface materialization. | Sorts env ids, maps action dict to joint-action arrays, materializes policy/env-id timesteps, splits output into one `MockBaseEnvTimestep` per env, copies observation and action mask per env. | This is the main "batched underneath, scalar above" boundary. It makes LightZero compatibility possible, but it prevents a large architecture-level speedup. |
| JAX batched observation renderer | `source_state_batched_observation_boundary_profile.py` lines `2597-2673`, `3220-3285` | `_copy_state_to_device` H2D, `output_device.block_until_ready()`, `np.asarray(output_device)` D2H. | State is packed owner-ordered, reshaped to row/player order, then copied into host output arrays. | This boundary is explicit and measurable. It is no longer enough to explain the full-loop wall alone, but any resident-stack design must remove this D2H from the actor hot path. |
| Policy initial inference and root prep | `lightzero_curvyzero_stacked_debug_visual_survival_train.py` lines `2015-2171` | `model.initial_inference` runs on GPU. Root prep does `pred_values.detach().cpu().numpy()` and `policy_logits.detach().cpu().numpy().tolist()`. | Per-env action-mask normalization, Python list masks, root noises, root value/policy lists, LightZero root objects. | Direct search starts after a CPU root contract has already been paid. This is a key reason direct latent search cannot deliver synthetic-sidecar speedups in stock train. |
| Direct CTree GPU latent search | `lightzero_curvyzero_stacked_debug_visual_survival_train.py` lines `1833-1974`; sidecar mirror in `source_state_batched_observation_boundary_profile.py` lines `5575-5724` | Per simulation: Python/NumPy traverse output to CUDA index/action tensors, recurrent inference on GPU, then `reward/value/policy_logits.detach().cpu().numpy()`. Sidecar also has explicit CUDA syncs around action tensor, recurrent, and D2H buckets. | `tree_muzero.batch_traverse` returns Python list-shaped indices/actions; `batch_backpropagate` consumes listified reward/value/policy; loop is Python over `num_simulations`; CTree owns CPU tree state. | This is the current hottest remaining boundary. Direct latent avoids CPU latent encode/decode, but CTree still requires CPU scalar arrays and list APIs each simulation. |
| Action output assembly | `lightzero_curvyzero_stacked_debug_visual_survival_train.py` lines `2198-2247` | CPU only after search. | `roots.get_distributions()`, `roots.get_values()`, dict output per `env_id`, per-env fallback `select_action` if not all legal. | The output fast path mostly solved this local fanout: current output assembly is only `~0.077s` in the no-RND direct matched run. This is no longer the next big target. |
| Replay and learner | `lightzero_curvyzero_stacked_debug_visual_survival_train.py` lines `1671-1686`, `1805-1811`; LightZero train entry via lines `6189-6200` | Learner train wrapper may synchronize for timing. Replay sample wrapper can synchronize for timing. Stock learner then moves sampled host data to model device internally. | Stock game segments, replay samples, target construction, learner batch objects. | Direct collect/search does not remove replay target-building or learner-batch object topology. Full-loop speedups hit Amdahl limits once collect/search is only partially improved. |
| RND | `exploration_bonus.py` lines `88-90`, `488-543`, `669-680`, `762-871`; RND class patch in `lightzero_curvyzero_stacked_debug_visual_survival_train.py` lines `4977-5026` | `_to_numpy_cpu` uses `detach().cpu().numpy()`; state hashes use `tensor.detach().cpu().contiguous().numpy()`; train loss uses `loss.detach().cpu().item()`; estimate uses `mse.detach().cpu().numpy()` and reward stats `.cpu().item()`. | Extracts latest gray64 frames from policy obs, stores many cloned tensors in a list, deep-copies target reward, returns reward list/copies, writes metrics snapshots. | RND is not the largest measured slice after hash fixes, but it is a separate CPU/object lane that prevents a resident device pipeline. It must be redesigned before a Coach-facing direct path is considered production-safe. |

## Host/Device Sync Inventory

Intrinsic hot-path synchronization:

- `model_output.reward.detach().cpu().numpy()`, `value.detach().cpu().numpy()`, and `policy_logits.detach().cpu().numpy()` inside direct recurrent search. This is paid every simulation batch before CTree backprop can proceed.
- `pred_values.detach().cpu().numpy()` and `policy_logits.detach().cpu().numpy().tolist()` during root preparation.
- JAX renderer `block_until_ready()` and `np.asarray(output_device)` when GPU-rendered observations are materialized as host arrays.
- RND CPU conversions and scalar reads: `_to_numpy_cpu`, tensor state hashing, `loss.detach().cpu().item()`, `mse.detach().cpu().numpy()`, and stats `.cpu().item()`.
- H2D conversions from host arrays back into CUDA tensors for policy/recurrent/RND inputs, including direct-search `last_actions` and hidden-state index tensors each simulation.

Profiling synchronization:

- `_PhaseTimer`/`_LightZeroPhaseProfiler` can synchronize CUDA before and after timed sections. The wrappers are installed around model inference, recurrent inference, policy forward, collector collect, learner train, MCTS search, replay sample, and RND hooks. These syncs are useful for attribution but should not be confused with the product algorithm unless the profile flag is left enabled.

Host-only scalarization and object fanout:

- `BaseEnvTimestep` and `MockBaseEnvTimestep` per env.
- Observation dicts with `observation`, `action_mask`, `to_play`, and `timestep`.
- Per-env action dict in, per-env timestep dict out.
- `np.asarray(...).item()` scalar reward/done conversions.
- Info dict/list per row/player.
- `roots.get_distributions()`, `roots.get_values()`, and action dict output.
- RND sample lists and deep-copied rewards.

## Why Previous Changes Stayed Small

The earlier fixes were locally correct, but they attacked boundaries whose whole-loop share had already shrunk.

1. Observation/render optimizations improved an older bottleneck. The whole-loop zero-observation comparison now suggests deleting observation work is only about a `~1.25x` ceiling in the relevant row.
2. `direct_ctree_gpu_latent` removed the latent roundtrip and public wrapper overhead, but left the CTree CPU/list contract intact.
3. The output fast path removed most per-env output assembly. The direct no-RND run shows only `~0.077s` there, so more polishing cannot move the denominator much.
4. The full loop still spends `19.41s` in collect and `10.31s` in policy collect on the direct no-RND run, with `8.06s` MCTS and `2.47s` direct D2H. The absolute wall gain was `8.88s` (`37.82s -> 28.94s`), which matches "partial improvement to a large slice" rather than an end-to-end architecture shift.
5. RND and learner/replay remain independent denominators. In RND runs, direct search improves policy/MCTS time but RND train/hash/estimate stays roughly flat.

So `~1.3x` is not a failure of the direct patch. It is the expected ceiling for a change that accelerates the model/search subpath while retaining the stock LightZero collect, tree, replay, and RND contracts.

## P0 Validation Gaps Before Coach-Facing Direct Use

These are production gates, not performance wishes.

1. Semantic parity of direct search versus stock CTree over fixed seeds, including action distributions, selected actions, root values, visit counts, legal-mask handling, Dirichlet noise, temperature, and terminal/truncated rows.
2. Full all-legal and masked-action validation. The fast output path is only safe when every row is genuinely legal-mask-compatible; fallback behavior must be parity-tested.
3. Recurrent-state indexing parity. `batch_traverse` path indices, latent pool writes, and root/batch row order must be proven stable across sim counts, env counts, done/autoreset, and player ordering.
4. RND interaction parity. Direct collect cannot change reward-shaping inputs, latest-frame extraction, target reward mutation semantics, RND metrics, or hash behavior in Coach-facing runs.
5. Replay/learner target parity. Game segment fields, search policy distributions, value prefixes, rewards, dones, to-play, and action histories must match stock expectations.
6. Determinism envelope. Seeded stock/direct matched runs need a documented tolerance model because GPU order, CPU list ordering, and exploration noise can otherwise mask semantic drift.
7. Profiling-mode isolation. Any claim used for promotion needs both synchronized attribution and an unsynchronized throughput run so measurement syncs are not accidentally promoted as algorithmic cost.

## Radical Optimization Candidates

### 1. Array-Native Collect ABI

Replace the stock env-manager surface for the optimized lane with a compact batch ABI:

```text
obs_uint8_or_float[batch, 4, 64, 64]
legal_mask[batch, 3]
reward[batch]
done[batch]
to_play[batch]
env_id[batch]
```

The bridge to stock `BaseEnvTimestep` should become a debug/compat adapter, not the hot path. This removes the repeated dict/timestep/info fanout in `source_state_batched_observation_mock_collector.py` and lets policy input batching become a direct tensor handoff.

Fast falsifier: add a profile-only collector stub that consumes a pre-materialized compact batch and bypasses `_split_timestep_by_env_id`. If throughput does not move materially, the main wall is deeper in CTree/replay/learner.

### 2. Array-Native Fixed-Action CTree

Introduce an optimized tree API for fixed action space `A=3`:

```text
prepare(values[batch], policies[batch,3], legal_mask[batch,3])
traverse() -> path_index[batch], last_action[batch]
backprop(reward[batch], value[batch], policy[batch,3])
output() -> action[batch], visit_counts[batch,3], root_value[batch]
```

This attacks the current hottest boundary: Python list-shaped `batch_traverse`/`batch_backpropagate`, root objects, policy logits `.tolist()`, and per-simulation CPU reward/value/policy arrays. It can be a C++/CUDA extension, a specialized CPU vector CTree, or a GPU-resident tree. The important part is that recurrent outputs no longer have to be materialized as Python lists every simulation.

Fast falsifier: build a no-model CTree microbench that feeds synthetic `reward/value/policy[batch,3]` arrays through the current CTree API and then through a flat-array prototype. If list/root overhead is not dominant there, prioritize recurrent/model batching instead.

### 3. GPU-Resident Search Service

Move the MCTS loop into a service that owns tree state and latent state, with only batched tensor calls at the boundary. The search service should keep latent pool, visit counts, priors, values, rewards, and actions resident and call recurrent inference without returning policy/value/reward to CPU each simulation.

This can be staged as:

1. CPU flat-array CTree, to remove Python objects first.
2. GPU-resident CTree or a CUDA/Triton/JAX implementation.
3. CUDA graph or compiled recurrent call for stable sim/batch shapes.

Fast falsifier: run sidecar direct search with recurrent inference replaced by precomputed resident tensors. If throughput is still far below synthetic CTree targets, tree/list topology is the limiting factor. If it jumps, recurrent launch/batching dominates.

### 4. Replay/Target Array Writer

Direct collect/search still eventually becomes stock game segments. A radical lane needs array-native replay writes:

```text
obs/action/reward/done/to_play/search_policy/root_value/action_mask
```

stored as contiguous batches, with target construction operating over arrays instead of Python game objects. This is required for a 5-10x full-loop result because otherwise the system just moves the bottleneck from collect/search to replay and learner input construction.

Fast falsifier: disable learner and replay writes separately in a matched dry profile, or replace replay push/sample with fixed prebuilt tensors, then compare direct collect ceiling. This should be done only in non-live profile jobs.

### 5. RND Resident Latest-Frame Path

RND currently re-extracts CPU latest gray64 frames, clones tensors into lists, hashes model tensors on CPU, deep-copies target reward, and returns CPU reward arrays/lists. A resident path should consume the same compact observation batch used by policy, keep latest-frame extraction on device, sample/train from a tensor ring buffer, and move metrics/hashes out of the hot cadence.

Fast falsifier: run matched RND profile with metrics hashing disabled and a pre-extracted latest-frame tensor cache. If speed barely changes, RND is not the next wall; if it moves, RND needs a resident redesign before Coach promotion.

### 6. Actor/Search Topology Split

If the goal is 5-10x rather than 1.5x, consider a topology split:

```text
batched env actors -> compact tensor queue -> central batched search/model service -> compact replay writer -> learner
```

The current stock LightZero path interleaves scalar env stepping, policy dict output, CTree Python APIs, and replay object writes in one synchronous loop. A service topology would let multiple actor batches feed one saturated GPU search/model service and keep replay writes array-native.

Fast falsifier: create a profile-only producer/consumer mock where env batches feed a fixed-shape search worker. If the worker can saturate H100-like throughput with synthetic tensors while the current integrated loop cannot, the integration topology is the ceiling.

## Fastest Falsifier Experiments, In Order

1. **No-model direct CTree API split.** Feed synthetic recurrent outputs through current CTree `batch_traverse`/`batch_backpropagate` and measure Python list/root overhead separately from model time.
2. **Precomputed recurrent-output direct search.** Keep the direct hook shape but replace recurrent inference output with resident tensors to separate recurrent launch/model cost from CPU CTree/list/D2H cost.
3. **Compact collector bypass.** In a profile-only lane, bypass `_ready_obs_by_env_id` and `_split_timestep_by_env_id` and feed compact arrays directly to the policy input builder.
4. **Replay/learner denominator ablation.** Profile direct collect with replay writes and learner train separately replaced by fixed synthetic/no-op components, preserving step count and reporting explicit semantic non-equivalence.
5. **RND resident/metrics ablation.** Profile RND with CPU hashing/metrics and latest-frame extraction removed from hot cadence, again profile-only and clearly non-production.
6. **Unsynchronized throughput companion.** For every synchronized attribution profile, run an unsynchronized throughput companion to avoid optimizing profile synchronization artifacts.

2026-05-22 update on falsifier 1:

```text
scripts/benchmark_lightzero_ctree_no_model.py
```

now prices the no-model CTree list ABI directly. First local rows show CTree
list/core work around `1M-1.6M nodes/sec`, while a simple vectorized fake-flat
update is `12x-42x` faster depending on root count and simulation count.

Fresh H100 refresh:

```text
Modal run ap-9hEH4WJk4kprHGTpcEiPte
roots=512,1024; simulations=16,32; legal=all3,mixed_2of3.

ctree-list:       ~0.51M-0.94M nodes/sec
ctree-torch-d2h:  ~0.58M-0.82M nodes/sec
fake-flat:        ~16M-22.6M nodes/sec
```

Read: this supports a bounded array-native CTree spike, but it also explains
why earlier small patches stayed small. The tree/list ABI is expensive, yet the
full loop also pays recurrent inference, stock collect fanout, replay, learner,
and RND. The next serious implementation must either vendor a fixed-A=3 CTree
flat API as a measured bridge or move to a compact batched search service.

Plain read:

```text
Replacing the LightZero CTree/list ABI is worth exploring, but raw CTree alone
is not slow enough to explain the whole current full-loop wall. The next
falsifier should keep the direct-search shape but replace recurrent inference
with precomputed resident tensors. That separates recurrent GPU launch/D2H/list
cost from CTree tree/list cost before we commit to a deeper tree rewrite.
```

## Concrete Boundary Claim

The exact hot boundary after `direct_ctree_gpu_latent` is:

```text
LightZero collect/search compatibility:
  host observation/timestep objects
  + GPU initial inference
  + CPU root prep
  + Python CTree traverse/backprop loop
  + per-simulation recurrent GPU output D2H
  + Python action dict/timestep fanout
  + stock replay/RND object lanes
```

`direct_ctree_gpu_latent` is valuable because it proves latent CPU roundtrip and output assembly were real costs. It is not a 5-10x architecture because it preserves the CPU/list CTree and stock collector/replay/RND contracts. The next meaningful move is to make one optimized lane array-native across collect, search, replay, and RND, with stock LightZero compatibility demoted to a validation adapter.
