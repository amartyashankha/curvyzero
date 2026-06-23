# Host Overhead Map

Date: 2026-05-20

Status: active. Fill this from the host-overhead/dataflow critique and fresh
profiles.

## Buckets To Measure

- env step physics;
- observation render;
- observation stack update;
- reset/autoreset;
- terminal `final_observation` copy;
- `BaseEnvTimestep` construction;
- Python `info` payload construction;
- subprocess serialization/IPC;
- policy forward during collection;
- MCTS/search;
- replay write and replay sample;
- learner train;
- RND collect/train/estimate;
- checkpoint/eval/GIF/artifact I/O, when intentionally included.

## Current Suspicion

After renderer work, the next full-loop bottleneck is probably not one function.
Fresh split rows narrow the most important current wall to the public
LightZero collect/search/output path after root model inference. Collection and
process topology still matter, but the newest H100 refresh says model initial
inference is around `9466` roots/sec while collect-forward sim8 is around
`2304` roots/sec on the same resident B512/A16 shape.

After the direct-surface and copy-reduction work, the profile-only B512 surface
step is about `0.123s` for `1024` policy rows. Device render is only about
`0.014s` of that row. That means the next local host/boundary questions are
payload shape, stack/pack layout, and whether LightZero can consume the vector
facade without collapsing back to scalar env workers.

Inside sampled env rows, `update_stack_sec` can still be a large measured
per-step bucket, and `BaseEnvTimestep` pickle time and payload size are visible
but not the obvious largest cost. The next practical split is:

- keep subprocess/high collector counts for real training recommendations;
- fix the RND/telemetry correctness gaps before using RND rows as speed proof;
- keep the vector/batched observation facade as the route to a real larger win.
- keep RND separate from renderer optimization. The profile-only RND cadence
  rows show CUDA helps, but `rnd_update_per_collect` dominates cost.
- split `collect_mode.forward` into model initial/recurrent calls versus
  non-model tree/output time before choosing a search rewrite or topology
  change.

## Instrumentation Gaps From Code Review

- Subprocess env-manager queue wait, scheduling, and actual IPC send/receive
  are still folded into collector wall today.
- BaseEnvTimestep construct/pickle time and payload bytes are now measured as a
  low-risk IPC proxy in registered profile env rows.
- Observation timing now splits stack update, stack copy, action-mask copy,
  `_base_info`, and terminal final-observation copy.
- Replay sample timing is coarse; target construction and tensor assembly are
  not split.
- RND `collect_data`, `estimate`, `train_with_data`, metrics snapshot/write, and
  state hash are now timed at the phase-profiler boundary.

## First Profile Ladder

Run matched rows:

```text
env_manager_type = base, subprocess
collector_env_num = 1, 8, 32
num_simulations = 8
batch_size = 32 or 64
mode = profile
eval/GIF/checkpoint I/O off
RND off
```

Read the subprocess-minus-base wall gap as the practical process/IPC/scheduling
effect. Then repeat one matched pair with `rnd_meter_v0` only after the compact
step counter is fixed and RND cadence semantics are clear enough to interpret.

## Profile Keys Added

- `base_env_timestep_construct_sec`
- `base_env_timestep_pickle_sec`
- `base_env_timestep_pickle_bytes`
- `base_env_timestep_array_bytes`
- `update_stack_sec`
- `observation_stack_copy_sec`
- `action_mask_copy_sec`
- `base_info_sec`
- `final_observation_copy_sec`
- `rnd_collect_data_sec`
- `rnd_train_with_data_sec`
- `rnd_estimate_sec`
- `rnd_metrics_snapshot_sec`
- `rnd_metrics_write_snapshot_sec`
- `rnd_state_hash_sec`

If these proxies are small but subprocess is still much faster or slower than
base, the next patch should inspect DI-engine queue/wait/send/receive directly.

## 2026-05-21 Added Split

The hybrid LightZero collect-forward probe now times model calls inside
`collect_mode.forward`:

- `lightzero_consumer_model_initial_inference_sec`
- `lightzero_consumer_model_initial_inference_calls`
- `lightzero_consumer_model_recurrent_inference_sec`
- `lightzero_consumer_model_recurrent_inference_calls`
- `lightzero_consumer_model_total_sec`
- `lightzero_consumer_collect_forward_non_model_sec`

Metric caveat: `lightzero_consumer_collect_forward_non_model_sec` means
"non-model time inside public `collect_mode.forward`." Direct CTree arrays rows
do not call public `collect_mode.forward`, so they now report
`lightzero_consumer_direct_boundary_non_model_sec` and
`lightzero_mcts_arrays_boundary_non_model_sec` instead.

This is the next Amdahl read. If model-call time is small and the non-model
residual is large, the next lane is LightZero CPU tree/output handling or a
deeper batched search boundary. If model-call time dominates, then topology,
batch size, and GPU choice matter more.

First H100 result:

```text
B512/A16/sim8, 120 measured calls
collect-forward wall: ~69.8s
timed model calls:    ~2.7s
non-model residual:   ~67s
```

Follow-up deeper split:

```text
B512/A16/sim8, H100
collect-forward wall:       35.36s
timed model calls:           1.81s
MCTS search wrapper:        10.97s
raw ctree traverse/backprop: 0.98s
outside-MCTS residual:      24.40s
```

Pure-policy collect reached about `6286` roots/sec, while MCTS collect was
about `2572` roots/sec. So the next lane is not raw ctree alone and not GPU
model inference. The wall is the MCTS branch representation path: root setup,
CPU/list conversion, wrapper/result handling, and per-root output fanout.
Hardware can still matter after this boundary changes, but H100 capacity is not
the missing piece in the public collect-forward path as currently measured.

Array-ceiling follow-up:

```text
H100 policy_arrays:    ~9958 roots/sec
H100 recurrent_toy:    ~8681 roots/sec
L4 policy_arrays:      ~5590 roots/sec
L4 recurrent_toy:      ~5030 roots/sec
```

The H100 recurrent toy still runs 8 real batched recurrent model calls, so this
strengthens the read that model-call pressure is not the wall. It also exposes
a second boundary cost: the toy spends about `2.4s` in host-stack to Torch H2D.
That is real cost in the current probe because the pre-scalar stack is a host
`uint8` array. It is probably avoidable with a resident Torch input path or a
clean pinned/dtype split, but it is not fake and should be measured before
being designed away.
