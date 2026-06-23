# Real-Consumer Canary Plan

Date: 2026-05-21

Status: implemented first profile-only gate. Do not launch live training,
resume a trusted run, promote checkpoints, write tournament/eval artifacts, or
change trainer defaults for this gate.

Implementation update, 2026-05-21:

- Flag: `--hybrid-lightzero-collect-forward-probe`.
- Code path:
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
- It builds a scratch CurvyTron MuZero policy through
  `lightzero_config_builder.build_visual_survival_configs`, then directly
  instantiates `lzero.policy.muzero.MuZeroPolicy`.
- It consumes the pre-scalar `[B,2,4,64,64]` stack, flattens it to
  `[B*2,4,64,64]`, uses real `[B*2,3]` action masks, passes player ids as
  `to_play`, and decodes every root.
- First L4/T4 Modal smoke passed with `calls_train_muzero=false`,
  `materialized_timestep_count=0`, `lightzero_policy_device=cuda:0`, and zero
  illegal actions.
- The profile image pins `torch==2.8.0` because older/newer floating Torch
  wheels conflicted with the JAX CUDA12/cuDNN runtime.

## Plain Read

The current resident chunk canary is useful, but it is still synthetic. It keeps
`[B,2,4,64,64]` `uint8` observations on GPU through replay/search-shaped work,
and the medium B512/A16/sim8 rows are strong:

| compute | scalar edge | roots/sec |
| --- | --- | ---: |
| H100 | off | `~10.98k` |
| H100 | on | `~7.62k` |
| L4/T4 | off | `~5.84k` |
| L4/T4 | on | `~4.13k` |

That proves resident batching has headroom. It does not prove that real
LightZero policy/search can consume the batch before the path collapses back to
CPU NumPy objects.

The smallest next canary should therefore consume the pre-scalar batched stack
with the real LightZero collect-policy API, while staying outside
`train_muzero`.

## 1. First Real Object To Hook Or Mimic

Hook the public collect-policy boundary first:

```python
policy.collect_mode.forward(
    obs_tensor,
    action_mask=action_mask,
    temperature=temperature,
    to_play=to_play,
    epsilon=epsilon,
    ready_env_id=ready_env_id,
)
```

This is the practical API used by existing CurvyTron eval/tournament/smoke code
for LightZero action selection. It enters `MuZeroPolicy._forward_collect`, runs
the model's initial inference, and then runs LightZero's collect search path.

Do not hook `MuZeroCollector.collect` first. That path immediately rebuilds a
batch from scalar env-id dictionaries and scalar `GameSegment` objects, which is
exactly the boundary we are trying to test around.

Do not hook LightZero's private C++/Python MCTS classes first. A private tree
hook would be more invasive, less stable across installed LightZero versions,
and still would not solve the known `detach().cpu().numpy()` conversion of
latent roots/logits before MCTS.

The canary should mimic only the collector-to-policy handoff:

```text
pre-scalar [B,P,4,64,64] uint8 stack
-> flatten active roots to [N,4,64,64]
-> torch float32 normalized tensor on policy device
-> MuZeroPolicy.collect_mode.forward(...)
-> decode actions/visit distributions/root values
-> discard or optionally store actions for a later profile-only feedback gate
```

## 2. Can We Call Actual Policy/Search Without A Full Trainer?

Yes, with an important caveat.

We can call actual `MuZeroPolicy.collect_mode.forward` in profile-only mode
without a full trainer run. The repo already has direct call sites that build a
Torch observation batch, pass action masks, `to_play`, `epsilon`,
`temperature`, and `ready_env_id`, then decode the returned actions and MCTS
metadata. This does not require `train_muzero`, `MuZeroCollector`, replay,
target construction, learner updates, checkpoint writes, eval, GIFs, or live
run state.

The caveat: this is real LightZero collect forward/search, but it is not
device-resident MCTS. In LightZero v0.2.0, collect forward can run initial
inference on the model device, then converts `pred_values`,
`latent_state_roots`, and `policy_logits` to CPU NumPy/list before MCTS. That
CPU boundary is the point of the canary, not something to hide.

So the first gate should be named:

```text
lightzero_collect_forward_consumer
```

not:

```text
resident_lightzero_mcts
```

If actual LightZero policy construction is flaky in the profile image, the
fallback falsifier is a two-row probe:

1. actual policy model `initial_inference` over the same pre-scalar batch;
2. the existing synthetic resident replay/search probe over the same batch.

That fallback is still closer than arbitrary conv loops because it uses the real
MuZero model contract, but it is not a substitute for the collect-forward row.

## 3. Required Inputs And Outputs

Inputs from the profile-only hybrid/boundary path:

| name | shape/type | notes |
| --- | --- | --- |
| `observation` | `uint8[B,P,4,64,64]` | Pre-scalar stack, preferably contiguous. `P=2` for current CurvyTron rows. |
| `action_mask` | `bool[B,P,3]` or `bool[N,3]` | Must be the real merged actor mask, not an all-true replacement. |
| `active_root_mask` | `bool[B,P]`, optional first pass | First no-death row can use all roots. Normal-death row must filter inactive roots. |
| `policy_env_row` | `int32[N]` | Row index for decoding/checks. |
| `policy_player` | `int16[N]` | Player/view index for decoding/checks. |
| `ready_env_id` | `int64[N]` | Prefer dense `0..N-1`; this is the key LightZero uses in output maps. |
| `to_play` | `list[int]` length `N` | For fixed-opponent single-agent policy, use `-1` for all roots. For two-seat/profile policy checks, use `policy_player` only if the policy config expects player ids. |
| `temperature` | `float` | Use the same collect setting as the comparable stock profile row. |
| `epsilon` | `float` | Use the same collect exploration setting as the comparable stock profile row. |

Tensor conversion contract:

```text
flat_u8:      [N,4,64,64] uint8
obs_tensor:   [N,4,64,64] torch.float32 on policy device, normalized to [0,1]
action_mask: [N,3] float32 or bool NumPy array, matching existing call sites
```

Outputs required from the canary:

| name | shape/type | notes |
| --- | --- | --- |
| `actions` | `int64[N]` | Decoded from LightZero output. Must satisfy `0 <= action < 3` and legal mask. |
| `visit_count_distribution` | `float32[N,3]`, optional | Use when present; otherwise report missing. |
| `root_value` | `float32[N]`, optional | Prefer `searched_value`, then `predicted_value`, then `value`. |
| `output_key_sample` | list/string | Compact evidence of actual LightZero output keys. |
| `action_checksum` | scalar | Cheap readback proof that outputs were consumed. |
| `illegal_action_count` | int | Must be zero for a pass row. |
| `lightzero_policy_forward_calls` | int | Measured calls, excluding warmup. |
| `lightzero_root_count` | int | Sum of roots passed to collect forward. |
| `lightzero_roots_per_call` | int | Should equal `B*P` in no-death rows. |

Timing and byte fields:

```text
lightzero_consumer_total_sec
lightzero_consumer_tensor_prepare_sec
lightzero_consumer_h2d_sec
lightzero_consumer_normalize_sec
lightzero_consumer_collect_forward_sec
lightzero_consumer_decode_sec
lightzero_consumer_readback_sec
lightzero_consumer_input_bytes
lightzero_consumer_output_bytes
lightzero_consumer_policy_device
lightzero_consumer_policy_class
lightzero_consumer_policy_surface
lightzero_consumer_num_simulations
```

Required identity flags:

```text
profile_only=true
calls_train_muzero=false
touches_live_runs=false
stock_lightzero_integrated=false
consumer_semantics=lightzero_collect_forward_search_cpu_tree
materialize_scalar_timestep=false
materialized_timestep_count=0
```

## 4. Profile-Only Implementation Steps

Keep edits inside profile-only surfaces:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`, only if
  the probe protocol needs optional action/checksum fields
- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
- profile-only tests for the above

Implementation sequence:

1. Add a `LightZeroCollectForwardStackProbe` in the Modal boundary profile file.
   It should implement the existing `HybridBatchedStackProbe` protocol if
   possible, so the training-side profile loop remains a telemetry-only
   injection point.

2. Construct or load a profile-only policy inside the Modal profile entrypoint.
   Prefer an existing CurvyTron LightZero policy/config loader helper if it can
   be reused without touching live run state. If no checkpoint is provided, use
   a scratch/random initialized policy and label it as such; speed is the gate,
   not policy quality.

3. Convert the pre-scalar `uint8[B,P,4,64,64]` stack to flat roots. Normalize on
   the policy device. Do not call `materialize_lightzero_scalar_timestep` for
   the primary row.

4. Call `policy.collect_mode.forward` for measured steps after warmup. Force
   synchronization around timing on CUDA. Decode actions, root values, and visit
   distributions with the same permissive helpers used by existing smoke/eval
   code.

5. Add flags for paired rows:

```text
--hybrid-lightzero-collect-forward-probe
--hybrid-lightzero-policy-checkpoint-ref <optional scratch/ref>
--hybrid-lightzero-consumer-num-simulations 8
--hybrid-lightzero-consumer-temperature 1.0
--hybrid-lightzero-consumer-epsilon 0.0
--hybrid-lightzero-consumer-action-feedback false
```

6. Keep action feedback off for the first row. The first falsifier is whether
   real collect forward/search can consume the batch without scalar
   materialization. Feeding actions back into actor steps is a second gate
   because it changes loop semantics and row/player validation surface.

7. Preserve compact output fields. The result summary must carry consumer
   identity, shapes, call counts, roots, timing splits, byte counts, and the
   known CPU-MCTS boundary label. A row without those fields should fail closed.

## 5. Tests And Pass/Fail Gates

Local unit tests:

- Fake-policy probe test: a dummy `collect_mode.forward` records `obs_tensor`,
  `action_mask`, `to_play`, `ready_env_id`, `temperature`, and `epsilon`, then
  returns deterministic per-root actions. Assert shapes, dtype normalization,
  legal-action checks, and telemetry.
- No-scalarization test: with the LightZero consumer enabled and scalar edge
  disabled, assert `materialized_timestep_count == 0`,
  `last_flat_obs_shape == [0,4,64,64]`, and profile flags remain false for
  trainer/live-run integration.
- Illegal-action fail test: if decoded actions violate `action_mask`, the
  canary reports failure and does not quietly count the row as valid.
- Grid-builder test: ensure new flags are emitted only for profile-only hybrid
  rows, and cannot be combined with incompatible scalar-only policy-search
  probes unless explicitly requested for an A/B.

Remote/profile smoke:

- Tiny H100 or L4/T4 row with `B=16`, `A=2`, `sim=1-2`, warmup enabled, no
  scalar materialization, and no live run writes.
- Assert LightZero import/policy creation works in the profile image.
- Assert output includes `consumer_semantics=lightzero_collect_forward_search_cpu_tree`.

First real comparison rows:

| row | consumer | scalar edge | purpose |
| --- | --- | --- | --- |
| A | existing synthetic resident replay/search | off | current anchor |
| B | real `MuZeroPolicy.collect_mode.forward` | off | smallest real-consumer gate |
| C | real `MuZeroPolicy.collect_mode.forward` | on | price stock-like scalar edge after real search |
| D | model `initial_inference` only, optional | off | isolate model forward from CPU tree search |

Pass gates:

- no `train_muzero` call, no live-run writes, no checkpoint/eval/tournament
  side effects;
- primary row has `materialize_scalar_timestep=false` and zero materialized
  scalar timesteps;
- `lightzero_root_count == measured_steps * B * P` for no-death rows;
- decoded actions are legal and output checksum changes across non-identical
  observations or seeds;
- compile/warmup is excluded from measured timing;
- H100 B512/A16/sim8 real collect-forward scalar-off row stays materially above
  the scalar-on synthetic range, or the timing clearly identifies MCTS CPU tree
  work as the collapse;
- collect-forward timing is large enough to prove the real consumer ran, not an
  optimized-away placeholder.

Fail gates:

- throughput collapses to scalar-on or stock full-loop range before scalar
  materialization is enabled;
- a hidden `materialize_lightzero_scalar_timestep` or float32 NumPy row path
  appears in the primary row;
- `action_mask` is all-true by accident, shape-mismatched, or ignored;
- output cannot be tied back to `ready_env_id`, row, and player;
- LightZero collect forward silently falls back to CPU model execution when the
  requested row is GPU;
- compact results omit identity/timing/shape fields needed to interpret the row.

## Critical Verdict

Real MCTS as a device-resident resident-stack consumer is too invasive for this
next gate. LightZero's current collect path intentionally converts latent roots
and logits back to CPU objects for MCTS. Removing that boundary means changing
LightZero policy/search internals or replacing search, which is an architecture
prototype, not a canary.

The next closest honest falsifier is to call the real public
`MuZeroPolicy.collect_mode.forward` from the pre-scalar batched stack and label
the CPU-tree boundary explicitly. If that row is still fast enough, resident
batching has survived contact with the real policy/search consumer. If it
collapses, we learn where: tensor conversion, model initial inference, CPU MCTS
tree/search, output decode, or scalar materialization.
