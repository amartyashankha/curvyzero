# Current Boundary Code Audit, 2026-05-22

Scope: read-only audit of the current profile/search/replay boundary. I edited
only this document and did not touch live runs.

Files inspected:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/source_state_batched_observation_mock_collector.py`
- `src/curvyzero/training/exploration_bonus.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/*.md`

## Plain Conclusion

The current wall is not "MCTS is not on GPU" by itself.

The current wall is the boundary shape:

```text
batched CurvyTron state
-> host NumPy observation/action-mask sidecars
-> Torch CUDA initial inference
-> CPU NumPy/list root prep for CTree
-> Python simulation loop
-> CUDA recurrent inference
-> CPU NumPy/list reward/value/policy for CTree backprop
-> compact arrays or stock env-id dicts
-> replay/RND object materialization
```

The direct GPU-latent CTree mode removes one important copy: it no longer
round-trips latent states to CPU for each recurrent call. It does not remove
the CPU/list CTree API, the per-simulation output readback, scalar/env-id
objects, or RND/replay materialization.

That matches the observed pattern in the docs: small direct-hook speedups in
matched full-loop rows, larger profile-only roots/sec wins, and no evidence yet
for a 5-10x full-loop change from this hook alone.

## Current Dataflow

### 1. Hybrid actor/profile surface

`source_state_hybrid_observation_profile.py` owns the profile-only batch shape.

Important pieces:

- `HybridCompactBatch` carries `[B,P,4,64,64]` observation, `[B,P,3]`
  action masks, row/player ids, reward, done, `to_play`, `done_root`,
  `active_root_mask`, terminal/autoreset/final-observation sidecars
  (`source_state_hybrid_observation_profile.py:183-209`).
- Actors step `VectorMultiplayerEnv` on CPU and copy scalar fields into either
  payload objects or parent compact buffers (`:277-345`, `:347-395`).
- The manager updates a rolling stack, optionally calls the renderer, and then
  builds `HybridCompactBatch` before scalar materialization (`:557-651`).
- If `materialize_scalar_timestep=True`, it creates `MockBaseEnvTimestep`,
  `flat_obs`, `target_reward`, and info rows (`:706-730`).
- It can pickle compact payloads for byte profiling (`:732-736`).

CPU/object materialization here:

- actor payload dataclasses;
- NumPy copies of reward/done/alive/action masks/joint actions;
- optional merged render-state dictionaries;
- optional `MockBaseEnvTimestep`;
- optional compact pickle.

### 2. Scalar LightZero bridge

`source_state_batched_observation_mock_collector.py` is the key "batched below,
scalar above" adapter.

Important pieces:

- `BatchedLightZeroScalarActionBridge.step()` sorts scalar env ids, maps the
  action dict back to joint actions, steps one batched surface, then exposes
  scalar LightZero-shaped output again (`:220-263`).
- `_joint_action_from_scalar_actions()` loops through the action dict and builds
  a `[B,P]` joint-action NumPy array (`:265-294`).
- `_output_from_loop_step()` builds `policy_env_id`, then calls
  `_ready_obs_by_env_id()` and `_split_timestep_by_env_id()` (`:296-370`).
- `materialize_lightzero_scalar_timestep()` reshapes `[B,P,4,64,64]` into
  `[B*P,4,64,64]`, repeats row-level done/reward, and builds info dicts
  (`:738-810`).
- `_ready_obs_by_env_id()` copies each observation and action mask into one
  env-id keyed dict (`:1069-1085`).
- `_split_timestep_by_env_id()` copies each observation/action-mask row again
  into one `MockBaseEnvTimestep` per env id (`:1102-1139`).

CPU/object materialization here:

- sorted env-id arrays;
- action dict to joint-action loop;
- per-env observation/action-mask `.copy()`;
- per-env `MockBaseEnvTimestep`;
- per-env `info` dicts;
- repeated reward/done/to_play arrays.

This is a real denominator if the fast lane has to remain stock-LightZero
compatible.

### 3. LightZero collect/search boundary

`_LightZeroCollectForwardStackProbe` in
`source_state_batched_observation_boundary_profile.py` is the current profile
consumer (`:4457+`).

Common setup:

- `run()` accepts `[B,P,4,64,64]` plus `[B,P,3]`, flattens to roots, drops
  zero-mask roots, and prepares tensors for the policy device (`:4533-4576`,
  `:4700-4712`).
- The stock facade calls `policy.collect_mode.forward(...)`, then decodes the
  public LightZero output back into compact arrays (`:4751-4823`).
- The direct boundary dispatches to `_run_direct_mcts_arrays()` for
  `direct_ctree_arrays`, `direct_ctree_gpu_latent`, and
  `direct_ctree_gpu_latent_precomputed_recurrent` (`:4714-4749`).

The direct MCTS arrays path:

- Runs real `model.initial_inference(obs_tensor)` on the policy device
  (`:5279-5283`).
- Unpacks root outputs (`:5285-5289`).
- Copies root values and root policy logits to CPU NumPy
  (`pred_values.detach().cpu().numpy()`, `policy_logits.detach().cpu().numpy()`)
  for LightZero root prep (`:5290-5304`).
- For non-GPU-latent direct CTree, also copies latent roots to CPU NumPy
  (`:5299-5302`).
- Listifies policy logits and builds Python `legal_actions`, Dirichlet noises,
  `roots`, and `roots.prepare(...)` (`:5311-5323`).
- Runs either stock `mcts.search(...)` or the custom GPU-latent CTree loop
  (`:5325-5348`).
- Reads root distributions and root values from CTree and assembles compact
  action/value/visit arrays (`:5378-5500`).

CPU/GPU syncs here:

- `model.initial_inference` is followed by `_sync_torch_device_if_cuda()`
  (`:5279-5283`).
- Root value/logit `.cpu().numpy()` is an intrinsic GPU-to-CPU synchronization
  (`:5290-5304`).
- CTree root prep is CPU/list-shaped (`:5311-5323`).

### 4. GPU-latent CTree loop

`_run_direct_ctree_gpu_latent_search()` is the exact current hot CTree loop
(`source_state_batched_observation_boundary_profile.py:5723-5940`).

What stays on GPU:

- `latent_pool` is allocated on `device` and root latents are copied into it
  (`:5764-5770`).
- Each simulation gathers selected latents from `latent_pool` on device
  (`:5825-5839`).
- Real mode calls `model.recurrent_inference(latent_states, last_actions_tensor)`
  on device (`:5850-5855`).
- `next_latent_state` is copied back into `latent_pool` on device (`:5893`).

What still synchronizes or materializes on CPU:

- `tree_muzero.batch_traverse(...)` is called from Python and returns Python
  list-like path indices, batch indices, actions, and virtual to-play
  (`:5805-5823`).
- `last_actions` are converted through `np.asarray(last_actions)` and then
  `torch.as_tensor(..., device=device)`, followed by a CUDA sync (`:5841-5848`).
- Recurrent reward/value/policy logits are concatenated and copied to CPU
  NumPy every simulation (`torch.cat(...).detach().cpu().numpy()`, `:5877-5892`).
- The NumPy arrays are converted to Python lists with `.tolist()` before
  `tree_muzero.batch_backpropagate(...)` (`:5895-5909`).

This means the GPU-latent mode is still a Python-controlled CTree loop with
one GPU-to-CPU recurrent output payload per simulation.

### 5. Compact target/replay sidecar

`compact_policy_row_bridge.py` proves compact search outputs can feed the
repo target-row contract.

Important pieces:

- `CompactPolicySearchArraysV0` holds active-root arrays for policy row,
  env row, player, action, action mask, visit policy, and root value (`:28-40`).
- `build_policy_row_records_from_compact_search_v0()` still builds one
  `PolicyRowRecordV0` object per active root (`:43-99`).
- `build_compact_target_rows_from_search_arrays_v0()` avoids that intermediate
  policy-record object list, but still loops over active roots to build row
  dicts and then calls `_target_rows_from_dicts(...)` (`:102-231`).
- Validation is array-first and checks row/player/action/reward/final-observation
  alignment (`:234+`).

CPU/object materialization here:

- object path: one `PolicyRowRecordV0` per active root;
- direct compact path: one Python dict per target row before target-row arrays;
- lots of `np.asarray(...)`, `.copy()`, scalar `int/float/bool` conversion.

This is good validation scaffolding, not yet an array-native replay writer.

### 6. RND path

`exploration_bonus.py` is independent from rendering/search and remains a
separate CPU/device boundary.

Important pieces:

- `_to_numpy_cpu()` converts tensors with `detach().cpu().numpy()` (`:86-89`).
- RND latest-frame extraction accepts stock or compact observations but first
  converts to NumPy CPU (`:481-536`, `:539-626`).
- `collect_data()` stores latest frames by cloning one Torch tensor per frame
  into `self.train_obs` (`:849-871`).
- `train_with_data()` samples a Python list, stacks tensors, moves to device,
  trains the predictor, then reads loss with `loss.detach().cpu().item()`
  (`:874-904`).
- `_state_hash()` copies every parameter tensor to CPU NumPy when metrics hash
  state (`:756-767`).
- `estimate()` builds a Torch tensor from NumPy, runs the RND model, reads MSE
  to CPU NumPy, reads scalar stats to CPU, and if weight is nonzero deep-copies
  and augments target reward on CPU (`:906-960`).

CPU/GPU syncs here:

- tensor-to-NumPy conversions for input extraction, model state hashes, MSE, and
  reward stats;
- loss `.item()`;
- target reward deepcopy and CPU NumPy delta checks.

RND is not currently the largest measured bucket after hash fixes, but it is
not compatible with a fully resident device pipeline as written.

## Grid/Tooling State

`scripts/build_curvytron_hybrid_observation_profile_grid.py` is profile-only:

- It targets `curvyzero.infra.modal.source_state_batched_observation_boundary_profile`
  (`:1-10`).
- It marks emitted rows `profile_only=True`, `calls_train_muzero=False`,
  `touches_live_runs=False` (`:390-405`).
- It supports `direct_ctree_gpu_latent_precomputed_recurrent` as an explicit
  MCTS arrays-boundary implementation (`:24-33`, `:438-451`).
- The fixed `--next-direct-ctree-comparison-preset` intentionally compares only
  `stock_facade`, `direct_ctree_arrays`, and `direct_ctree_gpu_latent`; it does
  not include the synthetic precomputed recurrent mode by default (`:37-45`,
  `:114-129`).

That separation is correct. The precomputed recurrent mode is a falsifier, not
a default speed recommendation.

## Critique: `direct_ctree_gpu_latent_precomputed_recurrent`

The mode is present:

- constant and allowed impl: `source_state_batched_observation_boundary_profile.py:182-193`;
- backend/semantics dispatch: `:4489-4510`;
- direct dispatch with `keep_latents_on_device=True` and
  `precompute_recurrent_outputs=True`: `:4714-4749`;
- synthetic recurrent payload branch: `:5790-5801`, `:5850-5875`.

What it correctly tests:

- It bypasses real `model.recurrent_inference`.
- It still calls real LightZero `tree_muzero.batch_traverse(...)` and
  `tree_muzero.batch_backpropagate(...)` once per simulation.
- It still pays action tensor creation, latent indexing, recurrent-output
  CPU readback/listification, CTree backprop, root output extraction, and compact
  output assembly.
- It is explicitly profile-only and kept out of the default fixed comparison
  preset.

Main caveats:

1. `model_eval_count` is misleading for this mode.

   `_run_direct_mcts_arrays()` always reports
   `active_root_count * (1 + num_simulations)` (`:5539`) even when recurrent
   calls are zero. The more truthful fields are
   `lightzero_consumer_model_recurrent_inference_calls` and
   `lightzero_consumer_gpu_latent_precomputed_recurrent_enabled`.

2. It uses trivial all-zero reward/value/policy tensors.

   This is fine for a falsifier, but it changes tree priors, values, and
   expansion dynamics. Read it as "CTree/list/control shell with recurrent model
   removed", not as an exact subtractive decomposition of real search.

3. It skips scalar-transform cost for synthetic reward/value.

   The real branch inverse-transforms recurrent reward/value before CPU
   readback (`:5857-5866`). The synthetic branch directly assigns plain zeros
   (`:5867-5875`). That makes the row slightly more optimistic than "real
   recurrent outputs already available on device".

4. D2H and listification are not fully separated.

   `model_output_d2h_sec` times `torch.cat(...).cpu().numpy()` plus sync
   (`:5877-5892`), but `.tolist()` is untimed separately and is folded into
   broader search wall (`:5895-5897`). If the next question is "D2H or Python
   list?", add a separate listify timer before making architecture calls.

5. It still pays some work a future all-device search would remove.

   It still builds `last_actions_tensor` from Python/NumPy, syncs it, indexes
   `latent_pool`, copies `next_latent_state`, and calls CPU CTree APIs. That is
   useful for this falsifier but not a lower bound for a real GPU-native search.

## Exact Boundary Claims

The current code has four distinct boundaries:

1. **Batched env to host observation/reward/action sidecars.**

   This is mostly NumPy copies and payload dataclasses in
   `source_state_hybrid_observation_profile.py`.

2. **Host sidecars to LightZero scalar/public surface.**

   This is env-id dicts, `MockBaseEnvTimestep`, per-env copies, info dicts, and
   optional pickles in `source_state_batched_observation_mock_collector.py`.

3. **LightZero model/search to CPU CTree lists.**

   This is root value/logit `.cpu().numpy()`, policy logit `.tolist()`,
   CTree `batch_traverse`, recurrent output `.cpu().numpy()`, reward/value/policy
   `.tolist()`, and CTree `batch_backpropagate` in
   `source_state_batched_observation_boundary_profile.py`.

4. **Replay/RND materialization.**

   Compact target rows still become Python row dicts before arrays; RND stores
   Python lists of tensors, copies tensors to CPU for metrics, and returns CPU
   reward arrays.

## Recommended Next Measurement

The next clean split is not another renderer profile. Run same-denominator
profile-only rows that compare:

```text
direct_ctree_gpu_latent
direct_ctree_gpu_latent_precomputed_recurrent
mock_search_service / recurrent_toy ceiling
```

Use enough roots and simulations to make CTree/recurrent time visible. Then
read:

- If precomputed recurrent is close to direct GPU-latent, CTree/list/control is
  the wall.
- If it jumps toward the toy ceiling, recurrent launch/output handling is the
  wall.
- If both remain far below compact sidecar ceilings, the next move is a
  compact search/replay owner, not more small LightZero wrapper patches.

Add two small telemetry fixes before relying on charts:

- report `actual_model_eval_count` separately from logical MuZero search shape;
- split recurrent-output `.cpu().numpy()` time from NumPy `.tolist()` time.

## Bottom Line

The code supports the current high-level optimizer read:

```text
Small patches around LightZero's stock boundary can plausibly give 1.2x-1.5x.
The 5-10x lane needs compact batched ownership across search and replay, or a
device/array-native search service. The precomputed recurrent mode is a useful
falsifier for deciding which half of that boundary hurts more, but it is not a
training path and not a semantic-equivalence path.
```
