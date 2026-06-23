# Subagent Host Overhead And Sync Audit, 2026-05-22

Scope: code-review only. I inspected:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- persistent renderer pieces in
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- focused contract tests around compact replay, resident stack, and MCTX root
  value extraction

No source code, live Coach/training run, checkpoint, eval, GIF, tournament
artifact, or Modal run was touched. This document is the only write.

## Current Read

The direct root-node extraction fix changes the priority stack. `_extract_mctx_root_values`
now reads `search_tree.node_values[:, 0]` / equivalent direct arrays before
falling back to `search_tree.summary()` (`mctx_synthetic_benchmark.py:691-743`).
The focused test covers this direct path and payload byte count
(`tests/test_mctx_synthetic_benchmark_legality.py:147-171`).

Latest profile-only denominator from `world_model.md:17-24`:

```text
H100, B1024/P2, loop24, no-death compact loop,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack, no root observation copy,
explicit resident-stack sync off,
vectorized delta pack off.
```

Latest replay-valid rows:

```text
sim16: 55.0k roots/sec, total 0.894s
  env_step 0.656s, search 0.157s, root_value_extract 0.019s,
  replay_index 0.010s

sim32: 38.1k roots/sec, total 1.289s
  env_step 0.771s, search 0.418s, root_value_extract 0.024s,
  replay_index 0.012s
```

Plain Amdahl read: root-value extraction is no longer the wall. Replay-index
construction is also small in the latest replay-valid rows. The dominant sim16
wall is now the `env_step_sec` bucket, but that bucket is mostly observation /
next-search-input handoff rather than game mechanics. At sim32, search becomes
large enough that both handoff and MCTX search scaling matter.

## Timer Semantics

### Mostly Exclusive Top-Level Timers

These are sequential spans in the repeated closed-loop records
(`mctx_synthetic_benchmark.py:2636-2951`) and are summed into bucket totals at
`mctx_synthetic_benchmark.py:2978-3005`:

| Timer | Exclusive enough? | Notes |
| --- | --- | --- |
| `root_build_sec` | Yes, for `build_compact_root_batch_v1`. | Includes validation and sidecar copies in `compact_policy_row_bridge.py:124-251`. With `copy_observation=False`, observation is a view, but sidecars are still copied. |
| `root_sidecar_sec` | Yes, but partly benchmark glue. | Includes row-major guard, `np.asarray(loop_root_batch.observation)`, legal-mask inversion, active-mask conversion, and `np.flatnonzero` (`mctx_synthetic_benchmark.py:2648-2672`). In resident mode the observation is not the hot search input, so this can overprice validation. |
| `h2d_sec` | Yes as wall, misleading by source. | Host mode transfers observation plus invalid mask. Resident mode reshapes the resident stack, blocks on it, and transfers only invalid mask (`mctx_synthetic_benchmark.py:2674-2689`). If resident stack sync is deferred, wait can move here. |
| `search_sec` | Yes as synchronized search wall. | Full replay-valid mode blocks on `action_weights`; action-only/deferred/overlap modes block only on `action` (`mctx_synthetic_benchmark.py:2691-2708`), so those search timers are different denominators. |
| `d2h_sec` | Yes, but incomplete payload label. | Reads selected actions and, in full mode, action weights (`mctx_synthetic_benchmark.py:2710-2721`). Root values are outside this timer. Because `search_sec` already blocks on `action_weights`, this often measures host transfer/cast more than compute wait. |
| `root_value_extract_sec` | Yes, now small. | Direct root node read happens at `mctx_synthetic_benchmark.py:2777-2781`; fallback to `summary()` would be a regression. |
| `search_result_validate_sec` | Yes. | `validate_compact_search_result_v1` checks legality, normalized visits, finite values, and copies result arrays (`compact_policy_row_bridge.py:254-345`). |
| `joint_action_build_sec` | Yes. | Allocates default joint action and indexed-writes active selected actions (`mctx_synthetic_benchmark.py:2803-2818`). |
| `env_step_sec` | Yes as loop wall, very inclusive. | Contains `HybridBatchedObservationProfileManager.step(...)` plus optional resident JAX stack update and block (`mctx_synthetic_benchmark.py:2820-2841`). It is not pure physics. |
| `replay_index_sec` | Yes. | Calls `build_compact_replay_index_rows_v1_from_search_result` (`mctx_synthetic_benchmark.py:2861-2891`). Latest cost is small, but the strict builder remains validation-heavy (`compact_policy_row_bridge.py:424-586`). |

### Nested Or Misleading Timers

`next_step_timings_sec` is diagnostic attribution inside `env_step_sec`; it is
not an exclusive profile.

- `actor_step_wall_sec` is a wall around the in-process actor loop and row
  writes (`source_state_hybrid_observation_profile.py:655-783`). It contains
  `actor_env_*`, compact writes, render-state writes, and in-process loop
  overhead. `actor_idle_wait_sec` is residual math, not real async actor idle.
- `actor_env_runtime_sec`, `actor_env_reward_sec`, and
  `actor_env_post_runtime_bookkeeping_sec` are the closest leaves for game
  mechanics. They are nested under `actor_step_wall_sec`.
- `observation_sec` wraps `_update_observation(...)`, terminal final-observation
  copy, and autoreset observation handling (`source_state_hybrid_observation_profile.py:794-823`).
- `renderer_stack_update_sec` is assigned equal to `observation_sec`
  (`source_state_hybrid_observation_profile.py:823-824`). Treat it as an alias,
  not another bucket.
- `renderer_render_sec` is inclusive. In the persistent renderer it is the sum
  of production-to-compact, delta pack, H2D, persistent update, device render,
  and D2H (`source_state_batched_observation_boundary_profile.py:2968-2977`).
- `renderer_device_render_sec` is the raw compose/draw wait, but only after
  H2D and persistent layer update have already happened.
- `host_observation_setup_sec`, `host_setup_plus_fresh_boundary_sec`, and
  `closed_one_step_search_replay_edge_sec` are setup / synthetic add-up fields,
  not the repeated closed-loop denominator. The end-to-end fresh loop always
  does fresh `jax.device_put(obs_host)` (`mctx_synthetic_benchmark.py:2306-2331`),
  so it is not the same as the resident-stack closed loop.
- `steady_search_plus_h2d_plus_policy_d2h_median_sec` sums medians from separate
  loops. Useful for a quick boundary smell test, but not a real iteration wall.

## Remaining Host And Sync Risks

### 1. `env_step_sec` hides compact-batch construction

The compact capture probe in the MCTX benchmark deliberately reports
`total_sec: 0.0` (`mctx_synthetic_benchmark.py:1095-1107`). In the manager,
`timings["batched_stack_probe_sec"]` prefers probe telemetry `total_sec` over
the measured span (`source_state_hybrid_observation_profile.py:901-931`).

Consequence: `_make_compact_batch(...)` (`source_state_hybrid_observation_profile.py:1695-1748`)
is inside `env_step_sec`, but can be reported as zero in nested telemetry. The
closed-loop whitelist also omits `batched_stack_probe_sec`
(`mctx_synthetic_benchmark.py:2893-2927`). This is a concrete residual trap.

Recommended telemetry:

- `compact_batch_build_sec`
- `compact_batch_capture_probe_sec`
- `hybrid_step_return_copy_sec`
- `compact_payload_copy_sec`
- `env_step_unattributed_sec`

### 2. Persistent renderer still has host state packing and hard H2D waits

The persistent renderer still does:

```text
production state -> compact state -> delta state -> device_put each array
-> persistent layer update -> compose/render
```

Code:

- production-to-compact: `source_state_batched_observation_boundary_profile.py:2883-2890`
- visual compact adapter: `source_state_batched_observation_boundary_profile.py:3059-3138`
- delta state: `source_state_batched_observation_boundary_profile.py:3248-3415`
- H2D: `source_state_batched_observation_boundary_profile.py:2912-2927`
- `_copy_state_to_device` blocks every copied device value:
  `source_state_gpu_render_benchmark.py:1741-1745`
- update/render blocks unless async device-only profile defers them:
  `source_state_batched_observation_boundary_profile.py:2931-2942`

The async renderer flag only defers update/render blocks. It does not make
`_copy_state_to_device` asynchronous because that helper still blocks every
device-put. That matches the latest docs: async internal renderer did not beat
the simpler resident-sync-off row.

Recommended telemetry:

- `renderer_compact_state_bytes`
- `renderer_delta_state_bytes`
- `renderer_compose_state_bytes`
- `renderer_device_put_array_count`
- `renderer_device_put_block_sec`
- `renderer_update_queue_sec` and `renderer_update_wait_sec`
- `renderer_compose_queue_sec` and `renderer_compose_wait_sec`
- `persistent_delta_state_mode`: exact, vectorized, or fallback

### 3. Resident stack wait can move between buckets

Resident mode avoids search-observation D2H/H2D, but the benchmark keeps a
separate JAX FIFO stack:

```text
jnp.concatenate((device_stack[:, :, 1:], latest_device), axis=2)
```

and may block immediately (`mctx_synthetic_benchmark.py:1244-1262` and
`mctx_synthetic_benchmark.py:2823-2841`). If
`compact_visual_resident_sync=False`, the same wait can move into `h2d_sec`
because `loop_obs.block_until_ready()` still runs before search
(`mctx_synthetic_benchmark.py:2674-2689`).

Recommended telemetry:

- `resident_stack_concat_queue_sec`
- `resident_stack_concat_wait_sec`
- `resident_stack_bytes_shifted_logical`
- `resident_stack_wait_absorbed_in_h2d_sec`
- split `h2d_invalid_mask_sec` from `h2d_resident_stack_ready_sec`

### 4. Search output readback is now small but under-specified

Full mode reads action and visit policy in `d2h_sec`, then root value in
`root_value_extract_sec` (`mctx_synthetic_benchmark.py:2710-2721` and
`mctx_synthetic_benchmark.py:2777-2781`). The direct root fix makes this cheap,
but the code still has a fallback to `search_tree.summary()`
(`mctx_synthetic_benchmark.py:716-741`).

The fallback should be treated as a profile regression, not an innocent
compatibility path, on the current MCTX version.

Recommended telemetry:

- `d2h_action_sec`
- `d2h_visit_policy_sec`
- `d2h_root_value_sec`
- `root_value_source`
- `root_value_fallback_summary_used`
- readback bytes for action, visit policy, and root value

### 5. Strict compact replay validation is correct but not the next wall

`build_compact_replay_index_rows_v1_from_search_result` correctly avoids
observation materialization, but it still validates identities, masks, reward,
done, final-observation row masks, and copies all output arrays
(`compact_policy_row_bridge.py:424-586`). Latest replay-valid rows show this is
around `0.010-0.012s` over 24 steps, so it is not the next P0 speed target.

Keep it strict for now. Add a fast sampled-validation builder only after the
larger `env_step_sec` split shows replay-index becoming a double-digit wall.

### 6. No-copy root observation has aliasing risk

`copy_observation=False` is validated by focused tests
(`tests/test_compact_search_replay_contract.py:668-684`) and is the right
profile denominator. But if any future deferred payload design holds root
batches across manager steps, no-copy roots can alias mutable compact state.

Guardrail telemetry/tests:

- root batch `observation_copied` must be reported
- deferred payload rows must keep stable root sidecars
- keep the non-prefix deferred payload parity test
  (`tests/test_compact_search_replay_contract.py:579-658`)

## Amdahl Telemetry To Add Next

Add instrumentation only, with no behavior change, around the current
borrowed/resident replay-valid row:

```text
closed_loop_total_sec
measured_bucket_sum_sec
residual_sec
env_step_sec
env_step_known_nested_sec
env_step_unattributed_sec
```

Then split `env_step_unattributed_sec`:

- `compact_batch_build_sec`
- `capture_probe_dispatch_sec`
- `hybrid_step_return_copy_sec`
- `compact_payload_copy_sec`
- `final_observation_copy_sec`
- `autoreset_reset_observation_sec`

Split renderer:

- production-to-compact
- delta pack
- H2D array count / bytes / block time
- persistent update queue vs wait
- compose/render queue vs wait
- D2H, expected zero in resident device-only mode

Split search boundary:

- action readback
- visit-policy readback
- root-value readback
- root-value source
- search-result validation
- joint-action allocation vs indexed assignment

Split root/replay:

- root observation reshape/view vs copy
- root sidecar validation
- root sidecar copy bytes
- active-root `flatnonzero`
- invalid-mask H2D bytes
- replay validation vs final array construction

These fields clarify whether the next 20% win comes from renderer/stack
residency, Python compact-object churn, or search scaling.

## Smallest Next Profile Rows

Use the same current denominator unless a row says otherwise:

```text
H100, B1024/P2, loop24, body_capacity=4096,
rollout_steps=4, hidden_dim=64, max_depth=16,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
compact_visual_observation_source=resident_gpu,
compact_root_copy_observation=False,
closed_loop_replay_index=True,
compact_visual_resident_sync=False,
persistent_vectorized_delta_pack_profile=False,
persistent_renderer_async_device_only_profile=False
```

| Row | Sims | Switch delta | Purpose | Read |
| --- | ---: | --- | --- | --- |
| H0 baseline rerun | 16, 32 | none | Anchor after direct root extraction. | total roots/sec, env/search fractions, root-value source, residual |
| H1 sync relocation | 16 | `compact_visual_resident_sync=True` | Check whether explicit resident stack wait is still a real wall. | total wall, not just `resident_stack_update_sec` |
| H2 host-vs-resident | 16, 32 | `compact_visual_observation_source=host` | Price resident stack benefit after root-value fix. | `env_step_sec`, `h2d_sec`, renderer D2H, total roots/sec |
| H3 refresh-off ceiling | 16, 32 | `hybrid_refresh_observation_stack=False` | Upper-bound deletion of observation/render handoff. | gap to H0; do not treat as replay/training realism |
| H4 replay off, full payload | 16 | `closed_loop_replay_index=False`, action-only off | Verify replay-index still tiny after direct root extraction. | total delta vs H0 should roughly match `replay_index_sec` |
| H5 compact batch timer patch row | 16 | instrumentation only | Reveal `_make_compact_batch` and return-copy cost hidden inside `env_step_sec`. | `env_step_unattributed_sec` should shrink |
| H6 search scaling | 8, 16, 32 | only `num_simulations` | Find sim count where search overtakes handoff. | search fraction slope and roots/sec slope |

Rows not worth prioritizing right now:

- Serial deferred payload flush. Direct root extraction removed the old
  root-value wall, and serial flush is no longer the next big move.
- Async internal renderer as a speed setting. It did not improve total wall and
  does not defer H2D blocks.
- More replay-index toggles before the env/renderer/resident-stack split.
  Latest replay-index is about one percent of total wall.

## Recommendation

The next useful work is not another MCTX search-only row and not another
action-only ceiling. Keep the replay-valid compact MCTX loop as the denominator
and split `env_step_sec` until it is exclusive enough to identify the largest
real wall.

The smallest high-signal patch is instrumentation: expose compact-batch build,
return-copy, renderer H2D/sync bytes, resident stack wait relocation, and
root/search readback source. After that, choose between:

1. a resident/compact renderer state owner if production-to-compact/delta/H2D
   remains dominant;
2. compact object/copy reduction if `_make_compact_batch` and return-side
   copies are double-digit wall;
3. search scaling work only where sim32+ shows search fraction dominating.
