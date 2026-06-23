# Mechanics vs Observation Audit, 2026-05-22

Scope: read-only audit of the compact closed-loop profile code. I did not edit
source and did not touch live runs.

## Short Answer

`env_step_sec` is fuzzy because it is not just the CurvyTron game step. In the
compact MCTX closed loop it means:

```text
selected actions -> actor/env step -> compact sidecars -> render-state/stack
refresh -> next compact observation/search-input handoff
```

The current mechanics leaf is small. The hot part is mostly observation and
search-input handoff.

## 1. Definition Of `env_step_sec`

In `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`, the closed-loop
profile starts `loop_step_started` immediately before:

```python
loop_next_step = compact_visual_manager_for_replay.step(loop_joint_action)
```

Then, in resident-GPU-stack mode, it also updates the resident stack before
stopping the timer. The recorded field is:

```python
"env_step_sec": loop_step_sec
```

So `env_step_sec` includes:

- the manager step;
- in-process actor stepping;
- `VectorMultiplayerEnv.step(...)`;
- compact sidecar writes;
- render-state writes or borrowed render-state setup;
- observation render/update;
- host stack shift/latest-frame update when enabled;
- terminal/final-observation and autoreset observation work when rows end;
- resident GPU stack FIFO update when enabled.

It excludes:

- `root_build_sec`;
- top-level root/search-input `h2d_sec`;
- `search_sec`;
- search-output `d2h_sec`;
- `replay_index_sec`;
- unlabeled loop glue outside those buckets.

## 2. Leaf Timers That Are Real Mechanics

The current plain breakdown in `mctx_synthetic_benchmark.py` groups strict
mechanics as:

- `actor_env_runtime_sec`
- `actor_env_post_runtime_bookkeeping_sec`
- `actor_env_reward_sec`

What those mean:

- `actor_env_runtime_sec` is the wall around
  `VectorMultiplayerEnv._advance_runtime_for_public_step(...)`. That calls
  `vector_runtime.step_many(...)`, sometimes once per decision and sometimes in
  a source-frame substep loop. It also includes natural bonus timer advance and
  elapsed-time updates, so it is the best current mechanics leaf but still not
  a perfect physics-only timer.
- `actor_env_post_runtime_bookkeeping_sec` covers death correction, death
  appends, episode-step increments, overflow/timeout/truncation flags, and
  terminal/warmdown flags after the runtime step.
- `actor_env_reward_sec` covers reward-map computation.

Mechanics-adjacent but not strict mechanics:

- `actor_env_public_prepare_sec`: action/source-move setup, disabled masks,
  timer advance setup, random-tape headroom checks.
- `actor_env_final_observation_sec`: terminal observation capture.
- `actor_env_public_info_sec`: public info dict construction.
- `actor_env_batch_pack_sec`: public batch packaging.
- `actor_autoreset_sec`: reset work for terminal rows.

`vector_runtime.py` has finer internal phase names such as `movement_sec`,
`normal_point_append_sec`, `body_collision_sec`, `bonus_catch_sec`,
`terminal_score_state_sec`, and `tick_sec`, but those are not surfaced in the
current compact closed-loop rows.

## 3. Leaf Timers That Are Observation/Search-Input Handoff

Inside `env_step_sec`, these are handoff, not game mechanics:

- `actor_compact_write_sec`: writes reward/done/mask/identity/action sidecars.
- `actor_payload_copy_sec`: actor payload copy cost.
- `actor_render_state_write_sec`: writes renderer input buffers.
- `actor_render_state_write_visual_trail_sec`
- `actor_render_state_write_player_sec`
- `actor_render_state_write_bonus_sec`
- `actor_render_state_write_other_sec`
- `gather_merge_sec`: merges actor payloads in non-native-buffer mode.
- `observation_sec`: wall around `_update_observation(...)` plus reset
  observation handling.
- `stack_shift_sec`: host stack FIFO shift.
- `stack_latest_update_sec`: latest rendered frame write into host stack.
- `resident_stack_update_sec`: resident GPU stack FIFO update outside the
  manager step but inside `env_step_sec`.

Renderer leaves nested under observation:

- `renderer_production_to_compact_sec`
- `renderer_persistent_delta_pack_sec`
- `renderer_host_to_device_sec`
- `renderer_persistent_update_sec`
- `renderer_device_render_sec`
- `renderer_device_to_host_sec`

Treat these labels carefully:

- `renderer_render_sec` is inclusive renderer time, not an exclusive leaf.
- `renderer_stack_update_sec` is currently the same broad wall as
  `observation_sec`.
- `actor_step_wall_sec` and `actor_step_sec` are inclusive actor-loop labels.

Outside `env_step_sec`, but still part of preparing/consuming the next search
input:

- `root_build_sec`: builds `CompactRootBatchV1`.
- `h2d_sec`: readies observations/masks on device.
- `search_sec`: MCTX search itself.
- `d2h_sec`: selected action/action-weight/root-value readback.
- `replay_index_sec`: compact replay-index row construction after search.

## 4. Amdahl Read From Latest Rows

Latest profile-only rows in the current docs say:

| row | roots/sec | env frac | search frac | read |
| --- | ---: | ---: | ---: | --- |
| resident sim16 copied | `34.1k` | `64.6%` | `10.2%` | parent render-state copy still present |
| resident sim16 borrowed | `51.8k` | `52.5%` | `16.0%` | parent render-state copy removed |
| resident sim32 borrowed | `38.5k` | `42.6%` | `30.6%` | search is already a large wall |
| resident sim16 refresh off | `61.9k` | `32.1%` | `19.3%` | invalid training lane, ceiling only |
| lazy-sync sim16 borrowed | `54.7k` | `52.4%` | `16.7%` | profiling switch; sim32 regressed |

Plain Amdahl read:

- Removing the parent render-state copy was real: sim16 resident moved
  `34.1k -> 51.8k`, about `1.52x`.
- After that, deleting all observation refresh only has about a `1.2x` ceiling
  at sim16 (`51.8k -> 61.9k`). If using the noisy lazy-sync sim16 row, the
  remaining ceiling is even smaller (`54.7k -> 61.9k`, about `1.13x`).
- At sim32, search is already about `30%` of wall in the borrowed row. Pure
  render/observation work is no longer a plausible 10x lever in this profile
  denominator.
- The next high-value work is either remaining state/observation handoff
  ownership or the search/service boundary. Game mechanics alone is not the
  current Amdahl lever.

## 5. Instrumentation Gaps

1. Surface `vector_runtime` phase timers into the compact closed-loop rows.
   Today `actor_env_runtime_sec` is one coarse mechanics leaf. We need the same
   MCTX rows to expose `movement_sec`, `body_collision_sec`,
   `normal_point_append_sec`, `bonus_catch_sec`, `tick_sec`, and related phases.

2. Make observation timing exclusive. Right now `observation_sec`,
   `renderer_render_sec`, `renderer_stack_update_sec`, renderer H2D/update/draw,
   and resident stack update are partly nested. The next profile split should
   make render call, delta pack, H2D, persistent update, device draw, host stack
   update, resident stack update, terminal/final-observation handling, and sync
   wait time clearly separate.
