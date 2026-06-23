# Current GPU Boundary Audit

Date: 2026-05-21

Status: code-grounded audit only. No trainer defaults, tournament defaults, live
training runs, or active Modal runs were touched.

## Plain Read

The current batched GPU/hybrid observation lane is still host-owned at the
environment, stack, LightZero payload, and manager boundaries. The GPU candidate
copies compact render state to device, renders a batched two-view `uint8`
frame batch, immediately reads those frames back to CPU, then updates CPU
`float32` `[B, P, 4, 64, 64]` stacks. Scalar LightZero rows are materialized only
after that host stack exists.

That is the right profile boundary for safety, but it is not an end-to-end GPU
RL loop. The likely 10x blockers are: one-process manager topology, host stack
and scalar timestep materialization, policy/search still consuming scalar
Python/NumPy rows, RND cadence/data movement, and the lack of device residency
across env/observation/policy/search.

## CPU-Owned Data

- `VectorMultiplayerEnv` owns the source state as NumPy arrays. The batched
  profile facade passes `self.env.state` into `SourceStateBatchedRenderRequest`
  and stores `_raw_frames` and `_stacks` as host NumPy arrays
  (`src/curvyzero/training/source_state_batched_observation_profile.py:210`,
  `src/curvyzero/training/source_state_batched_observation_profile.py:225`,
  `src/curvyzero/training/source_state_batched_observation_profile.py:233`,
  `src/curvyzero/training/source_state_batched_observation_profile.py:363`).
- The trainer-visible renderer-backed stack also owns a host `float32` stack and
  host `uint8` render buffer. It renders all player views through one renderer
  call, then writes the latest frame into the CPU stack
  (`src/curvyzero/training/multiplayer_source_state_trainer_surface.py:146`,
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:150`,
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:222`,
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:255`).
- The hybrid actor scaffold is also CPU-first: in-process actors step
  `VectorMultiplayerEnv`, copy reward/done/alive/action/state arrays into
  payloads, and the parent merges them into CPU NumPy arrays
  (`src/curvyzero/training/source_state_hybrid_observation_profile.py:138`,
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:181`,
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:301`,
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:567`).

## GPU Copy Boundary

- The Modal boundary profile spells out the intended path: CPU env step,
  production-to-compact, owner-ordered pack, H2D, fused GPU render, readback,
  row-major conversion, CPU stack update, CPU parity checks
  (`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:7`).
- The exact H2D point is `_copy_state_to_device(jax=jax, state=compact_state)`;
  immediately before it, the production state is converted and owner-packed on
  CPU (`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1489`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1497`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1522`).
- The device work is only `render_fn(device_state)` plus
  `output_device.block_until_ready()`. The result is then read back with
  `np.asarray(output_device)` and converted from view-major to row-major on CPU
  (`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1526`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1532`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1537`).
- The dynamic renderer always computes a full batched render from the full
  production state. Partial row/player requests are handled after readback by
  slicing `frames[rows, players]`, so partial autoreset support does not yet
  reduce GPU work (`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1653`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1660`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1663`).

## Return To CPU

- Render output returns to CPU as `uint8` frames, then `_push_row_major_frames_into_stack`
  shifts CPU `float32` stacks and scales `uint8 / 255.0` into the newest channel
  (`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1779`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1795`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1826`).
- The trainer surface packages CPU arrays after render: legal masks, live mask,
  policy row/player arrays, selected policy observations, final observations,
  reward maps, and info dictionaries (`src/curvyzero/training/multiplayer_source_state_trainer_surface.py:520`,
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:526`,
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:527`,
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:541`,
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:549`).
- The current policy payload is `float32` `[4,64,64]`; docs note B512 direct-GPU
  profile pickles about `67.1MB` per step because the payload is
  `1024 * 4 * 64 * 64 * float32`, and `uint8` is only a research note, not an
  implemented contract (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/uint8_payload_design_note.md:9`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/uint8_payload_design_note.md:21`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/uint8_payload_design_note.md:25`).

## Scalar LightZero Rows

- In the pure batched facade and boundary profile, scalar LightZero rows are
  materialized after the host stack exists via `materialize_lightzero_scalar_timestep`
  (`src/curvyzero/training/source_state_batched_observation_mock_collector.py:674`,
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py:687`,
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py:714`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:504`).
- In the trainer-surface path, `materialize_trainer_surface_policy_timestep`
  copies/contiguizes `surface_step.policy_observation` and builds
  `MockBaseEnvTimestep` with scalar reward/done/info rows
  (`src/curvyzero/training/source_state_batched_observation_mock_collector.py:728`,
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py:734`,
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py:824`).
- The profile env manager still exposes scalar env IDs and splits timesteps into
  one object per scalar env id. `_ready_obs_by_env_id` copies each observation
  row, and `_split_timestep_by_env_id` creates per-env `MockBaseEnvTimestep`
  objects (`src/curvyzero/training/source_state_batched_observation_mock_collector.py:329`,
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py:383`,
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py:984`,
  `src/curvyzero/training/source_state_batched_observation_mock_collector.py:1017`).
- The hybrid scaffold has the same outer scalarization point: after actor merge
  and observation update, it calls `materialize_lightzero_scalar_timestep` and
  reports `policy_env_id`, `policy_env_row`, and `policy_player`
  (`src/curvyzero/training/source_state_hybrid_observation_profile.py:325`,
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:342`,
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:348`).

## Likely 10x Blockers

- Renderer-only work is no longer a 10x lever at the current best C512 shape.
  The docs put best real render at about `1439.84 steps/s` versus a
  zero-observation ceiling of about `1805.22 steps/s`, or roughly `1.25x`
  remaining observation-only upside
  (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:105`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:113`).
- The one-process batched manager can cross the stock LightZero boundary, but
  early matched rows showed subprocess CPU-oracle still faster at C64 because
  subprocesses hide CPU env/render work better than one batched surface in one
  process (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:217`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:226`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:230`).
- Timed C64 showed collection and manager/stack/render dominating: manager step
  `109.44s`, surface stack update `94.07s`, renderer render `92.02s`, device
  render `82.28s`, with H2D `4.49s` and D2H `0.41s`. That points at the
  observation/stack path plus manager topology, not only transfer bytes
  (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:246`).
- Current code still collapses back to Python/NumPy scalar LightZero rows. The
  world-model note says the profile lane batches rendering but still copies back
  to host, updates host stacks, and returns scalar timestep objects; this explains
  why it can help without yielding 5-10x (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:147`).
- RND is separable but real: docs record RND meter mode as a `10-12%` C512 cost,
  while high update cadences dominate when enabled. It should stay an independent
  axis, not be mixed into renderer claims
  (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:95`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:193`).
- The hybrid zero-observation scaffold has topology headroom, but it excludes
  real render, policy/search/replay/learner/RND, and IPC. The task board records
  B64/A4, B256/A8, and B512/A16 at roughly `15.4k`, `21.6k`, and `24.9k` scalar
  timesteps/sec, and explicitly says the next gate is real dynamic JAX renderer
  injection plus terminal/final-observation semantics
  (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/task_board.md:156`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/task_board.md:167`,
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/task_board.md:170`).

## Audit Conclusion

The current path is best described as CPU env + CPU scalar LightZero boundary
with a batched GPU renderer island. The immediate next performance truth is not
whether the renderer can be micro-optimized; it is whether CurvyZero can keep a
large batch alive across env step, render, policy/search, and payload formation,
or regain subprocess-style CPU actor parallelism while preserving one central
batched GPU observation service.
