# Current data movement audit, 2026-05-21

Scope: read-only audit of the current profile and training-adjacent path. I used `rg` and `sed` locally and did not edit production code or launch training.

## Simple flow

```text
VectorMultiplayerEnv.state
  CPU NumPy source arrays: pos/radius/alive/body or visual_trail/bonus/timers/etc.
    |
    | VectorMultiplayerEnv.step(actions)
    |   vector_runtime.step_many mutates CPU NumPy state
    |   VectorMultiplayerBatch packages observation/action_mask/reward/done/info
    v
SourceStateMultiplayerTrainerSurface / profile facade
  observation stack storage [B,P,4,64,64] or [B,4,64,64]
    |
    | renderer-backed path:
    |   state -> compact/pack -> JAX device_put -> render -> np.asarray(readback)
    |   -> stack roll/update -> policy rows
    |
    | CPU oracle/profile path:
    |   per row/player CPU render -> stack roll/update
    v
LightZero-facing materialization
  policy_observation/action_mask/reward/done -> scalar env ids and MockBaseEnvTimestep
    |
    | BatchedLightZeroScalarActionBridge maps scalar actions back to joint [B,P]
    v
Stock LightZero loop
  Collector -> MuZeroPolicy forward -> MCTS search -> GameBuffer replay push/sample/update
```

The important current shape is: environment and trainer surfaces are CPU NumPy; the GPU renderer/probes are profile-only sidecars; stock LightZero receives scalarized NumPy/Python structures.

## Conversion, copy, and scalar materialization sites

Notes:
- `np.asarray` does not always copy. It is still a boundary worth recording because dtype/device-backed input would materialize on host or fail, and dtype changes copy.
- The list below is limited to the audited files and directly imported training-adjacent helpers.

### Env runtime and public batch packaging

- `VectorMultiplayerEnv.step` copies hot row state before the runtime step: `pre_alive`, `pre_death_count`, `pre_active`, timer and disabled masks, terminal masks, reward/final reward, action sidecars, source moves, and terminal rows. See `src/curvyzero/env/vector_multiplayer_env.py:1026`, `:1039`, `:1040`, `:1079`, `:1083`, `:1088`, `:1100`.
- `_batch` converts and copies terminal masks, final observations/rewards, reward/done/terminated/truncated, and calls `_observe_array()` for the public observation even though trainer observation is later rebuilt elsewhere. See `src/curvyzero/env/vector_multiplayer_env.py:2718`, `:2722`, `:2725`, `:2729`, `:2733`, `:2757`, `:2759`.
- `vector_runtime` validates input with `np.asarray` for `step_ms`, `source_moves`, print-manager mode, masks, and scalar row-float expansion. See `src/curvyzero/env/vector_runtime.py:6008`, `:6014`, `:6022`, `:6031`, `:6037`, `:6074`, `:6078`.
- Runtime internals have many per-row `int(...)` and `float(...)` reads for timers, bonuses, events, collision bookkeeping, and row loops. These are CPU scalar materializations, not device syncs today, but they are an obvious blocker if state ever becomes device-resident.

### Batched observation profile facade

- CPU oracle render converts row index/control arrays and loops row-by-row, materializing `controlled_player` and `source_row` as Python ints for every output row. See `src/curvyzero/training/source_state_batched_observation_profile.py:142`, `:143`, `:153`, `:154`, `:157`.
- `observation` returns `self._stacks.copy()`. Reset and step copy observation, reward, done, joint actions, final observations, row masks, and final rows into result/info. See `src/curvyzero/training/source_state_batched_observation_profile.py:242`, `:266`, `:267`, `:268`, `:304`, `:305`, `:306`, `:470`, `:474`, `:475`, `:476`, `:482`, `:483`.
- Terminal capture allocates `np.zeros_like(self._stacks)`, computes `np.flatnonzero(batch.done)`, and copies terminal stack rows. See `src/curvyzero/training/source_state_batched_observation_profile.py:294`, `:296`, `:297`, `:298`.
- Stack update shifts/normalizes host arrays, either by view reshape and `np.multiply` or by full stack slice assignment. See `src/curvyzero/training/source_state_batched_observation_profile.py:376`, `:378`, `:383`, `:384`, `:391`, `:392`.
- Row/control helpers create repeated/tiled row-major arrays and materialize row/player pairs as Python ints. See `src/curvyzero/training/source_state_batched_observation_profile.py:401`, `:403`, `:409`, `:414`.
- `_joint_actions`, `_controlled_players`, and `_action_vector` convert scalar/vector action inputs with `np.asarray`, `.item()`, Python `int`, and per-row Python assignment. See `src/curvyzero/training/source_state_batched_observation_profile.py:428`, `:443`, `:445`, `:622`, `:623`, `:641`, `:642`.
- Palette construction reads state shape/color arrays and scalar row/player color entries through `np.asarray`, `int`, and `max`. See `src/curvyzero/training/source_state_batched_observation_profile.py:498`, `:504`, `:506`, `:509`, `:521`.

### Trainer surface packaging

- Renderer-backed stack path validates renderer output with `np.asarray(result.frames)` after render. If the renderer returns a JAX array this is a D2H readback. See `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:243`.
- `_surface_step` copies masks/done flags, converts observation to float32, selects policy rows, copies policy action masks, builds final observation/reward maps, and finally copies joint action/reward/final outputs. See `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:529`, `:531`, `:545`, `:555`, `:562`, `:566`, `:575`, `:577`, `:622`, `:623`, `:627`, `:629`.
- The renderer-backed full-row-major fast path avoids copying `policy_observation` by reshape when all live rows are full row-major; other cases do `observation_array[policy_env_row, policy_player].astype(..., copy=True)`. See `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:546`, `:550`, `:555`.
- `_info` copies a large metadata payload: row masks, policy rows, policy player, policy action mask, joint action, legal/action masks, optional final observations/rewards, reward, and shape probes. See `src/curvyzero/training/multiplayer_source_state_trainer_surface.py:651`, `:674`, `:743`, `:757`, `:761`, `:769`, `:770`, `:771`, `:773`, `:781`, `:791`, `:806`, `:810`.

### LightZero scalar bridge and mock collector

- `BatchedLightZeroScalarActionBridge.step` sorts scalar env ids into a NumPy array, counts actions, builds a joint `[B,P]` NumPy action matrix through Python dict/set/int loops, handles autoreset masks, and stores profile timing. See `src/curvyzero/training/source_state_batched_observation_mock_collector.py:226`, `:228`, `:240`, `:248`, `:265`, `:267`, `:287`, `:289`, `:290`.
- `_output_from_loop_step` converts policy row/player arrays, builds scalar env ids, constructs `ready_obs` dicts, optionally rematerializes env-id-specific timesteps, splits timesteps into per-env dictionaries, and copies policy ids/rows/players/masks into output. See `src/curvyzero/training/source_state_batched_observation_mock_collector.py:309`, `:310`, `:311`, `:317`, `:331`, `:332`, `:346`, `:354`, `:363`, `:367`.
- `materialize_lightzero_scalar_timestep` converts observation to float32, optionally normalizes `uint8`, makes a contiguous flattened observation, repeats done, flattens action masks, generates row/player arrays, builds Python `info` list with `int(row)`, `int(player)`, `bool(done[index])`, and copies terminal final observations per row/player. See `src/curvyzero/training/source_state_batched_observation_mock_collector.py:734`, `:735`, `:737`, `:741`, `:744`, `:748`, `:752`, `:760`, `:761`, `:774`, `:776`, `:777`, `:780`, `:783`.
- `materialize_trainer_surface_policy_timestep` makes contiguous policy observations, converts rows/players/action masks/rewards/done/final masks/final observations/final rewards, gathers reward/done by row/player, builds Python info entries, and copies final observations/rewards for terminal rows. See `src/curvyzero/training/source_state_batched_observation_mock_collector.py:805`, `:811`, `:812`, `:820`, `:824`, `:830`, `:831`, `:835`, `:856`, `:873`, `:881`, `:882`, `:884`, `:892`, `:893`.
- `materialize_trainer_surface_env_id_timestep` converts scalar env ids to rows/players, gathers observation/action mask/reward/done by env id, and rechecks final observation/reward maps. See `src/curvyzero/training/source_state_batched_observation_mock_collector.py:916`, `:924`, `:933`, `:941`, `:949`, `:954`, `:963`, `:972`.
- Per-env timestep update paths materialize scalar reward/done through `np.asarray(...).reshape(()).item()` and Python `float`/`bool`. See `src/curvyzero/training/source_state_batched_observation_mock_collector.py:557`, `:558`, `:1142`, `:1147`.

### GPU boundary/profile-only code

- Non-persistent candidate render measures CPU production-to-compact and owner-ordered pack, copies compact state to JAX device, blocks for render, reads output back with `np.asarray(output_device)`, then converts view-major to row-major. See `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:139`, `:2169`, `:2170`, `:2176`, `:2180`, `:2184`, `:2193`.
- Persistent renderer validates row/player/out with `np.asarray`, computes CPU compact/delta state, copies delta and compose state to device, blocks on update and compose, reads frames back with `np.asarray(output_device)`, writes back into host `out`, and copies previous avatar color. See `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2290`, `:2291`, `:2292`, `:2297`, `:2304`, `:2313`, `:2326`, `:2334`, `:2342`, `:2347`, `:2352`, `:2357`, `:2359`, `:2363`.
- Hybrid policy/search probe copies flat observations to device, blocks, runs a synthetic device loop, blocks, then materializes a scalar with `float(output)`. See `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2861`, `:2863`, `:2864`, `:2868`, `:2869`, `:2873`.
- Hybrid batched-stack probe copies action masks to device and either copies full observations to device or reuses `last_output_device`; then it blocks for device stack update, normalize, probe, and scalar `float(output)` readback. See `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2950`, `:2963`, `:2964`, `:2971`, `:2972`, `:2998`, `:3002`, `:3009`, `:3015`, `:3019`.

### Stock training launcher/profiler

- Existing LightZero phase profiler can insert CUDA sync around policy forward, MCTS search, learner train, and replay sample. It measures phase wall time, not the internal tensor/NumPy copies within those phases. See `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1538`, `:1555`, `:1588`, `:1654`, `:1723`, `:1779`.
- Replay instrumentation wraps GameBuffer init, push, remove-oldest, sample, and priority update. It does not inspect payload array types, sizes, or CPU/GPU residency. See `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1765`, `:1768`, `:1775`, `:1781`, `:1788`.
- Background GIF policy action path is not the main training loop, but it is a concrete torch boundary: `np.asarray([observation["observation"]])` -> `torch.as_tensor(..., device=policy_device)`, action mask stays NumPy, `to_play` becomes Python list, `ready_env_id` is NumPy. See `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9879`, `:9880`, `:9884`, `:9885`, `:9886`.
- Torch RNG checkpoint helpers call `.cpu()` on RNG states. This is checkpoint/resume plumbing, not per-step training movement. See `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3922`, `:3925`.

## Measured today vs invisible today

Measured today:

- Profile boundary timing fields include `env_step_sec`, `production_to_compact_sec`, `owner_ordered_pack_sec`, `host_to_device_sec`, `device_render_sec`, `device_to_host_sec`, `view_major_to_row_major_sec`, `stack_sec`, `final_obs_sec`, `lightzero_scalarize_sec`, pickle, RND, and mock collector fields. See `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:139`.
- Candidate non-persistent render explicitly measures H2D, render block, D2H readback, and row-major conversion. See `source_state_batched_observation_boundary_profile.py:2169-2198`.
- Persistent renderer measures production compact, delta pack, H2D, persistent update, device render, D2H, and aggregate `render_sec`. See `source_state_batched_observation_boundary_profile.py:2295-2374`.
- Renderer-backed trainer surface measures package mask copy, live mask, policy rows, policy observation, policy action mask, final observation, info packaging, and output copy when `TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE` is active. See `multiplayer_source_state_trainer_surface.py:524-612`.
- Scalar bridge measures action id sorting, joint-action build, loop step, autoreset reset/materialize, policy env-id build, ready-obs dict build, env-id timestep materialization, split-by-env-id, counts. See `source_state_batched_observation_mock_collector.py:223-360`.
- LightZero launcher measures coarse collector, policy forward, MCTS search, learner train, replay, and optional CUDA sync time. See `lightzero_curvyzero_stacked_debug_visual_survival_train.py:1525-1797`.
- Hybrid probes measure synthetic H2D, device compute, D2H scalar readback, bytes for stack/action-mask transfer, and whether latest device frames were reused. See `source_state_batched_observation_boundary_profile.py:2863-2888` and `:2962-3035`.

Invisible or under-instrumented today:

- `VectorMultiplayerEnv.step` internal copy volume: pre-state snapshots, action sidecars, public observation/action-mask packaging, `_public_info` metadata arrays, and `_batch` copies are not reported as bytes or timing buckets.
- `vector_runtime.step_many` CPU scalar row loops and `int`/`float` extraction hotspots are not attributed separately from env step wall time.
- Trainer surface `_info` payload copy volume is not broken out beyond `package_info_sec`; it does not report bytes or number of copied arrays.
- The full `MockBaseEnvTimestep` materialization payload is partially timed in profile paths, but there is no byte accounting for `flat_obs`, `action_mask`, `reward`, `done`, final observations, Python `info` list, or dict split fanout.
- Stock LightZero policy/search/replay internals do not report tensor device, host/device transfer count, NumPy-to-Torch conversion sites, batch item sizes, replay sample bytes, replay push bytes, or per-phase CPU serialization/materialization.
- CUDA sync time is aggregated by the profiler, but sync cause is not attributed. A long `policy_forward_collect_sec` or `mcts_search_sec` may hide D2H conversion, kernel work, Python work, or framework sync.
- Persistent renderer still reads frames to host for the training-adjacent surface; the profile-only `device_latest_provider` can avoid the full observation H2D in the synthetic stack probe, but that is not yet stock training data flow.

## Lowest-risk instrumentation additions

1. Add byte/count accounting to existing profile-only timing dicts around materialization boundaries.
   - In `materialize_trainer_surface_policy_timestep`, `materialize_trainer_surface_env_id_timestep`, and `materialize_lightzero_scalar_timestep`, record `flat_obs_nbytes`, `action_mask_nbytes`, `reward_nbytes`, `done_nbytes`, `final_observation_nbytes`, `info_count`, and whether each array is C-contiguous.
   - This is low risk because these functions already produce profile outputs and mock timesteps; no behavior change is needed.

2. Add `VectorMultiplayerEnv.step` packaging/copy counters under an opt-in profile flag.
   - Time `_advance_runtime_for_public_step`, `_reward`, `_observe_array`, `_action_mask`, `_public_info`, and `_batch` separately; add approximate bytes for copied state snapshots and public batch arrays.
   - This isolates the "env state -> public batch" host tax without touching vector physics semantics.

3. Extend the stock LightZero phase profiler to sample payload residency and sizes, not contents.
   - In policy forward wrappers, record input argument summaries: type, shape, dtype, `nbytes`/`numel * element_size`, torch device, NumPy/Torch flag, and whether CUDA is synchronized by the wrapper.
   - In MCTS search and replay buffer wrappers, record root batch count plus replay push/sample object size summaries for observation/action/reward/action-mask arrays when cheaply visible.
   - Keep samples capped like existing profiler samples to avoid log blowup.

## What not to optimize yet

- Do not rewrite `vector_runtime` or `VectorMultiplayerEnv.state` to device arrays. The current bottleneck map is not precise enough, and Python scalar row logic is still a semantic anchor.
- Do not remove copies from trainer/info/final-observation packaging before measuring which downstream consumers mutate or retain references. Many copies currently protect LightZero-facing contracts.
- Do not optimize background GIF/eval tensor conversion as part of the training loop. It is a real boundary, but not the main collector/search/replay path.
- Do not move replay storage format or GameBuffer internals yet. First add byte/type sampling around push/sample/update so we know whether replay is actually a copy sink.
- Do not chase every `np.asarray` in isolation. Prioritize places that cross device/host, allocate contiguous observation payloads, split into Python per-env dicts/lists, or copy full `[B,P,4,64,64]` stacks.
- Do not treat the resident/profile canary as proof that stock training can stay device-resident. It proves a pressure-probe shape, but the actual LightZero path still scalarizes and uses host-visible arrays.
