# Stock Boundary Batch Death Audit - 2026-05-21

Scope: read-only audit of the trusted stock LightZero `train_muzero` path for
`source_state_fixed_opponent`, compared with
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
I did not modify production code or launch training.

## 1. What is proven by code

- The trusted launcher still calls stock LightZero. The CurvyTron Modal trainer
  imports `lzero.entry`, resolves the configured entrypoint, builds LightZero
  configs, and calls:
  `train_muzero([patched["main_config"], patched["create_config"]], seed=..., max_train_iter=..., max_env_step=...)`
  (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5336-5342`,
  `:5717-5728`). The default env variant is delegated from
  `lz_config.DEFAULT_ENV_VARIANT`, and the source-state fixed-opponent alias is
  wired in the same launcher (`:433-439`).

- The config builder exposes collector parallelism to LightZero as many scalar
  env instances, not one batched CurvyTron surface. It patches
  `policy.collector_env_num`, `policy.n_episode`, `policy.batch_size`, and the
  env config's `collector_env_num`, then sets the env `type/import_names` for
  the selected wrapper (`src/curvyzero/training/lightzero_config_builder.py:1268-1325`).

- The stock fixed-opponent env wrapper is scalar at the LightZero API. Each
  env has `VectorMultiplayerEnv(batch_size=1, player_count=2)` internally and
  accepts one ego action. It builds a `(1, 2)` joint action by filling the
  opponent action inside the wrapper, steps one physical row, then computes one
  next LightZero observation (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:824-921`).

- The first hard batch death on the trusted stock path is inside
  `_lightzero_observation`. `_update_stack()` returns one env's ego stack, then
  `_lightzero_observation` copies it and returns a Python dict:
  `{"observation": observation, "action_mask": action_mask, "to_play": -1,
  "timestep": int(...)}` (`curvyzero_source_state_visual_survival_lightzero_env.py:1238-1254`).
  Even when `policy_observation_backend == jax_gpu`, the scalar renderer writes
  into NumPy output buffers, copies one ego frame and one opponent frame, then
  normalizes and shifts NumPy stacks (`:1256-1271`, `:1341-1352`). There is no
  resident tensor crossing the env boundary.

- Upstream LightZero `v0.2.0` then re-scalarizes by env id. `train_muzero`
  builds DI-engine env managers from env factories and creates
  `Collector(env=collector_env, policy=policy.collect_mode, ...)`
  (LightZero `lzero/entry/train_muzero.py:72-118`). `MuZeroCollector.collect`
  reads `self._env.ready_obs`, a dict keyed by scalar env id, converts
  `init_obs[i]["observation"]`, masks, `to_play`, and timestep through
  `to_ndarray`, and creates one `GameSegment` per env id
  (`lzero/worker/muzero_collector.py:347-387`).

- Before policy/search, the collector rebuilds a batch from scalar segment
  objects: `stack_obs = {env_id: game_segments[env_id].get_obs() ...}`;
  then `stack_obs = list(stack_obs.values())`. It separately filters masks and
  `to_play` dictionaries by scalar env id, calls policy forward, unpacks a dict
  back by env id, and calls `self._env.step(actions)` with an action dict
  (`lzero/worker/muzero_collector.py:413-429`, `:491-513`). This is Python
  dict/list orchestration around a reconstructed model batch.

- Search does not keep the model-side batch resident either. In LightZero
  `v0.2.0` MuZero collect forward, the initial inference can run on the model
  device, but then `pred_values`, `latent_state_roots`, and `policy_logits` are
  detached to CPU NumPy/list before MCTS roots/search:
  `pred_values = ...detach().cpu().numpy()`,
  `latent_state_roots = ...detach().cpu().numpy()`,
  `policy_logits = ...detach().cpu().numpy().tolist()`
  (`lzero/policy/muzero.py:724-760`). So even a better env boundary would still
  hit a CPU/search boundary unless search is also changed.

- Replay is NumPy/list segment storage. `GameSegment.append` appends scalar
  transition pieces to Python lists (`lzero/mcts/buffer/game_segment.py:137-148`);
  the collector appends `to_ndarray(obs["observation"])` and later updates the
  observation window with another `to_ndarray(obs["observation"])`
  (`lzero/worker/muzero_collector.py:481-509`).

- The boundary profile is not the trusted stock path. It explicitly builds a
  `SourceStateMultiplayerTrainerSurface`, wraps it in
  `BatchedLightZeroScalarActionBridge`, and exposes a profile env-manager facade
  (`source_state_batched_observation_boundary_profile.py:1568-1577`). The
  scalar LightZero payload is optional and explicit:
  `materialize_lightzero_scalar_timestep(...)` plus optional pickle timing
  (`:535-550`).

- That materialization function is exactly the canary's scalarization point:
  it accepts `[B, P, 4, 64, 64]`, casts/normalizes to float32 NumPy, reshapes to
  `[B*P, 4, 64, 64]`, builds repeated done/mask/to_play arrays, and creates a
  `MockBaseEnvTimestep` (`src/curvyzero/training/source_state_batched_observation_mock_collector.py:724-790`).

- The actual trainer contains a profile-only hook that can replace LightZero's
  `create_env_manager` with a batched profile manager, but the hook itself
  says it is a profile manager and requires even scalar env count because one
  physical row exposes two scalar player views
  (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:1820-1858`).
  The stock-shaped adapter still returns `env_id -> timestep` mappings and
  converts each profile timestep to a stock `BaseEnvTimestep`
  (`source_state_batched_observation_mock_collector.py:445-456`, `:550-577`).

## 2. What is only inferred

- It is likely, but not proven from local runtime measurements in this audit,
  that the dominant loss from resident GPU observation is the repeated
  materialization/copy path: GPU/JAX render output -> NumPy stack -> copied obs
  dict -> env-manager ready dict -> collector `to_ndarray` -> Python list batch
  -> Torch tensor -> CPU NumPy MCTS roots -> GameSegment NumPy/list replay.

- It is likely that the profile-only resident canary's throughput drop when
  scalarizing/pickling is measuring the same class of cost as stock LightZero,
  but the canary is not a proof of exact stock overhead because its surface,
  bridge, and env-manager adapter are repo-owned profile code, not the trusted
  default `subprocess` env manager path.

- It is likely that optimizing render alone is now below the highest-leverage
  frontier. The stock collector/search/replay path intentionally disassembles
  the batch into scalar env ids and CPU objects at several boundaries. A faster
  renderer can reduce `_update_stack`, but cannot remove collector dict/list
  traffic, MCTS CPU conversion, or replay materialization.

- It is not proven here whether a full batched env-manager replacement is
  behaviorally safe for trusted training. The code has a profile-only adapter,
  but the current trusted evidence remains the scalar `source_state_fixed_opponent`
  train path.

## 3. Likely highest-leverage interface change

The highest-leverage change is not "make observation rendering faster"; it is
to move the ownership boundary up one level:

- Introduce a stock-compatible batched collector/env-manager interface where
  the environment manager owns one resident `SourceStateMultiplayerTrainerSurface`
  and exposes a batch table (`policy_env_id`, `row`, `player`, observation,
  mask, reward, done) without per-env object copies.

- Feed policy/search from that table directly, preserving `[N, 4, 64, 64]`
  device/contiguous storage through initial inference. Only convert to scalar
  ids at the last point LightZero absolutely requires, ideally after policy
  output, not immediately after observation.

- Treat MCTS as the next boundary, not an implementation detail. LightZero
  `v0.2.0` currently converts latent roots and logits back to CPU NumPy before
  search. If we keep stock MCTS, the best env change may still be capped by
  CPU search. If we want full resident speed, policy/search must accept
  device-resident latent roots or a batched search implementation.

Concretely, the existing `BatchedLightZeroStockEnvManagerAdapter` is useful as
a compatibility canary, but not as the final speed interface: it deliberately
returns `env_id -> BaseEnvTimestep` and therefore preserves the stock scalar
boundary. The next architecture should either fork/wrap `MuZeroCollector` for
batched rows, or add a narrow "batched ready_obs" fast path before stock
collector scalarization.

## 4. Minimal next experiment that does not touch live training

Run a profile-only canary against fresh scratch run ids, with background
eval/GIF disabled, no checkpoint promotion, and a hard stop after the first
learner call. This is not live training evidence; it is a bounded interface
profile.

1. Use a fresh run id and `mode=profile`, `stop_after_learner_train_calls=1`,
   `env_manager_type=curvyzero_batched_profile`, `profile_volume_commit=false`,
   background eval/GIF disabled, checkpoint cadence set beyond the profile
   horizon, and `source_state_fixed_opponent` semantics.
   This exercises stock `train_muzero` plus the profile env-manager hook, but
   stops after one learner call and should not be treated as training evidence.

2. Run the paired `env_manager_type=curvyzero_batched_zero_obs_profile` canary
   with identical LightZero/search settings. The delta isolates renderer and
   surface cost from collector/search/learner overhead.

3. In the output, compare:
   `batched_profile_renderer_*`, `batched_profile_bridge_*`,
   `batched_profile_env_manager_*`, `collector_collect_sec`,
   `model_initial_inference_sec`, `mcts_search_sec`,
   `learner_train_sec`, and envstep throughput.

4. Decision rule: if zero-observation and real-render canaries are close, the
   batch is dying after observation and the next change should be collector/search
   interface work. If real-render remains much slower, keep optimizing the
   resident renderer/surface before changing collector contracts.

This experiment is deliberately a canary, not live training: no production code
edit is required, no trusted run should be resumed, no Modal volume cleanup or
promotion should run, and the result should be filed as architecture evidence
only.
