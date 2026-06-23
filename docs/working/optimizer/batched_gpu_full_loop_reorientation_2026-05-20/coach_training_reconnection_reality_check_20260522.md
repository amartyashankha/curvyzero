# Coach Training Reconnection Reality Check

Date: 2026-05-22

Status: doc-only read. Do not mutate live runs, checkpoints, Modal volumes,
trainer defaults, or production code from this note.

## Answer

`direct_ctree_gpu_latent` now has a profile-only hook inside the trusted stock
`train_muzero` launcher, but it is not production and it is not Coach launch
advice. The C64/sim16/3-learner repeat was stock `445.19` steps/sec versus
direct `438.56` steps/sec.

The trusted Coach path still enters stock LightZero through `lzero.entry.train_muzero`
from
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
The hook only runs in `mode=profile`, restores afterward, and is rejected
outside profile mode.

This is not a solved config/plumbing change. A config-only change can alter
stock knobs such as `collector_env_num`, `num_simulations`, `batch_size`,
`env_manager_type`, reward settings, RND, and opponent mix. It cannot remove
the remaining CTree CPU/list boundary or the stock per-env output fanout.

To make the optimization affect Coach training, we would need either:

1. a train-safe LightZero policy/search extension that preserves stock collector
   output semantics while swapping the internal search implementation, or
2. a custom collector/replay path that owns compact arrays and then proves it is
   a documented replacement for stock `train_muzero` semantics.

The first route is the smaller semantic replacement. The second route is a
trainer replacement.

## Current Coach Contract

Launcher:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `_run_visual_survival_train(...)` owns launch validation and command metadata
  at lines `4670-4765`.
- it loads `lzero.entry` and chooses `train_muzero` or the RND variant at lines
  `5338-5343`.
- it builds CurvyTron configs at lines `5344-5365`.
- for `mode in {"train", "profile"}`, it installs hooks and then calls
  `train_muzero([main_config, create_config], seed=..., max_train_iter=...,
  max_env_step=...)` at lines `5542-5728`.

Config builder:

- `src/curvyzero/training/lightzero_config_builder.py`
- `VisualSurvivalConfigSpec` exposes run, training, timing, observation,
  behavior, reward, opponent, and RND fields at lines `560-644`.
- `_build_visual_survival_configs_from_builder_kwargs(...)` copies the stock
  Atari MuZero config, sets `create_config.env_manager.type`, and patches
  `policy.collector_env_num`, `policy.num_simulations`, `policy.batch_size`,
  model shape/action size, env metadata, reward target config, opponent config,
  and RND config at lines `1160-1475`.
- There is no field for `mcts_arrays_boundary_impl`,
  `direct_ctree_gpu_latent`, or alternate collect/search backend.

Trusted source-state env:

- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv` is a native single-ego
  LightZero env over `VectorMultiplayerEnv`; its config declares
  `source_state_fixed_opponent`, `two_seat_self_play=False`, and
  `opponent_policy_kind=fixed_straight` by default at lines `332-368`.
- `reset()` creates one source-state episode and returns a scalar LightZero
  observation at lines `690-724`.
- `step(action)` receives one ego scalar action, chooses the opponent internally,
  builds a `[1,2]` joint action, steps the vector env, computes the scalar
  trainer reward, and returns one timestep at lines `824-946`.
- `_lightzero_observation()` returns NumPy `observation`, `action_mask`,
  `to_play=-1`, and `timestep` at lines `1227-1255`.
- the registered env converts the local timestep to DI-engine
  `BaseEnvTimestep` at lines `2678-2683`.

This is the stock LightZero contract Coach currently trusts:

```text
scalar env reset/step
-> BaseEnvTimestep
-> stock LightZero env manager
-> stock collector
-> MuZeroPolicy.collect_mode.forward
-> stock CTree MCTS
-> stock replay/target/learner
```

## Profile-Only Boundary Contract

Profile sidecar:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- module identity is explicitly profile-only at lines `112-114`.
- profile outputs set `profile_only=True`, `calls_train_muzero=False`, and
  `touches_live_runs=False` in several canaries, including lines `1601-1608`,
  `1890-1900`, and `2352-2360`.
- `direct_ctree_gpu_latent` is declared at lines `161-170`.
- `_run_direct_mcts_arrays(...)` starts at lines `5020-5043`; it calls the real
  LightZero model, builds CTree roots, and can delegate to
  `_run_direct_ctree_gpu_latent_search(...)`.
- `_run_direct_ctree_gpu_latent_search(...)` starts at lines `5530-5547`; it
  keeps latent tensors on GPU but still feeds CPU reward/value/policy-logit
  arrays to LightZero CTree traversal/backprop at lines `5596-5680`.

Batched LightZero-shaped bridge:

- `src/curvyzero/training/source_state_batched_observation_mock_collector.py`
- the module docstring says it does not call LightZero or change trainer
  defaults at lines `1-7`.
- the batched loop and scalar bridge are marked `profile_only=True`,
  `stock_lightzero_integrated=False`, `touches_live_runs=False` at lines
  `127-174`.
- `BatchedLightZeroScalarActionBridge.step(...)` requires complete physical
  rows and maps scalar env ids back to one batched joint action at lines
  `220-290`.
- `BatchedLightZeroStockEnvManagerAdapter` is only a
  `profile_canary_only` stock-shaped adapter at lines `446-456`; its `step`
  returns `env_id -> BaseEnvTimestep`-like mappings at lines `551-578`.

Coach profile hook:

- `lightzero_curvyzero_stacked_debug_visual_survival_train.py` includes
  `curvyzero_batched_profile` and `curvyzero_batched_zero_obs_profile` as env
  manager choices at lines `420-430`.
- `_run_visual_survival_train(...)` rejects those env managers outside
  `mode="profile"` at lines `4845-4851`.
- `_install_batched_profile_env_manager_hook(...)` patches
  `train_muzero.__globals__["create_env_manager"]` to create a batched profile
  manager at lines `1809-1827` and `2287-2318`.
- `compile_config` is skipped for those managers only because they are
  profile-only hook-installed managers at lines `7521-7550`.

So the only current stock-loop reconnection is a bounded profile canary. It is
not a training setting and it is not the direct CTree GPU-latent search path.

## Why Config-Only Is Not Enough

`direct_ctree_gpu_latent` sits inside a profile consumer that bypasses the stock
public collect facade. Actual training calls LightZero's stock collector and
policy collect mode. The config builder can set `policy.num_simulations`; it
does not select a replacement CTree execution path or compact-array output
schema.

A train-facing direct CTree path would need to preserve, at minimum:

- observation shape and surface contract: `[4,64,64]`, owner perspective,
  `to_play=-1`;
- binary action-mask semantics and legal action mapping;
- collect-mode output fields consumed by LightZero game segments and replay:
  action, visit distribution/policy target, searched value/root value, reward
  handling, and any LightZero side metadata;
- reset/death/autoreset behavior;
- RND/no-RND behavior;
- checkpoint, resume, eval, tournament, GIF, and telemetry compatibility.

If the code instead keeps compact arrays all the way through replay/targets,
that is no longer stock `train_muzero`; it is a replacement training loop and
must be labeled that way.

## Required Gate Before Coach Advice

Before recommending `direct_ctree_gpu_latent` for Coach training, run a matched
full-loop profile gate after the parity gates pass.

Minimum gate:

- same code revision/image;
- fresh `run_id`s, `mode=profile`, `profile_allow_auto_resume=false`,
  `profile_volume_commit=false`;
- `env_variant=source_state_fixed_opponent`;
- same `collector_env_num`, `num_simulations`, `batch_size`, `source_max_steps`,
  reward variant, RND setting, death/autoreset setting, and opponent mix;
- `called_train_muzero=true`;
- at least one learner train call, so collection, search, replay, target build,
  and learner are all in the denominator;
- no background eval/GIF sidecars in the timing row unless both arms enable the
  same sidecars;
- compare stock public search versus the candidate direct GPU-latent search
  hook using the same env-manager topology.

Pass criteria:

- semantic identity report says stock `train_muzero` consumer semantics are
  still used, or the replacement semantics are explicitly documented;
- zero illegal actions and zero illegal visit mass;
- target/replay rows match the expected LightZero policy/value/reward schema;
- reset/death/autoreset counts match the stock control;
- RND metrics match the selected RND mode;
- wall-clock throughput improves in the full-loop denominator, not only in
  roots/sec sidecar numbers.

Until that gate exists, the safe Coach-facing recommendation is:

```text
Keep direct_ctree_gpu_latent as profile evidence and search-boundary baseline.
Do not present it as an actual Coach training speedup.
```
