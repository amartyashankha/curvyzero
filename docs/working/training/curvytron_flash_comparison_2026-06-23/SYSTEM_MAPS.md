# System Maps

Status: compact source-read map.

## Flash

Recovered root:

```text
artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/
```

Flash is a Modal-only Curvytron bot lab:

- `modal_curvytron.py`: standard playable web server.
- `modal_curvytron_flash.py`: `curvytron-flash`, `us-west`, class lifecycle,
  `modal.experimental.flash_forward`.
- `modal_bots.py`: standalone WebSocket bot runner and policy sanity check.
- `modal_reference.py`: Node/reference smoke, eval, pin, benchmark, check.
- `modal_accelerated.py`: H100 accelerated env check, validate, parity,
  benchmark, sweep.
- `modal_train.py`: PPO, preset pipeline, profiler, DDP, eval, promotion,
  distillation, JS export.

Runtime shape:

- Playable service is the old Curvytron Node server.
- Rooms require at least two ready players.
- Bots wait for `min_players`, then re-ready after games.
- `curvytron.toml` enables RL bots `mort`, `oog`, and `grug`, all using
  `/checkpoints/exports/latest.json`.
- JS RL policy validates `curvytron.rl_mlp.v0` export format and exact
  `raycast_v1` field order.

Environment/training shape:

- `reference/` wraps real Curvytron code in a deterministic Node VM with seeded
  RNG, synthetic time, deterministic timers, reset/step/snapshot, scripted eval,
  policy match, and policy-batch scenarios.
- `accelerated/` is GPU-resident PyTorch/Triton with optimized 2-player Triton,
  dense N-player Triton, generic torch N-player, CUDA Graph benchmark support,
  and explicit `grid_overflow` reporting.
- CUDA Graph support is limited to `accelerated.benchmark` raw fixed-shape step
  replay. PPO, profile, and DDP training loops do not use CUDA Graph capture.
- "GPU-resident" means env tensors, rollout buffers, and optimized kernels live
  on CUDA; PPO/DDP still contain host synchronization for done/reset control and
  metrics.
- PPO trains over `AcceleratedCurvytronEnv`, usually 2-player grid/raycast,
  `observation_mode="raycast_v1"`, terminal plus shaped rewards, optional
  reachability rewards, and optional auxiliary heads.
- Export path is practical: checkpoint -> `curvytron.rl_mlp.v0` JSON ->
  `/checkpoints/exports/latest.json` -> live JS bot.

## CurvyZero

CurvyZero has two active runtime centers:

- Stock-ish source-state visual LightZero/MuZero training on Modal.
- Profile-only compact owner/search/replay/learner speed experiments.

Important files:

- `goal.md`: speed-claim compass. Accepted baseline is OPT-104 H100 full-loop;
  target is repeatable 2x, current evidence is below target.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`:
  Modal LightZero wrapper, hooks, train/profile/dry modes, H100 surfaces.
- `src/curvyzero/training/lightzero_config_builder.py`: env/reward/observation
  config builder and compact experiment surface.
- `src/curvyzero/training/reward_contracts.py`: reward schemas, hashes,
  reward-space bounds, LightZero support config.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`: compact
  profile harness and owner-boundary proof surface.
- `src/curvyzero/training/compact_owned_loop.py`: profile-only compact replay
  sampling cadence, learner-edge calls, policy-version lineage, deferred workers.
- `scripts/run_curvytron_optimizer_profile_manifest.py`: Modal result collection
  and profile manifest path.

CurvyZero distinctions:

- Production training calls LightZero `train_muzero`; compact-owned profile
  harnesses explicitly do not.
- Multi-GPU CurvyZero evidence should be treated as a runtime row, not a config
  claim: record device count, `lightzero_multi_gpu`, entrypoint, and whether
  `train_muzero` was called before using it in comparisons.
- RND is a stock-ish learning hypothesis, not compact speed progress.
- Same-work H100 full-loop rows are the only speed-claim currency.
- Reward defaults must be named by layer: low-level `auto`, normalized env
  default, experiment-facing default, and Wave A intended arm can differ.

## Keep

- Keep Flash immutable under `artifacts/modal_deploy_downloads/` unless it
  becomes an active fork.
- Store Flash local/control outputs under `artifacts/local/flash_controls/`.
- Store CurvyZero comparison controls under
  `artifacts/local/curvyzero_comparison_controls/`.
- Do not feed Flash rows into CurvyZero compact coach speed-row ledgers.
