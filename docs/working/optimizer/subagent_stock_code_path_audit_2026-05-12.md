# Stock Code Path Audit

Date: 2026-05-12

Scope: code-path audit for stock LightZero `train_muzero` with
`env_variant=source_state_fixed_opponent` and a frozen checkpoint opponent. No
code changes.

## Bottom Line

The stock frozen-opponent lane is split correctly: GPU compute sets the live
LightZero learner/search policy to CUDA, while `opponent_use_cuda=false` keeps
the frozen checkpoint opponent policy on CPU.

The main Amdahl caveat is attribution. With `env_manager_type=base`, env/render
and frozen-opponent work can show up in the parent-process profiler. With
`env_manager_type=subprocess`, most env-worker internals are hidden; the parent
mostly sees collector wall time plus learner/search/model timings.

## 1. GPU Vs CPU In Stock Frozen-Opponent Profiles

GPU side:

- The Modal train path imports and calls stock `lzero.entry.train_muzero`, not
  the custom two-seat trainer, in `_run_visual_survival_train`
  (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3052`,
  `:3212`).
- GPU compute is converted to `cuda=True` by `_compute_uses_cuda(compute)` and
  passed into the LightZero config (`...train.py:624`, `:3058`).
- `_build_visual_survival_configs` writes that into `main_config.policy.cuda`
  and keeps `policy.multi_gpu` separate (`...train.py:3975`, `:3976`).
- The profiled CUDA-capable LightZero pieces are policy forward, model
  initial/recurrent inference, MCTS search, learner train, and replay sample
  (`...train.py:1015`, `:1355`, `:1457`, `:1528`, `:1585`). Plainly: live
  policy/model inference and learner work can use CUDA; the MCTS tree/bookkeeping
  itself should still be treated as mostly CPU around GPU model calls unless
  LightZero says otherwise.

CPU/env-worker side:

- The registered env is the source-state visual survival env
  (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1464`).
- Env reset/step, action repeat, reward accumulation, vector env stepping,
  stack/render update, and telemetry are Python/NumPy/env work
  (`...lightzero_env.py:578`, `:599`, `:617`, `:734`, `:756`, `:1243`).
- Frozen checkpoint opponent action selection is called from inside env
  `step()` before the vector env advances (`...lightzero_env.py:606`,
  `:792`).

## 2. `opponent_use_cuda=false`

Yes. It is decoupled from learner CUDA.

Evidence:

- Default is false (`...train.py:338`).
- The command records it separately from `compute` (`...train.py:2964`).
- The env config only adds it for frozen checkpoint opponents
  (`...train.py:4089`, `:4103`).
- The env builder passes it to the frozen opponent provider
  (`...lightzero_env.py:1731`, `:1757`).
- The provider sets `cfg.policy.cuda = bool(use_cuda)` and
  `cfg.policy.device = "cuda" if use_cuda else "cpu"`
  (`src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:235`).
- Opponent eval tensors are created on that provider device
  (`...opponent_provider.py:322`).

So a GPU profile with `compute=gpu-*` and `opponent_use_cuda=false` gives:

- live learner/search policy: CUDA;
- frozen checkpoint opponent inference: CPU;
- env/render/vector runtime: CPU/env-worker.

If `opponent_use_cuda=true`, each env process that constructs the frozen
opponent can lazy-load a GPU policy. With subprocess managers, that can create
GPU contention and makes attribution harder.

## 3. Base Vs Subprocess Timing Visibility

The env manager type is written into `create_config.env_manager.type`
(`...train.py:3963`, `:3969`) and surfaced in summaries
(`...train.py:4133`, `:4135`). The default is `subprocess`
(`...train.py:306`).

The profiler describes itself as single-process timing (`...train.py:687`) and
patches env methods, render helpers, vector env `step`, and vector runtime
`step_many` in the current process (`...train.py:1213`, `:1301`, `:1316`).

Visible with `base`:

- `collector_collect_sec`;
- `env_step_sec`, `env_reset_sec`, obs pack, stack update;
- render/downsample helpers;
- vector env/runtime step timers;
- frozen opponent work if timed through policy/model hooks in the same process.

Hidden or blurred with `subprocess`:

- per-worker env step internals;
- per-worker render/stack/vector-runtime costs;
- frozen opponent inference inside worker envs;
- env IPC/wait overhead as a separate bucket.

The parent still sees coarse `collector_collect_sec` (`...train.py:1397`) and
usually live policy/search/model/learner timers, but subprocess env cost is
mostly folded into collector wall time.

## 4. Most Useful Missing Timers

Minimal next profiling patch, in priority order:

1. Frozen opponent timers: `env_opponent_action_sec`,
   `frozen_opponent_select_action_sec`, `frozen_opponent_policy_forward_sec`,
   and `frozen_opponent_load_sec`. Add device/use_cuda/load summary to telemetry.
   This directly answers how much CPU opponent inference costs.
2. Env manager parent round trip: `env_manager_step_roundtrip_sec`,
   `env_manager_reset_roundtrip_sec`, and maybe `env_manager_wait_sec`, tagged by
   manager type and env count. This is the missing subprocess Amdahl bucket.
3. Context-split model inference time. The current profiler records model batch
   sizes by context, but not context-specific time. Add
   `model_initial_inference_in_mcts_search_sec`,
   `model_recurrent_inference_in_mcts_search_sec`, and learner/eval equivalents.
4. Worker-surviving env breakdown in `info` or telemetry:
   opponent action, vector step loop, reward/info assembly, stack/render, and
   telemetry write. This survives subprocess because `info` comes back from the
   worker.
5. Artifact/post-train timers outside `train_muzero_wall_sec` for checkpoint
   scan/mirror/action telemetry summary if end-to-end Modal profile wall matters.

## 5. Flags/Defaults That Can Profile The Wrong Lane

- `mode` must be `profile` or `train` on the stock branch. `mode=two-seat-selfplay`
  jumps to the old custom trainer path (`...train.py:582`, `:8738`, `:7752`).
- `opponent_policy_kind` defaults to fixed-straight (`...train.py:337`). For the
  frozen-opponent lane, explicitly pass
  `opponent_policy_kind=frozen_lightzero_checkpoint` and a checkpoint ref.
- `env_variant` defaults to the right stock lane, `source_state_fixed_opponent`
  (`...train.py:322`). `source_state_joint_action` is stock `train_muzero` but
  centralized joint control, not the frozen-opponent lane; it auto-normalizes the
  opponent kind to centralized-none when using defaults (`...train.py:422`).
- `env_manager_type` defaults to `subprocess` (`...train.py:306`). Good for
  throughput checks, bad for env attribution. Use `base` for component timing.
- Background eval and GIF defaults are on (`...train.py:344`, `:353`). For clean
  optimizer profiles, turn them off unless the artifact/eval path is the target.
- `disable_death_for_profile=true` changes the env death mode and is rejected
  outside profile mode (`...train.py:2825`; `...lightzero_env.py:424`). Do not
  compare it as normal training.
- `profile_cuda_sync_enabled=false` by default (`...train.py:298`), so CUDA
  sub-timers are not synchronization-accurate. Use a small sync profile when
  assigning GPU time precisely.
- Reused run IDs can trigger auto-resume checks; profile blocks resume unless
  explicitly allowed (`...train.py:3102`). Use fresh run IDs for timing.
