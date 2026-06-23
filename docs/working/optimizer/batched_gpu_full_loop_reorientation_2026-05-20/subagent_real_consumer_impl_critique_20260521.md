# LightZero collect-forward real-consumer critique - 2026-05-21

Scope: source audit only. No live Coach runs, Modal launches, checkpoint loads,
tournaments, or source edits were performed.

## Summary

The new canary is mostly in the right containment box: it is behind the
profile-only hybrid path, constructs a scratch `MuZeroPolicy`, calls
`policy.collect_mode.forward`, flattens `[B,P,4,64,64]` to `[B*P,4,64,64]`,
decodes each returned root, and preserves `materialized_timestep_count=0` when
the scalar edge is disabled.

The main risks are contract drift, not live-run leakage:

1. the direct Modal CLI defaults can run the LightZero probe on `float32` stacks
   and the older block render surface;
2. the scratch policy is built as the source-state fixed-opponent env, but the
   collect call passes player ids as `to_play` instead of the scalar env's
   `-1` convention;
3. terminal rows can hand LightZero zero action masks because the probe decodes
   every row without filtering active roots;
4. some timing/byte telemetry names still sound more device-resident than the
   implementation really is.

Plain interpretation: this canary answers "what happens if the current
pre-scalar CurvyTron stack is handed to the public LightZero collect call?" It
does not answer "what would a fully device-resident MuZero search cost?" and it
does not answer "what is stock `train_muzero` wall time?" Treat it as the
boundary check between our batched observation work and LightZero's existing
collection/search consumer.

## Findings

### 1. LightZero mode does not enforce the intended uint8/direct canary contract

The manifest builder defaults are correct for the intended row:

- `--hybrid-stack-storage-dtype uint8`
- `--render-surface direct_gray64`
- `--observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile`

Refs:

- `scripts/build_curvytron_hybrid_observation_profile_grid.py:112`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py:116`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py:249`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py:250`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py:253`

But the Modal local entrypoint defaults remain:

- `hybrid_stack_storage_dtype="float32"`
- `observation_renderer_backend=SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND`
- `render_surface=RENDER_SURFACE_BLOCK_704_GRAY64`

Refs:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:4528`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:4541`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:4542`

The LightZero branch validates temperature/epsilon and rejects device-latest,
but it does not require `stack_storage_dtype == "uint8"` or the policy-space
direct surface:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1894`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1903`

By contrast, the resident chunk probe explicitly enforces uint8:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1904`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1911`

Impact: a direct command with only `--hybrid-lightzero-collect-forward-probe`
can produce telemetry that looks like the real-consumer canary while not
testing the advertised pre-scalar `uint8[B,2,4,64,64]` handoff or the intended
direct gray64 policy surface.

Recommended guard: when `hybrid_lightzero_collect_forward_probe` is true,
require `hybrid_stack_storage_dtype == "uint8"` and either require or
explicitly label any non-`direct_gray64` renderer as a different experiment.

### 2. `to_play` semantics are ambiguous against the scratch policy config

The probe passes player ids as `to_play`:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3731`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3735`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3763`

That matches the two-seat current-policy smoke pattern, but the scratch policy
constructed here uses `ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT`:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3616`

The source-state fixed-opponent scalar env emits `to_play = -1`:

- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1249`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1252`

The scalar materializer also uses `-1` for every flattened row:

- `src/curvyzero/training/source_state_batched_observation_mock_collector.py:799`
- `src/curvyzero/training/source_state_batched_observation_mock_collector.py:803`

Impact: if LightZero's non-board MuZero path ignores `to_play`, this is harmless.
If it does not, the probe is not matching the fixed-opponent source-state policy
surface it builds. The test currently locks in `[0, 1, 0, 1]`:

- `tests/test_source_state_batched_observation_boundary_profile.py:447`

Recommended guard: either document this as deliberately following the two-seat
current-policy collect semantics and build the matching policy/config, or pass
`[-1] * root_count` for the fixed-opponent scratch policy. At minimum, report a
telemetry field distinguishing `to_play_policy=player_id` from
`to_play_policy=always_-1_non_board_game`.

### 3. Terminal/zero-mask rows are not filtered before collect-forward

The manager runs the pre-scalar probe before optional scalar materialization and
passes the merged actor `action_mask` directly:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py:428`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py:433`

The LightZero probe then forwards every root and only detects illegal selected
actions after the call:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3730`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3766`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3777`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3803`

In the default no-death/max-ticks profile this likely stays fine, but a
terminal row can have an all-zero action mask. A real collect search over a
zero-mask root may crash, choose an illegal fallback, or skew timings.

Recommended guard: either keep this probe explicitly no-terminal and assert
`action_mask.any(axis=-1).all()` before calling LightZero, or add an active-root
filter and report `roots_requested`, `roots_consumed`, and `terminal_roots_skipped`.

### 4. Telemetry is useful, but a few fields overclaim precision

Good labels:

- `semantics = "lightzero_collect_forward_search_cpu_tree"` is explicit and
  avoids claiming device-resident MCTS.
- The known gap says LightZero tree internals still cross to CPU.

Refs:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3688`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2051`

Telemetry caveats:

- `device_sec` is set to the whole `collect_mode.forward` wall time, which
  includes CPU tree work, not only device kernels.
- `model_eval_count = root_count * num_simulations` is inferred, not observed.
- `host_to_device_bytes` includes the NumPy action mask even though the mask is
  passed to LightZero as a NumPy array, not explicitly copied as a torch tensor.
- `readback_sec` is mostly output serialization size measurement; tensor
  detach/CPU conversion happens earlier in `_plain_lightzero_value`.

Refs:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3814`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3826`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3832`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3836`

Recommended wording: keep the existing generic timing fields for grid
compatibility, but add LightZero-specific aliases such as
`lightzero_collect_forward_wall_sec`, `lightzero_search_cpu_tree=true`,
`estimated_model_eval_count`, and `action_mask_host_numpy=true`.

### 5. Amdahl read: useful, but only for this boundary

If a medium row shows most wall time under `batched_stack_probe_sec` or
`lightzero_consumer_collect_forward_sec`, the next bottleneck is not the CurvyTron
renderer in this profile. It is the public LightZero collect/search path plus
the conversions around that path.

That is still a real answer. It means a faster renderer can help only up to the
fraction still spent in `observation_sec`/`renderer_render_sec`. If the collect
consumer dominates, the higher-leverage choices are:

- reduce unnecessary scalar/materialization around the consumer;
- isolate model initial inference from CPU tree/search in a separate probe;
- investigate a batched/device-resident search path as a larger architecture
  change;
- compare against stock `train_muzero` only after the boundary profile tells us
  where the wall moved.

Do not read `batched_stack_probe_device_sec` as pure GPU time. In this
implementation it is populated from `device_sec`, and for the LightZero probe
`device_sec` is the whole `collect_mode.forward` wall time. That wall includes
GPU model work, CPU tree/search work, internal tensor-to-CPU conversions, Python
bookkeeping, and LightZero output construction.

### 6. Warmup and measured window are basically sound

The manager calls the batched-stack probe inside `manager.step(...)`, including
warmup iterations. The outer loop skips accumulating warmup timings, so the
LightZero policy/search path is warmed before the measured window. That is the
right shape for Modal/GPU timing.

Caveat: explicit `torch.cuda.synchronize(...)` calls make the timing splits
easier to reason about but more conservative than an overlapped production
pipeline. That is acceptable for this canary, as long as the result is read as a
boundary wall-time profile rather than a perfect lower bound.

### 7. One small type/documentation drift

`HybridObservationProfileStep.batched_stack_probe_telemetry` is still annotated
as `dict[str, float]`, while the LightZero probe now intentionally returns
strings, booleans, lists, and nested dicts in telemetry. Runtime behavior is
fine because `_plain_telemetry_value(...)` preserves those values, but the type
hint should become `dict[str, Any]` in a cleanup pass.

## Tests

Strong coverage added:

- fake collect-mode test verifies flattening to `[B*2,4,64,64]`, mask flattening
  to `[B*2,3]`, uint8 normalization, player-id `to_play`, dense `ready_env_id`,
  root decoding, and illegal-action count.
- manifest test verifies the LightZero flag, consumer simulation forwarding, and
  synthetic probe simulation zeroing.

Refs:

- `tests/test_source_state_batched_observation_boundary_profile.py:370`
- `tests/test_source_state_batched_observation_boundary_profile.py:456`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py:93`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py:111`

Missing coverage I would add before trusting more rows:

- direct CLI/config validation for LightZero mode requiring uint8/direct surface,
  or an explicit test showing direct defaults are intentionally allowed;
- a zero-mask/terminal-row test;
- a real `run_hybrid_observation_profile(... materialize_scalar_timestep=False,
  batched_stack_probe=fake_lightzero_probe)` assertion that the compact result
  keeps `materialized_timestep_count=0` and reports the LightZero semantics;
- output-shape variants for LightZero dict-of-arrays, dict keyed by string ids,
  and list outputs;
- an illegal action fake output test that proves the legality check fails loud.

## Side-effect review

I did not find evidence that this path touches live Coach runs. The LightZero
branch:

- constructs a scratch policy;
- uses `compile_config(..., save_cfg=False)`;
- sets a `/tmp` telemetry path;
- does not call `train_muzero`;
- passes `opponent_checkpoint=None`, `opponent_snapshot_ref=None`, and
  `opponent_checkpoint_state_key=None`;
- reports `profile_only=true`, `calls_train_muzero=false`,
  `stock_lightzero_integrated=false`, and `touches_live_runs=false`.

Refs:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3598`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3601`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3649`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3655`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:3658`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2009`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2015`

The manifest builder writes local manifest/command files when not in
`--stdout-only` mode, but does not submit or launch runs by itself.
