# Latest Renderer Surface Audit - 2026-05-22

Scope: read-only audit of the current renderer/observation surface so optimizer profiles do not accidentally measure an old path. I did not edit source and did not touch live runs.

## Short Answer

The current trusted policy observation surface is:

```text
browser_lines + simple_symbols, [4,64,64], player perspective, cpu_oracle
```

The current fast optimizer surface is:

```text
jax_gpu_persistent_policy_framebuffer_profile + direct_gray64,
browser-line semantics + simple_symbols, profile-only
```

`body_circles_fast` still exists in generic renderer code and old notes, but the current source-state training env only supports `browser_lines` for policy observations. The tournament path also requires the default policy surface and default CPU policy-observation backend. A profile command that says `body_circles_fast` is stale unless it is deliberately hitting a legacy/generic render benchmark.

## Current Sources Of Truth

Files checked:

- `src/curvyzero/env/observation_surface_contract.py`
- `src/curvyzero/contracts/curvytron.py`
- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/env/vector_trainer_observation.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `src/curvyzero/training/lightzero_config_builder.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
- `scripts/run_curvytron_hybrid_observation_profile_manifest.py`
- `src/curvyzero/tournament/curvytron/contracts.py`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`

Current global policy constants:

- `POLICY_TRAIL_RENDER_MODE = "browser_lines"`
- `POLICY_BONUS_RENDER_MODE = "simple_symbols"`
- `POLICY_RENDER_SURFACE_LABEL = "browser_lines+simple_symbols"`
- `POLICY_STACK_SHAPE = (4, 64, 64)`
- `DEFAULT_POLICY_OBSERVATION_BACKEND = "cpu_oracle"`
- `CURRENT_RELIABLE_POLICY_OBSERVATION_BACKEND = "cpu_oracle"`
- `EXPERIMENTAL_SCALAR_POLICY_OBSERVATION_BACKEND = "jax_gpu"`
- production direction is a batched GPU backend, not the scalar `jax_gpu` env backend.

Current CurvyTron default training knobs from `contracts/curvytron.py`:

- decision source frames: `1`
- simulations: `8`
- collector envs: `256`
- episodes per collect: `256`
- learner batch size: `64`
- policy batch size: `8`
- default compute: `gpu-l4-t4-cpu40`

## Trusted Policy Surface

Training and tournament both point at the policy contract.

The source-state LightZero env now restricts policy surface modes:

```text
SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES = ("browser_lines",)
SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES = ("simple_symbols",)
```

That means `body_circles_fast` is not a valid current training policy surface. It can still exist as a generic renderer constant, but it is not accepted by the current source-state env path.

The trusted CPU surface does this:

1. Render source-state data into a 704x704 RGB canvas-like image.
2. Use connected rounded line semantics for trails from `visual_trail_*` arrays when available.
3. Draw simple symbolic bonus marks instead of browser sprite-sheet art.
4. Draw heads after bonuses.
5. Convert RGB to BT.601 grayscale.
6. Area-downsample to 64x64.
7. Keep a 4-frame player-perspective stack for LightZero.

This is not proven browser-pixel exact. It is the current policy observation contract.

## Fast/GPU Surface

The optimizer fast lane is profile-only unless explicitly promoted later.

The current profile-only persistent GPU backend is:

```text
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
render_mode=browser_lines
bonus_render_mode=simple_symbols
```

Important guardrails:

- `source_state_batched_observation_boundary_profile.py` marks these paths as `profile_only=true`, `calls_train_muzero=false`, and `touches_live_runs=false`.
- `direct_gray64` is only allowed for canary/profile/hybrid probes.
- `jax_gpu_persistent_policy_framebuffer_profile` requires `render_surface="direct_gray64"`.
- `scripts/run_curvytron_hybrid_observation_profile_manifest.py` rejects manifests that are not profile-only or that touch live runs.
- `mctx_synthetic_benchmark.py` uses this persistent GPU backend by default for compact visual sample profiles.

Plainly: the fast GPU path is the right optimizer experiment surface, but it is not yet the production trainer/tournament surface.

## Fidelity Preserved

The current GPU/profile surface preserves the things that matter most for a policy-space probe:

- 2-seat/player perspective.
- `[4,64,64]` observation shape.
- Browser-line trail semantics rather than `body_circles_fast`.
- `visual_trail_*` position events, trail owner, radius/thickness, write cursor, and break-before connectivity.
- Simple-symbol bonus identities.
- Bonus symbols overwrite trail pixels in the symbol mask.
- Heads are drawn over bonus symbols.
- Tests cover all twelve bonus identities, bonus-over-trail, and heads-over-bonus for the direct-gray64 renderer.

## Fidelity Lost Or Approximate

The GPU `direct_gray64` path is a policy-space approximation:

- It does not render the full 704x704 RGB source canvas and then area-downsample in the same way as the CPU oracle.
- It is not browser-pixel exact.
- It does not use the original browser bonus sprite sheet or bounce animation.
- It does not preserve every antialiasing/downsample detail from the CPU `browser_lines + simple_symbols` oracle.
- It is validated by canaries and tolerant divergence checks, not exact equality to the CPU oracle.

This is acceptable for optimizer profiling and may be acceptable for future training after promotion gates, but it should not be mislabeled as the current trusted production surface.

## Tournament Wiring

Tournament code is currently fail-closed:

- Tournament defaults come from `curvyzero.env.observation_surface_contract`.
- It requires the default trail mode and bonus mode.
- It requires the default policy observation backend, currently `cpu_oracle`.
- It extracts policy surface metadata from checkpoints and rejects mismatches.

So tournament should not silently evaluate a `body_circles_fast` or GPU-profile checkpoint as if it were the trusted surface.

## Training Wiring

The canonical launcher `lightzero_curvyzero_stacked_debug_visual_survival_train.py` passes policy observation settings through the LightZero config builder.

Current train-facing default is still:

```text
source_state_trail_render_mode=browser_lines
source_state_bonus_render_mode=simple_symbols
policy_observation_backend=cpu_oracle
```

The scalar env `policy_observation_backend="jax_gpu"` branch exists, but it is experimental and scalar. It is not the same thing as the fast batched persistent GPU profile backend.

## Stale-Path Risk

Main risk is not the current canonical code. Main risk is old commands/docs.

Stale or confusing names still present:

- `body_circles_fast`: generic renderer mode; old speed lens; not current trusted training surface.
- `fast_gray64_direct`: old/custom naming; translate mentally to current profile-only `direct_gray64` only if the command also uses the persistent GPU profile backend and canary guards.
- `policy_observation_backend="jax_gpu"`: scalar experimental env backend; not the batched persistent GPU profile backend.
- old custom/two-seat scripts and docs: may mention surfaces that no longer represent the current trusted path.

Safe command signature for optimizer profile-only fast surface:

```text
--observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile
--render-surface direct_gray64
--hybrid-observation-canary
```

Safe command signature for trusted production trainer/tournament surface:

```text
policy_observation_backend=cpu_oracle
policy_trail_render_mode=browser_lines
policy_bonus_render_mode=simple_symbols
```

## Amdahl Notes

Do not call the whole `env_step_sec` bucket "game mechanics." In the hybrid/compact profile path, that bucket can include actor render-state handoff, production-to-compact conversion, renderer/stack update, root-batch handoff, and reset/final-observation bookkeeping.

The actual `VectorMultiplayerEnv.step()` mechanics are still CPU. They include source physics, collision, bonus effects, and visual-trail event writes. Recent timing notes show this is not the largest current bucket in compact visual profiles, but it can become important for very long no-death trajectories because trail/collision state grows.

Current optimizer priority after this audit:

1. Keep renderer-surface labels precise.
2. Keep profiling the persistent GPU `direct_gray64` lane separately from production CPU oracle.
3. For real speedups, keep attacking the larger collect/search/replay boundary after the render-state handoff is measured cleanly.
4. Avoid spending new effort on `body_circles_fast` unless the explicit experiment is an old-renderer ablation.

