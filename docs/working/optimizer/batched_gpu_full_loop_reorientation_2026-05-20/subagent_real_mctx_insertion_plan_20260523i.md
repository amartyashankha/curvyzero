# Real MCTX Shadow Insertion Plan, 2026-05-23i

Scope: bounded sidecar inspection. I did not edit code, start Modal jobs, touch
live runs, or change trainer defaults. Trusted Coach training remains
read-only; this is optimizer profile-only.

## Current Shape

- `src/curvyzero/training/mctx_compact_search_service.py` is already the right
  boundary: `MctxCompactSearchServiceV1.run()` validates `CompactRootBatchV1`,
  moves active roots/masks to JAX, calls a jitted `run_search`, reads back
  selected actions, visit policy, and root value, then returns a checked
  `CompactSearchResultV1`.
- The smallest real-model insertion point is
  `MctxCompactSearchServiceV1._backend()`. Today it builds toy params plus local
  toy `representation`, `prediction`, and `recurrent_fn` functions before
  calling `mctx.gumbel_muzero_policy`.
- `src/curvyzero/training/lightzero_jax_shadow_model_parity.py` already has the
  useful model surface: `JaxMuZeroShadowModel.initial_inference`,
  `recurrent_inference`, `representation`, `dynamics`, `prediction`, and
  checkpoint coverage helpers.
- The only profile instantiation of `MctxCompactSearchServiceV1` is in
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  inside the compact rollout slab branch. Current `gpu_mctx_image` installs
  JAX/MCTX but not LightZero/Torch or the runs volume, so real checkpoint mode
  needs a separate combined profile image/path.

## Smallest Clean Insertion

Keep `CompactSearchServiceV1` and `CompactRolloutSlab` unchanged. Add an
optional LightZero JAX shadow backend inside `MctxCompactSearchServiceV1`; leave
the toy backend as the default for existing rows.

Shape:

```text
CompactRootBatchV1
-> MctxCompactSearchServiceV1.run(...)
-> _backend(..., backend="toy" | "lightzero_jax_shadow")
-> mctx.gumbel_muzero_policy(...)
-> CompactSearchResultV1
```

Implementation sketch:

- Add a constructor/config path that accepts a prebuilt
  `JaxMuZeroShadowModel` or a strictly loaded state dict plus checkpoint labels.
  Do checkpoint/Torch loading once during service construction or a small
  factory, not inside `run()`.
- Split current `_backend()` into toy and shadow helpers. The shadow helper
  closes over `JaxMuZeroShadowModel` and returns the same `(jnp, mctx, params,
  run_search)` shape.
- In shadow `run_search`, normalize uint8 roots to float32 `[0,1]` exactly like
  the LightZero profile path, call `shadow.representation(obs)`, then
  `shadow.prediction(latent)`, and pass the latent as the MCTX root embedding.
- In shadow `recurrent_fn`, call `shadow.dynamics(latent, action)` and
  `shadow.prediction(next_latent)`.
- Convert LightZero categorical value/reward logits to scalar JAX arrays before
  handing them to MCTX. The existing NumPy
  `inverse_scalar_transform_logits()` is the reference; the search path needs a
  JAX equivalent.

## Likely Edit Map

- `src/curvyzero/training/mctx_compact_search_service.py`
  - `MctxCompactSearchConfig`: add backend/checkpoint metadata knobs or a
    compact `model_backend` enum.
  - `MctxCompactSearchServiceV1.__init__`: accept optional shadow model/state
    dict and preserve default toy behavior.
  - `MctxCompactSearchServiceV1._backend`: dispatch to toy vs real shadow
    backend.
  - New helper: JAX LightZero support-logit scalar transform.
  - Metadata: add `mctx_compact_search_service_model_backend`,
    checkpoint ref/SHA/load summary, and coverage status.
- `src/curvyzero/training/lightzero_jax_shadow_model_parity.py`
  - Add/reuse a JAX scalar-transform helper so value/reward logits become MCTX
    scalars without NumPy readback.
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  - Add a combined profile image with `mctx`, `LightZero`, `torch`, JAX, and
    `cloudpickle`.
  - Add immutable checkpoint ref/state-key config for MCTX shadow mode.
  - Mount/read the runs volume only for this profile-only checkpoint mode.
  - Pass the shadow backend into the existing instantiation at the compact slab
    branch; keep toy MCTX rows working.
- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
  - Add pass-through args for the shadow checkpoint ref/state key if manifest
    rows need to launch this mode.
- Tests likely to extend:
  - `tests/test_mctx_compact_search_service.py`
  - `tests/test_lightzero_jax_shadow_model_parity.py`
  - `tests/test_source_state_batched_observation_boundary_profile.py`
  - `tests/test_curvytron_hybrid_observation_profile_grid_builder.py`
  - `tests/test_curvytron_hybrid_observation_profile_manifest_runner.py`

## Validation Risks

- LightZero value/reward outputs are categorical support logits, while MCTX
  wants scalar value/reward. A wrong transform can make search look fast but
  semantically useless.
- Recent parity says raw root representation has small drift. The real MCTX
  path starts from JAX observation representation, not a PyTorch latent, so that
  drift can compound through search.
- MCTX/Gumbel MuZero is still not LightZero CTree. The first proof should claim
  legal selected actions, compact replay identity, and profile timing only.
- Current MCTX Modal image cannot load the real checkpoint because it lacks
  LightZero/Torch and the runs mount.
- Keep immutable checkpoint refs only. Do not accept `latest.pth.tar` or
  `ckpt_best.pth.tar`.
- Separate one-time checkpoint load/JAX compile from steady-state search timing.

## Tiny Smoke To Run After Wiring

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --hybrid-observation-canary \
  --compute gpu-h100 \
  --batch-size 4 \
  --actor-count 2 \
  --steps 2 \
  --warmup-steps 1 \
  --trail-slots 128 \
  --body-capacity 128 \
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-compact-rollout-slab-probe \
  --hybrid-mctx-compact-search-probe \
  --hybrid-mctx-num-simulations 2 \
  --hybrid-mctx-shadow-checkpoint-ref training/lightzero-curvytron-visual-survival/cz26a-r001-out0-n0-imm0-b20w05r1/attempts/try-cz26a-r001-out0-n0-imm0-b20w05r1/train/lightzero_exp/ckpt/iteration_260000.pth.tar
```

Expected first-pass read: `ok=true`, JAX backend is GPU, compact slab commits
all active root index rows, all selected actions are legal, profile labels still
say profile-only/not train_muzero, and telemetry identifies
`model_backend=lightzero_jax_shadow`.
