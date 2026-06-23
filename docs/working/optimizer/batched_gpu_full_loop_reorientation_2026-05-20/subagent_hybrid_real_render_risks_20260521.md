# Hybrid Real Render Risk Note

Date: 2026-05-21

Scope: implementation-risk note only. No code inspected here was changed.

## Context

`src/curvyzero/training/source_state_hybrid_observation_profile.py` is currently
an in-process, profile-only hybrid scaffold. Actors step real
`VectorMultiplayerEnv` rows, the parent merges compact row metadata, then writes
zero `[B,P,4,64,64]` stacks and materializes scalar LightZero-shaped rows.

The tempting next step is to replace the zero stack update with the existing
batched renderer protocol from `source_state_batched_observation_profile.py` or
the Modal-backed candidate renderer. That is the right direction for profiling,
but it has several correctness traps.

## Main Risks

- Row/player order can silently flip. The training-side protocol wants
  row-major output rows: `[(row0,p0), (row0,p1), ...]`. The Modal JAX two-view
  render path naturally produces view-major frames and then converts them with
  `_view_major_to_row_major_frames`. A direct hybrid integration must make the
  expected request/output order explicit and tested at the hybrid boundary, not
  rely on a downstream reshape looking plausible.

- Scalar env ids must stay aligned with rendered views. The hybrid scaffold maps
  scalar env id as `row * player_count + player`, with `policy_env_row` equal to
  `[0,0,1,1,...]` and `policy_player` equal to `[0,1,0,1,...]`. If rendered
  frames arrive player-major, actor-local-row-major, or partition-major, the
  final LightZero rows will contain the wrong player's perspective while all
  shapes still pass.

- Terminal final observations are currently absent from the hybrid zero path.
  The batched profile and trainer surface capture visual final observations
  before reset. Real render in the hybrid scaffold must decide whether terminal
  rows render before actor autoreset, after actor autoreset, or from a saved
  terminal source-state snapshot. Rendering after `autoreset_done_rows` would
  train on the next episode's first frame as the previous episode's terminal
  frame.

- Partial rows are correctness-friendly but timing-hostile. The current dynamic
  JAX renderer accepts partial row/player requests by rendering the full batch
  and gathering the requested frames. That is useful for reset-row paths and
  final-observation paths, but the hybrid profile must record when it happened;
  otherwise a partial-row-heavy run may look like it made efficient batched
  requests while actually doing full renders behind the protocol.

- Actor partitions can break full-batch assumptions. The zero scaffold merges
  actor payloads back into global row order and validates coverage. A renderer
  backed mode must render from the merged global state, or it must preserve a
  stable mapping from actor-local rows to global rows before stack writes. Any
  per-actor render shortcut needs tests with uneven actor partitions and
  nontrivial `actor_count` so row 2 from actor 1 cannot masquerade as global
  row 0.

- Host copies can erase the measured win. The current promising JAX boundary
  still does CPU production-state conversion, owner-ordered packing,
  host-to-device copies, device render sync, device-to-host readback,
  view-major-to-row-major conversion, host float32 stack writes, and scalar
  LightZero object materialization. If the hybrid adds actor IPC plus pickled
  source-state payloads on top, it may profile IPC and copies rather than render
  speed. Telemetry should separate pack, H2D, device render, D2H, row-order
  conversion, stack update, final-observation copy, and scalarization.

- Import boundaries matter. `source_state_hybrid_observation_profile.py` lives
  under `curvyzero.training`; importing Modal internals directly would couple
  a local training/profile module to `curvyzero.infra.modal` and its JAX/Modal
  setup assumptions. Prefer a small renderer object passed into the hybrid
  manager, implementing `SourceStateBatchedObservationRenderer`, with any Modal
  dynamic renderer adapter constructed by scripts or infra code.

- Hidden CPU fallback would invalidate the experiment. The existing profile
  facade and trainer surface reject `jax_gpu_batched_profile` unless an explicit
  renderer is provided. The hybrid real-render mode should keep the same rule:
  profile-only CPU oracle is fine when named as CPU oracle; GPU candidate mode
  must fail closed if the renderer is missing or reports the wrong backend.

## Test Gates Before Replacing Zeros

- A hybrid row/player sentinel renderer test: fill output frame `(row,player)`
  with a unique value, run `batch_size > actor_count`, and assert
  `policy_env_id`, `policy_env_row`, `policy_player`, `flat_obs`, timestep
  info, and stack latest channels all agree in row-major scalar order.

- An uneven partition test: use a batch/actor split such as `batch_size=5`,
  `actor_count=2`, then verify global row order after merge and render. The
  test should fail if actor-local row ids are used as global ids.

- A terminal/autoreset test with `max_ticks=1`: assert final observations are
  real rendered stacks from the terminal state before reset, with row masks and
  rows matching `terminal_global_rows`. Also assert post-reset observation, if
  modeled, is not reused as `final_observation`.

- A partial-row telemetry test: force a reset-row or terminal-row render subset
  and assert the result marks partial requests distinctly from full row-major
  requests. This protects Amdahl interpretation.

- A dependency-boundary test or import audit: `source_state_hybrid_observation_profile.py`
  should depend on the renderer protocol and training/env modules, not on
  `curvyzero.infra.modal.*`.

- A no-hidden-fallback test: GPU candidate mode without an explicit renderer,
  or with a renderer whose `backend_name` is not the expected profile backend,
  should raise before any profile loop starts.

## Practical Integration Shape

Keep the hybrid manager profile-only and inject a renderer:

```text
actors step CPU env rows
-> parent merges global source state/metadata
-> renderer.render(SourceStateBatchedRenderRequest)
-> parent writes [B,P,4,64,64] stack in row-major order
-> parent materializes scalar LightZero-shaped rows
```

Do not make the first real-render patch solve subprocess services, trainer
defaults, or live `train_muzero` integration. The first success criterion is
boring but valuable: exact row/player/final-observation semantics with honest
timing buckets, using the same scalar materialization surface as the zero
scaffold.
