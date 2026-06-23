# Compact Ownership Plan

Date: 2026-05-23

Purpose: make "compact ownership" concrete enough to implement or kill.

## Current State

The compact path exists as a profile-side proof system:

- `CompactRootBatchV1`;
- `CompactSearchServiceV1`;
- `CompactSearchResultV1`;
- `CompactReplayIndexRowsV1`;
- `CompactRolloutSlab`;
- direct CTree, compact Torch, and MCTX/JAX comparators.
- compact Torch now owns a real two-phase profile path: action-only hot return,
  delayed replay payload flush, and compact identity checks. It is still not the
  Coach training backend.

It is not the Coach training backend.

## Gate A: Stock Full-Loop Baseline

Use the existing stock profile wrapper. Do not add compact ownership here.

Files:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `scripts/run_curvytron_optimizer_profile_manifest.py`
- `scripts/summarize_curvytron_optimizer_profile_results.py`

Acceptance:

- `called_train_muzero=True`;
- stock MuZero trainer entrypoint, or the explicit RND entrypoint when RND is
  the tested condition;
- same collector count, `n_episode`, batch size, simulations, reward/RND,
  observation backend, source horizon, death mode, and sidecar settings;
- report `train_muzero_wall`, env steps, MCTS roots, learner calls, replay
  sample calls, and RND metrics when enabled.

## Gate B: Compact-Owned Proof

Stay outside live Coach training. Build a separate profile lane where compact
arrays own collection, search, replay refs, and learner/RND probes.

Current implementation progress:

- slab final flush/close exists;
- cumulative committed replay-index rows exist;
- active-seat-only action checking exists;
- multi-group materialization and sampling helpers exist;
- profile output now reports stored committed groups, stored rows, and dropped
  pending tail searches;
- focused compact tests pass locally.
- `CompactTorchSearchServiceV1` is the real compact search-owner candidate. It
  is wired through compact rollout slab mode with `run_action_step()` and
  `flush_replay_payload()`, so the hot env step only needs selected actions.
- `FixedShapeBatchedSearchOwnerV0` remains useful as a fixed-shape first-legal
  control/floor. It is a boundary probe, not real MCTS.

Likely files:

- `src/curvyzero/training/compact_rollout_slab.py`
  - add final flush/close semantics for pending search payloads;
  - expose cumulative replay index rows.
- `src/curvyzero/training/compact_torch_search_service.py`
  - keep compact search tensors device-owned until action readback;
  - expose delayed replay payload flush;
  - count hidden host copies honestly.
- `src/curvyzero/training/compact_policy_row_bridge.py`
  - add helpers for sampling multiple compact replay-index groups;
  - keep full materialization at sampler edge.
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  - route compact Torch through the two-phase slab adapter;
  - keep `fixed_shape_search_owner` labeled as a control.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
  - extend compact slab mode into a compact-owned proof mode.
- `src/curvyzero/training/compact_owned_loop_profile.py`
  - new isolated ownership loop, if the existing profile file gets too large.
- `src/curvyzero/training/exploration_bonus.py`
  - reuse the existing latest-frame extraction for RND; do not invent a second
    RND model.

Next implementation boundary:

- Harden the existing compact-owned trainer-like profile path before opening
  another architecture lane.
- Do not wire it into live Coach training first.
- Keep `search_feedback` for learning-shaped rows; `scripted_random` remains
  timing-only because it drops replay commits.
- The sample gate now uses a bounded compact replay ring with explicit pair
  capacity, eviction counters, empty-ring behavior, RNG-order-preserving
  sampling, and canonical sample metadata.
- Local compact replay sampling can require committed successor targets for the
  learner gate.
- Local learner-gate modes are now explicit: `toy_probe` for the old tiny timing
  consumer, `compact_muzero` for a one-step MuZero-shaped optimizer update.
- Modal/grid/result wiring for those learner-gate modes has a first passed
  H100 proof row: `optimizer-compact-muzero-gate-smoke-20260526`, row `001`.
- Next, run a larger compact MuZero profile wave and carry real RND
  latest-frame/update/estimate cadence through the compact path, or explicitly
  mark RND mocked.
- Make terminal/autoreset checks match trainer semantics, not profile no-death
  convenience.
- Keep fail-closed metadata on every row: whether it calls `train_muzero`, what
  stock path it replaces, whether scalar timesteps were materialized, and what
  replay/learner/RND work actually ran.

Acceptance:

- scalar timestep materialization is off in compact collection;
- selected action from search at step `k` drives env step `k+1`;
- replay rows stay hidden until replay payload is attached;
- action steps fail closed before env action if schema, handle, sidecars,
  legality, or selected-action digest are stale;
- final observation and autoreset identity are checked;
- RND is either real and measured or clearly mocked;
- learner is either real sample consumption or clearly mocked with finite loss
  and no claim of optimizer state parity.

## Gate C: Promotion Or Kill

Only after Gate B is promoted into a matched trainer-profile denominator and
beats Gate A there. Compact profile wins alone are not comparable to stock
`train_muzero` profile wins.

Possible files if promoted:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  - add profile-only compact backend guard; reject normal `mode="train"` first.
- `src/curvyzero/training/lightzero_config_builder.py`
  - expose compact ownership metadata without changing stock defaults.
- `scripts/run_curvytron_optimizer_profile_manifest.py`
  - include compact-owned attestation fields.
- `scripts/summarize_curvytron_optimizer_profile_results.py`
  - refuse to summarize profile-only rows as Coach speedups.

Promotion criteria:

- meaningful speedup over Gate A after learner/RND accounting, preferably
  `1.5x+`;
- no action, mask, final observation, autoreset, RND latest-frame, or replay
  identity mismatch;
- no hidden fallback to stock scalar materialization;
- no live Coach run touched.

Kill criteria:

- less than `25-30%` speedup after scalar materialization is removed;
- frequent fallback to Python row materialization;
- RND or replay correctness requires rebuilding full scalar timesteps in the
  hot path;
- semantic mismatch is large and not accepted as a deliberate algorithm change.
