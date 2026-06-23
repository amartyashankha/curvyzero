# Code Inventory

Date: 2026-05-23

Purpose: separate useful pieces from profile-only experiments.

## Keep

- `src/curvyzero/training/compact_policy_row_bridge.py`
  - compact root/search/replay row contracts.
- `src/curvyzero/training/compact_search_service.py`
  - search service boundary and two-phase action/replay payload idea.
- `src/curvyzero/training/compact_rollout_slab.py`
  - compact proof harness.
- `src/curvyzero/env/observation_surface_contract.py`
  - current policy observation surface contract.

## Profile-Only Until Promoted

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/compact_torch_search_service.py`
- MCTX/JAX bridge work and compact slab rows.

These are useful for experiments, but they should not be described as Coach
training speedups unless they call the same loop or get promoted through a
clear gate.

## Confusion Risks

- old names: `body_circles_fast`, `fast_gray64_direct`, old two-seat custom
  trainer language;
- old docs that call profile rows "current";
- single-file profile launchers that contain renderer, search, replay,
  ceilings, Modal orchestration, and summaries in one place;
- untracked docs/scripts that look as current as real source.

## Cleanup Rule

Prefer labeling and archiving over deleting while other agents are active.

If a path is not Coach-ready, say so at the top of the file or doc.

