# Observation Path Purge - 2026-05-11

Scope: stale wording around the active 2P visual/trainer observation path. This
is a doc cleanup ledger, not a new runtime contract.

## Current Guardrail

There is one active product visual path under hardening:

```text
source-state browser-like RGB64 raw frame -> deterministic gray64 -> frame stack
```

The trail renderer split for that path is now explicit:

- `browser_lines`: default fidelity path for connected rounded browser-style
  trails.
- `body_circles_fast`: explicit approximation for speed, profiling, and old
  bead-style exact fixtures.

This split is a source-state/native renderer fact, not a browser pixel parity
claim. Real browser/canvas parity still needs a separate harness and golden
browser canvas reference.

`bonus64` and any "rich" tensor are diagnostic/proof surfaces for hidden bonus
facts only. They are not a parallel product observation path and must not be
described as the trainer default.

Browser/canvas pixel parity is optional later debug evidence, not P0. The
current source-state gray64 gate proves covered source-shaped and vector states
produce the same model-observation raster. It does not prove browser pixels.

Trainer wrapper, replay, and final-observation propagation remain open unless a
doc names the exact trainer/replay proof. A visual harness pass alone is not a
trainer-readiness claim.

## Purged In This Pass

- `active_lanes.md`: added the front-door guardrail, renamed the current gate as
  source-state gray64, and fenced bonus64/rich tensors as diagnostics only.
- `visual_fidelity_harness_2026-05-11.md`: tightened the scope to source-state
  RGB64 -> gray64 -> stack, removed wording that made canvas-gray64 sound like
  browser pixel parity, and kept trainer propagation explicitly open.
- `two_player_fidelity_gap_catalog_2026-05-11.md`: separated the source-state
  gray64 product image path from diagnostic bonus64 and left two-seat
  trainer/replay promotion open.
- `remaining_reconstruction_gap_catalog_2026-05-11.md`: reframed source-state
  visual evidence as model-raster plumbing only, not browser pixels or proven
  trainer/replay propagation.
- `browser_rendering_spec_2026-05-11.md`: recorded the two renderer modes,
  marked `browser_lines` as the default source-state renderer, and kept browser
  pixel parity as a separate missing harness.
- This update: made `browser_lines` default fidelity vs `body_circles_fast`
  explicit approximation wording consistent across the scoped status docs.

## Still Risky Outside This Write Scope

- `coverage_tracker.md`: the visual row still has older scenario-count wording
  and should be synced to the 34-scenario source-state gray64 gate.
- `full_curvytron_one_shot_spec_2026-05-10.md`: still mixes debug occupancy
  gray64 and LightZero stacked-input wording; review before treating it as a
  current product-path source.
- `optimizer_handoff_2026-05-10.md`,
  `lightzero_env_requirements_2026-05-10.md`, and
  `optimizer_visual_tensor_handoff_2026-05-10.md`: mostly compatible, but should
  point at this guardrail when the visual trainer path is updated again.
- `trainer_observation_reward_contract_v0_2026-05-09.md` and
  `observation_fidelity_plan.md`: historical learned-observation/ray planning
  docs; keep them from being read as the active 2P source-state visual trainer
  path.
