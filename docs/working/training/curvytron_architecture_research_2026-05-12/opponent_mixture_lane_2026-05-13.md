# Opponent Mixture Lane

Status: first-class local/trainer wiring exists for static weighted opponent
mixtures in the trusted stock LightZero path. Keep this separate from the live
300-row rescue batch.

## Supported Now

Path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
env_variant=source_state_fixed_opponent
```

Selection point: `CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.reset()`.
The selected entry is stored on the env instance and used by `_opponent_action()`
for every step until the next reset. There is no per-step opponent switching.

Mixture spec: explicit JSON object or list with weighted entries. Each entry
must name `opponent_policy_kind` and `weight`; invalid or ambiguous entries
raise instead of falling back.

Supported entry families:

- `fixed_straight` plus `opponent_runtime_mode=blank_canvas_noop`;
- `fixed_straight` plus `opponent_death_mode=immortal` as the passive dirty
  control;
- `proactive_wall_avoidant`;
- `frozen_lightzero_checkpoint` with immutable checkpoint refs. Use
  `age_label=recent`, `somewhat_recent`, or `old` only as labels on explicit
  immutable refs, not as dynamic selectors.

Example shape:

```json
{
  "seed": 17,
  "entries": [
    {"name": "blank", "weight": 4, "opponent_policy_kind": "fixed_straight", "opponent_runtime_mode": "blank_canvas_noop"},
    {"name": "passive", "weight": 1, "opponent_policy_kind": "fixed_straight", "opponent_death_mode": "immortal"},
    {"name": "wall", "weight": 2, "opponent_policy_kind": "proactive_wall_avoidant"},
    {"name": "recent_001", "age_label": "recent", "weight": 1, "opponent_policy_kind": "frozen_lightzero_checkpoint", "opponent_checkpoint_ref": "training/lightzero-curvytron-visual-survival/<run>/checkpoints/lightzero/iteration_123.pth.tar"}
  ]
}
```

## Gaps

- No live rolling same-run self-play refresh.
- No directory scan or `latest.pth.tar` checkpoint selection.
- Background checkpoint eval/GIF now carries mixture config and records the
  selected episode component in eval rows and GIF summaries; it still needs a
  corrected tiny remote canary before scale.
- `scripts/build_curvytron_opponent_mixture_manifest.py` builds the canary and
  batch manifests. The first 100-row draft is stale; the next draft should use
  `save_ckpt_after_iter=10000` and the 228-row compact base grid.

## Proposed Second Batch Shape

After one corrected tiny canary passes with background eval/GIF enabled, run a
second batch rather than changing the 300 rescue wave:

- core baseline grid: `body_circles_fast` and `browser_lines`, sim8, C32, B32,
  with repeat levels `rep0`, `repM`, and `repH`;
- sentinel baseline grid: sim16, C64, and B64 probes, each paired across fast
  and browser render;
- main mixtures: eight readable recipes, each with recent frozen checkpoint at
  50%;
- controls: recent-only, mid-only, old-only, blank-only, scripted-only, and
  passive-only;
- working scale: 228 rows before any pruning;
- keep reward at `survival_plus_bonus_no_outcome` and episode cap at `65536`.

Do not call this true live self-play. Label it as stock ego training against a
static weighted episode-opponent mixture.

## Launch Artifacts

Current local artifacts:

- canary manifest:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix-recent-canary-20260513a.json`
- stale planned 100-row batch manifest:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix-recent-20260513a.json`

Next manifest names should use the clearer `curvy-mix2-*` prefix.

The provisional frozen refs point to preserved v1b files:

- recent: `iteration_20000.pth.tar`
- mid: `iteration_10000.pth.tar`
- old: `iteration_0.pth.tar`

The earlier dense-run checkpoint refs are not valid now because that run was
cleaned from the Modal volume.
