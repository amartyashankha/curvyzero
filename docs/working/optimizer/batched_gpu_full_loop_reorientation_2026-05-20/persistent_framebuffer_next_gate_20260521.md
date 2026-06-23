# Persistent Policy-Space Framebuffer Next Gate

Date: 2026-05-21

Status: optimizer plan. Do not treat this as a live training recommendation yet.

## Plain Goal

Stop paying to redraw every old trail point on every frame.

The current long no-death GPU profile gets slow because each observation frame
redraws the full trail history. The synthetic benchmark shows the better cost
model: keep a persistent policy-space image, stamp only new trail segments, and
compose heads/bonuses on top.

## Evidence So Far

Dynamic GPU full-redraw ladder, B512/A16:

| measured steps | scalar steps/s | observation wall | renderer wall |
| ---: | ---: | ---: | ---: |
| 20 | `3975.90` | `4.141s` | `3.499s` |
| 100 | `1779.81` | `51.815s` | `48.085s` |
| 200 | `1149.00` | `168.225s` | `161.786s` |
| 500 | `598.01` | `821.539s` | `802.428s` |

Current local CPU dirty-cache path:

| steps | env steps/s | render % | notes |
| ---: | ---: | ---: | --- |
| 100 | `430.5` | `76.9%` | no fallbacks after cold start |
| 500 | `433.1` | `76.8%` | no fallbacks after cold start |
| 1000 | `425.5` | `76.6%` | no fallbacks after cold start |

Synthetic persistent framebuffer benchmark:

| row | total speedup | device speedup | parity |
| --- | ---: | ---: | --- |
| H100 B128/S64/64x64 | `3.67x` | `7.86x` | exact |
| H100 B512/S512/64x64 | `8.48x` | `40.26x` | exact |
| H100 B2048/S256/64x64 | `5.06x` | `48.13x` | exact |
| L4 B512/S512/64x64 | `10.86x` | `53.23x` | exact |
| H100 B512/S512/64x64 no readback | `38.57x` | `38.80x` | exact |
| H100 B512/S256/128x128 | `4.60x` | `40.21x` | exact |

Read: update-only rendering is the first plausible 10x-class renderer lane.
Readback and stack ownership still matter.

## What This Is Not

- Not browser pixel parity.
- Not a trainer default.
- Not a tournament default.
- Not checkpoint-compatible metadata until promoted.
- Not a claim that a semantic/lower-fidelity surface learns as well.

## Real Profile-Only Renderer Requirements

Add a new explicit renderer label behind `SourceStateBatchedObservationRenderer`,
for example:

```text
jax_gpu_persistent_policy_framebuffer_profile
```

It should consume current source-state arrays:

- `visual_trail_active`
- `visual_trail_write_cursor`
- `visual_trail_pos`
- `visual_trail_radius`
- `visual_trail_owner`
- `visual_trail_break_before`
- `pos`, `radius`, `present`, `alive`
- `bonus_active`, `bonus_pos`, `bonus_radius`, `bonus_type`, `bonus_id`
- `avatar_color`

It should maintain per-row cache state:

- persistent trail layer, at least `[B, 2, H, W]`;
- previous cursor;
- previous map size;
- previous visual-trail prefix signature or safe rebuild rule;
- previous head snapshot;
- previous bonus snapshot;
- previous reset generation if available.

## Cost Model

For normal append-only steps:

1. Detect rows whose visual trail cursor advanced.
2. Gather only new trail slots.
3. Stamp new segments into persistent trail layers.
4. Compose transient heads and bonuses.
5. Read back or hand off the current `[B,2,1,H,W]` frames.

For invalidation:

- reset rows: clear those row layers and rebuild from current post-reset state;
- cursor regression: clear and rebuild;
- prefix mutation: clear and rebuild;
- map-size or palette change: clear and rebuild;
- game clear: clear affected trail layers;
- unsupported state: fail closed or fall back with explicit telemetry, never
  silently hide it.

## Composition Rules

Keep current policy semantics unless a new surface is explicitly created:

- view 0 means player 0 as self, player 1 as other;
- view 1 means player 1 as self, player 0 as other;
- spatial board stays global, not egocentric;
- draw order remains background, trails, bonuses, heads;
- bonus symbols must stay distinguishable at policy resolution;
- heads draw last.

If the renderer changes line geometry, resolution, grayscale values, or bonus
glyphs enough to stop being equivalent, label it as a new observation surface.

## Minimal Profile Gate

Before using this in any Coach run:

1. Run local/unit parity against a stateless reference on asymmetric rows:
   diagonal trails, overlapping owners, radius changes, break-before gaps,
   bonus overlap, head overlap, reset, and game clear.
2. Run Modal profile-only rows:
   B512/A16 no-death 20/100/200/500 measured steps.
3. Report:
   renderer update time, readback, stack update, dirty rows, rebuild rows,
   fallbacks, active trail count, and wall time.
4. Compare against the current dynamic full-redraw GPU ladder and the local CPU
   dirty-cache shape.

Promotion rule: only recommend this to Coach if it is faster end-to-end and
has explicit metadata that the tournament/checkpoint path can honor.

## Current Recommendation

Implement the real profile-only persistent renderer next. Do not spend another
large pass optimizing the full-redraw JAX block renderer unless this gate fails.

