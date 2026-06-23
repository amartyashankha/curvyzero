# Subagent Critique: Lower-Fidelity Training Observations

Date: 2026-05-21

Status: recommendation note only. No implementation, trainer default, tournament
default, checkpoint contract, or live run was changed.

## Context

The trajectory-length ladder changes the renderer question. With dynamic trail
slots, B512/A16 no-probe rows measured about `3976` scalar steps/s at 20 profile
steps, `1780` at 100, `1149` at 200, and `598` at 500. The 500-step no-death
row spent about `822s` of `856s` in observation and about `802s` in renderer
render. For longer-lived policies, cumulative full redraw of browser-line
history becomes the wall again.

The training observation does not need browser/GIF fidelity. It does need to
preserve the model-facing game signal:

- controlled-player perspective: self/other semantics are stable even when
  physical seat changes;
- global board coordinates: no camera-centering, rotation, mirroring, or crop
  unless a new policy contract is explicitly created;
- trail occupancy, current heads, walls/bounds, and nearby future collision
  cues are visible enough to learn survival;
- bonus type/effect semantics remain learnable, not collapsed into an
  undifferentiated dot;
- terminal `final_observation`, autoreset, RND latest-frame extraction, and
  replay/model dtype handling remain explicit gates.

## Plain Recommendation

The best next research lane is a named low-fidelity policy surface based on a
persistent semantic occupancy buffer, preferably `128x128` internally with a
cheap `2x2` target-space reduction to `64x64`. It should update only newly
drawn trail/head/bonus changes per step, then emit the current `4x64x64`
policy stack or a clearly named successor. This attacks the long-trajectory
cost shape: render work becomes proportional to new motion and changed symbols,
not total trail history.

The safest probe ladder is:

1. Persistent `128x128` semantic buffer -> `64x64` observation.
2. Persistent `64x64` semantic buffer as the maximum-speed variant.
3. Direct full-redraw `128x128` and `64x64` rasters as comparison anchors.

Do not silently label these as `browser_lines+simple_symbols`. If adopted for
training, they need a new observation surface id and checkpoint metadata.

## Recommendation Matrix

| Candidate | Representation | Expected Speed | Fidelity Risk | Learnability Risk | Implementation Risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| Persistent `128x128` semantic occupancy, reduced to `64x64` | Maintain per env/player target-space masks for self trail, other trail, heads, walls, and bonus glyphs. Stamp only changed segments/symbols; reduce with `2x2` max or max-plus-average. | Very high on long rows. Render-only could plausibly drop by `5-10x` versus cumulative redraw; B512/500 end-to-end could move by several-x if stack/manager costs do not become the next wall. | Medium-low. `128x128` preserves more diagonal/near-miss structure than direct `64x64`, while still discarding source-canvas antialiasing. | Low if channels/intensities stay stable and bonus glyphs are distinct. | Medium. Needs careful reset, game-clear, borderless, gap, death, and final-observation invalidation. | Best first prototype. It balances speed and signal, and it gives a gentler fallback if direct `64x64` aliases too hard. |
| Persistent `64x64` semantic occupancy | Same as above, but stamp directly into the policy grid with explicit thickness and priority composition. | Highest. Could approach zero-observation-ish render cost for no-death long rows, because old trails are not redrawn. | Medium-high. Thin curves, grazing collisions, and close parallel trails can alias or merge. | Medium. Survival may still learn if trails are thick enough, but bad aliasing can make collision margins noisy. | Medium. Simpler memory footprint than `128x128`, but more tuning pressure on thickness/glyphs. | Run as maximum-speed ablation after `128x128`. Promote only if survival/eval does not regress. |
| Direct full-redraw `128x128` line raster -> `64x64` | Ignore browser/source resolution; redraw all current trail segments into `128x128`, stamp heads/bonuses, then reduce to `64x64`. | Medium-high. Avoids `704 -> 64` browser-like draw/downsample, but still grows with trail length. | Low-medium. Better geometric continuity than direct `64x64`; loses exact browser luma/coverage. | Low if bonus symbols are target-space and thick. | Low-medium. Easier to reason about than persistent mutation because each frame is rebuilt. | Useful comparison anchor and semantic oracle for persistent buffers. Not the long-term winner if 500-step trails stay dominant. |
| Direct full-redraw `64x64` line raster | Redraw current trail history directly into the final policy grid with fixed-width target-space lines and symbols. | Medium. Likely faster than browser redraw, but still `O(total trail history)` and may degrade with survival length. | High for geometry. Diagonals and tight gaps are fragile. | Medium-high unless lines/symbols are deliberately over-visible. | Low. Fastest to prototype and easiest to inspect. | Good throwaway ablation. Too risky as the main training surface unless results are surprisingly strong. |
| Dense browser/source redraw with cheaper downsample | Keep source-canvas line semantics but replace the expensive resize/area path with cheaper sampling or coarser source resolution. | Low-medium. May save `1.2-2x`, unlikely to solve an `802s` render wall. | Low if area-ish; medium if nearest/center sampling. | Low-medium. Familiar observation contract, but weaker antialiasing can erase small symbols. | Low-medium. Mostly renderer plumbing. | Not enough for long no-death rows. Use only as a conservative fallback or parity control. |
| Semantic multichannel masks | Emit separate channels for self trail, other trail, self head, other head, bonus classes/effects, walls, maybe recent trail age. Could be `C x 64 x 64` rather than grayscale stack. | Mixed. Raster can be fast, but payload/model input expands unless packed as `uint8` or indexed codes. | Low visual fidelity risk because it is intentionally semantic, not visual. | Lowest if the model and RND can consume the channels. Bonus semantics become explicit. | High. Changes model/replay/RND/checkpoint contracts more than a renderer swap. | Strong future design, weak drop-in. Consider only if accepting a new policy architecture/input contract. |
| Indexed single-channel semantic code map | One `uint8` grid stores class ids or priority-coded intensities; model path maps to float/embedding/one-hot. | High for render and payload if kept `uint8`; lower if expanded early. | Medium. Overlap priority can hide lower-priority objects. | Medium-low if codes are stable and decoded near the model. | High. Requires explicit cast/decode path across collector, learner, replay, and RND. | Interesting second-phase contract. Do not hide it behind the current float grayscale contract. |
| RGB/color policy observation | Use RGB-like channels for self/other/bonus classes instead of grayscale. | Low or negative versus grayscale because bytes/channels expand. | Low visual ambiguity, but not browser parity. | Low for class separation if model changes are allowed. | Medium-high. Changes input shape and policy metadata. | Not recommended for speed. Use semantic channels or indexed codes instead of literal RGB. |
| Egocentric crop/rotation/heading-normalized view | Center or rotate around the controlled player. | Potentially high due to smaller local field, but requires resampling and new semantics. | Very high relative to current contract. Loses global board convention and can hide long-range traps/bonuses. | Unknown. It may help local avoidance but changes the task the policy learned. | High. New policy contract, eval comparability, and tournament interpretation issues. | Do not use in this optimizer lane. Keep player perspective as self/other encoding on global coordinates. |

## Encoding Guidance

### Persistent Occupancy

Persistent buffers are the highest-leverage idea because they change the cost
model. Current long-row redraw repeatedly re-renders all old trails. An
occupancy buffer can keep old trails resident and stamp only:

- each player's new segment or head movement;
- trail gaps or non-printing intervals;
- bonus spawn, pickup, expiry, and active-effect overlays;
- full invalidations for reset, terminal final snapshot, game clear, and any
  rule that removes or globally changes trail visibility.

The risk is stale state. The buffer needs an explicit dirty/invalidation model,
not incidental mutation. A safe profiler should compare persistent output
against a full-redraw low-res oracle every N steps and on reset/terminal rows.

### Direct `64x64` Versus `128x128`

Direct `64x64` is tempting because the policy already consumes `64x64`, but it
is the aliasing danger zone. CurvyTron survival depends on thin curved trails,
gaps, tight near-misses, and the current heading/head location. A direct
`64x64` line needs target-space thickness tuned for learnability, not visual
beauty.

`128x128` internal occupancy is a better first compromise:

- draw lines/glyphs with integer target-space coverage;
- reduce to `64x64` with max-pooling for occupancy-critical layers;
- optionally use average only for soft intensity, not for object existence;
- keep heads and bonuses at a minimum visible footprint after reduction.

This gives a cheap two-sample supersampling effect without reopening the dense
source-canvas browser renderer.

### Downsampling

For policy observations, area-averaging from `704` is not sacred. The useful
property is that small but important objects do not disappear. Prefer:

- max or OR reduction for trail/head/bonus existence;
- capped intensity or small blur only if jagged lines destabilize learning;
- deterministic priority composition after reduction, not racing scatter writes;
- no center-sampling for thin symbols unless a canary proves they survive.

If the lower-fidelity renderer uses grayscale, choose values for separability,
not source luma mimicry.

### Grayscale, Color, And Channels

The cheapest drop-in stays close to the current `float32 [4,64,64]` stack:
self trail/head, other trail/head, bonuses, and walls are encoded by stable
grayscale values in one frame, then stacked over time. This preserves model
shape but forces all semantics through intensity and priority.

A more learnable representation is semantic channels:

```text
self_trail, other_trail, self_head, other_head,
neutral_bonus_type_or_group, wall/border, optional active_effect_self,
optional active_effect_other
```

That is probably a better RL input, but it is no longer the same policy
contract. It changes model input shape, checkpoint compatibility, replay, RND,
and possibly MuZero representation dynamics. Treat it as a new surface, not a
renderer optimization.

Literal RGB is not the right middle ground. It costs more bandwidth than
grayscale and is less explicit than semantic channels.

### Bonus Symbols

A lower-fidelity observation must not turn bonuses into identical dots. Bonuses
need type/effect information at the policy scale. The robust target-space rule
is:

- use fixed-size glyphs at `64x64` visibility, at least `3x3` after reduction
  and preferably `5x5` for rare/high-impact effects;
- use asymmetric glyphs where orientation or mirroring cannot make two classes
  look identical;
- encode raw bonus type/effect group consistently, not browser sprite color;
- keep map bonus objects seat-invariant, matching the current contract that
  neutral bonus rendering is seat-invariant;
- if active effects are rendered on players, encode them in controlled-player
  terms: active effect on self versus active effect on other.

Suggested effect grouping for a low-fidelity probe:

| Bonus group | Learnable target-space cue |
| --- | --- |
| Self-affecting avatar modifiers | Distinct neutral glyph class; active effect overlay on whichever player caught it. |
| Enemy-affecting avatar modifiers | Distinct neutral glyph class; active effect overlay on the affected player after pickup. |
| All-player modifiers | Separate glyph; optional dual active-effect overlay. |
| Game/global modifiers such as borderless or clear | Separate high-salience glyph because the strategic implication is global. |

If there are too many raw types for grayscale intensities, group by effect first
and keep raw type in a future semantic-channel/indexed representation.

### Player Perspective

Do not lower fidelity by collapsing player identity into physical colors. The
policy still needs controlled-player view:

```text
observation[b, 0] = player 0 as self, player 1 as other
observation[b, 1] = player 1 as self, player 0 as other
```

The spatial board should stay global. The lower-fidelity renderer can change
line thickness, symbol shape, resolution, and channel encoding, but not the
self/other convention. Both player views must be tested from the same asymmetric
state so a swapped or fixed-player-zero bug is visible.

## Concrete Gates Before Promotion

- Profile gates: B512/A16 20/100/200/500-step no-death rows, reporting
  renderer render, observation total, stack update, H2D/D2H, payload bytes, and
  wall. Compare against current dynamic browser-line redraw and zero-observation.
- Visual gates: save small frame grids for asymmetric two-player states,
  diagonal trails, close parallel trails, trail gaps, heads overlapping bonuses,
  border/wall proximity, and every bonus group.
- Semantic gates: both controlled-player views, random physical learner seat,
  opponent view, terminal `final_observation`, autoreset, game clear, borderless
  state, bonus spawn/pickup/expiry, and RND latest-frame extraction.
- Learning gates: run paired short training/eval canaries with the same seeds
  and label checkpoints by observation surface. Do not compare checkpoint Elo
  across old and new surfaces without explicit metadata.

## Final Ranking

1. **Persistent `128x128` semantic occupancy -> `64x64`:** best balance.
2. **Persistent `64x64` semantic occupancy:** highest speed, higher alias risk.
3. **Direct full-redraw `128x128` -> `64x64`:** best oracle/control.
4. **Direct full-redraw `64x64`:** quick ablation, risky mainline.
5. **Semantic multichannel or indexed-code observations:** promising but a new
   policy contract, not a renderer swap.
6. **Cheaper dense/browser downsample or RGB:** unlikely to solve the current
   long-trajectory renderer wall.

