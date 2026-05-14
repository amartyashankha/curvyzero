# Browser Fidelity Reconstructor Handoff

Date: 2026-05-13

Scope: concise side-lane handoff from Optimizer to Environment Reconstruction.
This does not change the current training or optimizer parity oracle.

## Current Truth

The trusted trainer visual contract today is CPU-reference source-state
fidelity, not proven browser-pixel fidelity:

```text
source-state RGB 704x704
-> browser-sprite bonuses
-> live heads
-> BT.601 luma
-> exact 11x11 area average
-> uint8[1,64,64]
-> FIFO stack [4,64,64]
```

Optimizer owns performance and exact parity against that CPU reference. Dirty
cache, GPU renderers, or other optimized paths must byte-match the CPU
reference surface they replace. Browser pixel claims belong to Environment
Reconstruction.

The browser/reference source is in `third_party/curvytron-reference`. It is an
old Node/Gulp/Bower app, and the checkout does not currently include generated
full-app JS/server artifacts. Treat full-app browser replay as a later target,
not the first useful proof.

## What Reconstructor Should Own

- Define what "browser/source fidelity" means for real canvas pixels.
- Build and maintain the browser golden-frame harness and artifacts.
- Decide when a captured browser frame is strong enough evidence to update
  source-state renderer claims.
- Keep browser-canvas evidence separate from source-state gray64 evidence.

## What Optimizer Needs

- A small, reproducible browser golden-frame fixture or fixture set.
- Saved outputs: browser composite PNG, CPU-reference RGB PNG, gray64 if useful,
  diff PNG, and metrics JSON.
- Clear verdict fields: exact match, toleranced match, known browser/backend
  variance, or source-state renderer bug.
- A stable note on whether the CPU-reference visual contract should remain
  unchanged or needs a targeted renderer fix.

Optimizer does not need full multiplayer app boot, Socket.IO lifecycle, live
Modal runs, or training-volume artifacts for this side lane.

## Minimal Browser Golden Harness

Start with a tiny browser page, not the whole app:

1. Launch Chromium/Playwright with viewport `704x704` and
   `deviceScaleFactor=1`.
2. Create the four source canvas layers: `background`, `bonus`, `game`,
   `effect`.
3. Load `third_party/curvytron-reference/web/images/bonus.png` from the same
   local origin as the page.
4. Draw one deterministic 2P source-state frame using the browser drawing
   primitives and one or two typed map bonuses.
5. Compose `background + bonus + game` in `page.evaluate()` into an offscreen
   canvas and export PNG. Leave `effect` out of the first proof.
6. Render the same fixture through the current CPU-reference source-state
   renderer at RGB 704x704, then optionally through the gray64 downsample.
7. Save tiny local artifacts under a non-training path, for example
   `artifacts/local/browser_golden_probe/...`.

First fixture should be boring: background, two trails, two live heads, one
active bonus sprite. Add clears, gaps, wrap, death/effects, resize persistence,
and full-app event replay only after the boring fixture is trustworthy.

## Do Not Block On

- Old full-app Node/Gulp/Bower resurrection.
- WebSocket room lifecycle or spectator resync.
- Historical trail replay for arbitrary rich states.
- Exact browser antialiasing policy before a saved diff/metric exists.
- Death particles, idle arrows, HUD stack icons, or resize persistence.
- Any live training run, Modal volume, Coach batch, or optimizer profile.

## Exact Success Criteria

This side lane is successful when Reconstructor has:

1. a checked-in or documented command that captures one deterministic Chromium
   `background + bonus + game` composite at 704x704;
2. a matching CPU-reference render for the same fixture;
3. saved `browser.png`, `cpu_reference.png`, `diff.png`, and `metrics.json`
   outside training artifacts;
4. metrics that report at least image shape, max absolute RGB diff, mismatch
   pixel count, and whether gray64 after BT.601 plus 11x11 area average matches;
5. a written verdict that says one of:
   - CPU-reference renderer agrees with browser for this fixture;
   - CPU-reference renderer differs only by accepted browser/backend tolerance;
   - CPU-reference renderer has a concrete bug, with the smallest failing
     fixture named;
6. no change to the optimizer oracle unless a targeted renderer fix is proposed
   and separately validated against the CPU reference and training contract.

Related optimizer notes:

- `docs/working/optimizer/browser_golden_capture_side_lane_2026-05-13.md`
- `docs/working/optimizer/downsample_reference_fidelity_2026-05-13.md`
- `docs/working/optimizer/current_plate_map_2026-05-13.md`
- `docs/working/optimizer/system_architecture_map_2026-05-13.md`
