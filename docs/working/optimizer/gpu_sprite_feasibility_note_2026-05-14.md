# GPU Sprite Feasibility Note

Date: 2026-05-14
Updated: 2026-05-15

Scope: side-lane critique for original browser sprites on GPU.

## Bottom Line

Original `browser_sprites` on GPU are not the current policy-observation target.
The current production target is CPU `cpu_oracle`
`browser_lines + simple_symbols`.

CPU `browser_lines + simple_symbols` is the production backend and parity oracle
today. A future batched GPU backend must match it; `browser_sprites` stay in the
artifact/reference/browser fidelity lane.

Do not block the policy renderer on exact original-sprite GPU parity.

## Why Original Sprites Are A Separate Project

Putting `bonus.png` on the GPU is not enough. Exact original-sprite parity would
need to match the CPU reference composition:

- 3x4 RGBA atlas from
  `third_party/curvytron-reference/web/images/bonus.png`;
- type-to-sprite mapping and cached stamp sizing;
- world placement, `_canvas_round`, clipping, nearest sampling, and tile choice;
- draw order: trails, active bonus sprites, live heads;
- source-over RGB alpha blending, rounding, clipping, BT.601 luma, and exact
  11x11 area downsample;
- deterministic behavior for overlapping translucent sprites;
- byte comparison against CPU `browser_lines + browser_sprites` on real rows.

That work is moderate-to-high complexity and belongs to browser-fidelity or
artifact needs, not the current policy-observation target.

## Historical Evidence

Earlier GPU notes showed no-bonus line/head rows could be exact or near exact.
Bonus rows were the visible mismatch: about `1.2%` to `1.6%` of checked gray64
pixels differed in synthetic bonus comparisons, with large max diffs around
missing or differently shaped sprite pixels.

Those rows explain why original sprites are hard. They do not change the current
target to `browser_sprites`.

## Current Policy Implication

Use simple symbols for policy observations:

- simpler device representation;
- no RGBA atlas sampling on the hot path;
- no translucent sprite composition on the hot path;
- easier CPU/GPU equality testing;
- stable machine-visible class identity after downsample.

The semantic honesty rule still applies: call the policy surface
`browser_lines + simple_symbols`, not browser-sprite parity.

## If Original Sprites Are Reopened

The next credible step would be a small isolated parity prototype:

1. one or a few real source-state rows;
2. atlas on device;
3. exact sprite-over-trail and head-over-sprite cases;
4. byte comparison to CPU gray64;
5. separate timing for transfer, render, and readback.

Until that passes, GPU `browser_sprites` should stay advertised as a known gap
in the artifact/reference lane.
