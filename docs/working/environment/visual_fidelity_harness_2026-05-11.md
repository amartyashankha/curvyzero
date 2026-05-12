# Visual Fidelity Harness - 2026-05-11

Scope: 2-player CurvyTron source-state visual checks for
`VectorMultiplayerEnv`. This is a working-memory note, not a replacement for
the full 2P gap catalog.

## Plain Answer

Status: active source-state/native visual consistency gate. The active product
path is source-state RGB renderer -> gray64 -> frame stack. `browser_lines` is
the default browser-style source-state renderer, and `body_circles_fast` is an
explicit circle-per-body approximation for speed and legacy profiling. This is
still not a browser pixel parity claim.

`scripts/compare_2p_raw_visual_observation.py` checks whether the source-shaped
2P state and the fast vector runtime produce the same source-state/native
product image frame. The active product path is one clean image path:
source-state browser-style RGB64 raw frame -> deterministic gray64
`uint8[1,64,64]` -> normalized frame stack. The harness directly compares
gray64; trainer wrapper/replay propagation must be proven separately. This is
not a real browser/DOM canvas pixel-parity claim.

For each covered scenario, the harness renders:

- source truth: `CurvyTronSourceEnv` snapshot plus source world/bonus bodies
- vector candidate: `VectorMultiplayerEnv` state

It compares the two frames at reset/setup and after each scripted step until
the scenario ends or `--max-steps` stops it. A clean pass means exact pixel
equality: `max_abs_diff=0` and `mismatch_pixels=0`.

Current working-doc state: `core2p` is an active source-vs-vector native render
gate. It was reported exact across 35 scenarios under the shared source-state
renderer. Keep that result as source-state raster regression evidence only, not
browser canvas pixel evidence. The harness also has two intentional mismatch
canaries that remove a visible world body or a visible map bonus and must fail.
The PrintManager random-call-order fixture stays outside gray64 because it
proves RNG/event order, not a different rendered frame.

There is also a separate bonus64 v1 typed/status gate. It does not replace
gray64. It checks active map bonus mask/type planes for all 12 source-default
bonus types and post-catch self/other/game status planes against source-derived
values. Treat bonus64/rich tensors as diagnostic/proof surfaces for hidden
bonus facts, not as the product trainer default and not as a parallel rich
observation path.

Use `--suite full2p` as the current combined source-state/native visual
consistency command. The recorded 2026-05-11 result was:

```bash
uv run python scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain
# PASS full_2p_source_state_visual_gate canvas_gray64=35/35 typed_bonus=12/12 final_obs=pass canaries=2/2 mismatch_pixels=0 max_abs_diff=0.0 expected_canary_mismatch_pixels=78
```

Read that line carefully: it means only that source and vector matched through
the source-state/native render path and that the separate diagnostic bonus64
gate passed. It does not mean real browser canvas parity, wrapper/replay
propagation, or a full trainer-ready environment has been proven. It also does
not mean the trainer consumes bonus64. The intended product path stacks gray64,
and trainer/replay propagation remains open until a direct proof names it.

Raw pictures or debug renders may look grayscale by design when they are showing
the gray64 tensor. That says nothing about the browser canvas palette. Keep the
candidate model-observation tensor surface separate from the browser/canvas
render surface.

## Why It Is Useful

- It catches mistakes in the candidate model-observation raster, not just simulator
  state.
- It exposes missing vector geometry: heads, trails, bodies, wall deaths,
  bonus bodies, cleared bodies, terminal frames, and reset/setup placement.
- It gives a cheap failure pointer: first mismatching tick, `(y, x)`, source
  value, vector value, nonzero pixel counts, and optional PGM diff images.
- It has intentional failure canaries, so we know the alarm fires when a visible
  source fact is missing.
- It keeps the candidate LightZero visual path honest: the tensor can be source-state
  faithful before anyone claims learning quality.
- It separates source-state observation fidelity from browser/render fidelity,
  which should be a later human/debug check.

## What It Cannot See

- Browser/canvas pixels, antialiasing, palette, viewport scaling, sprite style,
  browser timing, or network render payloads.
- Real browser trail-shape parity. `browser_lines` is the default browser-style
  source-state renderer, but it is still a native/source-state implementation.
  `body_circles_fast` must stay labeled approximate and should not be cited as
  browser-style fidelity evidence.
- Hidden state that a grayscale screenshot-like observation does not encode.
  Bonus type and active stack timers are not separate channels in gray64.
- Bonus stack/status facts. The bonus64 v1 companion gate covers the first
  typed/status slice, but it still does not encode `BonusAllColor` post-catch
  color rotation or a typed post-catch `BonusGameClear` status plane. It is a
  diagnostic/proof gate, not the trainer observation.
- PrintManager random-call order or event order when the final visible raster is
  unchanged.
- Replay completeness, reward correctness, two-seat self-play policy mapping,
  or final observation propagation through trainer/replay wrappers.
- 3P/4P behavior. This harness is intentionally 2P-only.
- Bugs shared by both source-state gray64 renderers, or state differences that
  quantize to the same 64x64 pixels.

Also keep the size wording straight: the source 2P arena is 88 source units.
`64x64` is only the learned observation raster size.

## How To Use It To Catch Holes

First prove the source rule with a JS/source fixture or `CurvyTronSourceEnv`
claim. Then use this harness to ask whether that trusted state is visible in
the model tensor.

Run the current gate:

```bash
uv run python scripts/compare_2p_raw_visual_observation.py --suite core2p --format plain
```

Run the fuller current 2P visual gate:

```bash
uv run python scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain
```

Run one scenario with detailed JSON:

```bash
uv run python scripts/compare_2p_raw_visual_observation.py \
  --scenario scenarios/environment/source_normal_wall_death_step.json \
  --format json
```

When debugging one visible mismatch, write PGM artifacts for a single scenario:

```bash
uv run python scripts/compare_2p_raw_visual_observation.py \
  --scenario scenarios/environment/source_bonus_game_clear_immediate_step.json \
  --out-dir /private/tmp/curvy-visual-hole
```

Use `--max-steps` to bisect long traces. Read `first_mismatch`, then compare
the previous frame, failing frame, and `final_absdiff.pgm`. If the source and
vector states differ, fix the runtime/state path. If the states agree but the
raster differs, fix `vector_visual_observation`. If the important fact is not
representable in gray64, do not hide it in this harness; add or extend a
diagnostic typed/status gate without making it the trainer default.

Natural bonus spawn/retry/cap fixtures use a special harness path because their
RNG starts at `BonusManager` setup, not the ordinary forced-state seeding path.

## Next Best 2P Scenarios

Add source-state gray64 cases only when the visual geometry should actually change.
Expand bonus64 v1 only when source-fidelity proof needs hidden bonus facts that
are still not represented.

Best next additions:

- Build a real browser/canvas pixel harness with golden browser frames. The
  existing `full2p` gate is source-state/native visual consistency only.
- JS/original fixture parity for the programmatic bonus stress cases:
  SelfMaster body-hit protection, borderless expiry before wall death, and
  GameClear before body collision.
- Final observation before autoreset for visual terminal frames.
- `BonusAllColor` color-rotation diagnostic plane if that hidden fact needs
  source/vector proof.
- Natural emitted own-body loop collision and broader natural trail-gap cases,
  especially around terminal frames.
- A non-gray64 companion gate for PrintManager RNG/event-order canaries, so
  they stay protected without pretending they are visible pixels.

Do not move to browser pixels or 3P/4P visual claims until the 2P source-state
and bonus-status story is boring.
