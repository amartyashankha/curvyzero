# Visual Fidelity Harness - 2026-05-11

Scope: 2-player CurvyTron source-state visual checks for
`VectorMultiplayerEnv`. This is a working-memory note, not a replacement for
the full 2P gap catalog.

## Plain Answer

`scripts/compare_2p_raw_visual_observation.py` checks whether the source-shaped
2P state and the fast vector runtime produce the same trainer visual frame.
The product path is one clean image path: source-state browser-like RGB64 raw
frame -> deterministic grayscale `uint8[1,64,64]` -> normalized stack
`float32[4,64,64]`. This is still not a real browser/DOM canvas pixel-parity
claim.

For each covered scenario, the harness renders:

- source truth: `CurvyTronSourceEnv` snapshot plus source world/bonus bodies
- vector candidate: `VectorMultiplayerEnv` state

It compares the two frames at reset/setup and after each scripted step until
the scenario ends or `--max-steps` stops it. A clean pass means exact pixel
equality: `max_abs_diff=0` and `mismatch_pixels=0`.

Current working-doc state: `core2p` is the active 2P canvas-gray64 gate and is
reported as exact across 34 scenarios. That includes the long no-bonus wall
terminal, movement, normal-wall/draw, collision-order, borderless,
`BonusSelfSmall`, `BonusGameClear`, `BonusGameBorderless`, four natural bonus
spawn/retry/cap fixtures, and eight programmatic source-snapshot stress cases,
including printing trail emission, opponent/own body collision edges,
PrintManager trail-gap boundary emission, and explicit 2P survivor
warmdown-frame movement/death.
The harness also has two intentional mismatch canaries that remove a visible
world body or a visible map bonus and must fail. The PrintManager
random-call-order fixture stays outside gray64 because it proves RNG/event
order, not a different rendered frame.

There is also a separate bonus64 v1 typed/status gate. It does not replace
gray64. It checks active map bonus mask/type planes for all 12 source-default
bonus types and post-catch self/other/game status planes against source-derived
values. Treat bonus64/rich tensors as diagnostic/proof surfaces for hidden
bonus facts, not as the product trainer default and not as a parallel rich
observation path.

Use `--suite full2p` when you want the current one-line visual answer. On
2026-05-11 it passed:

```bash
uv run python scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain
# PASS full_2p_source_state_visual_gate canvas_gray64=34/34 typed_bonus=12/12 canaries=2/2 mismatch_pixels=0 max_abs_diff=0.0 expected_canary_mismatch_pixels=81
```

Read that line carefully: it means the current 2P canvas-gray64 visual gate and
the separate diagnostic bonus64 gate pass. It does not mean real browser canvas
parity, wrapper/replay propagation, or a full trainer-ready environment has
been proven. It also does not mean the trainer consumes bonus64; the
source-state training wrapper consumes stacked canvas-gray64.

Raw pictures or debug renders may look grayscale by design when they are showing
the gray64 tensor. That says nothing about the browser canvas palette. Keep the
model-facing tensor surface separate from the browser/canvas render surface.

## Why It Is Useful

- It catches mistakes in the actual model-facing raster, not just simulator
  state.
- It exposes missing vector geometry: heads, trails, bodies, wall deaths,
  bonus bodies, cleared bodies, terminal frames, and reset/setup placement.
- It gives a cheap failure pointer: first mismatching tick, `(y, x)`, source
  value, vector value, nonzero pixel counts, and optional PGM diff images.
- It has intentional failure canaries, so we know the alarm fires when a visible
  source fact is missing.
- It keeps the LightZero visual path honest: the tensor can be source-state
  faithful before anyone claims learning quality.
- It separates source-state observation fidelity from browser/render fidelity,
  which should be a later human/debug check.

## What It Cannot See

- Browser/canvas pixels, antialiasing, palette, viewport scaling, sprite style,
  browser timing, or network render payloads.
- Hidden state that a grayscale screenshot-like observation does not encode.
  Bonus type and active stack timers are not separate channels in the trainer
  image.
- Bonus stack/status facts. The bonus64 v1 companion gate covers the first
  typed/status slice, but it still does not encode `BonusAllColor` post-catch
  color rotation or a typed post-catch `BonusGameClear` status plane. It is a
  diagnostic/proof gate, not the trainer observation.
- PrintManager random-call order or event order when the final visible raster is
  unchanged.
- Replay completeness, reward correctness, two-seat self-play policy mapping,
  or final observation propagation through trainer/replay wrappers.
- 3P/4P behavior. This harness is intentionally 2P-only.
- Bugs shared by both canvas-gray64 renderers, or state differences that
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

Add canvas-gray64 cases only when the visual geometry should actually change.
Expand bonus64 v1 only when source-fidelity proof needs hidden bonus facts that
are still not represented.

Best next additions:

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
