# Visual Fidelity Harness - 2026-05-11

Scope: 2-player CurvyTron source-state visual checks for
`VectorMultiplayerEnv`. This is a working-memory note, not a replacement for
the full 2P gap catalog.

## Plain Answer

`scripts/compare_2p_raw_visual_observation.py` checks whether the source-shaped
2P state and the fast vector runtime produce the same learned raw visual frame.
The frame is `uint8[1,64,64]` gray64. It is a CurvyZero training observation
raster from source state, not the original browser canvas.

For each covered scenario, the harness renders:

- source truth: `CurvyTronSourceEnv` snapshot plus source world/bonus bodies
- vector candidate: `VectorMultiplayerEnv` state

It compares the two frames at reset/setup and after each scripted step until
the scenario ends or `--max-steps` stops it. A clean pass means exact pixel
equality: `max_abs_diff=0` and `mismatch_pixels=0`.

Current working-doc state: `core2p` is the active 2P gray64 gate and is reported
as exact across 31 scenarios. That includes the long no-bonus wall terminal,
movement, normal-wall/draw, collision-order, borderless, `BonusSelfSmall`,
`BonusGameClear`, `BonusGameBorderless`, four natural bonus spawn/retry/cap
fixtures, and five programmatic source-snapshot stress cases, including
printing trail emission and explicit 2P survivor warmdown-frame movement/death.
The harness also has two intentional mismatch canaries that remove a visible
world body or a visible map bonus and must fail. The PrintManager
random-call-order fixture stays outside gray64 because it proves RNG/event
order, not a different rendered frame.

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
- Hidden state that gray64 does not encode. All active map bonuses collapse to
  one value, `208`, so gray64 cannot tell bonus type.
- Bonus stack/status facts: speed, radius, inverse controls, straight-angle,
  invincibility, borderless active/expired state, color stacks, expiry timers,
  and current effective properties.
- PrintManager random-call order or event order when the final visible raster is
  unchanged.
- Replay completeness, reward correctness, two-seat self-play policy mapping,
  or final observation propagation through trainer/replay wrappers.
- 3P/4P behavior. This harness is intentionally 2P-only.
- Bugs shared by both gray64 renderers, or state differences that quantize to
  the same 64x64 pixels.

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
representable in gray64, do not hide it in this harness; add a typed/status
observation schema and give it its own gate.

Natural bonus spawn/retry/cap fixtures use a special harness path because their
RNG starts at `BonusManager` setup, not the ordinary forced-state seeding path.

## Next Best 2P Scenarios

Add gray64 v0 cases only when the visual geometry should actually change.
Add a new typed/status visual schema when the policy needs hidden bonus facts.

Best next additions:

- Natural `BonusGameClear` catch that clears bodies, then probes the cleared
  collision site and the terminal/final frame.
- Natural `BonusGameBorderless` catch, wrap, expiry, then normal-wall death
  after expiry.
- Natural or seeded `BonusSelfMaster` catch, blocked body/trail death, then
  wall death, with the final observation checked before autoreset.
- `BonusAllColor` overlap and expiry/restore in a typed/status schema; gray64
  cannot see the color-stack rule.
- Speed/radius/control bonus effects that change later movement or collision:
  self/enemy slow/fast, enemy big, inverse, and straight-angle.
- Natural emitted own-body loop collision and broader natural trail-gap cases,
  especially around terminal frames.
- A non-gray64 companion gate for PrintManager RNG/event-order canaries, so
  they stay protected without pretending they are visible pixels.

Do not move to browser pixels or 3P/4P visual claims until the 2P source-state
and bonus-status story is boring.
