# GPU Renderer Adversarial Gates

Date: 2026-05-15

Purpose: red-team the isolated JAX/H100 observation renderer before any trainer
integration. CPU `cpu_oracle` remains production. The GPU renderer is a lab lane
until it passes these gates against the production `browser_lines +
simple_symbols` observation contract.

## Highest-Risk Untested Holes

1. Avatar color is still a trap. The CPU oracle palette can respect
   `state["avatar_color"]`, but the compact GPU state currently carries trail
   owners and head indices, not avatar-color indices. A swapped-color or
   `BonusAllColor` row can pass identity-color smoke tests and still render the
   wrong controlled-player self/other view.

2. Two checked real-env rows do not exercise enough geometry. The current smoke
   is strongest on ordinary appended trails. It does not prove crossings where
   the winning RGB owner has lower luma, coincident heads, terminal heads over
   bonuses, radius changes inside one owner path, or slots that are active but
   stale because the cursor wrapped/regressed/reset.

3. The lab renderer is only an observation renderer. It does not prove trainer
   step semantics: reset frame stack, final observation, done/truncated,
   per-seat metadata, tournament seat mapping, replay row mapping, or whether
   the policy consumes the same player perspective after roster/seat swaps.

4. Long-horizon states are qualitatively different. A `real_env512` smoke may
   miss cursor wrap, visual-trail overflow, print/hole cadence after many bonus
   effects, stale high slots, and cache invalidation patterns that only appear
   after thousands of source ticks.

5. Composition order must be attacked, not sampled. Production order is grouped
   browser-line owners, then bonuses, then heads, with RGB overwrite before
   grayscale downsample. Exact parity on typical rows can hide wrong answers
   when the top object has the same or nearby luma.

## P0 Gates Before Trainer Integration

- **Avatar-color matrix:** exact CPU-oracle parity for identity colors, swapped
  colors, duplicated color indices, high color indices, and `BonusAllColor`
  active/expired rows. Include both controlled players and verify that
  self/other luma follows avatar color, not owner index.

- **Adversarial geometry batch:** exact parity for owner sequences
  `[0, 1, 0]`, `[1, 0, 1]`, `[0, 1, 1, 0]`, inactive holes between same-owner
  points, and overlapping crossings where player 0 wins over player 1 despite
  lower luma. Repeat with `break_before` on every possible slot.

- **Radius discontinuity batch:** exact parity when one owner has radii
  `4, 4, 8, 8, 4`, including caps at the discontinuity, same-position repeated
  points, near-zero radius, and head radius changed by self/enemy bonus effects.

- **Cursor/stale-slot batch:** exact parity when active bits exist past
  `visual_trail_write_cursor`, cursor is `0`, cursor is `capacity`, stale slots
  contain bright crossed geometry, and prefix slots are inactive but later slots
  remain active. GPU must ignore exactly what CPU ignores.

- **Bonus overlap batch:** exact parity for all 12 simple-symbol types centered
  on live head, dead head, trail crossing, trail cap, wall edge, clipped corner,
  and another bonus. Verify bonus drawn after trails and before heads.

- **Multi-player perspective batch:** exact parity for 3P/4P rows, absent
  players, dead players, invalid owners, and controlled-player views for every
  seat. This is required even if the first trainer target is 2P, because
  tournament metadata and shared code already carry 3P/4P assumptions.

- **Reset/final-observation batch:** exact parity for terminal step obs,
  `final_observation`, autoreset first obs, stack FIFO contents, and per-row
  reset in a mixed batch. A renderer-only pass is not enough; the batch object
  handed to LightZero must match CPU.

- **Metadata drift audit:** assert trainer/tournament artifacts name the same
  `policy_observation_backend`, `trail_render_mode`, `bonus_render_mode`,
  `controlled_player`, seat mapping, stack depth/order, `decision_source_frames`,
  and source physics step. Reject GPU rows that cannot prove these fields.

- **Long-horizon corpus:** collect real CPU states at source ticks
  `0, 1, 2, 8, 64, 512, 2048, 8192`, plus terminal and post-reset rows. The GPU
  gate should sample from this corpus, not only synthetic or early rollout rows.

- **Pipeline timing gate:** benchmark render + stack update + normalization +
  policy-forward stub + replay handoff. Report compile excluded, warmup
  excluded, host readback counted, and GPU memory/HLO size. Kernel-only speed is
  not an integration proof.

## Smallest Kill/Strengthen Experiment

Build a fixed `B=32` adversarial production-state corpus with 16 hand-authored
rows and 16 long-horizon CPU rollout rows. Render both controlled-player views
through the CPU oracle and the JAX H100 renderer.

The hand-authored rows should include:

- swapped `avatar_color` and one `BonusAllColor` row
- same-owner interleaving with inactive holes and `break_before` toggles
- radius changes inside one owner path
- active stale slots beyond cursor containing bright overlapping geometry
- all 12 bonus symbols split across rows with head/trail/wall overlap
- one terminal row and one immediately reset row with stack expectations
- one 4P row with absent/dead players and invalid-owner stale body slots

Pass criterion: exact parity for every trainer-visible observation frame and
explicit agreement on stack/final/reset metadata. Report first differing row,
field, pixel, owner, bonus slot, and cursor on failure.

Kill criterion: any unexplained trainer-visible mismatch, any avatar-color
controlled-view mismatch, any hidden host readback in the proposed hot loop, or
speedup disappearing once stack/reset/readback are included.

## Concrete Test Targets

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`
- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `tests/test_source_state_gpu_render_benchmark_cpu.py`
- `tests/test_vector_visual_observation.py`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`

