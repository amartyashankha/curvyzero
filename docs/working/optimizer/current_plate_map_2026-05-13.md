# Current Optimizer Plate Map

Date: 2026-05-13

Purpose: one plain page to prevent old optimizer notes from pulling us back
into stale lanes.

## Main Lane

The current trusted training/profile lane is stock LightZero:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--opponent-use-cuda=false
```

The current trusted visual surface is CPU-reference `browser_lines`:

```text
source-state RGB 704x704
-> browser-sprite bonuses
-> live heads
-> BT.601 luma
-> exact 11x11 area average
-> uint8[1,64,64]
-> frame stack
```

Say **CPU-reference fidelity**, not browser-pixel fidelity.

The near-term LightZero input contract is `[4,64,64]` grayscale stack. The env
wrapper owns the FIFO stack and LightZero is configured as a conv model with
`image_channel=4` and `frame_stack_num=1`. Any RGB or semantic-plane canary is a
separate model/config lane and needs its own smoke before Coach relies on it.

Latest Coach refactor context:
[system architecture map](system_architecture_map_2026-05-13.md). Plain read:
the stock `--mode train` lane is still the current source of truth. The
refactor has extracted timestamped checkpoint discovery and opponent assignment
parsing, and the focused local gate currently passes. Do not interfere with the
live Coach batches while profiling or documenting optimizer work.

## Active Plates

1. **Reference/optimized parity**
   Dirty-cache and GPU paths must byte-match the CPU reference. Gray64-only
   paths need exact gray64 parity. RGB-cached paths also need exact RGB parity.

2. **Downsample information quality**
   The 11x11 area average is the right anti-aliased family, but grayscale can
   still lose category information. Current 2P self/other colors are separated;
   future 3P/4P needs a palette or channel decision. Working doc:
   [downsample/reference fidelity](downsample_reference_fidelity_2026-05-13.md).
   Overnight-safe recommendation remains `[4,64,64]` gray until diagnostics or
   canaries prove that the contract should change.

3. **Full-loop Amdahl**
   Render matters in long-survival/no-death profiles, but after dirty-cache
   fixes it may not be the whole full-loop bottleneck. Re-check stock LightZero
   profiles before choosing the next production target.

4. **GPU renderer research**
   Keep isolated. Do not promote until real-state feed, bonus sprite parity,
   exact CPU-reference parity, transfer timing, and device-resident policy
   handoff are proven.

5. **Browser golden-frame side lane**
   Useful, not the optimizer oracle. Environment Reconstruction owns browser
   pixel claims. Optimizer can help define a tiny capture/proof harness, but
   training speed work should not block on it. Working note:
   [browser golden capture side lane](browser_golden_capture_side_lane_2026-05-13.md).

## Do Not Drift Back To

- `fast_gray64_direct` as a current stock-path recommendation. That name belongs
  to the old custom two-seat adapter.
- `body_circles_fast` as the trusted visual surface. It is a control/ablation.
- Old `--mode two-seat-selfplay` learning conclusions as evidence against
  CurvyTron or LightZero.
- Browser-pixel claims without a browser golden-frame harness.

## Immediate Next Checks

- Keep the downsample regression tests passing.
- Keep the refactor-focused local gate green when optimizer advice depends on
  current Coach scaffolding:
  `tests/test_lightzero_timestamped_checkpoint_discovery.py`,
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py`,
  `tests/test_curvytron_run_status.py`, `tests/test_opponent_mixture.py`, and
  `tests/test_opponent_registry.py`.
- Add broader downsample metrics only when they answer a concrete question:
  luma collision, phase sweep, bonus signature separability, or small-feature
  visibility.
- Wait for the visual-RL preprocessing research and browser-capture feasibility
  side reports, then fold the useful parts into the docs.
- Before changing live training, run a fresh full-loop stock profile with named
  buckets: env step, render/observation, frozen opponent, MCTS/search,
  replay/sample, learner, checkpoints, eval/GIF, and artifact I/O.
