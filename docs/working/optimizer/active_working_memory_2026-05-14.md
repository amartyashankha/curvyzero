# Optimizer Active Working Memory

Date: 2026-05-14

Purpose: short live state for the optimizer lane. Read this before older
optimizer notes if the docs disagree.

## Role

Optimizer owns speed, profiling, setup advice, and renderer/search performance
experiments. Coach owns learning claims and which checkpoints are good. Do not
touch live overnight training runs except read-only.

## Current Trusted Path

Training/profiling source of truth is stock LightZero:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
```

The trusted visual reference is CPU `browser_lines`:

```text
source-state RGB 704-style canvas
-> browser-sprite bonuses
-> BT.601 luma
-> 11x11 area average
-> uint8[1,64,64]
-> [4,64,64] stack for LightZero
```

This is CPU-reference fidelity, not browser-pixel fidelity.

## Current Speed Read

- Local long no-death env profiles say render is still a large bucket for
  `browser_lines`. The latest direct-fast comparison showed about `3.4x` to
  `8.2x` env-only speedup for the fast gray64 path versus `browser_lines`.
- Full stock LightZero profiles are less renderer-pure. The measured
  full-loop gain from the fast approximation was about `1.3x` to `1.5x`,
  because collection, search, policy forward, subprocess overhead, and learner
  time remain in the loop.
- Amdahl read: render work can still matter a lot for long-survival policies,
  but render-only work cannot produce a `10x` full-loop win unless render is
  more than `90%` of the actual full-loop wall time.
- Bigger GPUs are not automatically better. H100 helped more when search sims
  were heavier, but the sim8 full-loop rows did not justify H100 by default.

## Active Plates

1. **Bonus symbol renderer**
   We have 12 active bonus sprite types. A simplified symbol renderer makes
   sense as an opt-in training/GPU experiment if it preserves type, group,
   position, size, and grayscale visibility. Reference stays `browser_sprites`.
   Plan: [bonus symbol render plan](bonus_symbol_render_plan_2026-05-14.md).
   Current fast luma circles are only semi-principled: values are spaced apart,
   but the scheme was not chosen from a real separability sweep. The 2026-05-14
   probe found no exact centered collisions, but did find near pairs and a
   `BonusGameClear` player-head-remap caveat.

2. **GPU browser-lines renderer**
   The GPU prototype is promising but not trusted. Trail/head parity is close;
   bonus sprites and exact draw semantics are the gap. Current note:
   [GPU render parity gap](gpu_render_parity_gap_2026-05-13.md).

3. **Full-loop scaling**
   Keep measuring stock LightZero with clear buckets. Relevant knobs are
   collector width, search sims, learner batch, CPU count, GPU class, and
   trajectory length. Do not infer learning quality from speed profiles.

4. **Moving environment**
   Environment Reconstruction is still changing source-state/render details.
   Every speed number must name code state, render mode, bonus mode, warmup,
   trajectory length, and whether search/learner are included.

## To Do

- [x] Harvest sprite inventory and practical symbol-design subagents.
- [x] Write down the 12-bonus symbol plan and Amdahl implication.
- [x] Harvest the running signature-probe subagent and fold results into the
  symbol plan.
- [ ] Harvest the symbol-mask design, symbol-lane value critique, and original
  sprite-shape probe subagents.
- [ ] If symbol signatures look safe, add an explicit opt-in
  `simple_symbols` bonus render mode rather than changing defaults.
- [ ] Gate any new render mode with separability, offset, size, CPU/GPU parity,
  and train/eval render-mode metadata tests.
- [ ] Rerun small env-only and stock full-loop profiles only after the current
  environment/render changes settle enough to compare.

## Operating Pattern

- Main thread is for planning, delegation, orchestration, and short synthesis.
- Use subagents for bounded code reading, toy probes, literature/research, and
  critique.
- Keep docs current before details fall out of context.
- Do not change live training runs. New renderer experiments must be opt-in.
