# Optimizer System Architecture Map

Date: 2026-05-13
Updated: 2026-05-15

Purpose: keep optimizer work aligned with the current CurvyTron stock LightZero
path.

## Current Training Path

Use the Coach/stock LightZero lane:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
```

This path calls stock `lzero.entry.train_muzero`. LightZero owns collection,
MCTS/search, replay, target construction, learner updates, and stock checkpoint
creation.

CurvyZero owns config construction, env registration, run metadata,
progress/status files, checkpoint discovery, resume helpers, checkpoint eval/GIF
scheduling, and Modal entrypoints.

The old `--mode two-seat-selfplay` path is historical/custom-adapter evidence.
Do not use it for current optimizer recommendations unless the user explicitly
reopens that lane.

## Current Visual Contract

Current policy-observation target:

```text
source-state geometry
-> browser_lines trails
-> simple_symbols bonuses
-> live heads
-> BT.601 luma
-> exact 11x11 area average
-> uint8[1,64,64]
-> FIFO stack [4,64,64]
```

The target future implementation is batched GPU-rendered.

CPU `browser_lines + simple_symbols` is the production backend and parity oracle
today. It remains the recommendation until a batched GPU backend proves the
same observation contract inside the trainer.

`browser_sprites` are for artifact/reference/browser-fidelity views.
`body_circles_fast` and `fast_gray64_direct` are historical/prototype renderer
names, not current training or tournament policy surfaces.

## Current Coach Context

The refactor folder is:

```text
docs/working/training/lightzero_train_refactor_2026-05-13/
```

Relevant status from the earlier optimizer read:

- timestamped `lightzero_exp*` checkpoint discovery was extracted into
  `src/curvyzero/training/lightzero_checkpoints.py`;
- frozen-opponent assignment parsing was started in
  `src/curvyzero/training/opponent_registry.py`;
- focused local gates passed at the time of the note;
- tiny Modal CPU train and artifact smokes prove plumbing, not learning.

Do not interfere with Coach live runs while profiling or documenting optimizer
work.

## Profile Boundaries

For optimizer profiles:

- turn off background eval/GIF unless that overhead is the explicit target;
- do not write website markers for profile runs;
- keep checkpoint cadence sparse unless measuring checkpoint overhead;
- record whether eval, GIF, checkpoint, and volume I/O are included;
- record whether render was GPU target, CPU oracle/fallback, artifact sprites,
  or a historical control.

## Current Optimizer Questions

1. Can GPU `browser_lines + simple_symbols` byte-match the CPU oracle on real
   source-state rows?
2. Can LightZero consume that rendered observation without a forced CPU round
   trip at the env manager boundary?
3. In full-loop stock profiles, how much time remains in render versus
   collection, search, opponent inference, replay, learner, and artifact I/O?
4. What cost remains while CPU `browser_lines + simple_symbols` is used before
   the batched GPU backend exists?

Historical profile answer, 2026-05-13: fresh stock full-loop rows said the
sim8 profile was mainly collection/search/process limited, not pure render and
not GPU-bound. C1 was about `10.8` env steps/sec, C32 `153.6`, C64 `408.4`,
C96 `487.3`, and C64 H100 `321.0`. Use that as speed history, not as a reason
to change the current render target.

## Ownership

- Coach owns learning claims, run matrices, checkpoint quality, eval curves,
  and current live jobs.
- Environment Reconstruction owns source fidelity and browser-pixel claims.
- Optimizer owns timing, profiling, renderer/search performance probes, and
  clear Amdahl reports.
