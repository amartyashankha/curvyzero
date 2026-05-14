# Optimizer System Architecture Map

Date: 2026-05-13

Purpose: keep one plain map of the current CurvyTron training system so the
optimizer does not drift back into old custom paths.

## Current Trusted Training Path

Use the Coach/stock LightZero lane:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
```

This path calls stock `lzero.entry.train_muzero`. LightZero owns collection,
MCTS/search, replay, target construction, learner updates, and stock checkpoint
creation.

CurvyZero owns the surrounding scaffolding: config construction, env
registration, run metadata, progress/status files, checkpoint discovery,
resume helpers, checkpoint eval/GIF scheduling, and Modal entrypoints.

The old `--mode two-seat-selfplay` path is historical/custom-adapter evidence.
Do not use it for learning claims or current optimizer recommendations unless
the user explicitly reopens that lane.

## Current Coach Refactor Status

The active refactor folder is:

```text
docs/working/training/lightzero_train_refactor_2026-05-13/
```

Latest read:

- Timestamped `lightzero_exp*` checkpoint discovery was extracted into
  `src/curvyzero/training/lightzero_checkpoints.py`.
- Frozen-opponent assignment parsing was started in
  `src/curvyzero/training/opponent_registry.py`.
- The giant Modal launcher still exists, but tests now pin several important
  contracts before larger cuts.
- Tiny Modal CPU train and artifact smokes are recorded as passed in the Coach
  docs. Those prove plumbing, not learning.
- The current open contract is action cadence: the trusted path should mean one
  policy action per granular source game step unless action repeat is explicitly
  enabled.

Local gate rerun by Optimizer on 2026-05-13:

```text
uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_curvytron_run_status.py \
  tests/test_opponent_mixture.py \
  tests/test_opponent_registry.py -q

102 passed, 1 skipped

uv run ruff check src/curvyzero/training/lightzero_checkpoints.py \
  src/curvyzero/training/opponent_registry.py \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  tests/test_lightzero_timestamped_checkpoint_discovery.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_opponent_registry.py \
  docs/working/training/lightzero_train_refactor_2026-05-13

All checks passed
```

## Current Live Run Context

Read-only Coach docs say the current large surfaces are:

- `curvy-survive-bonus-large-20260513b`: 300-row survival-plus-bonus diagnostic
  batch.
- `curvy-mix2-clean-20260513a`: 156-row opponent-mixture batch.
- `curvy-mix3-currentckpt-20260513a`: 300-row current-checkpoint mixture batch.

Do not interfere with those live runs. Optimizer may read docs and local
artifacts, but should not cancel calls, mutate Modal volumes, or publish
profile artifacts into the GIF/tournament websites unless explicitly asked.

## Current Visual Contract

Trusted input surface:

```text
source-state RGB 704x704
-> browser-sprite bonuses
-> live heads
-> BT.601 luma
-> exact 11x11 area average
-> uint8[1,64,64]
-> FIFO stack [4,64,64]
```

This is CPU-reference `browser_lines` fidelity. It is not a browser-pixel
claim. Optimized CPU dirty-cache or GPU render paths must byte-match this CPU
reference before being treated as equivalent.

`body_circles_fast` is an ablation/control path in the stock lane. It is not
the trusted visual target.

## Artifact And Eval Boundaries

Training writes checkpoints and progress. Background eval/GIF workers read
checkpoints and write human/debug artifacts. Tournament code is a separate
artifact consumer.

For optimizer profiles:

- turn off background eval/GIF unless that overhead is the explicit target;
- do not write website markers for profile runs;
- keep checkpoint cadence sparse unless measuring checkpoint overhead;
- record whether eval, GIF, checkpoint, and volume I/O are included.

## Current Optimizer Questions

1. Is the post-refactor stock lane still bottlenecked by render/observation,
   MCTS/search, frozen-opponent inference, replay/sample, learner, or artifact
   I/O?
2. Does one LightZero env step now equal one granular source game step in the
   trusted defaults?
3. After the environment/refactor churn settles, what fresh full-loop profile
   should be used as the next Amdahl baseline?
4. Can a GPU/compiled/block-local renderer beat the CPU dirty-cache path while
   preserving exact CPU-reference gray64 output?

## Ownership

- Coach owns learning claims, run matrices, checkpoint quality, eval curves,
  and current live jobs.
- Environment Reconstruction owns source fidelity and browser-pixel claims.
- Optimizer owns timing, profiling, setup advice, renderer/search performance
  probes, and clear Amdahl reports.

