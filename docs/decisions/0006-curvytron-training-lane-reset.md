# 0006 - CurvyTron Training Lane Reset

Date: 2026-05-12

## Decision

Reset CurvyTron training claims around the stock LightZero loop.

Do not treat `--mode two-seat-selfplay` as the main learning lane until it
either feeds native LightZero replay/targets or has parity-tested repo-owned
targets.

## Context

The May 12 scaled CurvyTron runs used a custom two-seat path. That path solved
same-current-policy joint action collection, but it also bypassed stock
`train_muzero`, stock collector, native `GameSegment`, and `MuZeroGameBuffer`.

The resulting flat survival curves do not prove CurvyTron is unlearnable. They
show that the custom training contract was not a safe thing to scale.

## Consequences

- Stock fixed/frozen opponent runs are controls, not full self-play.
- The stock frozen-checkpoint opponent route has a passed tiny CPU canary:
  `stock-frozen-canary-source-state-s304-20260512` called stock
  `train_muzero` and loaded the frozen checkpoint strictly.
- The same stock frozen-checkpoint route also has a passed tiny L4 GPU canary:
  `stock-frozen-gpu-base-canary-source-state-s304-20260512b`.
- Stock centralized joint-action runs are controls, not competitive self-play.
- Turn-commit remains blocked for training.
- The true two-seat path needs a native replay bridge or target parity tests.
