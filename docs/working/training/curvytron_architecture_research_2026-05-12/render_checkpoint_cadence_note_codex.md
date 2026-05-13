# Render Checkpoint Cadence Note

Date: 2026-05-13

Context: `curvy-mix2-clean-20260513a` launched at 2026-05-13 07:49 EDT with
156 rows under `curvyzero-lightzero-curvytron-visual-survival-train`.

## Main Finding

Do not treat the first all-row status sweep as evidence that
`body_circles_fast` checkpoints faster than `browser_lines`.

The manifest is paired by render mode, but its row order is fast-first inside
each recipe/base block:

- main recipes emit 9 `body_circles_fast` rows, then 9 `browser_lines` rows;
- control recipes emit 3 `body_circles_fast` rows, then 3 `browser_lines`
  rows;
- the grouped submitter spawns rows in manifest order.

That means the first sweep split, `body_circles_fast` 34/78 at `iteration_0`
versus `browser_lines` 22/78 at `iteration_0`, is launch-order confounded. It
is a startup/queue read, not a render-speed read.

## Current Evidence

Manifest constants for this batch remove several alternative explanations
within the batch:

- render rows: 78 `body_circles_fast`, 78 `browser_lines`;
- `num_simulations=8` for all rows;
- `collector_env_num=32`, `n_episode=32`, `batch_size=32` for all rows;
- `save_ckpt_after_iter=10000` for all rows;
- varying knobs that remain relevant: opponent mixture recipe and action-repeat
  profile (`rep0`, `repM`, `repH`).

A later status pull still had only one observed `iteration_10000` row:
`r50-blank50-rf-s8-c32-l32-rep0-k10-c2`. That is not enough to compare render
fidelity. Its matched browser row had not also produced `k10`, so this is still
startup/order evidence, not paired cadence evidence.

## First Valid k10 Read

For first-checkpoint cadence, compare only matched pairs where both renders have
started and both have produced `iteration_10000`.

Matched key should be:

- same recipe;
- same repeat profile;
- same copy/seed;
- same `num_simulations`, collector count, `n_episode`, batch size, and
  checkpoint interval.

For each matched pair, record:

- train call spawn position or submission order;
- trainer-owned heartbeat/start timestamp if available;
- `iteration_0` checkpoint mtime;
- `iteration_10000` checkpoint mtime;
- elapsed time from trainer start to `k10`;
- elapsed time from `iteration_0` to `k10`;
- poller/eval status separately from trainer checkpoint status.

The render comparison should use paired deltas, for example
`browser_k10_elapsed - fast_k10_elapsed`, then summarize median/min/max by
repeat profile and by opponent recipe. Rows without both sides at `k10` should
stay in the liveness bucket.

## Future Manifest Shape

Future manifests should interleave or deterministically shuffle render order if
we want startup or checkpoint timing to answer a speed question. A clean order
would emit matched pairs adjacent with alternating first render, for example:

- recipe A, rep0, seed 1: fast then browser;
- recipe A, rep0, seed 2: browser then fast;
- recipe A, repM, seed 1: fast then browser;
- recipe A, repM, seed 2: browser then fast.

The important part is not randomness by itself; it is removing the systematic
fast-first submission advantage.
