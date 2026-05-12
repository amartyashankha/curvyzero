# CurvyTron Overnight40 Launch - 2026-05-12

## Source

Launch plan comes from:

- `docs/working/optimizer/coach_next_training_run_recommendations_2026-05-12.md`
- `docs/working/optimizer/current_status_2026-05-09.md`

## Launcher

Use only the canonical two-seat current-policy self-play launcher:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode two-seat-selfplay
```

The launch script is:

```text
scripts/launch_curvytron_overnight40_20260512.zsh
```

## Global Settings

- Batch tag: `overnight40a`
- Max train iterations: `3000`
- Checkpoint cadence: every `50` iterations
- Stock LightZero in-loop eval: off via `--lightzero-eval-freq 0`
- CurvyZero checkpoint eval: on by default
- CurvyZero self-play GIF: on by default
- Background eval launch kind: `poller`
- Initial checkpoints: default on
- Death mode: `normal`
- Replay scope: `accumulated`
- Max replay rows: `65536`

## Main Surface

The main training surface is `fast_gray64_direct`.

`browser_lines` appears only as rows `39` and `40`, as slow fidelity sentinels.
They are not a gate for the main approximation-heavy matrix.

## Naming Rule

Rows `01`-`22` were launched before the short-name patch and use:

```text
curvy2seat-selfplay-overnight40a-<row>-<lane>-<render>-b<B>-sim<S>
```

Rows `23`-`40` use the shorter safe form:

```text
curvy2seat-overnight40a-<row>-<lane>-<renderTag>-b<B>-sim<S>
```

Reward, stochasticity, and learning-rate variants are appended to the run id
when they differ from default.

## Launch Status

As of `2026-05-12T05:29Z`, all 40 rows have run directories in the
`curvyzero-runs` Modal volume. The resumed rows `23`-`40` all returned
`status=spawned` in `logs/curvytron_overnight40a_launch_20260512.log`.

Website visibility marker:

```text
training/lightzero-curvytron-visual-survival/<run_id>/show_in_gif_browser.flag
```

was verified for representative rows `01`, `23`, `33`, `39`, and `40`.
Rows `39` and `40` appeared after normal startup delay.

Background checkpoint handling was verified for row `40`: the checkpoint eval
poller status exists at `train/checkpoint_eval_poller.json` with `status=running`.
The row also wrote `progress_latest.json` with `event=start`.

## Checkpoint Timing Snapshot

Checkpoint cadence is configured as every `50` training iterations for every row
via `--save-ckpt-after-iter 50`.

The initial checkpoint `iteration_0.pth.tar` appears immediately. The first
useful training checkpoint is `iteration_50.pth.tar`.

As of `2026-05-12T05:31Z`, row `01` reached iteration `10` after `307.0`
seconds. That implies roughly `30.7` seconds per iteration, or about `25.6`
minutes per 50-iteration checkpoint for the main `B64/sim8/collect64/updates4`
fast-gray run.

Expected first useful main-row checkpoints: roughly `2026-05-12T05:48Z` to
`2026-05-12T05:52Z`, depending on launch time and row shape. Browser sentinel
and larger H100/B128 variants need their own first iteration-10 progress before
we should trust a timing estimate.

## Artifact Paths

For each run id and attempt id:

- Train summary:
  `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/summary.json`
- Action observability:
  `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/action_observability.json`
- Checkpoint eval poller status:
  `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/checkpoint_eval_poller.json`
- Checkpoints:
  `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero`

## Matrix

The 40 rows match the optimizer recommendation:

- Rows 01-32: L4/T4 `fast_gray64_direct` main, seeds, search, batch, collect,
  update, learner sample, learning-rate, reward, and stochasticity variants.
- Rows 33-38: H100 scale probes.
- Rows 39-40: `browser_lines` sentinels only.
