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

## Action Selection Snapshot

Current overnight rows use:

```text
action_selection_mode=collect
collect_temperature=1.0
collect_epsilon=0.25
```

The two-seat adapter passes those directly to
`MuZeroPolicy.collect_mode.forward(...)`. This is the training collection path,
not the deterministic checkpoint GIF/eval path.

Meaning:

- training collection uses LightZero collect-mode MCTS and samples from MCTS
  visit counts with temperature `1.0`;
- the checkpoint GIF/eval path calls `MuZeroPolicy.eval_mode.forward(...)`,
  which LightZero documents as choosing the highest-value action rather than
  sampling;
- therefore a one-action greedy GIF does not by itself prove the training
  collection data is collapsed.

Observed row `01` at iteration `10`: training collect actions were varied, with
top action fractions around `0.38` per player and no collapse warning.

Important caveat: the 40-run matrix does not currently sweep collect
temperature or collect/eval action-selection mode. That is a valid follow-up
knob if these runs do not show useful learning signal.

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

## Collect-Mode GIF Subscriber

Added on `2026-05-12` after the 40 runs were already live:

```text
src/curvyzero/infra/modal/curvytron_collect_t1_gif_subscriber.py
```

Purpose: keep the existing training jobs alive and backfill the new
`collect_t1.gif` view without relaunching training. The old per-run pollers were
started before the two-GIF worker change, so they can keep writing only
`raw.gif`. This sidecar scans the 40 overnight run checkpoint directories and
spawns the current GIF worker for checkpoints missing:

```text
attempts/<attempt_id>/eval/live_checkpoint_iteration_N/selfplay/collect_t1.gif
```

Behavior:

- deployed as Modal app `curvyzero-curvytron-collect-t1-gif-subscriber`;
- runs every five minutes via `modal.Period(minutes=5)`;
- uses the hard-coded overnight40 run and attempt ids from this launch;
- writes state outside training attempts under:
  `sidecars/lightzero-curvytron-visual-survival/collect-t1-gif-subscriber/overnight40a-20260512/`;
- writes `manifest.json`, `latest_tick.json`, and `ticks/<utc_stamp>.json`;
- does not touch training jobs, checkpoints, or run markers;
- dispatches one Modal worker function per missing checkpoint, so GIF creation is
  parallel across checkpoints;
- each worker writes the same `raw.gif` and `collect_t1.gif` pair that the
  current checkpoint GIF worker would have written if the run had started with
  the newer code.

Validation:

- `py_compile` and `ruff` passed for the subscriber module.
- Dry run found all 40 run roots and no missing roots.
- One real smoke created `collect_t1.gif` for row 01 `iteration_100`.
- A broad manual tick then spawned `80` checkpoint GIF workers with no spawn
  failures.
- A later scan saw `81` existing `collect_t1.gif` files across `106` observed
  checkpoints.
- A second broad manual tick spawned `26` more workers for the remaining
  observed gaps.
- Follow-up dry run at `2026-05-12T06:42Z`: `109` of `112` observed
  checkpoints already had `collect_t1.gif`; `2` were still pending from a recent
  spawn, and `1` new checkpoint had arrived after the manual backfill. The
  scheduled five-minute tick should keep closing these small gaps as checkpoints
  continue to arrive.

Important caveat: the sidecar manifest records spawned jobs, and a later tick
marks them done when `collect_t1.gif` appears. New checkpoints that arrive after
the manual backfill are handled by the scheduled five-minute tick.

## Matrix

The 40 rows match the optimizer recommendation:

- Rows 01-32: L4/T4 `fast_gray64_direct` main, seeds, search, batch, collect,
  update, learner sample, learning-rate, reward, and stochasticity variants.
- Rows 33-38: H100 scale probes.
- Rows 39-40: `browser_lines` sentinels only.
