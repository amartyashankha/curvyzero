# LightZero Dummy Pong Eval Ladder - 2026-05-09

Worker lane: define the immediate eval ladder only. This note does not
implement a LightZero dummy Pong adapter, checkpoint loader, exporter, trainer,
or core evaluator change.

## Current Boundary

The immediate LightZero dummy Pong eval must reuse the existing CurvyZero Pong
baselines and telemetry shape:

- evaluator: `src/curvyzero/training/dummy_pong_eval.py`
- scoreboard wrapper: `scripts/run_dummy_pong_checkpoint_scoreboard.py`
- fixed policies already present: `random_uniform`, `lagged_track_ball_1`,
  `track_ball`

The current scoreboard can load CurvyZero supervised raster `.npz`
checkpoints through `learned:` specs. A narrow CurvyZero-owned direct
policy-head path can also score the tiny LightZero dummy Pong `.pth.tar`
checkpoint greedily.

Tiny LightZero dummy Pong train state:

- Run: `lz-dpong-20260509T141607Z-3696aa333028`.
- Attempt: `attempt-20260509T141607Z-98662e4917b4`.
- Mirrored checkpoints: `ckpt_best.pth.tar`, `iteration_0.pth.tar`,
  `iteration_2.pth.tar`.
- Probe passed policy-head access, but strict full model load failed due a
  dynamics-only key mismatch.
- Probe ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/attempts/attempt-20260509T141607Z-98662e4917b4/probe/lightzero_checkpoint_probe_20260509T143137Z.json`.
- Greedy policy-head scoreboard refs:
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/summary.json`
  and
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/episodes.jsonl`.

This scoreboard is not MCTS eval and not proof of a good learned policy. Raw
matchups show `lightzero_best` used constant up in every LightZero row:
`[N, 0, 0]`. Do not treat 21/40 versus `random_uniform` as learning proof. The
ladder below remains the required eval ladder for honest future scoring.

## Immediate Ladder

Run rows in this order.

1. `random_uniform`
   - Purpose: sanity floor and stochastic canary.
   - Expected use: a learned LightZero checkpoint should eventually beat this
     on score/wins. If wins are unstable early, survival/loss-delay still
     reports whether the policy is doing anything less random.

2. `lagged_track_ball_1`
   - Purpose: first score-pressure rung.
   - Reason: this target keeps the default dummy Pong geometry and normal
     resets, but the one-step tracking lag makes it scoreable.
   - Expected use: primary early win target once the LightZero policy can be
     loaded for independent eval.

3. `track_ball`
   - Purpose: survival/tie floor, not a hard win gate.
   - Reason: in the current default geometry, exact search found no normal-reset
     ego sequence that scores against deterministic `track_ball` before the
     120-step cap.
   - Expected use: report survival, truncation, and shaped loss-delay. Do not
     reject early checkpoints only because they have `0/N` wins here.

4. learned-vs-learned
   - Purpose: regression and checkpoint progression check.
   - Gate: do this only after a LightZero checkpoint loader or exporter exists
     and can load at least two LightZero checkpoints through the same inference
     boundary.
   - Expected use: compare `previous`, `latest`, and later `selected_best`.
     This is never a replacement for the fixed baseline rows above.

## Metric Bundle

Every LightZero dummy Pong eval row must report score/win metrics plus
survival/loss-delay metrics.

Required score fields:

- wins by policy
- losses by policy, inferred from opponent wins when the episode terminates
- truncations/timeouts
- score return stats by policy

Required survival/loss-delay fields:

- mean, median, p90, min, max, and std survival steps
- truncation rate
- shaped loss-delay return stats by policy
- action histograms; if summary rows omit them, read raw matchups and fix the
  summary reporting

Use the existing shaped diagnostic:

```text
win:     +1.0
loss:    -1.0 + 0.5 * (episode_steps / max_steps)
timeout:  0.0
```

This shaped value is diagnostic and may be useful for early checkpoint
selection. It is not the environment reward. The honest dummy Pong reward stays:

```text
ego scores:       +1
opponent scores:  -1
no score event:    0
```

Survival alone is acceptable as an early signal when wins are not meaningful,
especially against `track_ball`. Any report using survival alone must say that
the row is a survival/tie read, not a scoring claim.

## Current Policy-Head Scoreboard

Direct policy-head greedy rows from the Modal scoreboard:

| Opponent | LightZero wins | Opponent wins | Truncations | Mean score | Shaped mean | Mean steps | p90 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_uniform` | 21/40 | 19/40 | 0 | 0.05 | 0.081875 | 13.775 | 20.1 |
| `lagged_track_ball_1` | 16/40 | 20/40 | 4 | -0.1 | -0.0764583 | 21.95 | 29.1 |
| `track_ball` | 0/40 | 38/40 | 2 | -0.95 | -0.8736458 | 24.325 | 41.0 |

Read this as loader/inference evidence only. Raw matchups include
`action_histogram_by_policy` and show constant-up LightZero actions:
`[N, 0, 0]` in every LightZero row. The `scoreboard_rows` summary omits that
field, so surfacing action histograms in summary rows is a telemetry/reporting
cleanup item.

## Exact Next LightZero Eval Plan

After strict load or a documented compatible full-model loader exists:

1. Load one LightZero dummy Pong checkpoint as a CurvyZero policy with:
   - exact training observation schema
   - exact action schema `0=up`, `1=stay`, `2=down`
   - checkpoint path/ref, hash, run id, attempt id, config ref, seed, and
     adapter/exporter schema id recorded
   - clear statement of eval mode: greedy policy head, LightZero eval-mode
     policy, or MCTS

2. Run a selection split scoreboard with paired seats against:
   - `random_uniform`
   - `lagged_track_ball_1`
   - `track_ball`

3. Rank the checkpoint read in this order:
   - score/win rate against `lagged_track_ball_1`
   - score/win rate against `random_uniform`
   - survival/loss-delay against `track_ball`
   - action histogram and action-logit sanity; constant actions must be called
     out directly

4. If there are at least two loadable LightZero checkpoints from the same
   adapter/exporter path, add learned-vs-learned rows:
   - `previous` vs `latest`
   - `selected_best` vs `latest` only after a selection record exists

5. Run heldout only after a checkpoint is selected. Heldout must include the
   selected checkpoint, `latest`, all three fixed baselines, and any
   learned-vs-learned regression rows that were used in selection.

## Reporting Rules

- Name the algorithm as `LightZero MuZero` in the run title and summary.
- Do not call CEM, supervised MLP, imitation, NumPy self-play, or Mctx benchmark
  rows LightZero or MuZero progress.
- Do not collapse Pong reports to `0/N wins`; include survival steps,
  truncation rate, score return, shaped loss-delay return, and variance/std.
- Do not report LightZero scorecard rows without action histograms. If summary
  rows omit `action_histogram_by_policy`, read the raw matchups and fix the
  reporting.
- Do not require wins against `track_ball` for early progress; treat
  `track_ball` as a survival/tie floor unless the geometry or reset support
  changes and is re-proven scoreable.
- Do not run learned-vs-learned before the full LightZero checkpoint loading or
  export boundary exists.
