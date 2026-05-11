# 2026-05-09 dummy pong geometry CEM smoke

## Question

Can one tiny alternative learner baseline replace the failed old self-play lane
as a scoreable, survival-aware Pong checkpoint?

## Setup

- New trainer:
  `src/curvyzero/training/dummy_pong_cem_train.py`
- Learner: cross-entropy/random search over only the six geometry features in
  the existing raster feature encoding, embedded back into the normal full
  linear raster checkpoint shape.
- Initial mean: a geometry prior equivalent to `track_ball`.
- Selection opponents: `random_uniform` weighted `0.25`, `track_ball` weighted
  `0.75`.
- Selection proxy:
  `win=+1.0`, `loss=-1.0 + 0.5 * steps / max_steps`, `truncation=0.0`.
- Smoke size: 2 generations, 8 candidates, 3 elites, 4 paired-seat eval games
  per opponent inside training.

## Command

```sh
uv run python -m py_compile src/curvyzero/training/dummy_pong_cem_train.py
```

```sh
uv run python -m curvyzero.training.dummy_pong_cem_train \
  --generations 2 \
  --population-size 8 \
  --elite-count 3 \
  --eval-games 4 \
  --seed 8050911 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-geometry-cem-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 9050911 \
  --split-id dummy_pong_geometry_cem_smoke \
  --split-role smoke \
  --checkpoint cem=artifacts/local/dummy-pong-geometry-cem-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-geometry-cem-scoreboard-smoke-2026-05-09
```

## Results

Compile passed. Training and scoreboard both passed. The checkpoint loaded
through the existing `learned:<checkpoint.npz>` scoreboard path as
`learned_cem`, with checkpoint schema
`dummy_pong_geometry_cem_policy_checkpoint_v0`.

Trainer final eval, 8 paired-seat episodes per opponent:

| Opponent | Wins | Losses | Truncations | Mean steps | Median steps | Mean shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_uniform` | 8/8 | 0/8 | 0/8 | 25.875 | 19.0 | 1.0 |
| `track_ball` | 0/8 | 0/8 | 8/8 | 120.0 | 120.0 | 0.0 |

Existing checkpoint scoreboard, 16 paired-seat learned-vs-baseline episodes:

| Row | Learned wins | Baseline wins | Truncations | Mean steps | Mean reward, learned |
| --- | ---: | ---: | ---: | ---: | ---: |
| `learned_cem` vs `random_uniform` | 16/16 | 0/16 | 0/16 | 21.75 | 1.0 |
| `learned_cem` vs `track_ball` | 0/16 | 0/16 | 16/16 | 120.0 | 0.0 |

The best selected search candidate was generation 1 candidate 0, the
`track_ball` geometry prior. Later random perturbations matched the same
selection score but did not beat it on the tie-breaks.

## Interpretation

This should replace the old blind self-play lane as the immediate narrow Pong
survival baseline: it is tiny, scoreable by the existing scoreboard, and its
failure mode is legible.

It should not be promoted as a strategic Pong learner. It learns/selects a
geometry policy that beats random and survives forever against `track_ball`,
but it still has 0 wins against `track_ball`. The right read is "survival floor
established, win pressure absent."

There is no dedicated Modal wrapper for this CEM trainer yet. The existing Pong
Modal train wrapper targets the old self-play replay/train path, so do not
launch a long Modal run from this smoke without first adding a CEM-specific
wrapper or intentionally broadening the old wrapper.

## Artifacts

- `artifacts/local/dummy-pong-geometry-cem-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-geometry-cem-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-geometry-cem-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-geometry-cem-scoreboard-smoke-2026-05-09/episodes.jsonl`

## Follow-ups

Use this as the cheap survival-aware baseline when evaluating future Pong
learners. A next learner should beat this row on at least one hard dimension:
win against `track_ball`, or preserve the 120-step truncation floor while
showing deliberate score pressure in a separate diagnostic.

Update after the beatability probe: default `track_ball` is unwinnable in this
geometry, so the next CEM run should not use it as the hard win target. CEM-v2
is prepared as a score-primary learner for the ladder-selected target: configure
geometry and opponents with `--width`, `--height`, `--paddle-height`, repeated
`--opponent-weight POLICY=WEIGHT`, and `--target-opponent-id`. See
`docs/experiments/2026-05-09-dummy-pong-cem-v2-score-pressure-plan.md`.
