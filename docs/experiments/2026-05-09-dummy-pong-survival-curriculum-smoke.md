# 2026-05-09 dummy pong survival curriculum smoke

## Question

Can a simpler on-policy visual Pong trainer expose a learning signal beyond
wins, especially survival/loss delay against `track_ball`, while still writing
a checkpoint the existing scoreboard can load?

## Setup

- New trainer:
  `src/curvyzero/training/dummy_pong_survival_curriculum_train.py`
- Policy: per-ego-agent linear softmax over the existing raster one-hot plus
  geometry features.
- Curriculum: `random_uniform` -> noisy `weak_track_ball` ->
  `track_ball`.
- Training return:
  `score_return + 0.5 * episode_steps / max_steps + 0.25 if truncated`.
- Explicit reported signals: wins, losses, truncations, truncation rate, mean
  steps, mean survival fraction, score return, and shaped training return.
- This is a tiny local smoke, not a quality claim.

## Command

```sh
uv run python -m curvyzero.training.dummy_pong_survival_curriculum_train \
  --epochs 3 \
  --games-per-epoch 4 \
  --eval-games 2 \
  --seed 6050911 \
  --max-steps 120 \
  --learning-rate 0.08 \
  --output-dir artifacts/local/dummy-pong-survival-curriculum-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 4 \
  --seed 7050911 \
  --split-id dummy_pong_survival_curriculum_smoke \
  --split-role smoke \
  --checkpoint survival=artifacts/local/dummy-pong-survival-curriculum-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-survival-curriculum-scoreboard-smoke-2026-05-09
```

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_survival_curriculum_train.py
```

## Results

Trainer final eval, 4 paired-seat episodes per opponent:

| Opponent | Wins | Losses | Truncations | Mean steps | Mean survival | Mean score return | Mean shaped return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_uniform` | 2/4 | 2/4 | 0/4 | 13.5 | 0.1125 | 0.0 | 0.05625 |
| `weak_track_ball` | 2/4 | 2/4 | 0/4 | 13.5 | 0.1125 | 0.0 | 0.05625 |
| `track_ball` | 0/4 | 2/4 | 2/4 | 75.0 | 0.625 | -0.5 | -0.0625 |

Existing scoreboard, 8 paired-seat learned-vs-baseline episodes:

| Row | Learned wins | Baseline wins | Truncations | Mean steps |
| --- | ---: | ---: | ---: | ---: |
| `learned_survival` vs `random_uniform` | 6/8 | 2/8 | 0/8 | 16.25 |
| `learned_survival` vs `track_ball` | 0/8 | 6/8 | 2/8 | 42.875 |

The scoreboard also confirmed the checkpoint loaded through the existing
`learned:<checkpoint.npz>` path with checkpoint schema
`dummy_pong_survival_curriculum_policy_checkpoint_v0`.

## Interpretation

This is a better artifact shape than the failed blind self-play sweep because
the trainer's own artifacts make survival/loss-delay visible. It does not yet
prove a strong policy. The checkpoint still gets 0 wins against `track_ball`;
the useful signal is that the same artifact reports how often it survives to
truncation and how long losses take.

Follow-up correction: broad new-trainer work should pause until the
survival/loss-delay audit is used to decide whether the existing self-play
trainer is broken, undertrained, or just too short. This smoke should stay as a
fallback baseline, not replace the old lane by default.

## Artifacts

- `artifacts/local/dummy-pong-survival-curriculum-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-survival-curriculum-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-survival-curriculum-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-survival-curriculum-scoreboard-smoke-2026-05-09/episodes.jsonl`

## Follow-ups

Do not broaden this trainer until the old-trainer longer-run feasibility call
is made. If this lane resumes, run a modest monitor job with fixed scoreboard
and heldout only after selection. Treat a useful result as a joint signal:
random performance, `track_ball` mean steps, `track_ball` truncation rate,
shaped return, and wins. Do not summarize success only as wins versus
`track_ball`.
