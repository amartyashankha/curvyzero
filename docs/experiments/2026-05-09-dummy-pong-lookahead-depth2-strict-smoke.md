# 2026-05-09 dummy pong lookahead depth-2 strict smoke

## Question

Does two-step ego action-sequence lookahead create useful strict Pong labels
against fixed `track_ball` on the default geometry?

## Setup

- Replay output:
  `artifacts/local/dummy-pong-lookahead-depth2-strict-smoke-2026-05-09`
- Collector: random ego versus fixed `track_ball`.
- Labeling: evaluate all 9 two-action ego sequences while the opponent uses
  `track_ball` on both forced steps; then roll out both agents with
  `track_ball` through the 32-step horizon.
- Ties: strict replay, so all-tied sampled states are filtered.
- No pytest. No training.

## Commands

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_lookahead_replay.py \
  scripts/build_dummy_pong_lookahead_replay.py
```

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 2 \
  --seed 19 \
  --max-steps 80 \
  --lookahead-steps 32 \
  --ego-sequence-depth 2 \
  --collector-policy random_uniform \
  --output-dir artifacts/local/dummy-pong-lookahead-depth2-strict-smoke-2026-05-09
```

## Results

Replay:

- Rows: 10 emitted from 65 sampled states.
- Filtered all-tied states: 55.
- Return spread: all 10 emitted rows had spread `1.0`.
- Target source counts:
  - best sequence return tie broken by `track_ball`: 8.
  - unique best sequence return: 2.
- Target returns: 10 zero, 0 positive, 0 negative.
- Target score-delta returns: 10 zero, 0 positive, 0 negative.
- Targets different from collector action: 7.
- Targets different from `track_ball`: 0.
- Candidate sequence returns were score-bearing in the avoided-loss sense:
  each sequence had negative returns on 5 or 6 of the 10 emitted rows, but the
  chosen sequence targets were all zero-return survival labels.
- Chosen sequence histogram included `down/up`, `stay/up`, `stay/down`,
  `down/down`, and `up/up`.

## Interpretation

Depth-2 search is wired correctly and exposes candidate sequence return tables
in both replay rows and the summary. The strict smoke did produce non-tied
sequence labels: the emitted rows separated zero-return survival sequences from
losing sequences.

This is not enough to train. The target first action matched `track_ball` on all
10 emitted rows, and there were no positive-return targets. Training would mostly
teach another `track_ball` copy rather than a scoring policy.

## Artifacts

- `artifacts/local/dummy-pong-lookahead-depth2-strict-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-depth2-strict-smoke-2026-05-09/replay_rows.jsonl`

## Next

- Run a larger depth-2 replay only if the goal is to estimate how often avoided
  losses become non-`track_ball` labels.
- For scoring labels, default geometry still looks weak; the smaller-width
  geometry probe remains the more direct next experiment.
