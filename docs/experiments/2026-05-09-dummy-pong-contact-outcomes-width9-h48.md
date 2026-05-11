# 2026-05-09 dummy-pong-contact-outcomes-width9-h48

## Question

Does a narrower width-9 dummy Pong geometry make top, center, and bottom
near-contact choices produce different score-delta returns against
`track_ball`?

## Setup

- Environment: `dummy_pong_v0`
- Config: `width=9`, `height=9`, `paddle_height=3`, `max_steps=48`
- States: 64 controlled near-contact snapshots
- Candidate contacts per state: top `-1`, center `0`, bottom `1`
- Opponent and post-contact policy: `track_ball`
- Seed: 17

## Commands

```sh
uv run python -m py_compile scripts/build_dummy_pong_contact_outcomes.py
```

```sh
uv run python scripts/build_dummy_pong_contact_outcomes.py \
  --states 64 \
  --seed 17 \
  --horizon 48 \
  --width 9 \
  --height 9 \
  --paddle-height 3 \
  --output-dir artifacts/local/dummy-pong-contact-outcomes-width9-h48-2026-05-09
```

## Results

- Candidate rows: 192 from 64 sampled states.
- Contacts: 192/192 candidate rows.
- Missing contacts: 0.
- Return histogram: `{"0.0": 192}`.
- Mean score-delta return by offset: top `0.0`, center `0.0`, bottom `0.0`.
- Mean post-contact score-delta return by offset: top `0.0`, center `0.0`,
  bottom `0.0`.
- Truncations: 64/64 rows for each candidate offset.
- Actual outgoing `ball_vy` still differed perfectly by candidate:
  top `-1` on 64 rows, center `0` on 64 rows, bottom `1` on 64 rows.
- Outcome differences:
  `outgoing_ball_vy_differs_state_count=64`,
  `score_delta_return_differs_state_count=0`,
  `terminal_outcome_differs_state_count=0`.
- Best candidate impact offset histogram: `{"all_tied": 64}`.
- Same-state `track_ball` baseline: 64 rows, 64 contacts, return histogram
  `{"0.0": 64}`, mean score-delta return `0.0`.
- Baseline actual impact offsets: `-1: 26`, `0: 21`, `1: 17`.

## Interpretation

Width 9 did not create score-delta differences between contact choices over a
48-step horizon. It preserved the immediate angle signal, but every candidate
and baseline rollout still ended as a truncation with zero score-delta return.

This narrows the failure mode: the geometry can express top/center/bottom bounce
differences, but width 9 alone is still not harsh enough for one controlled
contact to become a score-bearing label against `track_ball`.

## Artifacts

- `artifacts/local/dummy-pong-contact-outcomes-width9-h48-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-contact-outcomes-width9-h48-2026-05-09/contact_rows.jsonl`
