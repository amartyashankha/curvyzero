# 2026-05-09 dummy-pong-contact-outcomes-smoke

## Question

Can a tiny local probe create near-contact Pong rows that compare top, center,
and bottom ego contacts against `track_ball` over a short score-delta horizon?

## Setup

- Environment: `dummy_pong_v0`
- Rows: controlled near-contact snapshots, three candidate ego contacts per
  state: top `-1`, center `0`, and bottom `1`
- Opponent and post-contact policy: `track_ball`
- Horizon: 24 steps
- State control: this first probe writes private `PongEnv` fields to place the
  ball and candidate paddle before the first real step.
- Audit note: off-center rate alone is weak, because pure `track_ball` can also
  make off-center hits from lag. The useful row fields are predicted hit row,
  desired/actual impact offset, reachable target center, outgoing `ball_vy`,
  and short post-contact score delta.

## Command

```sh
uv run python -m py_compile scripts/build_dummy_pong_contact_outcomes.py
```

```sh
uv run python scripts/build_dummy_pong_contact_outcomes.py \
  --states 4 \
  --seed 0 \
  --horizon 24 \
  --output-dir artifacts/local/dummy-pong-contact-outcomes-smoke
```

## Results

- Wrote `summary.json` and `contact_rows.jsonl`.
- Candidate rows: 12 from 4 sampled states.
- Contacts: 12/12 candidate rows.
- Actual outgoing `ball_vy` differed by candidate on every sampled state:
  top `-1`, center `0`, bottom `1`.
- Score-delta returns did not differ: every candidate return was `0.0`.
- Terminal outcomes did not differ: every candidate row truncated at 24 steps.
- Same-state pure `track_ball` baseline also returned `0.0` on all 4 states,
  and made off-center contacts on 3/4 states.

## Interpretation

The probe now creates inspectable contact-outcome rows, but the first short
score-delta signal is flat. Contact choices changed the immediate bounce angle,
not the score outcome, against `track_ball` in this default geometry.

If this persists on larger samples, the default Pong geometry is probably too
forgiving for a one-contact score-delta label. The next geometry knobs are
smaller width, smaller paddle, or faster ball.

## Artifacts

- `artifacts/local/dummy-pong-contact-outcomes-smoke/summary.json`
- `artifacts/local/dummy-pong-contact-outcomes-smoke/contact_rows.jsonl`

## Follow-ups

- Try a slightly harsher geometry before fitting a chooser.
- Keep comparing against same-state `track_ball`, not just off-center contact
  counts.

## Implementation Follow-up

The smallest score-pressure implementation is now an opt-in custom dummy Pong
reset curriculum: `pong_reset_profile=contact_pressure`. It starts episodes
near paddle-contact states and records reset metadata in env info, but
`env.step()` still returns only sparse score rewards: `+1`, `-1`, or `0`. The
first tiny LightZero MuZero train plus matching MCTS scorecard passed
mechanically but did not produce a quality checkpoint. See
`docs/experiments/2026-05-09-lightzero-dummy-pong-contact-pressure-curriculum-smoke.md`.
