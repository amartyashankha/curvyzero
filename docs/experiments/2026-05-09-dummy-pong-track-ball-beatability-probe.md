# 2026-05-09 dummy pong track_ball beatability probe

## Question

Under the current default geometry
`PongConfig(width=15,height=9,paddle_height=3,max_steps=120)`, can any policy
score against deterministic `track_ball` from normal resets?

This is the key check after geometry-CEM reached full survival/tie with no
score pressure.

## Setup

- Probe script: `scripts/probe_dummy_pong_track_ball_beatable.py`
- Opponent: fixed deterministic `track_ball`
- Ego control: exact dynamic programming over all legal ego actions
  (`up`, `stay`, `down`)
- Reset support: all 20 states produced by normal resets:
  - both paddles centered at top row `3`
  - `ball_x=7`
  - `ball_y in {2,3,4,5,6}`
  - `ball_vx in {-1,1}`
  - `ball_vy in {-1,1}`
- Player seats checked: both `player_0` and `player_1` as ego for every reset
  state, for 40 reset/player cases.
- Bound: exact search to the 120-step episode cap. The loose finite-state bound
  for this config is 4,802,490 `(paddles, ball, velocity, step)` states; the
  reachable memoized states touched across the 40 reset/player cases totaled
  174,476, with 3,107 to 10,086 states per case.
- Transition parity check: the script compared its pure transition model
  against `PongEnv.step` for all reset-support states, both ego seats, and all
  ego actions. It passed 120/120 single-step cases.

## Command

```sh
uv run python -m py_compile scripts/probe_dummy_pong_track_ball_beatable.py
```

```sh
uv run python scripts/probe_dummy_pong_track_ball_beatable.py \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-track-ball-beatable-probe-2026-05-09
```

## Results

Compile passed. The exact bounded search found no winning trace.

| Metric | Result |
| --- | ---: |
| Normal reset states | 20 |
| Reset/player cases | 40 |
| Cases where ego can score | 0 |
| Winning traces | 0 |
| Max depth | 120 steps |
| Reachable memo states, total over cases | 174,476 |
| Reachable memo states, min per case | 3,107 |
| Reachable memo states, max per case | 10,086 |
| Transition parity cases | 120/120 passed |

Plain answer: no, within the full default episode bound, no legal policy can
score against `track_ball` from the current normal reset support.

## Interpretation

`track_ball` is not merely a hard baseline in this toy geometry; it is a bad
hard win baseline. The best possible result against it from normal resets is a
full-length tie. That explains why geometry-CEM could reach perfect survival
without score pressure: there is no score pressure to discover under this
target.

The current scoreboard can keep `track_ball` as a survival/stability baseline,
but it should not be the hard win gate for this default toy. A learner that
fails to beat `track_ball` in this geometry may be optimal with respect to the
score outcome.

## Artifacts

- `artifacts/local/dummy-pong-track-ball-beatable-probe-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-track-ball-beatable-probe-2026-05-09/rows.jsonl`

## Follow-ups

Change the Pong target before asking a learner to beat `track_ball`: use a
scoreable opponent, biased/near-contact starts, or a geometry change such as
smaller paddles, different width/height, faster ball, or a different bounce
rule. Keep the current `track_ball` row only as a full-survival/tie floor.
