# 2026-05-09 dummy pong target ladder probe

## Question

After the exact beatability probe showed default deterministic `track_ball` is
unwinnable from normal resets, what is the smallest winnable Pong target that
still creates meaningful score pressure?

## Setup

- Probe script: `scripts/probe_dummy_pong_target_ladder.py`
- Search: exact dynamic programming over legal ego actions
  (`up`, `stay`, `down`) to the 120-step cap.
- Transition model: same pure transition model used by
  `scripts/probe_dummy_pong_track_ball_beatable.py`.
- Control target: default
  `PongConfig(width=15,height=9,paddle_height=3,max_steps=120)` with
  deterministic `track_ball`.
- Compact sweep:
  - one-step and two-step lagged `track_ball`;
  - track every 2 or 3 steps, otherwise stay;
  - symmetric paddle heights 2 and 1;
  - width 9 and height 11 geometry tweaks;
  - default `track_ball` from near-opponent-contact starts at distances 2 and 1.

## Command

```sh
uv run python -m py_compile scripts/probe_dummy_pong_target_ladder.py
```

```sh
uv run python scripts/probe_dummy_pong_target_ladder.py \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-target-ladder-probe-2026-05-09
```

## Results

Compile passed. The ladder probe found that one-step lagged `track_ball` is
scoreable from every normal reset/player case while keeping the default
geometry.

| Target | Reset mode | Cases scoreable | Median win steps | Read |
| --- | --- | ---: | ---: | --- |
| `default_track_ball_normal` | normal | 0/40 | n/a | control remains unwinnable |
| `lag1_track_ball_normal` | normal | 40/40 | 19.0 | best ladder target |
| `lag2_track_ball_normal` | normal | 40/40 | 19.0 | also winnable, less minimal |
| `every2_track_ball_normal` | normal | 40/40 | 19.0 | winnable but less like normal tracking |
| `every3_track_ball_normal` | normal | 40/40 | 19.0 | winnable but weaker target |
| `paddle2_track_ball_normal` | normal | 38/40 | 107.0 | promising geometry, not complete |
| `paddle1_track_ball_normal` | normal | 20/40 | 8.0 | too sparse and too geometry-specific |
| `width9_track_ball_normal` | normal | 0/40 | n/a | width alone still not useful |
| `height11_track_ball_normal` | normal | 0/56 | n/a | height alone still not useful |
| `default_track_ball_near_contact_d2` | biased | 4/20 | 4.0 | diagnostic starts only |
| `default_track_ball_near_contact_d1` | biased | 4/20 | 3.0 | diagnostic starts only |

Shortest lag-1 trace:

- target: `lag1_track_ball_normal`
- ego: `player_0`
- reset: `reset-02-y2-vx+1-vy-1`
- initial state:
  `player_0_y=3, player_1_y=3, ball_x=7, ball_y=2, ball_vx=1, ball_vy=-1`
- actions: eight `up` actions
- score: `player_0` wins at step 8 with `ball_x=15`

The lag-1 target touched only 4,290 memoized states across the 40 cases, versus
174,476 for the default unwinnable control. That is expected: once the target is
scoreable, the search finds short traces instead of exhausting tie loops.

## Interpretation

Use `lag1_track_ball_normal` as the next Pong target ladder rung. It is better
than default `track_ball` because it keeps the same board, paddle height, action
space, reset support, and visual observation shape, but turns the target from a
survival-only tie floor into a scoreable opponent with exact traces in every
normal reset/player case.

The target still teaches meaningful score pressure: the ego must create or
exploit the opponent's one-step tracking error to score, instead of merely
surviving. It is also easier to encode and evaluate than asymmetric paddles or
special reset distributions: add a deterministic `lagged_track_ball_1` baseline
policy, then run the same scoreboard shape on default geometry.

Do not promote the near-contact starts as the main target. They produce short
scores in only 4/20 biased cases and are better as a diagnostic pressure slice.

## Artifacts

- `artifacts/local/dummy-pong-target-ladder-probe-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-target-ladder-probe-2026-05-09/rows.jsonl`

## Follow-ups

Done for the immediate learner path: `lagged_track_ball_1` is now available to
CEM and the checkpoint scoreboard while preserving default `track_ball` as the
full-survival/tie floor. The first learner check should report wins against
`lagged_track_ball_1`, survival against default `track_ball`, and random-opponent
wins so the new target does not hide a regression.
