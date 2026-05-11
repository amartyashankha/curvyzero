# 2026-05-09 dummy-pong-paddle-angle-smoke

## Question

Does dummy Pong already change the outgoing vertical ball velocity based on
where the ball hits the paddle?

## Setup

- Environment: `dummy_pong_v0`
- Config: default `PongConfig`, especially `paddle_height=3`
- Controlled contact: place the ball one column left of `player_1`'s paddle,
  set horizontal velocity toward that paddle, and keep both paddles still.
- Mini North Star: learn to choose off-center paddle returns to beat
  `track_ball`, not merely track the ball row.

## Command

```sh
uv run python -c $'from curvyzero.training.dummy_pong import PongEnv\n\n\ndef hit(label, hit_y):\n    env = PongEnv()\n    env.reset(seed=0)\n    env._paddle_y = {"player_0": 3, "player_1": 3}\n    env._ball_x = env._paddle_x["player_1"] - 1\n    env._ball_y = hit_y\n    env._ball_vx = 1\n    env._ball_vy = 0\n    step = env.step({"player_0": 1, "player_1": 1})\n    impact = step.infos["last_hit_impact"]\n    print("{}: hit_y={} impact_offset={} outgoing_ball_vy={}".format(label, hit_y, impact["impact_offset"], step.infos["ball"]["vy"]))\n\nfor label, hit_y in (("top", 3), ("center", 4), ("bottom", 5)):\n    hit(label, hit_y)\n'
```

## Results

```text
top: hit_y=3 impact_offset=-1 outgoing_ball_vy=-1
center: hit_y=4 impact_offset=0 outgoing_ball_vy=0
bottom: hit_y=5 impact_offset=1 outgoing_ball_vy=1
```

## Interpretation

The current bounce rule already has paddle-angle mechanics. For the default
three-cell paddle, top/center/bottom contacts produce three different outgoing
vertical velocities: `-1`, `0`, and `1`.

This makes the next strategic target sharper: a learned policy should discover
useful off-center returns that beat the simple `track_ball` baseline, rather
than only imitating `track_ball`'s chase-the-row action.

## Artifacts

- No file artifacts; this was a direct no-pytest smoke.
- Code metadata now names the rule as `dummy_pong_paddle_offset_bounce_v0`.

## Follow-ups

- Keep reward unchanged as score delta only.
- Future eval can track whether learned policies create scoring returns against
  `track_ball`, not just whether they survive rallies.
