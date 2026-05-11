# Two-Player Toy Game Plan

Date: 2026-05-08

Status: implemented as `src/curvyzero/training/dummy_line_duel.py` and
`scripts/run_dummy_line_duel_train.py`; still a dummy scaffold, not real
MuZero or a quality benchmark.

Update: Pong is now the near-term visual training toy because it has a raster
observation path. Tiny Line Duel remains useful later for CurvyTron-like
simultaneous movement and trail collisions.

Goal: pick the smallest post-solo-survival game that exercises simultaneous
two-player rollout, ego-perspective replay, and shared-policy training without
pulling in real CurvyTron fidelity work.

## Options

| Option | Fit | Cost | Verdict |
| --- | --- | --- | --- |
| Pong-like duel | Familiar 1v1, dense ball-control signal, easy scripted opponent. | Adds paddle/ball physics, serving, bounce edge cases, and a different action/reward shape from CurvyTron. | Too much new toy logic. |
| One-dimensional tug/lane duel | Extremely small; good for replay plumbing. | Too abstract: little spatial observation, no trail collision, weak bridge to CurvyTron. | Useful only as a unit-test fixture. |
| Tiny line duel | Reuses solo survival grid, heading, left/straight/right actions, wall/trail crashes, sparse terminal reward, and simultaneous steps. | Needs two-phase collision and tie handling. | Choose this. |

## Chosen Game: Tiny Line Duel

Implement as a new dummy-training task, not as source-fidelity environment
work. It can live beside `dummy_survival.py` and reuse the MuZero-shaped dummy
pieces: `representation`, `dynamics`, `prediction`, planner, replay buffer,
updater, checkpoint, summary artifacts.

### Config

- Grid: `11 x 11`.
- Agents: `player_0`, `player_1`.
- Actions: `0=turn_left`, `1=straight`, `2=turn_right`.
- Max decision steps: `80`.
- Discount: keep solo default, `0.997`.
- No bonuses, no speed, no action repeat, no stochastic events after reset.

### Reset

`reset(seed) -> observations_by_agent`

- Clear occupancy grid.
- Spawn `player_0` at `(2, height // 2)`, heading east.
- Spawn `player_1` at `(width - 3, height // 2)`, heading west.
- Mark both spawn cells as occupied.
- Initialize `step=0`, both alive, no death cause.
- Optional later variety: seed-controlled rotation/reflection of this symmetric
  setup. Do not add it for the first implementation.

### Step

`step(actions_by_agent) -> observations, rewards, terminated, truncated, infos`

- Require one action for each live player. Dead players are no-op only after a
  terminal transition, so the first version should raise if stepped after done.
- Apply turns to both headings before moving.
- Compute both proposed next cells.
- Detect deaths in a two-phase pass before writing new occupancy:
  - out of bounds,
  - collision with existing occupied cell,
  - both players choose the same next cell,
  - players cross-swap cells in the same step.
- Mark only non-dead proposed cells.
- Increment `step`.
- Terminate when zero or one players remain alive.
- Truncate when `max_steps` is reached with both players alive.

### Observation

Use one ego-relative observation object per player. Keep it tabular-friendly:

```text
LineDuelObservation(
  ego_x,
  ego_y,
  ego_heading,
  left_clearance,
  straight_clearance,
  right_clearance,
  opponent_dx_forward,
  opponent_dy_right,
  opponent_heading_relative,
  opponent_alive,
  step,
)
```

Clearances are measured from the ego position after rotating into each candidate
heading, stopping at walls or any occupied cell. Opponent offsets are in ego
coordinates, not absolute board coordinates. Bucket these fields in
`representation` the same way solo survival buckets clearances and board
regions.

### Reward

- Nonterminal: `0.0` for both players.
- Ego wins by being the only survivor: winner `+1.0`, loser `-1.0`.
- Same-step double death: both `0.0`.
- Max-step truncation with both alive: both `0.0`, `truncated=True`,
  `terminated=False`.

This is zero-sum except draws/timeouts, which keeps value targets simple while
exercising multiplayer terminal metadata.

### Replay Shape

Store replay as ego-perspective rows, one row per live ego player per decision
step:

```text
episode_id
seed
step
ego_agent
opponent_agent
observation_key
ego_action
joint_action_by_agent
reward
next_observation_key | None
done
truncated
target_return
opponent_policy_id
ruleset_id = "dummy_line_duel_v0"
observation_schema_id = "line_duel_ego_tabular_v0"
reward_schema_id = "win_loss_draw_v0"
```

For the first trainer, search/control both players with the same dummy model or
use ego search plus a random opponent. In both cases, replay rows stay
ego-perspective: `player_0` and `player_1` each see their own rotated view and
share one policy/value table.

### Training Harness Mapping

- `LineDuelConfig`: mirrors `SurvivalConfig` with `players=2`.
- `LineDuelEnv`: mirrors `SoloTurningSurvivalEnv`, but returns dicts by agent.
- `LineDuelObservation`: dataclass above.
- `LineDuelStep`: dict observations/rewards/done/truncated/info.
- `ReplayTransition`: extend or make a sibling dataclass with `ego_agent`,
  `joint_action`, `truncated`, and schema ids.
- `DummyMuZeroModel`: keep one shared tabular model; `action_count=3`.
- `DummyPlanner`: unchanged at first; it selects one ego action from one ego
  observation.
- Rollout policy:
  - Phase A: `shared_dummy_vs_random`, store only the controlled ego seat.
  - Phase B: `shared_dummy_self_play`, select actions independently for both
    ego views and store both rows.
- Evaluation: run paired seeds with seat swap once reset variety exists; until
  then report wins, losses, draws, truncations, mean steps, and action histogram
  by player.

### Later Multiplayer Path

This toy is intentionally a 1v1 bridge to the ADR-0003 formulation:

- One environment step still consumes a joint action.
- One replay row still represents one ego player.
- The model still predicts only the ego action/value.
- Opponents can later be random, scripted, current shared policy, old
  checkpoints, or league members without changing the replay row shape.
- N-player extension becomes rank payoff plus `opponents[]` metadata, not a new
  trainer architecture.

## Implementation Order

1. Add the env and dataclasses with deterministic reset, simultaneous step, and
   textual episode stats.
2. Add rollout that can run random-vs-random and shared-policy-vs-random.
3. Add ego-perspective replay rows and target returns.
4. Reuse the dummy model/planner/updater with the new observation key.
5. Add a small script matching `scripts/run_dummy_survival_train.py`.
6. Smoke locally with a short command and write summary/checkpoint artifacts.

No pytest is required for this planning step. When implementation starts, the
smallest useful checks are hand-authored episodes for wall death, same-cell
head-head draw, cross-swap draw, and winner reward.
