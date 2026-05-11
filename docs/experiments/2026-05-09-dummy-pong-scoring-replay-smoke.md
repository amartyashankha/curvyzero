# 2026-05-09 dummy-pong-scoring-replay-smoke

## Question

Can dummy Pong produce learner-ready replay rows with nonzero score-delta
reward after `track_ball` versus `track_ball` produced only zero-reward
truncations?

This creates scoring replay data. It still does not implement MuZero,
self-play, or a learned policy.

## Setup

- Matchups: `track_ball` vs `random_uniform` in both seats.
- Row filter: emit rows only for the `track_ball`-controlled ego.
- Reward: score-delta-only rewards already produced by `PongEnv`.
- Max steps: 120.
- Games: 4 per seat, 8 total.

Each replay row preserves:

- `raster_grid`
- `ego_agent`
- `behavior_policy_id`
- `behavior_action_id` / `behavior_action_label`
- `target_action_id` / `target_action_label`
- `joint_action_by_agent`
- `reward_after_step`
- `next_raster_grid`
- `terminated` / `truncated`

## Commands

```sh
uv run python scripts/build_dummy_pong_scoring_replay.py \
  --games-per-seat 4 \
  --seed 0 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-scoring-replay-smoke-2026-05-09
```

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-scoring-replay-smoke-2026-05-09 \
  --sample-frames 0
```

```sh
uv run python -c 'import json; from pathlib import Path; p=Path("artifacts/local/dummy-pong-scoring-replay-smoke-2026-05-09/replay_rows.jsonl"); rows=[json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]; print(json.dumps({"rows": len(rows), "nonzero_reward_rows": sum(float(r["reward_after_step"]) != 0.0 for r in rows), "terminated_rows": sum(bool(r["terminated"]) for r in rows), "truncated_rows": sum(bool(r["truncated"]) for r in rows), "ego_agents": sorted(set(r["ego_agent"] for r in rows)), "behavior_policy_ids": sorted(set(r["behavior_policy_id"] for r in rows)), "reward_values": sorted(set(float(r["reward_after_step"]) for r in rows))}, sort_keys=True))'
```

## Results

- `replay_rows.jsonl` rows: 196.
- Nonzero reward rows: 8.
- Terminated rows: 8.
- Truncated rows: 0.
- Reward values in emitted rows: `0.0`, `1.0`.
- Ego agents present: `player_0`, `player_1`.
- Behavior policy IDs present on emitted rows: `track_ball`.
- Raster shape observed by inspector: `9x15` for all 196 rows.
- Inspector quality note: no obvious count or raster-shape problems detected.

`track_ball` won all 8 games:

- `track_ball_p0__random_uniform_p1`: 4 wins for `player_0`, 0 truncations,
  mean 16.25 steps.
- `random_uniform_p0__track_ball_p1`: 4 wins for `player_1`, 0 truncations,
  mean 32.75 steps.

## Interpretation

This is the smallest clean reward-learning data path for the coach lane:
score-delta reward is nonzero, the visual observation path is present, and the
opponent action remains available in joint metadata. Because rows are emitted
only for `track_ball`, the reward rows are all positive in this smoke. This
gives score events, but it does not give losing ego examples or negative value
targets from the emitted-row perspective.

This artifact is best used for a first policy/value smoke around winning
`track_ball` behavior. A v1 scoring replay should include both ego policies, or
add an explicit option to include `random_uniform` ego rows, if the next learner
needs negative value targets.

## Artifacts

- `artifacts/local/dummy-pong-scoring-replay-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-replay-smoke-2026-05-09/replay_rows.jsonl`

## Follow-ups

- Feed this replay into the first policy/value smoke for winning `track_ball`
  behavior.
- Add both-policy ego rows, or an explicit random-ego inclusion option, before
  relying on this path for negative value targets.
- Keep eval separate and outcome-based.
- Do not add shaped reward until score-delta-only learning has a concrete
  failure mode.
