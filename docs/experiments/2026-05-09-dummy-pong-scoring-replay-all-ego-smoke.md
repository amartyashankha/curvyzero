# 2026-05-09 dummy-pong-scoring-replay-all-ego-smoke

## Question

Can scoring replay include both ego policies so `random_uniform` losing to
`track_ball` creates negative terminal reward rows?

This is still scripted scoring replay. It is not self-play, MuZero, or a learned
policy.

## Setup

- Matchups: `track_ball` vs `random_uniform` in both seats.
- Row policy: `all`, emitting both `track_ball` and `random_uniform` ego rows.
- Reward: score-delta-only rewards from `PongEnv`.
- Max steps: 120.
- Games: 4 per seat, 8 total.

All-ego scoring replay is better for value targets than track-ball-only rows
because it includes both positive and negative terminal reward rows from the
ego perspective.

## Commands

```sh
uv run python scripts/build_dummy_pong_scoring_replay.py \
  --games-per-seat 4 \
  --seed 0 \
  --max-steps 120 \
  --row-policy all \
  --output-dir artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09
```

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09 \
  --sample-frames 0
```

```sh
uv run python -c 'import json; from pathlib import Path; p=Path("artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09/replay_rows.jsonl"); rows=[json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]; print(json.dumps({"rows": len(rows), "positive_reward_rows": sum(float(r["reward_after_step"]) > 0.0 for r in rows), "negative_reward_rows": sum(float(r["reward_after_step"]) < 0.0 for r in rows), "nonzero_reward_rows": sum(float(r["reward_after_step"]) != 0.0 for r in rows), "terminated_rows": sum(bool(r["terminated"]) for r in rows), "truncated_rows": sum(bool(r["truncated"]) for r in rows), "ego_agents": sorted(set(r["ego_agent"] for r in rows)), "behavior_policy_ids": sorted(set(r["behavior_policy_id"] for r in rows)), "reward_values": sorted(set(float(r["reward_after_step"]) for r in rows))}, sort_keys=True))'
```

## Results

- `replay_rows.jsonl` rows: 392.
- Environment steps: 196.
- Nonzero reward rows: 16.
- Positive reward rows: 8.
- Negative reward rows: 8.
- Terminated rows: 16.
- Truncated rows: 0.
- Reward values in emitted rows: `-1.0`, `0.0`, `1.0`.
- Ego agents present: `player_0`, `player_1`.
- Behavior policy IDs present on emitted rows: `random_uniform`, `track_ball`.
- Raster shape observed by inspector: `9x15` for all 392 rows.

`track_ball` won all 8 games:

- `track_ball_p0__random_uniform_p1`: 4 wins for `player_0`, 0 truncations,
  mean 16.25 steps.
- `random_uniform_p0__track_ball_p1`: 4 wins for `player_1`, 0 truncations,
  mean 32.75 steps.

Inspector note: it reported replay reward totals of `0.0` by ego agent because
the positive and negative terminal rewards cancel across seats. The direct row
count above confirms nonzero positive and negative reward rows are present.

## Artifacts

- `artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09/replay_rows.jsonl`

## Next Recommendation

Use `--row-policy all` for the first value-target smoke so terminal labels
include both wins and losses. Keep policy/eval work separate, and do not treat
this artifact as self-play or MuZero evidence.
