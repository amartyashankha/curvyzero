# 2026-05-09 dummy-pong-eval-smoke

## Question

Can a tiny project-owned Pong-like two-player environment and fixed baseline
eval produce deterministic sanity artifacts?

## Setup

- Environment: `dummy_pong_v0`
- Observations: `pong_ego_tabular_v0`
- Actions: `pong_vertical_actions_v0` with `up`, `stay`, `down`
- Rewards: `win_loss_score_v0`
- Baselines: `random_uniform`, `track_ball`
- Implementation provenance: original simple project implementation; no
  external Pong code was copied or adapted.

## Command

```sh
PYTHONPATH=src python scripts/run_dummy_pong_eval.py --episodes 8 --seed 123 --output-dir artifacts/local/dummy-pong-smoke
```

## Results

- `random_uniform` vs `random_uniform`: 8 episodes, 4 wins per seat, mean 16.25
  steps, 0 truncations.
- `track_ball` vs `random_uniform`, paired seats: `track_ball` won 16/16
  episodes, mean reward 1.0, 0 truncations.
- `track_ball` vs `track_ball`: 8/8 truncations at the 120-step cap.

## Interpretation

The environment/eval contract is working as a tiny scaffold: random games score,
the simple tracking heuristic strongly beats random from both seats, and a
symmetrical heuristic matchup exposes time-limit behavior. This is a sanity
baseline only, not a trainer or a claim about learning.

## Artifacts

- `artifacts/local/dummy-pong-smoke/summary.json`
- `artifacts/local/dummy-pong-smoke/episodes.jsonl`

## Follow-ups

- Add golden trace tests when this lane graduates from scaffold to shared
  contract.
- Consider a slightly imperfect scripted opponent if self-play truncations make
  later eval tables too flat.
