# 2026-05-09 dummy pong checkpoint eval smoke

## Question

Can dummy Pong eval load the supervised raster imitation checkpoint and compare
it against the fixed baselines on both seats?

## Setup

- Environment: `dummy_pong_v0`
- Eval seed: 0
- Episodes per seating: 2
- Learned checkpoint:
  `artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz`
- Learned policy id in eval: `learned_dummy_pong_imitation_train_smoke`
- Checkpoint schema: `dummy_pong_imitation_policy_checkpoint_v0`
- Learned input path: `env.raster_observation()` plus ego agent
- Fixed baselines: `random_uniform`, `track_ball`

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/run_dummy_pong_eval.py
```

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 2 \
  --seed 0 \
  --checkpoint-policy learned:artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-checkpoint-eval-smoke
```

## Results

- `py_compile` completed.
- Eval completed and wrote `summary.json` plus `episodes.jsonl`.
- Total episodes: 16.
- Loaded policies:
  - `random_uniform`
  - `track_ball`
  - `learned_dummy_pong_imitation_train_smoke`
- `learned_dummy_pong_imitation_train_smoke` vs `track_ball`:
  - Episodes: 4 across both seatings
  - Truncations: 4
  - Mean steps: 120.0
  - Mean reward by policy: both 0.0
  - Wins by policy: none
  - Learned action histogram in each seating: `[30, 181, 29]`
  - Track-ball action histogram in each seating: `[29, 181, 30]`
- `learned_dummy_pong_imitation_train_smoke` vs `random_uniform`:
  - Episodes: 4 across both seatings
  - Truncations: 0
  - Mean steps: 16.25
  - Wins by policy: learned 1, random 3
  - Mean reward by policy: learned -0.5, random 0.5
- Baseline sanity:
  - `track_ball` vs `track_ball`: 2 truncations, mean steps 120.0
  - `random_uniform` vs `track_ball`: track_ball won 4 of 4

## Interpretation

This proves the eval harness can load the supervised imitation checkpoint,
include checkpoint path/schema metadata in policy specs, and call the learned
policy from raster observations plus ego agent during environment rollouts. On
this tiny seed smoke, the learned checkpoint behaves like `track_ball` when
evaluated against `track_ball`: both seatings timed out with zero reward.

This does not prove reward learning, MuZero, planning, self-play improvement,
or a learned winning objective. The checkpoint is a supervised raster clone of
`track_ball`, and the source imitation data had no score reward.

## Artifacts

- `artifacts/local/dummy-pong-checkpoint-eval-smoke/summary.json`
- `artifacts/local/dummy-pong-checkpoint-eval-smoke/episodes.jsonl`

## Follow-ups

- Use a larger heldout seed set before treating learned-vs-random results as
  meaningful.
- Keep reward-learning claims blocked until there is score-bearing replay and a
  reward-trained checkpoint.
