# 2026-05-09 dummy pong checkpoint scoreboard distinct smoke

## Question

Can the Pong scoreboard compare the distinct learned policy checkpoints we
already have, and does any one of them look meaningfully better against
`track_ball`?

## Setup

- Environment: `dummy_pong_v0`
- Eval split: `dummy_pong_monitor_v0`, role `monitor`
- Episodes per seating: 8
- Seed: 0
- Checkpoints:
  - `imitation_v0`: trained from `track_ball` versus `track_ball` imitation
    replay.
  - `scoring_expert`: trained from score-bearing expert rows against random.
  - `scoring_all_ego`: trained from all-ego scoring rows that include random
    actions.

These are distinct policy attempts, not a clean generation sequence.

## Command

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor \
  --checkpoint imitation_v0=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --checkpoint scoring_expert=artifacts/local/dummy-pong-scoring-imitation-train-smoke-2026-05-09/checkpoint.npz \
  --checkpoint scoring_all_ego=artifacts/local/dummy-pong-scoring-all-ego-imitation-train-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-checkpoint-scoreboard-distinct-smoke-2026-05-09
```

## Results

Baseline sanity:

- `track_ball` beat `random_uniform` 16/16.
- `track_ball` versus `track_ball` truncated 8/8.

Learned versus random:

- `imitation_v0` beat random 10/16.
- `scoring_expert` beat random 10/16.
- `scoring_all_ego` beat random 12/16.

Learned versus `track_ball`:

- `imitation_v0` won 0/16; `track_ball` won 1/16; 15/16 truncated.
- `scoring_expert` won 0/16; `track_ball` won 7/16; 9/16 truncated.
- `scoring_all_ego` won 0/16; `track_ball` won 14/16; 2/16 truncated.

Learned versus learned:

- `imitation_v0` beat `scoring_expert` 7/16 to 5/16, with 4 truncations.
- `scoring_expert` beat `scoring_all_ego` 11/16 to 5/16.
- `scoring_all_ego` edged `imitation_v0` 7/16 to 6/16, with 3 truncations.

## Interpretation

The scoreboard is now doing the right simple job: it shows fixed baselines,
learned-versus-baseline rows, and learned-versus-learned rows in one artifact.

The policy result is not good yet. All learned policies beat random more often
than not, but none beat `track_ball`. The all-ego action clone is especially
bad against `track_ball`, which matches the earlier warning that random-action
rows are poor expert policy targets.

This argues for better policy training data or a real policy-improvement step
before more angle/contact diagnostics.

## Artifacts

- `artifacts/local/dummy-pong-checkpoint-scoreboard-distinct-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-checkpoint-scoreboard-distinct-smoke-2026-05-09/episodes.jsonl`

## Follow-ups

- Make future Pong training attempts save periodic policy checkpoints.
- Use this scoreboard after each attempt.
- Treat `track_ball` as the main current gate.
