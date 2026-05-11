# 2026-05-09 dummy pong checkpoint eval e32

## Question

After the tiny learned-checkpoint eval wiring smoke, what happens on a larger
32-episode Pong eval?

## Setup

- Environment: `dummy_pong_v0`
- Eval seed: 0
- Episodes per seating: 32
- Learned checkpoint:
  `artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz`
- Learned policy id in eval: `learned_dummy_pong_imitation_train_smoke`
- Fixed baselines: `random_uniform`, `track_ball`

## Command

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 32 \
  --seed 0 \
  --checkpoint-policy learned:artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-checkpoint-eval-e32-seed0
```

## Results

- Total episodes: 256.
- `track_ball` versus `random_uniform`: `track_ball` won 64/64, no truncations.
- learned checkpoint versus `random_uniform`: learned won 43/64, no
  truncations.
- learned checkpoint versus `track_ball`: learned won 0/64, `track_ball` won
  6/64, and 58/64 games truncated.
- `track_ball` versus `track_ball`: 32/32 truncations.

## Interpretation

The learned checkpoint is now runnable in the eval harness and beats random
more often than it loses on this seed set. It is still much weaker than the
scripted `track_ball` policy it was trying to copy.

The likely reason is simple: the training replay came from `track_ball` versus
`track_ball`, where all games timed out and the states were narrow. Against
random opponents, the ball visits different states, and the copied policy is
less reliable. This is a data problem before it is an algorithm problem.

This is not reward learning, MuZero, planning, or self-play improvement.

## Artifacts

- `artifacts/local/dummy-pong-checkpoint-eval-e32-seed0/summary.json`
- `artifacts/local/dummy-pong-checkpoint-eval-e32-seed0/episodes.jsonl`

## Follow-ups

- Train from score-bearing `track_ball` versus `random_uniform` replay.
- Add both-policy rows or random-ego rows before relying on the scoring replay
  for negative value targets.
- Keep `track_ball` as the scripted ceiling for the next imitation checkpoint.
