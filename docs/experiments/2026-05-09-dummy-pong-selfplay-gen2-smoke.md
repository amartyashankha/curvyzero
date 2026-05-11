# 2026-05-09 Dummy Pong Staged Replay Pass 2 Smoke

Correction note: this is a historical project-owned staged replay experiment,
not the current LightZero core plan and not final multiplayer self-play. Do not
make manual generations or promotions the plan. The current plan is trusted
LightZero whole-job Modal runs, honest checkpoint curves, actor/step scaling,
then frozen-checkpoint and later multiplayer self-play.

## Question

Does one more staged replay/update pass improve the first Pong checkpoint?

## Setup

This pass collected games from the previous epoch-50 checkpoint with epsilon
exploration. Training initialized from the same previous checkpoint.

## Command

```sh
uv run python scripts/build_dummy_pong_selfplay_replay.py \
  --games 32 \
  --seed 41 \
  --max-steps 80 \
  --policy learned:artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/checkpoints/epoch-000050/checkpoint.npz \
  --epsilon 0.1 \
  --output-dir artifacts/local/dummy-pong-selfplay-gen2-replay-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_selfplay.py \
  --replay-path artifacts/local/dummy-pong-selfplay-gen2-replay-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-selfplay-gen2-train-smoke-2026-05-09 \
  --seed 1 \
  --epochs 75 \
  --policy-learning-rate 0.05 \
  --value-learning-rate 0.001 \
  --validation-fraction 0.2 \
  --initial-checkpoint artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/checkpoints/epoch-000050/checkpoint.npz \
  --checkpoint-every-epochs 25
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 16 \
  --seed 431 \
  --split-id dummy_pong_selfplay_gen2_monitor \
  --split-role monitor \
  --checkpoint gen1_50=artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/checkpoints/epoch-000050/checkpoint.npz \
  --checkpoint gen2_25=artifacts/local/dummy-pong-selfplay-gen2-train-smoke-2026-05-09/checkpoints/epoch-000025/checkpoint.npz \
  --checkpoint gen2_50=artifacts/local/dummy-pong-selfplay-gen2-train-smoke-2026-05-09/checkpoints/epoch-000050/checkpoint.npz \
  --checkpoint gen2_75=artifacts/local/dummy-pong-selfplay-gen2-train-smoke-2026-05-09/checkpoints/epoch-000075/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-selfplay-gen2-scoreboard-smoke-2026-05-09
```

## Results

- The second-pass replay wrote 1,392 rows from 32 games.
- Replay had no truncations; player_0 won 15 and player_1 won 17.
- Mean shaped return was about `0.104`, up from about `0.084` in the first
  pass.
- The trainer wrote epoch 25, 50, and 75 checkpoints.
- No second-pass checkpoint beat `track_ball`.
- `gen2_50` beat `random_uniform` 20/32, which is better than `gen1_50` at
  19/32 on this monitor split.
- Direct old-vs-new rows rejected the child checkpoint:
  - `gen1_50` beat `gen2_25` 20/32.
  - `gen1_50` beat `gen2_50` 20/32.
  - `gen1_50` beat `gen2_75` 17/32.
- The second-pass policy predictions became narrow: the final checkpoint used
  no `down` actions in the training summary's predicted-action histogram.

## Interpretation

The staged loop works, but the update is not stable enough. The second pass
shows a tiny random-baseline bump for epoch 50, but it regresses against the
parent checkpoint and still cannot beat `track_ball`.

Do not accept the second-pass checkpoint. Do not scale this objective blindly.

The next useful work is the critique decision: repair this crude trainer or
switch to a known simple baseline/curriculum. If this trainer is repaired, start
with the simple aliasing bug: `policy_grad = probs` likely should be
`policy_grad = probs.copy()` when later code mutates the gradient.

If repaired, the trainer needs:

- keep an entropy or action-diversity term so the policy does not collapse;
- compare against the parent checkpoint and fixed baselines before any child is
  a candidate;
- normalize advantages per ego/seat or per game so seat quirks do not dominate;
- emit iteration metrics, action histograms by seat, entropy/collapse metrics,
  terminal causes, and failure examples.

## Artifacts

- `artifacts/local/dummy-pong-selfplay-gen2-replay-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-selfplay-gen2-train-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-selfplay-gen2-scoreboard-smoke-2026-05-09/summary.json`

## Follow-ups

- Keep this as historical evidence, not the core plan.
- If this old trainer is repaired later, add entropy/action-diversity visibility
  and parent-plus-baseline candidate gates before Modal scaling.
- Keep `track_ball` as the main hard baseline.
