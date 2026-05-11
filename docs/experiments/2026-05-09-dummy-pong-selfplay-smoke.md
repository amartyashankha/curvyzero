# 2026-05-09 Dummy Pong Staged Replay Smoke

Correction note: this is a historical project-owned random/replay trainer
smoke, not the current LightZero core plan and not final multiplayer self-play.
It proved a staged loop shape only. Do not use manual generations or promotion
language as the main plan. The core plan is trusted LightZero whole-job Modal
runs, honest checkpoint curves, scaling actors/steps, then frozen-checkpoint
and later multiplayer self-play.

Supersession note: the later second-pass smoke showed this trainer is not a
source of quality: the child checkpoint lost to the parent and won 0 games
against `track_ball`. Read
`docs/experiments/2026-05-09-dummy-pong-selfplay-gen2-smoke.md` and
`docs/working/pong_training_critique_wave_2026-05-09.md` before treating this
as a next path.

## Question

Can the Pong lane run a first staged loop shape: replay collection, shaped
training return, policy/value update, checkpoint save, and checkpoint
scoreboard?

## Setup

The run used `dummy_pong_v0` with raster observations. The behavior policy was
`random_uniform` for both seats. The training target kept raw score separate
from shaped return:

```text
win:      +1.0
loss:     -1.0 + 0.5 * episode_steps / max_steps
timeout:   0.0
```

## Command

```sh
uv run python scripts/build_dummy_pong_selfplay_replay.py \
  --games 16 \
  --seed 23 \
  --max-steps 80 \
  --policy random_uniform \
  --epsilon 0.0 \
  --output-dir artifacts/local/dummy-pong-selfplay-random-replay-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_selfplay.py \
  --replay-path artifacts/local/dummy-pong-selfplay-random-replay-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001 \
  --seed 0 \
  --epochs 50 \
  --policy-learning-rate 0.1 \
  --value-learning-rate 0.001 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 25
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 331 \
  --split-id dummy_pong_selfplay_smoke_monitor \
  --split-role monitor \
  --checkpoint selfplay25=artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/checkpoints/epoch-000025/checkpoint.npz \
  --checkpoint selfplay50=artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/checkpoints/epoch-000050/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-selfplay-random-scoreboard-smoke-2026-05-09
```

## Results

- Replay wrote 498 rows from 16 games.
- Replay had no truncations; player_0 won 3 games and player_1 won 13.
- Mean shaped return was about `0.084`.
- The first value-learning default `0.05` diverged, so the default was lowered
  to `0.001`.
- With `0.001`, value MSE stayed finite and improved from about `0.80` to
  about `0.43` on all rows.
- Checkpoints loaded in the existing Pong scoreboard.
- `selfplay50` tied `random_uniform` at 8/16.
- `selfplay25` lost slightly to `random_uniform`, 7/16 versus 9/16.
- Neither checkpoint beat `track_ball`.
- `selfplay50` beat `selfplay25` 10/16 in the learned-vs-learned row.

## Interpretation

This proves the correct lane exists. It does not prove policy quality yet.

The policy is still weak and small-seed noisy. The useful change is structural:
we now have a replay format, shaped return fields, a policy/value
checkpoint, and a scoreboard path that can compare old and new checkpoints.

That second pass has now run and failed the important gates. The next useful
step is not more manual generations; it is the LightZero-first whole-job Modal
plan, with survival steps and shaped score reported next to wins.

## Artifacts

- `artifacts/local/dummy-pong-selfplay-random-replay-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-selfplay-random-replay-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/summary.json`
- `artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/checkpoints/epoch-000025/checkpoint.npz`
- `artifacts/local/dummy-pong-selfplay-random-train-smoke-2026-05-09-lr001/checkpoints/epoch-000050/checkpoint.npz`
- `artifacts/local/dummy-pong-selfplay-random-scoreboard-smoke-2026-05-09/summary.json`

## Follow-ups

- Do not add more manual generations by default.
- Keep this as historical evidence, not the core training plan.
- Keep shaped return out of the scoreboard.
