# 2026-05-09 Modal Dummy Pong Train Attempt Smoke

## Question

Can Modal run the current Pong replay plus training code and save the outputs
to the `curvyzero-runs` Volume?

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --games 1 \
  --epochs 1 \
  --seed 0 \
  --max-steps 16 \
  --checkpoint-every-epochs 1
```

## Results

Passed.

- Modal app run: `ap-4jUUDYiI3yk0BLmSkEEMIT`
- Run id: `pong-train-20260509T035946Z-37dd161e5de5`
- Attempt id: `attempt-20260509T035946Z-e495fc4171ab`
- Remote elapsed time: about `0.90s`
- Replay: `1` game, `16` rows, `8` steps, player 0 won.
- Train: `1` epoch, `16` rows, one periodic checkpoint.
- The model predicted `stay` for all `16` rows. That is expected for this tiny
  smoke and is not a quality signal.

Important Volume refs:

```text
training/dummy-pong/pong-train-20260509T035946Z-37dd161e5de5/run.json
training/dummy-pong/pong-train-20260509T035946Z-37dd161e5de5/latest_attempt.json
training/dummy-pong/pong-train-20260509T035946Z-37dd161e5de5/attempts/attempt-20260509T035946Z-e495fc4171ab/replay/summary.json
training/dummy-pong/pong-train-20260509T035946Z-37dd161e5de5/attempts/attempt-20260509T035946Z-e495fc4171ab/train/summary.json
training/dummy-pong/pong-train-20260509T035946Z-37dd161e5de5/attempts/attempt-20260509T035946Z-e495fc4171ab/train/checkpoint.npz
training/dummy-pong/pong-train-20260509T035946Z-37dd161e5de5/checkpoints/latest.json
```

## Interpretation

This proves the remote train wrapper, Volume writes, checkpoint save, and JSON
refs work for Pong. It does not prove the trainer is good.

The next check is to score the saved checkpoint with the existing Modal Pong
scoreboard wrapper.

## Follow-ups

- Run a tiny remote scoreboard against the saved checkpoint.
- If that passes, run the small 64-game Modal repair attempt from
  `docs/working/pong_angle_learning_next_steps_2026-05-09.md`.
