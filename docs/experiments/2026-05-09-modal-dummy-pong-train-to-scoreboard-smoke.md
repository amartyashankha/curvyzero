# 2026-05-09 Modal Dummy Pong Train To Scoreboard Smoke

## Question

Can a checkpoint produced by the Modal Pong train wrapper be loaded and scored
by the Modal Pong scoreboard wrapper using only a Volume ref?

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints smoke=ref:training/dummy-pong/pong-train-20260509T035946Z-37dd161e5de5/attempts/attempt-20260509T035946Z-e495fc4171ab/train/checkpoint.npz \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_pong_modal_train_smoke_monitor \
  --split-role monitor
```

## Results

Passed.

- Modal app run: `ap-QNBuVI2T2cL2VufAKo0DEJ`
- Scoreboard run id: `pong-scoreboard-20260509T040137Z-cde53ae65a0c`
- Attempt id: `attempt-20260509T040137Z-ccbd13de4606`
- Remote elapsed time: about `1.10s`
- Eval summary ref:
  `training/dummy-pong/pong-scoreboard-20260509T040137Z-cde53ae65a0c/attempts/attempt-20260509T040137Z-ccbd13de4606/eval/checkpoint-scoreboard/summary.json`

Scoreboard:

| Row | Result |
| --- | --- |
| learned smoke vs random | 2 wins each over 4 seated games |
| learned smoke vs `track_ball` | learned 0 wins, `track_ball` 3 wins, 1 truncation |
| `track_ball` vs random | `track_ball` 4 wins over 4 seated games |
| `track_ball` vs `track_ball` | 2 truncations |

## Interpretation

The remote train-to-eval path works. A checkpoint can be trained on Modal,
stored in `curvyzero-runs`, loaded from a Volume ref, and scored on Modal.

The checkpoint quality is poor and expected to be poor. This was a one-game,
one-epoch smoke.

## Follow-ups

- Run the small 64-game Modal Pong repair attempt.
- Score its periodic checkpoints on Modal.
- If it still shows no `track_ball` improvement, switch learner or curriculum
  instead of running another generation.
