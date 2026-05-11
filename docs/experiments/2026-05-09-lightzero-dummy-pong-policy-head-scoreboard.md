# 2026-05-09 LightZero Dummy Pong Policy-Head Scoreboard

## Question

Can the tiny LightZero MuZero dummy Pong checkpoint be scored by a
CurvyZero-owned evaluator outside the LightZero training process?

## Source Training Run

- Algorithm: LightZero MuZero.
- Run: `lz-dpong-20260509T141607Z-3696aa333028`.
- Attempt: `attempt-20260509T141607Z-98662e4917b4`.
- Mirrored checkpoints: `ckpt_best.pth.tar`, `iteration_0.pth.tar`,
  `iteration_2.pth.tar`.

The checkpoint probe passed policy-head access, but strict full model load
failed due a dynamics-only key mismatch. Probe ref:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/attempts/attempt-20260509T141607Z-98662e4917b4/probe/lightzero_checkpoint_probe_20260509T143137Z.json
```

## Scoreboard Artifacts

```text
eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/summary.json
eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/episodes.jsonl
```

This scorecard uses direct greedy policy-head inference. It is not MCTS eval
and not proof that the tiny smoke learned a good policy.

## Rows

| Opponent | LightZero wins | Opponent wins | Truncations | Mean score | Shaped mean | Mean steps | p90 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_uniform` | 21/40 | 19/40 | 0 | 0.05 | 0.081875 | 13.775 | 20.1 |
| `lagged_track_ball_1` | 16/40 | 20/40 | 4 | -0.1 | -0.0764583 | 21.95 | 29.1 |
| `track_ball` | 0/40 | 38/40 | 2 | -0.95 | -0.8736458 | 24.325 | 41.0 |

## Read

The important result is that CurvyZero-owned scoring now exists for a
LightZero checkpoint and reports survival plus shaped loss-delay, not only
wins.

The crucial raw-matchup read is that `lightzero_best` is constant-up in every
LightZero row: `[N, 0, 0]`. Do not treat the 21/40 result versus
`random_uniform` as learning proof. The weak rows, constant action histogram,
and strict-load failure mean this is still an eval plumbing milestone, not a
learning-quality milestone.

Telemetry cleanup is now done for later scoreboards: `scoreboard_rows` include
`action_histogram_by_policy`, so constant or broken policies are visible
without opening the raw matchups.

Immediate next lanes:

- Fix strict full-model loading and LightZero MCTS/eval-mode scoring.
- Fix the constant-up policy-head/action-logit issue or prove it is only a
  greedy policy-head limitation.
- Keep Modal Volume refs explicit for training, probe, and eval artifacts.
- Run longer LightZero whole-job experiments only after the eval path is
  honest.

## 512/8 Scoreboard

Source training run:

```text
run_id: lz-dpong-20260509T144635Z-eb5a0ed35de0
attempt_id: attempt-20260509T144635Z-ece79bad80d0
train_summary: training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/train/summary.json
```

Scoreboard artifacts:

```text
eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-20260509T144736Z/summary.json
eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-20260509T144736Z/episodes.jsonl
```

This scored both `ckpt_best` and `iteration_8` with direct greedy policy-head
inference. It is still not MCTS eval.

| Checkpoint | Opponent | LightZero wins | Opponent wins | Truncations | Action histogram `[up, stay, down]` | Shaped mean | Mean steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ckpt_best` | `random_uniform` | 30/64 | 34/64 | 0 | `[864, 0, 0]` | -0.03047 | 13.5 |
| `iteration_8` | `random_uniform` | 31/64 | 33/64 | 0 | `[787, 0, 0]` | -0.00117 | 12.2969 |
| `ckpt_best` | `lagged_track_ball_1` | 21/64 | 35/64 | 8 | `[1595, 0, 0]` | -0.1919 | not recorded here |
| `iteration_8` | `lagged_track_ball_1` | 29/64 | 35/64 | 0 | `[677, 0, 0]` | -0.0691 | not recorded here |
| `ckpt_best` | `track_ball` | 0/64 | 60/64 | 4 | `[1510, 0, 0]` | -0.8704 | not recorded here |
| `iteration_8` | `track_ball` | 0/64 | 53/64 | 11 | `[2217, 0, 0]` | -0.7697 | 34.6406 |

Read:

Both 512/8 checkpoints are still constant-up under the exported/reconstructed
greedy policy head. The trainer-side run reported many env/evaluator wins, but
that is not a policy-quality proof. Do not scale more until MCTS/eval or
loader/inference correctness is fixed.

`scoreboard_rows` now include `action_histogram_by_policy`, so this failure is
visible in compact summaries.

MCTS loader smoke on this run's `iteration_8` failed with wrapper error
`missing cfg.policy.device`:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_mcts_loader_smoke_20260509T144731Z.json
```
