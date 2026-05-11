# 2026-05-09 LightZero Dummy Pong Lag1 Shaped Knob Run

## Question

After exposing the LightZero dummy Pong trainer knobs, does a better-shaped
data-scale sanity run improve survival, shaped score, action diversity, and
checkpoint scorecards?

This is repeated ego play against a chosen opponent, not full two-policy
self-play. The train run below uses LightZero MuZero for `player_0` against the
scripted `lagged_track_ball_1` opponent in `DummyPongLightZeroEnv`.

## Config Surface

The knob-exposure patch makes these train-attempt CLI/config fields available:

```text
n_episode
game_segment_length
random_collect_episode_num
eps_greedy_exploration_in_collect
eps_start
eps_end
eps_decay
fixed_temperature_value
```

After this run started, `policy.action_type` was also made explicit in the
patched config surface as `fixed_action_space`. That is config hygiene and
likely matches the previous LightZero default; this train image was already
running/completed before that explicit field was added to the persisted summary.

No pytest was run.

## Dry Smoke

First command failed at the Modal CLI boolean parser:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode dry --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 4096 --max-train-iter 64 --num-simulations 16 --batch-size 32 --update-per-collect 8 --n-episode 4 --game-segment-length 120 --random-collect-episode-num 8 --eps-greedy-exploration-in-collect true --eps-start 0.3 --eps-end 0.05 --eps-decay 4096 --seed 7 --run-id lz-dpong-knob-dry-s7 --attempt-id dry-args
```

Error:

```text
Got unexpected extra argument (true)
```

Corrected command uses the boolean as a presence flag:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode dry --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 4096 --max-train-iter 64 --num-simulations 16 --batch-size 32 --update-per-collect 8 --n-episode 4 --game-segment-length 120 --random-collect-episode-num 8 --eps-greedy-exploration-in-collect --eps-start 0.3 --eps-end 0.05 --eps-decay 4096 --seed 7 --run-id lz-dpong-knob-dry-s7 --attempt-id dry-args
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-Smk2oTavrjfGAvxaERdAZc`

Result:

```text
ok: true
run_id: lz-dpong-knob-dry-s7
attempt_id: dry-args
summary_ref: training/lightzero-dummy-pong/lz-dpong-knob-dry-s7/attempts/dry-args/train/summary.json
called_train_muzero: false
problems: []
```

## Data-Scale Sanity Train

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 4096 --max-train-iter 64 --num-simulations 16 --batch-size 32 --update-per-collect 8 --n-episode 4 --game-segment-length 120 --random-collect-episode-num 8 --eps-greedy-exploration-in-collect --eps-start 0.3 --eps-end 0.05 --eps-decay 4096 --seed 7 --run-id lz-dpong-lag1-shaped-s7 --attempt-id train-4096x64-sim16
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-OQAscgQCGY2l1pSEZkWYrD`

Run refs:

```text
run_id: lz-dpong-lag1-shaped-s7
attempt_id: train-4096x64-sim16
summary_ref: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/attempts/train-4096x64-sim16/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/attempts/train-4096x64-sim16/train/episodes.jsonl
training_signals_ref: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/attempts/train-4096x64-sim16/train/lightzero_training_signals.json
lightzero_artifacts_ref: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/attempts/train-4096x64-sim16/train/lightzero_artifacts_manifest.json
```

Checkpoint refs:

```text
ckpt_best: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: 3b517be8f7e60190be007fb5517e967984e12dd1392f879f6996d96a91e9d297

iteration_0: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/checkpoints/lightzero/iteration_0.pth.tar
  sha256: 09eb778b2f81ae67a5556f8b40a1fa3f4448a123171da821acabbd851c92b5c1

iteration_64: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/checkpoints/lightzero/iteration_64.pth.tar
  sha256: 2c67af326d06c7c298da016a3511f54cb74f42d764b310e4a46d70fd33f23264
```

Train-side telemetry:

| Metric | Value |
| --- | ---: |
| Env-side episodes | 40 |
| Survival steps mean | 11.85 |
| Survival steps median | 8.0 |
| Survival steps p90 | 19.0 |
| Survival steps max | 30.0 |
| Score return mean | -0.35 |
| Shaped loss-delay return mean | -0.349005 |
| Wins / losses / timeouts | 13 / 27 / 0 |
| Truncation rate | 0.0 |
| Unique seeds | 32 |
| Top seed frequency | 2 / 40 |

Train action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` LightZero ego | 97 | 85 | 292 |
| `player_1` lagged opponent | 184 | 132 | 158 |

Read: the run collected varied actions during training, including substantial
`down` actions from the LightZero-controlled ego. It did not show a positive
train-side result against the chosen scripted opponent: mean shaped score and
mean score return are both negative, with 13 wins and 27 losses.

## Abandoned Large Scorecard

The first independent paired-seat MCTS scorecard was too large for this
debugging loop and was killed/abandoned.

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter64=ref:training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/checkpoints/lightzero/iteration_64.pth.tar --episodes 20 --seed 77 --run-id lz-dpong-lag1-shaped-s7 --attempt-id train-4096x64-sim16 --eval-id mcts-scoreboard-iter64-paired-s77 --max-env-step 4096 --num-simulations 16 --paired-seats
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-Oja5g8fWVnJ49rGAynGC4u`

Result: abandoned after running too long. No scorecard rows from this command
should be used as source of truth for this pass.

## Small Independent MCTS Scorecard

Replacement command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter64=ref:training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/checkpoints/lightzero/iteration_64.pth.tar --episodes 4 --seed 77 --run-id lz-dpong-lag1-shaped-s7 --attempt-id train-4096x64-sim16 --eval-id mcts-scoreboard-iter64-paired-s77-small --max-env-step 4096 --num-simulations 8 --paired-seats
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-nG3BY4ldN4loLg0n0GGsuz`

Artifacts:

```text
eval_dir: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/attempts/train-4096x64-sim16/eval/mcts-scoreboard-iter64-paired-s77-small
summary_json: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/attempts/train-4096x64-sim16/eval/mcts-scoreboard-iter64-paired-s77-small/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/attempts/train-4096x64-sim16/eval/mcts-scoreboard-iter64-paired-s77-small/episodes.jsonl
```

Checkpoint load:

```text
policy_id: lightzero_iter64
adapter: LightZero MCTS eval-mode
num_simulations: 8
load_state_dict: strict=true, ok=true, missing_keys=[], unexpected_keys=[]
strict_full_model_load_variant: res_connection_in_dynamics_true
```

LightZero rows:

| Row | Episodes | Survival mean / median / p90 | LZ shaped mean | LZ score mean | Wins | LZ actions |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `lightzero_iter64_vs_lagged_track_ball_1` | 8 | 10.75 / 8.0 / 19.0 | 0.000656 | 0.0 | LZ 4, lagged 4 | `[0, 0, 86]` |
| `lightzero_iter64_vs_random_uniform` | 8 | 17.625 / 19.0 / 22.3 | -0.248886 | -0.25 | LZ 3, random 5 | `[0, 0, 141]` |
| `lightzero_iter64_vs_track_ball` | 8 | 13.5 / 13.5 / 19.0 | -0.998352 | -1.0 | LZ 0, track 8 | `[0, 0, 108]` |

Baseline context rows:

| Row | Episodes | Survival mean / median / p90 | Shaped mean | Score mean | Wins | Actions |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `random_uniform_vs_lagged_track_ball_1` | 8 | 13.5 / 8.0 / 22.3 | random 0.000992, lagged 0.000656 | random 0.0, lagged 0.0 | random 4, lagged 4 | random `[30,45,33]`, lagged `[41,25,42]` |
| `random_uniform_vs_track_ball` | 8 | 19.0 / 19.0 / 30.0 | random -0.997681, track 1.0 | random -1.0, track 1.0 | random 0, track 8 | random `[41,53,58]`, track `[57,35,60]` |
| `lagged_track_ball_1_vs_track_ball` | 8 | 19.0 / 19.0 / 30.0 | lagged -0.997681, track 1.0 | lagged -1.0, track 1.0 | lagged 0, track 8 | lagged `[60,24,68]`, track `[63,22,67]` |
| `track_ball_vs_track_ball` | 4 | 4096 / 4096 / 4096 | 0.0 | 0.0 | track 0, truncs 4 | track `[90,32570,108]` |

Read: the independent scorecard does not show a robust survival gain. Against
`lagged_track_ball_1`, the checkpoint ties wins and shaped return is basically
zero. Against `random_uniform`, it survives longer on average but loses 3-5 and
has negative shaped return. Against `track_ball`, it loses every game. The
action histogram is also collapsed in eval-mode MCTS: the LightZero checkpoint
chooses only action id 2 in these rows.

## Frozen-Checkpoint Opponent Practicality

Frozen-checkpoint opponents are now technically practical enough for a tiny
smoke. The env and train wrapper support:

```text
--opponent-policy lightzero_policy_head_checkpoint
--opponent-policy lightzero_mcts_checkpoint
--opponent-checkpoint ref:...
--opponent-checkpoint-adapter policy_head_greedy|mcts_eval_mode
--opponent-checkpoint-num-simulations N
```

The small independent scorecard strict-loaded `iteration_64` through the MCTS
adapter, so the checkpoint loading path is not the blocker. The caution is
quality and cost: this checkpoint is weak and eval-mode MCTS collapsed to one
action, while MCTS checkpoint opponents inside collection will be slower than a
scripted opponent. Next step should be a tiny frozen-opponent smoke, not a
large self-play claim.

## Conclusion

The exposed knobs compile and the better-shaped data-scale lane runs end to end
with checkpoints and telemetry. This pass should be read as repeated play
against `lagged_track_ball_1`, not full self-play. The telemetry is mostly
negative: train-side shaped score is below zero, and the small independent MCTS
scorecard shows no robust survival or shaped-score gain. The run is still useful
because it validates the larger config knobs, checkpoint mirroring, strict MCTS
load, paired-seat small eval, and the next frozen-checkpoint-opponent smoke
path.
