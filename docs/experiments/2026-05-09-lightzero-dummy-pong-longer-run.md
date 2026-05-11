# 2026-05-09 LightZero Dummy Pong Longer Run

Worker C lane: cheap whole-job scaling for LightZero MuZero on the custom dummy
Pong wrapper.

## Cap Finding

`src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py` was a true
tiny smoke wrapper: it rejected `max_env_step > 64` and `max_train_iter > 2`.
To keep that CLI tiny, I added:

- internal helper caps in `lightzero_dummy_pong_tiny_train_smoke.py`, defaulting
  to the original tiny limits;
- a separate scaled wrapper:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`.

The scaled wrapper reuses the tiny smoke's run-management, trainer invocation,
artifact scan, checkpoint mirror, and env-side scorecard plumbing. It sets
conservative explicit caps of `max_env_step <= 1024` and
`max_train_iter <= 16`.

## Attempt 0: Failed Before Training Loop

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy random_uniform --max-env-step 512 --max-train-iter 8 --num-simulations 4 --batch-size 16 --n-evaluator-episode 4 --seed 0
```

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-vF3ziuioUoSufcuLtYriTj
```

Run refs:

```text
run_id: lz-dpong-20260509T144454Z-50d82017ab95
attempt_id: attempt-20260509T144454Z-090d1ec9a8ce
summary_ref: training/lightzero-dummy-pong/lz-dpong-20260509T144454Z-50d82017ab95/attempts/attempt-20260509T144454Z-090d1ec9a8ce/train/summary.json
```

Result:

```text
ok: false
called_train_muzero: true
problem: LightZero train_muzero failed: KeyError: 'timestep'
checkpoints: none
```

The failure was in `lzero/worker/muzero_evaluator.py`, which warned that
`init_obs[0]["timestep"]` was missing and then indexed it directly. I kept the
fix local to the scaled wrapper by installing a compatibility patch that adds a
`timestep` key to dummy Pong observations before calling the reused trainer
helper.

Env-side telemetry before the crash:

```text
episodes: 1
wins: 1
losses: 0
survival_steps.mean: 8.0
score_return.mean: 1.0
shaped_loss_delay_return.mean: 1.0
truncation_rate: 0.0
```

## Attempt 1: Completed

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --seed 1 --opponent-policy random_uniform --max-env-step 512 --max-train-iter 8 --num-simulations 4 --batch-size 16 --update-per-collect 1 --n-evaluator-episode 4
```

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-qbOR8K3WUDR2yjtme5VGBG
```

Run refs:

```text
run_id: lz-dpong-20260509T144618Z-8aca41c726e9
attempt_id: attempt-20260509T144618Z-90daf215d879
summary_ref: training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/attempts/attempt-20260509T144618Z-90daf215d879/train/summary.json
attempt_manifest: training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/attempts/attempt-20260509T144618Z-90daf215d879/attempt.json
latest_attempt: training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/latest_attempt.json
```

Config:

```text
env: dummy_pong_lag1
feature_mode: tabular_ego
seed: 1
opponent_policy: random_uniform
max_env_step: 512
max_train_iter: 8
num_simulations: 4
batch_size: 16
update_per_collect: 1
n_evaluator_episode: 4
collector_env_num: 1
evaluator_env_num: 1
cuda: false
```

Top-level result:

```text
ok: true
called_train_muzero: true
problems: []
elapsed_sec: 12.31995
checkpoint_iterations: [0, 8]
max_checkpoint_iteration: 8
```

Mirrored checkpoint refs:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/checkpoints/lightzero/ckpt_best.pth.tar
  bytes: 27829507
  sha256: c809978f574583ceaaed6c2c8381b6ab450dccbf750ce34f6ef0c44ec474bfe2

training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/checkpoints/lightzero/iteration_0.pth.tar
  bytes: 55565027
  sha256: 78402cb44a466e92fd7c0f510509c6270ee0289ff804955fe25fba2af80a5151

training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/checkpoints/lightzero/iteration_8.pth.tar
  bytes: 55565027
  sha256: c67d9835ac9adfcef6b02ff919ba7abbd6dba40cafb59fa2c76f267d2148b8fd

training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/checkpoints/lightzero/manifest.json
  sha256: 6598aa236f866d4b1c11b324ef4a4b1f478254fb15cc033ad1efb617bc927787
```

Artifact refs:

```text
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/attempts/attempt-20260509T144618Z-90daf215d879/train/episodes.jsonl
  sha256: e241a7bc3d5fc6c9a312d85c0408e4db246abf3eabd74aa2fd68235556e38e5e

training_signals: training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/attempts/attempt-20260509T144618Z-90daf215d879/train/lightzero_training_signals.json
  sha256: 24bfcf13916de398f67eb9d80225721eadde0649e1f34863bb0151bafff0c496

lightzero_artifacts: training/lightzero-dummy-pong/lz-dpong-20260509T144618Z-8aca41c726e9/attempts/attempt-20260509T144618Z-90daf215d879/train/lightzero_artifacts_manifest.json
  sha256: 5af49383e19ee53b4569f01d1e5c999ac9faf014b3c91a919ad9346b1d13cd63
```

Env-side scorecard:

```text
episodes: 42
wins: 37
losses: 5
timeouts: 0
truncation_rate: 0.0

survival_steps.count: 42
survival_steps.mean: 9.047619047619047
survival_steps.median: 8.0
survival_steps.p90: 8.0
survival_steps.min: 8.0
survival_steps.max: 30.0
survival_steps.std: 4.023457296502604

score_return.mean: 0.7619047619047619
score_return.median: 1.0
score_return.min: -1.0
score_return.max: 1.0
score_return.std: 0.6476890718445448

shaped_loss_delay_return.mean: 0.7633463541666666
shaped_loss_delay_return.median: 1.0
shaped_loss_delay_return.min: -0.9921875
shaped_loss_delay_return.max: 1.0
shaped_loss_delay_return.std: 0.6437700776102411

player_0 action_counts: up=115, stay=208, down=57
player_1 action_counts: up=138, stay=103, down=139
```

LightZero evaluator signal:

```text
final_rewards: 32 values, all 1.0
last evaluator table: train_iter 7, episode_count 4, envstep_count 32
last eval_episode_return: [1.0, 1.0, 1.0, 1.0]
```

## Scorecard Readiness

This run is ready as a checkpoint/artifact source for an independent scorecard:
the run has `ckpt_best`, `iteration_0`, `iteration_8`, env-side episode rows,
and a summary manifest. The independent CurvyZero checkpoint scorecard itself
was not run in this lane; the existing summary still marks it blocked until the
LightZero checkpoint inference/loader path is wired for standalone scoring.

## 2026-05-09 Corrected Existing 512/8 MCTS Scorecard

Before launching a longer train, I reran the existing 512/8 `iteration_8`
checkpoint through the patched MCTS scorecard path. The config bug worker's
source fixes were present: persistent per-episode `random_action()` RNG,
LightZero observations include `timestep`, and LightZero checkpoint scorecards
use `PongConfig(max_steps=lightzero_max_env_step)`.

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter8=ref:training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/checkpoints/lightzero/iteration_8.pth.tar --episodes 16 --seed 2 --eval-id mcts-scoreboard-512x8-iter8-maxstep512 --max-env-step 512 --num-simulations 8
```

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-ISzvokza20CZzt5S7OsEvK
```

Refs:

```text
summary: eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-maxstep512-20260509T151044Z/summary.json
episodes: eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-maxstep512-20260509T151044Z/episodes.jsonl
checkpoint: training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/checkpoints/lightzero/iteration_8.pth.tar
checkpoint sha256: c67d9835ac9adfcef6b02ff919ba7abbd6dba40cafb59fa2c76f267d2148b8fd
```

LightZero MCTS rows, `episodes=32` because the paired eval seats the checkpoint
on both sides:

```text
lightzero_iter8_vs_random_uniform:
  wins: lightzero_iter8 15, random_uniform 17
  mean_steps: 15.90625; p90_steps: 30.0; truncation_rate: 0.0
  mean_score_return: -0.0625
  mean_shaped_loss_delay_return: -0.053314208984375
  action_histogram lightzero_iter8 [up, stay, down]: [507, 2, 0]

lightzero_iter8_vs_lagged_track_ball_1:
  wins: lightzero_iter8 12, lagged_track_ball_1 18
  mean_steps: 42.59375; p90_steps: 19.0; truncation_rate: 0.0625
  mean_score_return: -0.1875
  mean_shaped_loss_delay_return: -0.18109130859375
  action_histogram lightzero_iter8 [up, stay, down]: [1361, 2, 0]

lightzero_iter8_vs_track_ball:
  wins: lightzero_iter8 0, track_ball 30
  mean_steps: 50.15625; p90_steps: 39.90000000000002; truncation_rate: 0.0625
  mean_score_return: -0.9375
  mean_shaped_loss_delay_return: -0.919769287109375
  action_histogram lightzero_iter8 [up, stay, down]: [1599, 6, 0]
```

Read: the corrected `512/8` checkpoint remains effectively up-only even after
the horizon/config mismatch fix.

## 2026-05-09 Longer Cheap CPU Train: 4096/64

I raised only the scaled train wrapper caps to allow this deliberate run:
`MAX_ALLOWED_ENV_STEP=8192` and `MAX_ALLOWED_TRAIN_ITER=64`. The tiny smoke
wrapper stayed tiny. No GPU flags were added.

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy random_uniform --seed 2 --max-env-step 4096 --max-train-iter 64 --num-simulations 8 --batch-size 32 --update-per-collect 1 --n-evaluator-episode 8
```

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-Uj3XVYgEnzr9oSVan3NILH
```

Refs:

```text
run_id: lz-dpong-20260509T151212Z-b95b61de2eb0
attempt_id: attempt-20260509T151212Z-8b9db08f8fcb
summary: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/train/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/train/episodes.jsonl
```

Checkpoint refs:

```text
ckpt_best: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: 49c4df93dabbfc1cef0cd1f62b04e396bbc886cb88d9c01bbc8e67004f3aa7b0
iteration_0: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/checkpoints/lightzero/iteration_0.pth.tar
  sha256: 9d6fb0534dce7fc2f2719c621faedec9511124ffb2b3338d247269a2ff0c3301
iteration_64: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/checkpoints/lightzero/iteration_64.pth.tar
  sha256: 11a0cc80f797ce8e63150e0a6018efc163b7858bed9efd92b77dda8cadaf95e4
```

Trainer-side env scorecard:

```text
ok: true
episodes: 578
wins: 535
losses: 43
timeouts: 0
truncation_rate: 0.0
survival_steps.mean: 18.18166089965398
survival_steps.median: 19.0
survival_steps.p90: 19.0
survival_steps.max: 52.0
score_return.mean: 0.8512110726643599
shaped_loss_delay_return.mean: 0.8513278631190527
player_0 action_counts: up=9539, stay=800, down=170
player_1 action_counts: up=1836, stay=3328, down=5345
```

## 2026-05-09 Post-Train 4096/64 Iteration 64 MCTS Scorecard

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter64=ref:training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/checkpoints/lightzero/iteration_64.pth.tar --episodes 16 --seed 2 --run-id lz-dpong-20260509T151212Z-b95b61de2eb0 --attempt-id attempt-20260509T151212Z-8b9db08f8fcb --eval-id mcts-scoreboard-4096x64-iter64-maxstep4096 --max-env-step 4096 --num-simulations 8
```

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-G8BlfW9uUBtT7jTKxgtx0U
```

Refs:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/eval/mcts-scoreboard-4096x64-iter64-maxstep4096/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/eval/mcts-scoreboard-4096x64-iter64-maxstep4096/episodes.jsonl
```

LightZero MCTS rows:

```text
lightzero_iter64_vs_random_uniform:
  wins: lightzero_iter64 13, random_uniform 19
  mean_steps: 15.90625; p90_steps: 30.0; truncation_rate: 0.0
  mean_score_return: -0.1875
  mean_shaped_loss_delay_return: -0.18620681762695312
  action_histogram lightzero_iter64 [up, stay, down]: [290, 219, 0]

lightzero_iter64_vs_lagged_track_ball_1:
  wins: lightzero_iter64 11, lagged_track_ball_1 19
  mean_steps: 266.25; p90_steps: 19.0; truncation_rate: 0.0625
  mean_score_return: -0.25
  mean_shaped_loss_delay_return: -0.24916839599609375
  action_histogram lightzero_iter64 [up, stay, down]: [6475, 2045, 0]

lightzero_iter64_vs_track_ball:
  wins: lightzero_iter64 0, track_ball 31
  mean_steps: 144.34375; p90_steps: 30.0; truncation_rate: 0.03125
  mean_score_return: -0.96875
  mean_shaped_loss_delay_return: -0.9667549133300781
  action_histogram lightzero_iter64 [up, stay, down]: [3335, 1284, 0]
```

Read: longer training changed the MCTS action mix from almost all-up to
up-plus-stay, but still never selected down in these scorecard rows and did not
beat random or scripted baselines. The strong trainer-side wins are therefore
not reliable learning evidence.

## Main-Thread 512/8 Run: Completed

The main thread also completed a separate 512/8 LightZero MuZero dummy Pong
run.

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-4XcfDjKPeDhMG2uI93QcYN
```

Run refs:

```text
run_id: lz-dpong-20260509T144635Z-eb5a0ed35de0
attempt_id: attempt-20260509T144635Z-ece79bad80d0
summary_ref: training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/train/summary.json
```

Config:

```text
max_env_step: 512
max_train_iter: 8
num_simulations: 4
batch_size: 16
n_evaluator_episode: 4
seed: 1
```

Trainer-side telemetry:

```text
episodes: 42
wins: 37
losses: 5
timeouts: 0
survival_steps.mean: 9.0476
survival_steps.p90: 8.0
shaped_loss_delay_return.mean: 0.7633
score_return.mean: 0.7619
checkpoint_iterations: [0, 8]
```

Read:

This is a real completed LightZero MuZero training run, but trainer-side
env/evaluator wins are not a policy-quality proof. The independent policy-head
scoreboard for this run found constant-up greedy behavior in both `ckpt_best`
and `iteration_8`.

After the loader fix, the strict-config direct policy-head rerun also completed
with strict `load_state_dict` true:

```text
eval_id: policy-head-scoreboard-512x8-strictcfg
modal_url: https://modal.com/apps/modal-labs/shankha-dev/ap-Q7sPmscebJQWisowuweBxV
summary: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/summary.json
episodes: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/episodes.jsonl
```

The split residual dynamics config fix is working, but direct policy-head
behavior is still constant-up: vs lagged `[590,0,0]`, 13 wins vs 17, 2 truncs,
mean steps 18.4375, shaped -0.0987; vs random `[388,0,0]`, 12 wins vs 20, mean
steps 12.125, shaped -0.2134; vs track `[968,0,0]`, 0 wins vs 28, 4 truncs,
mean steps 30.25, shaped -0.8115.

Correction: the earlier MCTS loader smoke failure on missing
`cfg.policy.device` is stale. Device/action-mask issues were fixed, and the
512/8 `iteration_8` MCTS loader smoke now passes:

```text
probe_ref: training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_mcts_loader_smoke_20260509T145607Z.json
ok: true
mcts_eval_status: ok
strict_full_model_load_ok: true
strict_full_model_load_variant: res_connection_in_dynamics_true
call_shape: data [1,10], action_mask [[1,1,1]], to_play [-1], ready_env_id [0]
action: 0
visit_count_distributions: [2,1,1]
predicted_policy_logits: [0.0170983, 0.00644484, 0.0132326]
predicted_value: about 0.0000259
searched_value: about 0.000114
```

Do not scale more until a full MCTS/eval-mode scorecard exists across
episodes/opponents. The direct policy-head greedy scoreboard remains
constant-up and is still not MCTS proof.
