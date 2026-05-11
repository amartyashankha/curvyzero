# Training Smokes

These commands run the current toy training code. They are not proof that the
models are good.

## Next Actual MuZero Smoke Decision

Current answer as of 2026-05-09: the only actual MuZero-like training that has
already run is stock LightZero CartPole MuZero on CPU Modal. The real trainer
path is `lightzero_cartpole_tiny_train_smoke --mode train`, followed by the
slightly longer `--mode progression` run that emitted evaluator metrics,
learner losses, TensorBoard events, and LightZero checkpoints. No stock Pong
MuZero trainer has run. The Mctx smokes are search-only; the dummy Pong
self-play, CEM, and supervised MLP lanes are useful baselines or scaffolds, but
they are not MuZero.

Current spike order is LightZero-first:

0. LightZero feature-fit audit gate.
1. LightZero custom-env config/import smoke.
2. LightZero tiny dummy Pong MuZero train smoke on Modal.
3. Project-owned Mctx tiny trainer only as fallback/comparison.

Do not start with a project-owned Mctx trainer. The immediate question is
whether LightZero can run a real MuZero trainer on our custom dummy Pong env
without hiding the CurvyZero scorecard and trace metadata.

Immediate next commands, after the named modules exist:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode feature-fit \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --seed 0
```

Do not run either command until the module it names has been added under
`src/curvyzero/infra/modal/`. The first command is the LightZero feature-fit
plus config/import gate. The second command is the tiny LightZero MuZero train
smoke, and it only follows after the feature-fit/config-import gate passes.

First implementation gate: add a LightZero feature-fit audit before or inside
the custom-env config/import smoke. This command is valid after
`curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke --mode feature-fit`
exists:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode feature-fit \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

The feature-fit audit must report these fields before training:

- env reset and one step through the LightZero-facing wrapper;
- observation shape, dtype, schema id, and chosen feature mode;
- legal action handling: accepted `action_mask` or a clear all-actions-legal
  note for `A=3`;
- custom reward and `info` telemetry: seed, step, joint action, winner,
  terminated, truncated, wins, losses, score return, survival steps, mean
  survival steps, median survival steps, p90 survival steps, survival standard
  deviation, truncation rate, shaped loss-delay return, shaped-return standard
  deviation, opponent policy id, and trace hash;
- trainer entrypoint fit: import/signature and confirmation that the patched
  config targets the custom env;
- checkpoint discovery plan: experiment directory and checkpoint filename
  patterns;
- independent CurvyZero scorecard path for evaluating a learned checkpoint
  outside LightZero's own evaluator.

Pass if every required feature is present or deliberately not needed, with no
`missing`, `unknown`, or `hidden_by_framework` fields.

Fail if reset/step cannot run, observation shape is ambiguous, legal actions
cannot be represented or ruled unnecessary, reward/info telemetry is lost, the
trainer target is not the custom env, checkpoint discovery is unclear, or no
independent scorecard path exists.

Second work item: add the custom-env config/import smoke. This command is valid
after `curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke` exists:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

Pass if LightZero/DI-engine imports, sees the custom dummy Pong env,
create/reset/step works, `A=3` and the observation/action surfaces are clear,
the feature-fit report is included, and the patched MuZero config is returned
without calling the trainer.

Fail if the adapter needs invasive framework surgery, hides seed/action/reward
traces, cannot step before training, or falls back to stock CartPole/Pong.

Third work item: add the capped trainer smoke. This command is valid after
`curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke` exists and the
feature-fit plus config/import smoke pass:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --seed 0
```

Pass if it calls LightZero's real MuZero trainer, returns `ok: true`, emits
learner/evaluator signals, writes at least one LightZero checkpoint, and
reports dummy Pong wins/losses, survival, truncation, score return,
loss-delay/shaped proxy, seeds, actions, checkpoint refs, and artifact refs.
It must also run or schedule an independent CurvyZero scorecard outside
LightZero's own evaluator.

Fail if LightZero cannot preserve required telemetry, the artifacts cannot be
mapped into CurvyZero run records, checkpoint discovery fails, the independent
scorecard cannot run, or the adapter becomes larger than the tiny project-owned
fallback trainer.

Fallback work item: only if LightZero-first fails, or after it passes and needs
a complexity comparison, add/run the project-owned Mctx tiny trainer:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_known_env_tiny_train_smoke \
  --env dummy_line_duel \
  --iterations 2 \
  --episodes-per-iter 4 \
  --num-simulations 4 \
  --batch-size 8 \
  --seed 0
```

Keep this third. It is not the next spike.

Existing stock reference command, useful only to recheck the already-proven
external trainer lane:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode progression
```

Treat CartPole as existing-example replication, not as the next CurvyZero
adapter spike.

Follow-up lane: unblock stock LightZero Pong environment creation, not Pong
training yet. The current Pong env smoke reaches ALE reset and fails because
the Pong ROM is missing. The exact unblocker, after explicit Atari ROM license
approval, is to make a separate ROM-enabled Modal image step such as:

```python
image = image.uv_pip_install("AutoROM[accept-rom-license]")
image = image.run_commands("AutoROM --accept-license")
```

Then rerun the env check:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke
```

Only after that command returns `lightzero_path_ok: true` with reset and step
passing should we consider a brutally capped stock Atari
`train_muzero_segment` Pong smoke. This is separate from the immediate custom
dummy Pong LightZero-first lane above.

## Local Toy Baseline

Runs random-vs-random and a privileged heuristic-vs-random on the current
`CurvyTronEnv` toy environment.

```sh
uv run python scripts/run_toy_baseline.py --episodes 100 --seed 0
```

Optional local artifacts:

```sh
uv run python scripts/run_toy_baseline.py \
  --episodes 2 \
  --seed 0 \
  --output-dir artifacts/local/toy_baseline_smoke
```

Current interpretation: the command proves structured baseline summaries work.
It does not prove learnability; the first heuristic smoke lost to random.

## Dummy Survival Training

Runs the first single-player training loop. The code uses MuZero-like names,
but the learner is a simple tabular NumPy dummy learner, not real MuZero.

```sh
uv run python scripts/run_dummy_survival_train.py \
  --iterations 5 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_smoke
```

Optional periodic checkpoints for selection sweeps:

```sh
uv run python scripts/run_dummy_survival_train.py \
  --iterations 8 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2 \
  --checkpoint-every-iterations 2
```

Expected files:

- `summary.json`
- `checkpoint.npz`
- `iteration_metrics.jsonl`

Use this command to check the actor/replay/update/checkpoint/eval path before
adding JAX, Mctx, LightZero, or multiplayer.

## Dummy Survival Baseline Eval

Runs the first fixed-baseline table for the single-player dummy survival task.
This evaluates random and simple scripted policies; it can also load explicit
dummy checkpoints. It does not train or compute ratings.

```sh
uv run python scripts/run_dummy_survival_eval.py \
  --episodes 50 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_eval_smoke
```

Expected files:

- `summary.json`
- `episodes.jsonl`

Default policies are now `random_uniform`, `one_step_safe`, and
`untrained_model_same_planner`. The planner-only baseline is required because
the safety-aware planner can solve the tiny monitor split even with an empty
model.

Use this command as the first debugging floor before comparing learned
checkpoints against random and scripted policies.

Optional learned-checkpoint comparison:

```sh
uv run python scripts/run_dummy_survival_eval.py \
  --episodes 10 \
  --seed 123 \
  --split-id dummy_survival_monitor_v0 \
  --split-role monitor \
  --checkpoint-policy learned:artifacts/local/dummy_survival_smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy_survival_checkpoint_eval_smoke
```

Periodic checkpoint sweep:

```sh
uv run python scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 10 \
  --seed 123 \
  --split-id dummy_survival_monitor_v0 \
  --split-role monitor \
  --output-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2_eval_seed123_e10
```

Expected sweep files:

- `summary.json`
- `checkpoint_eval.jsonl`
- `best_checkpoint.json`
- `best_checkpoint_path.txt`
- `selection_record.json`

The sweep summary also includes `selected_checkpoint`, `latest_checkpoint`,
`eval_split`, and `heldout_required`. For real selection runs, use
`--split-role selection`, then confirm the selected checkpoint once on a
separate `heldout` split before making quality claims.

Current interpretation: after the safety-aware planner patch, iteration 2 and
iteration 4 checkpoints can match `one_step_safe` on the fixed seed-123 smoke,
while later checkpoints degrade. A follow-up planner-only baseline showed that
`untrained_model_same_planner` also solves the same tiny monitor split, so this
is not clean learning evidence. Use sweeps to expose checkpoint damage and
best-checkpoint choice, not to claim robust learning yet.

Experimental collection variant:

```sh
uv run python scripts/run_dummy_survival_train.py \
  --iterations 8 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2 \
  --checkpoint-every-iterations 2 \
  --safety-filter-epsilon
```

The first smoke for this flag was negative/mixed, so keep it opt-in.

Survival diagnostic option:

```sh
uv run python scripts/run_dummy_survival_train.py \
  --iterations 8 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_unknown_pessimism_i8_e20_seed0_c2 \
  --checkpoint-every-iterations 2 \
  --planner-unknown-next-value -1.0
```

The first run with this option did not beat `untrained_model_same_planner`.
Keep it as a diagnostic flag, not a main training path.

Heldout confirmation for a preselected checkpoint:

```sh
uv run python scripts/run_dummy_survival_selection_holdout.py \
  --selection-record artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/selection_record.json \
  --episodes 100 \
  --seed 456 \
  --split-id dummy_survival_heldout_v0 \
  --output-dir artifacts/local/dummy_survival_selection_holdout_smoke
```

Expected artifacts:

- `summary.json`
- `episodes.jsonl`
- `holdout_confirmation.json`

The confirmation status is strict: the selected checkpoint must beat both
`latest` and `untrained_model_same_planner`; tying the planner-only baseline is
`inconclusive`, not a learning claim.

## Modal Dummy Survival Ephemeral

Runs the same dummy survival code as one CPU Modal job. No Volume is attached
yet; remote files are temporary, so the returned JSON summary is the useful
local result.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_survival \
  --iterations 1 \
  --episodes-per-iter 2 \
  --seed 0 \
  --eval-episodes 2
```

Use this only to confirm the Modal source-copy/import/summary path.

## Modal Volume Dummy Survival

Runs the same tiny dummy survival training code as one CPU Modal job, writes
summary/checkpoint/metrics files to the durable `curvyzero-runs` Volume,
commits the Volume, and returns compact file refs plus final eval.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.volume_dummy_survival \
  --iterations 1 \
  --episodes-per-iter 2 \
  --seed 0 \
  --eval-episodes 2
```

Expected Volume path:

```text
training/dummy-survival/volume-smoke/seed-0/iterations-1/episodes-per-iter-2/eval-episodes-2
```

Expected file refs:

- `summary.json`
- `checkpoint.npz`
- `iteration_metrics.jsonl`

This only proves that files persist. It does not implement resume, checkpoint
selection, or a training runner. Rerunning the exact same seed/iteration/episode
tuple writes the same fixed quick-run path.

## Modal Dummy Survival Train Attempt

Runs the tiny dummy survival trainer as one Modal job and writes files in the
run/attempt layout. This is the preferred Modal wrapper for new dummy survival
training smokes.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_survival_train_attempt \
  --iterations 1 \
  --episodes-per-iter 2 \
  --seed 0 \
  --eval-episodes 2
```

Expected Volume path:

```text
training/dummy-survival/<run_id>/attempts/<attempt_id>/train
```

Expected files:

- `training/dummy-survival/<run_id>/run.json`
- `training/dummy-survival/<run_id>/latest_attempt.json`
- `training/dummy-survival/<run_id>/attempts/<attempt_id>/attempt.json`
- `training/dummy-survival/<run_id>/attempts/<attempt_id>/train/summary.json`
- `training/dummy-survival/<run_id>/attempts/<attempt_id>/train/checkpoint.npz`
- `training/dummy-survival/<run_id>/attempts/<attempt_id>/train/iteration_metrics.jsonl`

Optional periodic checkpoints:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_survival_train_attempt \
  --iterations 2 \
  --episodes-per-iter 2 \
  --seed 0 \
  --eval-episodes 2 \
  --checkpoint-every-iterations 1
```

When periodic checkpoints are enabled, the trainer still writes its checkpoints
under the attempt train directory. The wrapper also copies each one to
`training/dummy-survival/<run_id>/checkpoints/iteration-00000N/checkpoint.npz`
and writes `training/dummy-survival/<run_id>/checkpoints/latest.json`.

This wrapper does not resume a stopped run. Use a new attempt id for another
try.

## Modal Mctx Dependency Smoke

Runs a contained JAX/Mctx dependency and synthetic search check in Modal. This
is the first MuZero-family runtime smoke; it is not Pong, not LightZero, and not
training.

CPU import/search smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu
```

Optional cheap GPU import/search smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

The GPU function requests `["L4", "T4"]`. The image pins `mctx==0.0.6`; CPU
uses `jax==0.7.0`/`jaxlib==0.7.0`, while GPU uses `jax[cuda12]==0.7.0`. The
remote profile is deliberately tiny: `B=4`, `A=3`, hidden dim 8, 4 simulations,
and max depth 4.

Interpretation:

- Pass: returned JSON has `ok: true`, package versions are present,
  `action_weights_finite: true`, row sums are near 1.0, and the GPU run reports
  a GPU JAX backend.
- Fail/diagnose: `ok: false`, missing packages, CPU backend in the GPU run, or
  non-finite/unnormalized action weights.

Expected cost/runtime risk: first invocation may spend a few minutes building
the Modal image and downloading JAX wheels. The actual smoke search should be
seconds once the image exists. Do not increase batch size, simulations, or GPU
class in this module; use a separate benchmark when the dependency path passes.

## Modal Mctx Synthetic Benchmark

Runs one contained synthetic Gumbel MuZero search benchmark on a cheap Modal GPU.
This is not a trainer: there is no real environment, replay buffer, checkpoint,
or optimizer. It measures fixed-shape JAX/Mctx runtime behavior only.

Tiny remote smoke profile:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 8 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

The function requests `["L4", "T4"]` and uses the same passing GPU dependency
pins as the smoke: `mctx==0.0.6`, `jax[cuda12]==0.7.0`, and `numpy>=1.26`.

Returned JSON includes:

- package versions, JAX backend/devices, and `nvidia-smi`;
- the exact shape/config for the profile;
- compile-plus-first-run time;
- warmup and steady-state wall times;
- median decisions/sec and simulations/sec;
- action histogram plus finite and normalized `action_weights` checks;
- `problems`, which is empty on a clean run.

Interpretation:

- Pass: `ok: true`, GPU backend in `jax.default_backend`, finite action
  weights, row sums near 1.0, and steady-state times clearly separated from
  compile time.
- Diagnose before scaling: `ok: false`, CPU backend on the GPU function,
  missing packages, non-finite weights, unnormalized weights, or unexpectedly
  slow steady-state timing after the image is already built.

After the tiny profile passes, sweep one shape at a time. This larger profile
has now passed and can be used as the first non-tiny L4 reference:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 64 \
  --num-simulations 16 \
  --hidden-dim 64 \
  --max-depth 16 \
  --warmup-runs 2 \
  --steady-runs 5
```

Latest observed tiny-profile pass on 2026-05-09:

```text
B=8, simulations=4, hidden_dim=32, max_depth=4, warmup=1, steady=2
Modal app: ap-E6lvbtm5xPQQ21nAnE2HHM
GPU: NVIDIA L4, 23034 MiB total, 17140 MiB used, 0%, driver 580.95.05
JAX backend: gpu/cuda:0, jax/jaxlib 0.7.0, mctx 0.0.6
compile_plus_first_run_sec: 4.855567563
warmup_times_sec: [0.0021835409999990674]
steady_times_sec: [0.001847238999999945, 0.0017432480000003636]
steady_median_sec: 0.0017952435000001543
decisions_per_sec_median: 4456.220005809414
simulations_per_sec_median: 17824.880023237656
action_weights_finite: true
action_weights_normalized: true
problems: []
```

Observed larger-profile pass on 2026-05-09:

```text
B=64, simulations=16, hidden_dim=64, max_depth=16, warmup=2, steady=5
Modal app: ap-ULhQNpnV6a1lsn0uQLUbnX
GPU: NVIDIA L4, JAX backend gpu/cuda:0
compile_plus_first_run_sec: 8.080801095000002
steady_median_sec: 0.005292786999998356
decisions_per_sec_median: 12091.928127850202
simulations_per_sec_median: 193470.85004560323
action_weights_finite: true
action_weights_normalized: true
```

## Modal LightZero Stock Config Smoke

Runs a contained CPU Modal import/config check for stock LightZero MuZero
examples. This is the first stock-example replication step. It does not call
`train_muzero`, does not call `train_muzero_segment`, and does not run CartPole
or Pong training.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dependency_smoke
```

What it checks:

- installs pinned `LightZero==0.2.0` in the remote CPU image;
- imports `lzero`, `ding`, `torch`, and `easydict`;
- imports the stock CartPole MuZero config module and checks that it is
  `CartPole-v0`, `muzero`, `cartpole_lightzero`, MLP, action space 2;
- imports the stock Atari Pong MuZero segment config module, monkeypatches
  `lzero.entry.train_muzero_segment` to a capture function, then calls the stock
  config builder for `PongNoFrameskip-v4` without training.

Interpretation:

- Pass: returned JSON has `ok: true`, package versions are present, all imports
  are `ok`, CartPole has the expected MuZero config surface, and the captured
  Pong config is `PongNoFrameskip-v4`, `muzero`, `atari_lightzero`, conv,
  action space size 6, `num_simulations: 50`, `batch_size: 256`, and
  `captured_max_env_step: 500000`.
- Fail/diagnose: missing packages, failed imports, CartPole config mismatch, or
  Pong capture/config mismatch. Do not start a LightZero trainer until this
  smoke passes or the dependency/config drift is understood.

Observed pass on 2026-05-09:

```text
LightZero 0.2.0, DI-engine 0.5.3, torch 2.11.0
CartPole: max_env_step 100000, collector_env_num 8, num_simulations 25,
  batch_size 256, cuda true
Pong: observation_shape [4, 96, 96], max_env_step 500000,
  collector_env_num 8, num_simulations 50, batch_size 256, cuda true
```

Note: the first Modal image build pulled the full PyTorch/CUDA dependency stack
even though this smoke runs on CPU. That is acceptable for this one dependency
smoke, but it is not a cheap trainer image yet.

Next command after a pass is not Pong. It is a separate, brutally capped
CartPole MuZero trainer smoke with CPU, tiny env counts, tiny simulation count,
tiny batch size, and a very low `max_env_step`.

## Modal LightZero Stock CartPole Tiny Train Smoke

Prepares the next stock LightZero replication step after the config smoke. This
is still the stock LightZero CartPole MuZero path, not CurvyZero's trainer and
not Pong. The default command is intentionally dry: it imports the installed
stock CartPole config, copies it, applies the tiny CPU caps, inspects
`lzero.entry.train_muzero`, and reports the exact patched surface without
starting training.

Safe dry/config-patch smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke
```

Default caps:

```text
mode: dry
seed: 0
max_env_step: 4
max_train_iter: 1
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 2
batch_size: 4
update_per_collect: 1
cuda: false
```

Dry pass criteria:

- returned JSON has `ok: true`;
- `label` is `stock LightZero CartPole MuZero tiny trainer smoke`;
- `mode` is `dry` and `call_policy` is `dry_config_patch_only`;
- patched surface is `CartPole-v0`, `muzero`, `cartpole_lightzero`, MLP,
  action space 2, and `cuda: false`;
- caps are at or below the defaults above;
- `trainer_entrypoint` is `lzero.entry.train_muzero`.

Opt-in real tiny trainer command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode train
```

Real trainer stop criteria:

- Modal function timeout is 8 minutes;
- the LightZero trainer is called with `max_train_iter: 1` and
  `max_env_step: 4`;
- if returned JSON has `train_result.ok: true`, the stock entrypoint completed
  under the cap;
- if it fails or times out, do not increase caps. Diagnose the config/env
  manager/dependency failure first.

Risk note: even with `max_env_step: 4`, LightZero may complete one CartPole
episode or an initial evaluation episode before the trainer loop observes the
cap. That is still small for CartPole, but the real `--mode train` command is
not the default until the dry smoke has passed.

Observed pass on 2026-05-09:

```text
dry smoke: ok true, remote_elapsed_sec 12.847235
train smoke: ok true, remote_elapsed_sec 13.269896
train_result: ok true, return_type MuZeroPolicy, elapsed_sec 4.356072
patched caps: collector/evaluator env 1, n_episode 1, n_evaluator_episode 1,
  num_simulations 2, batch_size 4, update_per_collect 1, cuda false,
  max_train_iter 1, max_env_step 4
trainer signature included max_train_iter and max_env_step
initial evaluator episode logged reward 9.0 / envstep_count 9.0
```

Interpretation: this proves that the pinned stock LightZero CartPole MuZero
entrypoint can start and stop under the tiny caps on Modal CPU. It is not a
quality result, not a Pong result, and not a CurvyZero trainer result.

Slightly longer stock CartPole progression command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode progression
```

Progression caps:

```text
mode: progression
seed: 0
max_env_step: 128
max_train_iter: 4
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 5
batch_size: 16
update_per_collect: 4
eval_freq: 1
cuda: false
```

Progression pass criteria:

- returned JSON has `ok: true` and `train_result.ok: true`;
- `log_signals.max_checkpoint_iteration` is `4`;
- `log_signals.table_metrics` includes evaluator metrics such as
  `reward_mean` and `eval_episode_return_mean`;
- `log_signals.table_metrics` includes learner metrics such as
  `total_loss_avg`, `policy_loss_avg`, and `value_loss_avg`;
- `artifact_summary.checkpoint_files` includes `iteration_4.pth.tar`.

Observed progression pass on 2026-05-09:

```text
ok: true, remote_elapsed_sec 15.603642, train elapsed_sec 5.651327
return_type: MuZeroPolicy
final_rewards: [33.0]
max_checkpoint_iteration: 4
table_metrics: reward_mean 33.0, eval_episode_return_mean 33.0,
  total_loss_avg 45.577473, policy_loss_avg 3.855631,
  value_loss_avg 38.391567
artifacts: ckpt_best.pth.tar, iteration_0.pth.tar, iteration_4.pth.tar,
  log/evaluator/evaluator_logger.txt, log/learner/learner_logger.txt,
  log/serial/events.out.tfevents.*
```

Interpretation: the existing-example lane is now validated for stock CartPole
MuZero progression on CPU Modal. This is still not a convergence or quality
claim, and stock Pong training remains unvalidated.

## Modal LightZero Stock Pong Dry Config Smoke

Runs the next stock LightZero reference check for Atari Pong, but keeps it
dry. It imports the installed stock Atari MuZero segment config, monkeypatches
`lzero.entry.train_muzero_segment` to capture the generated configs, applies
tiny CPU patches to the captured config, and returns the before/after surfaces.
It does not instantiate ALE/Gym/EnvPool, does not require Atari ROMs, and does
not call the trainer.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_dry_config_smoke
```

Default dry caps:

```text
seed: 0
max_env_step: 4
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
num_simulations: 2
batch_size: 4
update_per_collect: 1
cuda: false
```

Dry pass criteria:

- returned JSON has `ok: true`;
- `label` is `stock LightZero Atari Pong MuZero dry config smoke`;
- `mode` is `dry` and `train_result` is `null`;
- patched surface is `PongNoFrameskip-v4`, `muzero`, `atari_lightzero`,
  convolutional model, observation shape `[4, 96, 96]`, action space 6, and
  `cuda: false`;
- caps are at or below the defaults above;
- `trainer_entrypoint` is `lzero.entry.train_muzero_segment`.

Observed pass on 2026-05-09:

```text
ok: true, remote_elapsed_sec 12.235089
packages: LightZero 0.2.0, DI-engine 0.5.3, torch 2.11.0, easydict 1.13
original Pong: collector_env_num 8, evaluator_env_num 3,
  n_evaluator_episode 3, num_simulations 50, batch_size 256, cuda true,
  max_env_step 500000, observation_shape [4, 96, 96]
patched Pong: collector/evaluator env 1, n_evaluator_episode 1,
  num_simulations 2, batch_size 4, update_per_collect 1, cuda false,
  max_env_step 4
```

Interpretation: this is useful as a stock visual MuZero config reference only.
Do not promote it to a real Pong trainer yet. The dry run intentionally dodges
Atari environment creation, ROM availability, EnvPool/Gym behavior, replay, and
the heavyweight visual trainer loop. If stock Pong becomes necessary, add a
separate Atari environment-creation smoke before calling
`train_muzero_segment`.

## Modal LightZero Stock Pong Env Smoke

Runs the next no-train stock Pong check. It imports the same stock Atari MuZero
segment config, applies the same tiny CPU patches, then tries to create, reset,
and step `PongNoFrameskip-v4` through DI-engine/LightZero env paths. Plain
Gym/Gymnasium creation is included only as a diagnostic fallback.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke
```

Pass criteria:

- returned JSON has `ok: true`;
- `lightzero_path_ok` is `true`;
- the first successful path includes `reset.ok: true` and `step.ok: true`;
- `train_result` is `null`.

Observed result on 2026-05-09:

```text
ok: false
env_ok: false
lightzero_path_ok: false
packages: LightZero 0.2.0, DI-engine 0.5.3, gym 0.25.1,
  gymnasium 0.28.0, ale-py 0.8.1,
  opencv-python-headless 4.11.0.86, AutoROM missing, envpool missing
imports: cv2 ok
LightZero/DI-engine failure: reset reaches ALE, then Pong ROM is missing
plain Gym/Gymnasium fallback: unable to find game "Pong"; ROM missing
```

Interpretation: OpenCV is prepared, but the current image cannot reset/step the
stock Pong env because the ROM is not installed. Do not add ROMs silently. If
the Atari ROM license is acceptable for the project, the explicit next prep is
one clearly documented image step such as `AutoROM --accept-license` or
installing the AutoROM accept-license extra. Then rerun this env smoke before
any `train_muzero_segment` attempt.

## Dummy Pong Observability Harness

Runs a tiny deterministic fixed-policy Pong trace harness. This is for
inspecting game dynamics and future visual learner inputs, not for training or
ratings. Tabular ego observations remain in the step rows for debugging and
eval continuity; the MuZero-facing Pong path is the compact raster grid in
`frames.jsonl`.

```sh
uv run python scripts/run_dummy_pong_observability.py \
  --games-per-match 1 \
  --seed 123 \
  --max-steps 16 \
  --output-dir artifacts/local/dummy-pong-observability-smoke
```

Expected artifacts:

- `summary.json`
- `games.jsonl`
- `steps.jsonl`
- `frames.jsonl`

Use `steps.jsonl` for policy/action/reward/terminal debugging and join
`raster_frame_id` to `frames.jsonl` when inspecting the tiny visual observation
for each post-step state. `frames.jsonl` also includes one reset frame at
`step_index: 0` per game, so frame rows are `step_rows + total_games`.
The short `--max-steps` cap is intentional for quick-run size and can create
truncations that are harness behavior rather than policy quality.

## Dummy Pong Current Decision Path

Use this path when checking Pong artifacts:

```text
fixed-baseline scoreboard -> critique decision
                          -> repair self-play or switch baseline/curriculum
```

The self-play commands below are a reproduction path under critique, not an
active training recommendation. Gen2 lost to its parent and won 0 games against
`track_ball`. Older imitation, scoring replay, value-target, lookahead, and
contact commands are historical plumbing or diagnostics. Use the scoreboard for
policy checkpoint claims; use probes as diagnostics.

## Dummy Pong Self-Play Replay Reproduction

Builds visual self-play rows for both ego players. This reproduces the current
self-play scaffold under critique.

```sh
uv run python scripts/build_dummy_pong_selfplay_replay.py \
  --games 16 \
  --seed 23 \
  --max-steps 80 \
  --policy random_uniform \
  --epsilon 0.0 \
  --output-dir artifacts/local/dummy-pong-selfplay-random-replay-smoke-2026-05-09
```

Expected artifacts:

- `summary.json`
- `replay_rows.jsonl`

Rows include the played action, joint action, raw `score_return`, and shaped
`shaped_return`. The shaped target gives losing episodes partial credit for
lasting longer and is not an eval metric.

## Dummy Pong Self-Play Train Reproduction

Trains the tiny raster policy/value checkpoint from self-play rows. Do this to
reproduce or inspect the current trainer, not as the default next path.

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

The first smoke wrote epoch 25 and epoch 50 checkpoints. A higher
`--value-learning-rate 0.05` diverged, so keep the default at `0.001` until a
better optimizer exists.

Score the periodic checkpoints:

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

Current result: the loop works, but quality is weak. `selfplay50` tied random
8/16, beat `selfplay25` 10/16 in learned-vs-learned, and still won 0 games
against `track_ball`.

Before touching this trainer, check the simple aliasing bug noted in the plan:
`policy_grad = probs` likely should be `policy_grad = probs.copy()` if the
diversity regularizer or later gradient edits should not mutate `probs`.

Generation 2 smoke:

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

Current generation-2 result: do not promote. `gen2_50` beat random 20/32, but
all gen2 checkpoints lost to the parent checkpoint and won 0 games against
`track_ball`. Do not run more generations until the critique decision chooses
repairing this trainer over a simpler known baseline/curriculum.

## Modal Dummy Pong Train Attempt

Runs the current dummy Pong self-play replay builder and tiny NumPy trainer as
one CPU Modal job. This is the preferred way to reproduce the current Pong
self-play trainer because it writes replay, checkpoint, summaries, manifests,
and the latest checkpoint pointer to the durable `curvyzero-runs` Volume.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --games 16 \
  --epochs 5 \
  --seed 0 \
  --max-steps 120 \
  --policy random_uniform \
  --epsilon 0.0 \
  --checkpoint-every-epochs 5
```

Expected Volume path:

```text
training/dummy-pong/<run_id>/attempts/<attempt_id>/
```

Expected files:

- `training/dummy-pong/<run_id>/run.json`
- `training/dummy-pong/<run_id>/latest_attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/replay/summary.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/replay/replay_rows.jsonl`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/summary.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoint.npz`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoints/...` when `--checkpoint-every-epochs` is set
- `training/dummy-pong/<run_id>/checkpoints/latest.json`

For learned self-play or warm starts, prefer Volume refs:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --games 16 \
  --epochs 5 \
  --seed 1 \
  --policy learned:ref:training/dummy-pong/RUN_ID/attempts/ATTEMPT_ID/train/checkpoint.npz \
  --initial-checkpoint ref:training/dummy-pong/RUN_ID/attempts/ATTEMPT_ID/train/checkpoint.npz
```

This wrapper proves Modal artifact discipline and enables remote Pong
reproduction. It does not prove the current self-play objective is correct,
final, or capable of learning a strong Pong policy. Score any resulting
checkpoint with the Modal scoreboard before making policy-quality claims.

## Modal Dummy Pong CEM Train Attempt

Runs the score-primary geometry CEM learner as one CPU Modal job. This is the
Modal-backed reproduction path for the CEM-v2 `lagged_track_ball_1` lane. It
writes the trainer summary, loadable checkpoint, compact CEM rows, manifests,
and a latest checkpoint pointer to the durable `curvyzero-runs` Volume.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_cem_train_attempt \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --generations 8 \
  --population-size 32 \
  --elite-count 8 \
  --eval-games 16 \
  --seed 8050913 \
  --opponent-weights lagged_track_ball_1=1.0,random_uniform=0.10,track_ball=0.10 \
  --target-opponent-id lagged_track_ball_1 \
  --loss-delay-weight 0.5 \
  --truncation-value 0.0
```

Expected Volume path:

```text
training/dummy-pong/<run_id>/attempts/<attempt_id>/train
```

Expected files:

- `training/dummy-pong/<run_id>/run.json`
- `training/dummy-pong/<run_id>/latest_attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/summary.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoint.npz`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/cem_rows.jsonl`
- `training/dummy-pong/<run_id>/checkpoints/latest.json`

Observed CEM-v2 Modal monitor pass on 2026-05-09:

```text
Modal app run: ap-SzIu3KSSe7NRAq2Iqn33Yu
run_id: pong-cem-20260509T045950Z-e8b06974a402
attempt_id: attempt-20260509T045950Z-f16d342d760b
final eval vs lagged_track_ball_1: 25/32 learner wins, 7 truncations,
  mean steps 38.6875, mean score return 0.78125
final eval vs random_uniform: 30/32 learner wins, mean score return 0.875
final eval vs track_ball: 32/32 truncations, mean steps 120.0
```

Score the resulting checkpoint remotely:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints cem_v2=ref:training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/attempts/attempt-20260509T045950Z-f16d342d760b/train/checkpoint.npz \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_cem_v2_modal_lagged_track_ball_1 \
  --split-role monitor
```

Observed scoreboard pass:

```text
Modal app run: ap-nulgA7l3s4pfcMZUZhOyuO
run_id: pong-scoreboard-20260509T050220Z-84b0c61e5ab9
attempt_id: attempt-20260509T050220Z-b0a25fb91c80
learned_cem_v2_vs_lagged_track_ball_1: 53/64 learned wins,
  1 opponent win, 10 truncations, mean steps 31.34375
learned_cem_v2_vs_random_uniform: 60/64 learned wins, mean steps 22.4375
learned_cem_v2_vs_track_ball: 64/64 truncations, mean steps 120.0
```

This wrapper proves the CEM-v2 lane can train remotely with durable artifacts.
It is still a CPU NumPy baseline, not MuZero or a GPU training path.

## Modal Dummy Pong Imitation Train Attempt

Runs the existing supervised dummy Pong imitation trainer as one CPU Modal job
from an already-built replay. This is the simplest Modalization for the
stack-2 `raster_only` MLP lane: the wrapper resolves a Volume replay ref, copies
`replay_rows.jsonl` into the attempt's `replay/` directory, trains into
`train/`, writes manifests and a latest checkpoint pointer, commits
`curvyzero-runs`, and returns compact refs.

First place the local lag-1 stack-2 replay on the Volume:

```sh
modal volume put curvyzero-runs \
  artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09/replay_rows.jsonl \
  training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl
```

Then launch the tiny CPU NumPy MLP trainer:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_imitation_train_attempt \
  --replay-path ref:training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl \
  --seed 0 \
  --epochs 800 \
  --learning-rate 0.005 \
  --validation-fraction 0.2 \
  --class-weighting balanced \
  --feature-mode raster_only \
  --frame-stack 2 \
  --model-type mlp \
  --hidden-dim 128
```

Expected Volume path:

```text
training/dummy-pong/<run_id>/attempts/<attempt_id>/
```

Expected files:

- `training/dummy-pong/<run_id>/run.json`
- `training/dummy-pong/<run_id>/latest_attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/replay/replay_rows.jsonl`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/summary.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoint.npz`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoints/...` when `--checkpoint-every-epochs` is set
- `training/dummy-pong/<run_id>/checkpoints/latest.json`

Score the resulting checkpoint with the Modal scoreboard before making a
heldout or CEM-v2 comparison claim:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints mlp_stack2=ref:training/dummy-pong/RUN_ID/attempts/ATTEMPT_ID/train/checkpoint.npz \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_lag1_raster_only_mlp_modal \
  --split-role monitor
```

This wrapper proves Modal artifact discipline for the visual-only imitation
learner. It does not prove MuZero, on-policy learning, or GPU training.

Observed 2026-05-09 pass:

```text
train Modal app run: ap-D7PXnC4IvX4gezXH7uhwRm
train run_id: pong-imitation-20260509T055813Z-4506d7a50304
train attempt_id: attempt-20260509T055813Z-944cfc7b4c22
checkpoint ref: training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/attempts/attempt-20260509T055813Z-944cfc7b4c22/train/checkpoint.npz
scoreboard Modal app run: ap-nQPH5XM4aXszCOtWUDi7bX
scoreboard run_id: pong-scoreboard-20260509T055921Z-402ab3dba50c
scoreboard attempt_id: attempt-20260509T055921Z-3bf5d8d94220
scoreboard summary: training/dummy-pong/pong-scoreboard-20260509T055921Z-402ab3dba50c/attempts/attempt-20260509T055921Z-3bf5d8d94220/eval/checkpoint-scoreboard/summary.json
learned_mlp_stack2_vs_lagged_track_ball_1: 49/64 learned wins, 4 lag-1 wins, 11 truncations, mean steps 33.265625
learned_mlp_stack2_vs_random_uniform: 34/64 learned wins, 30 random wins, mean steps 14.875
learned_mlp_stack2_vs_track_ball: 0/64 learned wins, 39 track_ball wins, 25 truncations, mean steps 66.1875
```

## Dummy Pong Imitation Replay

Builds learner-ready replay rows from `track_ball` actions over Pong raster
observations. This is historical raster plumbing, not the current self-play
training objective.

```sh
uv run python scripts/build_dummy_pong_imitation_replay.py \
  --games 32 \
  --seed 0 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-imitation-replay-v0
```

Expected artifacts:

- `summary.json`
- `replay_rows.jsonl`

Use `replay_rows.jsonl` as the first supervised input for a tiny raster policy
learner. Each row has the pre-step `raster_grid`, `ego_agent`,
`target_action_id`, `target_action_label`, reward after the step, and the next
raster grid.

Current v0 note: `track_ball` versus `track_ball` kept all 32 games alive until
the step cap, so this replay is useful for imitation but not for reward-learning
claims. Use score-delta reward for reward-learning tests, and keep rally length
as a log.

## Dummy Pong Artifact Inspector

Inspects Pong replay or trace directories without opening raw JSONL by hand.
Use this before training from an artifact.

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-imitation-replay-v0 \
  --sample-frames 1
```

Useful fields:

- detected files
- row counts
- raster shape and schema
- action histograms
- reward totals
- terminated/truncated counts
- sample frame strings when `frames.jsonl` exists
- quality notes

Current v0 note: the inspector flags the imitation replay as useful for
copying `track_ball`, but not useful for reward learning, because all replay
rewards are zero and every game truncated at the step cap.

## Dummy Pong Imitation Training

Trains the first tiny learned Pong checkpoint from raster replay rows. This
copies the scripted `track_ball` policy. It is not MuZero, self-play, or reward
learning.

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-imitation-replay-v0 \
  --output-dir artifacts/local/dummy-pong-imitation-train-smoke \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 1.0 \
  --validation-fraction 0.2
```

Expected artifacts:

- `summary.json`
- `checkpoint.npz`

Optional periodic checkpoints for old-vs-new eval:

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-imitation-replay-smoke \
  --output-dir artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09 \
  --seed 0 \
  --epochs 3 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 1
```

Additional expected artifacts:

- `checkpoints/epoch-000001/checkpoint.npz`
- `checkpoints/epoch-000002/checkpoint.npz`
- `checkpoints/epoch-000003/checkpoint.npz`

When enabled, `summary.json` includes `checkpoints.count`,
`checkpoints.refs`, and `checkpoints.latest`; the root `checkpoint.npz` remains
the final policy checkpoint.

Current smoke result: validation accuracy was about 99.0% against held-out
rows from the same replay. This proves the raster replay can feed a small
supervised learner and save a reloadable checkpoint. It does not prove reward
learning because the source replay had zero score rewards.

## Dummy Pong Scoring Data Check

Checks which fixed-policy Pong matchups produce real score events under the
score-delta reward.

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 32 \
  --seed 0 \
  --output-dir artifacts/local/dummy-pong-scoring-data-smoke-2026-05-09
```

Current smoke result: `random_uniform` versus `random_uniform` scored in all 32
games, and `track_ball` beat `random_uniform` in all 64 seated games with no
truncations. `track_ball` versus `track_ball` still timed out every game.

Use random opponents for the first reward-data replay. Do not add rally-length
reward or biased starts for v0.

Scoring replay export:

```sh
uv run python scripts/build_dummy_pong_scoring_replay.py \
  --games-per-seat 4 \
  --seed 0 \
  --max-steps 120 \
  --row-policy all \
  --output-dir artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09
```

Use `--row-policy all` when value targets need both winning and losing ego
examples. All-ego replay is good for value labels because it has wins and
losses. It is bad for expert action cloning because it has random actions. Use
the default `--row-policy track_ball` only for expert-action copying. The
all-ego smoke produced 8 positive and 8 negative terminal reward rows with no
truncations.

## Dummy Pong Value/Reward Target Smoke

Trains a small value regressor from all-ego scoring replay. This backs up
score-delta rewards into `target_return` labels. It does not improve a policy.

```sh
uv run python scripts/train_dummy_pong_value.py \
  --replay-path artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-scoring-all-ego-value-train-smoke-2026-05-09 \
  --seed 0 \
  --validation-fraction 0.2 \
  --discount 1.0 \
  --ridge-l2 0.000001
```

Expected artifacts:

- `summary.json`
- `checkpoint.npz`

Current smoke result: 392 rows became 16 `(game_index, ego_agent)` return
groups. Target returns had 196 positive rows and 196 negative rows. Validation
MSE was about 1.67, so this proves the target/checkpoint path, not strong value
prediction.

The checkpoint scoreboard now exists. Keep fixed baseline rows stable, then use
the scoreboard to compare candidate/latest checkpoints against older or best
checkpoints. Periodic Pong policy checkpoints now exist, but the first periodic
scoreboard smoke proves plumbing only.

## Dummy Pong Angle-Control Probe

Runs a focused scripted probe that tries to make top or bottom paddle contacts.
This measures the mini North Star directly before adding a learner.

```sh
uv run python scripts/run_dummy_pong_angle_control_probe.py \
  --episodes 16 \
  --seed 0 \
  --output-dir artifacts/local/dummy-pong-angle-control-probe-smoke
```

Expected artifacts:

- `summary.json`
- `episodes.jsonl`

Current smoke result: `angle_control` beat `random_uniform` 32/32 across both
seats and made 23/23 off-center contacts. Against `track_ball`, all 32 games
truncated, even though `angle_control` made 172/172 off-center contacts.

Interpretation: off-center contact is scriptable and measurable, but a naive
off-center policy does not beat `track_ball`. The next Pong step is a
contact-outcome dataset that compares top, center, and bottom contacts over a
short horizon.

Important caveat: `track_ball` already creates many off-center hits because of
one-step movement lag. Treat off-center rate as a debug count. The useful
metric is short score-delta return after a contact.

## Dummy Pong Contact-Outcome Dataset Probe

Builds controlled near-contact rows for top, center, and bottom ego paddle
contacts. Each row records predicted hit row, desired and actual impact offset,
whether the target center was reachable from the base state in one step,
outgoing `ball_vy`, and short post-contact score delta. The summary also
compares same-state pure `track_ball`, since off-center rate alone is weak.

```sh
uv run python scripts/build_dummy_pong_contact_outcomes.py \
  --states 4 \
  --seed 0 \
  --horizon 24 \
  --output-dir artifacts/local/dummy-pong-contact-outcomes-smoke
```

Expected artifacts:

- `summary.json`
- `contact_rows.jsonl`

Current smoke result: top/center/bottom contacts changed outgoing `ball_vy` on
all sampled states, but all candidate score-delta returns stayed `0.0` against
`track_ball` and all rows truncated. Same-state pure `track_ball` also returned
`0.0`, and made off-center contacts on 3/4 states. The default geometry may be
too forgiving; next knobs are smaller width, smaller paddle, or faster ball.

## Dummy Pong Lag-1 Trace Frame Stack

Builds exact lag-1 trace replay rows with optional chronological raster history.
`--frame-stack 2` emits `raster_frame_stack` and `next_raster_frame_stack`;
training with `--frame-stack 2` widens the checkpoint feature axis, and eval
maintains per-agent raster history from checkpoint metadata.

```sh
uv run python scripts/build_dummy_pong_lag1_trace_replay.py \
  --max-steps 120 \
  --repeats 1 \
  --frame-stack 2 \
  --include-vertical-mirror \
  --balance-actions oversample \
  --balance-seed 0 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-policy-stack2-e400-smoke-2026-05-09 \
  --seed 0 \
  --epochs 400 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --feature-mode raster_plus_geometry \
  --frame-stack 2
```

Current 2026-05-09 smoke result: stack 2 did not improve supervised label
accuracy over stack 1, but it weakly improved closed-loop behavior: 13/32 wins
versus `lagged_track_ball_1` instead of 11/32, 15/32 versus random instead of
12/32, and 6/32 truncations versus default `track_ball` instead of 4/32. Treat
this as a small positive input-feature signal, not a solved visual lane.

## Dummy Pong Lookahead Replay

Builds raster replay rows whose `target_action` is selected by short
score-delta lookahead instead of plain behavior cloning. For each sampled ego
state, the builder clones the toy env, tries all three immediate ego actions
against fixed-opponent `track_ball`, rolls out `track_ball` for both agents,
and labels the best score-delta return.

Strict score-separated smoke:

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 2 \
  --seed 0 \
  --max-steps 60 \
  --lookahead-steps 16 \
  --collector-policy random_uniform \
  --output-dir artifacts/local/dummy-pong-lookahead-replay-smoke-2026-05-09
```

Angle-tie variant that emits equal-return states and prefers the existing
`angle_control` action when it is tied best:

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 2 \
  --seed 0 \
  --max-steps 60 \
  --lookahead-steps 16 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy angle_control \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-replay-smoke-2026-05-09
```

Expected artifacts:

- `summary.json`
- `replay_rows.jsonl`

The replay can train the existing tiny supervised raster policy:

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lookahead-angle-tie-replay-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-policy-train-smoke-2026-05-09 \
  --seed 0 \
  --epochs 80 \
  --learning-rate 0.5 \
  --validation-fraction 0.2
```

Then score behavior, not just label accuracy:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 4 \
  --seed 5 \
  --split-id dummy_pong_lookahead_angle_tie_smoke \
  --split-role monitor \
  --checkpoint lookahead_angle_tie=artifacts/local/dummy-pong-lookahead-angle-tie-policy-train-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-scoreboard-smoke-2026-05-09
```

Current smoke result: strict score-separated replay emitted 9 rows and all
targets matched `track_ball`, so it is not a useful improvement signal by
itself. The angle-tie replay emitted 131 rows with 41 targets different from
`track_ball`; the tiny trained checkpoint still scored 0/8 wins against
`track_ball`, but forced 2/8 truncations. Treat this as a pressure diagnostic,
not a success claim.

Larger angle-tie check:

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 32 \
  --seed 7 \
  --max-steps 120 \
  --lookahead-steps 32 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy angle_control \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-replay-g32-h32-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lookahead-angle-tie-replay-g32-h32-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09 \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 250
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 313 \
  --split-id dummy_pong_lookahead_angle_tie_g32_h32_monitor \
  --split-role monitor \
  --checkpoint lookahead250=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz \
  --checkpoint lookahead500=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000500/checkpoint.npz \
  --checkpoint lookahead750=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz \
  --checkpoint lookahead1000=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --checkpoint imitation1000=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-scoreboard-monitor-2026-05-09
```

Current larger result: the replay produced 1,669 rows and 442 targets different
from `track_ball`, but the scoreboard still selected imitation epoch 1000. All
lookahead checkpoints won 0/64 against `track_ball`; the best lookahead random
row was 41/64 versus imitation's 44/64. Stop scaling one-step angle-tie labels
unless a bug is found.

## Dummy Pong Learned Checkpoint Eval

Runs the Pong eval matrix with a learned raster checkpoint.

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 32 \
  --seed 0 \
  --checkpoint-policy learned:artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-checkpoint-eval-e32-seed0
```

Current 32-episode result: the learned checkpoint beat random 43/64, but
scripted `track_ball` beat random 64/64. Against `track_ball`, the learned
checkpoint won 0/64 and most games timed out. Treat this as a runnable
checkpoint/eval proof, not a strong policy.

Training from the small positive-only scoring replay moved learned-vs-random
from 43/64 to 44/64 and did not close the gap to `track_ball`. Training a plain
action clone on all-ego replay dropped learned-vs-random to 41/64 because random
action rows are not expert policy targets.

## Dummy Pong Checkpoint Scoreboard

Runs the small Pong checkpoint scoreboard. Main rows are learned checkpoints
against `random_uniform` and `track_ball`; when multiple checkpoints are passed,
the command also includes learned-vs-learned paired-seat rows. Baseline sanity
rows stay in the artifact so random and `track_ball` behavior is visible.

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor \
  --checkpoint latest=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-checkpoint-scoreboard
```

Two-checkpoint plumbing smoke:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor \
  --checkpoint latest=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --checkpoint previous=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-checkpoint-scoreboard-smoke-2026-05-09
```

Expected artifacts:

- `summary.json`
- `episodes.jsonl`

The summary includes config, optional `eval_split`, checkpoint specs,
`pair_groups`, `scoreboard_rows`, and exact artifact paths. The 2026-05-09
smoke reused the same checkpoint under two labels, so its learned-vs-learned row
is only a same-checkpoint plumbing check, not a meaningful policy result.

Distinct-checkpoint monitor smoke:

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

Current distinct-checkpoint result: all three learned checkpoints beat random
more often than not, but none beat `track_ball`. Treat `track_ball` as the main
current gate.

Periodic-checkpoint scoreboard smoke:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor \
  --checkpoint epoch_1=artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000001/checkpoint.npz \
  --checkpoint epoch_3=artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000003/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09
```

Current periodic-checkpoint smoke result: the command loaded epoch 1 and epoch
3 checkpoints and wrote scoreboard rows. Epoch 3 beat random 2/4 versus epoch
1's 1/4, but both checkpoints won 0/4 against `track_ball`. Epoch 1 versus
epoch 3 tied 2/4 to 2/4. This proves periodic-checkpoint scoreboard plumbing
only.

Selection record from an existing scoreboard summary:

```sh
uv run python scripts/select_dummy_pong_checkpoint.py \
  --summary artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09/summary.json \
  --output-dir artifacts/local/dummy-pong-checkpoint-selection-record-smoke-2026-05-09
```

Expected artifact:

- `selection_record.json`

The selector does not rerun eval. It ranks learned checkpoint candidates by win
rate against `track_ball`, then win rate against `random_uniform`, then lower
truncation rate across those required baseline rows. It records the selected
label/path/policy id when available, metric details, split metadata, the source
summary path/hash, and the candidate rows used. It refuses to overwrite an
existing record unless `--force` is passed.

This is selection-split bookkeeping only. It does not prove final Pong
checkpoint quality; use a separate heldout confirmation before making quality
claims.

Selection record from the longer imitation v0 scoreboard:

```sh
uv run python scripts/select_dummy_pong_checkpoint.py \
  --summary artifacts/local/dummy-pong-imitation-periodic-v0-e1000-scoreboard-selection-2026-05-09/summary.json \
  --output-dir artifacts/local/dummy-pong-imitation-v0-e1000-selection-record-2026-05-09
```

Current result: selected `epoch1000`. All candidates had 0 wins against
`track_ball`, so the selector used the next rule and chose the checkpoint with
the best `random_uniform` win rate, 42/64. This is still not a quality claim.

Heldout check for that selected checkpoint:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 211 \
  --split-id dummy_pong_imitation_v0_heldout \
  --split-role heldout \
  --checkpoint selected_best=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --checkpoint previous=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-imitation-v0-e1000-scoreboard-heldout-2026-05-09
```

Current heldout result: selected beat previous 50/64 and lost less often to
`track_ball`, but still won 0/64 against `track_ball`. This is useful pressure,
not success.

## Modal Dummy Pong Checkpoint Scoreboard Attempt

Runs the same dummy Pong checkpoint scoreboard as one CPU Modal job. It writes
`summary.json` and `episodes.jsonl` to the `curvyzero-runs` Volume in the
run/attempt eval layout, then returns compact refs and hashes.

Use a Volume ref for checkpoints when running remotely:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints latest=ref:training/dummy-pong/RUN_ID/attempts/ATTEMPT_ID/train/checkpoint.npz \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor
```

Multiple checkpoints are comma-separated:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints latest=ref:training/dummy-pong/RUN_ID/attempts/ATTEMPT_ID/train/checkpoint.npz,previous=ref:training/dummy-pong/RUN_ID/attempts/PREVIOUS_ATTEMPT_ID/train/checkpoint.npz \
  --episodes 2 \
  --seed 0
```

Expected Volume path:

```text
training/dummy-pong/<run_id>/attempts/<attempt_id>/eval/checkpoint-scoreboard
```

Expected files:

- `training/dummy-pong/<run_id>/run.json`
- `training/dummy-pong/<run_id>/latest_attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/attempt.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/eval/checkpoint-scoreboard/summary.json`
- `training/dummy-pong/<run_id>/attempts/<attempt_id>/eval/checkpoint-scoreboard/episodes.jsonl`

Checkpoint specs can be `LABEL=ref:VOLUME_REF`, `LABEL=volume:VOLUME_REF`, or
an absolute path inside the remote container. This proves remote eval and
artifact plumbing only. It does not prove policy quality.

Remote smoke with manually uploaded checkpoints:

```sh
modal volume put curvyzero-runs \
  artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz \
  training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-000250/checkpoint.npz
```

```sh
modal volume put curvyzero-runs \
  artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-001000/checkpoint.npz
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints latest=ref:training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-001000/checkpoint.npz,previous=ref:training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-000250/checkpoint.npz \
  --episodes 2 \
  --seed 19 \
  --split-id dummy_pong_modal_smoke_v0 \
  --split-role monitor \
  --run-id modal-pong-scoreboard-smoke-20260509 \
  --attempt-id attempt-000001
```

Current remote smoke result: the CPU Modal job read both checkpoints from
`curvyzero-runs`, wrote `summary.json` and `episodes.jsonl` back to the Volume,
and returned file refs and hashes. Latest beat random 3/4, latest versus
`track_ball` truncated 4/4, and this proves remote eval plumbing only.

## Dummy Line Duel Training

Runs the first two-player training loop. The code uses MuZero-like names, but
the learner is a shared tabular NumPy dummy learner, not real MuZero.

```sh
uv run python scripts/run_dummy_line_duel_train.py \
  --iterations 3 \
  --episodes-per-iter 10 \
  --seed 0 \
  --output-dir artifacts/local/dummy_line_duel_smoke
```

Expected files:

- `summary.json`
- `checkpoint.npz`
- `iteration_metrics.jsonl`
- `replay_rows.jsonl`

Use this command to check simultaneous two-player stepping, ego-perspective
replay rows, and shared-policy dummy updates before adding real multiplayer
training.

## Dummy Line Duel Eval Matrix

Runs the first fixed-policy EVAL2 baseline matrix for Tiny Line Duel. This
evaluates `random_uniform`, `random_sticky`, and `one_step_safe` across
same-policy mirrors and paired seat assignments for mixed policy pairs. It does
not compute ratings. Optional checkpoint policies are evaluated only against
the fixed baselines.

```sh
uv run python scripts/run_dummy_line_duel_eval.py \
  --episodes 20 \
  --seed 0 \
  --split-id dummy_line_duel_monitor_v0 \
  --split-role monitor \
  --output-dir artifacts/local/dummy_line_duel_eval_smoke
```

Expected artifacts:

- `summary.json`
- `episodes.jsonl`

Use this command to check fixed-seed baseline signal, paired-seat behavior,
action histograms, reward/win tables, truncations, and death-cause counts before
adding learned checkpoint opponents.

The summary includes `eval_split`, `pair_groups`, `paired_seat_group_count`, and
seat deltas. Use `pair_groups` as the multiplayer claim unit.

Optional learned-checkpoint comparison:

```sh
uv run python scripts/run_dummy_line_duel_eval.py \
  --episodes 5 \
  --seed 123 \
  --checkpoint-policy learned:artifacts/local/dummy_line_duel_smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy_line_duel_checkpoint_eval_smoke
```

## CurvyTron Trainer Observation Learnability Diagnostic

Runs simple fixed policies directly against the 1v1 no-bonus trainer observation
surface. Use this before scaling RL when the question is whether the current
CurvyTron observation/action interface has any exploitable survival signal.

```sh
uv run python -m curvyzero.training.curvytron_baseline_eval \
  --episodes 64 \
  --batch-size 64 \
  --max-steps 2048 \
  --policy-kinds straight,left,right,random_legal,wall_avoid,ray_clearance \
  --observation-summary-dir artifacts/local/curvytron_learnability_probe \
  --observation-summary-limit 8
```

Expected signal: `ray_clearance` or at least `wall_avoid` should survive longer
than `random_legal` and straight/no-op on the same seed set. The JSON summary
prints mean/median/max survival steps, terminal reasons, winners, and action
histograms. Optional observation summaries are saved as per-policy JSON files
with ray-channel minima, forward rays, scalars, and legal action names.

## Modal Dummy Line Duel Ephemeral

Runs the same dummy line-duel scaffold as one coarse CPU Modal job. No Volume is
attached yet; artifacts are remote ephemeral files and the returned JSON summary
is the durable local signal.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_line_duel \
  --iterations 1 \
  --episodes-per-iter 2 \
  --seed 0 \
  --eval-episodes 2
```

Use this only to confirm the Modal source-copy/import/summary path.

## Next Useful Smokes

- Repeat the checkpoint eval commands after trainer changes to see whether
  learned checkpoints beat both scripted floors and
  `untrained_model_same_planner`.
- For dummy survival, prefer checkpoint-sweep artifacts over final-checkpoint
  artifacts until the degradation issue is understood.
- For Pong, use the checkpoint scoreboard for latest/older/best checkpoint
  comparisons once multiple real policy checkpoints exist. Angle-control and
  contact-outcome probes should stay diagnostic, not policy improvement claims.
- Use `scripts/inspect_dummy_pong_artifacts.py` before training from a Pong
  artifact so reward holes, shape mismatches, and truncation-only data are
  obvious.
- For Pong reward data, use random opponents before adding any shaping or
  biased starts.
- Modal follow-up: add resume only after the run/attempt wrapper has been used
  on a few tiny jobs.
- Mctx: keep `docs/working/mctx_spike_checklist.md` as fallback/search
  benchmark context. It is third behind the two LightZero custom-env smokes.

Planning docs:

- `docs/working/two_player_toy_game_plan.md`
- `docs/research/simple_training_environment_options.md`
- `docs/research/training_evaluation.md`
- `docs/working/mctx_spike_checklist.md`
