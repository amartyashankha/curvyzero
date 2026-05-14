# Granular Action Cadence E2E Smoke Plan

Date: 2026-05-13

## Goal

Run the smallest honest post-patch training-loop smoke for the trusted
CurvyZero stock LightZero lane.

The smoke must call real `lzero.entry.train_muzero`, instantiate the real
CurvyZero source-state visual env, step that env through LightZero collection,
and write real LightZero checkpoint artifacts.

## Best Command

Use the trusted trainer module in waited CPU Modal mode:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute cpu \
  --run-id curvytron-cadence-e2e-smoke-20260513 \
  --attempt-id train-smoke-001 \
  --max-train-iter 1 \
  --max-env-step 64 \
  --source-max-steps 64 \
  --collector-env-num 1 \
  --n-episode 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 1 \
  --batch-size 4 \
  --lightzero-eval-freq 0 \
  --save-ckpt-after-iter 1 \
  --env-telemetry-stride 1 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --wait-for-train \
  --output-detail compact
```

Why this command:

- It uses the active trusted entrypoint,
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`.
- `--compute cpu` avoids a GPU allocation for this gate.
- `--wait-for-train` makes the command return the actual training result
  instead of only a background function id.
- `--max-train-iter 1`, `--max-env-step 64`, one collector env, one episode,
  one simulation, and batch size 4 keep the run small.
- Background eval and GIF work are disabled so the smoke only checks the
  training loop.
- The existing refactor log records this same shape as a passing tiny Modal
  train smoke with `ok=true` and `called_train_muzero=true`.

## Expected Pass Signals

The returned compact JSON should show:

```text
ok: true
called_train_muzero: true
trainer_entrypoint: lzero.entry.train_muzero
mode: train
compute: cpu
command.env_variant: source_state_fixed_opponent
command.decision_source_frames: 1
command.policy_action_repeat_min: 1
command.policy_action_repeat_max: 1
command.policy_action_repeat_extra_probability: 0.0
problems: []
```

It should also report non-empty action telemetry:

```text
action_observability.row_count > 0
```

## Expected Artifacts

The run should write under the Modal Volume task root:

```text
training/lightzero-curvytron-visual-survival/curvytron-cadence-e2e-smoke-20260513/
```

Expected attempt files:

```text
attempts/train-smoke-001/config.json
attempts/train-smoke-001/command.json
attempts/train-smoke-001/train/summary.json
attempts/train-smoke-001/train/env_steps.jsonl
attempts/train-smoke-001/train/action_observability.json
attempts/train-smoke-001/train/target_audit.json
attempts/train-smoke-001/train/lightzero_artifacts_manifest.json
attempts/train-smoke-001/train/stdout_tail.txt
attempts/train-smoke-001/train/stderr_tail.txt
attempts/train-smoke-001/train/lightzero_exp*/ckpt/iteration_*.pth.tar
```

Expected mirrored checkpoint files:

```text
checkpoints/lightzero/iteration_*.pth.tar
```

The previous tiny train pattern produced `iteration_0.pth.tar`,
`iteration_1.pth.tar`, `iteration_2.pth.tar`, and `ckpt_best.pth.tar`.
For this smoke, the exact iteration count can vary with LightZero internals,
but at least one `iteration_*.pth.tar` must exist and be mirrored.

## Local Tests Are Not Enough

The focused local tests are useful guards, but they are not the final smoke:

```sh
uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q
```

Reason: the stock-entrypoint regression test fakes
`lzero.entry.train_muzero`. It proves local wiring, not a real LightZero
collector/replay/learner loop.

The boundary test
`test_stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action`
does instantiate and step the registered env, but it still does not call stock
`train_muzero`.

## Risks

- Modal image startup may dominate runtime even though the training work is
  tiny.
- CPU LightZero collection can still take longer than a unit test; this is a
  real training-loop smoke, not a local fast test.
- `--mode train` can auto-resume if the same `run-id` already has checkpoints.
  Use a fresh `run-id` for a clean post-patch smoke.
- This is a fixed-opponent stock-loop control. It is not current-policy
  two-seat self-play.
- A pass proves the cadence patch did not break the stock training loop. It
  does not prove policy quality or long-run learning.
