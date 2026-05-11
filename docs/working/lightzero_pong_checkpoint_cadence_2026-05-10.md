# LightZero Pong Checkpoint Cadence

Question: what checkpoint cadence should we expect from the current detached
frequent-checkpoint run?

## Current Read

The first frequent-checkpoint control run was launched around
`2026-05-10T00:24:21Z`:

- run id: `lz-visual-pong-exact-installed-0.2.0-s0`
- attempt id: `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath`
- Modal app: `ap-7Wd3QGsjT0RucAc2DC26mS`
- launch state: Modal reported `Running (1/1 containers active)` after worker
  assignment, then stopped at `2026-05-10T00:29:55Z` before `iteration_1000`.

This run kept stock reward/control semantics. The only intended training
surface changes were the faithful-short env-step cap, relative artifact root,
and the checkpoint cadence override.

That attempt is no longer the active attempt. After the wrapper was patched so
`--mode train` uses `train_fn.spawn(...)`, the current clean post-patch launch
is:

- Current app: `ap-h7bpMSwDDW6f0eIOv7Cfdl`
- Function call id: `fc-01KR7NR7XQNTQ4GF75EQ3Z6TE0`
- Current attempt:
  `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-spawn-relpath`
- Run id: `lz-visual-pong-exact-installed-0.2.0-s0`
- Claim: this is a clean robust background-launch validation using
  `modal run --detach`, with frequent checkpoints every `1000` learner
  iterations.
- Non-claim: there is no policy-quality result yet; wait for strict same-run
  eval of `iteration_1000+` against this run's `iteration_0`.
- Expected checkpoints: `iteration_0`, then every `1000` learner iterations,
  plus the final after-run checkpoint.
- Progress ref:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-spawn-relpath/train/progress/latest.json`.

Killed attempt status:

- Claim: the attempt stopped because the Modal launch client disconnected, so
  it did not run long enough to test or use the `iteration_1000` checkpoint
  cadence.
- Non-claim: this is not a learning/no-learning result, not a failed strict eval,
  and not evidence that LightZero or the `save_ckpt_after_iter=1000` knob is
  broken.
- Progress artifact:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath/train/progress/progress_failed_20260510T002955Z.json`.
- Progress readout: `phase=failed`, `train_elapsed_sec=340.736058747`,
  `actual_save_ckpt_after_iter=1000`, `checkpoint_count=2`, newest checkpoints
  `ckpt/iteration_0.pth.tar` and `ckpt/ckpt_best.pth.tar`.
- App log cause: `Stopping app - local client disconnected. Use modal run
  --detach to keep apps running even if your local client disconnects.`
- Learner log reached `Training Iteration 400` at `2026-05-10 00:29:50`; no
  `iteration_1000.pth.tar`, final checkpoint, train summary, or eval manifest
  exists for this attempt.

Launch note: first app `ap-15WcpcwIbSoNSZkx0kUxJf` failed before training with
`KeyError('learn')` in the new cadence override path. The wrapper was patched to
create `policy.learn.learner.hook` when setting `save_ckpt_after_iter`, matching
the older tiny-train override pattern; the replacement app above started
training, then stopped early when the local client disconnected.

## Exact Knob

The prior stock-cadence active run's saved `total_config.py` had:

```python
policy.learn.learner.hook.save_ckpt_after_iter = 10000
policy.learn.learner.hook.save_ckpt_after_run = True
```

The new run was launched with:

```bash
--save-ckpt-after-iter-override 1000
```

The wrapper patches:

```python
main_config.policy.learn.learner.hook.save_ckpt_after_iter
```

The saved config value was confirmed as:

```python
policy.learn.learner.hook.save_ckpt_after_iter = 1000
policy.learn.learner.hook.save_ckpt_after_run = True
```

## Observed Pace From Prior 32768 Run

Learner log:

- `iteration_0` at `2026-05-09 23:41:37`
- `iteration_5100` at `2026-05-10 00:05:56`

Collector log:

- first collect end: `14822` total envsteps at `23:50:36`
- second collect end: `21813` total envsteps at `00:00:51`

The run cap is `max_env_step_override=32768`. With `replay_ratio=0.25`, the
final learner iteration should be roughly one quarter of final envsteps, plus
batch/rounding effects. The prior 8192 run ended at `14791` envsteps and saved
final `iteration_3697`, matching the same ratio.

## Estimate

For the current detached `ckpt1000` spawn run, expect normal checkpoints near:

- `iteration_0` near startup
- `iteration_1000`
- `iteration_2000`
- `iteration_3000`
- continuing every `1000` learner iterations while training runs
- a final after-run checkpoint

Most likely final useful checkpoint: roughly `iteration_8200` to
`iteration_9500`, after the collector overshoots the `32768` env-step cap.
This should give several same-run checkpoints before the final, unlike the
stock `10000` cadence.

Progress snapshots for the current detached attempt are expected every `120`
seconds at:

```text
training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-spawn-relpath/train/progress/latest.json
```

Quality rule: do not infer learning from trainer logs alone. Eval later
same-run checkpoints with strict load, no fallback, and compare against the
same-run `iteration_0` baseline. For the killed attempt, there is no later
same-run checkpoint to evaluate. For the current detached spawn attempt, wait
for a normal later checkpoint before making any quality claim.

## Future Change

For future short rehearsal runs, keep using:

```bash
--save-ckpt-after-iter-override
```

When unset, stock behavior is unchanged. When set, the wrapper changes only
checkpoint cadence. Suggested short-run value remains `1000` or `2000`; stock
`10000` is too sparse for `8192` and `32768` env-step rehearsals.

Also launch long-enough Modal training jobs detached, or otherwise keep the
local client alive. The killed attempt was stopped by Modal when the client
disconnected before the first periodic checkpoint. The current replacement
attempt was launched with `--detach` after the wrapper patch moved background
training under `train_fn.spawn(...)`.

Launched command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath \
  --progress-interval-sec 120 \
  --max-env-step-override 32768 \
  --save-ckpt-after-iter-override 1000
```

Detached replacement command shape:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath \
  --progress-interval-sec 120 \
  --max-env-step-override 32768 \
  --save-ckpt-after-iter-override 1000
```

Current clean post-patch command shape:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-spawn-relpath \
  --progress-interval-sec 120 \
  --max-env-step-override 32768 \
  --save-ckpt-after-iter-override 1000
```

No pytest was run.
