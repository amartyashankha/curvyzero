# LightZero Exact Reproduction Decision - 2026-05-09

Scope: exact installed-package LightZero Atari Pong MuZero replication, plus a
clearly labeled faithful-short rehearsal path. Exact means `LightZero==0.2.0`,
`zoo.atari.config.atari_muzero_config`, `PongNoFrameskip-v4`, stock config
values, stock `lzero.entry.train_muzero`, `max_env_step=200000`, and the stock
trainer/evaluator path. The only exact-mode config mutation is Modal
artifact/output placement. Train mode sets `exp_name` to a relative Volume ref
and runs from `/runs`, so DI-engine `./` checkpoint paths stay inside the Modal
Volume.

Faithful-short is not exact reproduction. It keeps the same installed package
config and the same `exp_name` artifact patch, but additionally passes a shorter
`max_env_step_override` to `train_muzero`.

No pytest was run. No exact full training was started. One faithful-short GPU
train has now completed.

## Immediate Verdict

Run dry now: yes. It is safe, short, and already passed once.

Start exact `--mode train --compute gpu-l4-t4` immediately after dry: not yet.
Dry passing proves the installed package surface and the single exact-mode
`exp_name` patch. The faithful-short run proves the relpath wrapper can train
and keep checkpoints under the intended Volume root. It does not prove learning
or exact reproduction. The first full exact run should still be watched because
it can run for hours and write many checkpoints. Training on CPU is blocked by
the wrapper before `train_muzero` is called.

First command I recommend is the dry validation, even if repeated before a
go/no-go:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode dry \
  --compute cpu \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id dry-exact-installed-0.2.0-stock-surface
```

If this dry command passes, the next action should be an explicit human
approval for a managed full train, not an automatic launch.

## Decision

Feasible now, but not cheap enough to launch casually.

The exact installed `LightZero==0.2.0` Atari Pong replication is mechanically
ready after adding a separate exact wrapper:

```text
src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py
```

The wrapper imports the installed package config, deep-copies it, patches only
`main_config.exp_name` to a relative Modal Volume ref, changes the process
working directory to `/runs` in train mode, and then calls:

```python
lzero.entry.train_muzero([main_config, create_config], seed=0, max_env_step=200000)
```

It does not pass `max_train_iter`, does not cap episode length, does not reduce
collector/evaluator counts, does not reduce MCTS simulations, does not change
batch size, does not force checkpoint cadence, and does not change
`update_per_collect`.

If `--max-env-step-override` is provided, the wrapper labels the run
`faithful-short`, records `run_kind`, `actual_max_env_step`, and the extra
`train_muzero.max_env_step` patch in the result summary, and passes that shorter
value to `train_muzero`. This is a rehearsal of the installed package path, not
an exact reproduction claim.

## Dry Validation Run

Command run:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode dry --compute cpu --seed 0 --run-id lz-visual-pong-exact-installed-0.2.0-s0 --attempt-id dry-exact-installed-0.2.0-stock-surface
```

Second dry validation after the later wrapper hardening also passed:

```text
Modal app: ap-Xz1gqGamx5CX0tCZfKknk8
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-exact-installed-0.2.0-stock-surface-v2/dry_exact_summary.json
sha256: d57a1122ddd88279b743e01f4228591dba41af19cdeab491786982bcaf1e6813
```

Progress-argument dry validation after adding the watcher also passed:

```text
Modal app: ap-yVqnbnmm7WxdJJUtC5cEkK
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-exact-installed-0.2.0-progress-arg-smoke/dry_exact_summary.json
sha256: 5c9c15d72f2eb32fddf3f431194e4a5b3938f6bb3a1c76a8cee95234fb805a3e
progress: disabled in dry mode, interval recorded as 300 seconds
```

Dry faithful-short wrapper hardening smoke also passed. This validates the
short-run labeling and patch accounting without claiming exact reproduction:

```text
Modal app: ap-ODFjMCC7IzU5sceWaxSL5l
run_id: lz-visual-pong-exact-installed-0.2.0-s0
attempt_id: dry-faithful-short-installed-0.2.0-8192
ok: true
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-faithful-short-installed-0.2.0-8192/dry_exact_summary.json
sha256: 21b535e6bfff46cb2ede82d465c115f479bf286d7a3c72cd46d8705eda9d6c77
run_kind: faithful-short
is_exact_reproduction: false
stock_max_env_step: 200000
actual_max_env_step: 8192
only extra patch: train_muzero.max_env_step 200000 -> 8192
stock surface: 50 sims, 8 collectors, 3 evaluators, batch 256, segment 400,
  no episode caps, cuda true
```

Modal app:

```text
ap-GwF0NH5Pm3iIEry5PQtuLO
```

Result:

```text
ok: true
call_policy: dry_import_config_patch_exp_name_only_no_env_no_train
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-exact-installed-0.2.0-stock-surface/dry_exact_summary.json
sha256: 3af537d06b27d8c0088f9077e3f83b7744e76880e45fe9f9728d13ecff800fe0
remote_elapsed_sec: 8.377718
```

Validated package surface:

| Field | Value |
| --- | --- |
| `LightZero` | `0.2.0` |
| `DI-engine` | `0.5.3` |
| `torch` | `2.11.0` |
| `gym` | `0.25.1` |
| `gymnasium` | `0.28.0` |
| `ale-py` | `0.8.1` |
| `AutoROM` | `0.6.1` |
| module | `zoo.atari.config.atari_muzero_config` |
| trainer | `lzero.entry.train_muzero` |
| env id | `PongNoFrameskip-v4` |
| env type | `atari_lightzero` |
| env manager | `subprocess` |
| policy type | `muzero` |
| model type | `conv` |
| observation shape | `[4, 64, 64]` |
| action space | `6` |
| collector envs / episodes | `8` / `8` |
| evaluator envs / episodes | `3` / `3` |
| MCTS simulations | `50` |
| batch size | `256` |
| `update_per_collect` | `None` |
| `replay_ratio` | `0.25` |
| `game_segment_length` | `400` |
| `eval_freq` | `2000` |
| CUDA | `True` |
| learning rate | `0.2` |
| target update freq | `100` |
| replay buffer size | `1000000` |
| stock max env step | `200000` |
| stock episode caps | unset |
| stock `save_ckpt_after_iter` | unset |

Only patch observed:

```text
exp_name:
  old: data_muzero/Pong_atari_stack4_muzero_ns50_upcNone-rr0.25_seed0
  new: /runs/training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-exact-installed-0.2.0-stock-surface/train/lightzero_exp
```

Local syntax check:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py
```

## Exact Full Train Command

Do not run this without an explicit go-ahead:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-exact-installed-0.2.0-stock-surface \
  --progress-interval-sec 300
```

That command leaves `--max-env-step-override` unset. It uses the installed
package's `max_env_step=200000` and stock config values, with only the
`exp_name` artifact patch. Its summary should report `run_kind: exact`,
`is_exact_reproduction: true`, `actual_max_env_step: 200000`, and
`extra_patch: null`.

CPU train is intentionally rejected:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute cpu
```

That returns a failed summary with
`call_policy: blocked_cpu_train_before_train_muzero`; it does not call
`train_muzero`.

CPU train guard smoke, after the wrapper hardening, failed by design before
`train_muzero`:

```text
Modal app: ap-OaX6ZKqwp3JwMHzwXEsKCy
attempt_id: train-cpu-guard-installed-0.2.0
ok: false
call_policy: blocked_cpu_train_before_train_muzero
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-cpu-guard-installed-0.2.0/train/summary.json
sha256: 168722d09056fe705208a5d857f21a7b9278061c6624c1fc50d7b0a032b236b8
```

## Faithful-Short Rehearsal

Use this only to rehearse the installed package path at smaller cost. It is not
exact reproduction.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-8192 \
  --progress-interval-sec 300 \
  --max-env-step-override 8192
```

The result summary should say:

- `run_kind: faithful-short`
- `is_exact_reproduction: false`
- `stock_max_env_step: 200000`
- `actual_max_env_step: 8192`
- `extra_patch.path: train_muzero.max_env_step`

The stock installed LightZero config remains unchanged except for `exp_name`.
The additional difference is the trainer `max_env_step` argument.

The first GPU faithful-short train was launched and then killed after
`Training Iteration 400` because checkpoints were logging under `.//runs/...`,
while progress snapshots scanning `/runs/...` saw no checkpoints. Make no
quality or eval claim from this interrupted run:

```text
Modal app: ap-lDxY0C7O0GGDwu3jjxuMaI
attempt_id: train-faithful-short-installed-0.2.0-s0-8192
max_env_step_override: 8192
progress_interval_sec: 120
last observed trainer log: Training Iteration 400
reason killed: artifact path mismatch, .//runs/... logs vs /runs/... progress scan
```

Artifact-root safety note: the running faithful-short train log reported
`ckpt_best` under `.//runs/training/.../lightzero_exp/ckpt/ckpt_best.pth.tar`,
while the 120s progress snapshot scanning the intended absolute
`/runs/training/.../lightzero_exp` saw only config files and zero checkpoints.
The likely cause is LightZero or DI-engine prepending `./` to an absolute
`exp_name`, turning it into a cwd-relative `./runs/...` path. The wrapper now
uses a relative `exp_name` and runs train mode from `/runs`, so future
checkpoint saves should land inside the Modal Volume even if DI-engine prepends
`./`. Progress snapshots and final summaries also scan both the configured root
and the plausible cwd-relative alternate root, and record alternate roots with
checkpoint counts.

Relative-exp dry faithful-short smoke passed after the wrapper patch:

```text
Modal app: ap-BJS7mWOdsqqzbA6DafF82z
attempt_id: dry-faithful-short-installed-0.2.0-8192-relative-exp
ok: true
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-faithful-short-installed-0.2.0-8192-relative-exp/dry_exact_summary.json
sha256: bbff8da5746cb8999f03e8e28f4e26ba5ba05b361dd8e17be3d49de7afd791b8
```

Relpath GPU faithful-short run completed:

```text
Modal app: ap-ipdfYJmWQitQtIBxrKU2E9
attempt_id: train-faithful-short-installed-0.2.0-s0-8192-relpath
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/summary.json
sha256: c97dc26094462ec17d1dd970370d86e392433a8059aed9b1eaea1e5614ed2a06
train_ok: true
gpu: L4
torch_cuda: true
actual_max_env_step: 8192
collector_env_steps: 14791
remote_elapsed_sec: about 1326
checkpoint_bytes: 256,613,692
checkpoints: ckpt_best, iteration_0, iteration_3697
alternate_roots: none
```

This was faithful-short, not exact reproduction. The only intended patches were
`exp_name` for Modal artifact placement and `train_muzero.max_env_step 200000
-> 8192`. The collector overshot the cap in one batch and reached `14791` env
steps.

Corrected post-train eval:

```text
Modal app: ap-ov622Yu6wEnN74V2Laf8HG
manifest: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/eval/faithful-short-periodic-custom512-stockeval-s0-8192-relpath/manifest_custom_steps512_seed0_20260509T232832Z.json
checkpoints: iteration_0, iteration_3697
strict_load: true for both
fallback: false for both
manual_eval_steps: 512 for both
return: -13 for both
nonzero_rewards: 13 for both
positive_rewards: 0 for both
iteration_0 action_entropy: 0.963778
iteration_0 dominant_action: 2, share 0.345703
iteration_3697 action_entropy: 0.755427
iteration_3697 dominant_action: 0, share 0.535156
```

Manual/stock first-prefix match was false, so do not use stock/manual parity as
clean evidence for this eval. Use the manual 512-step telemetry as the score
read. The earlier low eval was invalid and misleading because manual
`max_episode_steps` stayed `64` while stock used `512`.

Plain result: no learning signal. Final changed the action mix, but did not
improve return, survival cap, or positive rewards versus `iteration_0`.

## Expected Time, Cost, And Artifact Size

Blunt estimate: budget `6-18h` wall time and `$9-$30` all-in on an L4/T4 class
run. I would not call it insane, but I would not start it as a background
"quick check."

Reasoning:

- Prior non-exact 8192/sim25 run on L4 took `428.997874s` inside
  `train_muzero`, `919.7004s` remote elapsed, with `4` collectors,
  `batch_size=64`, `game_segment_length=128`, and stock
  `update_per_collect=None` restored.
- Exact installed-package scale is `200000` env steps, `50` simulations,
  `8` collectors, `batch_size=256`, `game_segment_length=400`, and
  `3` evaluator envs. Straight env-step/search scaling alone is about `49x`
  the 8192/sim25 train-call work before batch/segment/eval overhead.
- Modal pricing checked on 2026-05-09 lists L4 at `$0.000222/sec`, T4 at
  `$0.000164/sec`, CPU at `$0.0000131/core/sec`, and memory at
  `$0.00000222/GiB/sec`.
- The exact GPU wrapper reserves `8` CPU cores and `32GiB` memory. That is
  roughly `$1.43/h` on L4 or `$1.22/h` on T4 before incidental image/build
  overhead.

Artifact estimate:

- Each prior Atari Pong checkpoint from the 8192/sim25 lane was about `95-100MB`.
- The bad 8192/sim25 artifact explosion was caused by our smoke wrapper forcing
  `save_ckpt_after_iter=1`; that mirrored `934` checkpoint files and about
  `90GB`.
- Exact mode does not force checkpoint-every-iteration, and dry validation shows
  stock `save_ckpt_after_iter` is unset. That lowers expected artifact size, but
  the resolved DI-engine learner hook cadence is still not visible from config
  import alone.
- Practical planning envelope: `5-30GB` if stock checkpoint cadence is sane;
  `100GB+` if the resolved learner hook saves far more often than expected;
  multi-TB is the theoretical disaster case if it degenerates into every-learner
  iteration over a full `200000` env-step run.

Pricing source: https://modal.com/pricing

## Required Wrapper Cap And Timeout

Training semantics cap: none. Exact mode must not add LightZero caps.

Execution guardrails:

- GPU function timeout: `18h`.
- GPU resource: `["L4", "T4"]`.
- CPU/memory: `8` cores, `32GiB`.
- Dry CPU timeout: `8m`.
- Train CPU guard: fail clearly before `train_muzero`; use GPU for train mode.
- Full train should be manually killed if startup stalls, CUDA is unavailable,
  or no useful train/eval logging/checkpoint progress appears in the first
  `30-45m`.
- Before launch, decide who is watching the first `30-45m` and who is allowed
  to kill the run if Volume growth is obviously wrong.
- Do not pass `max_train_iter`. The stock trainer signature has a huge default;
  exact package replication should rely on `max_env_step=200000`.

The old tiny smoke wrapper remains unsuitable for exact mode because it patches
training semantics. Use only the exact wrapper above.

Current wrapper behavior: train mode writes `exp_name` as a relative ref under
the Modal Volume, changes cwd to `/runs`, and writes periodic progress JSON under
`attempts/<ATTEMPT_ID>/train/progress/`. The progress watcher scans file count,
total bytes, checkpoint count, checkpoint bytes, newest checkpoints, and largest
files across both the configured `exp_name` root and the plausible
cwd-relative alternate root caused by a `./` prefix. Final summaries also carry
an `artifact_scan` with per-root scans and `alternate_artifact_roots`; report
those alternate roots and checkpoint counts in any completion handoff. The
watcher still does not capture stdout tails or prune artifacts automatically.
That is enough to make a supervised exact run less blind; it is still not a
reason to fire-and-forget an 18h job.

## Checkpoint Retention Plan

Use direct Volume-backed `exp_name`; do not rely on end-of-run checkpoint copy.
The 8192/sim25 run showed that checkpoint volume can explode if checkpoint
cadence is patched. Exact mode therefore keeps stock checkpoint cadence
untouched.

Retention plan for a full exact run:

1. During run, keep the entire stock `lightzero_exp` directory on the Modal
   Volume so interruption does not lose artifacts.
2. During run, read `train/progress/latest.json` to watch file count,
   checkpoint count, checkpoint bytes, and newest checkpoint names.
3. Immediately after completion, write a manifest with package versions, stock
   surface, command, Modal app/task id, checkpoint listing, file sizes, and
   hashes for retained checkpoints.
4. Long-term keep `ckpt_best`, the first checkpoint, the final checkpoint,
   every stock periodic checkpoint needed to reconstruct the learning curve,
   TensorBoard/log files, and the exact config/manifests.
5. After hashes and selected checkpoint refs are recorded, prune excess
   transient checkpoints from the Volume. This is artifact retention only; it
   must happen after training and must not alter LightZero training behavior.

If the stock run still produces tens or hundreds of GB, stop treating exact
replication as a casual control and promote it to a managed training job with
explicit storage budget.

## Bottom Line

Yes, exact installed `LightZero==0.2.0` Atari Pong replication is feasible now.
The dry validation passed. The full train is operationally risky but bounded:
likely single-digit to low-double-digit hours and tens of dollars, not hundreds.

Do not start the full train just because dry passes. Start it only after someone
explicitly accepts the runtime, storage, monitoring, and retention plan.
