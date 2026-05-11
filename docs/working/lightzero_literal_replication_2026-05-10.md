# LightZero Literal Replication - 2026-05-10

## Target

Earlier weak-signal target:

- run id: `lz-visual-pong-exact-installed-0.2.0-s0`
- old attempt: `train-faithful-short-installed-0.2.0-s0-32768-relpath`
- setup: official/control LightZero Atari Pong, seed `0`,
  `max_env_step_override=32768`, installed `LightZero==0.2.0`, stock-ish
  config surface, no custom dummy Pong.
- earlier signal: same-run `iteration_9092` improved versus same-run
  `iteration_0` under strict eval, but remained weak.

Claim: the new launch is a literal repeat of the earlier weak-signal
official/control Pong lane, with only the checkpoint cadence intentionally
changed to `1000` for denser same-run eval.

Non-claim: this launch does not yet prove learning, exact upstream
reproduction, solved Pong, or CurvyTron readiness; no result claim is valid
until same-run strict eval compares later checkpoints against this run's
`iteration_0`.

## Launch

H100 was not used. The wrapper has partial H100 constants, but the train guard
and local entrypoint still accept only `cpu` and `gpu-l4-t4`, so
compute-scaling support is not landed enough for a clear H100 command.

Command:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4 --seed 0 --run-id lz-visual-pong-exact-installed-0.2.0-s0 --attempt-id train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath --progress-interval-sec 300 --max-env-step-override 32768 --save-ckpt-after-iter-override 1000
```

Launch metadata:

- attempt id:
  `train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath`
- Modal app id: `ap-VjL82fpC66tEWd5PzSKT6j`
- Modal function call id: `fc-01KR7PPMJEHYFWRRHRMVRJVTDH`
- compute: `gpu-l4-t4`
- detached pattern: `modal run --detach` plus wrapper `Function.spawn`
- progress ref:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/train/progress/latest.json`

CLI spawned payload:

```json
{
  "attempt_id": "train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath",
  "compute": "gpu-l4-t4",
  "function_call_id": "fc-01KR7PPMJEHYFWRRHRMVRJVTDH",
  "mode": "train",
  "progress_ref": "training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/train/progress/latest.json",
  "run_id": "lz-visual-pong-exact-installed-0.2.0-s0",
  "schema": "curvyzero_lightzero_pong_background_launch/v1",
  "seed": 0,
  "status": "spawned"
}
```

## Eval Trigger

Use strict checkpoint loads with no fallback. Compare only within this attempt:

1. Eval `iteration_0`.
2. Eval each visible frequent checkpoint:
   `iteration_1000`, `iteration_2000`, `iteration_3000`, and so on.
3. Eval the final checkpoint, whatever its final iteration is, against
   same-run `iteration_0`.
4. If compute/time allows, also run a long-cap eval for the best later
   checkpoint and `iteration_0` to check whether the 512-step cap hides a
   return difference.

Report fields must start with:

1. `steps_survived`
2. `delta_steps_survived` versus same-run `iteration_0`
3. raw/manual return
4. stock return if available
5. action histogram / collapse
6. checkpoint path and attempt id

## First Checkpoint Eval

Strict stock-evaluator eval was run for `iteration_0` versus
`iteration_1000` from the literal repeat A attempt, using the 2048-step cap
first because survival is the priority signal.

Eval id:
`repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0`.

Remote manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012006Z.json`.

Local manifest:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012006Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`
(`survival_fraction=0.371582`), did not survive to cap, manual return `-21`,
stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict load
true, fallback false, stock/manual action prefix match true, dominant action
`2` share `1.0`, entropy `0`, verdict `collapsed_action`.
`iteration_1000` survived `761/2048`, `delta_steps_survived=0`,
manual return `-21` (`delta_return=0`), stock return `-21`
(`delta_stock_return=0`), `21` nonzero rewards, `0` positive rewards, strict
load true, fallback false, stock/manual action prefix match true, dominant
action `1` share `1.0`, entropy `0`, verdict `collapsed_action`.

Claim: repeat A `iteration_1000` strict-loads with no fallback, but does not
improve long-horizon survival, manual return, stock return, reward count, or
positive rewards versus same-run `iteration_0`.

Non-claim: this is not evidence of learning progress, solved Pong, or
CurvyTron readiness. It is also not evidence that a later repeat A checkpoint
cannot improve; it only rejects a first-checkpoint improvement claim at
`iteration_1000`.

## Second Checkpoint Eval

Strict stock-evaluator eval was run for `iteration_0` versus
`iteration_2000` from the literal repeat A attempt, using the same 2048-step
survival-first cap and strict no-fallback checkpoint load.

Eval app: `ap-oNRa4eVcbLiIIg4h7qG8nw`.

Eval id:
`repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0`.

Remote manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012735Z.json`.

Local manifest:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012735Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`
(`survival_fraction=0.371582`), did not survive to cap, manual return `-21`,
stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict load
true, fallback false, stock/manual action prefix match true, dominant action
`2` share `1.0`, entropy `0`, verdict `collapsed_action`.
`iteration_2000` survived `761/2048`, `delta_steps_survived=0`,
manual return `-21` (`delta_return=0`), stock return `-21`
(`delta_stock_return=0`), `21` nonzero rewards, `0` positive rewards, strict
load true, fallback false, stock/manual action prefix match true, dominant
action `0` share `0.582129`, action histogram `{"0": 443, "2": 318}`,
entropy `0.980449`, verdict `negative_return`.

Claim: repeat A `iteration_2000` strict-loads with no fallback and no stock
evaluator mismatch, but does not improve long-horizon survival, manual return,
stock return, reward count, or positive rewards versus same-run `iteration_0`.

Non-claim: this is not a solved-Pong or learning-progress result. The policy is
less fully collapsed than `iteration_0` or `iteration_1000`, but the gameplay
metrics that matter for this survival-first control remain flat.

## Compact Latest Curve Eval

Repeat A advanced beyond the original prompt. A checkpoint poll found
`iteration_0`, `iteration_1000`, every 1000-step checkpoint through
`iteration_9000`, and latest visible `iteration_9559`. The compact 2048-step
strict no-fallback stock-evaluator curve evaluated requested checkpoints
`iteration_0`, `iteration_1000`, `iteration_5000`, `iteration_9000`, and
latest `iteration_9559`. An in-flight manifest also contains `iteration_2000`;
the compact claim below reports only the requested curve.

Eval apps: `ap-hKrDI5A93bIG3vGorxuEkT` for `0/1000/2000/9559` and
`ap-qHvUiewiwJ9nc4kJf7Z8kf` for `5000/9000`.

Remote manifests:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration0-1000-2000-9559-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T034656Z.json`
and
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration5000-9000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T035104Z.json`.

Local manifests:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-1000-2000-9559-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T034656Z.json`
and
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration5000-9000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T035104Z.json`.
Local combined summary:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-compact-curve-stockish2048-stockeval-s0-summary.tsv`.

| checkpoint | steps survived / 2048 | delta vs iteration_0 | saturation | manual return | stock return | positive rewards | negative/nonzero rewards | action usage / collapse |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | `761/2048` | `0` | not saturated | `-21` | `-21` | `0` | `21/21` | action `2` only, collapsed |
| `iteration_1000` | `761/2048` | `0` | not saturated | `-21` | `-21` | `0` | `21/21` | action `1` only, collapsed |
| `iteration_5000` | `761/2048` | `0` | not saturated | `-21` | `-21` | `0` | `21/21` | all six actions; dominant `2` share `0.529566` |
| `iteration_9000` | `761/2048` | `0` | not saturated | `-21` | `-21` | `0` | `21/21` | all six actions; dominant `2` share `0.574244` |
| `iteration_9559` | `973/2048` | `+212` | not saturated | `-21` | `-17` | `0` | `21/21` | all six actions; dominant `1` share `0.34224` |

All compact rows strict-loaded and used no model fallback. `iteration_0` and
`iteration_1000` had stock/manual action-prefix match true; `iteration_5000`,
`iteration_9000`, and `iteration_9559` had stock/manual action-prefix match
false, so the stock return should be read as the stock-evaluator score and the
manual action histogram as separate policy telemetry.

Claim: latest repeat A checkpoint `iteration_9559` improves survival by
`+212` manual eval steps over same-run `iteration_0` and improves stock return
from `-21` to `-17` under strict no-fallback 2048-cap eval. Earlier compact
curve checkpoints `iteration_1000`, `iteration_5000`, and `iteration_9000`
remain flat on survival, manual return, stock return, and positive rewards.

Non-claim: this is not solved Pong and not a clean return-learning result. The
manual return remains `-21`, positive rewards remain `0`, no checkpoint
survives to the 2048 cap, and the later non-collapsed checkpoints have
manual/stock action-prefix mismatch.

## Notes

Do not compare this repeat's later checkpoints to the old relpath run except
as context. The admissible training-signal claim is same-run later checkpoint
versus same-run `iteration_0`.
