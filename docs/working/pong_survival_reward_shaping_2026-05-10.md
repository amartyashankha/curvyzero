# Pong Survival Reward Shaping - 2026-05-10

Purpose: inspect the official LightZero Atari Pong path and add the simplest
survival-shaped training ablation without changing stock/control Pong.

## Decision

Feasible, with a narrow opt-in wrapper.

Worker inspection status: implemented, not blocked. The control path remains
stock by default, and the shaped path is available only when the caller passes a
positive `--survival-reward-per-step` and uses `survival-shaped` in both ids.

Stock/control path stays unchanged:

- env type: `atari_lightzero`
- env import: `zoo.atari.envs.atari_lightzero_env`
- reward: ALE/LightZero Atari Pong reward stream
- exact wrapper default: `survival_reward_per_step=0.0`

Shaped ablation path is separate:

- env type: `atari_lightzero_survival_shaped`
- env import: `curvyzero.training.lightzero_atari_survival_env`
- schema: `curvyzero_lightzero_atari_survival_step_reward/v1`
- reward change: add `survival_reward_per_step` on non-terminal env steps

This is not an exact reproduction and not a stock/control Pong result.

## Implementation

Added:

- `src/curvyzero/training/lightzero_atari_survival_env.py`

The new env subclasses LightZero's Atari env inside the Modal LightZero runtime
and returns a fresh `BaseEnvTimestep` with shaped reward. It also records
`curvyzero_reward_shaping` in `info` with base reward, applied bonus, shaped
reward, cumulative bonus, and schema id.

Patched:

- `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py`

The exact wrapper now accepts:

```text
--survival-reward-per-step FLOAT
```

Default is `0.0`, so all existing stock/control commands keep the stock env type
and pass the existing exact-surface validation.

When the value is positive, the wrapper:

- switches only the patched run config to `atari_lightzero_survival_shaped`;
- imports only `curvyzero.training.lightzero_atari_survival_env`;
- writes reward-shaping metadata into the summary and attempt manifest;
- marks `is_exact_reproduction=false`;
- requires both `run_id` and `attempt_id` to contain `survival-shaped`.

## Conservative Command Pattern

Use a tiny bonus first:

```bash
uv run --extra modal modal run --detach \
  -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-survival-shaped-step0.001-s0 \
  --attempt-id train-survival-shaped-step0.001-32768-ckpt1000-s0 \
  --progress-interval-sec 300 \
  --max-env-step-override 32768 \
  --save-ckpt-after-iter-override 1000 \
  --survival-reward-per-step 0.001
```

Report it as:

```text
survival-shaped official Atari Pong ablation, per-step bonus 0.001
```

Do not mix it into stock/control tables except as a clearly separate shaped row.
Evaluate shaped checkpoints with the existing unshaped eval wrapper and compare
against the same shaped run's `iteration_0`.

## Why This Shape

A separate env type is safer than monkeypatching the stock env because the
artifact surface shows the reward contract change. The id marker requirement is
a guardrail against accidentally launching shaped training under the stock run
name.

The bonus is per non-terminal step, not a timeout win. In these runs it is
bounded by episode termination and eval/train step limits, but it is not a final
reward design. If a future environment lets a policy stall forever, this reward
can teach stalling instead of winning. Treat it as a cheap signal probe only:
true return, stock return, positive rewards, timeout rate, action histogram,
entropy, and survival deltas must stay side by side.

## Verification

No pytest was run.

Compiled:

```bash
python -m py_compile \
  src/curvyzero/training/lightzero_atari_survival_env.py \
  src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py
```

Result: compile passed locally.

## Live Shaped Eval - H100 Seed 3

Claim: shaped H100 seed `3` checkpoints `iteration_0`, `iteration_1000`,
and newly visible latest `iteration_2000` strict-loaded with no model fallback
and ran the stock evaluator under a 2048-step cap.

Non-claim: this is not stock/control Pong and it does not show shaped learning
progress. Survival, manual return, stock return, reward counts, and positive
rewards were flat versus same-run `iteration_0`.

Launched through the live eval queue with per-checkpoint fan-out:

```bash
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-survival-shaped-step0p001-s3-h100 \
  --attempt-id train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100 \
  --eval-id shaped-h100-s3-iteration0-1000-stock2048-seed3 \
  --compute gpu-l4-t4 \
  --eval-pass custom \
  --seed 3 \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --max-env-step 32768 \
  --group-size 1 \
  --max-parallel-launches 4 \
  --execute
```

Queue behavior: by the corrected seed-3 launch, `iteration_2000` was visible,
so the pending compact wave was `0/1000/2000`.

Tooling note: the first launch exposed a queue-helper bug where `--seed` was
used for expected output-dir names but was not forwarded to the evaluator. The
helper was patched to pass `--seed` through, then the H100 wave was relaunched.
The accidental seed-0 artifacts remain in the same eval root but are excluded
from the seed-3 shaped result below.

Seed-3 manifests:

- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s3-h100/attempts/train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100/eval/shaped-h100-s3-iteration0-1000-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T040419Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s3-h100/attempts/train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100/eval/shaped-h100-s3-iteration0-1000-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T040430Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s3-h100/attempts/train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100/eval/shaped-h100-s3-iteration0-1000-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T040438Z.json`

Local seed-3-only manifest copy:
`artifacts/local/lightzero-eval-manifests/shaped-h100-s3-iteration0-1000-2000-stock2048-seed3-only/`.

Survival/return first:

| checkpoint | steps/cap | delta steps vs iteration_0 | manual return | stock return | positive rewards | nonzero rewards | strict load | fallback | dominant action | action entropy | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| `iteration_0` | 762/2048 | 0 | -21 | -21 | 0 | 21 | true | false | 1 @ 0.602362 | 0.688281 | manual_stock_mismatch |
| `iteration_1000` | 762/2048 | 0 | -21 | -21 | 0 | 21 | true | false | 0 @ 1.0 | 0.0 | collapsed_action |
| `iteration_2000` | 762/2048 | 0 | -21 | -21 | 0 | 21 | true | false | 3 @ 0.681102 | 0.535120 | manual_stock_mismatch |

The stock/manual mismatch verdict is a status flag from the eval summary; the
reported manual and stock returns are both `-21` in these rows. The useful
survival-first read is flat.

## Live Shaped Eval - H100 Seed 3 Compact Latest

Claim: shaped H100 seed `3` checkpoints `iteration_0`, `iteration_1000`,
`iteration_3000`, and latest-visible `iteration_5000` strict-loaded with no
model fallback and ran the stock evaluator under a 2048-step cap.

Non-claim: this is not stock/control Pong and does not show stock-return
learning. Latest `iteration_5000` improved manual survival and manual positive
rewards, but stock return and stock positive rewards stayed flat.

Pre-eval poll found checkpoints `0/1000/2000/3000/4000/5000`; `iteration_5000`
was latest-visible. The compact eval used `0/1000/3000/5000`,
per-checkpoint fan-out, and stock `update_per_collect=None` via
`--update-per-collect -1`:

```bash
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-survival-shaped-step0p001-s3-h100 \
  --attempt-id train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100 \
  --eval-id shaped-h100-s3-compact-0-1000-3000-5000-stock2048-seed3 \
  --compute gpu-l4-t4 \
  --eval-pass custom \
  --seed 3 \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --max-env-step 32768 \
  --update-per-collect -1 \
  --selected-iterations 0,1000,3000,5000 \
  --group-size 1 \
  --max-parallel-launches 4 \
  --execute
```

Manifests:

- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s3-h100/attempts/train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100/eval/shaped-h100-s3-compact-0-1000-3000-5000-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T041557Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s3-h100/attempts/train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100/eval/shaped-h100-s3-compact-0-1000-3000-5000-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T041558Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s3-h100/attempts/train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100/eval/shaped-h100-s3-compact-0-1000-3000-5000-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T041601Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s3-h100/attempts/train-survival-shaped-step0p001-s3-32768-ckpt1000-spawn-h100/eval/shaped-h100-s3-compact-0-1000-3000-5000-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T041638Z.json`

Local summary:
`artifacts/local/lightzero-eval-manifests/shaped-h100-s3-compact-0-1000-3000-5000-stock2048-seed3/summary_baseline_deltas.tsv`.

Simple facts:

| checkpoint | stock return | stock survival | manual survival | positive rewards | reached cap |
| --- | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | -21 | 758/2048 | 762/2048 | stock 0, manual 0 | no |
| `iteration_1000` | -21 | 758/2048 | 762/2048 | stock 0, manual 0 | no |
| `iteration_3000` | -21 | 758/2048 | 762/2048 | stock 0, manual 0 | no |
| `iteration_5000` | -21 | 898/2048 | 839/2048 | stock 0, manual 1 | no |

## Live Shaped Eval - L4 Seed 0

Claim: shaped L4 seed `0` checkpoints `iteration_0`, `iteration_1000`, and
latest-visible `iteration_3000` strict-loaded with no model fallback and ran the
stock evaluator under a 2048-step cap.

Non-claim: this is not stock/control Pong and does not show learning progress.
Stock return, manual survival, and positive rewards were flat versus same-run
`iteration_0`.

Pre-eval poll found only `iteration_0`, `iteration_1000`, and
`iteration_3000`; `iteration_3000` was therefore also latest-visible. The eval
used per-checkpoint fan-out:

```bash
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-survival-shaped-step0p001-s0 \
  --attempt-id train-survival-shaped-step0p001-s0-32768-ckpt1000-spawn-relpath \
  --eval-id shaped-l4-s0-0-1000-3000-latest-stock2048-seed0 \
  --compute gpu-l4-t4 \
  --eval-pass custom \
  --seed 0 \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --max-env-step 32768 \
  --group-size 1 \
  --max-parallel-launches 4 \
  --selected-iterations 0,1000,3000 \
  --execute
```

Manifests:

- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s0/attempts/train-survival-shaped-step0p001-s0-32768-ckpt1000-spawn-relpath/eval/shaped-l4-s0-0-1000-3000-latest-stock2048-seed0/manifest_custom_steps2048_seed0_20260510T041539Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s0/attempts/train-survival-shaped-step0p001-s0-32768-ckpt1000-spawn-relpath/eval/shaped-l4-s0-0-1000-3000-latest-stock2048-seed0/manifest_custom_steps2048_seed0_20260510T041542Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s0/attempts/train-survival-shaped-step0p001-s0-32768-ckpt1000-spawn-relpath/eval/shaped-l4-s0-0-1000-3000-latest-stock2048-seed0/manifest_custom_steps2048_seed0_20260510T041545Z.json`

Local summary:
`artifacts/local/lightzero-eval-manifests/shaped-l4-s0-0-1000-3000-latest-stock2048-seed0/summary_baseline_deltas.tsv`.

Simple facts:

| checkpoint | stock return | stock survival | manual survival | positive rewards | reached cap |
| --- | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | -21 | 760/2048 | 761/2048 | 0 stock, 0 manual | no |
| `iteration_1000` | -21 | 760/2048 | 761/2048 | 0 stock, 0 manual | no |
| `iteration_3000` | -21 | 821/2048 | 761/2048 | 0 stock, 0 manual | no |

`iteration_3000` had `stock_manual_match=false`, so read its stock survival
and manual survival separately.

## Live Shaped Eval - L4 Seed 1

Claim: shaped L4 seed `1` checkpoints `iteration_0`, `iteration_1000`,
`iteration_3000`, and newly visible `iteration_4000` strict-loaded with no
model fallback and ran the stock evaluator under a 2048-step cap.

Non-claim: this is not stock/control Pong and not confirmed return learning.
`iteration_4000` has a survival-first bump, but stock return stays `-21`.

Launched with per-checkpoint fan-out under eval id
`shaped-l4-s1-compact-0-1000-3000-stock2048-seed1`. The first wave selected
`0,1000,3000`; after `iteration_4000` appeared, the same queue command was
rerun with `--selected-iterations 0,1000,3000,4000`, and duplicate filtering
launched only `iteration_4000`.

Local manifests:
`artifacts/local/lightzero-eval-manifests/shaped-l4-s1-compact-0-1000-3000-stock2048-seed1/`.

Simple facts:

| checkpoint | stock return | stock survival | manual survival | positive rewards | reached cap |
| --- | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | -21 | 761/2048 | 763/2048 | stock 0, manual 0 | no |
| `iteration_1000` | -21 | 761/2048 | 763/2048 | stock 0, manual 0 | no |
| `iteration_3000` | -21 | 761/2048 | 763/2048 | stock 0, manual 0 | no |
| `iteration_4000` | -21 | 883/2048 | 842/2048 | stock 0, manual 1 | no |

Read: `iteration_4000` is a weak survival bump only. Manual return was `-20`,
but stock return stayed `-21`, so keep stock/control claims separate and
conservative.

## Live Shaped Eval - L4 Seed 2

Claim: shaped L4 seed `2` checkpoints `iteration_0` and `iteration_1000`
strict-loaded with no model fallback and ran the stock evaluator under a
2048-step cap.

Non-claim: this is not stock/control Pong and does not show learning progress.
Survival, stock return, manual return, and positive rewards were flat.

Launched with per-checkpoint fan-out under eval id
`shaped-l4-s2-iteration0-1000-stock2048-seed2`.

Local manifests:
`artifacts/local/lightzero-eval-manifests/shaped-l4-s2-iteration0-1000-stock2048-seed2/`.

Simple facts:

| checkpoint | stock return | stock survival | manual survival | positive rewards | reached cap |
| --- | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | -21 | 760/2048 | 762/2048 | stock 0, manual 0 | no |
| `iteration_1000` | -21 | 760/2048 | 762/2048 | stock 0, manual 0 | no |

## Queue Tooling Note

During this watch, `scripts/lightzero_live_eval_queue.py` was patched to pass
`--seed` through to `lightzero_pong_eval_smoke` and to support
`--selected-iterations`. The second flag lets shaped evals use compact waves
without evaluating every intermediate checkpoint.

## Next Steps

1. Run a dry shaped command first with the same ids and bonus to verify the
   patched surface reports `atari_lightzero_survival_shaped`.
2. If dry passes, launch one short shaped ablation.
3. Evaluate with unshaped strict no-fallback eval and report same-run
   `iteration_0` deltas.
4. Keep stock/control runs as the main official Pong lane.
