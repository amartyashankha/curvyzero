# LightZero Shaped Queue Status - 2026-05-10

Purpose: side-lane readiness and eval launch status for survival-shaped Pong
runs only. These runs add `0.001` reward per non-terminal step during training.
They are telemetry for survival-shaped behavior, not stock/control Pong proof.

## 14:36 Long Shaped Side-Lane Read

Two already-active long shaped H100 runs were evaluated with the unshaped stock
evaluator. No new shaped training run was launched because these long shaped
streams were already active and useful: s32 had checkpoints through
`iteration_54181`; s33 had checkpoints through `iteration_51963`.

This is side-lane telemetry only. It must not be mixed into stock/control Pong
proof.

| lane/run | checkpoint | mean `stock_steps_survived` | delta vs `iteration_0` | `stock_return_mean` | `stock_positive_reward_count_sum` | dominant action summary/action entropy | read |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| shaped `lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16` | `iteration_0` | `976` | `0` | `-20` | `1` | stock action `0 @ 0.638`, entropy `0.722` | broader than one-action baseline, already one positive reward |
| shaped `lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16` | `iteration_54000` | `2048` | `+1072` | `3` | `9` | stock action `0 @ 0.437`, entropy `0.873` | reached 2048 cap; broad actions across all six; useful shaped telemetry |
| shaped `lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16` | `iteration_0` | `762` | `0` | `-21` | `0` | stock action `1 @ 1.000`, entropy `0.000` | collapsed one-action baseline |
| shaped `lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16` | `iteration_51000` | `1282` | `+520` | `-20` | `1` | stock action `2 @ 0.392`, entropy `0.894` | no longer one-action; broad actions across all six |

Exact eval commands:

```sh
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16 --attempt-id train-survival-shaped-step0p001-s32-199000-ckpt1000-spawn-h100cpu16-relpath --eval-id shaped-long-s32-0-54000-stock2048-seed32 --compute gpu-l4-t4-cpu40 --eval-pass custom --seed 32 --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step 199000 --selected-iterations 0,54000 --group-size 1 --max-parallel-launches 2 --execute
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16 --attempt-id train-survival-shaped-step0p001-s33-199000-ckpt1000-spawn-h100cpu16-relpath --eval-id shaped-long-s33-0-51000-stock2048-seed33 --compute gpu-l4-t4-cpu40 --eval-pass custom --seed 33 --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step 199000 --selected-iterations 0,51000 --group-size 1 --max-parallel-launches 2 --execute
```

Artifact refs:

- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16/attempts/train-survival-shaped-step0p001-s32-199000-ckpt1000-spawn-h100cpu16-relpath/eval/shaped-long-s32-0-54000-stock2048-seed32/manifest_custom_steps2048_seeds32_20260510T143339Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16/attempts/train-survival-shaped-step0p001-s32-199000-ckpt1000-spawn-h100cpu16-relpath/eval/shaped-long-s32-0-54000-stock2048-seed32/manifest_custom_steps2048_seeds32_20260510T143548Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16/attempts/train-survival-shaped-step0p001-s33-199000-ckpt1000-spawn-h100cpu16-relpath/eval/shaped-long-s33-0-51000-stock2048-seed33/manifest_custom_steps2048_seeds33_20260510T143321Z.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16/attempts/train-survival-shaped-step0p001-s33-199000-ckpt1000-spawn-h100cpu16-relpath/eval/shaped-long-s33-0-51000-stock2048-seed33/manifest_custom_steps2048_seeds33_20260510T143425Z.json`

Local fetch roots used for summary:

- `/private/tmp/shaped-long-s32-0-54000-stock2048-seed32`
- `/private/tmp/shaped-long-s33-0-51000-stock2048-seed33`

Summary commands:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --survival-curve --survival-aggregate --format tsv '/private/tmp/shaped-long-s32-0-54000-stock2048-seed32/**/*.json'
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --survival-curve --survival-aggregate --format tsv '/private/tmp/shaped-long-s33-0-51000-stock2048-seed33/**/*.json'
```

Eval contract used for launch attempts:

- strict no-fallback checkpoint load
- stock evaluator enabled
- `--compute gpu-l4-t4-cpu8`
- `--eval-pass custom`
- `--max-eval-steps 2048`
- `--max-episode-steps 2048`
- `--step-detail-limit 8`
- `--group-size 1`
- `--max-parallel-launches 3`
- stock steps survived should be reported first by the harvester

Follow-up correction: later CPU64 eval attempts failed because Modal caps this
workspace's function CPU at 40 cores. Future retry commands should use
`--compute gpu-l4-t4-cpu40` with `--max-parallel-launches 64`; the CPU8 line
above describes the successful early shaped evals.

No eval results ledger was edited.

## Launch Outcome

Successful eval roots launched and completed in this pass:

| Seed | Eval id | Selected iterations at launch | Status |
| ---: | --- | --- | --- |
| 30 | `shaped-s30-0-1000-stock2048-seed30` | `0,1000` | completed; strict artifacts written |
| 31 | `shaped-s31-0-1000-stock2048-seed31` | `0,1000` | completed; strict artifacts written |
| 32 | `shaped-s32-0-1000-stock2048-seed32` | `0,1000` | completed; strict artifacts written |

Ready but blocked by launch-time Modal validation:

| Seed | Eval id attempted | Selected iterations attempted | Status |
| ---: | --- | --- | --- |
| 33 | `shaped-s33-0-1000-stock2048-seed33` | `0,1000` | blocked before artifact: `Function CPU request out of bounds. Must be between 0.125 and 40 cores.` |
| 34 | `shaped-s34-0-1000-stock2048-seed34` | `0,1000` | blocked before artifact: same Modal CPU validation error |
| 35 | `shaped-s35-0-1000-stock2048-seed35` | `0,1000` | blocked before artifact: same Modal CPU validation error |
| 36 | `shaped-s36-0-1000-stock2048-seed36` | `0,1000` | blocked before artifact: same Modal CPU validation error |
| 37 | `shaped-s37-0-1000-stock2048-seed37` | `0,1000` | blocked before artifact: same Modal CPU validation error |

Note: the successful s30-s32 launch poll saw only `0,1000` as the selected
ready range. A later end-of-pass poll saw newer checkpoints on several runs,
including `iteration_5000`; those are documented below for the next harvester
or retry pass.

## End-of-Pass Checkpoint Visibility

| Seed | Run id | Attempt id | Max env | Visible iteration range at final poll | 5000 visible | Eval state |
| ---: | --- | --- | ---: | --- | --- | --- |
| 30 | `lz-visual-pong-survival-shaped-step0p001-s30-65k-l4cpu16` | `train-survival-shaped-step0p001-s30-65536-ckpt1000-spawn-l4cpu16-relpath` | 65536 | `0,1000,2000,3000,4000,5000` | yes | `0,1000` eval completed |
| 31 | `lz-visual-pong-survival-shaped-step0p001-s31-65k-l4cpu16` | `train-survival-shaped-step0p001-s31-65536-ckpt1000-spawn-l4cpu16-relpath` | 65536 | `0,1000,2000,3000,4000,5000,6000` | yes | `0,1000` eval completed |
| 32 | `lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16` | `train-survival-shaped-step0p001-s32-199000-ckpt1000-spawn-h100cpu16-relpath` | 199000 | `0,1000,2000,3000,4000,5000,6000,7000` | yes | `0,1000` eval completed |
| 33 | `lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16` | `train-survival-shaped-step0p001-s33-199000-ckpt1000-spawn-h100cpu16-relpath` | 199000 | `0,1000,2000,3000,4000,5000,6000,7000` | yes | ready; eval launch blocked |
| 34 | `lz-visual-pong-survival-shaped-step0p001-s34-65k-l4cpu16` | `train-survival-shaped-step0p001-s34-65536-ckpt1000-waveB-l4cpu16-relpath` | 65536 | `0,1000,2000,3000` | no | ready; eval launch blocked |
| 35 | `lz-visual-pong-survival-shaped-step0p001-s35-65k-l4cpu16` | `train-survival-shaped-step0p001-s35-65536-ckpt1000-waveB-l4cpu16-relpath` | 65536 | `0,1000` | no | ready; eval launch blocked |
| 36 | `lz-visual-pong-survival-shaped-step0p001-s36-65k-l4cpu16` | `train-survival-shaped-step0p001-s36-65536-ckpt1000-waveB-l4cpu16-relpath` | 65536 | `0,1000,2000,3000` | no | ready; eval launch blocked |
| 37 | `lz-visual-pong-survival-shaped-step0p001-s37-65k-l4cpu16` | `train-survival-shaped-step0p001-s37-65536-ckpt1000-waveB-l4cpu16-relpath` | 65536 | `0,1000,2000,3000` | no | ready; eval launch blocked |

No shaped run in this watch set is still pending on the original
`iteration_0` plus `iteration_1000` readiness gate as of the final poll. The
remaining blocker is eval launch infrastructure, not checkpoint readiness.

## Fetch Commands For Completed Evals

```sh
uv run --extra modal modal volume get curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s30-65k-l4cpu16/attempts/train-survival-shaped-step0p001-s30-65536-ckpt1000-spawn-l4cpu16-relpath/eval/shaped-s30-0-1000-stock2048-seed30 artifacts/local/lightzero-eval-manifests/shaped-s30-0-1000-stock2048-seed30 --force
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --baseline-deltas --format tsv artifacts/local/lightzero-eval-manifests/shaped-s30-0-1000-stock2048-seed30
```

```sh
uv run --extra modal modal volume get curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s31-65k-l4cpu16/attempts/train-survival-shaped-step0p001-s31-65536-ckpt1000-spawn-l4cpu16-relpath/eval/shaped-s31-0-1000-stock2048-seed31 artifacts/local/lightzero-eval-manifests/shaped-s31-0-1000-stock2048-seed31 --force
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --baseline-deltas --format tsv artifacts/local/lightzero-eval-manifests/shaped-s31-0-1000-stock2048-seed31
```

```sh
uv run --extra modal modal volume get curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16/attempts/train-survival-shaped-step0p001-s32-199000-ckpt1000-spawn-h100cpu16-relpath/eval/shaped-s32-0-1000-stock2048-seed32 artifacts/local/lightzero-eval-manifests/shaped-s32-0-1000-stock2048-seed32 --force
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --baseline-deltas --format tsv artifacts/local/lightzero-eval-manifests/shaped-s32-0-1000-stock2048-seed32
```

## Retry Commands After Modal CPU Validation Is Cleared

Use `0,1000,5000` where `iteration_5000` is already visible at the final poll;
use `0,1000` otherwise.

```sh
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16 --attempt-id train-survival-shaped-step0p001-s33-199000-ckpt1000-spawn-h100cpu16-relpath --eval-id shaped-s33-0-1000-5000-stock2048-seed33 --compute gpu-l4-t4-cpu40 --eval-pass custom --seed 33 --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step 199000 --selected-iterations 0,1000,5000 --group-size 1 --max-parallel-launches 64 --execute
```

```sh
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-survival-shaped-step0p001-s34-65k-l4cpu16 --attempt-id train-survival-shaped-step0p001-s34-65536-ckpt1000-waveB-l4cpu16-relpath --eval-id shaped-s34-0-1000-stock2048-seed34 --compute gpu-l4-t4-cpu40 --eval-pass custom --seed 34 --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step 65536 --selected-iterations 0,1000 --group-size 1 --max-parallel-launches 64 --execute
```

```sh
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-survival-shaped-step0p001-s35-65k-l4cpu16 --attempt-id train-survival-shaped-step0p001-s35-65536-ckpt1000-waveB-l4cpu16-relpath --eval-id shaped-s35-0-1000-stock2048-seed35 --compute gpu-l4-t4-cpu40 --eval-pass custom --seed 35 --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step 65536 --selected-iterations 0,1000 --group-size 1 --max-parallel-launches 64 --execute
```

```sh
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-survival-shaped-step0p001-s36-65k-l4cpu16 --attempt-id train-survival-shaped-step0p001-s36-65536-ckpt1000-waveB-l4cpu16-relpath --eval-id shaped-s36-0-1000-stock2048-seed36 --compute gpu-l4-t4-cpu40 --eval-pass custom --seed 36 --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step 65536 --selected-iterations 0,1000 --group-size 1 --max-parallel-launches 64 --execute
```

```sh
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-survival-shaped-step0p001-s37-65k-l4cpu16 --attempt-id train-survival-shaped-step0p001-s37-65536-ckpt1000-waveB-l4cpu16-relpath --eval-id shaped-s37-0-1000-stock2048-seed37 --compute gpu-l4-t4-cpu40 --eval-pass custom --seed 37 --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step 65536 --selected-iterations 0,1000 --group-size 1 --max-parallel-launches 64 --execute
```
