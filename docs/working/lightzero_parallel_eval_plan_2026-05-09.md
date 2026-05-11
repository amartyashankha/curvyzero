# LightZero Parallel Eval Plan - 2026-05-09

Scope: official LightZero Atari Pong eval only. No training and no pytest were
run for this plan.

## Goal

Stop waiting on serial checkpoint evals. The official Atari eval wrapper now
supports a Modal `map` fan-out over checkpoint refs while preserving the old
single-checkpoint path.

## Wrapper Contract

Use `curvyzero.infra.modal.lightzero_pong_eval_smoke`.

Compute selection is explicit:

- `--compute cpu` is the default and preserves the original CPU-only eval path.
- `--compute gpu-l4-t4` routes checkpoint workers to a separate Modal function
  with `gpu=["L4", "T4"]`, patches LightZero with `policy.cuda=true`, and
  records `config.runtime_compute` plus the compiled policy `cuda/device`.

Both compute modes use the same eval implementation after Modal placement.
There is no automatic CPU fallback for GPU eval; failures should stay visible
in the per-checkpoint JSON and manifest.

Single eval remains compatible:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --checkpoint-ref training/lightzero-official-visual-pong/<RUN_ID>/checkpoints/lightzero/iteration_16.pth.tar \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --max-eval-steps 256 \
  --step-detail-limit 8 \
  --no-allow-model-fallback
```

Single eval on cheap GPU:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --compute gpu-l4-t4 \
  --checkpoint-ref training/lightzero-official-visual-pong/<RUN_ID>/attempts/<ATTEMPT_ID>/train/lightzero_exp/ckpt/iteration_<N>.pth.tar \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --max-eval-steps 512 \
  --max-episode-steps 512 \
  --step-detail-limit 8 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

Parallel eval uses one of two selectors:

- `--checkpoint-refs` for an explicit comma-separated list.
- `--checkpoint-ref-template` plus `--selected-iterations` for iteration
  curves. The template uses `{iteration}`.

## Low-Detail First Pass

Run cheap checkpoint triage first. This keeps details tiny and writes one JSON
artifact per checkpoint plus a manifest.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --parallel \
  --eval-pass low \
  --eval-id curve-low-sim10 \
  --checkpoint-ref-template 'training/lightzero-official-visual-pong/<RUN_ID>/checkpoints/lightzero/iteration_{iteration}.pth.tar' \
  --selected-iterations 0,4,8,16 \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --max-env-step 4096 \
  --max-train-iter 32 \
  --collector-env-num 2 \
  --evaluator-env-num 1 \
  --num-simulations 10 \
  --batch-size 32 \
  --update-per-collect 2 \
  --max-episode-steps 512 \
  --game-segment-length 64 \
  --no-allow-model-fallback
```

Default low-detail settings are `64` eval steps and `2` detailed step records.
Override with `--low-detail-max-eval-steps` and
`--low-detail-step-detail-limit` if needed.

For real checkpoint curves, do not use the tiny smoke cap. Use a higher step
cap and score the selected checkpoints in one parallel call. The default curve
shape should be one command over the checkpoint set, for example
`iteration_0,4,8,16,max`, with enough eval steps to see serve/opening rewards.

The same command can infer the template from a representative checkpoint ref if
the ref ends in `iteration_<n>.pth.tar`:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --parallel \
  --eval-pass low \
  --eval-id curve-low-sim10 \
  --checkpoint-ref training/lightzero-official-visual-pong/<RUN_ID>/checkpoints/lightzero/iteration_16.pth.tar \
  --selected-iterations 0,4,8,16 \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --num-simulations 10 \
  --max-episode-steps 512 \
  --no-allow-model-fallback
```

## Optional High-Detail Rerun

Only rerun promising checkpoints. Keep the selected list small.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --parallel \
  --eval-pass high \
  --eval-id curve-high-sim10 \
  --checkpoint-refs 'training/lightzero-official-visual-pong/<RUN_ID>/checkpoints/lightzero/iteration_8.pth.tar,training/lightzero-official-visual-pong/<RUN_ID>/checkpoints/lightzero/iteration_16.pth.tar' \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --max-env-step 4096 \
  --max-train-iter 32 \
  --collector-env-num 2 \
  --evaluator-env-num 1 \
  --num-simulations 10 \
  --batch-size 32 \
  --update-per-collect 2 \
  --max-episode-steps 512 \
  --game-segment-length 64 \
  --no-allow-model-fallback
```

Default high-detail settings are `512` eval steps and `8` detailed step
records. Override with `--high-detail-max-eval-steps` and
`--high-detail-step-detail-limit`.

## Artifact Naming And Retention

Per-checkpoint artifacts are written under:

```text
training/lightzero-official-visual-pong/<RUN_ID>/attempts/<ATTEMPT_ID>/eval/<EVAL_ID>/<CHECKPOINT_LABEL>_<PASS>_steps<STEPS>_seed<SEED>/lightzero_visual_pong_eval_<CHECKPOINT_LABEL>_<PASS>_steps<STEPS>_seed<SEED>_<UTC>.json
```

The manifest is written under the same eval root:

```text
training/lightzero-official-visual-pong/<RUN_ID>/attempts/<ATTEMPT_ID>/eval/<EVAL_ID>/manifest_<PASS>_steps<STEPS>_seed<SEED>_<UTC>.json
```

Retention: keep all per-checkpoint JSON artifacts and the manifest until the
run is manually archived. Low-detail and high-detail outputs intentionally do
not overwrite each other.

The manifest contains both full `results` and a compact `table`. The table is
the first readout surface and includes:

- `checkpoint`
- `eval_cap_steps`
- `strict_load`
- `steps_survived`
- `return`
- `nonzero_reward_count`
- `positive_reward_count`
- `action_histogram`
- `dominant_action`
- `dominant_action_share`
- `action_entropy`
- `stock_manual_match`
- `fallback_used`
- `elapsed_sec`
- `verdict`

The verdict is only a first-pass label. `collapsed_action` means one action
accounted for at least 98% of the manual eval actions. It is not a learning
claim.

## Readout Rule

Use the manifest first. Promote checkpoints to high-detail only if low-detail
shows cleaner no-fallback behavior, broader action support, a less-negative
return, or any nonzero positive Pong reward. Flat collapsed checkpoints should
not get expensive reruns.

## Live 32768 GPU Eval Command

The active `32768` faithful-short train is app `ap-xiGLACKHPZLvL1eYgygqvm`,
run `lz-visual-pong-exact-installed-0.2.0-s0`, attempt
`train-faithful-short-installed-0.2.0-s0-32768-relpath`. It writes checkpoints
under the attempt train root while training continues. Once the desired
periodic files exist, score only selected `iteration_*.pth.tar` files, not
`ckpt_best`, with:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --compute gpu-l4-t4 \
  --parallel \
  --eval-pass low \
  --eval-id faithful-short-32768-periodic-low-stockeval-gpu-s0 \
  --checkpoint-refs 'training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt/iteration_0.pth.tar,training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt/iteration_<FINAL_OR_MID>.pth.tar' \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --low-detail-max-eval-steps 512 \
  --max-episode-steps 512 \
  --low-detail-step-detail-limit 8 \
  --max-env-step 200000 \
  --max-train-iter 1 \
  --collector-env-num 8 \
  --evaluator-env-num 3 \
  --num-simulations 50 \
  --batch-size 256 \
  --update-per-collect 1 \
  --game-segment-length 400 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

Replace only `<FINAL_OR_MID>` with an existing live checkpoint iteration. Keep
the eval cap and `max_episode_steps` matched at `512`. This command intentionally
uses stock-ish eval knobs and strict/no-fallback behavior; GPU only changes
where the policy model runs.

## Local Manifest Summary Helper

Use `scripts/summarize_lightzero_pong_eval_manifest.py` after the Modal eval
finishes. It is read-only: it consumes the manifest JSON and prints one row per
checkpoint with checkpoint ref, strict load, fallback, manual/stock match,
steps survived, manual return, stock return when present, reward counts,
dominant action/share, entropy, and verdict.

Score rule: when stock `MuZeroEvaluator` runs cleanly with `strict_load=true`,
`fallback_used=false`, and stock-ish eval config, use `stock_return` as the
main score. Use the manual 512-step telemetry for survival, reward-count,
action-collapse, and parity diagnostics. A manual/stock mismatch is an
eval-harness parity warning, not a checkpoint-load failure. Beware wrapper
defaults: `DEFAULT_NUM_SIMULATIONS=2` and tiny env counts are useful for code
path checks, but they are not the final stock-ish scoring config.

Fetch the manifest from the Modal Volume, replacing `<MANIFEST_REF>` with the
manifest ref printed by the eval command:

```sh
mkdir -p artifacts/local/lightzero-eval-manifests
uv run --extra modal modal volume get curvyzero-runs '<MANIFEST_REF>' artifacts/local/lightzero-eval-manifests/
```

Then summarize the fetched manifest:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  artifacts/local/lightzero-eval-manifests/<MANIFEST_FILE>.json
```

For spreadsheet or doc paste, write TSV:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --format tsv \
  --output artifacts/local/lightzero-eval-manifests/<EVAL_ID>.tsv \
  artifacts/local/lightzero-eval-manifests/<MANIFEST_FILE>.json
```

If the Modal Volume is locally mounted at `/runs`, the helper can read a ref
directly:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  'ref:<MANIFEST_REF>'
```

## Completed Faithful-Short Post-Train Eval

This note records the corrected eval for the completed faithful-short installed
package train. The earlier low eval is not valid because manual
`max_episode_steps` stayed at `64` while stock used `512`.

Target train identity:

```text
run_id: lz-visual-pong-exact-installed-0.2.0-s0
attempt_id: train-faithful-short-installed-0.2.0-s0-8192-relpath
run_kind: faithful-short
```

The corrected command used `iteration_0` and final `iteration_3697`, strict
load, no fallback, manual cap `512`, and stock evaluator score read:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --parallel \
  --eval-pass custom \
  --eval-id faithful-short-periodic-custom512-stockeval-s0-8192-relpath \
  --checkpoint-refs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/lightzero_exp/ckpt/iteration_0.pth.tar,training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/lightzero_exp/ckpt/iteration_3697.pth.tar \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-8192-relpath \
  --max-eval-steps 512 \
  --max-episode-steps 512 \
  --step-detail-limit 8 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

Result:

| Checkpoint | Manual return | Stock return | Steps | Positive rewards | Manual/stock match | Fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `iteration_0` | `-13` | `-13` | `512` | `0` | `false` | `false` |
| `iteration_3697` | `-13` | `-8` | `512` | `0` | `false` | `false` |

Read: this eval used the stock `MuZeroEvaluator` code path, but still used tiny
wrapper defaults such as `num_simulations=2`. Treat it as useful evidence, not
the final stock-ish score. The code-path stock returns were `iteration_0=-13`
and `iteration_3697=-8`; manual telemetry still showed both at `-13`, no
positive rewards, and no survival-cap improvement. Manual/stock first-prefix
match was false, so report the mismatch as an eval-harness parity warning and
keep the manual telemetry side-by-side; do not call it a checkpoint-load
failure or a clean learning claim.

Corrected stock-ish eval completed:

```text
app: ap-81xAvfiyvnU8flV3eElPSH
eval_id: faithful-short-periodic-stockish512-stockeval-s0-8192-relpath
checkpoints: iteration_0, iteration_3697
strict/no fallback: true
num_simulations: 50
evaluator_env_num: 3
collector_env_num: 8
batch_size: 256
game_segment_length: 400
max_env_step: 200000
max_train_iter: 1
update_per_collect: 1
cuda: false
eval cap: 512
max_episode_steps: 512
```

The eval wrapper ran on CPU and compiled a policy config, so
`max_train_iter=1` and `update_per_collect=1` were part of this eval wrapper
configuration. Result:

| Checkpoint | Manual return | Stock return | Steps | Nonzero rewards | Positive rewards | Dominant action/share | Entropy | Manual/stock match | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| `iteration_0` | `-13` | `-13` | `512` | `13` | `0` | `0` / `0.521484` | `0.805545` | `false` | `false` |
| `iteration_3697` | `-5` | `-8` | `512` | `7` | `1` | `0` / `0.714844` | `0.644585` | `false` | `false` |

Read: this is the first weak signal that final is less bad than init under
stock-ish eval, but it is one seed/two checkpoints and manual-stock mismatch
remains. It is not solved Pong and not exact reproduction.

Keep `--max-episode-steps` equal to the eval step cap in future evals. If the
manual cap is `512`, the episode cap must also be `512`.

The first readout is the manifest `table`, not individual JSON spelunking. In
particular check: `checkpoint`, `checkpoint_ref`, `strict_load`,
`eval_cap_steps`, `steps_survived`, `return`,
`stock_return`, `nonzero_reward_count`, `positive_reward_count`,
`action_histogram`, `dominant_action`, `dominant_action_share`,
`action_entropy`, `stock_manual_match`, `fallback_used`, `elapsed_sec`, and
`verdict`.

To inspect or archive a future manifest, fetch and summarize the printed
manifest ref:

```sh
mkdir -p artifacts/local/lightzero-eval-manifests
uv run --extra modal modal volume get curvyzero-runs '<MANIFEST_REF>' artifacts/local/lightzero-eval-manifests/
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  artifacts/local/lightzero-eval-manifests/<MANIFEST_FILE>.json
```

## Smoke Evidence

Tiny parallel eval smoke passed on 2026-05-09 against run
`lz-visual-pong-8192-sim25-s0`.

Command shape used:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --parallel \
  --eval-pass low \
  --low-detail-max-eval-steps 32 \
  --selected-iterations 0,100 \
  --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/iteration_100.pth.tar \
  --run-id lz-visual-pong-8192-sim25-s0 \
  --attempt-id train-8192-sim25-b64-env4-auto \
  --eval-id checkpoint_curve_parallel_smoke \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

Modal app: `ap-GroNH8bnBAadark30VLY51`.

Manifest:

```text
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve_parallel_smoke/manifest_low_steps32_seed0_20260509T202709Z.json
sha256 52f8393deca27361143020d5108a4205d0fc1306eb74e302a92b935cd0eb6314
```

Observed tiny table signal:

| Checkpoint | Action histogram | Return | Meaning |
| --- | --- | ---: | --- |
| `iteration_0` | `{3: 32}` | `0` | Workflow proof only; 32 steps is too short for quality. |
| `iteration_100` | `{0: 32}` | `0` | Workflow proof only; 32 steps is too short for quality. |

This proves the fan-out/manifest path works for selected checkpoints, including
stock evaluator capture. It is not evidence that either checkpoint is good.

A second tiny smoke after the manifest-table patch also passed:

```text
Modal app: ap-gfDhVFnKEAbniTrVuf9E2c
manifest: training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve_parallel_table_smoke/manifest_low_steps4_seed0_20260509T202922Z.json
```

Its table rows were:

| Checkpoint | Steps survived | Return | Action histogram | Meaning |
| --- | ---: | ---: | --- | --- |
| `iteration_0` | `4` | `0` | `{3: 4}` | Table-shape proof only. |
| `iteration_100` | `4` | `0` | `{0: 4}` | Table-shape proof only. |

Four steps is deliberately too short to score learning. The only claim is that
future real evals can now read the compact manifest table first.
