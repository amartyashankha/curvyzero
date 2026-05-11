# 2026-05-09 Modal LightZero Exact Faithful-Short 8192 Relpath

## Question

Can the exact installed-package wrapper run a short LightZero Atari Pong train
on Modal with correct artifact roots, sane checkpoint output, and a readable
512-step eval?

## Setup

Scope: installed `LightZero==0.2.0` official Atari Pong only.

This is faithful-short, not exact reproduction. It uses the installed package
config and stock trainer path, but changes two things:

- `exp_name` is patched so Modal artifacts land under the intended Volume root.
- `train_muzero.max_env_step` is shortened from `200000` to `8192`.

Train app: `ap-ipdfYJmWQitQtIBxrKU2E9`

Run:
`lz-visual-pong-exact-installed-0.2.0-s0`

Attempt:
`train-faithful-short-installed-0.2.0-s0-8192-relpath`

## Command

Train command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-8192-relpath \
  --max-env-step-override 8192 \
  --progress-interval-sec 120
```

Eval command:

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

## Results

Train summary:

```text
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/summary.json
sha256: c97dc26094462ec17d1dd970370d86e392433a8059aed9b1eaea1e5614ed2a06
train_ok: true
gpu: L4
torch_cuda: true
actual_max_env_step: 8192
collector_env_steps: 14791
remote_elapsed_sec: about 1326
checkpoint_bytes: 256,613,692
alternate_roots: none
```

The collector overshot the requested train cap in one batch. The train cap was
`8192`, but the final collected env-step count was `14791`.

Checkpoints were under the correct Volume root:

- `ckpt_best`
- `iteration_0`
- final `iteration_3697`

Corrected eval app:
`ap-ov622Yu6wEnN74V2Laf8HG`

Corrected eval manifest:

```text
training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/eval/faithful-short-periodic-custom512-stockeval-s0-8192-relpath/manifest_custom_steps512_seed0_20260509T232832Z.json
```

| Checkpoint | Manual return | Stock return | Steps | Nonzero rewards | Positive rewards | Action entropy | Dominant action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | `-13` | `-13` | `512` | `13` | `0` | `0.963778` | action `2`, share `0.345703` |
| `iteration_3697` | `-13` | `-8` | `512` | `13` | `0` | `0.755427` | action `0`, share `0.535156` |

Both checkpoints strict-loaded. Fallback was false for both.

Manual/stock first-prefix match was false for both rows. The stock
`MuZeroEvaluator` code path ran and saw final as less bad, but this eval still
used tiny wrapper defaults such as `num_simulations=2`. Treat it as useful
evidence, not final scoring. Report the manual 512-step telemetry side-by-side
for survival/action diagnostics. The mismatch is an eval-harness parity
warning, not a checkpoint-load failure.

Corrected stock-ish eval app:
`ap-81xAvfiyvnU8flV3eElPSH`

Corrected stock-ish eval id:
`faithful-short-periodic-stockish512-stockeval-s0-8192-relpath`

```text
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
`max_train_iter=1` and `update_per_collect=1` were part of the wrapper config.

| Checkpoint | Manual return | Stock return | Steps | Nonzero rewards | Positive rewards | Action entropy | Dominant action | Manual/stock match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `iteration_0` | `-13` | `-13` | `512` | `13` | `0` | `0.805545` | action `0`, share `0.521484` | `false` |
| `iteration_3697` | `-5` | `-8` | `512` | `7` | `1` | `0.644585` | action `0`, share `0.714844` | `false` |

The earlier low eval was invalid and misleading because manual
`max_episode_steps` stayed `64` while stock used `512`.

## Interpretation

The train wrapper now works for this faithful-short run. CUDA was available,
the run finished, checkpoints landed in the intended Volume root, and there was
no alternate-root artifact problem.

The stock-ish eval is now the main read for this two-checkpoint run. It is the
first weak signal that final is less bad than init under stock-ish eval:
manual return improved from `-13` to `-5`, stock return moved from `-13` to
`-8`, nonzero rewards dropped from `13` to `7`, and final saw one positive
reward. But this is one seed and two checkpoints, and manual-stock mismatch
remains. Do not call this solved Pong or exact reproduction.

This result does not prove LightZero cannot learn Pong. It says this bounded
faithful-short `8192` rehearsal did not produce a useful Pong policy.

## Artifacts

- Train summary:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/summary.json`
- Corrected tiny-wrapper eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/eval/faithful-short-periodic-custom512-stockeval-s0-8192-relpath/manifest_custom_steps512_seed0_20260509T232832Z.json`
- Corrected stock-ish eval:
  app `ap-81xAvfiyvnU8flV3eElPSH`, eval id
  `faithful-short-periodic-stockish512-stockeval-s0-8192-relpath`

## Follow-ups

- Keep calling this `faithful-short`, not exact reproduction.
- Do not use the earlier low eval.
- Use the completed stock-ish eval as the main score read for this small
  checkpoint pair; keep manual 512-step telemetry beside it for diagnostics.
- Treat the next larger run as a scale/accounting choice, not as evidence that
  learning is solved.
- Report manual/stock mismatch as an eval-harness parity warning, not a
  checkpoint-load failure.

No pytest was run.
