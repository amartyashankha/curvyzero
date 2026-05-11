# 2026-05-09 Modal LightZero Official Atari Pong 8192 Sim25

## Scope

Official LightZero Atari Pong only. This run targets the installed
`LightZero==0.2.0` current-config surface:

- `zoo.atari.config.atari_muzero_config`
- `lzero.entry.train_muzero`
- `PongNoFrameskip-v4`
- 64x64 four-frame visual conv MuZero

LightZero contains a full MuZero implementation: policy/model/MCTS/trainer,
collector, evaluator, replay-buffer integration, and Atari wrappers. The issue
in this lane is not a missing MuZero implementation. The issue is whether our
selected LightZero package surface, config, scale, checkpoint/eval protocol,
and project fit can reproduce useful stock Atari evidence cleanly enough before
we adopt it for CurvyZero work.

It does not target current GitHub upstream, whose exact-upstream dry validation
is tracked separately. It does not use the older 96x96/downsample pretrained
checkpoint lane. It is not custom dummy Pong and not CurvyTron.

No pytest was run.

## Wrapper Change

Changed only `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`.

- Raised official Atari validation caps to allow `8192` env steps, `64`
  requested train iter, `4` collectors, `25` simulations, `batch_size=64`,
  `1024` train/eval episode caps, and `game_segment_length=128`.
- Raised the cheap GPU function timeout to `60m`.
- Added `--update-per-collect=-1` as a CLI sentinel for restoring stock
  LightZero `update_per_collect=None` and the installed config's replay-ratio
  accounting.

Syntax check:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py
```

## Dry Config Validation

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute cpu --mode dry --max-env-step 8192 --max-train-iter 64 --collector-env-num 4 --evaluator-env-num 1 --num-simulations 25 --batch-size 64 --update-per-collect=-1 --max-episode-steps 1024 --game-segment-length 128 --run-id lz-visual-pong-8192-sim25-s0 --attempt-id dry-8192-sim25-b64-env4-auto
```

Modal app: `ap-VasQbApDzGd18EaB38hM59`

Result: pass. The patched surface was:

| Setting | Installed 0.2.0 stock | Patched rung |
| --- | ---: | ---: |
| `max_env_step` | `200000` | `8192` |
| `collector_env_num` / `n_episode` | `8` / `8` | `4` / `4` |
| `evaluator_env_num` / `n_evaluator_episode` | `3` / `3` | `1` / `1` |
| `num_simulations` | `50` | `25` |
| `batch_size` | `256` | `64` |
| `update_per_collect` | `None` | `None` |
| `game_segment_length` | `400` | `128` |
| episode caps | unset | `1024` |

## Train Command

Launched before follow-up pause guidance arrived; no further eval or sweep was
started after the follow-ups.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train --max-env-step 8192 --max-train-iter 64 --collector-env-num 4 --evaluator-env-num 1 --num-simulations 25 --batch-size 64 --update-per-collect=-1 --max-episode-steps 1024 --game-segment-length 128 --run-id lz-visual-pong-8192-sim25-s0 --attempt-id train-8192-sim25-b64-env4-auto
```

Modal app: `ap-qnwMaN8FlOUNJwLNo1mZKs`

## Train Result

Status: completed.

```text
run_id: lz-visual-pong-8192-sim25-s0
attempt_id: train-8192-sim25-b64-env4-auto
summary_ref: training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/train/summary.json
attempt_ref: training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/attempt.json
checkpoint_root: training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero
manifest_ref: training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/manifest.json
remote_elapsed_sec: 919.7004
train_elapsed_sec: 428.997874
runtime GPU: NVIDIA L4
CUDA available: true
final trainer-side reward: -21.0
```

The run mirrored `934` checkpoint files: `ckpt_best` plus `iteration_0` through
`iteration_932`.

Selected checkpoint refs:

```text
ckpt_best:
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/ckpt_best.pth.tar
sha256 71bc4988adfca1546f4ea70bb77d3f148c04d46ffa08198aa8b1aed5de722e04

iteration_0:
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/iteration_0.pth.tar
sha256 ccd9f5dbc33d19090dc878d15956a0f488cf76798fe48bddd9266473c7635fdb

iteration_100:
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/iteration_100.pth.tar
sha256 9f15cd39b642ff05f31f7f0847dc9c837adb9fd3f0e19ad14393a22eb887195e

iteration_500:
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/iteration_500.pth.tar
sha256 3aae481567c8b59cd026002d518ea08e65adfc489ead215587d301f982dfd7e8

iteration_900:
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/iteration_900.pth.tar
sha256 933a9aceb5465eb767121a30bf91aca6a11744f451cae4b297a2f49f8824dc96

iteration_932:
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/iteration_932.pth.tar
sha256 cb4de8cd15a1cd4b9b7698e418067c4df34f16e5fd60bf1dd616462f83a3498b
```

## Metadata Captured

Profile metadata:

- LightZero `0.2.0`, DI-engine `0.5.3`, torch `2.11.0`, ale-py `0.8.1`,
  gym `0.25.1`, gymnasium `0.28.0`, AutoROM `0.6.1`.
- Modal app `ap-qnwMaN8FlOUNJwLNo1mZKs`; Modal task
  `ta-01KR74M25J0K7EAP2V9XF84J4N`.
- Requested compute `gpu-l4-t4`; runtime GPU `NVIDIA L4`; CUDA available.

Contracts:

- Env id `PongNoFrameskip-v4`; env type `atari_lightzero`; policy type
  `muzero`; model type `conv`; observation shape `[4, 64, 64]`; action space
  size `6`; subprocess env manager.
- Config ref and full patched/original surfaces are in the train summary and
  command artifact.
- Post-train eval now exists for the selected checkpoint curve below. All eval
  rows used strict checkpoint load, no model fallback, manual
  `MuZeroPolicy.eval_mode.forward`, and stock `lzero.worker.MuZeroEvaluator`
  where possible.

Timing/throughput:

- Remote elapsed `919.7004s`.
- Trainer call elapsed `428.997874s`.
- LightZero log parser saw training iteration markers at
  `0,100,...,900`; no per-step throughput was emitted by this wrapper.

Checkpoint ids:

- `run_id`, `attempt_id`, selected refs, and selected sha256 values are listed
  above.
- Full checkpoint manifest is at
  `training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/manifest.json`.

Seed/reset:

- Seed `0`.
- Dynamic reset/evaluator seed behavior was not independently probed in this
  train run; prior stock MuZeroEvaluator parity covered the matching eval path.

Non-claims:

- This is not solved Atari Pong: the strict no-fallback eval curve below is
  mostly action-collapse evidence, with only `ckpt_best` showing action
  diversity in the manual path.
- This is not custom dummy Pong or CurvyTron evidence.
- This is not evidence for the current GitHub upstream surface or the older
  96x96/downsample pretrained checkpoint surface.

## Interpretation

This run is closer to installed `LightZero==0.2.0` stock Atari than 4096/sim10:
it moves search from `10` to `25`, collectors from `2` to `4`, batch from `32`
to `64`, env steps from `4096` to `8192`, episode cap from `512` to `1024`,
game segment length from `64` to `128`, and restores stock
`update_per_collect=None`.

It also exposed a practical adoption risk. This is a setup/fit/scale risk, not
an implementation-completeness risk. Restoring stock auto-update semantics
caused a much larger learner/checkpoint burst than the wrapper's
`max_train_iter=64` label suggests: the run mirrored checkpoints through
`iteration_932` and produced roughly 90 GB of checkpoint artifacts. That means
LightZero's stock update/replay accounting is powerful but not transparent
enough for casual bounded runs unless the wrapper explicitly controls checkpoint
cadence and learner-update reporting.

## Strict Checkpoint Curve Eval

No training and no pytest were run for this follow-up. The eval used the
existing stock-MuZeroEvaluator/manual parity wrapper on CPU with a 256-step cap,
`num_simulations=25`, `--no-allow-model-fallback`, and
`--run-stock-evaluator`.

Representative command shape:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/<checkpoint>.pth.tar \
  --run-id lz-visual-pong-8192-sim25-s0 \
  --attempt-id train-8192-sim25-b64-env4-auto \
  --output-ref training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve256/<checkpoint>_stock_evaluator_parity256.json \
  --max-env-step 8192 --max-train-iter 64 --collector-env-num 4 \
  --evaluator-env-num 1 --num-simulations 25 --batch-size 64 \
  --update-per-collect=-1 --max-episode-steps 1024 \
  --game-segment-length 128 --max-eval-steps 256 \
  --step-detail-limit 8 --no-allow-model-fallback --run-stock-evaluator
```

Artifacts:

```text
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve256/iteration_0_stock_evaluator_parity256.json
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve256/iteration_100_stock_evaluator_parity256.json
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve256/iteration_500_stock_evaluator_parity256.json
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve256/iteration_900_stock_evaluator_parity256.json
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve256/iteration_932_stock_evaluator_parity256.json
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve256/ckpt_best_stock_evaluator_parity256.json
```

Eval contracts were shared across rows:

- Packages: LightZero `0.2.0`, DI-engine `0.5.3`, torch `2.11.0`, ale-py
  `0.8.1`, gym `0.25.1`, gymnasium `0.28.0`, AutoROM `0.6.1`.
- Env/action contract: `PongNoFrameskip-v4`, actions
  `0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE`.
- Observation contract: reset env observation `[1, 64, 64]`; policy input
  `[4, 64, 64]`; stock evaluator policy data `[1, 4, 64, 64]`.
- Action masks: manual `[1, 1, 1, 1, 1, 1]`; stock
  `[[1, 1, 1, 1, 1, 1]]`.
- Checkpoint load: strict `as_is` state-dict load, no missing keys, no
  unexpected keys.
- Fallback: disabled and unused for all rows.

| Checkpoint | Manual action histogram | Manual return | Nonzero reward steps | Stock first-32 histogram | Stock return | Manual/stock first-32 match | Manual elapsed / steps-sec | Stock elapsed |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `iteration_0` | `{3: 256}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | `{3: 32}` | `-6.0` | yes | `17.617s / 14.531` | `11.411s` |
| `iteration_100` | `{0: 256}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | `{0: 32}` | `-6.0` | yes | `19.225s / 13.316` | `14.453s` |
| `iteration_500` | `{5: 256}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | `{5: 32}` | `-6.0` | yes | `31.722s / 8.070` | `22.061s` |
| `iteration_900` | `{0: 256}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | `{0: 32}` | `-6.0` | yes | `18.919s / 13.531` | `14.289s` |
| `iteration_932` | `{1: 256}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | `{1: 32}` | `-6.0` | yes | `18.400s / 13.913` | `13.353s` |
| `ckpt_best` | `{0: 38, 1: 45, 2: 42, 3: 30, 4: 50, 5: 51}` | `0.0` | `73:+1, 138:-1, 173:-1, 238:+1` | `{0: 6, 1: 2, 2: 3, 3: 12, 4: 5, 5: 4}` | `-6.0` | no | `29.547s / 8.664` | `18.073s` |

Checkpoint sha256 values:

```text
iteration_0:   ccd9f5dbc33d19090dc878d15956a0f488cf76798fe48bddd9266473c7635fdb
iteration_100: 9f15cd39b642ff05f31f7f0847dc9c837adb9fd3f0e19ad14393a22eb887195e
iteration_500: 3aae481567c8b59cd026002d518ea08e65adfc489ead215587d301f982dfd7e8
iteration_900: 933a9aceb5465eb767121a30bf91aca6a11744f451cae4b297a2f49f8824dc96
iteration_932: cb4de8cd15a1cd4b9b7698e418067c4df34f16e5fd60bf1dd616462f83a3498b
ckpt_best:     71bc4988adfca1546f4ea70bb77d3f148c04d46ffa08198aa8b1aed5de722e04
```

Read:

- The checkpoint naming/path was not blocked. All six requested refs existed
  and strict-loaded.
- The periodic checkpoints are not improving over the 256-step eval window.
  They each collapse to one action and lose six points by steps
  `60, 95, 130, 165, 200, 235`.
- `ckpt_best` is mechanically different from the periodic checkpoints: the
  manual path chooses all six actions and reaches a zero return in 256 steps,
  but stock evaluator first-32 actions do not match the manual prefix and stock
  return is still `-6.0`. The recorded `ckpt_best` policy logits are all zero
  in the step details, so this row should be treated as a parity warning rather
  than policy-quality evidence.
- The post-episode subprocess reset warnings from the stock evaluator appeared
  after the completed capped episode and did not block artifact creation.

The next action is not a larger train. The useful question is why `ckpt_best`
has a smaller checkpoint file, all-zero policy logits in the eval details,
manual/stock prefix mismatch, and a different return from the periodic
checkpoints. A single-ego wrapper remains possible for custom game work; the
issue to solve before adoption is observability, control, and metadata, not
impossibility.
