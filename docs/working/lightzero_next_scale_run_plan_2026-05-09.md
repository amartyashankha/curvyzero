# LightZero Next Scale Run Plan - 2026-05-09

Scope: next bounded faithful-short installed-package Atari Pong run after the
current `8192` faithful-short run is evaluated. This is not exact reproduction,
not custom dummy Pong, and not CurvyTron.

No pytest was run. Code observability was compiled with `py_compile`.

## Current Read

The current faithful-short run has completed:

```text
run_id: lz-visual-pong-exact-installed-0.2.0-s0
attempt_id: train-faithful-short-installed-0.2.0-s0-8192-relpath
run_kind: faithful-short
max_env_step_override: 8192
train_app: ap-ipdfYJmWQitQtIBxrKU2E9
train_summary: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/summary.json
train_summary_sha256: c97dc26094462ec17d1dd970370d86e392433a8059aed9b1eaea1e5614ed2a06
```

Operational read:

- Train ok `true`.
- GPU was L4 and torch CUDA was true.
- Actual max env step was `8192`.
- The collector overshot to `14791` env steps in one batch.
- Remote elapsed was about `1326s`.
- Checkpoints were under the intended Volume root only:
  `ckpt_best`, `iteration_0`, and final `iteration_3697`.
- Checkpoint bytes were `256,613,692`, well below `2 GiB`.
- No alternate roots were found.

Corrected eval read:

```text
eval_app: ap-ov622Yu6wEnN74V2Laf8HG
manifest: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/eval/faithful-short-periodic-custom512-stockeval-s0-8192-relpath/manifest_custom_steps512_seed0_20260509T232832Z.json
```

Eval covered `iteration_0` and final `iteration_3697` only. Both strict-loaded,
both used no fallback, and both ran 512 manual steps. Manual returns were `-13`
for both, with `13` nonzero rewards and `0` positive rewards. Stock
`MuZeroEvaluator` returns were `-13` for `iteration_0` and `-8` for
`iteration_3697`.

The stock `MuZeroEvaluator` code path returned `-13` for `iteration_0` and
`-8` for `iteration_3697`, but this eval still used tiny wrapper defaults such
as `num_simulations=2`. Treat it as useful evidence, not the final stock-ish
scoring read. Manual 512-step telemetry is still required side-by-side for
survival, reward-count, action-collapse, and parity diagnostics. Manual/stock
first-prefix match was false; that is an eval-harness parity warning, not a
checkpoint-load failure. The earlier low eval was invalid because manual
`max_episode_steps` stayed `64` while stock used `512`.

Corrected stock-ish eval completed: app `ap-81xAvfiyvnU8flV3eElPSH`, eval id
`faithful-short-periodic-stockish512-stockeval-s0-8192-relpath`, same two
checkpoints, strict/no fallback, `num_simulations=50`, `evaluator_env_num=3`,
`collector_env_num=8`, `batch_size=256`, `game_segment_length=400`,
`max_env_step=200000`, `max_train_iter=1`, `update_per_collect=1`, CUDA false
because the eval wrapper ran on CPU, manual cap `512`, and
`max_episode_steps=512`.

| Checkpoint | Manual return | Stock return | Steps | Nonzero rewards | Positive rewards | Dominant action/share | Entropy | Manual/stock match |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `iteration_0` | `-13` | `-13` | `512` | `13` | `0` | `0` / `0.521484` | `0.805545` | `false` |
| `iteration_3697` | `-5` | `-8` | `512` | `7` | `1` | `0` / `0.714844` | `0.644585` | `false` |

Conclusion: artifact accounting passed. The previous corrected eval is useful
but not the full story. The stock-ish eval gives the first weak signal that
final is less bad than init, but it is one seed/two checkpoints and
manual-stock mismatch remains. A larger run is an operational scale choice,
not confirmation of solved learning.

## Checkpoint Accounting

The exact wrapper does not set checkpoint cadence. It only patches
`exp_name`, changes cwd to `/runs` in train mode, and optionally shortens the
`train_muzero(max_env_step=...)` argument for faithful-short mode.

Installed-package dry validation showed stock Atari config values:

- `eval_freq=2000`
- `update_per_collect=None`
- `replay_ratio=0.25`
- `save_ckpt_after_iter` unset in the imported LightZero config
- `save_ckpt_after_run` not visible in that config surface, but supplied by
  DI-engine learner defaults after config compilation

LightZero/DI-engine source behavior:

- DI-engine `BaseLearner` default hook config has
  `save_ckpt_after_iter=10000` and `save_ckpt_after_run=True`.
- DI-engine `SaveCkptHook` writes `iteration_<last_iter>.pth.tar` when
  `last_iter % freq == 0`, and writes the final checkpoint from the after-run
  hook.
- LightZero `MuZeroEvaluator` calls `learner.save_checkpoint('ckpt_best.pth.tar')`
  when eval reward improves.
- LightZero `train_muzero` runs one eval before the main loop, then evals when
  `evaluator.should_eval(learner.train_iter)` triggers.
- With `update_per_collect=None`, LightZero computes learner updates from
  collected transitions and `replay_ratio`; changing `replay_ratio`,
  `update_per_collect`, collector counts, episode caps, or segment length
  changes the number of learner iterations and therefore checkpoint exposure.

Expected checkpoint count for the exact wrapper's next small faithful-short
rung:

- `ckpt_best.pth.tar` from initial/new-best eval.
- `iteration_0.pth.tar` from the first learner after-iter hook.
- A final `iteration_<N>.pth.tar` from `save_ckpt_after_run`.
- Add `iteration_10000.pth.tar` only if learner iterations reach `10000`.

For `32768` env steps, the likely learner-iteration count should stay below
`10000`, based on the previous `8192/sim25` stock-update run reaching
`iteration_932` and the rough replay-ratio envelope. So plan for `3` checkpoint
files, tolerate a few more if `ckpt_best` is overwritten or eval timing differs,
and treat `>10` checkpoint files or `>2 GiB` checkpoint bytes as an accounting
surprise.

Do not force `save_ckpt_after_iter=1` again. That was the source of the prior
`934`-checkpoint / roughly `90 GB` burst.

## Recommended Next Run

If continuing this lane, run `32768`, not full exact yet. It is a clean 4x step
increase over the completed `8192`, still far below installed stock `200000`,
and should remain below the default `10000`-iteration periodic checkpoint.

This is not justified by a clean learning claim. It is justified only if the
next question is bounded scale, cleaner accounting, and whether more
faithful-short steps move stock-ish return, manual 512-step diagnostics, or
positive rewards.

Use the same cheap GPU wrapper and keep the progress watcher at `120s` so
Volume growth is visible early:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --progress-interval-sec 120 \
  --max-env-step-override 32768
```

Launch record:

```text
launched_at: 2026-05-09
app: ap-xiGLACKHPZLvL1eYgygqvm
run_id: lz-visual-pong-exact-installed-0.2.0-s0
attempt_id: train-faithful-short-installed-0.2.0-s0-32768-relpath
status: running at time of doc update
latest_progress: learner iteration 500; artifact root healthy; 1 ckpt_best
  around 64 MB; no alternate roots
claim: bounded scale/accounting probe only, not a learning claim
```

This preserves stock installed LightZero settings except:

- Modal artifact placement via `exp_name`.
- `train_muzero.max_env_step` shortened from installed stock `200000` to
  `32768`, so the run is explicitly `faithful-short`.

It does not change `num_simulations`, `collector_env_num`,
`evaluator_env_num`, `batch_size`, `game_segment_length`, `update_per_collect`,
`replay_ratio`, `eval_freq`, episode caps, or checkpoint cadence.

## Watch And Stop Rules

Watch `train/progress/latest.json` during the first `10-20m`.

Stop the run if:

- CUDA is unavailable.
- Artifacts appear under an alternate root instead of the intended Volume tree.
- Checkpoint count grows past `10` before the run is near completion.
- Checkpoint bytes exceed `2 GiB` during this bounded rung.
- No checkpoints, logs, or env-step progress appear after startup.

Those are operational stop rules only. They do not judge learning.

## Retention And Eval

For this `32768` rung, retain all stock checkpoints if checkpoint count is
single-digit. If the count unexpectedly exceeds `10`, write a manifest first,
then keep only:

- `ckpt_best.pth.tar` for debugging only, not as quality evidence until the
  best-save behavior is explained.
- `iteration_0.pth.tar`
- one middle periodic/final-ish checkpoint if present
- final `iteration_<N>.pth.tar`
- config, logs, TensorBoard files, progress snapshots, summary JSON, and eval
  manifest JSON

While training is running, it is safe to run low-detail strict/no-fallback eval
over periodic `iteration_*.pth.tar` files as they appear in the Modal Volume.
Use stock-ish eval settings rather than the wrapper's tiny defaults. Do not put
`ckpt_best` in the first quality curve.

Avoid duplicate evals by listing the eval root and skipping any checkpoint that
already has an artifact directory named
`iteration_<N>_low_steps512_seed0`.

Template:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --compute gpu-l4-t4 \
  --parallel \
  --eval-pass low \
  --eval-id faithful-short-32768-periodic-low-stockeval-s0 \
  --checkpoint-refs '<ITERATION_0_REF>,<MID_ITERATION_REF>,<FINAL_ITERATION_REF>' \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --max-env-step 200000 \
  --collector-env-num 8 \
  --evaluator-env-num 3 \
  --num-simulations 50 \
  --batch-size 256 \
  --game-segment-length 400 \
  --max-episode-steps 512 \
  --low-detail-max-eval-steps 512 \
  --low-detail-step-detail-limit 8 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

The first readout is the manifest table: strict load, fallback use,
manual/stock match, manual return, `stock_return`, nonzero/positive reward
counts, action histogram, dominant action share, entropy, elapsed time, and
verdict. If stock-ish eval is strict/no-fallback clean, rank by `stock_return`
and use manual telemetry as diagnostics.

## Bottom Line

The active larger run is `32768` env steps on `gpu-l4-t4`, not a full
`200000` exact run. The completed `8192` relpath run cleared the artifact-root
and checkpoint-byte gate. Its stock-ish eval gives a weak less-bad final
signal, but not a solved-learning claim. The purpose of `32768` is bounded
scale plus another honest stock-ish-return/manual-telemetry read, not
confirmation of solved learning.
