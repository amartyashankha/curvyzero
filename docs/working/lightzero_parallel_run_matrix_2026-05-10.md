# LightZero Parallel Run Matrix - 2026-05-10

Purpose: organized parallel official/control LightZero Atari Pong runs for
signal search, using the corrected background pattern:
`modal run --detach` locally plus `Function.spawn(...)` inside
`lightzero_pong_exact_reproduction.py`.

Live queue note: the current checkpoint/eval queue has been split out to
`docs/working/lightzero_live_queue_status_2026-05-10.md` so this matrix stays
readable. That queue doc records the latest poll, WaveB ids, shaped-vs-normal
separation, user-launched eval ids to avoid duplicating, and next eval gates.

## Claim

These launches prove only that four official/control LightZero Pong training
attempts were spawned as background Modal function calls with unique run ids,
attempt ids, and Volume progress refs. All use the installed `LightZero==0.2.0`
Atari Pong config surface, `PongNoFrameskip-v4`, seed-specific training, and
checkpoint cadence override `save_ckpt_after_iter=1000`.

## Non-claim

These launches and first-checkpoint evals do not prove solved Pong, exact
upstream GitHub reproduction, CurvyTron readiness, or completion of all remote
trainers. First-checkpoint evals are survival-first same-run comparisons only.
Do not compare these runs to dummy Pong.

## Long Run Status

Verified `2026-05-10`: the previously mentioned `199000` max-env-step long
runs are not alive. They should not be polled as active work and should not be
used as completed long-run evidence.

Evidence:

- `modal app list` shows all candidate long-run apps as `stopped` with `0`
  tasks: seed `4` app `ap-cYFKqDKr0rer2BgP29wS7e`, seed `5` app
  `ap-xQYWssVUsflEPBUExzar7g`, seed `6` app
  `ap-mRkytZQQM3VcaqKFwT4XiC`, seed `7` app
  `ap-TXPKFEWk9iwRENASU7LRhu`, seed `8` app
  `ap-kiaPJJMJfGViYu2BMcVOHD`, and seed `9` app
  `ap-s4EmO5wEXoQUINOhuELQnK`.
- Each checked Volume `progress/latest.json` says `phase: failed`, with final
  timestamps from `2026-05-10T04:26:37Z` through
  `2026-05-10T04:26:40Z`. App logs show stop/interrupt shutdown
  (`KeyboardInterrupt`, `BrokenPipeError`, `Runner terminated`).
- Seed `4` H100 long199k has `iteration_0.pth.tar` and `ckpt_best.pth.tar`
  only. Seed `5` H100 long199k has `iteration_0.pth.tar`,
  `iteration_5000.pth.tar`, and `ckpt_best.pth.tar`.
- Seeds `6` and `7` H100CPU16 long199k, and seeds `8` and `9` L4CPU16
  long199k, have only `ckpt_best.pth.tar` under the checked attempt ids. No
  `iteration_0.pth.tar` or periodic checkpoint is present for those four.
- The newer seed `20` through `23` H100CPU16 long199k launch references are
  stale too. `modal app list` shows app ids `ap-H6TbFZNa339kszHVqh1cVH`,
  `ap-DFBC3D78TaG8lwc9QTrIy0`, `ap-2VgJbQGadEuFc139yWfoaX`, and
  `ap-R84GpxSHpxUjwqei5AuwNQ` as `stopped` with `0` tasks. Targeted Volume
  checks for each expected
  `training/lightzero-official-visual-pong/{run_id}/attempts/{attempt_id}/train/progress`
  path returned `No such file or directory`, so there is no progress or
  checkpoint evidence under those recorded ids.

Safe next action: do not rely on these as live or completed long runs. Finish
the `65536` sweep first-checkpoint evals first. Launch fresh `199000` runs only
after those evals show that more scale is worth the Modal spend; if launched,
use new run ids and attempt ids.

## Matrix

All runs use:

- module: `curvyzero.infra.modal.lightzero_pong_exact_reproduction`
- mode: `train`
- launch pattern: `uv run --extra modal modal run --detach -m ...`
- training call pattern: `Function.spawn(...)`, per printed launch JSON
- installed LightZero package target: `LightZero==0.2.0`
- env: `PongNoFrameskip-v4`
- max env step override: `65536`
- checkpoint cadence: `--save-ckpt-after-iter-override 1000`
- progress interval: `120s`
- no custom/dummy Pong

| seed | compute | app id | function call id | run id | attempt id | progress ref |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `gpu-l4-t4` | `ap-KPcjy4bRO859DM1VjVyYv4` | `fc-01KR7PP0W00WHTAMCRZXV43QNR` | `lz-visual-pong-exact-installed-0.2.0-s1` | `train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath` | `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath/train/progress/latest.json` |
| 2 | `gpu-l4-t4` | `ap-QfQmdGXM2aFCtQyKUaiSi5` | `fc-01KR7PPMJWZPWE3H2F4C5HS2H9` | `lz-visual-pong-exact-installed-0.2.0-s2` | `train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath` | `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s2/attempts/train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath/train/progress/latest.json` |
| 3 | `gpu-l4-t4` | `ap-Fk9zl5nYNXVzmwYICzsptz` | `fc-01KR7PQ869KZED12MH8GWDYRBH` | `lz-visual-pong-exact-installed-0.2.0-s3` | `train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath` | `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s3/attempts/train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath/train/progress/latest.json` |
| 1 | `gpu-h100` | `ap-t3blwBFV4TsXL4Idsk08tr` | `fc-01KR7PRDC7J3PR4AAHPX7ATRY1` | `lz-visual-pong-exact-installed-0.2.0-s1-h100` | `train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-h100-relpath` | `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1-h100/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-h100-relpath/train/progress/latest.json` |

The H100 run was added only after the wrapper visibly exposed
`--compute gpu-h100` and Modal created the H100 function. It is named with
`h100` in both run id and attempt id so it stays separate from the L4/T4 seed
matrix.

## Launched Commands

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4 --seed 1 --run-id lz-visual-pong-exact-installed-0.2.0-s1 --attempt-id train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath --progress-interval-sec 120 --max-env-step-override 65536 --save-ckpt-after-iter-override 1000
```

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4 --seed 2 --run-id lz-visual-pong-exact-installed-0.2.0-s2 --attempt-id train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath --progress-interval-sec 120 --max-env-step-override 65536 --save-ckpt-after-iter-override 1000
```

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4 --seed 3 --run-id lz-visual-pong-exact-installed-0.2.0-s3 --attempt-id train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath --progress-interval-sec 120 --max-env-step-override 65536 --save-ckpt-after-iter-override 1000
```

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-h100 --seed 1 --run-id lz-visual-pong-exact-installed-0.2.0-s1-h100 --attempt-id train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-h100-relpath --progress-interval-sec 120 --max-env-step-override 65536 --save-ckpt-after-iter-override 1000
```

## Monitoring

Poll progress refs in the `curvyzero-runs` Modal Volume. The first useful
remote checkpoint signal is `iteration_1000.pth.tar`; later checkpoints should
appear every `1000` learner iterations, plus whatever final checkpoint the run
writes.

Example checkpoint dirs:

```bash
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath/train/lightzero_exp/ckpt
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s2/attempts/train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath/train/lightzero_exp/ckpt
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s3/attempts/train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath/train/lightzero_exp/ckpt
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1-h100/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-h100-relpath/train/lightzero_exp/ckpt
```

## Eval Triggers

Run strict same-run eval when a target checkpoint appears. The eval set must
include that run's own `iteration_0.pth.tar`, use `--no-allow-model-fallback`,
and run the stock evaluator. First trigger per run is `iteration_1000`; then
evaluate later normal checkpoints, especially around `9000/10000`, `16000`,
`32000`, `48000`, and the latest/final checkpoint near `65536`.

Use `scripts/lightzero_live_eval_queue.py` to print or execute pending strict
eval commands. Eval supports `cpu`, `gpu-l4-t4`, and `gpu-l4-t4-cpu8`; prefer
`gpu-l4-t4-cpu8` for this live sweep when available. Using L4/T4 eval for an
H100-trained checkpoint is acceptable because checkpoint quality, not training
hardware, is the comparison target.

Seed 1 L4/T4:

```bash
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-exact-installed-0.2.0-s1 --attempt-id train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath --eval-id live-s1-65536-ckpt1000-stockish512-stockeval-s1 --compute gpu-l4-t4 --eval-pass low --seed 1 --max-env-step 65536
```

Seed 2 L4/T4:

```bash
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-exact-installed-0.2.0-s2 --attempt-id train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath --eval-id live-s2-65536-ckpt1000-stockish512-stockeval-s2 --compute gpu-l4-t4 --eval-pass low --seed 2 --max-env-step 65536
```

Seed 3 L4/T4:

```bash
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-exact-installed-0.2.0-s3 --attempt-id train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath --eval-id live-s3-65536-ckpt1000-stockish512-stockeval-s3 --compute gpu-l4-t4 --eval-pass low --seed 3 --max-env-step 65536
```

Seed 1 H100:

```bash
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-exact-installed-0.2.0-s1-h100 --attempt-id train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-h100-relpath --eval-id live-s1-h100-65536-ckpt1000-stockish512-stockeval-s1 --compute gpu-l4-t4 --eval-pass low --seed 1 --max-env-step 65536
```

Every eval report should lead with:

1. `steps_survived / eval_cap_steps`
2. saturation status (`survived_to_cap`)
3. `delta_steps_survived` versus same-run `iteration_0`
4. raw/manual return/points and stock return
5. positive and negative reward counts
6. action histogram / dominant action share / entropy
7. checkpoint ref, run id, attempt id, strict load status, fallback status

If a 512-step eval saturates, use or report the longer-cap 2048-step result as
the meaningful survival signal. Do not summarize only win/loss or stock return.

Tooling update: the eval queue now prints compact output by default, and the
summarizer now leads with stock survival. Details are already recorded in
`docs/working/lightzero_eval_tooling_2026-05-10.md`.

## Broad Normal Sweep 65k Tracker

Purpose: track the broad normal LightZero Atari Pong 65k sweep that the main
thread is launching. These are normal stock/control runs, not survival-shaped
runs and not dummy Pong.

Claim: this section records the intended normal sweep ids, compute labels, and
readiness gate for seeds `10` through `19`, plus the later L4/T4+CPU16 seeds
`24` through `27`.

Non-claim: this does not claim any run has produced a useful checkpoint,
finished training, improved Pong, solved Pong, or established CurvyTron
readiness. Do not eval or report any sweep result until the same run has at
least `iteration_1000.pth.tar` plus its own `iteration_0.pth.tar`.

Shared settings:

- installed LightZero package target: `LightZero==0.2.0`
- env: `PongNoFrameskip-v4`
- run family: normal official/control Atari Pong
- max env step override: `65536`
- checkpoint cadence: `--save-ckpt-after-iter-override 1000`
- first useful eval trigger: `iteration_1000.pth.tar`
- first eval comparison: same-run `iteration_0` versus `iteration_1000`
- later eval plan: compact curve after more checkpoints appear, for example
  `0/1000/5000/10000/16000/latest`, using strict no-fallback stock evaluator
- claim rule: lead with survival/return metrics and include plain Claim /
  Non-claim before interpreting the run

Live readiness update: targeted Volume checks at `2026-05-10 00:46 EDT` found
first useful checkpoints for repeatB seed `1` and seed `18`. A follow-up watch
poll at `2026-05-10 00:52 EDT` found same-run `iteration_0.pth.tar` plus
`iteration_1000.pth.tar` for seeds `12`, `13`, `15`, `16`, and `17`; seeds
`12`, `13`, and `17` also had `iteration_2000.pth.tar`.

Main-thread eval ids already launched and not duplicated here:
`sweep65k-s10-0-1000-stock2048-seed10`,
`sweep65k-s11-0-1000-stock2048-seed11`,
`sweep65k-s14-0-1000-2000-stock2048-seed14`,
`sweep65k-s18-0-1000-2000-3000-stock2048-seed18`,
`sweep65k-s19-0-1000-2000-stock2048-seed19`, and
`repeatB-s1-0-1000-2000-3000-stock2048-seed1`.

| seed | compute | compute label | run id | attempt id | checkpoint readiness |
| ---: | --- | --- | --- | --- | --- |
| 1 repeatB | `gpu-l4-t4` | `l4` | `lz-visual-pong-exact-installed-0.2.0-s1-repeatB-65536-l4` | `train-normal-repeatB-s1-65536-ckpt1000-spawn-l4-relpath` | `iteration_0`, `1000`; tracker eval done, main broader eval launched |
| 10 | `gpu-l4-t4` | `l4` | `lz-visual-pong-exact-installed-0.2.0-s10-sweep65k-l4` | `train-normal-sweep65k-s10-ckpt1000-spawn-l4-relpath` | main-thread first eval launched |
| 11 | `gpu-l4-t4` | `l4` | `lz-visual-pong-exact-installed-0.2.0-s11-sweep65k-l4` | `train-normal-sweep65k-s11-ckpt1000-spawn-l4-relpath` | main-thread first eval launched |
| 12 | `gpu-l4-t4` | `l4` | `lz-visual-pong-exact-installed-0.2.0-s12-sweep65k-l4` | `train-normal-sweep65k-s12-ckpt1000-spawn-l4-relpath` | `iteration_0`, `1000`, `2000`; tracker eval done |
| 13 | `gpu-l4-t4` | `l4` | `lz-visual-pong-exact-installed-0.2.0-s13-sweep65k-l4` | `train-normal-sweep65k-s13-ckpt1000-spawn-l4-relpath` | `iteration_0`, `1000`, `2000`; tracker eval done |
| 14 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s14-sweep65k-l4cpu16` | `train-normal-sweep65k-s14-ckpt1000-spawn-l4cpu16-relpath` | main-thread first eval launched |
| 15 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s15-sweep65k-l4cpu16` | `train-normal-sweep65k-s15-ckpt1000-spawn-l4cpu16-relpath` | `iteration_0`, `1000`; tracker eval done |
| 16 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s16-sweep65k-l4cpu16` | `train-normal-sweep65k-s16-ckpt1000-spawn-l4cpu16-relpath` | `iteration_0`, `1000`; tracker eval done |
| 17 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s17-sweep65k-l4cpu16` | `train-normal-sweep65k-s17-ckpt1000-spawn-l4cpu16-relpath` | `iteration_0`, `1000`, `2000`; tracker eval done |
| 18 | `gpu-h100-cpu16` | `h100cpu16` | `lz-visual-pong-exact-installed-0.2.0-s18-sweep65k-h100cpu16` | `train-normal-sweep65k-s18-ckpt1000-spawn-h100cpu16-relpath` | main broader eval launched; tracker early duplicate done under different id |
| 19 | `gpu-h100-cpu16` | `h100cpu16` | `lz-visual-pong-exact-installed-0.2.0-s19-sweep65k-h100cpu16` | `train-normal-sweep65k-s19-ckpt1000-spawn-h100cpu16-relpath` | main-thread first eval launched |
| 24 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s24-sweep65k-l4cpu16` | `train-normal-sweep65k-s24-ckpt1000-spawn-l4cpu16-relpath` | launched about `2026-05-10 01:00 EDT`; readiness not polled here |
| 25 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s25-sweep65k-l4cpu16` | `train-normal-sweep65k-s25-ckpt1000-spawn-l4cpu16-relpath` | launched about `2026-05-10 01:00 EDT`; readiness not polled here |
| 26 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s26-sweep65k-l4cpu16` | `train-normal-sweep65k-s26-ckpt1000-spawn-l4cpu16-relpath` | launched about `2026-05-10 01:00 EDT`; readiness not polled here |
| 27 | `gpu-l4-t4-cpu16` | `l4cpu16` | `lz-visual-pong-exact-installed-0.2.0-s27-sweep65k-l4cpu16` | `train-normal-sweep65k-s27-ckpt1000-spawn-l4cpu16-relpath` | launched about `2026-05-10 01:00 EDT`; readiness not polled here |

Later launch details for seeds `24` through `27`: these are normal
stock/control runs, with no survival reward shaping,
`max_env_step_override=65536`, and `save_ckpt_after_iter=1000`.

| seed | app id | function call id |
| ---: | --- | --- |
| 24 | `ap-bM4vaOcdwXp4fiAGlQFB56` | `fc-01KR843ZMSJ6YBEMNA6VXFCAFX` |
| 25 | `ap-O1lEEybIwFW9QJi4TEOUHG` | `fc-01KR843ZQC5X6CCBAC9T22800P` |
| 26 | `ap-M1SyZ2F875Tl6gpkcmLBZI` | `fc-01KR843ZJ9G0QTNH5AK3RDH44N` |
| 27 | `ap-Kef3glrFAB8YzcKFMgzhxH` | `fc-01KR843ZFVDQXGE87EWJA4GPVX` |

Checkpoint path pattern:

```text
training/lightzero-official-visual-pong/{run_id}/attempts/{attempt_id}/train/lightzero_exp/ckpt
```

Eval queue pattern once `iteration_1000.pth.tar` is visible:

```bash
uv run python scripts/lightzero_live_eval_queue.py --run-id {run_id} --attempt-id {attempt_id} --eval-id sweep65k-s{seed}-0-1000-stock2048-seed{seed} --compute gpu-l4-t4-cpu8 --eval-pass custom --seed {seed} --max-eval-steps 2048 --max-episode-steps 2048 --max-env-step 65536 --update-per-collect -1 --selected-iterations 0,1000 --group-size 1 --max-parallel-launches 2
```

Add `--execute` only after confirming both checkpoints are present. Do not run
eval before `iteration_1000`; `iteration_0` alone is a baseline, not a training
signal.

Ready evals were launched for seeds `12`, `13`, `15`, and `17`. Seed `17` had
one bad local launch using `max_episode_steps=8` under eval id
`sweep65k-s17-0-1000-2000-3000-stock2048-seed17`; it was killed locally. Ignore
that bad eval id if artifacts appear. The corrected relaunch uses eval id
`sweep65k-s17-0-1000-2000-3000-stock2048b-seed17`.

This tracker lane also launched/completed first useful evals for seeds `12`,
`13`, `15`, `16`, and `17` after the `00:52 EDT` readiness poll. Modal app ids
are captured below; per-checkpoint function call ids were not emitted by the
`modal run` output. These runs used `gpu-l4-t4-cpu8`, group size `1`, strict
no-fallback, stock evaluator, a 2048-step cap, matching seed, and
`--update-per-collect -1`.

Claim: the tracker-launched evals strict-loaded visible same-run checkpoints
and produced stock evaluator telemetry. First-checkpoint signal is mostly flat;
only seed `17` shows a small stock-survival bump from `760/2048` at
`iteration_0` to `880/2048` at `iteration_1000`, while stock return remains
`-21`.

Non-claim: these first-checkpoint evals do not prove solved Pong, durable
learning, exact upstream reproduction, or CurvyTron readiness. The seed `18`
tracker eval overlaps the main-thread broader seed-18 eval and is redundant
early telemetry, not the primary seed-18 claim.

Current high-level read: normal early `1000`-`3000` checkpoint rows are mostly
flat. The useful old seed-1 signal appeared late, around `10000`-`18000`, so
the next normal-lane priority is later compact curves, not more early-only
interpretation.

| eval id | selected checkpoints | Modal app ids | local manifest dir | stock-side result |
| --- | --- | --- | --- | --- |
| `live-normal-repeatB-s1-0-1000-stock2048-seed1-upcnone-cpu8` | `0,1000` | `ap-9Rjvih6lcaayiZQbG9wEFl`, `ap-THg5WLuiY6yBNJX5XvM6UE` | `artifacts/local/lightzero-eval-manifests/live-normal-repeatB-s1-0-1000-stock2048-seed1-upcnone-cpu8/` | both rows stock `761/2048`, return `-21`, positive rewards `0` |
| `live-sweep65k-s18-0-1000-stock2048-seed18-upcnone-cpu8` | `0,1000` | `ap-JW6Fa75mrJAPevX0k0oR5s`, `ap-lY6TkxnHqVuIBcPYdahr6E` | `artifacts/local/lightzero-eval-manifests/live-sweep65k-s18-0-1000-stock2048-seed18-upcnone-cpu8/` | both rows stock `759/2048`, return `-21`, positive rewards `0` |
| `sweep65k-s12-0-1000-2000-stock2048-seed12` | `0,1000,2000` | `ap-nro1WlYzNFGF71x6GGnzw1`, `ap-GZrgADYrw8WO1WYxx7HGgz`, `ap-ABcuD1rfRwdO4nXWPvHbhR` | `artifacts/local/lightzero-eval-manifests/sweep65k-s12-0-1000-2000-stock2048-seed12/` | all rows stock `757/2048`, return `-21`, positive rewards `0` |
| `sweep65k-s13-0-1000-2000-stock2048-seed13` | `0,1000,2000` | `ap-4fykk8JyoCogkcZk59IO16`, `ap-rcjdOCobDa8GCM3yKiBw47`, `ap-ainAzBTOhCkhgiblZDNckq` | `artifacts/local/lightzero-eval-manifests/sweep65k-s13-0-1000-2000-stock2048-seed13/` | all rows stock `760/2048`, return `-21`, positive rewards `0` |
| `sweep65k-s15-0-1000-stock2048-seed15` | `0,1000` | `ap-T3eRgXCQ71bCYv3t2D0JRi`, `ap-8GfOc7X5huZZ5urkEjhqNT` | `artifacts/local/lightzero-eval-manifests/sweep65k-s15-0-1000-stock2048-seed15/` | both rows stock `759/2048`, return `-21`, positive rewards `0`; `iteration_1000` has `stock_manual_match=false` |
| `sweep65k-s16-0-1000-stock2048-seed16` | `0,1000` | `ap-SNKoGvdC6UlZgxQSzsGcQH`, `ap-QJ4MFXPOlpR9Fxi3SvHDKA` | `artifacts/local/lightzero-eval-manifests/sweep65k-s16-0-1000-stock2048-seed16/` | both rows stock `761/2048`, return `-21`, positive rewards `0` |
| `sweep65k-s17-0-1000-2000-stock2048-seed17` | `0,1000,2000` | `ap-zvC3ie4gsqsDuIeBgDO91h`, `ap-6dFCDTakpywxIFbZEtizwp`, `ap-fehwbtk5sOBTJRDnvfSH7V` | `artifacts/local/lightzero-eval-manifests/sweep65k-s17-0-1000-2000-stock2048-seed17/` | stock `760/2048`, `880/2048`, `760/2048`; stock return `-21` and positive rewards `0` for all rows |

## Survival-Shaped Side Lane

Claim: this section records a newly launched survival-shaped side lane. These
runs are intentionally separate from normal stock/control Pong.

Non-claim: these are not stock/control Pong evidence and must not be mixed into
normal sweep claims, normal long-run claims, or CurvyTron readiness claims.

Launched around `2026-05-10 01:05 EDT`.

Shared settings:

- reward shaping: `reward_shaping_enabled=true`
- survival reward: `survival_reward_per_step=0.001`
- checkpoint cadence: `save_ckpt_after_iter_override=1000`
- progress interval: `120s`

| seed | compute | max env step | app id | function call id | run id | attempt id |
| ---: | --- | ---: | --- | --- | --- | --- |
| 30 | `gpu-l4-t4-cpu16` | `65536` | `ap-Y012lkdhQsoCBdfTZwEOxe` | `fc-01KR84DT9XJQ6950G8FNTFMFA4` | `lz-visual-pong-survival-shaped-step0p001-s30-65k-l4cpu16` | `train-survival-shaped-step0p001-s30-65536-ckpt1000-spawn-l4cpu16-relpath` |
| 31 | `gpu-l4-t4-cpu16` | `65536` | `ap-hXNZ3KegIzNvUunuhepVSL` | `fc-01KR84DTEBC1K8EY1VFK9T7V5X` | `lz-visual-pong-survival-shaped-step0p001-s31-65k-l4cpu16` | `train-survival-shaped-step0p001-s31-65536-ckpt1000-spawn-l4cpu16-relpath` |
| 32 | `gpu-h100-cpu16` | `199000` | `ap-sfp8Npo0Jwg8zwIGbEEMwG` | `fc-01KR84DTP2M8M9VPYJ5PANGF5C` | `lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16` | `train-survival-shaped-step0p001-s32-199000-ckpt1000-spawn-h100cpu16-relpath` |
| 33 | `gpu-h100-cpu16` | `199000` | `ap-dszTPF0BR1k4y0a80eLItB` | `fc-01KR84DTNJ6WT5SSE4KCM38BEN` | `lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16` | `train-survival-shaped-step0p001-s33-199000-ckpt1000-spawn-h100cpu16-relpath` |

Eval gate: wait for same-run `iteration_0.pth.tar` plus target checkpoints,
then eval under strict no-fallback. Reports must lead with survival steps,
stock return, reward counts, action histograms, checkpoint refs, and an
explicit reminder that this is the shaped side lane.

## Longer H100+CPU16 Wave

Claim: this wave gives us longer normal stock/control Pong curves while the
65k sweep runs. It uses the faster `gpu-h100-cpu16` lane and still keeps
checkpoint cadence at `1000`, so eval can start early.

Non-claim: this does not replace the cross-seed 65k sweep and does not prove
quality until same-run checkpoint evals land.

Shared settings:

- max env step override: `199000`
- checkpoint cadence: `--save-ckpt-after-iter-override 1000`
- reward shaping: off
- first useful eval trigger: `iteration_1000.pth.tar`

| seed | compute | run id | attempt id | Modal app | call id |
| ---: | --- | --- | --- | --- | --- |
| 20 | `gpu-h100-cpu16` | `lz-visual-pong-exact-installed-0.2.0-s20-long199k-h100cpu16-ckpt1000` | `train-normal-long199k-s20-ckpt1000-spawn-h100cpu16-relpath` | `ap-H6TbFZNa339kszHVqh1cVH` | `fc-01KR834MZJ9WVQZKVTXRJ6N353` |
| 21 | `gpu-h100-cpu16` | `lz-visual-pong-exact-installed-0.2.0-s21-long199k-h100cpu16-ckpt1000` | `train-normal-long199k-s21-ckpt1000-spawn-h100cpu16-relpath` | `ap-DFBC3D78TaG8lwc9QTrIy0` | `fc-01KR834N116VA7TXGSW76ZVCS3` |
| 22 | `gpu-h100-cpu16` | `lz-visual-pong-exact-installed-0.2.0-s22-long199k-h100cpu16-ckpt1000` | `train-normal-long199k-s22-ckpt1000-spawn-h100cpu16-relpath` | `ap-2VgJbQGadEuFc139yWfoaX` | `fc-01KR834N13MJ2KM30GNSR56QB5` |
| 23 | `gpu-h100-cpu16` | `lz-visual-pong-exact-installed-0.2.0-s23-long199k-h100cpu16-ckpt1000` | `train-normal-long199k-s23-ckpt1000-spawn-h100cpu16-relpath` | `ap-R84GpxSHpxUjwqei5AuwNQ` | `fc-01KR834N0EGFNBKJCPCHDMPN0V` |

Readiness poll, `2026-05-10 00:46 EDT`: exact checkpoint dirs for seeds
`20`-`23` returned `No such file or directory`. Keep these four runs on the
watch list behind the 65k sweep, but make no eval claim until same-run
`iteration_0.pth.tar` plus `iteration_1000.pth.tar` are visible.

Correction: the actual later detachedB relaunch ids for long seeds `20` and
`21` are separate from the stale non-detachedB rows above:

- seed `20`: run
  `lz-visual-pong-exact-installed-0.2.0-s20-long199k-h100cpu16-ckpt1000-detachedB`,
  attempt `train-normal-long199k-s20-ckpt1000-detachedB-h100cpu16-relpath`.
- seed `21`: run
  `lz-visual-pong-exact-installed-0.2.0-s21-long199k-h100cpu16-ckpt1000-detachedB`,
  attempt `train-normal-long199k-s21-ckpt1000-detachedB-h100cpu16-relpath`.

Latest queue poll at `2026-05-10 01:23 EDT` found seed `20` through
`iteration_5000` and seed `21` through `iteration_7000`. Harvest reported
early long seed `20`/`21` rows as flat, so the live queue is holding further
long-run eval launch until a stronger later gate, preferably `10000+`.

## Seed 1 Compact Latest Curve Eval

Visible checkpoint poll before eval found `iteration_0`, every 1000-step
checkpoint through `iteration_18000`, and latest `iteration_18459`. Per the
coach-lane follow-up, this eval uses the compact curve only:
`iteration_0`, `iteration_1000`, `iteration_5000`, `iteration_10000`,
`iteration_16000`, and latest `iteration_18459`.

Eval ids:

- `matrix-s1-l4-compact-0-1000-5000-10000-16000-18459-stock2048-seed1`
  for `iteration_0`, `iteration_1000`, `iteration_5000`, and
  `iteration_10000`. The larger parallel call hit the old 480s per-row
  timeout before `iteration_16000` and `iteration_18459`, but completed and
  wrote the first four strict artifacts.
- `matrix-s1-l4-compact-late-16000-18459-stock2048-seed1` for
  `iteration_16000` and `iteration_18459`, run as a small late group after a
  temporary local evaluator timeout bump. The timeout change was restored after
  the run; eval semantics stayed strict load/no fallback with stock evaluator.

Late manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath/eval/matrix-s1-l4-compact-late-16000-18459-stock2048-seed1/manifest_custom_steps2048_seed1_20260510T040804Z.json`.

Local fetched artifacts:

- `artifacts/local/lightzero-eval-manifests/matrix-s1-l4-compact-0-1000-5000-10000-16000-18459-stock2048-seed1/`
- `artifacts/local/lightzero-eval-manifests/matrix-s1-l4-compact-late-16000-18459-stock2048-seed1/manifest_custom_steps2048_seed1_20260510T040804Z.json`

Local combined summary:
`artifacts/local/lightzero-eval-manifests/matrix-s1-l4-compact-combined-stock2048-seed1-summary.tsv`.

| Checkpoint | Steps / cap | Delta steps | Saturated | Manual return/points | Stock return | Positive rewards | Negative/nonzero rewards | Dominant action | Action entropy | Strict load | Fallback | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `iteration_0` | `763/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `0` share `1.0` | `0` | `true` | `false` | `collapsed_action` |
| `iteration_1000` | `763/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `0` share `0.984273` | `0.116727` | `true` | `false` | `collapsed_action` |
| `iteration_5000` | `763/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `4` share `0.988204` | `0.06209` | `true` | `false` | `collapsed_action` |
| `iteration_10000` | `1118/2048` | `+355` | `false` | `-20` | `-17` | `1` | `21/22` | `0` share `0.616279` | `0.681224` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_16000` | `1713/2048` | `+950` | `false` | `-20` | `-17` | `1` | `21/22` | `0` share `0.381203` | `0.878778` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_18459` | `2048/2048` | `+1285` | `true` | `-12` | `13` | `4` | `16/20` | `0` share `0.473633` | `0.852286` | `true` | `false` | `manual_stock_mismatch` |

Claim: seed 1 L4/T4 compact checkpoints `iteration_0`, `iteration_1000`,
`iteration_5000`, `iteration_10000`, `iteration_16000`, and latest
`iteration_18459` all strict-loaded with no fallback and ran the stock
evaluator under a 2048-step cap. Survival-first signal improves from
`763/2048` at same-run `iteration_0` to `2048/2048` at latest
`iteration_18459`, a `+1285` step gain and saturation to the cap; latest also
has stock return `+13`.

Non-claim: this does not prove solved Pong, exact upstream reproduction, or a
stable final policy. Manual/stock action traces mismatch from `iteration_10000`
onward, so manual survival and stock return are both useful but distinct
telemetry. The earlier checkpoints remain collapsed or near-collapsed, and the
late result needs confirmation across seeds and later/longer-cap evals before
claiming durable learning.

## Seed 2 First Checkpoint Eval

Eval id:
`matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2`.

Manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s2/attempts/train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath/eval/matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2/manifest_custom_steps2048_seed2_20260510T011941Z.json`.

Local fetched manifest copy:
`artifacts/local/lightzero-eval-manifests/matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2/manifest_custom_steps2048_seed2_20260510T011941Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `762/2048` steps
(`survival_fraction=0.37207`) and was not saturated because it did not survive
to the cap. Return/points were `-21`; rewards were `0` positive and `21`
negative. Strict load true, fallback false, stock return `-21`, dominant
action `1` share `1.0`, entropy `0`, verdict `collapsed_action`.
`iteration_1000` survived `762/2048` steps (`delta_steps_survived=0`,
`survival_fraction=0.37207`) and was also not saturated. Return/points were
`-21` (`delta_return=0`); rewards were `0` positive and `21` negative. Strict
load true, fallback false, stock return `-21` (`delta_stock_return=0`),
dominant action `1` share `0.838583`, entropy `0.430975`, verdict
`manual_stock_mismatch`.

Claim: seed 2 L4/T4 `iteration_0` and `iteration_1000` both strict-loaded with
no fallback, ran the stock evaluator, and produced a valid 2048-cap
survival-first comparison. Because the meaningful survival read is the longer
cap, the report uses `762/2048` as the first-line metric rather than a
512-step saturation-style summary.

Non-claim: this does not show learning progress at `iteration_1000`. Survival
time, saturation status, return/points, positive rewards, and negative rewards
are unchanged versus same-run `iteration_0`. The action distribution is less
purely collapsed, but still dominated by action `1` and not associated with a
survival or return gain.

## Seed 2 Through Iteration 3000 Eval

Eval id:
`matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2`.

Manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s2/attempts/train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath/eval/matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2/manifest_custom_steps2048_seed2_20260510T013455Z.json`.

Local fetched manifest copy:
`artifacts/local/lightzero-eval-manifests/matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2/manifest_custom_steps2048_seed2_20260510T013455Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2/summary_baseline_deltas.tsv`.

| Checkpoint | Steps / cap | Delta steps | Manual return | Stock return | Positive rewards | Negative/nonzero rewards | Dominant action | Action entropy | Strict load | Fallback | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `iteration_0` | `762/2048` | `0` | `-21` | `-21` | `0` | `21` | `1` share `1.0` | `0` | `true` | `false` | `collapsed_action` |
| `iteration_1000` | `762/2048` | `0` | `-21` | `-21` | `0` | `21` | `1` share `0.832021` | `0.429028` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_2000` | `762/2048` | `0` | `-21` | `-21` | `0` | `21` | `4` share `0.94357` | `0.313109` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_3000` | `762/2048` | `0` | `-21` | `-21` | `0` | `21` | `0` share `0.611549` | `0.648552` | `true` | `false` | `manual_stock_mismatch` |

Claim: seed 2 L4/T4 checkpoints `iteration_0`, `iteration_1000`,
`iteration_2000`, and `iteration_3000` all strict-loaded with no fallback,
ran the stock evaluator, and produced a valid 2048-cap survival-first
comparison. Every row survived `762/2048`, so all deltas versus same-run
`iteration_0` are `0`.

Non-claim: this does not show learning progress through `iteration_3000`.
Manual return, stock return, positive rewards, negative/nonzero rewards, and
survival are flat versus same-run `iteration_0`; the only movement is action
distribution churn from action-1 collapse to action-4 near-collapse and then a
less pure action-0 dominant policy at `iteration_3000`.

## Seed 2 Compact Latest Curve Eval

Visible checkpoint poll before eval found `iteration_0`, every 1000-step
checkpoint through `iteration_16000`, and latest `iteration_16829`. Per the
coach-lane follow-up, this eval uses the compact curve only:
`iteration_0`, `iteration_1000`, `iteration_5000`, `iteration_10000`,
`iteration_16000`, and `iteration_16829`.

Eval id:
`matrix-s2-l4-compact-0-1k-5k-10k-16k-16829-stock2048-seed2`.

Manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s2/attempts/train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath/eval/matrix-s2-l4-compact-0-1k-5k-10k-16k-16829-stock2048-seed2/manifest_custom_steps2048_seed2_20260510T034755Z.json`.

Local fetched manifest copy:
`artifacts/local/lightzero-eval-manifests/matrix-s2-l4-compact-0-1k-5k-10k-16k-16829-stock2048-seed2/manifest_custom_steps2048_seed2_20260510T034755Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/matrix-s2-l4-compact-0-1k-5k-10k-16k-16829-stock2048-seed2/summary_baseline_deltas.tsv`.

| Checkpoint | Steps / cap | Delta steps | Saturated | Manual return/points | Stock return | Positive rewards | Negative/nonzero rewards | Dominant action | Action entropy | Strict load | Fallback | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `iteration_0` | `762/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `1` share `1.0` | `0` | `true` | `false` | `collapsed_action` |
| `iteration_1000` | `762/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `1` share `0.829396` | `0.454914` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_5000` | `762/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `2` share `0.540682` | `0.544536` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_10000` | `882/2048` | `+120` | `false` | `-21` | `-21` | `0` | `21/21` | `5` share `0.386621` | `0.859586` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_16000` | `781/2048` | `+19` | `false` | `-21` | `-21` | `0` | `21/21` | `0` share `0.421255` | `0.795995` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_16829` | `840/2048` | `+78` | `false` | `-20` | `-21` | `1` | `21/22` | `1` share `0.77619` | `0.452982` | `true` | `false` | `manual_stock_mismatch` |

Claim: seed 2 L4/T4 compact checkpoints `iteration_0`, `iteration_1000`,
`iteration_5000`, `iteration_10000`, `iteration_16000`, and latest
`iteration_16829` all strict-loaded with no fallback, ran the stock evaluator,
and produced valid non-saturated 2048-cap survival reads. The best survival in
this compact curve is `iteration_10000` at `882/2048`, `+120` steps versus
same-run `iteration_0`; latest `iteration_16829` is `840/2048`, `+78` steps.

Non-claim: this does not prove solved Pong or clean return learning. Stock
return remains `-21` for every compact checkpoint; only latest `iteration_16829`
has a manual-return/positive-reward bump (`-20`, one positive reward), while
still ending with `21` negative rewards and no survival to the 2048-step cap.
Action usage is less collapsed at `iteration_10000` and `iteration_16000`, but
the useful quality signal is still limited to weak survival gains.

## Seed 3 First Checkpoint Eval

Eval id:
`matrix-s3-65536-l4t4-iteration0-1000-2000-stockish2048-stockeval-s3`.

Manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s3/attempts/train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath/eval/matrix-s3-65536-l4t4-iteration0-1000-2000-stockish2048-stockeval-s3/manifest_custom_steps2048_seed3_20260510T012941Z.json`.

Local fetched manifest copy:
`artifacts/local/lightzero-eval-manifests/matrix-s3-65536-l4t4-iteration0-1000-2000-stockish2048-stockeval-s3/manifest_custom_steps2048_seed3_20260510T012941Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/matrix-s3-65536-l4t4-iteration0-1000-2000-stockish2048-stockeval-s3/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `762/2048` steps
(`survival_fraction=0.37207`) and was not saturated because it did not survive
to the cap. Return/points were `-21`; rewards were `0` positive and `21`
negative. Strict load true, fallback false, stock return `-21`, dominant
action `1` share `0.883202`, entropy `0.520086`, verdict
`manual_stock_mismatch`. `iteration_1000` survived `822/2048` steps
(`delta_steps_survived=+60`, `survival_fraction=0.401367`) and was also not
saturated. Return/points were `-21` (`delta_return=0`); rewards were `0`
positive and `21` negative. Strict load true, fallback false, stock return
`-21` (`delta_stock_return=0`), dominant action `4` share `0.594891`,
entropy `0.756305`, verdict `manual_stock_mismatch`. `iteration_2000`
survived `762/2048` steps (`delta_steps_survived=0`,
`survival_fraction=0.37207`) and was also not saturated. Return/points were
`-21` (`delta_return=0`); rewards were `0` positive and `21` negative. Strict
load true, fallback false, stock return `-21` (`delta_stock_return=0`),
dominant action `2` share `0.89895`, entropy `0.304373`, verdict
`manual_stock_mismatch`.

Claim: seed 3 L4/T4 `iteration_0`, `iteration_1000`, and `iteration_2000` all
strict-loaded with no fallback, ran the stock evaluator, and produced a valid
non-saturated 2048-cap survival comparison. The meaningful survival signal is
`822/2048` for `iteration_1000` versus `762/2048` for same-run
`iteration_0`, a `+60` step survival gain; `iteration_2000` does not retain
that gain and returns to `762/2048`.

Non-claim: this does not show return learning or solved Pong. All rows still
end at `-21` return/points and stock return, with `0` positive rewards and
`21` negative rewards, and no checkpoint survives to the 2048-step cap. The
action distribution is less collapsed at `iteration_1000`, but that is only
telemetry unless it persists with return or longer-horizon survival gains.

## Seed 3 Compact Latest Curve Eval

Visible checkpoint poll before eval found `iteration_0`, every 1000-step
checkpoint through `iteration_17000`, and latest `iteration_17010`. Per the
coach-lane follow-up, this eval uses the compact curve only: `iteration_0`,
`iteration_1000`, `iteration_5000`, `iteration_10000`, `iteration_16000`, and
latest `iteration_17010`.

Eval id:
`matrix-s3-l4-compact-stock2048-seed3`.

Manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s3/attempts/train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath/eval/matrix-s3-l4-compact-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T035431Z.json`.

Local fetched manifest copy:
`artifacts/local/lightzero-eval-manifests/matrix-s3-l4-compact-stock2048-seed3/manifest_custom_steps2048_seed3_20260510T035431Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/matrix-s3-l4-compact-stock2048-seed3/summary_baseline_deltas.tsv`.

| Checkpoint | Steps / cap | Delta steps | Saturated | Manual return/points | Stock return | Positive rewards | Negative/nonzero rewards | Dominant action | Action entropy | Strict load | Fallback | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `iteration_0` | `762/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `1` share `0.866142` | `0.567925` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_1000` | `882/2048` | `+120` | `false` | `-21` | `-20` | `0` | `21/21` | `4` share `0.643991` | `0.657883` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_5000` | `762/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `1` share `0.675853` | `0.59978` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_10000` | `762/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `1` share `0.547244` | `0.684933` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_16000` | `1605/2048` | `+843` | `false` | `-17` | `-18` | `4` | `21/25` | `1` share `0.447975` | `0.848373` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_17010` | `1236/2048` | `+474` | `false` | `-19` | `-21` | `2` | `21/23` | `1` share `0.574434` | `0.691895` | `true` | `false` | `manual_stock_mismatch` |

Claim: seed 3 L4/T4 compact checkpoints `iteration_0`, `iteration_1000`,
`iteration_5000`, `iteration_10000`, `iteration_16000`, and latest
`iteration_17010` all strict-loaded with no fallback, ran the stock evaluator,
and produced valid non-saturated 2048-cap survival reads. The best compact
survival point is `iteration_16000` at `1605/2048`, `+843` steps versus
same-run `iteration_0`; latest `iteration_17010` remains above baseline at
`1236/2048`, `+474` steps, but below the `iteration_16000` peak.

Non-claim: this does not prove solved Pong or cap survival. Return learning is
mixed: `iteration_16000` improves both manual return (`-17`) and stock return
(`-18`) versus baseline, but latest `iteration_17010` falls back to stock
return `-21` despite a manual-return and positive-reward bump. No checkpoint
survived to the 2048-step cap, and stock/manual mismatch remains on every
compact row.

No pytest was run.

## Seed 1 H100 Compact Latest Curve Eval

Visible checkpoint poll before eval found `iteration_0`, every 1000-step
checkpoint through `iteration_17000`, and final/latest `iteration_17504`.
Per the coach-lane follow-up, this eval uses the compact curve only:
`iteration_0`, `iteration_1000`, `iteration_5000`, `iteration_10000`,
`iteration_16000`, and latest `iteration_17504`.

Eval id:
`h100-s1-official-control-compact-stock2048-seed1`.

Local summary:
`artifacts/local/lightzero-eval-manifests/h100-s1-official-control-compact-stock2048-seed1/summary_baseline_deltas.tsv`.

Remote artifacts:

- `iteration_0`, `iteration_1000`, `iteration_5000`, `iteration_16000`, and
  `iteration_17504` are under
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1-h100/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-h100-relpath/eval/h100-s1-official-control-compact-stock2048-seed1/`.
- `iteration_10000` was completed as a single-checkpoint retry after the first
  parallel map hit the older 480s timeout. Its remote artifact is
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1-h100/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-h100-relpath/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260510T035747Z.json`.

| Checkpoint | Steps / cap | Delta steps | Saturated | Manual return/points | Stock return | Positive rewards | Negative/nonzero rewards | Dominant action | Action entropy | Strict load | Fallback | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `iteration_0` | `763/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `0` share `1.0` | `0` | `true` | `false` | `collapsed_action` |
| `iteration_1000` | `763/2048` | `0` | `false` | `-21` | `-21` | `0` | `21/21` | `1` share `0.689384` | `0.893883` | `true` | `false` | `negative_return` |
| `iteration_5000` | `792/2048` | `+29` | `false` | `-21` | `-21` | `0` | `21/21` | `0` share `0.436869` | `0.771637` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_10000` | `1731/2048` | `+968` | `false` | `-19` | `-16` | `2` | `21/23` | `0` share `0.27383` | `0.976163` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_16000` | `1471/2048` | `+708` | `false` | `-19` | `-19` | `2` | `21/23` | `1` share `0.239973` | `0.974604` | `true` | `false` | `manual_stock_mismatch` |
| `iteration_17504` | `782/2048` | `+19` | `false` | `-21` | `-21` | `0` | `21/21` | `3` share `0.393862` | `0.730883` | `true` | `false` | `manual_stock_mismatch` |

Claim: seed 1 H100 compact checkpoints `iteration_0`, `iteration_1000`,
`iteration_5000`, `iteration_10000`, `iteration_16000`, and latest
`iteration_17504` all strict-loaded with no fallback, ran the stock evaluator,
and produced valid non-saturated 2048-cap survival reads. The best compact
survival point is `iteration_10000` at `1731/2048`, `+968` steps versus
same-run `iteration_0`; `iteration_16000` remains above baseline at
`1471/2048`, `+708` steps, but latest `iteration_17504` mostly falls back to
baseline survival at `782/2048`, `+19`.

Non-claim: this does not prove solved Pong or cap survival. Return learning is
real but not durable: `iteration_10000` improves manual return to `-19`, stock
return to `-16`, and positive rewards to `2`, while `iteration_17504` returns
to `-21` manual and stock return with `0` positive rewards. No checkpoint
survived to the 2048-step cap, and stock/manual mismatch remains on the
non-baseline higher-signal rows.

No pytest was run.
