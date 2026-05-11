# LightZero 10-Run Launch Wave - 2026-05-10

Last updated: `2026-05-10 09:41 EDT`.

Purpose: keep training artifacts flowing while CPU40 eval harvest continues.
This is an exploration wave, not proof that Pong is solved.

Main signal for every later eval: stock steps survived versus the same run's
`iteration_0`. Score/return stays secondary. Seeds are for diagnosis and
reproducibility; the goal is sustained survival improvement over checkpoints,
not one lucky seed.

Training starts should stay varied/random. When practical, eval should report
survival over multiple starts. Compare normal stock reward against
survival-shaped runs, but do not let shaping hide broken learning.

## 09:41 Shaped/Randomization Lane Decision

No additional shaped/random-start training was launched from this pass. Finish
the existing wave10 and wave11 stock-evaluator survival reads first. The exact
hold gate and fallback `wave12-micro` launch matrix are recorded in
[lightzero_shaped_randomization_lane_decision_2026-05-10.md](lightzero_shaped_randomization_lane_decision_2026-05-10.md).

## Runs To Launch

| seed | lane | compute | max env step | run id | attempt id |
| ---: | --- | --- | ---: | --- | --- |
| 50 | normal | `gpu-l4-t4-cpu16` | 65536 | `lz-visual-pong-exact-installed-0.2.0-s50-wave10-l4cpu16` | `train-normal-wave10-s50-65536-ckpt1000-l4cpu16-relpath` |
| 51 | normal | `gpu-l4-t4-cpu16` | 65536 | `lz-visual-pong-exact-installed-0.2.0-s51-wave10-l4cpu16` | `train-normal-wave10-s51-65536-ckpt1000-l4cpu16-relpath` |
| 52 | normal | `gpu-l4-t4-cpu16` | 65536 | `lz-visual-pong-exact-installed-0.2.0-s52-wave10-l4cpu16` | `train-normal-wave10-s52-65536-ckpt1000-l4cpu16-relpath` |
| 53 | normal | `gpu-l4-t4-cpu16` | 65536 | `lz-visual-pong-exact-installed-0.2.0-s53-wave10-l4cpu16` | `train-normal-wave10-s53-65536-ckpt1000-l4cpu16-relpath` |
| 54 | normal | `gpu-l4-t4-cpu40` | 65536 | `lz-visual-pong-exact-installed-0.2.0-s54-wave10-l4cpu40` | `train-normal-wave10-s54-65536-ckpt1000-l4cpu40-relpath` |
| 55 | normal | `gpu-l4-t4-cpu40` | 65536 | `lz-visual-pong-exact-installed-0.2.0-s55-wave10-l4cpu40` | `train-normal-wave10-s55-65536-ckpt1000-l4cpu40-relpath` |
| 56 | normal | `gpu-h100-cpu16` | 199000 | `lz-visual-pong-exact-installed-0.2.0-s56-wave10-h100cpu16-long199k` | `train-normal-wave10-s56-199000-ckpt1000-h100cpu16-relpath` |
| 57 | normal | `gpu-h100-cpu40` | 199000 | `lz-visual-pong-exact-installed-0.2.0-s57-wave10-h100cpu40-long199k` | `train-normal-wave10-s57-199000-ckpt1000-h100cpu40-relpath` |
| 60 | shaped | `gpu-l4-t4-cpu16` | 65536 | `lz-visual-pong-survival-shaped-step0p001-s60-wave10-l4cpu16` | `train-survival-shaped-step0p001-wave10-s60-65536-ckpt1000-l4cpu16-relpath` |
| 61 | shaped | `gpu-h100-cpu16` | 199000 | `lz-visual-pong-survival-shaped-step0p001-s61-wave10-h100cpu16-long199k` | `train-survival-shaped-step0p001-wave10-s61-199000-ckpt1000-h100cpu16-relpath` |

All runs use `save_ckpt_after_iter_override=1000`.

## Launch Results

All ten local launch commands returned `status=spawned`.

| seed | app id | function call id |
| ---: | --- | --- |
| 50 | `ap-TqtxL2yZXoag0mfLcDqS0x` | `fc-01KR8XREFNF19K1YCA3R7CHKHM` |
| 51 | `ap-n6bxWxs3os3HxNKm3hN9XY` | `fc-01KR8XREP9DPE9BSYJ5K3BER54` |
| 52 | `ap-VpfmPA0gKw4gJrnddAH3yT` | `fc-01KR8XREEPG3BK672FZ6MRMH9B` |
| 53 | `ap-lIRoBNZwgddHUiHozSXReT` | `fc-01KR8XREKD9VQS3VXA818E5RJM` |
| 54 | `ap-84VgZC3VlNtL8YhLvZ8eDr` | `fc-01KR8XREMM2DY7RXDE5EDTD6JN` |
| 55 | `ap-u1JC7NcXJWJWO2j5Gnel2q` | `fc-01KR8XRYH2H10YF602Y7ZNVSGB` |
| 56 | `ap-FuYnRU2jVM4ExnrZJK53YV` | `fc-01KR8XRXRD97CPAEW4J28BCD6R` |
| 57 | `ap-8ErdSUpPAdw5aZCr1xqATG` | `fc-01KR8XRYKPGHKNAAQWZXDAJ3KE` |
| 60 | `ap-6ZydFmHGNUFODSlgGR1J8q` | `fc-01KR8XRXQCTP4G6PKJKAQR4KR1` |
| 61 | `ap-FaiguZq2bbUCSFecfyaR1p` | `fc-01KR8XRXRH9NGZYBB7FK7H7JJN` |

Next check: poll Modal Volume checkpoint dirs for `iteration_0` and
`iteration_1000`; do not eval until at least `0,1000,5000` exist unless a
launch looks broken.

## Checkpoint Readiness Poll

At `2026-05-10 08:31 EDT`, sampled seeds `50`, `54`, `56`, `57`, `60`, and
`61` had `ckpt_best.pth.tar` visible but not yet useful `iteration_*.pth.tar`
checkpoints. That suggests the jobs have started, but it is too early to eval.

At `2026-05-10 08:35 EDT`, sampled seeds `50`-`55`, `57`, `60`, and `61`
had `iteration_0.pth.tar`; seed `56` had `iteration_0.pth.tar` and
`iteration_1000.pth.tar`. Still too early for the main eval gate.

At `2026-05-10 08:36 EDT`, all ten checkpoint dirs were polled. All ten had
`iteration_0.pth.tar`. Seeds `56`, `57`, and `61` had
`iteration_1000.pth.tar`. No listed run had `iteration_5000.pth.tar`.

After a short wait, at `2026-05-10 08:37 EDT`, all ten checkpoint dirs were
polled again. All ten had `iteration_0.pth.tar` and `iteration_1000.pth.tar`.
No listed run had `iteration_5000.pth.tar`.

At `2026-05-10 08:38 EDT`, sampled seeds `50`, `54`, `56`, `57`, and `60`
still had only `iteration_0.pth.tar` and `iteration_1000.pth.tar`. No
`iteration_5000.pth.tar` yet. Use a longer sleep before the next poll.

At `2026-05-10 08:42 EDT`, sampled seeds `50`, `56`, and `57` had reached
`iteration_2000.pth.tar`; sampled seeds `54` and `60` were still at
`iteration_1000.pth.tar`. No sampled run had `iteration_5000.pth.tar` yet.
Keep waiting; no eval launch yet.

At `2026-05-10 08:46 EDT`, sampled seeds `50`, `56`, and `57` had reached
`iteration_3000.pth.tar`; sampled seeds `54` and `60` had reached
`iteration_2000.pth.tar`. No sampled run had `iteration_5000.pth.tar` yet.
Keep waiting; no eval launch yet.

At `2026-05-10 08:51 EDT`, seed `56` reached `iteration_5000.pth.tar`, seed
`57` reached `iteration_4000.pth.tar`, seeds `50`, `54`, and `60` had not yet
reached `5000` in the sampled poll.

Eval launched at `2026-05-10 08:52 EDT` for seed `56`:

- eval id: `wave10-normal-s56-0-1000-5000-stock2048-seed56`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `78461`

Eval result harvested for seed `56`:

- `iteration_0`: `stock_steps_survived=763`, `stock_return=-21`,
  `stock_positive_reward_count=0`; dominant action `4` collapsed.
- `iteration_1000`: `stock_steps_survived=763`, `stock_return=-21`,
  `stock_positive_reward_count=0`.
- `iteration_5000`: `stock_steps_survived=823`, `stock_return=-21`,
  `stock_positive_reward_count=0`; action mix was less collapsed.
- Plain read: survival-only bump of `+60` stock steps at `iteration_5000`.
  This is not solved Pong. Stock return and positive reward count did not
  improve.
- Manual rollout reached `1479` steps at `iteration_5000`, but stock/manual
  mismatch means stock evaluator is the primary readout.

At `2026-05-10 08:54 EDT`, dry queue polls found seeds `57` and `61` ready
with selected checkpoints `0,1000,5000`; their eval roots did not exist. Seeds
`50`, `51`, `53`, and `60` still exposed only selected checkpoints `0,1000`.
Some parallel dry polls for seeds `52`, `54`, `55`, and `56` hit Modal
`VolumeListFiles` rate limiting and need a later retry.

Eval launched at `2026-05-10 08:55 EDT` for seed `57`:

- eval id: `wave10-normal-s57-0-1000-5000-stock2048-seed57`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `92599`
- Modal app ids observed: `ap-6bGn2sqwkyOmZLLHX4XjRT`,
  `ap-HUjRDjCe7FNob1lK0iEvvD`, `ap-2pzUjIt0K1XKxRXzPupXQC`

Eval result harvested for seed `57`:

- `iteration_0`: `stock_steps_survived=759`, `stock_return=-21`,
  `stock_positive_reward_count=0`.
- `iteration_1000`: `stock_steps_survived=940`, `stock_return=-21`,
  `stock_positive_reward_count=0`.
- `iteration_5000`: `stock_steps_survived=759`, `stock_return=-21`,
  `stock_positive_reward_count=0`.
- Plain read: best survival is at `iteration_1000`; `iteration_5000` does not
  improve over same-run `iteration_0`.

Eval launched at `2026-05-10 08:55 EDT` for seed `61`:

- eval id: `wave10-shaped-s61-0-1000-5000-stock2048-seed61`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `68253`
- Modal app ids observed: `ap-peaRArNYV9p5MNd1fGQNrp`,
  `ap-FXi90ChkwVtEliR16KY3lh`, `ap-AZDPhDCwzlTY0i7HJYXq7F`

Eval result harvested for seed `61`:

- `iteration_0`: `stock_steps_survived=760`, `stock_return=-21`,
  `stock_positive_reward_count=0`.
- `iteration_1000`: `stock_steps_survived=760`, `stock_return=-21`,
  `stock_positive_reward_count=0`.
- `iteration_5000`: `stock_steps_survived=837`, `stock_return=-21`,
  `stock_positive_reward_count=0`.
- Plain read: survival-only bump of `+77` stock steps at `iteration_5000`;
  still dies before the `2048` step cap and has no return/positive-reward
  improvement.

At `2026-05-10 09:02 EDT`, direct checkpoint-dir polls found seeds `50`,
`53`, `54`, `55`, and `60` ready with selected checkpoints `0,1000,5000`.
Seeds `51` and `52` still exposed only selected checkpoints `0,1000`.

Eval launched at `2026-05-10 09:03 EDT` for seed `50`:

- eval id: `wave10-normal-s50-0-1000-5000-stock2048-seed50`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `17539`
- Modal app ids observed: `ap-DBcXVOdJdUE8roNi18RFnU`,
  `ap-091DlZqIu81F2tcTomORKZ`, `ap-QralT702al8xw8EJQzQQcb`

Eval result harvested for seed `50`:

- stock survival steps: `iteration_0=764`, `iteration_1000=764`,
  `iteration_5000=764`
- stock return stayed `-21`; stock positive reward count stayed `0`
- Plain read: flat survival; no `iteration_5000` improvement over same-run
  `iteration_0`

Eval launched at `2026-05-10 09:03 EDT` for seed `53`:

- eval id: `wave10-normal-s53-0-1000-5000-stock2048-seed53`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `96310`
- Modal app ids observed: `ap-739ZoE9GpsfaPBssYtTG7q`,
  `ap-rAXfxbnzvGGudnNXZ3Wuqx`, `ap-Bujzmh67rKxnF8UY6uULju`

Eval result harvested for seed `53`:

- stock survival steps: `iteration_0=757`, `iteration_1000=757`,
  `iteration_5000=757`
- stock return stayed `-21`; stock positive reward count stayed `0`
- Plain read: flat survival; no `iteration_5000` improvement over same-run
  `iteration_0`

Eval launched at `2026-05-10 09:03 EDT` for seed `54`:

- eval id: `wave10-normal-s54-0-1000-5000-stock2048-seed54`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `51392`
- Modal app ids observed: `ap-9l3pJicbS7goI5XurKPrjY`,
  `ap-6M0xs3e2BsTiYx6wiAde0P`, `ap-8G47jdrBiyCN3KCiJUTyjj`
- caveat: queue helper's eval-root listing hit a Modal rate-limit warning, but
  the checkpoint-dir gate was confirmed before launch

Eval result harvested for seed `54`:

- stock survival steps: `iteration_0=763`, `iteration_1000=763`,
  `iteration_5000=763`
- stock return stayed `-21`; stock positive reward count stayed `0`
- Plain read: flat survival; no `iteration_5000` improvement over same-run
  `iteration_0`

Eval launched at `2026-05-10 09:03 EDT` for seed `55`:

- eval id: `wave10-normal-s55-0-1000-5000-stock2048-seed55`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `3021`
- Modal app ids observed: `ap-nXCe3gHQDBkkJmV1Y73EXc`,
  `ap-XPpsgGfDMZdCxu00GVzsAn`, `ap-umdTgv7zVdL2nRVCO1iFtB`

Eval result harvested for seed `55`:

- stock survival steps: `iteration_0=814`, `iteration_1000=758`,
  `iteration_5000=758`
- stock return stayed `-21`; stock positive reward count stayed `0`
- Plain read: regression from `iteration_0`; no `iteration_5000` improvement

Eval launched at `2026-05-10 09:03 EDT` for seed `60`:

- eval id: `wave10-shaped-s60-0-1000-5000-stock2048-seed60`
- selected iterations: `0,1000,5000`
- compute: `gpu-l4-t4-cpu40`
- local session: `91584`
- Modal app ids observed: `ap-TV2ALuf1xmwnglIJdSjbCl`,
  `ap-ok2RwZcSkmUKXj1wN4TKnf`, `ap-vt9cuRYjWYfX5Q8JYAEklc`

Eval result harvested for seed `60`:

- stock survival steps: `iteration_0=764`, `iteration_1000=764`,
  `iteration_5000=908`
- stock return stayed `-21`; stock positive reward count stayed `0`
- Plain read: shaped `iteration_5000` has a survival-only bump of `+144`
  stock steps, but still dies before the `2048` step cap
- Tooling note: this eval root contains two completed `iteration_5000` eval
  artifacts with different stock survival values, `908` and `973`. The
  survival-curve tool now collapses duplicate checkpoint rows to the latest
  artifact and marks `duplicate_stock_steps_disagree=true`. Conservative read:
  quote `+144` latest delta, and record the disagreement instead of pretending
  the eval is perfectly stable.

At `2026-05-10 09:18 EDT`, later-checkpoint evals were launched for:

- seed `50`: added `iteration_6000` to existing eval id
- seed `53`: launched eval id
  `wave10-normal-s53-0-1000-5000-7000-stock2048-seed53`
- seed `54`: added `iteration_6000` to existing eval id
- seed `55`: launched eval id
  `wave10-normal-s55-0-1000-5000-6000-stock2048-seed55`
- seed `56`: added `iteration_8000` to existing eval id
- seed `57`: added `iteration_7000` to existing eval id
- seed `60`: added `iteration_6000` to existing eval id
- seed `61`: added `iteration_8000` to existing eval id

First later-checkpoint reads already harvested:

- seed `50`: latest `iteration_6000=792`, `+28` versus `iteration_0`.
- seed `53`: flat through `iteration_7000=757`.
- seed `54`: latest `iteration_6000=763`, flat versus its lower repeated
  baseline; earlier `iteration_0` had also been observed at `823`, so keep this
  run marked as inconsistent until summarized with duplicate checks.
- seed `55`: `iteration_0=814`, `iteration_1000=758`,
  `iteration_5000=758`, `iteration_6000=878`; latest is `+64` versus
  baseline and a new run best, but this is still survival-only.

At `2026-05-10 09:10 EDT`, seeds `51` and `52` were polled again and still
exposed only selected checkpoints `0,1000`; no eval launched for them.

| seed | iteration_0 | iteration_1000 | iteration_5000 | latest poll |
| ---: | --- | --- | --- | --- |
| 50 | yes | yes | yes | `2026-05-10 09:02 EDT` |
| 51 | yes | yes | no | `2026-05-10 09:10 EDT` |
| 52 | yes | yes | no | `2026-05-10 09:10 EDT` |
| 53 | yes | yes | yes | `2026-05-10 09:02 EDT` |
| 54 | yes | yes | yes | `2026-05-10 09:02 EDT` |
| 55 | yes | yes | yes | `2026-05-10 09:02 EDT` |
| 56 | yes | yes | yes | `2026-05-10 08:51 EDT` |
| 57 | yes | yes | yes | `2026-05-10 08:54 EDT` |
| 60 | yes | yes | yes | `2026-05-10 09:02 EDT` |
| 61 | yes | yes | yes | `2026-05-10 08:54 EDT` |

Status: evals are harvested for ready seeds `50`, `53`, `54`, `55`, `56`,
`57`, `60`, and `61`, with later-checkpoint evals still running for seeds
`56`, `57`, `60`, and `61`. Remaining checkpoint watchlist: keep polling seeds
`51` and `52` for `iteration_5000.pth.tar`.

## Worker D Gap Eval - s51/s52

At `2026-05-10 16:10 EDT`, Worker D checked the previously missing wave10
normal seeds `51` and `52`. Both still exposed only `iteration_0` and
`iteration_1000`; `iteration_5000` was not visible. Stock-only survival evals
were launched for the visible checkpoints with `gpu-l4-t4-cpu40`, `--parallel`,
`--summary-only`, strict no-fallback, `50` simulations/action, and `2048` step
cap.

```text
s51 run_id: lz-visual-pong-exact-installed-0.2.0-s51-wave10-l4cpu16
s51 attempt_id: train-normal-wave10-s51-65536-ckpt1000-l4cpu16-relpath
s51 eval_id: workerD-gap-s51-stock2048-rand8-20260510
s51 selected: 0,1000
s51 eval_seed_sampler_seed: 2064836082944090358
s51 eval_seeds: 868527726,1283581883,1966928644,1887373076,1433818073,283428049,1751292526,716316417
s51 app: ap-9iv9HaSz0GYoPmVC2GiwqD
s51 manifest_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s51-wave10-l4cpu16/attempts/train-normal-wave10-s51-65536-ckpt1000-l4cpu16-relpath/eval/workerD-gap-s51-stock2048-rand8-20260510/manifest_custom_steps2048_seeds868527726-1283581883-1966928644-1887373076-1433818073-283428049-1751292526-716316417_20260510T201044Z.json

s52 run_id: lz-visual-pong-exact-installed-0.2.0-s52-wave10-l4cpu16
s52 attempt_id: train-normal-wave10-s52-65536-ckpt1000-l4cpu16-relpath
s52 eval_id: workerD-gap-s52-stock2048-rand8-20260510
s52 selected: 0,1000
s52 eval_seed_sampler_seed: 7816813580378591393
s52 eval_seeds: 913960976,1122873373,501068440,973389931,719090981,2078808365,653901647,1767734643
s52 app: ap-QAmbJWz45iWIVJNIiJRRGi
s52 manifest_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s52-wave10-l4cpu16/attempts/train-normal-wave10-s52-65536-ckpt1000-l4cpu16-relpath/eval/workerD-gap-s52-stock2048-rand8-20260510/manifest_custom_steps2048_seeds913960976-1122873373-501068440-973389931-719090981-2078808365-653901647-1767734643_20260510T200956Z.json
```

Survival-first read:

```text
s51 stock steps survived
iteration_0:    mean 759.500, median 759.5, min 757, max 764
iteration_1000: mean 767.250, median 760.5, min 757, max 819

s52 stock steps survived
iteration_0:    mean 761.375, median 762.5, min 758, max 763
iteration_1000: mean 763.750, median 762.5, min 758, max 782
```

Plain read: both s51 and s52 are essentially flat at `iteration_1000`; the
small mean bumps are from one high-start seed each and stock return stayed
`-21`. Keep watching for `iteration_5000`.

## Follow-Up

After launch:

1. Record app id and call id.
2. Poll checkpoints at `0,1000,5000`.
3. Eval with `gpu-l4-t4-cpu40`, `--group-size 1`, `--max-parallel-launches 64`.
4. Fetch manifests and run the survival curve summary:

   ```sh
   uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
     --survival-curve \
     --format tsv \
     artifacts/local/lightzero-eval-manifests/<eval-id>
   ```

5. Report survival first.
6. TODO: run diverse training configs, but judge them by survival trend, not one
   lucky seed.
