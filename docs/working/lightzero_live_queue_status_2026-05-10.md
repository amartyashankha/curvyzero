# LightZero Live Queue Status - 2026-05-10

Last updated: `2026-05-10 01:23 EDT`.

Purpose: keep the live checkpoint/eval queue separate from the run matrix while
the normal proof lane and survival-shaped side lane are moving in parallel.

## Queue Rules

- Normal proof/control and survival-shaped side-lane runs stay separate.
- Eval summaries should lead with survival steps / cap, then stock return,
  positive rewards, strict-load status, and fallback status.
- New evals use the quiet queue helper, stock evaluator, strict no-fallback,
  `2048` eval cap, `--update-per-collect -1`, `gpu-l4-t4-cpu40`,
  `--eval-pass custom`, `--step-detail-limit 8`, `--group-size 1`, and
  `--max-parallel-launches 64`. CPU64 was invalid here; Modal caps function
  CPU at 40 cores. Older CPU8/CPU16 eval notes are historical.
- Normal seeds `24`-`27` and WaveB normal seeds `44`-`47`: readiness-gate
  until `iteration_0`, `iteration_1000`, and `iteration_5000` exist.
- Shaped seeds `30`-`37`: keep shaped eval ids; next shaped eval gate is
  `0/1000/5000` only when `iteration_5000` is visible.

## Eval Launches This Pass

I launched these shaped side-lane evals before the later shaped 5k gate update.
They are complete at the queue-helper level and wrote strict no-fallback stock
evaluator artifacts under their eval roots:

| lane | seed | eval id | selected | status |
| --- | ---: | --- | --- | --- |
| shaped | 30 | `shaped-s30-65k-0-1000-stock2048-seed30` | `0,1000` | complete; next gate `0,1000,5000` when visible |
| shaped | 31 | `shaped-s31-65k-0-1000-stock2048-seed31` | `0,1000` | complete; next gate `0,1000,5000` when visible |
| shaped | 32 | `shaped-s32-199k-0-1000-stock2048-seed32` | `0,1000` | complete; next gate `0,1000,5000` when visible |
| shaped | 33 | `shaped-s33-199k-0-1000-stock2048-seed33` | `0,1000` | complete; next gate `0,1000,5000` when visible |

Command pattern used:

```bash
uv run python scripts/lightzero_live_eval_queue.py --run-id {run_id} --attempt-id {attempt_id} --eval-id {shaped_eval_id} --compute gpu-l4-t4-cpu8 --eval-pass custom --seed {seed} --max-eval-steps 2048 --max-episode-steps 2048 --step-detail-limit 8 --max-env-step {65536_or_199000} --update-per-collect -1 --selected-iterations 0,1000 --group-size 1 --max-parallel-launches 4 --execute
```

## User-Launched Normal Evals To Avoid Duplicating

These were launched by the harvest/main thread while this pass was in progress.
Do not duplicate these exact eval ids.

| seed | session | eval id | selected | checkpoint visibility at launch | status |
| ---: | ---: | --- | --- | --- | --- |
| 13 | `15752` | `sweep65k-s13-compact-0-5k-8k-stock2048-seed13` | `0,5000,8000` | through `iteration_8000` | running/pending fetch here |
| 18 | `66292` | `sweep65k-s18-compact-0-7k-10k-13k-stock2048-seed18` | `0,7000,10000,13000` | through `iteration_13000` | running/pending fetch here |
| 19 | `26819` | `sweep65k-s19-compact-0-1k-5k-10k-stock2048-seed19` | `0,1000,5000,10000` | through `iteration_10000` | running/pending fetch here |
| 1 repeatB | `88820` | `repeatB-s1-compact-0-5k-8k-stock2048-seed1` | `0,5000,8000` | through `iteration_8000` | running/pending fetch here |

Reason: third-wave harvest found the best stock readout so far at seed `13`
`iteration_5000` with stock steps `866/2048`, stock return `-20`, and one
stock positive reward. Seed `18` at `7000` and seed `19` at `1000` had
stock-step-only bumps with return still `-21`; repeatB `5000` was flat.

## Normal Proof-Lane Readiness

| seed | run id | attempt id | latest visible | queue action |
| ---: | --- | --- | --- | --- |
| 13 | `lz-visual-pong-exact-installed-0.2.0-s13-sweep65k-l4` | `train-normal-sweep65k-s13-ckpt1000-spawn-l4-relpath` | through `8000` per user launch note | user launched compact `0,5000,8000`; next useful gate `10000+` or `16000/latest` |
| 18 | `lz-visual-pong-exact-installed-0.2.0-s18-sweep65k-h100cpu16` | `train-normal-sweep65k-s18-ckpt1000-spawn-h100cpu16-relpath` | through `13000` per user launch note | user launched compact `0,7000,10000,13000`; next useful gate `16000/latest` |
| 19 | `lz-visual-pong-exact-installed-0.2.0-s19-sweep65k-h100cpu16` | `train-normal-sweep65k-s19-ckpt1000-spawn-h100cpu16-relpath` | through `10000` per user launch note | user launched compact `0,1000,5000,10000`; next useful gate `16000/latest` |
| 1 repeatB | `lz-visual-pong-exact-installed-0.2.0-s1-repeatB-65536-l4` | `train-normal-repeatB-s1-65536-ckpt1000-spawn-l4-relpath` | through `8000` per user launch note | user launched compact `0,5000,8000`; next useful gate `10000+` |
| 20 detachedB | `lz-visual-pong-exact-installed-0.2.0-s20-long199k-h100cpu16-ckpt1000-detachedB` | `train-normal-long199k-s20-ckpt1000-detachedB-h100cpu16-relpath` | `0,1000,2000,3000,4000,5000` | hold; harvest says early long s20/s21 flat, next useful gate `10000+` |
| 21 detachedB | `lz-visual-pong-exact-installed-0.2.0-s21-long199k-h100cpu16-ckpt1000-detachedB` | `train-normal-long199k-s21-ckpt1000-detachedB-h100cpu16-relpath` | `0,1000,2000,3000,4000,5000,6000,7000` | hold; harvest says early long s20/s21 flat, next useful gate `10000+` |
| 24 | `lz-visual-pong-exact-installed-0.2.0-s24-sweep65k-l4cpu16` | `train-normal-sweep65k-s24-ckpt1000-spawn-l4cpu16-relpath` | `0,1000,2000` | hold until `0,1000,5000` exists |
| 25 | `lz-visual-pong-exact-installed-0.2.0-s25-sweep65k-l4cpu16` | `train-normal-sweep65k-s25-ckpt1000-spawn-l4cpu16-relpath` | `0,1000,2000,3000` | hold until `0,1000,5000` exists |
| 26 | `lz-visual-pong-exact-installed-0.2.0-s26-sweep65k-l4cpu16` | `train-normal-sweep65k-s26-ckpt1000-spawn-l4cpu16-relpath` | `0,1000,2000,3000` | hold until `0,1000,5000` exists |
| 27 | `lz-visual-pong-exact-installed-0.2.0-s27-sweep65k-l4cpu16` | `train-normal-sweep65k-s27-ckpt1000-spawn-l4cpu16-relpath` | `0,1000,2000,3000` | hold until `0,1000,5000` exists |

## WaveB Normal Watch List

All are normal proof/control lane, L4/T4+CPU16, max env step `65536`,
checkpoint cadence `1000`, no survival shaping.

| seed | app id | call id | run id | attempt id | latest visible | next gate |
| ---: | --- | --- | --- | --- | --- | --- |
| 44 | `ap-lpHqsvq0afWz4NrwV7Hkq6` | `fc-01KR851KR398YV1XKMZA8YB7C8` | `lz-visual-pong-exact-installed-0.2.0-s44-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s44-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |
| 45 | `ap-sJCA46yqOFhhLJw2atGGAG` | `fc-01KR851KPZPNGZBJK0KK8KSNC2` | `lz-visual-pong-exact-installed-0.2.0-s45-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s45-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |
| 46 | `ap-gwXD6wGpwxQa6fXRSQoWcV` | `fc-01KR851KTQ6D7KPTYR3PBKVC6D` | `lz-visual-pong-exact-installed-0.2.0-s46-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s46-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |
| 47 | `ap-kYSaD26oLzFTjH4rNfEWIP` | `fc-01KR851KNWABV8JV8KZ8RY9BFZ` | `lz-visual-pong-exact-installed-0.2.0-s47-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s47-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |

## Survival-Shaped Watch List

Seeds `30`-`33` already have early `0/1000` shaped evals from this pass.
Future shaped evals should wait for `iteration_5000` and use `0,1000,5000`.

| seed | app id | call id | run id | attempt id | latest visible | next gate |
| ---: | --- | --- | --- | --- | --- | --- |
| 30 | `ap-Y012lkdhQsoCBdfTZwEOxe` | `fc-01KR84DT9XJQ6950G8FNTFMFA4` | `lz-visual-pong-survival-shaped-step0p001-s30-65k-l4cpu16` | `train-survival-shaped-step0p001-s30-65536-ckpt1000-spawn-l4cpu16-relpath` | `0,1000,2000` | hold until `0,1000,5000` |
| 31 | `ap-hXNZ3KegIzNvUunuhepVSL` | `fc-01KR84DTEBC1K8EY1VFK9T7V5X` | `lz-visual-pong-survival-shaped-step0p001-s31-65k-l4cpu16` | `train-survival-shaped-step0p001-s31-65536-ckpt1000-spawn-l4cpu16-relpath` | `0,1000,2000,3000` | hold until `0,1000,5000` |
| 32 | `ap-sfp8Npo0Jwg8zwIGbEEMwG` | `fc-01KR84DTP2M8M9VPYJ5PANGF5C` | `lz-visual-pong-survival-shaped-step0p001-s32-199k-h100cpu16` | `train-survival-shaped-step0p001-s32-199000-ckpt1000-spawn-h100cpu16-relpath` | `0,1000,2000,3000` | hold until `0,1000,5000` |
| 33 | `ap-dszTPF0BR1k4y0a80eLItB` | `fc-01KR84DTNJ6WT5SSE4KCM38BEN` | `lz-visual-pong-survival-shaped-step0p001-s33-199k-h100cpu16` | `train-survival-shaped-step0p001-s33-199000-ckpt1000-spawn-h100cpu16-relpath` | `0,1000,2000,3000` | hold until `0,1000,5000` |
| 34 | `ap-alqyQEfGFX52Tfd7qjdzMO` | `fc-01KR851KQZ051WDVJYZ1AYN7HT` | `lz-visual-pong-survival-shaped-step0p001-s34-65k-l4cpu16` | `train-survival-shaped-step0p001-s34-65536-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |
| 35 | `ap-qVvUTu6yCMGi4BQw72tvNv` | `fc-01KR851KQ2AM0Q363QSTQVB039` | `lz-visual-pong-survival-shaped-step0p001-s35-65k-l4cpu16` | `train-survival-shaped-step0p001-s35-65536-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |
| 36 | `ap-HLFdsAnolt4NA3RMvecO3N` | `fc-01KR851KSC7FK8EV8TS6QG88F5` | `lz-visual-pong-survival-shaped-step0p001-s36-65k-l4cpu16` | `train-survival-shaped-step0p001-s36-65536-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |
| 37 | `ap-6kNm5Ae2lFCajuU3JKiSlG` | `fc-01KR851KRS1AP0KYKEPKQM84XW` | `lz-visual-pong-survival-shaped-step0p001-s37-65k-l4cpu16` | `train-survival-shaped-step0p001-s37-65536-ckpt1000-waveB-l4cpu16-relpath` | `iteration_0` only | hold until `0,1000,5000` |

No pytest was run.
