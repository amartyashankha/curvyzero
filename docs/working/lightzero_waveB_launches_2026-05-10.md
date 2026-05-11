# LightZero WaveB Launches - 2026-05-10

Purpose: record the extra parallel training runs launched after the eval stdout
quieting patch landed. These runs are meant to keep artifacts flowing while the
older evals and checkpoint polls finish.

## Claim

Eight new Modal training jobs were spawned: four normal LightZero Atari Pong
control runs and four survival-shaped side-lane runs. All use L4/T4 with 16
Modal CPUs, `65536` max env steps, and checkpoint cadence `1000`.

## Non-claim

These launches do not prove Pong learning, solved Pong, CurvyTron readiness, or
that survival-shaped reward is the right final objective. They only create more
checkpoints to evaluate.

## Normal Proof Lane

Normal runs use stock/control LightZero Atari Pong reward. Eval them as proof
lane candidates.

| Seed | App | Call | Run id | Attempt id |
| ---: | --- | --- | --- | --- |
| 44 | `ap-lpHqsvq0afWz4NrwV7Hkq6` | `fc-01KR851KR398YV1XKMZA8YB7C8` | `lz-visual-pong-exact-installed-0.2.0-s44-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s44-ckpt1000-waveB-l4cpu16-relpath` |
| 45 | `ap-sJCA46yqOFhhLJw2atGGAG` | `fc-01KR851KPZPNGZBJK0KK8KSNC2` | `lz-visual-pong-exact-installed-0.2.0-s45-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s45-ckpt1000-waveB-l4cpu16-relpath` |
| 46 | `ap-gwXD6wGpwxQa6fXRSQoWcV` | `fc-01KR851KTQ6D7KPTYR3PBKVC6D` | `lz-visual-pong-exact-installed-0.2.0-s46-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s46-ckpt1000-waveB-l4cpu16-relpath` |
| 47 | `ap-kYSaD26oLzFTjH4rNfEWIP` | `fc-01KR851KNWABV8JV8KZ8RY9BFZ` | `lz-visual-pong-exact-installed-0.2.0-s47-sweep65k-l4cpu16-waveB` | `train-normal-sweep65k-s47-ckpt1000-waveB-l4cpu16-relpath` |

## Survival-Shaped Side Lane

Shaped runs add `0.001` reward per non-terminal step. Keep them separate from
the normal proof lane. Their main use is checking whether steps survived gives
an earlier learning signal.

| Seed | App | Call | Run id | Attempt id |
| ---: | --- | --- | --- | --- |
| 34 | `ap-alqyQEfGFX52Tfd7qjdzMO` | `fc-01KR851KQZ051WDVJYZ1AYN7HT` | `lz-visual-pong-survival-shaped-step0p001-s34-65k-l4cpu16` | `train-survival-shaped-step0p001-s34-65536-ckpt1000-waveB-l4cpu16-relpath` |
| 35 | `ap-qVvUTu6yCMGi4BQw72tvNv` | `fc-01KR851KQ2AM0Q363QSTQVB039` | `lz-visual-pong-survival-shaped-step0p001-s35-65k-l4cpu16` | `train-survival-shaped-step0p001-s35-65536-ckpt1000-waveB-l4cpu16-relpath` |
| 36 | `ap-HLFdsAnolt4NA3RMvecO3N` | `fc-01KR851KSC7FK8EV8TS6QG88F5` | `lz-visual-pong-survival-shaped-step0p001-s36-65k-l4cpu16` | `train-survival-shaped-step0p001-s36-65536-ckpt1000-waveB-l4cpu16-relpath` |
| 37 | `ap-6kNm5Ae2lFCajuU3JKiSlG` | `fc-01KR851KRS1AP0KYKEPKQM84XW` | `lz-visual-pong-survival-shaped-step0p001-s37-65k-l4cpu16` | `train-survival-shaped-step0p001-s37-65536-ckpt1000-waveB-l4cpu16-relpath` |

## Eval Gate

- First check: wait for `iteration_0.pth.tar` and `iteration_1000.pth.tar`.
- Better check: if `iteration_5000.pth.tar` appears soon, eval
  `0/1000/5000` instead of only `0/1000`.
- Use strict no-fallback stock evaluator with a 2048-step cap.
- Report stock steps survived first, then stock return and stock reward counts.
- Do not mix shaped and normal tables without a clear label.
