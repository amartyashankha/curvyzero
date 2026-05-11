# LightZero CPU40 Eval Wave - 2026-05-10

Last updated: `2026-05-10 08:30 EDT`.

Purpose: evaluate existing live Pong runs before launching the next diverse
training wave. The signal is stock steps survived versus the same run's
`iteration_0`. Return and score are secondary.

## Correction

CPU64 was wrong. Modal rejected `cpu=64` with:

`Function CPU request out of bounds. Must be between 0.125 and 40 cores.`

Code now uses:

- eval compute: `gpu-l4-t4-cpu40`
- training compute options: `gpu-l4-t4-cpu40`, `gpu-h100-cpu40`
- eval fan-out: `--max-parallel-launches 64`

So "64" means up to 64 independent Modal eval calls, not 64 CPUs inside one
function.

## Launched Eval Queue Sessions

Normal proof lane, selected `0,1000,5000`, `2048` cap, strict stock evaluator,
CPU40 eval compute:

| seed | eval id | local session |
| ---: | --- | ---: |
| 24 | `normal-s24-0-1000-5000-stock2048-seed24` | `18822` |
| 25 | `normal-s25-0-1000-5000-stock2048-seed25` | `97314` |
| 26 | `normal-s26-0-1000-5000-stock2048-seed26` | `23466` |
| 27 | `normal-s27-0-1000-5000-stock2048-seed27` | `74956` |

Survival-shaped side lane, selected `0,1000,5000` for seeds with `5000`
visible, strict stock evaluator, CPU40 eval compute:

| seed | eval id | local session |
| ---: | --- | ---: |
| 30 | `shaped-s30-0-1000-5000-stock2048-seed30` | `5551` |
| 31 | `shaped-s31-0-1000-5000-stock2048-seed31` | `20005` |
| 32 | `shaped-s32-0-1000-5000-stock2048-seed32` | `78879` |
| 33 | `shaped-s33-0-1000-5000-stock2048-seed33` | `86499` |

Survival-shaped WaveB early retry, selected `0,1000`, strict stock evaluator,
CPU40 eval compute:

| seed | eval id | local session |
| ---: | --- | ---: |
| 34 | `shaped-s34-0-1000-stock2048-seed34` | `21995` |
| 35 | `shaped-s35-0-1000-stock2048-seed35` | `44643` |
| 36 | `shaped-s36-0-1000-stock2048-seed36` | `61298` |
| 37 | `shaped-s37-0-1000-stock2048-seed37` | `41771` |

## Completed Read

Harvested available roots from `curvyzero-runs` into
`artifacts/local/lightzero-eval-manifests/<eval-id>` and summarized each root
with:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --baseline-deltas --format tsv <local-eval-dir>
```

Summary files were written as `summary_baseline_deltas.tsv` under each fetched
root. All summarized rows report `strict_load=true` and `fallback_used=false`.

Good signal still means later checkpoints survive longer than their own
`iteration_0` on stock evaluator reads. Return is secondary.

## CPU40 Harvest Status

Claim: the CPU40 eval wave found two useful stock-survival bumps: normal seed
`27` and shaped seed `30`.

Non-claim: this is not solved Pong and not stable learning. Return stayed
`-21` and stock positive rewards stayed `0` across this wave.

### Normal proof lane

All four normal eval roots completed and were fetched.

| seed | eval id | fetched rows | stock steps survived vs `iteration_0` | stock return |
| ---: | --- | --- | --- | --- |
| 24 | `normal-s24-0-1000-5000-stock2048-seed24` | `0,1000,5000` | flat: `763 -> 763 -> 763` | all `-21` |
| 25 | `normal-s25-0-1000-5000-stock2048-seed25` | `0,1000,5000` | flat: `758 -> 758 -> 758` | all `-21` |
| 26 | `normal-s26-0-1000-5000-stock2048-seed26` | `0,1000,5000` | flat: `763 -> 763 -> 763` | all `-21` |
| 27 | `normal-s27-0-1000-5000-stock2048-seed27` | `0,1000,5000` | `758 -> 758 -> 848`; best delta `+90` at `iteration_5000` | all `-21` |

Normal read: seed 27 has the only stock-survival gain in this batch. It did
not get a stock-score gain: positive rewards stayed `0`.

### Survival-shaped side lane

All four exact `0,1000,5000` shaped roots completed and were fetched.

| seed | eval id | fetched rows | stock steps survived vs `iteration_0` | stock return |
| ---: | --- | --- | --- | --- |
| 30 | `shaped-s30-0-1000-5000-stock2048-seed30` | `0,1000,5000` | `763 -> 763 -> 824`; best delta `+61` at `iteration_5000` | all `-21` |
| 31 | `shaped-s31-0-1000-5000-stock2048-seed31` | `0,1000,5000` | flat: `758 -> 758 -> 758` | all `-21` |
| 32 | `shaped-s32-0-1000-5000-stock2048-seed32` | `0,1000,5000` | high start then lower: `849 -> 761 -> 761` | all `-21` |
| 33 | `shaped-s33-0-1000-5000-stock2048-seed33` | `0,1000,5000` | flat: `762 -> 762 -> 762` | all `-21` |

Shaped read: seed 30 has a stock-survival gain at `iteration_5000`, but no
stock-score gain. Seed 32 started high at `iteration_0` and then dropped, so it
is not a training gain.

### Survival-shaped WaveB early retry

All four `0,1000` WaveB retry roots completed and were fetched.

| seed | eval id | fetched rows | stock steps survived vs `iteration_0` | stock return |
| ---: | --- | --- | --- | --- |
| 34 | `shaped-s34-0-1000-stock2048-seed34` | `0,1000` | flat: `758 -> 758` | all `-21` |
| 35 | `shaped-s35-0-1000-stock2048-seed35` | `0,1000` | flat: `760 -> 760` | all `-21` |
| 36 | `shaped-s36-0-1000-stock2048-seed36` | `0,1000` | flat: `764 -> 764` | all `-21` |
| 37 | `shaped-s37-0-1000-stock2048-seed37` | `0,1000` | drops: `789 -> 761` | all `-21` |

WaveB read: no early shaped retry improves at `iteration_1000`. Seed 37 drops
from a high baseline.

## Next 10-Run Wave

The CPU40 read was enough to keep exploration moving, but not enough to claim
Pong learning. The ten-run launch wave has now spawned all requested runs:
normal seeds `50`-`57` and shaped telemetry seeds `60` and `61`.

Details live in
[lightzero_10run_launch_wave_2026-05-10.md](lightzero_10run_launch_wave_2026-05-10.md).

Next decision should use survival first: s27 normal `+90` and s30 shaped `+61`
are the only useful stock-step bumps here. Both still have stock return `-21`
and `0` positive rewards, so return does not confirm Pong learning.
