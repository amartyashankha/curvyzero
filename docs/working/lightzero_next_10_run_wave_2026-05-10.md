# LightZero Next 10 Run Wave - 2026-05-10

Status: launching as an exploration wave after partial CPU40 eval readout.

## Gate First

Current read from the later normal eval harvest: survival bumps exist, but they
are not stable enough to scale. Seed `13` improved at `iteration_5000` and
fell back by `8000`; seed `18` improved at `7000`, fell back at `10000`, then
improved again at `13000`; seed `19` improved at `5000` and fell back by
`10000`; repeatB seed `1` stayed flat through `8000`.

The stricter gate below is still the clean proof gate. However, the partial
CPU40 eval wave now shows enough weak survival signal to justify a mixed
exploration wave while eval harvest continues:

- normal seed `27`: stock survival `758 -> 848` by `iteration_5000`;
- shaped seed `30`: stock survival `763 -> 824` by `iteration_5000`;
- normal seeds `24`, `25`, and `26` are flat so far;
- earlier seeds `13`, `18`, and `19` had survival bumps that later fell back.

This is not stable proof. It is enough to keep compute busy and test seed /
compute diversity.

The clean proof gate remains:

- at least two normal proof-lane runs with `stock_steps_survived` above the
  same-run `iteration_0` at a late checkpoint (`10000+`, `16000/latest`
  preferred);
- stock evaluator rows, strict checkpoint load, `fallback_used=false`,
  `2048` eval cap, and `--update-per-collect -1`;
- survival-step improvement that is still present at the latest checked row or
  repeats at more than one late checkpoint.

Return and score are secondary. A `-21` to `-20` return move is not enough by
itself. Shaped runs cannot open the normal proof-lane scale gate.

## Launch Wave: 10 Runs

Keep the wave diverse, but mostly normal. Use fresh seeds that do not overlap
with the active `1`, `3`, `10`-`21`, `24`-`27`, `30`-`37`, or `44`-`47`
sets.

| slot | lane | seed | training compute | max env step | checkpoint cadence | purpose |
| --- | --- | ---: | --- | ---: | ---: | --- |
| N1 | normal | 50 | L4/T4 + CPU16 | 65536 | 1000 | normal seed diversity |
| N2 | normal | 51 | L4/T4 + CPU16 | 65536 | 1000 | normal seed diversity |
| N3 | normal | 52 | L4/T4 + CPU16 | 65536 | 1000 | normal seed diversity |
| N4 | normal | 53 | L4/T4 + CPU16 | 65536 | 1000 | normal seed diversity |
| N5 | normal | 54 | L4/T4 + CPU40 | 65536 | 1000 | CPU scaling check |
| N6 | normal | 55 | L4/T4 + CPU40 | 65536 | 1000 | CPU scaling check |
| N7 | normal | 56 | H100 + CPU16 | 199000 | 1000 | longer-curve check |
| N8 | normal | 57 | H100 + CPU40 | 199000 | 1000 | longer curve plus CPU scaling |
| S1 | shaped side lane | 60 | L4/T4 + CPU16 | 65536 | 1000 | shaped telemetry only |
| S2 | shaped side lane | 61 | H100 + CPU16 | 199000 | 1000 | shaped telemetry only |

CPU40 is valid; CPU64 is invalid in this Modal workspace. CPU40 training runs
are probes, not proven faster yet. Use their progress snapshots to compare
checkpoint arrival time against CPU16 runs.

## Eval Cadence For The Wave

Use the current eval contract for every checkpoint readout: stock evaluator,
strict no-fallback load, `2048` cap, `gpu-l4-t4-cpu40`, `--group-size 1`,
`--max-parallel-launches 64`, and `--update-per-collect -1`.

Evaluate each run at `0,1000,5000` once those checkpoints exist. If and only if
stock survival improves over same-run `iteration_0`, continue that run at
`8000,10000,13000,16000/latest`; for long H100 runs, add `20000/latest` if the
curve is still alive.

Stop scaling if late checkpoints fall back to baseline across the normal lane.
Keep shaped results separate and report them as telemetry, never as proof that
normal Pong learned.
