# Speed Timeline

Date: 2026-05-23

Purpose: keep the speed story honest. These rows are different currencies
unless stated otherwise.

## Actual Coach Training

These are the numbers that matter for wall-clock training jobs.

| Period / batch | Path | Hardware | Main shape | Speed read | Trust |
| --- | --- | --- | --- | --- | --- |
| CZ26 136-run batch | stock `train_muzero` | L4/T4 class, CPU40 | C256/N256, batch64, sim8, browser_lines + simple_symbols, sidecars on | derived completed rows around `15k` learner iters/hour | real Coach run, but not an official speed table |
| Later RND blank sweep | stock `train_muzero` | L4 | similar Coach lane, RND blank/meter context | documented mean `~18.4k`, median `~19.7k` learner iters/hour | real Coach run |
| Older r18fresh | stock `train_muzero` | H100 | older setup | documented `~31.5k` learner iters/hour | real run, but hardware/setup differ |

Plain conclusion: no optimizer candidate has yet been proven to beat CZ26 on
this exact denominator.

## Stock Full-Loop Profiles

These call the real LightZero training entry or profile that path closely
enough to compare within a matched profile batch.

| Candidate | Example speed | Delta | What it proves |
| --- | ---: | ---: | --- |
| Stock no-RND profile | `433` steps/sec | baseline | profile denominator only |
| Direct output-fast no-RND | `566` steps/sec | `1.31x` | output/search hook helps inside full-loop profile |
| Stock RND profile | `351` steps/sec | baseline | RND profile denominator only |
| Direct output-fast RND | `449` steps/sec | `1.28x` | same win survives RND profile shape |
| Latest H100 no-RND warm Gate A | `929 -> 1204` steps/sec | `1.30x` | strict matched stock `train_muzero` profile |
| Latest L4 no-RND warm Gate A | `591 -> 847` steps/sec | `1.43x` | strict matched stock `train_muzero` profile |
| Latest H100/L4 RND-meter warm Gate A | `1.28x` / `1.56x` | sidecar | fallback profile currency until raw collector-step telemetry is fixed |
| CPU-oracle observation profile | `586` steps/sec | baseline | current profile observation baseline |
| Batched GPU observation profile | `894` steps/sec | `1.5x` | observation path can be faster in profile |

Plain conclusion: these are useful, but still not the same as a completed Coach
training batch.

## Compact / Profile-Only Architecture Probes

These do not prove Coach speed by themselves.

| Candidate | Example speed | Delta | Why it is not enough |
| --- | ---: | ---: | --- |
| Direct CTree compact profile | `~8k-9k` steps/sec on large H100 rows | baseline | profile-only, not trainer |
| MCTX/JAX compact profile | up to `~19k` steps/sec in clean rows | `~2x+` | semantic mismatch with LightZero CTree remains |
| Service-tax / mock / zero-observation rows | varies | ceiling only | deliberately removes real work |

Plain conclusion: compact/MCTX rows are architecture evidence, not launch
recommendations.

## Current Honest Speedup Claim

The honest claim is:

- proven actual Coach speedup from optimizer work: `0x` so far, because it has
  not been promoted and measured on that denominator;
- plausible near-term full-loop gain if profile wins promote cleanly:
  `1.2x-1.5x`;
- plausible stronger compact/dataflow gain if compact ownership replaces hot
  scalar boundaries: `1.5x-2.5x` first, maybe more after search ownership
  changes;
- `5x-10x` requires a larger architecture change, not the patches already
  proven.
