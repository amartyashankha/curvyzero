# Actual Training Speed Read

Date: 2026-05-21

Purpose: keep actual Coach training speed separate from profile-only optimizer
speed. Do not mix learner iterations/hour, collected env steps/sec, and isolated
render/stack timings without naming the denominator.

## Current Plain Read

The last real Coach batch reviewed here is the RND blank sweep:

```text
rnd-blank-sweep-fastckpt-20260519a
gpu-l4-t4-cpu40
C256 / N256
batch_size=64
num_simulations=8
browser_lines + simple_symbols + cpu_oracle
stock train path, GIF/eval sidecars on
```

It did not use the profile-only batched GPU observation renderer.

Using the local artifact launch timestamp
`2026-05-19T23:43:33Z` and latest checkpoint mtimes:

| row | latest checkpoint | elapsed hours | checkpoint iters/hour |
| --- | ---: | ---: | ---: |
| stock-w0 | 77500 | 3.36 | 23036 |
| meter-w0 | 77500 | 5.38 | 14404 |
| rnd-w0p003 | 77500 | 5.33 | 14546 |
| rnd-w0p01 | 77500 | 3.93 | 19727 |
| rnd-w0p03 | 77500 | 3.78 | 20506 |
| rnd-w0p1 | 90611 | 4.81 | 18855 |
| rnd-w0p3 | 84231 | 4.23 | 19921 |
| rnd-w0p6 | 77500 | 3.76 | 20597 |
| rnd-w1p0 | 77500 | 5.39 | 14367 |

Summary:

```text
mean:   ~18.4k checkpoint iters/hour
median: ~19.7k checkpoint iters/hour
range:  ~14.4k to ~23.0k checkpoint iters/hour
```

The older r18fresh H100 batch was documented at about `31.5k` learner
iterations/hour average. So this L4/RND blank-sweep batch was roughly:

```text
0.58x the r18fresh learner-iteration rate on average
about 1.7x longer wall time for the same learner-iteration count
```

That is not an optimizer regression by itself: the batches used different
hardware, different sidecars/checkpoint cadence, and RND/reward-model paths.
It does mean we should not tell Coach that the new GPU-observation work has
already sped up real training.

## Profile-Only Speed Evidence

The optimizer speed work is currently ahead of production training:

- Profile-only batched GPU manager C512 real render: about `1439.84` collected
  env steps/sec.
- Matching C512 zero-observation ceiling: about `1805.22` collected env
  steps/sec.
- Current render/observation headroom in that one-process manager row is
  therefore only about `1.25x`.
- Hybrid actor plus central GPU-render profile rows reached thousands of scalar
  timesteps/sec, but they do not call `train_muzero` and exclude learner,
  replay, RND, and real subprocess behavior.

Plain conclusion:

```text
Actual Coach run speedup so far: not proven.
Best older full-loop profile gain: about 1.5x-2x versus some CPU-oracle
profile controls, but this is dated and denominator-specific.

Fresh 2026-05-22 train_muzero profile evidence:
  no-RND stock:              433.17 steps/sec
  no-RND direct output-fast: 566.19 steps/sec
  RND stock hash-fixed:      351.02 steps/sec
  RND direct hash-fixed:     448.52 steps/sec

So the current direct-search/output-fast probe has about 1.3x profile-loop
evidence. It is still profile-only and not a Coach default.

Large future wins require keeping bigger batches alive across env, observation,
policy/search, replay, and learner boundaries, not only making a render kernel
faster.
```

## Fresh Current-Code Profile Check

After writing the first read, a matched L4/C256/batch64/sim8 profile grid ran
against current code:

```text
opt-current-l4-c256-ab-rnd-20260521a
opt-current-l4-c256-ab-repeat-20260521a
```

No-RND combined means across three rows:

| manager | mean profile env steps/sec |
| --- | ---: |
| subprocess CPU oracle | `586.14` |
| batched GPU observation | `893.55` |
| zero observation | `964.39` |

Current profile estimate:

```text
batched GPU observation is ~1.52x faster than subprocess CPU oracle
in the current L4/C256 no-death profile shape.
```

This is still not actual training speed. It does say the batched GPU
observation path is a real speed candidate. It also says observation-only
headroom is now small in this profile shape: zero-observation is only about
`1.08x` above batched GPU observation when all three no-RND rows are included.

## Next Speed Gate

Before recommending any new default to Coach:

1. Keep actual Coach training speed separate from profile-only speed.
2. For optimizer promotion, use matched stock `train_muzero` profile A/B rows:
   stock versus direct output-fast, no-RND and `rnd_meter_v0`.
3. Require the direct-hook promotion gates before Coach-facing advice:
   [direct_ctree_promotion_gates_20260522.md](direct_ctree_promotion_gates_20260522.md).
4. Report learner iterations/hour and collected env steps/sec separately.
5. Keep eval/GIF/checkpoint sidecars explicit in the denominator.
