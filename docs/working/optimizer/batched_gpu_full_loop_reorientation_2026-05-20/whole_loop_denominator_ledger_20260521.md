# Whole Loop Denominator Ledger

Date: 2026-05-21

Purpose: stop mixing numbers from different experiments. Every optimizer claim
must say what path was measured, what unit it used, and what work was included.

Companion code map:
[current_code_dataflow_map_20260521.md](current_code_dataflow_map_20260521.md).

## Plain Current Read

The problem is not "render is dominant" or "render is not dominant." The
problem is that we have been timing several different systems:

1. Actual Coach training runs.
2. Stock `train_muzero` profile runs.
3. Profile-only boundary probes.

Those are all useful, but they are not the same currency.

## Currency 1: Actual Coach Training

Unit: checkpoint iterations per hour.

What it includes:

- stock LightZero `train_muzero`;
- CurvyTron env wrapper and policy observations;
- real collector/search/replay/learner work;
- RND when enabled;
- checkpoint/eval/GIF sidecars when the run enables them;
- Modal scheduling and ordinary production noise.

Current trusted read:

- Last real Coach batch `rnd-blank-sweep-fastckpt-20260519a` ran around
  `18.4k` checkpoint iterations/hour mean and `19.7k` median on L4/T4-class
  hardware.
- Older H100 `r18fresh` rows were around `31.5k` iterations/hour, but they are
  not directly comparable because the run setup changed.

Use this currency for Coach launch expectations. Do not use profile-only
roots/sec as direct training speed.

## Currency 2: Stock Full-Loop Profiles

Unit: collected env steps/sec or roots/sec inside a controlled profile.

What it includes:

- the real stock `train_muzero` entrypoint when launched through the training
  profile mode;
- LightZero public collect/search path;
- env manager behavior;
- scalar LightZero timestep materialization when the manager needs it;
- policy/search/manager overhead;
- optional profile controls such as no-death, no-RND, and zero observation.

Useful recent rows:

| shape | row | throughput | read |
| --- | --- | ---: | --- |
| L4/C256/batch64/sim8/no-RND | CPU oracle | `~586-641 steps/s` | current stock-ish CPU observation anchor |
| L4/C256/batch64/sim8/no-RND | batched GPU observation | `~893-943 steps/s` | about `1.47x-1.52x` over CPU oracle in this shape |
| C512/sim8 | real batched observation | `~1439.84 steps/s` | current useful real-observation profile row |
| C512/sim8 | zero observation | `~1805.22 steps/s` | deleting all observation work gives only about `1.25x` more in that row |

Plain read:

```text
The observation/render work got faster, but the stock loop still pays
LightZero collect/search/manager/replay-shaped costs. So full-loop speedup is
much smaller than render-only speedup.
```

This is the Amdahl point in plain language: once a component is no longer most
of the measured path, making that component faster cannot move the whole path
very much.

## Currency 3: Profile-Only Boundary Probes

Unit: roots/sec, scalar roots/sec, or surface steps/sec.

What it includes depends on the probe. These rows are for finding the next wall,
not for telling Coach what a training run will do.

Useful recent rows:

| probe | throughput | read |
| --- | ---: | --- |
| H100 resident synthetic uint8 stack, sim8, scalar off | `~13.8k roots/s` | if the batch stays alive, the hardware can go much faster |
| H100 real LightZero initial inference only | `~9466 roots/s` | the neural net root pass is not the wall |
| H100 public collect-forward sim8 | `~2304 roots/s` | public LightZero collect/search/output collapses most of the model-only headroom |
| H100 direct CTree arrays, fresh host uint8 | `~4564 roots/s` | profile-only direct arrays are about `1.85x` faster than stock facade |
| H100 direct CTree arrays, pinned repeat | `~4513 roots/s` | pinned input helps H2D but is not the big wall |
| H100 resident reuse ceiling | `~4885-5821 roots/s` | stale-input ceiling only; not a training mode |

Late P2 refresh on current code/current image:

| probe | throughput | read |
| --- | ---: | --- |
| H100 stock facade, host uint8 | `2670.68 roots/s` | public LightZero facade anchor |
| H100 direct CTree, host uint8 | `4764.06 roots/s` | direct boundary signal survives at about `1.78x` |
| H100 direct CTree, pinned uint8 | `3689.15 roots/s` | H2D cut, total wall worse |
| H100 direct CTree, resident stale ceiling | `3069.08 roots/s` | no H2D, but not useful here |

This refresh keeps the same conclusion: direct CTree over public facade is the
real profile-only signal; pinned/resident input is not the next big wall.

The important split:

```text
model root pass: fast
raw ctree traverse/backprop: small
public collect/search wrapper and output handling: large
```

In the H100 B512/A16/sim8 deep split, collect-forward spent about `35.36s`.
Only about `1.81s` was model calls. About `10.97s` was MCTS search. About
`24.40s` was outside the MCTS search call: root setup, action mask and to-play
plumbing, data conversion, action selection, visit/output assembly, and public
wrapper overhead.

## Why The Old Render Story Conflicted

The old render-heavy statements were not all fake. They were just measured in a
different path.

Render was large in:

- env-only long-trajectory rows;
- pre-persistent-render rows that redrew old trail history;
- no-death profiles where policies survive long enough for observation work to
repeat many times.

Render is much smaller in the current stock full-loop sim8 profiles because:

- persistent/incremental rendering and cursor-bound collision work already
removed a real chunk of observation wall time;
- MCTS/search/manager work is included in the denominator;
- LightZero still converts through scalar Python/NumPy-style structures at the
public boundary;
- zero-observation rows show the remaining non-observation floor.

So the correction is not "render did not matter." The correction is:

```text
Render mattered in the old denominator.
After the render/env fixes, the next measured wall moved.
```

## Current Whole-Loop Flow

Actual training flow:

```text
CurvyTron state
-> policy observation surface
-> scalar LightZero env timestep
-> stock LightZero collect/search
-> replay rows and target building
-> learner update
-> checkpoint/eval/GIF sidecars when enabled
```

Profile-only batched flow:

```text
CurvyTron state batch
-> compact batch
-> batched GPU observation stack [B,2,4,64,64]
-> optional scalar LightZero timestep edge
-> real or synthetic policy/search probe
```

The fast profile-only shape is promising because it keeps many rows together as
one batch. It is not yet the stock training shape.

## Current Priority

Stop treating renderer-only work as the main investigation. The next useful
question is:

```text
Can we keep the compact batch alive through policy/search/replay-shaped work
without changing MuZero semantics?
```

Near-term ordered work:

1. Keep `direct_ctree_arrays` profile-only until promotion gates pass.
2. Compare direct CTree arrays against stock facade with forced legal-mask,
   clear-preference, target-row, and statistical collect-row gates.
3. Run a matched full-loop profile only after the boundary is semantically
   credible.
4. Keep RND/death/autoreset as separate profile axes; do not mix those numbers
   into observation or MCTS claims.
5. Use actual Coach training speed only after a path is connected to stock
   `train_muzero` or a clearly documented replacement.

## What Would Count As A Real 5x-10x Path

Not this:

```text
same stock loop + faster renderer
```

More like this:

```text
compact actor/env batch
-> device/uint8 observation stack
-> batched model/search arrays
-> compact replay/target writes
-> scalar objects only at compatibility edges
```

That is a bigger architecture change. It may still preserve the same MuZero
algorithm, but it must pass semantic gates before it can touch Coach training.

## What To Tell Future Me

When a profile number appears, first ask:

1. Is this actual training, stock full-loop profile, or profile-only boundary?
2. Is the unit iterations/hour, env steps/sec, roots/sec, or surface steps/sec?
3. Is death on or off?
4. Is RND off, meter-only, or positive reward mode?
5. Did it call `train_muzero`?
6. Did it use stock public collect/search or direct CTree arrays?
7. Did it materialize scalar LightZero timesteps?

If those answers are missing, the number is not launch advice.
