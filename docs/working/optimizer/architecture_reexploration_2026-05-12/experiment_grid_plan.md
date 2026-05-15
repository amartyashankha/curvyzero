# CurvyTron Architecture Experiment Grid Plan

Date: 2026-05-12

Scope: aggressive but analyzable profiling grid for CurvyTron training
architecture. This is an Optimizer measurement plan, not a learning-quality
claim.

Trusted control path remains:

```text
stock LightZero train_muzero
  + env_variant=source_state_fixed_opponent
  + frozen checkpoint opponent
```

Do not mutate live Coach training runs. Do not use old custom
`two-seat-selfplay` timings as current evidence.

## Principles

- Use experiment families, not one giant Cartesian grid.
- Change one main architectural axis per family, with a small number of anchor
  profiles shared across families.
- Every job writes the same summary schema, even when the job is a stock
  trainer profile, collect-only fanout run, or MCTX benchmark.
- Treat speed as component timing plus denominator metadata. A faster
  `steps/sec` without render mode, death setting, MCTS sims, checkpoint, and
  worker shape is not usable.
- Scale only after the first wave is parseable and shows a real bottleneck.

## Shared Anchors

All first-wave jobs should share one immutable checkpoint/opponent pair unless
the checkpoint axis is the explicit family under test.

Default anchor:

```text
env_variant: source_state_fixed_opponent
opponent: frozen LightZero checkpoint
death mode: normal death enabled
render mode: browser_lines
env manager: subprocess unless the family tests env manager
MCTS simulations: current trusted stock value, plus one lower anchor if needed
collector width: current trusted stock value, plus ladder values below
artifact/eval/GIF: off unless required by stock path and reported separately
```

Recommended profile duration: long enough for warmup and at least several
complete episodes or a fixed minimum number of decisions. Do not compare tiny
early-exit jobs against steady jobs except as smoke tests.

## Experiment Families

### Family A: Stock Iteration Anatomy

Question: where does the trusted stock loop spend wall time?

Run 3-4 jobs:

| Job | Collector width | Env manager | Sims | Render | Death | Shape |
| --- | ---: | --- | ---: | --- | --- | --- |
| A0 smoke | current | current | current | browser_lines | on | current CPU/GPU |
| A1 anchor | current | subprocess | current | browser_lines | on | current CPU/GPU |
| A2 no-death lens | current | subprocess | current | browser_lines | off | current CPU/GPU |
| A3 fast-render lens | historical/profile only | subprocess | historical/profile only | body_circles_fast | off | control only |

Purpose: establish timing buckets and denominator correctness before scaling.
A2/A3 are bottleneck lenses, not fidelity claims.

### Family B: Collector Width Ladder

Question: how far does local stock collection scale before CPU, process, IPC,
or replay overhead wins?

Run 4-6 jobs:

```text
collector_env_num: 8, 16, 32, 64
optional: 96 only if 64 is clean and still rising
fixed: checkpoint, opponent, render, death, MCTS sims, CPU/GPU shape
```

If existing trusted width is not one of these, include it as an anchor. Keep
episode caps and learner caps identical. This family should report per-worker
telemetry whenever available.

### Family C: Env Manager And Render Surface

Question: is the bottleneck env manager overhead, render/observation, or search?

Run 4-8 jobs using two collector-width anchors from Family B: one modest width
and one near the best non-saturated width.

Axes:

```text
env manager: subprocess, base/sync if supported
render mode: browser_lines, body_circles_fast
death mode: on, off
```

Use a fractional grid, not all combinations. First compare env manager at
`browser_lines/death on`; then compare render at the same manager; then use
no-death only as a long-trajectory render lens.

### Family D: MCTS Simulation Ladder

Question: when do model/search costs dominate enough to justify batching or
MCTX work?

Run 4-6 jobs:

```text
num_simulations: 4, 8, 16, 32
collector width: best parseable Family B width plus one low-width anchor
fixed: checkpoint, opponent, render, death, env manager, CPU/GPU shape
```

Report decisions/sec and simulations/sec. The useful output is not just the
fastest sim count; it is the slope of wall time versus simulations and whether
GPU utilization rises.

### Family E: CPU/GPU Shape

Question: does more GPU help, or is the collector still CPU-bound?

Run 4-8 jobs only after Families A-D are parseable.

Axes:

```text
GPU: cheap/small, L4-class, larger GPU only if available without interfering
CPU: current default, higher CPU allocation
collector width: best Family B width
MCTS sims: one low and one high anchor
```

Do not test multi-GPU unless there is evidence the relevant LightZero search or
learner work actually uses it. If tested, report it as a separate shape, not as
an assumed upgrade.

### Family F: Checkpoint And Opponent Config

Question: are timings sensitive to policy strength or opponent setup?

Run 3-6 jobs:

```text
checkpoint K0: known trusted frozen baseline
checkpoint K1: newer/better or longer-lived policy if Coach supplies it
opponent: same frozen checkpoint, separate frozen checkpoint if used in proof
opponent device: CPU-safe subprocess default; CUDA only in a deliberately safe
  non-fork shape
```

Keep this family small. It exists to catch wrong conclusions from a bad-policy
short-episode checkpoint, not to make learning claims.

### Family G: Collect-Only Fanout

Question: does coarse actor fanout produce searched chunks faster, and can
those chunks be merged/imported?

Run 4-6 groups, not trainer jobs:

```text
fanout N actors: 1, 2, 4, 8
per actor collector_env_num: 1 or 2 first; then best local width if clean
fixed: checkpoint, opponent, render, death, MCTS sims, env manager
outputs: native LightZero GameSegment chunks plus strict manifests
merge: validate manifests, push into MuZeroGameBuffer, optionally sample batch
```

No learner train call, no checkpoint publish, no learning claim. If N=8 is
messy or merge/import dominates, stop before larger actor counts.

### Family H: MCTX Visual-Root Benchmark

Question: can a GPU-native batched search primitive beat or match the relevant
LightZero search bucket?

Run the narrow MCTX matrix from `mctx_visual_root_benchmark_plan.md`:

| Env rows B | Roots R | Sims | Hidden | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 8 | 16 | 8 | 64 | compile smoke |
| 16 | 32 | 16 | 64 | first useful timing |
| 64 | 128 | 16 | 64 | LightZero comparison bucket |
| 64 | 128 | 32 | 64 | search scaling |

Render comparison is optional and should be a second pass:
`browser_lines` versus `body_circles_fast`. This benchmark stays isolated from
the trainer.

## First Wave: 12-20 Jobs

Run this before any 50-100 job expansion:

| Block | Jobs | Purpose |
| --- | ---: | --- |
| Family A | 3 | trusted timing anatomy and denominator check |
| Family B | 4 | collector width ladder: 8/16/32/64 or local equivalent |
| Family C | 2-3 | env manager/render/death lens at selected width |
| Family D | 2-3 | MCTS sims slope |
| Family G | 1-2 groups | fanout N=1/2, optionally N=4, with merge/import |
| Family H | 1-2 | MCTX smoke plus first useful visual timing |

Target total: 13-17 first-wave jobs, with permission to reach 20 only if all
early summaries parse cleanly. Prefer fewer complete, parseable jobs over more
partial jobs.

## Expansion Wave: 50-100 Jobs

Expand only if first-wave summaries are machine-parseable and at least one
family shows a clear scaling question.

Expansion shape:

- Collector width: fill in 8, 16, 32, 64, 96 across two render/death anchors.
- MCTS sims: fill in 4, 8, 16, 32 across two collector widths.
- Env manager/render: complete only the combinations that explain the first
  wave; avoid all combinations if one axis is already dominated.
- CPU/GPU shape: test 2-3 machine shapes on the best two collector/sim anchors.
- Fanout: extend to N=8, 16, maybe 32 only if N=4 or N=8 has clean merge/import
  and aggregate decisions/sec keeps rising.
- MCTX: extend root batch and sims only if `B=64,R=128,sims=16` is stable,
  finite, and competitive with the LightZero search bucket.
- Checkpoint/opponent: add 2-3 checkpoint/opponent variants after the system
  bottleneck is understood.

Cap any single family at about 25 jobs unless it has already produced the main
bottleneck. The matrix should remain explainable as a small set of slices, not
a heap of unrelated runs.

## Required Metrics For Every Job

Every job writes one `summary.json` or equivalent row with:

- `run_id`, `family`, `job_id`, `created_at`, git commit, command, Modal image
  or local environment, package versions.
- `claim_label`: profiling/control/fanout/mctx, never learning unless Coach
  explicitly owns that run.
- checkpoint path/ref, checkpoint size/mtime/hash if available, collecting
  policy state key, opponent checkpoint path/ref/hash, opponent device.
- env config: `env_variant`, env manager, collector width, episode cap, env step
  cap, source max steps, decision ms, seed strategy.
- render config: render mode, visual stack shape, frame dtype/range, death mode,
  source-state-backed flag, fidelity caveat if using fast render.
- search config: MCTS sims, batch size/root batch if visible, model device,
  multi-GPU flag, action space, legal mask semantics.
- machine shape: CPU count/class, memory, GPU type/count, GPU memory, storage
  location, process count.
- denominators: wall sec, episodes, terminal episodes, env steps, ego decisions,
  active policy rows, search roots, MCTS simulations, learner updates, replay
  samples, chunk bytes.
- throughput: episodes/sec, env steps/sec, decisions/sec,
  active rows/sec, simulations/sec, learner updates/sec when applicable,
  chunk MB/sec, merge/import segments/sec for fanout.
- timing buckets: setup, env reset, env step, render/visual stack, opponent
  inference, policy/search, replay push, replay sample/batch build, learner
  forward/backward/update, checkpoint publish, artifact/eval/GIF, chunk write,
  merge validation, replay import, device transfer, compile, output copy.
- utilization: CPU average/peak if available, per-worker CPU timing, GPU util,
  GPU memory before/after, data-loader or queue wait if available.
- quality of profile: warmup count, steady count, p50/p95 or mean/stdev, number
  of failed workers/retries, timeout flag, OOM flag.
- output checks: non-empty trajectories, legal action masks valid, finite root
  values/visit targets, terminal reason counts, no unexpected all-invalid active
  rows, no unexpected JAX recompiles for MCTX.

For stock trainer jobs, include learner train calls and checkpoint/artifact
behavior even if capped. For collect-only fanout, include chunk manifest
compatibility and `MuZeroGameBuffer.push_game_segments` result. For MCTX,
include compile versus steady timing and finite/normalized action weights.

## Stop Rules

Abandon or pause a family when:

- summaries are missing required denominator fields;
- run paths or checkpoint refs are stale or ambiguous;
- the job touches live Coach training runs or writes into a live run directory;
- failure rate exceeds 20 percent for infrastructure reasons after one retry;
- p95 wall time is dominated by setup/compile/artifact work for a profile that
  is supposed to measure steady collection;
- increasing collector width gives less than 10 percent aggregate
  decisions/sec improvement over the previous width while CPU/process overhead
  rises sharply;
- fanout merge/import consumes more than 25-30 percent of total group wall time
  before N=8, unless merge itself is the bottleneck under study;
- MCTX visual roots fail correctness checks, recompile during steady fixed-shape
  runs, do not fit comfortably in GPU memory, or are slower than the comparable
  LightZero search bucket after visual setup and transfers are included;
- render/no-death profiles are being read as fidelity or learning evidence
  instead of bottleneck lenses.

## Expand Rules

Scale a family when:

- first-wave outputs parse into one table without manual interpretation;
- the same metric denominator is stable across compared jobs;
- at least two adjacent points show a clear slope or saturation point;
- component timing identifies the next plausible limiter;
- chunk/replay artifacts are compatible across jobs where merge is expected;
- CPU/GPU utilization supports the proposed next axis;
- Coach confirms any checkpoint/opponent variants are safe to use for profiling.

Specific scale gates:

- Collector width can expand past 64 only if decisions/sec still improves by at
  least 15 percent from the prior width and worker failures are zero.
- Fanout can expand past N=8 only if aggregate decisions/sec is still rising,
  chunk write plus merge/import stays below 20 percent of group wall time, and
  manifests validate without manual repair.
- MCTS sims can expand past 32 only if search is a first-order wall-clock
  bucket and GPU utilization rises without starving env workers.
- MCTX can expand only after `B=64,R=128,sims=16` is stable, finite, and
  competitive on total visual-root loop time, not just kernel time.
- CPU/GPU shape can expand only when Family A-D say whether the candidate
  bottleneck is CPU env/render, GPU search, learner, or replay.

## Known Risks

- Stale path: old run directories, old checkpoints, or old custom
  `two-seat-selfplay` artifacts can look current. Every row must include exact
  path, commit, checkpoint hash or mtime, and claim label.
- Live-run interference: profiling must not write into Coach-owned training
  outputs or mutate frozen checkpoints. Use isolated output roots and immutable
  checkpoint refs.
- Unparseable matrix: a giant grid with missing metadata will waste compute.
  Start with slices, require shared schema, and stop when rows need manual
  detective work.
- Wrong denominator: `steps/sec`, `episodes/sec`, `decisions/sec`,
  `active_rows/sec`, and `simulations/sec` answer different questions. Report
  all relevant denominators and compare only like with like.
- Bad-policy episode bias: early checkpoints may die quickly and hide render
  costs. Use no-death and stronger-checkpoint lenses only as profiling controls.
- Render fidelity confusion: `body_circles_fast` can be a speed lens, but it is
  not automatically the trusted final visual surface.
- GPU mirage: larger GPUs or multi-GPU runs can look impressive while CPU env
  workers starve the search/learner path. Component timing decides.
- Fanout waste: more actors can produce stale or unused data if learner/replay
  cannot digest it. Measure merge/import and learner-side work before scaling.
