# CurvyTron Optimizer Goal

Date: 2026-06-05

This file is the durable compass. It should be edited rarely: only when the
north-star goal, accepted baseline, selected architecture, active blocker, or
decision gates change. Run logs, artifact details, stale-lane cleanup, and
subagent notes belong in `docs/working/optimizer/reorientation_2026-05-23/`.

## Actual Goal

Build a compact-owned CurvyTron training loop that is genuinely faster than the
accepted OPT-104 same-work H100 full-loop baseline.

The first real target is a repeatable promotion-grade speedup. A `2x` same-work
H100 win is the near target. A `5x-10x` win is only plausible if the major
hot-loop data surfaces stop moving through parent Python and become resident,
fixed-shape owner/buffer handoffs.

The goal is not a prettier timer, a mock-speed row, a local-only proof, a
different denominator, or a single faster bucket. The goal is a faster whole
training loop.

## Baseline

```text
run: opt104-h100-normal-death-learner-validation-only-compacttorch-r1-20260531
hardware: H100
shape: B1024/A1
death mode: normal death
search: compact Torch
measured/warmup: 180 / 45
sample: interval 8, batch 512, replay capacity 4096
learner: unroll 2, train steps 1, refresh interval 4
speed: 12689.38 env steps/sec
wall: 14.5255s
```

Only same-work H100 full-loop rows can prove speed. Local rows, toy rows,
proof rows, mock rows, sidecar audits, and helper timers are decision inputs.

## Whole Loop Scope

The measured loop includes:

```text
mechanics / death / autoreset
observation and root-state production
search and action selection
replay append and storage
sample selection and learner-batch construction
learner update
policy/model refresh
proof, reporting, and final drain
```

A patch that makes one surface faster while another hot surface remains hidden
in parent Python is not enough.

## Architecture Model

Fast RL systems tend to rhyme:

```text
many envs write fixed buffers
search/inference consumes batched arrays or small handles
replay owns compact trajectory/index state
learner consumes fixed-shape batches and publishes versions
policy/search refresh is explicit
Python coordinates epochs, proof, launch, and reporting
```

The systems to copy conceptually are AlphaZero/MuZero/EfficientZero-style role
separation, PufferLib/EnvPool/Sample Factory/Isaac-style fixed buffers and
vectorized actors, and MCTX/JAX-style dense batched search. Do not blindly port
frameworks. Copy the ownership pattern: resident data, fixed shapes, coarse
sync points, and small handles across boundaries.

Target ownership graph:

```text
env/mechanics owner:
  receives previous actions
  steps mechanics, death, reward, autoreset
  publishes fixed observation/root/reward/done buffers

search owner:
  reads root/observation handles
  runs batched inference/search
  returns selected actions and compact replay handles
  keeps search/root state needed for replay materialization

replay owner:
  appends compact trajectory/index records
  maintains sampleable state
  exposes learner-ready or near-learner-ready tensors

learner owner:
  trains on fixed-shape batches
  publishes model/version/digest refreshes

parent Python:
  coordinates cadence, final drain, validation, logging, launch gates
```

## Current State

```text
stable speedup over OPT-104: modest repeat evidence, not promotion-grade
near-target 2x speedup: no
10x path justified: no
fundamental impossibility: not proven
selected candidate: vectorized reset RNG + owner-search threaded/background maintenance
active blocker: same-work headroom from modest 1.04x-1.10x toward 2x
```

The selected owner-search path is useful because the service that owns
search/root state also owns replay materialization derived from that state.
Recent H100 rows refined the active blocker:

```text
reset/autoreset Python RNG loop: fixed by vectorized CurvyZero-owned seeded source-random history with exact scalar parity
refresh cadence drift: fixed by matching OPT-104 refresh interval 4
inline owner maintenance: unstable/slab-heavy
threaded/background owner maintenance: repeated modest win and longer-row win
remaining gap to 2x: still open
```

The owner-search/threaded path has proved the important ownership invariants:

```text
parent committed/stored replay rows: 0
search-result payload bytes: 0
owner materializes replay: true
selected actions become applied/replay actions
owner replay/sample/train/drain counts close
inner two-phase compact Torch device replay is preserved
normal-death proof is preserved
```

The current H100 evidence is a real improvement but not the finish:

```text
accepted OPT-104 baseline: 12689.38 env/s, 14.5255s wall
threaded/background candidate repeat rows: 14002.12 and 13640.60 env/s
2x-long threaded/background row: 13145.34 env/s
proof status: tensor-native fallback none, terminal proof present, violations []
promotion status: not enough headroom; keep working toward 2x
```

Interpretation: the active work is no longer "prove any speedup exists." It is
to turn a modest, explained speedup into a larger same-work speedup by removing
the remaining actor/observation/autoreset and owner/search/replay/learner
surfaces from Python-heavy cadence paths. Do not reopen pure attribution
timers, cadence-only refresh, MPS placement, eager append pre-drain, or
same-process async learner overlap as primary speed paths.

## Why Progress Stalled

The failure mode was not that CurvyTron speedup is impossible. The failure mode
was repeatedly letting the latest failed row choose the next task instead of
returning to the owner graph.

The repeated mistakes to avoid:

```text
using goal.md as a run ledger
treating local proof as speed progress
launching H100 to answer architecture questions local gates could answer
optimizing labels and timers instead of moving ownership surfaces
staying on stale surfaces after evidence moved the P0 blocker elsewhere
using sidecars after decisions rather than as parallel falsifiers
accepting proof that did not fail closed on parent hot-loop transport
reopening stale lanes because planning docs had too many historical "nexts"
```

The correction is simple:

```text
move owner boundaries, not timer labels
prove local mechanics before H100
make speed claims fail closed
use sidecars for bounded critique while the main thread implements
keep goal.md stable and put evidence churn in the planning docs
after every row ask: did the whole-loop owner graph change, or only a timer?
```

## Next Gates

1. Candidate preservation gate.
   Keep vectorized reset RNG, refresh interval 4, threaded/background owner
   maintenance, fused learner batch, learner-ready unroll-2 cache,
   tensor-native replay, borrowed render state, normal death, and owner-search
   action-only proof intact. The row must still finish with no pending owner
   maintenance, no failed async/maintenance state, terminal proof present, and
   accepted-fast-path violations `[]`.

2. Next-headroom implementation gate.
   The next implementation must plausibly remove a measured remaining surface:
   actor/autoreset/observation work, owner-search dispatch residual, replay
   materialization/append, search batching, or learner publication/update
   cadence. It should change the owner graph or hot data movement, not only add
   a timer.

3. Same-work H100 repeat gate.
   Use exact same-work H100 rows to prove any new speed patch. Two exact rows
   above OPT-104 are the minimum; use a longer row when variance remains the
   question. A modest repeat win is progress, but the near target remains a
   repeatable `2x` same-work win.

## Decision Rules

```text
candidate repeat rows beat OPT-104 but remain far below 2x:
  keep the candidate, then attack the largest remaining measured owner/actor
  surface; do not declare the goal complete

inline owner row wins once but repeat regresses:
  treat owner/slab maintenance cadence as unstable and prefer background owner
  maintenance or a tighter owner-boundary patch

MPS, eager append, cadence-only refresh, or in-process async preserves proof but speed is flat/worse:
  stop that lane as a speed path

local proof preserves invariants and removes real hot work:
  run one H100 candidate, then exact repeat before promotion language

next patch only changes labels/timers:
  reject it unless it is required to fail closed on an imminent speed claim

threaded/background maintenance stops beating OPT-104:
  re-audit actor/autoreset/observation and owner/search/replay timing before
  launching more repeats

H100 row is unstable or below noise gate:
  no speed claim; compare against A/A noise and return to the measured owner
  surface that moved
```

## H100 Speed Claim Bar

Minimum future speed-claim requirements:

```text
same hardware and shape as OPT-104
same work identity and sample/order checksums
normal-death contract proof present
accepted-fast-path violations: none
host fallback: 0
render-state copy steps: 0
sample/learner/policy counts: identical
parent committed/stored rows: 0 in owner action-only mode
search-result payload bytes: 0
owner maintenance pending/inflight/failed: 0/false/false after final drain
wall spread: <= 5% of median
major bucket spread: <= 10% of median
repeat count: at least 2 exact rows; use 3 if unstable
optimization win: at least 2x measured A/A noise
```

## Document Discipline

Use the planning docs this way:

```text
CURRENT_STATE.md:
  current evidence and active blocker

FOLLOWUPS.md:
  active queue and closed-lane guards

ORCHESTRATION.md:
  what to run next and what not to run

SUBAGENT_DELEGATION.md:
  sidecar questions, results, and integration notes

PRIMARY_RESIDUAL_OWNERSHIP_2026-06-04.md:
  detailed owner-search evidence and artifact rows

MEASUREMENT_LEDGER.md / FINDINGS_LOG.md / ACTIVE_RUN_REGISTRY.md:
  durable run evidence
```

Before code changes, state the baseline row, target bucket, expected wall-time
win, and proof fields that must remain unchanged. Then implement. Do not do a
broad reorientation pass unless it ends in a code change, a toy-ceiling
artifact, a same-work row after local proof, or a comparison artifact.
