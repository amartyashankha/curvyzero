# Optimizer Measurement Plan

Date: 2026-05-09

Status: Amdahl-style measurement plan for future speed work.

## Short Read

Do not optimize by guessing. Measure the full loop, then optimize the largest
real bucket.

The first comparison-valid actor-loop report must include both throughput and
latency:

- useful games/minute;
- env transitions/sec;
- ego decisions/sec;
- replay rows/sec;
- p50/p95/p99 action latency;
- actor idle time;
- learner idle time;
- policy staleness.

## Buckets

Measure these as separate timers:

- reset/autoreset;
- env step;
- observation packing;
- live-player row compaction;
- policy forward;
- MCTS/search if present;
- CPU/GPU transfer if present;
- action scatter back to wrapper/replay `joint_action[B, P]`;
- replay/rollout staging;
- replay write or learner handoff;
- target construction;
- learner sample;
- learner update;
- checkpoint publish/load;
- evaluator/scorecard overhead;
- actor idle;
- learner idle.

## First Report Shape

Each report should state:

- environment backend: source env, toy env, vector env, or adapter;
- player count and batch size;
- observation schema;
- action schema;
- reward schema;
- opponent policy mix;
- policy/search implementation;
- replay/rollout writer;
- debug event mode;
- Modal/local hardware;
- dependency versions;
- total wall time;
- timing buckets;
- p50/p95/p99 action latency;
- completed games/minute;
- replay bytes/chunk;
- policy staleness.

The concrete report contract lives in
[profile report contract](profile_report_contract_2026-05-09.md). Keep that
contract neutral across repo-native and LightZero lanes; framework-specific
fields can be absent or `null`, but the report should say why.

## Amdahl Gates

Optimize env step if:

- debug events are off or sampled;
- observation/replay/final-observation path is production-like;
- real policy/search timing is included or calibrated;
- env step remains the largest bucket.

Optimize model/search if:

- real search/model timing is a large bucket;
- GPU batching or larger sync batches improve games/minute without unacceptable
  p95/p99 action latency;
- policy staleness stays inside the chosen budget.

Optimize replay if:

- production writes or learner handoff are visible in wall time;
- replay serialization blocks actors;
- replay chunk size or write frequency creates backpressure.

Optimize reset/autoreset if:

- terminal rows stall batches;
- final observations are expensive or mishandled;
- autoreset causes tail latency or replay corruption risk.

Do not optimize GPU env/native backend/distributed actors until these gates make
the target obvious.

## Decision Tree After Full-Loop Profile

Start this tree only after a comparison-valid actor-loop profile exists:
source-faithful CurvyTron observations/rewards/final observations, public
autoreset, replay writer or learner handoff, debug events off or sampled, and
real policy/search timing.

If contracts are missing, stop and hand the missing field list to the owner of
that contract. Optimizer should not optimize debug fixtures as the training
architecture.

If `env_step + observation packing + reset/autoreset` dominates:

- split movement/collision, trail/body insertion, ray/raster observation,
  terminal/final-observation, and reset/spawn/timer costs;
- remove debug event rows from the hot path and keep compact refs;
- finish source-faithful vector lifecycle before native or GPU rewrites;
- try larger sync batches and fixed-player grouping before actor pools;
- only consider Numba/C++/Rust/GPU env if this bucket remains largest with real
  policy/search and replay included.

If `policy/search` dominates:

- keep env-speed work secondary;
- for PPO, batch live ego rows better and compare compact versus padded rows;
- for MuZero/Mctx/LightZero, sweep simulations/depth, batch roots, and split
  host/device transfer from search;
- split MuZero-family search cost into representation/prediction/recurrent
  inference, tree bookkeeping, action selection, target construction,
  support/value transforms, replay sampling, and root-target diagnostics;
- compare search against pure policy at equal wall-clock;
- cap p95/p99 action latency and policy staleness before scaling actors.

If `CPU/GPU transfer` dominates:

- batch fewer, larger transfers: observation and mask to device once, actions
  and policy/search outputs back once;
- keep replay target construction on the side where data already lives when
  possible;
- do not jump to GPU env unless it removes transfers across env, observation,
  search, replay targets, and learner together.

If `replay stage/write/handoff` dominates:

- use chunky array replay, not JSON-per-row;
- increase rows per chunk until write overhead falls without actor stalls;
- add async writer or learner-adjacent queue only after measuring backpressure;
- record bytes/chunk, rows/chunk, write latency, actor wait, blocked chunks,
  and schema rejection cost.

If `learner update/sample/target construction` dominates:

- hand the evidence to the coach lane before redesigning algorithmic work;
- include learner share, sample time, target-build time, update time, GPU
  utilization, actor idle, learner idle, replay age, policy staleness, batch
  size, update ratio, loss/target diagnostics, checkpoint cadence, and whether
  eval improves;
- tune hardware/batching only after coach confirms the learner work is useful.

If `actor idle` dominates:

- identify whether actors wait on search, replay, reset/autoreset, learner, or
  checkpoint refresh;
- centralize or batch inference only if search causes the wait;
- use async chunk writes only if replay causes the wait;
- cap checkpoint refresh frequency if learner/checkpointing causes the wait.

If `learner idle` dominates:

- increase actor throughput or reduce replay write bottlenecks;
- add env rows or actor processes only if action latency and staleness stay
  bounded;
- do not add distributed actors before one-container batching is saturated.

## Debug Event Policy

Debug event rows are allowed for:

- fidelity probes;
- short diagnosis;
- sampled speed runs that explicitly measure debug overhead.

Debug event rows should be off or sampled for:

- training hot-loop reports;
- throughput comparisons;
- framework selection runs.

## Framework-Agnostic Gates

Owned PPO/IPPO-style runner remains the leading repo-native bench hypothesis
only if:

- it can produce the full measurement report quickly;
- its rollout buffer and profiler are transparent;
- it learns enough for the coach lane to use as a baseline;
- it keeps the all-player `[B, P]` wrapper shape.

Falsify or demote this hypothesis if LightZero produces a credible stock
Pong-like reproduction plus a metadata-preserving CurvyTron bridge, if PPO is
too weak to diagnose CurvyTron despite heuristics succeeding, or if building the
owned runner recreates framework machinery faster than it exposes loop
bottlenecks.

LightZero should move from serious replication/control into main-candidate
status only if:

- it preserves required metadata and replay/target visibility;
- its collector/search overhead is acceptable;
- coach-lane target-quality gates show useful search targets;
- it does not force CurvyTron into the wrong game semantics.

LightZero remains a serious replication/control lane even before it is favored
as the CurvyTron architecture. Optimizer should measure comparable throughput,
latency, checkpoint id, seed/reset, observation/action/reward schema, and target
metadata wherever the LightZero interface exposes them; coach owns whether that
lane has reproduced a credible Pong-like control or hit a clear blocker.

Mctx should move from search-module hypothesis into owned MuZero path only if:

- search is worth its cost after PPO/LightZero baselines;
- batching and device timing are favorable;
- the project is ready to own replay, targets, learner, checkpoints, and eval.

## Current Evidence To Respect

- Existing env-speed numbers are mostly debug/fixture evidence.
- Debug event emission is a large hot-path tax.
- In-memory replay staging has been small in scouts, but production writes are
  not proven.
- Synthetic policy/search is not a stand-in for real MuZero/MCTS or PPO model
  timing.
- The latest local contract check passed for source env, trainer contract,
  replay contract, and local LightZero-shaped adapter smoke.

## Source Anchors

- [Amdahl loop note](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md)
- [Self-play speed lane](../environment/selfplay_speed_lane_2026-05-09.md)
- [Measurement critique](../environment/measurement_critique_2026-05-09.md)
- [Actor-loop architecture](actor_loop_architecture_2026-05-09.md)
