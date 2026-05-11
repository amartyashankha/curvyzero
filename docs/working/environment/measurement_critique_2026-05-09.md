# Environment Measurement Critique - 2026-05-09

Status: working critique

This note is for making measurements less slippery. A number is useful only when
it is attached to the source claim or interface contract it protects.

## Current Truth From This Wave

Progress means source claim -> oracle/probe -> Python parity -> optimized
parity. The promoted claims are narrow mechanics such as kinematics,
border/borderless behavior, body and collision-order canaries, PrintManager
rules, trail cadence, trail gaps, one natural taped trail-gap loop, and old-body
metadata.

Optimized parity exists only for named fixture-backed vector and batch-row
slices. The actor bridge and Modal/Mctx runs remain boundary/runtime evidence
unless they consume real CurvyTron state, final observation/reward contracts,
and production replay rows.

Current fast-path truth: `vector_reset.py` is a reset boundary, `vector_spawn.py`
is a narrow first-round spawn helper, and the comparator/benchmarks are
fixture-backed tools. They are useful, but they are not a production
`step_many`/`reset_many` self-play environment. Treat latency and full-loop
bottlenecks as first-class measurements: real MCTS/model work may dominate the
loop before env stepping does.

Repository test and lint totals are hygiene footnotes after code changes. They
are not a headline and do not prove full reconstruction, production self-play
speed, or training readiness.

## What Each Measurement Means

| Measurement | What it means | What it does not mean | How it can mislead us |
| --- | --- | --- | --- |
| Source-claim acceptance output | Named source-derived scenarios match the expected JS/Python common-trace behavior for that mechanic. | It does not mean the whole mechanic is done. It does not cover untested geometry, long traces, lifecycle, bonuses, observations, or pixels unless those are in the source claim. | A large fixture set can hide a narrow slice. PrintManager evidence is still only the PrintManager claims we wrote down. |
| Common-trace JS/Python parity | The JS oracle and Python source runner agree after both are projected into the same trace shape. This is the best current source-fidelity signal. | It does not mean raw JS internals and Python internals are identical. It does not mean the browser rendering, network messages, or training observations match. | The projection can hide missing fields. If a field is not in common trace, parity says nothing about it. |
| Optimized parity | Fixture-seeded NumPy vector arrays reproduce the supported scalar/source transition fields and fixed event rows for supported fixtures. | It does not mean the vector env supports natural rollouts, arbitrary resets, all player counts, all body capacities, observations, rewards, or full row lifecycle. | A vector fixture set can sound broad while still excluding unsupported mechanics. |
| `B>1` batch-row benchmark | Rows can be stacked and stepped in one array path for the supported fixed-shape fixture slice. It gives CPU timing and checks batch output against single-row output before timing. | It does not mean production self-play is fast. It does not measure learned policy, real MCTS, final observations, replay, broad reset/autoreset, GPU, or network overhead. | Rows/sec can look great because the benchmark cycles known fixtures and skips real game recycling. Debug-event and no-event modes also measure different work. |
| Actor-loop benchmark | A first local bridge can run batched env steps, debug observation/reward packing, synthetic policy/search-shaped work, action feedback, and replay staging. | It is not real self-play. Policy/search is synthetic. Observations and rewards are debug schemas. Reset blocks and synthetic feedback are not the final lifecycle. | It can feel end-to-end because the boxes are connected. But some boxes are stand-ins, so total-loop speed is a plumbing number, not a training number. |
| Repository tests | The repository test suite is a broad regression check for implemented behavior. | It does not mean source fidelity is complete, speed is good, or untested mechanics are correct. | A single total can blur which tests are source fidelity, vector parity, toy env, docs, or utility checks. |
| Lint | Static lint checks code style and simple configured rules. | It says nothing about behavior, fidelity, speed, training quality, or correctness of docs. | Clean lint can sound like product readiness. It is only static hygiene. |
| Modal/Mctx runs | Remote Modal jobs can run the configured JAX/Mctx synthetic or boundary profiles, report hardware/package labels, and separate setup, device placement, compile, warmup, and steady search timing. | They are not CurvyTron rollout throughput unless real CurvyTron rollouts are inside the measured loop. They are not source fidelity, replay, trainer, or checkpoint evidence. | GPU throughput can distract from the CPU env, observation packing, transfer, reset, and replay costs. Tiny synthetic shapes can also exaggerate overhead or hide scaling limits. |

## Claim Notes To Report Going Forward

Report claim notes, not a scorecard.

1. Source claim: claim name, oracle/probe artifact, Python parity status, first
   mismatch if any, and unsupported source behavior.
2. Optimized parity: exact fixture set, vector or batch-row path, unsupported
   cases, and the source claim each optimized row protects.
3. Row lifecycle contract: reset/autoreset/timer/done status with open blockers.
4. Actor-loop contract: env rows/sec, ego rows/sec, p50/p95 action latency,
   reset overhead, observation/reward packing time, policy/search time, and
   replay time only after the interface contract is named.
5. Modal/Mctx runtime note: host setup time, host-to-device placement time,
   compile/first-run time, steady search time, ego decisions/sec, and whether
   inputs came from real actor-bridge output or synthetic tensors.
6. Hygiene footnote: repository tests and lint only after code changes, labeled
   as regression/static evidence rather than source progress.

Avoid one master score. One master score would invite overclaiming.

## The Real Comprehensiveness Check

The real answer is a claim matrix. Rows are mechanics. Columns are evidence
steps.

Suggested columns:

- Source fact named.
- JS oracle fixture exists.
- Python common-trace parity.
- Promoted source batch.
- Vector array parity.
- `B>1` batch support.
- Observation/reward support.
- Reset/autoreset/lifecycle support.
- Actor-loop support.
- Modal boundary support.
- Open gap or explicit out-of-scope label.

Suggested mechanic rows:

- Setup, spawn, player order, map size.
- Movement, elapsed time, turn rate.
- Normal walls and borderless wrap caveats.
- Body collision, own-body latency, strict overlap, old-body metadata.
- Collision order, same-frame death, scoring order.
- PrintManager toggles, delayed start, random call order, death stop.
- Trail cadence, trail gaps, emitted bodies, visual trail versus world body.
- Round lifecycle, reset, autoreset, terminal handling.
- Bonuses, RNG, durations, stacking.
- Observation and reward schemas.
- Replay chunks and training metadata.
- Server messages, per-player perspective.
- Browser pixels and human feel.

Missing-mechanics checklist today: round lifecycle, bonuses, final
trainer-facing observation/reward, broad reset/autoreset, natural long-run
trail-gap cadence, broader emitted-body trails, server messages, pixels, and
human feel. Those gaps are not failures. They are simply not proven yet.

## Later Links

Docs that should link here later:

- `docs/working/environment/coverage_tracker.md`
- `docs/working/environment/active_lanes.md`
- `docs/working/environment/selfplay_speed_lane_2026-05-09.md`
- `docs/design/environment/fidelity_measurement_plan.md`
- `docs/design/environment/fidelity_checklist.md`
- `docs/experiments/environment/2026-05-09-vector-batch-row-timing.md`
- `docs/experiments/environment/2026-05-09-vector-actor-loop-bridge.md`
- `docs/experiments/2026-05-09-modal-mctx-synthetic-benchmark.md`
