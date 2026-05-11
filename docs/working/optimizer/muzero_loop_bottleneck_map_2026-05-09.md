# MuZero Loop Bottleneck Map

Date: 2026-05-09

Status: optimizer working map. This is not a learning-quality report and not an
environment-fidelity claim.

## Short Read

The bottleneck is not knowable yet. Current evidence covers pieces: source-env
scouts, vector fixture kernels, strict native vector trainer plumbing,
debug actor-loop bridges, trainer-shaped repo-native dry runs, synthetic search
stand-ins, replay chunk validation, and LightZero controls. No run yet combines
source-faithful CurvyTron `[B, P]`
transitions, wrapper `[B, P]` action maps, trainer observations, final
observations, replay/handoff, real policy/search, learner update,
checkpoint/eval, and policy-version metadata.

Optimizer should therefore rank bottlenecks only after an actor-loop report. Until
then, the right work is to expose each bucket with honest metadata.

## Full Loop

```text
reset/autoreset + seed/RNG
  -> env state [B, P]
  -> source-faithful observation/reward/final-observation packing
  -> live ego-row compaction
  -> model initial inference
  -> MCTS/search recurrent inference and tree bookkeeping
  -> action selection
  -> action scatter to wrapper joint_action[B, P]
  -> trainer env step over elapsed-ms source frames
  -> replay or rollout staging/write/handoff
  -> target construction or reanalyse
  -> learner sample/update
  -> checkpoint publish/load
  -> evaluator/scorecard
  -> policy refresh/staleness accounting
```

## Bucket Map

| Bucket | What Exists | What Is Missing | Optimizer Risk |
| --- | --- | --- | --- |
| Source/env step | `CurvyTronSourceEnv` scalar scout, vector fixture kernels, and strict `VectorTrainerEnv1v1NoBonus` trainer-wrapper profile. | Production source-faithful vector loop, broad lifecycle, bonuses, multiplayer, final obs handoff. | Optimizing debug, fixture, or strict no-bonus kernels can miss reset, obs, replay, and search cost. |
| Reset/autoreset | Vector reset, spawn, lifecycle, autoreset planners, and `B=128` reset regression exist separately. | One public actor-loop boundary that stages final obs/reward before reset and preserves seed/episode metadata. | Tail latency or replay corruption can hide behind good one-step numbers. |
| Observation/reward packing | Trainer contract emits `[P,106]`; source-stepped trainer-shaped `[B,P,106]` no-train profile and strict vector trainer profile both exist. | Exact source trail/body geometry, final observation cost, vector-backed source-faithful rows, raster or history stacks if used. | Ray/raster packing may dominate before env physics does. |
| Row compaction/action scatter | Repo-native dry run exercises live ego rows and wrapper `joint_action[B,P]`. | Same mapping over source-stepped rows and mixed liveness/player counts. | Single-agent wrappers can hide multiplayer wrapper cost and metadata. |
| Policy/model forward | Tiny masked policies and NumPy stand-ins exist. | Real PPO, LightZero MuZero, or Mctx model timing on the same ego-row shape. | Env-only optimization is premature if model/search dominates. |
| MCTS/search | Synthetic NumPy and Modal/JAX/Mctx boundary probes exist. | Real learned recurrent model, real roots, tree bookkeeping split, H2D/D2H, target quality. | A fake search knob can mislead both speed and architecture decisions. |
| Replay/rollout | Replay-v0 chunk contract and debug sample writes exist. PPO-shaped rollout buffers exist in dry run. | Repeated production writes/read/sample/handoff, queue depth, schema rejection, Modal Volume/object-store path. | One `.npz` write is not a replay stream. |
| Target construction | PPO GAE smoke exists behind optional Torch. Replay-v0 target fields exist. | MuZero n-step targets, support transforms, root visit/value targets, reanalyse/priority. | Learner work can be invisible if target construction is folded into replay or update. |
| Learner update | Optional Torch PPO learner smoke exists, but local `uv` env currently skips it because Torch is missing. LightZero trainer smokes live in coach/control lane. | CurvyTron learner profile with sample/update/idle/GPU stats and checkpoint publish. | Optimizer cannot infer learning usefulness from loss fields. |
| Checkpoint/eval | Artifact paths and some smokes exist. | Checkpoint id/hash, strict load, eval wall time, eval episodes/sec, best/latest policy semantics. | Eval/checkpoint cadence can become hidden actor idle. |
| Policy staleness | Synchronous dry-run marker exists: `max_version_lag=0`. | Async actor/learner lag, refresh interval, dropped stale rollouts, p50/p95/p99 lag. | More actors can lower freshness even if rows/sec improves. |

## Current Mini-Probe Evidence

- `benchmark_policy_search_batch_standin.py`, `B=64`, `P=2`, `obs_dim=106`,
  `hidden_dim=128`, `simulations=4`: elapsed `0.00624s`, about `410k`
  policy rows/sec, recurrent-search stand-in about `80%` of measured bucket
  time.
- Same shape with `simulations=32`: elapsed `0.03913s`, about `65k` policy
  rows/sec, recurrent-search stand-in about `97%` of measured bucket time.
- `repo_native_ppo_learner_smoke.py` skipped locally because Torch is not
  importable in the uv environment. That is a setup fact, not a PPO verdict.
- `benchmark_source_env.py` on the narrow 111-step 1v1/no-bonus wall lifecycle:
  Python source env around `120k` steps/sec; persistent JS worker around `11k`
  steps/sec in the tiny local run. This is source-scout evidence, not
  production vector-loop throughput.
- `benchmark_vector_trainer_actor_loop_profile.py` on strict
  `1v1/no_bonus/P=2`: corrected native vector throughput was `1980/s` at
  `B=8,T=64`, `1760/s` at `B=16,T=64`, `1600/s` at `B=32,T=32`, and `1197/s`
  at `B=128,T=64`. Public step dominated loop time, and one-pass ray probes
  scaled from `0.0035s` at `B=8` to `0.0561s` at `B=128`.

## Next Decisive Profile

The next useful optimizer profile is a source-stepped and then source-faithful
`[B,2]` actor-loop report with:

- source-faithful observation/reward/final-observation payloads;
- public reset/autoreset ordering and metadata;
- live ego-row compaction and wrapper `joint_action[B,P]` scatter;
- calibrated real policy/search or clearly labeled fixed proxy;
- repeated replay/rollout write or learner handoff;
- canonical timing buckets, latency stats, throughput denominators, integrity
  checks, and policy/search metadata.

The strict native vector profile is a comparison rail for plumbing and replay
shape. It does not replace the source-backed JS/`CurvyTronSourceEnv` oracle.

Until that report exists, keep all framework and bottleneck rankings as working
hypotheses.
