# Optimizer World Model

Date: 2026-05-09

Status: optimizer working memory. This is not a training-quality report and not
an environment-fidelity report.

2026-05-10 note: for lane boundaries, see
[optimizer lane contract](lane_contract_2026-05-10.md).

## What This Project Is

CurvyZero is trying to train agents for a CurvyTron-like multiplayer game.
The original CurvyTron code is the reference. The training runtime should be a
source-faithful Python/NumPy environment that can run fast, produce replay, and
stay reproducible.

The first useful game target is 1v1/no-bonus. That is a proof slice, not the
whole project. The real game remains simultaneous multiplayer.

## Main Lanes

Environment lane:

- owns source behavior, JS oracle comparison, Python parity, vector parity, and
  trainer-facing observation/reward/final-observation contracts;
- has real source-derived slices and a scalar `CurvyTronSourceEnv`;
- has not proven full multiplayer gameplay, bonuses, broad lifecycle, or final
  production vector rollout.

Coach/training lane:

- owns learning claims, checkpoints, scorecards, target-quality audits, and
  LightZero/Pong results;
- has stock LightZero controls and a custom dummy Pong LightZero bridge;
- has not proven reliable custom dummy Pong improvement or CurvyTron learning.

Optimizer lane:

- owns the setup and speed worldview: what loop should run, what to measure,
  where Amdahl's law points, and when systems rewrites are justified;
- does not own whether a checkpoint improved;
- does not own whether a source-fidelity claim is true;
- uses those lanes' results as inputs.

## What Is Real Today

- The repo has source-derived CurvyTron environment slices and a scalar source
  env.
- The repo has trainer contract constants for 1v1/no-bonus egocentric rays,
  sparse round reward, and three turn actions.
- The repo has replay-v0 chunk contracts for observation, reward, action,
  action weights, root value, terminal flags, final observation, and final
  reward map.
- The repo has installed-runtime no-train LightZero/DI-engine smokes for the
  scalar CurvyTron adapter and the debug visual CurvyTron adapter. The debug
  visual env type is `curvyzero_debug_visual_tensor_lightzero`. This is
  setup/timing evidence, not trainer readiness or learning evidence.
- The repo has a real custom dummy Pong LightZero env and Modal train plumbing.
- The repo has speed scouts, including source-stepped trainer-shaped
  `[B,P,106]` no-train profiles, center-cell body scouts, and a stronger
  source body-circle ray mode backed by `world_bodies_snapshot()` plus
  `avatar_body_metadata_snapshot()`. Real policy/search/learner timing,
  browser-visible trail history, and bonus-body observations are still missing.

## What Is Not Real Yet

- No full CurvyTron trainer loop.
- No source-faithful visual tensor. Current visual tensor is
  `curvyzero_debug_occupancy_gray64/v0`, a debug occupancy smoke surface.
- No production actor loop with source-faithful trail/body observations, public
  autoreset, repeated replay/learner handoff, and real model/search timing.
- No project-owned PPO runner.
- No project-owned MuZero/Mctx trainer.
- No proof that LightZero is the right final backbone for CurvyTron.
- No proof that CPU env stepping is the final bottleneck once real model/search
  is included.

## Framework Roles

LightZero:

- serious MuZero replication/control lane and custom-env bridge;
- should not define CurvyTron core architecture by default;
- good for coach-lane target audits, stock Pong-like reproduction, checkpoint
  and eval plumbing, and comparison runs;
- can be reconsidered as a CurvyTron candidate if it preserves required
  metadata and passes target-quality and timing gates.

Owned PPO / TorchRL / CleanRL-style runner:

- current leading hypothesis for the first optimizer-friendly CurvyTron
  learnability and speed diagnosis;
- keeps env, rollout rows, scorecards, telemetry, and profiling under project
  control;
- should use the same multiplayer wrapper `[B, P]` shape the real game needs.
- demote if comparable evidence shows LightZero, TorchRL, Sample Factory, or
  Mctx exposes the same contracts faster or with better bottleneck visibility.

Mctx:

- likely owned-search path later;
- search library, not a trainer;
- should plug into a working actor loop as the policy/search box, not replace
  the environment or replay architecture.

Sample Factory, RLlib, SB3, OpenSpiel, and others:

- useful references or later scaling candidates;
- should not be adopted before the repo-native loop shape and bottlenecks are
  clear.

## Current Optimizer Hypothesis

The current architecture hypothesis is repo-native all-player self-play:

```text
env state [B, P]
  -> ego observations [B, P, ...]
  -> live ego rows
  -> batched policy/search
  -> wrapper joint_action [B, P]
  -> trainer env step over elapsed-ms source frames
  -> replay/rollout rows
  -> learner
  -> checkpoint/eval
```

Start with two players but keep the shape compatible with more players. Do not
model CurvyTron as alternating turns. Do not hide the multiplayer shape forever
inside a single-agent wrapper.

The first speed job is not to make one function faster. It is to measure the
whole actor loop and find the largest real bucket.

Current primary CurvyTron visual profiling target is non-ALE
`debug_visual_tensor` / `curvyzero_debug_occupancy_gray64/v0`: raw
`uint8[1,64,64]` CHW occupancy smoke, optionally normalized to
`float32[1,64,64]` CHW for LightZero-facing payloads. It is not source-faithful
visual truth.

Current local scalar-ray diagnostic bench is:

```text
CurvyTronSourceEnv
  -> source_snapshot_to_vector_trainer_state(...)
  -> observe_vector_1v1_egocentric_rays_v0(...)
  -> float32[B,2,106] + bool[B,2,3]
  -> policy-row mapping
  -> replay-v0 profile artifact
```

This is not Atari/ALE and not a real LightZero env. It is the repo-native
source/trainer diagnostic path, not the primary visual training target. The
batch action map is a wrapper/replay abstraction, not a native CurvyTron source
object.

## Boundary Correction

Optimizer may run tiny mechanical probes if they answer architecture or timing
questions. Optimizer should not chase Pong score, checkpoint quality, or
learning curves. Those belong to the coach lane.

If optimizer has a coach note, keep it short and point to docs. Do not take over
the coach lane.

## Source Anchors

- [Project README](../../../README.md)
- [Docs map](../../README.md)
- [Environment active lanes](../environment/active_lanes.md)
- [Training state index](../training_state_index_2026-05-09.md)
- [Optimizer setup synthesis](setup_synthesis_2026-05-09.md)
- [Trainer contract](../../../src/curvyzero/env/trainer_contract.py)
- [Replay chunk v0](../../../src/curvyzero/training/replay_chunk_v0.py)
- [Amdahl loop note](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md)
