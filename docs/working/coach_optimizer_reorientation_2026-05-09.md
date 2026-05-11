# Coach Optimizer Reorientation - 2026-05-09

Purpose: integrate the optimizer lane into the coach worldview. This is a
simple correction note, not a new experiment log.

## What Is Going On

The project is trying to train agents for a CurvyTron-like multiplayer game.
The environment lane is rebuilding source-faithful game behavior. The coach
lane is testing whether policies actually improve. The optimizer lane is asking
how the full training loop should be shaped and measured before we chase speed.

The important correction is this:

LightZero is not "just a reference." It remains a serious replication/control
obligation for the MuZero lane until credible reproduction or a clear blocker.
But it should not define CurvyTron's architecture by default. CurvyTron is a
multiplayer game with held controls for all live players. The repo-native
`[B,P]` lane is an architecture probe to keep that shape visible while we learn
what the final framework should be.

The coach should now think in two layers:

- LightZero/Pong lanes are replication controls, bridges, and target-quality
  audits.
- CurvyTron also needs a repo-native actor loop with explicit all-live-player
  wrapper rows.

Latest boundary facts:

- Matching 64x64 official Atari checkpoints can now use the stock LightZero
  `MuZeroEvaluator` path. The tiny Modal smoke fixed the missing
  `action_mask`/collation gap and matched manual actions.
- Pretrained OpenDILab Pong strict eval remains blocked separately: the public
  checkpoint is older 96x96/downsample Atari, while the current stock config is
  64x64. The replication obligation remains; use a matching checkpoint/config
  pair or continue from-scratch stock reproduction.
- The repo-native PPO on-policy learner smoke is optional-Torch and
  no-quality. The same tiny actor-critic collects and is updated, `[T,B,P]`
  arrays are preserved, one masked PPO update runs, metrics/checkpoint/report
  artifacts are written, and the local smoke reported
  `masked_action_violations=0`. It is not a learning result and not a
  LightZero replacement.
- Shared reporting requirement:
  `docs/working/shared_training_reporting_contract_2026-05-09.md`. Both lanes
  should emit comparable profile metadata, contracts, timing buckets,
  throughput, latency, checkpoint ids, seed/reset details, and explicit
  non-claims.

## Why LightZero Struggled

LightZero did run real MuZero training on controls and custom dummy Pong. That
matters. The problem is that the current evidence is mostly plumbing, not good
learning.

Official Atari Pong proved train/checkpoint/load/eval mechanics, but the scaled
smoke runs stayed weak and even collapsed to one action. That looks more like
off-recipe early training than an action-mapping bug.

Custom dummy Pong proved the adapter, Modal jobs, checkpoint mirroring,
scorecards, target sidecars, and frozen-checkpoint opponent plumbing. But the
held-out policies still collapse. Low-simulation MCTS can write root-visit
targets that miss a legal winning action. Trainer-side action diversity is not
enough if the stored targets and held-out MCTS rows are still bad.

So the struggle is not one simple bug. It is a setup mismatch:

- tiny smoke-scale budgets are being asked to prove learning quality;
- custom dummy Pong may not match stock LightZero assumptions well enough;
- root targets, support scale, and replay visibility are still under audit;
- single-agent wrapper shape hides the real multiplayer contract;
- speed measurements do not yet include the real full actor loop.

## What LightZero Is Still Good For

Keep LightZero for:

- stock controls such as CartPole, board games, and official Atari Pong;
- custom dummy Pong bridge tests;
- MuZero target audits, especially root visit distributions;
- checkpoint loading, MCTS eval, and scorecard comparison;
- reference behavior for what a full trainer provides;
- comparison against a repo-native runner.

Do not use LightZero as proof that CurvyTron learning is solved. Do not skip
LightZero replication because PPO looks cleaner. Do not force CurvyTron into
stock Atari semantics or turn-based board-game semantics just to fit a
framework.

LightZero can still be a plug-in later if it preserves the needed metadata,
final observations, replay contracts, scorecards, and profiling buckets.

## What Must Remain Repo-Native `[B, P]`

The repo-native CurvyTron architecture probe should keep the all-live-player
wrapper shape visible:

```text
env state [B, P]
  -> ego observations [B, P, ...]
  -> compact live player rows
  -> policy/search batch
  -> wrapper joint_action [B, P]
  -> trainer env step over elapsed-ms source frames
  -> replay or rollout rows
  -> learner
  -> checkpoint and eval
```

These contract surfaces should remain visible in the repo-native probe:

- reset/autoreset and wrapper step shape: `reset_many(seed[B])`,
  `step_many(wrapper joint_action[B, P])`;
- observation, legal-action mask, live mask, reward, done, final observation,
  and final reward map;
- mapping from player rows to batched policy/search rows and back to
  wrapper `joint_action[B, P]`;
- opponent/checkpoint assignment metadata;
- replay or rollout row schema;
- scorecards and plain eval summaries;
- profiler buckets for env step, observation packing, policy/search, replay,
  reset/autoreset, learner idle, actor idle, and policy staleness.

Start with two players, but keep the shape able to grow beyond two. Do not make
the real game alternating-turn unless the game itself becomes alternating-turn.

## Next High-Level Lanes

1. Keep the current LightZero lanes honest.
   Official Atari remains a stock reproduction/control lane. Custom dummy Pong
   remains a bridge and target-audit lane. Do not scale either as proof until
   held-out scorecards and target telemetry improve.

2. Build or specify the first repo-native PPO/CleanRL-style CurvyTron runner.
   This is a parallel architecture lane, not a replacement for LightZero
   replication. The current learner smoke now proves the policy-sampled
   on-policy collection/update boundary with the same tiny actor-critic while
   preserving the `[B, P]` game shape and `[T,B,P]` rollout rows. The next
   useful slice is shared-contract reporting plus scorecards and full-loop
   profiling around that shape before any quality claim. This is not a final
   framework decision.

3. Measure the whole actor loop before optimizing pieces.
   Include env step, observation packing, policy/search, replay write or
   learner handoff, reset/autoreset, action latency, completed games per
   minute, actor idle, learner idle, and policy staleness.

4. Keep Mctx as a later search box, not a whole trainer.
   Mctx is useful if search is worth its cost, but the repo would still own env,
   replay, targets, learner, checkpoints, eval, and Modal job shape.

5. Let environment fidelity and training quality stay separate.
   The environment lane decides source parity. The coach lane decides whether a
   checkpoint improved. The optimizer lane decides what loop shape and timing
   evidence justify systems work.

## Short Coach Reset

Do not ask LightZero to carry the whole worldview.

Use it as evidence, a replication control, and a comparison. Keep the
repo-native CurvyTron probe all-player, measurable, and shaped as `[B, P]`
while the final framework decision remains open.
