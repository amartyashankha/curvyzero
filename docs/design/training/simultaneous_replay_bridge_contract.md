# Simultaneous Replay Bridge Contract

Purpose: define the missing bridge for true CurvyTron two-seat current-policy
self-play.

## Problem

CurvyTron advances one physical tick after both players choose actions.

```text
state_t + action_player_0 + action_player_1 -> state_t+1 + rewards_per_player
```

Stock LightZero expects one scalar action and one reward per env transition.

The custom two-seat collector solved action collection, but then built replay
rows and targets itself. That is the untrusted part.

## Minimum Safe Bridge

A native-compatible bridge should:

1. collect both actions from the same pre-tick state;
2. step the env once with the joint action;
3. create one seat-perspective trajectory per active player;
4. encode each trajectory as native-compatible `GameSegment` data;
5. push those segments through `MuZeroGameBuffer`;
6. let LightZero sample targets;
7. prove target parity on tiny known trajectories before any large run.

## First Parity Test

Use one hand-authored trace with two seats and three physical ticks.

Example:

```text
joint actions: [(2, 0), (1, 2), (0, 1)]
seat 0 rewards: [1, 2, 4]
seat 1 rewards: [1, 0, -2]
terminal after tick 2
discount: 1.0
no bootstrap
```

Convert it into exactly two native-compatible trajectories:

- seat 0 trajectory: seat-0 observations, seat-0 actions, seat-0 rewards,
  seat-0 visit distributions;
- seat 1 trajectory: seat-1 observations, seat-1 actions, seat-1 rewards,
  seat-1 visit distributions.

Expected value targets from tick 0:

```text
seat 0: [7, 6, 4]
seat 1: [-1, -2, -2]
```

The test must prove:

- no fake pending rows;
- no joint action stored as a seat-local action;
- reward targets align with real physical ticks;
- policy targets match the stored visit distributions;
- `to_play` is deliberate, with `-1` as the starting default unless board-game
  semantics are intentionally tested.

Current status: this first tiny parity test passes in the Modal/LightZero
runtime. It builds native `GameSegment`s, pushes them into `MuZeroGameBuffer`,
and verifies deterministic reward/value/policy targets for the hand-authored
trace. This validates the bridge contract at toy scale only; it is not yet
wired into the active two-seat trainer.

## Things To Test

- No fake pending rows enter replay.
- Rewards belong to the correct seat perspective.
- `to_play` semantics are deliberate and tested.
- Terminal rows and bootstrap masks are correct.
- Policy visit targets are attached to the action actually taken by that seat.
- Replaying the same tiny trajectory gives expected value/reward/policy targets.

## Non-Goal

This contract does not require a massive distributed actor system first. It
only requires that the data entering the learner has the same meaning as stock
LightZero data.
