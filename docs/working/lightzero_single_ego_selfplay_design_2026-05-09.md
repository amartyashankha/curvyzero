# LightZero Single-Ego Self-Play Design

Date: 2026-05-09

Scope: practical LightZero wrapper path for CurvyTron 1v1, no bonuses.

This is a wrapper design, not a claim that LightZero is a natural multiplayer
self-play engine. The goal is smaller: make CurvyTron look like a single-agent
environment to LightZero, while the wrapper owns the second player's action and
keeps enough trace data to audit what really happened.

## Short Version

Use one LightZero-controlled ego player per episode.

At each step:

1. LightZero chooses the ego action.
2. The wrapper chooses the opponent action.
3. The wrapper builds the full joint action/control snapshot and advances the
   CurvyTron environment.
4. The wrapper returns only the ego observation, ego reward, ego done flag, and
   audit info to LightZero.

Train one shared policy by rotating which seat is the ego. Do not expose
joint-action search to LightZero for the first useful version.

## Base Environment Assumptions

Use the existing CurvyTron 1v1 environment with:

- exactly two players, `player_0` and `player_1`;
- no bonuses;
- the normal wrapper discrete steering action set;
- deterministic reset from seed;
- deterministic wrapper step from state plus joint action/control snapshot;
- ego-perspective observation available for either player;
- per-player rewards and terminal flags available after each step.

The wrapper should not change game rules. It should only choose who LightZero is
controlling and fill in the missing opponent action.

## Wrapper State

Each wrapper instance should track:

- `episode_id`;
- `seed`;
- `ruleset_id` and config hash;
- `observation_schema_id`;
- `reward_schema_id`;
- `action_schema_id`;
- `ego_player_id`;
- `opponent_player_id`;
- opponent kind and opponent version;
- current raw CurvyTron state or last per-player observations;
- ordered wrapper joint-action/control trace;
- per-step ego observation hashes;
- per-step reward and terminal records;
- final winner or draw.

This state is for honesty. LightZero can ignore most of it, but logs and replay
cannot.

## Ego Seat Rotation

LightZero still sees a single-agent problem. The wrapper changes which seat that
single agent controls.

Recommended schedule:

- episode 0: ego is `player_0`;
- episode 1: ego is `player_1`;
- keep alternating, or sample seats with a balanced random schedule.

For the policy, both seats use the same observation schema. The observation
function must produce "me versus opponent" features from either seat:

- ego position, heading, speed, trail, and alive flag are always in ego slots;
- opponent position, heading, speed, trail, and alive flag are always in
  opponent slots;
- relative geometry is expressed from the ego frame;
- action meanings are ego-local, for example left, straight, right.

This is the main trick. We do not train separate `player_0` and `player_1`
policies. We train one policy that always thinks it is "me".

## Step Flow

On reset:

1. Pick the ego seat.
2. Reset CurvyTron with the episode seed.
3. Store both players' initial observations or enough state to rebuild them.
4. Return the ego-perspective observation to LightZero.
5. Include the legal action mask.
6. Set LightZero's `to_play` to the single-agent convention used by the chosen
   LightZero config.

On step:

1. Receive one ego action from LightZero.
2. Validate it against the action schema and action mask.
3. Build the opponent observation from the same pre-step state.
4. Ask the configured opponent policy for one action.
5. Create a wrapper joint action/control snapshot:
   - if ego is `player_0`, `{player_0: ego_action, player_1: opponent_action}`;
   - if ego is `player_1`, `{player_0: opponent_action, player_1: ego_action}`.
6. Advance the CurvyTron wrapper once with that snapshot.
7. Log the pre-step state hash, both observations hashes, both chosen actions,
   reward map, terminal map, and next-state hash.
8. Return only the ego next observation, ego reward, ego done flag, and info.

The opponent action must be chosen from the pre-step state, not from the state
after the ego has acted. The wrapper applies all live-player control choices at
one decision boundary so neither side gets a hidden timing advantage.

## Opponent Modes

The wrapper should support three opponent families.

### Scripted Opponent

Use this for first learnability checks and deterministic replay.

Examples:

- random legal action;
- sticky random action;
- wall-avoidance heuristic;
- chase or pressure heuristic;
- fixed action trace loaded from a file.

Required metadata:

- `opponent_kind: scripted`;
- `opponent_id`;
- script version;
- script seed;
- script parameters.

Scripted opponents are easiest to audit. They should be the first gate before
any live self-play claim.

### Frozen Policy Opponent

Use this for stable eval and league-style progress checks.

The opponent loads a checkpoint and never changes during the episode or eval
batch. It receives the opponent's ego-perspective observation, because from its
point of view it is also "me".

Required metadata:

- `opponent_kind: frozen_policy`;
- checkpoint path or artifact id;
- checkpoint hash;
- model config hash;
- observation schema id;
- action schema id;
- inference seed, if sampling is used;
- deterministic or stochastic action mode.

Frozen opponents are the main way to answer, "Did the current learner actually
get better?"

### Current-Policy Opponent

Use this only after scripted and frozen gates are healthy.

The opponent uses the learner's current policy, or a periodically refreshed
copy of it. This gives cheap shared-policy self-play without teaching LightZero
about two live players.

There are two workable variants:

- Snapshot current policy at episode start and keep it fixed for that episode.
- Refresh from learner weights every N episodes or every N collector batches.

Prefer the snapshot version first. If the opponent weights change mid-episode,
the replay trace becomes harder to explain and repeat.

Required metadata:

- `opponent_kind: current_policy`;
- learner step or checkpoint id used for the snapshot;
- refresh rule;
- deterministic or stochastic action mode;
- action temperature or exploration settings.

Current-policy opponents are useful for pressure, but they are not clean
evidence by themselves. Eval still needs frozen opponents and a fresh recorded
pseudo-random eval seed list per wave.

## Joint-Action Logging

Every wrapper environment step should emit one compact audit record.

Minimum fields:

- `episode_id`;
- `step_index`;
- `seed`;
- `ego_player_id`;
- `opponent_player_id`;
- `pre_state_hash`;
- `ego_observation_hash`;
- `opponent_observation_hash`;
- `ego_action`;
- `opponent_action`;
- `joint_action_by_player`;
- `action_source_by_player`;
- `post_state_hash`;
- `reward_by_player`;
- `terminated_by_player`;
- `truncated_by_player`;
- `winner`;
- `ruleset_id`;
- `observation_schema_id`;
- `reward_schema_id`;
- `opponent_kind`;
- `opponent_version`;

LightZero replay may only need ego transitions, but CurvyZero needs the full
wrapper joint-action/control trace. If a run collapses into bad actions or
impossible wins, that trace is the first thing to inspect.

## Replay Honesty

There are two different replay concepts. Keep them separate.

LightZero replay stores the single-agent training view:

- ego observation;
- ego action;
- ego reward;
- ego done;
- policy/search metadata;
- value targets from the ego perspective.

CurvyZero audit replay stores the wrapper/game view:

- seed;
- ego seat;
- opponent identity;
- full joint action/control snapshot per wrapper transition;
- reward and terminal maps;
- hashes for states, observations, schemas, and rules.

Do not pretend LightZero's single-agent replay is a complete record of the game.
It is a training projection. The audit replay is the source of truth for
reconstruction.

For exact replay, the opponent action must either be logged directly or be
recomputed from fully logged opponent metadata. Logging it directly is safer and
cheaper.

## Evaluation Honesty

Evaluation should not use the same moving opponent that generated training
data, unless the result is clearly labeled as live self-play only.

Use a frozen-opponent eval ladder, sampled with a fresh pseudo-random eval seed
set for each eval wave:

- random legal opponent;
- sticky random opponent;
- simple wall-avoidance opponent;
- one or more frozen old checkpoints;
- current checkpoint on both seats, with seat balance reported.

For every eval batch, report:

- number of games;
- seed list or seed range;
- eval seed generator seed when a pseudo-random list was sampled;
- seat split;
- opponent ids and hashes;
- win/loss/draw;
- survival ticks;
- timeout rate;
- illegal action count;
- deterministic replay failures;
- action distribution by seat;
- mean return from each seat.

Seat balance matters. If the policy only wins as `player_0`, it is not yet a
shared ego policy.

## Training Data Shape

Each real CurvyTron episode can produce one LightZero episode from the selected
ego seat. That is the simplest path.

A later optimization can produce two ego-view training records from one real
joint episode: one as `player_0`, one as `player_1`. That is attractive, but it
is easier to get wrong because the second record's "policy action" came from
the opponent policy, not necessarily from LightZero's collector decision. Do not
use that shortcut until the one-ego-per-episode path is trustworthy.

## Practical Gates

Gate 1: scripted opponent

- deterministic seeds replay exactly;
- joint-action logs explain every terminal;
- ego seat alternation produces balanced data volume;
- eval against scripted opponents is stable.

Gate 2: frozen policy opponent

- checkpoint metadata is complete;
- frozen opponent inference is deterministic when requested;
- current learner can be evaluated against old checkpoints without changing the
  eval opponent mid-batch.

Gate 3: current-policy opponent

- opponent snapshot is fixed for the episode;
- refresh cadence is logged;
- replay records identify the exact opponent version;
- live self-play metrics are not mixed with fixed-opponent eval metrics.

## What This Cannot Do Cleanly

This wrapper does not make LightZero a true simultaneous multi-agent planner.

It cannot cleanly:

- search over both players' joint actions;
- model opponent adaptation inside MCTS;
- learn separate values for both players from one tree;
- represent general-sum multiplayer objectives;
- handle more than two players without more bookkeeping and weaker semantics;
- prove that self-play improved without a separate frozen eval ladder;
- make current-policy opponents exactly reproducible unless snapshots and
  sampled actions are logged;
- remove non-stationarity from live self-play;
- use LightZero's replay as the full game replay;
- explain seat bias unless eval reports seats separately.

It is also a poor fit for bonus-heavy CurvyTron at first. Bonuses add hidden
credit-assignment questions and action-context shifts. Keep this path on
1v1/no-bonus until the wrapper, logs, and eval ladder are boring.

## Recommendation

Build the wrapper as a contained experiment. Keep CurvyTron rules, observations,
and replay contracts project-owned. Let LightZero see one ego-controlled player,
and make the wrapper responsible for the opponent, seat rotation, joint-action
logs, and evaluation metadata.

This is practical enough to test whether LightZero helps. It is not clean enough
to become the long-term multiplayer abstraction unless the simple 1v1/no-bonus
path first proves learning, replayability, and honest eval.
