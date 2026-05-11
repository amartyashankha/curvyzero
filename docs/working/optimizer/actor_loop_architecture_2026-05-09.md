# Actor-Loop Architecture

Date: 2026-05-09

Status: framework-agnostic optimizer plan.

## Short Read

Whatever framework wins, the same pieces must exist:

- environment;
- observation builder;
- policy/search;
- opponent/checkpoint assignment;
- replay or rollout buffer;
- learner;
- checkpoint publisher;
- evaluator/scorecard;
- profiler.

The framework may provide some pieces. The repo must still own the contracts and
metadata.

Native CurvyTron is held player control state plus elapsed-ms server frames.
The `joint_action[B, P]` below is the trainer-wrapper/replay action map, not a
source-native object.

## Target Loop

```text
latest policy/checkpoint
  -> reset/autoreset env rows
  -> build obs[B, P, ...], mask[B, P, A], live[B, P]
  -> compact live players into ego rows
  -> policy/search batch returns action, action weights, root value
  -> scatter ego actions into wrapper joint_action[B, P]
  -> trainer env.step(wrapper joint_action)
  -> stage transition rows and final observations
  -> write replay/rollout chunks
  -> learner samples/updates
  -> checkpoint/eval/publish
```

For PPO, `action_weights` may be a one-hot or policy distribution/logprob
surface. For MuZero, `action_weights` are search visit/action weights.

## Environment Contract

Start with two players, but keep the shape multiplayer:

```text
reset_many(seed[B]) -> timestep
step_many(wrapper joint_action[B, P]) -> timestep
```

Timestep fields should include:

- `observation[B, P, ...]`;
- `legal_action_mask[B, P, 3]`;
- `live_mask[B, P]`;
- `reward[B, P]`;
- `done[B]`;
- `terminated[B]`;
- `truncated[B]`;
- `final_observation[B, P, ...]` when done;
- `final_reward_map[B, P]` when done;
- `episode_id`, `reset_seed`, `reset_source`;
- compact trace/event references, not full debug JSON in the hot path.

## Observation Path

First optimizer-friendly observation path:

- egocentric ray observations for 1v1/no-bonus;
- fixed shape and dtype;
- one row per player;
- no Python dicts inside the hot policy batch;
- schema/hash carried into replay.

Visual path:

- use stacked local raster or frame history before any visual-learning claim;
- avoid single-frame flat rasters as quality evidence;
- measure crop/packing cost separately from env stepping.

## Opponent And Self-Play

For 2-player first:

- current shared policy can control both players;
- scripted/random/heuristic opponents stay in the eval and curriculum ladder;
- checkpoint-pool opponents are explicit metadata, not hidden state.

For more players later:

- keep one ego row per live player;
- aggregate or mask other players in observation;
- keep centered rank payoff or per-player payoff maps;
- avoid joint-action MCTS as the default plan.

## Learner Shapes

PPO/IPPO runner:

- rollout buffer stores observations, actions, logprobs, values, rewards,
  done flags, episode ids, player ids, and opponent/checkpoint metadata;
- update loop is transparent and easy to profile;
- good first learnability/speed diagnostic.

MuZero runner:

- replay stores observations, actions, rewards, done flags, action weights,
  root values, policy/checkpoint id, and target metadata;
- model/search may be LightZero or project-owned Mctx;
- only promote after target quality and loop timing justify search complexity.

## LightZero Plug-In Boundary

LightZero can sit behind a single-ego adapter:

```text
one focal ego obs -> LightZero action/search -> wrapper supplies opponents
```

That is useful for serious MuZero replication/control, comparison, checkpoint
and eval plumbing, and target audits. It should not become the core CurvyTron
architecture unless it can preserve all-player self-play metadata, final
observations, replay contract, and profiling buckets without distorting the
game.

## Mctx Plug-In Boundary

Mctx should live only in the policy/search box:

```text
ego observations -> representation/prediction -> Mctx search -> action weights
```

The real environment stays outside MCTS. Replay, learner, checkpointing, and
Modal run management remain project-owned unless another framework proves it can
own them without hiding needed metadata.

## Source Anchors

- [Trainer contract](../../../src/curvyzero/env/trainer_contract.py)
- [Replay chunk v0](../../../src/curvyzero/training/replay_chunk_v0.py)
- [Policy row mapping](../../../src/curvyzero/training/policy_row_mapping.py)
- [Self-play speed lane](../environment/selfplay_speed_lane_2026-05-09.md)
- [Multiplayer self-play notes](../../research/multiplayer_selfplay_muzero.md)
