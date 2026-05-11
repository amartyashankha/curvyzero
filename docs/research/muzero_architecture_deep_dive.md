# MuZero Architecture Deep Dive

Status: Research note
Last updated: 2026-05-08

## Scope

This note is an implementation contract for CurvyZero's first MuZero path. It keeps theory only where it changes code shape.

Primary sources: [MuZero paper](https://arxiv.org/pdf/1911.08265), [Nature version](https://www.nature.com/articles/s41586-020-03051-4), [DeepMind explainer](https://deepmind.google/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules), and [google-deepmind/mctx](https://github.com/google-deepmind/mctx). Local sources: `docs/design/deterministic_environment.md`, `docs/research/env_fidelity_curriculum.md`, `docs/research/performance_vectorization.md`, `docs/design/modal_architecture.md`, and the CurvyTron reference under `third_party/curvytron-reference/`.

## One-Screen Data Flow

MuZero uses the real simulator for data collection and a learned latent model for planning.

```text
real obs_t
  -> representation(obs_t) = hidden_state_t
  -> prediction(hidden_state_t) = root_policy_logits_t, root_value_raw_t
  -> MCTS/Mctx recurrent search over learned dynamics
  -> root_action_weights_t, root_value_search_t, selected ego action_t
  -> real CurvyZero wrapper env.step_many(joint_actions_t)
  -> reward_{t+1}, obs_{t+1}, done_{t+1}
  -> replay chunk
  -> trainer samples sequence and unrolls model on stored wrapper actions
```

Implementation invariant: MCTS never calls the real environment. It only calls `recurrent_fn`, which wraps learned dynamics and prediction.

## Model Contract

Use one ego-perspective shared model for v0. Each batch row is "what should this ego player do from this observation?"

```python
class MuZeroModel:
    def representation(params, obs):
        """obs[B, ...] -> hidden[B, ...]"""

    def dynamics(params, hidden, action):
        """hidden[B, ...], action[B] -> reward[B], next_hidden[B, ...], discount[B]"""

    def prediction(params, hidden):
        """hidden[B, ...] -> policy_logits[B, A], value[B]"""
```

Recommended v0 shapes:

```text
obs                  float32[B, C, H, W] plus optional scalars
hidden               float32[B, 64, 8, 8] or float32[B, 64, 16, 16]
action               int32[B]
policy_logits        float32[B, 3]
value                float32[B]
reward               float32[B]
discount             float32[B]
```

### Representation

Purpose: encode the current ego-perspective observation into a fixed hidden state for root search.

CurvyZero v0 observation should be compact and simulator-native, not browser pixels:

- Ego-centered local raster.
- Channels for walls, own trail, opponent trail, own head, opponent head, and optionally recent trail age.
- Scalars for heading/speed/action repeat if not already encoded.
- History frames or previous actions if one frame is not Markov enough.

### Hidden State

The hidden state is not the simulator state. It only needs to preserve information useful for predicting policy, value, and reward during search. Keep it fixed-shape for JAX compilation and small enough for many MCTS node expansions.

### Dynamics / World Model

Purpose: learned transition inside search.

```text
reward, next_hidden, discount = dynamics(hidden, action)
```

For v0, make the wrapper environment deterministic so standard MuZero dynamics is enough:

- No bonuses.
- No stochastic trail gaps, or put solid trails and source-gap trails in separate named rulesets.
- Fixed wrapper decision cadence or action repeat.
- Deterministic tie/scoring rules.

### Action Encoding

Use a fixed action space. v0 recommendation:

```text
0 = turn_left
1 = straight / no_turn
2 = turn_right
```

Reason: the local CurvyTron v1 reference allows no-turn through `PlayerInput.resolve`, client `move: 0`, and server `updateAngularVelocity(0)`. If the target becomes strict CurvyTron 2 left/right, make that a separate two-action ruleset.

For a spatial hidden state, encode action as one-hot planes tiled to hidden resolution and concatenate with hidden channels before dynamics blocks.

### Prediction Heads

Policy head:

```text
policy_logits[B, A]
target = MCTS root action weights[B, A]
loss = cross entropy
```

Value head:

```text
value[B]
target = n-step bootstrapped ego return
loss = Huber or MSE for v0
```

Reward head:

```text
reward[B]
target = observed transition reward
loss = Huber or MSE for v0
```

Scalar value/reward heads are enough for 1v1 terminal rewards. Categorical support can wait until scalar training is unstable.

## Mctx Search Contract

Mctx expects root output plus a recurrent function.

```python
def make_root(params, obs):
    hidden = model.representation(params, obs)
    policy_logits, value = model.prediction(params, hidden)
    return mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=value,
        embedding=hidden,
    )

def recurrent_fn(params, rng_key, action, hidden):
    reward, next_hidden, discount = model.dynamics(params, hidden, action)
    policy_logits, value = model.prediction(params, next_hidden)
    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=policy_logits,
        value=value,
    ), next_hidden
```

Project-owned wrapper:

```python
class SearchPolicy:
    def search(params, obs_batch, legal_action_mask, rng_key, config):
        """Returns SearchOutput with action, action_weights, root_value, diagnostics."""
```

```python
@dataclass
class SearchOutput:
    action: IntArray          # [B]
    action_weights: FloatArray # [B, A]
    root_value: FloatArray    # [B]
    raw_value: FloatArray     # [B]
    diagnostics: dict
```

Store both root policy and root value:

- `action_weights`: training target for policy head.
- `root_value`: bootstraps future value targets.
- `raw_value`: diagnostic for search improvement.

Mctx's public `PolicyOutput` includes `action`, `action_weights`, and `search_tree`; compute `root_value` in the wrapper from the tree with one agreed formula and test it.

## Replay Record Fields

Store replay by chunk, not per-step files. Minimum per ego timestep:

```text
episode_id
timestep
ego_player_id
num_players
obs_t or obs_ref_t
action_t
joint_action_t
reward_t_plus_1
done_t_plus_1
truncated_t_plus_1
root_action_weights_t
root_value_t
raw_value_t
legal_action_mask_t
alive_mask_t
death_tick_t_plus_1
death_cause_t_plus_1
model_step_used_for_search
opponent_policy_versions
episode_seed
rules_hash
observation_schema_hash
reward_schema_hash
search_config_hash
```

Chunk manifest fields:

```text
chunk_id
created_at
producer_actor_id
model_step_min
model_step_max
num_episodes
num_steps
rules_hash
observation_schema_hash
reward_schema_hash
search_config_hash
storage_format_version
```

Training must refuse incompatible `rules_hash`, `observation_schema_hash`, or `reward_schema_hash` unless an explicit migration exists.

## Target Construction

Replay indexing convention:

```text
obs[t]          state before ego action action[t]
reward[t + 1]  reward observed after action[t]
root_policy[t] search action weights at obs[t]
root_value[t]  search root value at obs[t]
```

Sample start `t`, unroll length `K`, and bootstrap length `td_steps`.

For unroll `k = 0..K`:

```text
j = t + k
policy_target[k] = root_policy[j]
value_target[k]  = sum_{m=1..td_steps} discount^(m-1) * reward[j + m]
                   + discount^td_steps * root_value[j + td_steps]
reward_target[k] = masked at k=0
reward_target[k] = reward[t + k] for k>0
```

If terminal occurs before the bootstrap state:

- Drop bootstrap.
- Pad later unroll steps as absorbing state.
- Mask policy/value losses after terminal padding.
- Keep reward targets for observed terminal transition.

Trainer loss:

```text
loss =
  policy_ce(pred_policy[k], policy_target[k])
  + value_loss(pred_value[k], value_target[k])
  + reward_loss(pred_reward[k], reward_target[k])
  + weight_decay
```

Log losses by unroll depth. Reward/value off-by-one bugs often look like weak learning, not crashes.

## Self-Play Actor Loop

Actor owns real env stepping and search calls.

```text
load checkpoint params at step S
reset vectorized env batch
while actor is collecting:
  choose live ego rows needing a decision
  build obs_batch and legal_action_mask
  run SearchPolicy.search(params, obs_batch, mask, rng)
  choose ego actions from search output
  choose opponent actions from configured opponent policy
  assemble joint_actions
  env.step_many(joint_actions)
  append replay rows with search outputs and env results
  flush complete episodes or fixed-size chunks
  poll for newer checkpoint only between chunks/episodes
```

Opponent progression:

1. Random/heuristic for simulator gates.
2. Latest raw policy, no search.
3. Checkpoint pool.
4. Independent batched search for both players in 1v1 evaluation.

## Trainer Loop

Trainer owns parameters, optimization, checkpointing, and replay consumption.

```text
initialize params or load checkpoint
while training:
  discover replay chunks compatible with config hashes
  sample batch of sequence starts
  build obs_t, actions[t:t+K], targets[t:t+K]
  run initial inference and K recurrent unrolls
  compute policy/value/reward losses
  apply optimizer step
  periodically write immutable checkpoint
  update latest checkpoint pointer after write completes
  periodically launch/evaluate fixed checkpoint
```

Do not add Reanalyze to v0. Add it after the base actor/trainer path learns and replay integrity is stable.

## Evaluator Loop

Evaluator should be separate from trainer metrics.

```text
load explicit checkpoint step S
for fixed seeds and held-out seeds:
  run raw policy, no MCTS
  run search policy with evaluation search config
  record win rate, episode length, death cause, reward, seat bias
compare against random, heuristic, and older checkpoints
write evaluation artifact tied to checkpoint S
```

Always report raw policy and searched policy. If search helps a lot, the policy has not internalized search yet. If search hurts, suspect learned dynamics, root value calculation, masks, or search hyperparameters.

## Weight Handoff

Use coarse checkpoint handoff, not per-step RPC.

```text
trainer:
  writes checkpoints/step_000123456/
  fsyncs or completes artifact write
  updates latest.json -> step_000123456

actor:
  polls latest.json between chunks/episodes
  loads params to device memory
  records model_step_used_for_search in replay

evaluator:
  loads explicit step, never unresolved latest
```

Modal guidance:

- Keep env stepping, model inference, and MCTS in one process/container hot path.
- Use Modal storage for checkpoints/replay chunks/logs.
- Use Queue/Dict only for coarse coordination, not per tick, per inference, or per MCTS node.

## MCTS And Inference Batching

Batch roots across environments and ego perspectives.

```text
obs_batch[B, ...]
  -> jitted representation/prediction
  -> mctx.gumbel_muzero_policy or mctx.muzero_policy
  -> action[B], action_weights[B, A], search_tree
  -> wrapper computes root_value[B]
  -> env.step_many(joint_actions)
```

Static compile dimensions for first benchmarks:

```text
batch_size:       64, 256, 1024
num_actions:      3
num_simulations:  16, 32, 50
max_depth:        fixed or None, but benchmark explicitly
hidden_shape:     fixed
```

Avoid:

- One Python search call per env.
- Shape changes every decision.
- CPU/GPU transfers inside search.
- Modal network calls in the hot loop.

Measure separately:

- JAX compile time.
- Steady-state search throughput.
- Env step throughput.
- Observation generation time.
- Replay write throughput.

## Multiplayer Contract

Do not start with joint-action search. It changes the action space from `A` to `A^N`.

v0 and near-term multiplayer should use ego-perspective rows:

```text
one replay/search row = one ego player in one simultaneous-action game
model predicts ego action only
opponents act by configured policies
real env applies all player actions simultaneously
```

For N-player rewards, use ego-perspective rank reward:

```text
reward_i = 2 * score_i / (num_players - 1) - 1
```

Tie policy must be explicit and tested. In 1v1, winner `+1`, loser `-1`, exact tie `0/0` is the simplest v0 default.

`to_play` is not turn order in CurvyTron. It means ego perspective:

```text
ego_player_id
perspective_transform
alive_mask
opponent_policy_versions
```

## Minimal v0 Recommendation

Environment:

- `curvyzero-v0-1v1-no-bonus`.
- 1v1, one round per episode.
- Actions: left, straight, right.
- Fixed wrapper decision cadence or action repeat.
- Deterministic collision/scoring/tie rules.
- No bonuses.
- Solid-trail v0 or source-gap v0, but name the variant explicitly.
- Terminal reward: win `+1`, loss `-1`, exact tie `0`.

Model:

- JAX/Flax if Mctx smoke passes.
- Small CNN representation, 3-5 residual blocks.
- Spatial hidden state, about 64 channels at 8x8 or 16x16.
- Dynamics with action planes plus 1-2 residual blocks.
- Scalar reward and value heads.
- Policy head over 3 actions.
- Unroll `K = 5`.

Search:

- Project wrapper over Mctx.
- Prefer `gumbel_muzero_policy` for the first Mctx path unless smoke tests show standard `muzero_policy` is simpler or more stable.
- Start at 16 or 32 simulations; benchmark 50.
- Store `action_weights`, `root_value`, `raw_value`, and search config hash.

System:

- First implementation can be a single Modal GPU process group with vector envs, batched search, replay writer, and trainer.
- Preserve actor/trainer/evaluator boundaries in code and artifacts so it can split later.
- Checkpoint handoff through immutable checkpoint directories plus latest pointer.

Gates:

- Target construction tests on tiny hand-authored trajectories.
- Mctx synthetic throughput report.
- Deterministic rollout fingerprints.
- MuZero beats random, then challenges heuristic.
- Raw policy and searched policy both evaluated.

## Open Questions

- Is canonical v0 action space 3 actions from CurvyTron v1 source behavior, or 2 actions for a stricter CurvyTron 2-style target?
- What action repeat balances control fidelity and search cost?
- What exact root value formula should the Mctx wrapper use from `search_tree`?
- Does local raster beat ray/hybrid observations for first learnability?
- Should solid trails or source-randomized trail gaps be canonical v0?
- What same-tick death/tie scoring should be locked into golden tests?
- How stale can actor checkpoints be before policy lag hurts training?
- When adding 3+ players, are policy-only opponents enough, or is independent batched search needed?
