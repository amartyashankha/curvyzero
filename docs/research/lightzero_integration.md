# LightZero integration critique

Research snapshot: 2026-05-08.

## Short Answer

LightZero is a useful PyTorch/MuZero reference and may be worth a contained
spike, but it should not be the default CurvyZero training backbone yet. Its
documented environment interface fits single-agent Gym-style tasks and
alternating board games better than CurvyZero's all-live-player wrapper loop.

The most plausible minimal integration is an ego-agent wrapper: LightZero
controls one player, opponents are driven by a fixed random/heuristic/frozen
policy inside the wrapper, and the wrapper returns a scalar ego reward. That is
compatible with the current `CurvyTronEnv` shape and with the v0 training
decision to avoid joint-action search. A joint-action wrapper is technically
possible, but it makes the action space grow as `A ** N` and turns free-for-all
self-play into a single centralized controller problem.

Recommendation: postpone adoption until after baseline learnability, then run a
small LightZero spike with the ego wrapper, pure-policy collection, and low
simulation-count MuZero/Gumbel MuZero. Reject LightZero if the spike requires
forking the collector/search stack, cannot preserve CurvyZero deterministic
traces, or loses most of its time in MCTS and CPU/GPU conversion overhead.

## Evidence From LightZero

LightZero presents itself as a PyTorch toolkit for MCTS plus deep RL, with
AlphaZero, MuZero, Sampled MuZero, Stochastic MuZero, EfficientZero, Gumbel
MuZero, ReZero, and UniZero variants. Its public README also describes C++ and
Python MCTS implementations, `ctree` and `ptree`, and benchmarked environments
such as board games, Atari, continuous-control tasks, MiniGrid, and GoBigger.
See the [LightZero GitHub repo](https://github.com/opendilab/LightZero) and
the [LightZero docs home](https://opendilab.github.io/LightZero/index.html).

The paper frames LightZero as a unified benchmark/toolkit for general sequential
decision scenarios, but it also names the standard MCTS pain points that matter
for CurvyZero: complex action spaces, stochasticity/partial observability,
reliance on priors, simulation cost, and hard exploration. See
[LightZero: A Unified Benchmark for Monte Carlo Tree Search in General
Sequential Decision Scenarios](https://arxiv.org/abs/2310.08348) and the
[NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract-Datasets_and_Benchmarks.html).

## Custom Environment Interface

LightZero custom envs are based on DI-engine's `BaseEnv` pattern or a Gym
wrapper. The required observation is a dictionary, not a raw array:

```python
{
    "observation": obs,
    "action_mask": action_mask,
    "to_play": -1,
}
```

For non-board-game environments, docs set `to_play=-1`; discrete action masks
are all-ones arrays, while continuous action spaces use `None`. The `step`
method returns `BaseEnvTimestep(lightzero_obs_dict, reward, done, info)`. Board
games may add fields such as `board` and `current_player_index`, and are
expected to implement helpers such as `legal_actions`, `bot_action`, and
`random_action`. Source:
[custom environment tutorial](https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html).

`LightZeroEnvWrapper` is mainly a wrapper for Gym classic-control/box2d-style
tasks. Its source asserts an `is_train` flag, stores `env_id` and `continuous`,
converts reset/step observations into the LightZero dict, and accumulates
`eval_episode_return`. It assumes one reward and one done flag, not a
multi-agent result object. Source:
[LightZeroEnvWrapper source](https://opendilab.github.io/LightZero/_modules/lzero/envs/wrappers/lightzero_env_wrapper.html).

Current CurvyZero does not match this directly:

- `CurvyTronEnv.reset(seed)` returns `dict[player_id, np.ndarray]`.
- The trainer wrapper step call accepts one per-player wrapper action/control choice,
  then maps them onto held source controls for the elapsed-ms frame window.
- `StepResult` returns per-player observations, rewards, terminations,
  truncations, and infos.
- v0 currently supports exactly 2 players and usually 3 actions per player.

Therefore, the adapter should not wrap the existing env unchanged with
`LightZeroEnvWrapper`. A purpose-built `CurvyZeroLightZeroEnv(BaseEnv)` or
single-agent Gym-style wrapper is cleaner.

## Self-Play And Multi-Agent Limitations

LightZero's documented board-game modes are `self_play_mode`,
`play_with_bot_mode`, and `eval_mode`. In `self_play_mode`, each `step` places
one move and terminal reward is returned when the game is decided. In
`play_with_bot_mode`, the agent moves, then the bot moves. Eval records return
from player 1's perspective. Source:
[custom environment board-game methods](https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html#special-methods-for-board-game-environments).

That is not CurvyTron source semantics. CurvyTron holds player input/control
state while elapsed-ms server frames advance the world. For v0, LightZero can
only be made natural by hiding opponents inside the environment wrapper or by
centralizing the wrapper joint action/control snapshot.

Important limitations for CurvyZero:

- `to_play` helps distinguish single-agent logic from alternating board-game
  self-play, but it is not a complete general-sum multi-agent API.
- The public collector produces one action per active env step from the policy,
  not a per-agent action map.
- LightZero's policy/value targets are scalar and action-distribution based.
  Vector values, centered rank payoff for every player, checkpoint-pool
  opponents, and focal-agent-only searched self-play would require custom
  code around or inside the collector.
- The LightZero paper includes GoBigger in its benchmark discussion, so the
  project has some multi-agent experience. The documented public adapter path,
  however, is still MDP/board-game shaped enough that CurvyZero should treat
  general simultaneous multiplayer support as unproven until measured in a
  repo-local spike.

Practical conclusion: LightZero can test ego-vs-opponent MuZero. It should not
be treated as a turnkey free-for-all self-play system.

## MCTS Batching And Performance Risks

LightZero has a real performance advantage on paper: C++ tree operations and
batched model inference across root nodes. Its MCTS docs say the core
`batch_traverse` and `batch_backpropagate` functions are in C++, and the search
is parallel in model inference. Source:
[MCTS tree-search docs](https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html).

The risk is that this batching only helps if CurvyZero can feed enough active
roots with similar shapes. The policy source runs initial inference over the
active env batch, creates C++ or Python root batches, runs search, then selects
one action per env. Source:
[MuZero policy source](https://opendilab.github.io/LightZero/_modules/lzero/policy/muzero.html).

The MCTS source also shows a per-simulation Python loop around C++ traversal and
PyTorch recurrent inference. It converts latent states and actions into
tensors, calls `model.recurrent_inference`, then detaches outputs back to CPU
NumPy/lists for the C++ tree. Source:
[MuZero ctree source](https://opendilab.github.io/LightZero/_modules/lzero/mcts/tree_search/mcts_ctree.html).

CurvyZero-specific risks:

- The current raw simulator smoke benchmark is cheap, about 37k steps/sec in an
  early local run. LightZero overhead could dominate before the model is useful.
- If the batch is small, MCTS will not amortize Python control flow, C++ object
  setup, and tensor/NumPy transfers.
- If the wrapper drives opponent policies inside `env.step`, LightZero's MCTS
  does not search opponent choices unless we encode them into the action space
  or into the learned dynamics.
- Joint-action search grows quickly: 2 players with 3 wrapper actions is 9 joint
  actions; 6 players is 729 joint actions per wrapper decision before horizon
  expansion.
- CurvyTron episodes can be long, dense-tick control problems. Running 50+
  simulations at every tick may spend compute improving tiny steering choices
  while starving actual self-play data.
- The documented collector is episode-based and "serial" in orchestration, even
  with vectorized envs. Source:
  [MuZeroCollector docs](https://opendilab.github.io/LightZero/api_doc/worker/index.html)
  and
  [MuZeroCollector source](https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html).

The first LightZero benchmark should therefore separate pure wrapper overhead
from MCTS overhead. If pure-policy collection is already slow, MCTS will only
make the diagnosis noisier.

## Minimal CurvyZero Wrapper

The minimal wrapper should be deliberately single-agent from LightZero's point
of view:

```python
class CurvyZeroLightZeroEnv(BaseEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = CurvyTronEnv(cfg.curvy_config)
        self.ego = cfg.ego_player_id
        self.opponent_policy = cfg.opponent_policy
        self._eval_episode_return = 0.0
        self._last_obs_by_agent = None

    def reset(self):
        self._last_obs_by_agent = self.env.reset(seed=self._next_seed())
        self._eval_episode_return = 0.0
        return {
            "observation": self._last_obs_by_agent[self.ego],
            "action_mask": np.ones(self.env.config.action_count, dtype=np.int8),
            "to_play": -1,
        }

    def step(self, ego_action):
        actions = {self.ego: int(ego_action)}
        for agent in self.env.agents:
            if agent != self.ego:
                actions[agent] = self.opponent_policy.act(self._last_obs_by_agent[agent])

        result = self.env.step(actions)
        self._last_obs_by_agent = result.observations
        reward = float(result.rewards[self.ego])
        done = bool(result.terminated[self.ego] or result.truncated[self.ego])
        self._eval_episode_return += reward

        info = dict(result.infos[self.ego])
        info.update({
            "actions": actions,
            "rules_hash": self.env.config.rules_hash,
            "terminal_rewards": result.rewards,
        })
        if done:
            info["eval_episode_return"] = self._eval_episode_return

        return BaseEnvTimestep({
            "observation": result.observations[self.ego],
            "action_mask": np.ones(self.env.config.action_count, dtype=np.int8),
            "to_play": -1,
        }, reward, done, info)
```

Keep this wrapper out of `curvyzero_env/`; it belongs in a future wrappers or
training experiment package. The core simulator should not import LightZero,
DI-engine, Gym, or torch.

Two variants are worth testing:

- Ego-vs-scripted/frozen opponent: lowest-risk spike. This matches baseline
  learnability and tests LightZero without claiming full self-play.
- Rotating ego wrapper: collect one LightZero episode per ego seat and share the
  same network weights across seats. This approximates shared-policy self-play
  without changing LightZero internals.

Avoid for v0:

- Joint-action action space unless used only as a synthetic throughput test.
- Full n-player vector values.
- Modifying LightZero's C++ tree code before proving that the wrapper and
  collector are acceptable.

## What To Benchmark

Run the spike in stages and stop early when a gate fails.

1. Install/import smoke:
   - Can LightZero and DI-engine install cleanly in this repo's Python setup?
   - Does a tiny official example run without patching package versions?
   - Record PyTorch, CUDA/MPS/CPU backend, LightZero commit/tag, and DI-engine
     version.

2. Wrapper correctness:
   - Same seed and same scripted action trace produce byte-stable CurvyZero
     summaries with and without the LightZero wrapper.
   - `action_mask`, `to_play`, observation dtype/shape, reward, done, and
     `eval_episode_return` stay stable across reset/autoreset paths.
   - Opponent-policy actions are logged so any death is replayable.

3. Pure-policy throughput:
   - Raw `CurvyTronEnv` steps/sec.
   - Wrapper-only steps/sec with random ego and random opponent.
   - LightZero collector with `collect_with_pure_policy=True`.
   - Direct PyTorch policy runner with the same model, to isolate LightZero
     collector overhead.

4. MCTS throughput:
   - `num_simulations` in `{1, 4, 8, 16, 32, 50}`.
   - Batch sizes in `{1, 8, 32, 128}` active envs, if memory allows.
   - CPU time in env stepping, policy initial inference, MCTS search, recurrent
     inference, and tensor/NumPy transfers.
   - GPU/MPS utilization if using accelerator.

5. Learning smoke:
   - Ego-vs-random and ego-vs-one-step-safe opponent.
   - Tiny MLP observation model first; raster only after low-dimensional smoke.
   - Compare pure policy, MuZero, and Gumbel MuZero at the same wall-clock
     budget, not only the same env-step budget.
   - Evaluation on held-out seeds against random, sticky-random, and heuristic
     opponents.

6. Self-play stress:
   - Rotate ego seat and train one shared model.
   - Evaluate latest checkpoint against a small frozen checkpoint pool outside
     LightZero's default collector if necessary.
   - Track seat bias, action collapse, timeout rate, and deterministic replay
     failures.

## Choose Or Reject Criteria

Choose LightZero for the next CurvyZero spike only if all of these hold:

- The ego wrapper is small, local, deterministic, and does not require changing
  `curvyzero_env`.
- Pure-policy collection overhead is close enough to a direct PyTorch runner
  that LightZero is not the bottleneck before MCTS starts.
- Batched MCTS throughput improves meaningfully with env batch size and does
  not collapse under CPU/GPU transfer overhead.
- A low-simulation MuZero or Gumbel MuZero run beats, or at least clearly
  improves over, pure-policy training at the same wall-clock budget.
- The integration can preserve repo priorities: deterministic traces,
  replayable failures, rules/observation hashes, and local benchmark artifacts.
- No fork of LightZero's collector or C++ tree code is needed for the first
  useful result.

Reject or defer LightZero if any of these happen:

- The only viable integration is joint-action search for both players.
- Simultaneous self-play requires invasive collector changes before the 1v1
  ego-vs-opponent case learns.
- MCTS consumes most wall time while producing no win-rate lift over pure
  policy or PPO-style baselines.
- Version or dependency friction makes LightZero harder to reproduce than a
  small repo-local PyTorch runner.
- The wrapper obscures CurvyZero's deterministic seed/action trace or loses
  per-agent terminal details.
- Multi-agent ambitions push the project toward vector values, checkpoint
  leagues, or custom general-sum MCTS before baseline learnability is proven.

## Recommendation

Do not commit to LightZero in the core architecture. Keep it as an experiment
behind `docs/research`, later a thin wrapper package, and benchmark it against
the same environment/reward/observation contracts used for PPO and any custom
MuZero path.

The likely best use of LightZero is educational and comparative: it can expose
what a mature PyTorch MuZero stack expects, provide baseline MCTS code, and
serve as a wall-clock benchmark for "would a framework save us time?" It is not
currently the cleanest answer for CurvyTron's simultaneous multiplayer
self-play.

## Sources

- [LightZero documentation](https://opendilab.github.io/LightZero/index.html)
- [LightZero GitHub repository](https://github.com/opendilab/LightZero)
- [LightZero paper on arXiv](https://arxiv.org/abs/2310.08348)
- [LightZero NeurIPS 2023 page](https://proceedings.neurips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract-Datasets_and_Benchmarks.html)
- [How to customize environments in LightZero](https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html)
- [LightZeroEnvWrapper source](https://opendilab.github.io/LightZero/_modules/lzero/envs/wrappers/lightzero_env_wrapper.html)
- [LightZero MCTS tree-search docs](https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html)
- [LightZero MuZero ctree source](https://opendilab.github.io/LightZero/_modules/lzero/mcts/tree_search/mcts_ctree.html)
- [LightZero MuZero policy source](https://opendilab.github.io/LightZero/_modules/lzero/policy/muzero.html)
- [LightZero MuZeroCollector docs](https://opendilab.github.io/LightZero/api_doc/worker/index.html)
- [LightZero MuZeroCollector source](https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html)
- Local context: `curvyzero_env/core.py`, `curvyzero_env/config.py`,
  `docs/research/multiplayer_selfplay_muzero.md`,
  `docs/research/baseline_learnability.md`, and
  `docs/experiments/2026-05-08-env-smoke-benchmark.md`.
