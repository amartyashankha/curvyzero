# Simple Training Environment Options

Status: Proposed
Date: 2026-05-08

## Short Answer

Use a tiny project-owned line-duel environment as the next MuZero-shaped smoke test: two agents move simultaneously on a small discrete grid, leave lethal trails, receive terminal win/loss/draw rewards, and expose a PettingZoo `ParallelEnv` adapter plus an optional ego-vs-scripted Gymnasium adapter.

This is the simplest environment that validates the parts CurvyZero training actually stresses: all-player control collection, self-play data collection, perspective-swapped observations, terminal sparse rewards, tactical planning around trails, and adapter boundaries. TicTacToe, Connect Four, and matrix games are useful library smoke tests, but they miss continuous-survival pressure. Pong-like games add avoidable physics and dense-control quirks. A single-agent grid/survival task is even cheaper, but it does not exercise the multi-agent loop that is most likely to break.

## Comparison

| Option | Fit for MuZero-shaped loop | What it validates | Why not the default |
| --- | --- | --- | --- |
| Single-agent grid/survival | Good first single-agent smoke. | Gymnasium API, replay storage, value/reward/model unrolls, truncation handling. | Avoids self-play, opponent sampling, simultaneous actions, and ego-perspective value targets. |
| Two-player Pong-like | Medium. | Two-player control, reaction timing, possibly pixels/raster observations. | Ball/paddle physics are not CurvyTron-like; tends toward dense shaping and twitch control before trail tactics. |
| Line-duel / Tron-like | Best. | Simultaneous moves, trails, sparse terminal rewards, shared policy by perspective, joint rollout bookkeeping. | Requires writing a tiny environment, but the rules are small enough to own. |
| TicTacToe / Connect Four | Good algorithm sanity check. | Turn-based self-play, legal action masks, sparse terminal value learning, MCTS targets. | Turn-based and fully discrete; does not test simultaneous action collection or survival dynamics. |
| Rock-Paper-Scissors / matrix games | Useful interface test only. | Parallel step shape, policy entropy, exploitability/evaluation plumbing. | Too shallow for MuZero dynamics or planning; one-step games can pass while the real loop is broken. |
| PettingZoo built-ins | Good external conformance tests. | Known multi-agent API behavior, legal masks, parallel/AEC conversions, examples. | Existing games may drag in semantics that do not match CurvyTron; still need project-owned adapters. |
| Gymnasium / MiniGrid | Good single-agent baselines. | Standard single-agent tooling, vectorization, wrappers, seeding. | Multi-agent behavior has to be invented elsewhere. |
| OpenSpiel | Excellent game-theory reference. | Board games, matrix games, simultaneous games, evaluation concepts. | API and game formalism are heavier than needed for the first CurvyZero toy; use through an adapter only if needed. |

## Recommendation

Build `LineDuelTiny` before CurvyTron:

- `N x N` grid, start with `9x9` or `11x11`.
- Two players, simultaneous actions: turn-left, straight, turn-right, or absolute cardinal moves if implementation speed matters.
- Each player advances one cell per step and marks its trail.
- Death on wall, own trail, opponent trail, or contested head/head collision.
- Rewards: winner `+1`, loser `-1`, draw `0`; max-step timeout is `truncated`, not a normal terminal draw.
- Observation: ego-centered planes for walls/trails/heads, current heading, legal action mask if needed, and optional global debug state in `info`.
- Deterministic `reset(seed=...)`, fixed trace replay, and small rendered ASCII frames for failed cases.

This is small enough to solve with random, heuristics, and shallow search, but close enough to CurvyTron to catch the interface mistakes that matter: who acted at each tick, which observations become policy/value targets, how simultaneous deaths are scored, and how replay data names the ego player.

## Interface Best Practices

Keep the simulator core independent of external RL libraries. The project-owned core should expose typed reset/step functions, deterministic seeding, state serialization or replay fingerprints, and stable observation/reward schema versions. Adapters should be thin and tested against golden traces.

Use Gymnasium for single-agent wrappers. Gymnasium `Env.step()` returns `(observation, reward, terminated, truncated, info)`, and `reset()` returns `(observation, info)`. Preserve the distinction between terminal game outcomes and time-limit truncation because bootstrapping logic depends on it.

Use PettingZoo `ParallelEnv` for all-live-agent wrapper decisions. Parallel step accepts an action dictionary keyed by live agent id and returns observation, reward, termination, truncation, and info dictionaries keyed by agent. This matches CurvyZero's wrapper action-map needs better than AEC turn iteration.

Use an ego Gymnasium adapter only for baselines: one controlled player, scripted or sampled-policy opponents, rotated seats across episodes. Do not let this become the canonical simulator contract.

Use OpenSpiel as an optional reference/adaptation target, not the first owned interface. It is strong for game-theoretic environments, matrix games, turn-taking games, and simultaneous-move research, but its native concepts are broader than this smoke test needs.

## Why Not Start With Built-In Games

PettingZoo `rps_v2.parallel_env()` is the fastest possible parallel API smoke, but it cannot validate planning depth. PettingZoo TicTacToe and Connect Four provide legal action masks and sparse terminal rewards, but they are turn-based even when exposed through a parallel-compatible interface. MiniGrid gives clean single-agent grid tasks and curriculum knobs, but it bypasses multi-agent self-play. OpenSpiel is valuable for known game implementations and evaluation tools, but using it first risks adapting CurvyZero to OpenSpiel rather than proving a CurvyTron-shaped training loop.

The next toy should therefore be owned, tiny, and Tron-shaped. Use external libraries to validate adapters, not to define the problem.

## Sources

- Gymnasium Env API: `step`, `reset`, single-agent contract, and the `terminated` / `truncated` split: https://gymnasium.farama.org/api/env/
- Farama explanation of the terminated/truncated step API and bootstrapping motivation: https://farama.org/Gymnasium-Terminated-Truncated-Step-API
- PettingZoo Parallel API for simultaneous multi-agent environments and dictionary-keyed returns: https://pettingzoo.farama.org/api/parallel/
- PettingZoo overview: AEC for turn-based/sequential interaction and Parallel API for simultaneous actions: https://pettingzoo.farama.org/
- PettingZoo Rock-Paper-Scissors, including `parallel_env`, simultaneous choices, discrete observations/actions, and terminal rewards: https://pettingzoo.farama.org/environments/classic/rps/
- PettingZoo TicTacToe and Connect Four environment docs, including action masks and sparse terminal rewards: https://pettingzoo.farama.org/environments/classic/tictactoe/ and https://pettingzoo.farama.org/environments/classic/connect_four/
- MiniGrid environment docs for tunable single-agent gridworld curricula: https://minigrid.farama.org/environments/minigrid/index.html
- OpenSpiel introduction: single/multi-player, matrix games, sequential and simultaneous move games, zero/general-sum support: https://openspiel.readthedocs.io/en/latest/intro.html
- MuZero Nature paper: planning with a learned reward/policy/value model across Atari and board games: https://www.nature.com/articles/s41586-020-03051-4
