# Framework Replication Controls

Date: 2026-05-11

Status: optimizer control backlog. These are examples to replicate before
trusting a framework or migration path.

## Why This Exists

We should not choose a MuZero/AlphaZero framework only from architecture notes
or repo descriptions.
We need controls that already work and are close to CurvyTron along one or more
axes:

- visual observations
- delayed/survival reward
- searched MuZero/AlphaZero training
- 2-player/PvP or self-play
- simultaneous/parallel actions
- distributed actor/replay/learner shape

No maintained command-ready "Tron + visual + MuZero/AlphaZero + simultaneous
self-play" example has been found yet. The best controls split the target into
pieces.

## Highest-Signal Controls

### LightZero Atari Pong MuZero

Purpose: visual observation + delayed reward + MuZero search in the current
framework family.

Evidence:

- HuggingFace model card:
  https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero
- The card reports a LightZero/DI-engine MuZero setup for
  `PongNoFrameskip-v4`, `obs_shape=[4,96,96]`, `collector_env_num=8`,
  `evaluator_env_num=3`, CUDA on, `mcts_ctree=True`, and self-reported eval
  `mean_reward=20.4 +/- 0.49`.
- It includes a simple training snippet:

```python
from lzero.agent import MuZeroAgent

agent = MuZeroAgent(env_id="PongNoFrameskip-v4", exp_name="PongNoFrameskip-v4-MuZero")
return_ = agent.train(step=int(500000))
```

Useful repo config family:

- `opendilab/LightZero`
- `zoo/atari/config/atari_muzero_segment_config.py`
- related configs:
  `atari_gumbel_muzero_config.py`,
  `atari_efficientzero_config.py`,
  `atari_sampled_efficientzero_config.py`

What it proves if replicated: our LightZero install/container can reproduce a
known visual MuZero setup with real search and delayed Atari reward.

What it does not prove: 2-player simultaneous self-play.

### LightZero Board-Game MuZero

Purpose: same framework + PvP/search semantics + delayed terminal reward.

Candidates:

- TicTacToe MuZero:
  `zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py`
- Connect4 MuZero:
  `zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py`

Local Modal smoke surfaces exist for TicTacToe, and possibly Connect4 wrappers
around the same idea.

What it proves if replicated: LightZero `to_play`, board-game search, and
GameBuffer target behavior can work in a two-player setting.

What it does not prove: visual input or simultaneous actions.

### MiniZero Atari MuZero / Gumbel MuZero

Purpose: independent full-system visual MuZero stack with self-play workers,
optimizer, storage, and batched GPU inference.

Repo:

- https://github.com/rlglab/minizero

Documented command shape:

```bash
tools/quick-run.sh train atari mz 300 \
  -n ms_pacman_mz_n50 \
  -conf_str env_atari_name=ms_pacman:actor_num_simulation=50
```

Likely Pong variant: set `env_atari_name=pong`, but verify before relying on
it.

What it proves if replicated: MiniZero's visual MuZero system and worker/server
pipeline work locally/remotely.

What it does not prove: 2-player simultaneous CurvyTron semantics.

### MiniZero Board Games

Purpose: full-system self-play loop on a mature PvP domain.

Example command shape:

```bash
tools/quick-run.sh train go az 300 \
  -n go_9x9_az_n200 \
  -conf_str env_board_size=9:actor_num_simulation=200
```

What it proves: server, self-play workers, optimizer worker, storage, and
checkpoint loop.

What it does not prove: visual input or simultaneous actions.

### PettingZoo Atari Pong

Purpose: visual, true two-player, parallel-action API.

Competitive Atari Pong:

- https://pettingzoo.farama.org/environments/atari/pong/
- `from pettingzoo.atari import pong_v3`
- `pong_v3.parallel_env()`
- agents: `first_0`, `second_0`
- observation shape: `(210,160,3)`
- action range: `[0,5]`
- scoring: +1 to scorer, -1 to opponent

Cooperative Pong:

- https://pettingzoo.farama.org/environments/butterfly/cooperative_pong/
- `from pettingzoo.butterfly import cooperative_pong_v5`
- two agents, visual observation `(280,480,3)`, discrete actions
- objective is to keep the ball in play as long as possible

What it proves: a clean parallel multi-agent visual API for Pong-like dynamics
and survival/competitive reward shapes.

What it does not prove: MuZero search or AlphaZero-style training.

### OpenSpiel AlphaZero Connect Four

Purpose: canonical AlphaZero PvP/self-play control outside LightZero.

Docs:

- https://openspiel.readthedocs.io/en/latest/alpha_zero.html

Command shape:

```bash
python3 open_spiel/python/examples/alpha_zero.py \
  --game connect_four \
  --nn_model mlp \
  --actors 10
```

What it proves: actor/learner/replay/evaluator structure for turn-based
AlphaZero-style training.

What it does not prove: visual input, high production scale, or simultaneous
CurvyTron.

### MCTX Synthetic And Visual-Root Probe

Purpose: high-throughput batched GPU search primitive.

Existing repo surfaces:

- `src/curvyzero/infra/modal/mctx_dependency_smoke.py`
- `src/curvyzero/infra/modal/mctx_gpu_dependency_smoke.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `docs/experiments/2026-05-09-modal-mctx-synthetic-benchmark.md`

Current local evidence: synthetic Modal L4 benchmark recorded about `12k`
decisions/sec and `193k` simulations/sec for one synthetic shape.

Next control: visual-root profile over `[B,P,4,64,64]` roots with a tiny CNN
and `mctx.gumbel_muzero_policy`.

What it proves: whether JAX/MCTX can be the fast search core.

What it does not prove: replay/learner/checkpoint system.

## Recommended Replication Order

Plain English: replicate one known example for each thing we care about before
we trust any framework. Do not ask one example to prove everything.

1. LightZero Atari Pong MuZero:
   establish a known visual MuZero control in the current package family.
2. LightZero TicTacToe or Connect4 MuZero:
   establish PvP/search/GameBuffer semantics in the current package family.
3. PettingZoo Atari Pong or Cooperative Pong:
   establish visual parallel-action environment/reward behavior as a baseline.
4. MCTX visual-root benchmark:
   establish whether a fast batched search core is realistically available.
5. MiniZero Atari or board-game quick run:
   establish whether MiniZero is worth deeper migration work.

## Current Read

The examples support a split strategy:

- use LightZero controls to avoid losing the working MuZero machinery;
- use PettingZoo Pong controls to sanity-check visual 2-player parallel-action
  environment assumptions;
- use MCTX to test the fast-search future;
- use MiniZero to learn from a real actor/server/optimizer system, but do not
  assume it will accept CurvyTron cleanly.
