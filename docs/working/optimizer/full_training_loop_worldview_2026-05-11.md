# Full Training Loop Worldview

Date: 2026-05-11

Status: optimizer working map. This is the component checklist for CurvyTron
visual 2P MuZero-style training. It should evolve as profiling and framework
spikes produce harder evidence.

Related replication-control backlog:
`docs/working/optimizer/framework_replication_controls_2026-05-11.md`.

2026-05-15 correction: this file is a historical architecture map. Current
trusted guidance is stock LightZero `--mode train` with
`env_variant=source_state_fixed_opponent`, the frozen-opponent route, and CPU
`cpu_oracle` `browser_lines + simple_symbols` policy observations. The custom
`--mode two-seat-selfplay` path should not be treated as the trusted Coach
learning baseline.

## One-Line Goal

Current optimizer baseline:
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`
on `source_state_fixed_opponent`, source-state `[4,64,64]`, and fixed/frozen
opponents. Future goal: if the stock loop is too slow, test coarse synchronous
fanout of searched trajectory collection before designing any continuous
actor/replay service.

Plain English: we need actors to play games, search to choose good actions,
replay to store those games, a learner to update the model, checkpoints to move
new weights back to actors, and evals to tell whether anything improved.

## Future Loop

This is not the current stock `train_muzero` loop. It is the later decoupled
shape if coarse fanout proves useful.

```text
checkpoint N
  -> actors collect searched visual chunks from one frozen checkpoint
  -> replay stores physical-tick trajectory chunks
  -> learner samples unrolls and updates model
  -> checkpoint publisher writes checkpoint N+1
  -> evaluator scores checkpoints
  -> actors refresh policy on cadence
```

## Components

### Environment

Owns source-faithful CurvyTron state transition:

- reset/seed/rules metadata
- simultaneous `joint_action[B,P]`
- terminal/final observation before autoreset
- per-player rewards and alive/done masks
- visual tensor generation

Likely placement: CPU NumPy first, fixed-shape arrays. GPU env is a later
option only if env/render becomes the limiting measured bottleneck after search
batching.

### Observation / Visual Stack

Owns converting source state into network input:

- `uint8` source/debug frame or source-faithful visual frame
- frame stack, likely `[4,64,64]` or a later higher-fidelity shape
- egocentric/player-perspective transforms if used
- masks and metadata

Open issue: separate source-fidelity visual truth from optimizer debug visual.

### Collector / Actor

Owns turning env states into searched decisions and replay records:

- build live player rows from `[B,P]`
- run policy/search for each live row or for joint actions
- map selected actions back to `joint_action[B,P]`
- step env once per physical tick
- emit trajectory chunks with policy/search metadata

Current likely v1: custom CurvyTron actor around LightZero
`MuZeroPolicy.collect_mode.forward`, one row per live seat, then native-compatible
per-seat trajectory records.

Near-term scale v1 is simpler than a continuous actor service: freeze checkpoint
`K`, launch `N={1,2,4,8}` collect-only actors, have each write searched chunks
with checkpoint id/schema/seed/search settings, merge/import chunks, then run
the learner to publish `K+1`. All actors in that batch use the same checkpoint.
The caveats are wasted old/correlated data and merge/learner bottlenecks, not
hidden off-policy drift inside the synchronous batch.

### Search / Inference

Owns MCTS/Gumbel MuZero policy improvement:

- root priors, root values, legal masks
- visit-count policy targets
- root-value diagnostics
- batched neural inference
- search configuration and model/checkpoint id

Candidate implementations:

- LightZero C++ tree search via `collect_mode.forward`: shortest path now.
- MCTX/JAX: strongest GPU-batched search candidate, but requires repo-owned
  model/replay/learner.
- MiniZero: strongest full-system reference, but not a drop-in for faithful
  simultaneous CurvyTron.

### Replay

Owns storing and sampling training data:

- physical tick id, env row id, player id
- observation stack before action
- scalar ego action
- optional joint action for audit
- action mask
- MCTS visit distribution for ego action space
- root value
- scalar ego reward
- terminal/truncation/final-observation flags
- policy/checkpoint/search version
- rules/render schema id

Two viable shapes:

- LightZero-native-compatible: one `GameSegment` per seat perspective.
- Repo-native chunks: immutable arrays, then convert to learner batches.

Do not store one opaque joint trajectory unless learner semantics are explicit.

Concrete LightZero-native spike:

- keep the custom simultaneous collector for safe `[B,P]` action collection;
- group rows into one seat-perspective trajectory per `(episode_id, env_row_id,
  player_id)`;
- convert each group into upstream `GameSegment` with scalar ego action, scalar
  ego reward, `action_mask[3]`, `child_visit[3]`, root value, and `to_play=-1`;
- push those segments into `MuZeroGameBuffer`;
- call native `MuZeroGameBuffer.sample(...)` and `MuZeroPolicy.learn_mode.forward`;
- validate target reward/policy/value semantics before trusting learning.

This is the cleanest current way to reuse LightZero's replay/target/learner
machinery while keeping CurvyTron's simultaneous physical step.

### Learner / Trainer

Owns model update:

- sample replay sequences
- unroll dynamics
- policy loss against visit distribution
- value loss against return/bootstrap target
- reward loss
- optimizer state
- AMP / batch sizing / gradient accumulation
- checkpoint payload

Current short path: reuse LightZero `MuZeroPolicy.learn_mode.forward` if replay
can be made native-compatible. Longer path: repo-owned PyTorch or JAX learner.

### Checkpoint Publisher

Owns weight distribution and artifact semantics:

- write checkpoint payload first
- write small pointer manifest after payload is complete
- include model id, optimizer step, replay version, search config, env/render
  schema ids
- avoid frequent heavy writes in the hot loop

Actors should refresh by checkpoint id/cadence, not by per-step sync.

### Evaluator

Owns learning claims, not optimizer:

- stock/eval policy search settings
- fixed seeds and opponent pools
- survival/time/score curves
- action histograms and collapse checks
- checkpoint comparisons

Optimizer only owns eval speed and artifact plumbing.

### Observability

Every run should report:

- env steps/sec and physical ticks/sec
- games/sec and completed episodes
- MCTS simulations/sec
- neural inference calls/sec
- average inference batch size
- GPU utilization and CPU utilization where available
- replay rows/chunks/sec
- learner updates/sec
- checkpoint write/read time
- actor policy-version lag
- replay age distribution
- time split: env/render/reset/search/model/replay/learner/eval/artifacts

## Framework Examples To Replicate

The next framework decision should be grounded by controls we can reproduce:

- LightZero Atari Pong MuZero:
  `zoo/atari/config/atari_muzero_segment_config.py` or related Atari configs.
  This is the closest known "visual observation + delayed reward + MuZero
  search" control.
- LightZero TicTacToe/Connect4 MuZero:
  `zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py` and
  Connect4 configs. This probes delayed-reward PvP/search and LightZero
  `to_play`/GameBuffer semantics.
- MiniZero Atari MuZero/Gumbel MuZero:
  e.g. `tools/quick-run.sh train atari mz ... -conf_str env_atari_name=pong` or
  the documented MsPacman command. This checks another full-system visual MuZero
  stack.
- MiniZero Go/Othello/board games:
  proves server/self-play-worker/optimizer/storage loop on a mature
  self-play domain.
- OpenSpiel or AlphaZero.jl Connect Four:
  clean canonical AlphaZero PvP controls outside LightZero.
- PettingZoo Atari Pong v3:
  best runnable visual 2-player parallel-action environment control, but not a
  MuZero/search trainer.
- MCTX synthetic benchmark already exists locally; next is a visual-root MCTX
  profile over `[B,P,4,64,64]` roots.
- RLlib multi-agent PPO baseline: optional sanity check for CurvyTron env/reward,
  not a MuZero replacement.

No maintained command-ready "Tron + visual + MuZero/AlphaZero + simultaneous
self-play" framework example has been found yet. The runnable controls split
the target into pieces: visual MuZero Atari, turn-based PvP search, and visual
parallel-action Pong/MARL.

## Current Working Bets

- Simultaneous action is not the primary speed blocker. The primary speed blocker
  is searched experience throughput and batched inference/search.
- Sequential commit is a usable interface trick if pending actions are hidden
  correctly and replay is physical-tick based. It is not automatically wrong,
  but it must not leak hidden pending actions into search.
- LightZero remains the shortest path for near-term CurvyTron MuZero smokes.
- MiniZero is the strongest full-system reference, not yet a clear replacement.
- MCTX is the strongest fast-search primitive, not a full training framework.
- The most practical speed architecture starts one-container and batched, then
  scales actors/search workers only after the local batching limit is measured.

Practical next move: do not rewrite everything yet. First see whether the
custom CurvyTron collector can feed native LightZero `GameSegment` /
`MuZeroGameBuffer` replay. If that works, we keep LightZero's learner and target
builder while owning only the CurvyTron actor/replay boundary.

## Immediate Research Spikes

1. LightZero native-compatible replay: can custom CurvyTron actor chunks become
   one `GameSegment` per seat perspective and reuse native sampling/learner?
2. MCTX visual-root profile: can a tiny visual model plus Gumbel MuZero search
   run `B*P` roots on Modal GPU with honest timings?
3. MiniZero controls: can we reproduce MiniZero Atari or a small board-game run
   before considering a CurvyTron port?
4. Sequential commit audit: define exactly what observation/search state sees
   during pending-action phases and whether this can be made framework-safe.
