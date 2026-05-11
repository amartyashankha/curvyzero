# Optimizer Setup Synthesis

Date: 2026-05-09

Status: first synthesis from local docs plus setup, reward, multiplayer, and
alternatives scouts.

## Short Read

Use the custom ego-wrapper path as the CurvyZero bridge. Do not make CurvyTron
look like stock Atari at the game-contract level.

LightZero should see a single-agent fixed-action MDP: one ego observation, one
ego action, scalar ego reward, `action_mask`, and `to_play=-1`. The CurvyZero
wrapper owns the trainer/replay boundary: opponent actions, wrapper action
maps, seeds, trace metadata, and project telemetry. Native CurvyTron remains
held player control state advanced over elapsed-ms server frames.

## Current Repo Setup

- Stock Atari Pong is a control lane: ALE/Gym Pong through LightZero's native
  Atari env/config path, visual frame stack, conv model, and six Atari actions.
  It tests whether pinned LightZero can run on Modal and gives a baseline for
  visual MuZero behavior. It does not test CurvyZero's custom all-player
  wrapper environment contract.
- Custom dummy Pong is the real bridge: `DummyPongLightZeroEnv` registers a
  DI-engine `BaseEnv`, wraps project-owned `PongEnv`, lets LightZero control
  one ego paddle, supplies the opponent action inside the env, and returns
  LightZero-shaped observations.
- Custom dummy Pong training uses stock LightZero trainer/search/replay entry
  points through patched config, plus CurvyZero diagnostics such as target
  replay sidecars and independent scorecards.
- CurvyTron itself is not yet directly registered as a LightZero training env.
  The current CurvyTron LightZero work is adapter/contract smoke evidence, not
  a full trainer path.

## Correct First Setup

- Keep `tabular_ego` as the custom dummy Pong learning/debug lane.
- Use explicit `pong_episode_max_steps`; do not let training step budget double
  as episode horizon.
- Keep true sparse win/loss reward as the training reward unless an experiment
  is explicitly marked objective-changing.
- Raise MCTS simulations above smoke settings before drawing learning-quality
  conclusions; `num_simulations=2` can create misleading root visit targets.
- Verify the compiled value/reward support scale rather than trusting patched
  config intent.
- Gate learning claims with independent scorecards, not only collector returns.
- Treat `raster_flat` as smoke-only. A visual bridge should be stacked,
  channel-first, and eventually close to the stock Atari conv shape.

## Multiplayer Shape

For CurvyTron, the first serious shape should be shared ego-perspective policy
rows. Every live player can emit one ego row, batched inference maps rows back
to a wrapper `joint_action`, and the trainer env advances one elapsed-ms source
transition for the chosen wrapper cadence.

Use policy-sampled or frozen-checkpoint opponents first. Add focal-agent MCTS
later: search only the ego action space and sample other players from explicit
policy adapters. Avoid joint-action MCTS as the default path because branching
is `3^N` before any game-theoretic value complications.

## LightZero Stance

LightZero remains useful as a serious replication/control lane and contained
bridge while CurvyZero keeps custom logic at the env/feature/opponent/telemetry
boundary and uses stock trainer/search/model flow where possible.

The setup becomes a trap if it forces CurvyTron into stock Atari semantics,
turn-based board-game semantics, per-step remote calls, checkpoint-opponent
MCTS inside the hot env step, or always-on row-level JSON diagnostics during
speed runs.

The broader recommendation is to stop treating LightZero as the only candidate
backbone or as already rejected. It has mostly answered "can we plumb a full
MuZero trainer?" but has not yet answered "is this the right optimizer
architecture for custom multiplayer CurvyTron?" Current evidence supports
keeping LightZero for stock Atari/CartPole controls, target audits, checkpoint
scorecards, and comparison runs while the optimizer lane prototypes a
transparent project runner.

## Alternatives Stance

- Prototype a project-owned PyTorch/CleanRL-style PPO runner as the next
  measurement and speed-debug evidence source. Coach owns learnability claims.
- Use a PettingZoo Parallel-compatible shape as the CurvyTron environment
  contract: wrapper action mapping in, per-agent observation/reward/done
  mapping out. Do not treat that API shape as native CurvyTron source behavior.
- Keep project-owned JAX/Mctx MuZero or Gumbel MuZero as a possible owned-search
  path after policy-gradient baseline behavior and timing are understood.
- Defer Sample Factory, RLlib, EnvPool-style backends, MiniZero, Muax,
  muzero-general, EfficientZero, and TurboZero as adoption candidates. They may
  provide references, but they do not beat a simple owned loop for the next
  debugging step.
- The optimizer framework hypotheses are tracked separately in
  [framework working hypotheses](framework_decision_2026-05-09.md): the leading
  repo-native bench candidate is an owned PPO/IPPO-style runner; LightZero
  remains a serious replication/control lane; Mctx is a later search-module
  hypothesis.

## Reward Stance

Keep environment reward and MuZero reward targets on true game payoff. Survival
ticks, loss delay, contact pressure, terminal cause, timeout rate, and action
entropy are telemetry, curriculum diagnostics, or tie-breakers inside a true
score equivalence band.

For finite competitive rounds, treat `discount_factor < 1` as an
objective-changing ablation unless explicitly justified. The default value
target should be true undiscounted return or centered rank payoff.

Worth testing first:

- canonical true sparse reward with `gamma=1`;
- scoreable curricula that change start/opponent/seed distributions, not reward;
- replay priority for rare wins/deaths/near-death/action-sensitive states;
- target-quality improvements such as more simulations, reanalysis, and root
  visit telemetry.

Avoid serious mainline runs with living reward, raw survival bonus,
contact/clearance reward, or pressure reward inside `env.step()`.

## Optimizer Implications

- Measure the whole actor loop before optimizing the simulator in isolation:
  env step, observation packing, policy/search, action unmap, replay write,
  reset/autoreset, actor idle, learner idle, and policy staleness.
- For speed runs, disable or sample target replay sidecars and high-volume
  trace/debug payloads.
- Keep Modal boundaries coarse: whole jobs and sweeps, not per-step/per-player
  calls.
- Do not recommend GPU env, C++/Rust, EnvPool-style backends, or distributed
  actors until the comparison-valid profile shows the CPU env is the limiting
  bucket.

## Source Anchors

- [Dummy Pong LightZero env](../../../src/curvyzero/training/lightzero_dummy_pong_env.py)
- [Dummy Pong config patcher](../../../src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py)
- [Dummy Pong Modal train smoke](../../../src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py)
- [Frozen checkpoint policy adapter](../../../src/curvyzero/training/lightzero_dummy_pong_policy.py)
- [CurvyTron trainer contract](../../../src/curvyzero/env/trainer_contract.py)
- [CurvyTron LightZero smoke adapter](../../../src/curvyzero/training/curvyzero_lightzero_smoke.py)
- [Training state index](../training_state_index_2026-05-09.md)
- [Multiplayer self-play MuZero notes](../../research/multiplayer_selfplay_muzero.md)
- [Training-loop bottlenecks and Amdahl's law](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md)
- [Reward shaping notes](../../research/reward_shaping_for_pong_curvy_muzero.md)
