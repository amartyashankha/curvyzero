# MuZero Repo Baseline Options

Research snapshot: 2026-05-09.

## Short Answer

LightZero-first is the current direction for the next real dummy Pong MuZero
attempt. Do not treat "write a project-owned Mctx trainer" as the next move.

The practical split is:

- LightZero-first means adapting a complete MuZero framework to dummy
  Pong/Curvy-style envs, then using its existing trainer.
- Project-owned fallback means using Mctx for search and writing the env
  adapter, replay, model, update loop, checkpoints, Modal job, and scorecard
  ourselves.

LightZero gets the first custom-env attempt because it is the only full MuZero
trainer already proven on Modal. Project-owned Mctx may fit CurvyTron semantics
and Modal artifacts better later, but it asks us to write many training pieces
before we have a project-owned success.

LightZero-first implementation spike:

1. Build a LightZero custom-env config/import smoke for dummy Pong tabular ego
   observations, `observation_shape=10`, and `A=3`.
2. Run a brutally capped custom-env MuZero train smoke on Modal.
3. Reject it only if the adapter hides replay determinism, action traces,
   scorecard metrics, or CurvyTron semantics.

If the fallback project-owned path is chosen later, cap it like this:

1. Make a quarantined spike branch or worktree.
2. Pin `mctx` and JAX, build the synthetic Mctx benchmark from
   `docs/research/mctx_integration.md`.
3. Add the thinnest project-owned MuZero loop: model, replay, target builder,
   self-play actor, trainer, evaluator.
4. Run ego-vs-random only, with `A=3`, fixed shapes, tiny vector observations,
   low simulation counts, and deterministic trace artifacts.

This is more work than copying a repo. It is only worth doing if the spike stays
small and proves an end-to-end checkpoint/eval loop quickly.

## Implementation Support Audit

Plain answer: LightZero is the only external library in this audit that can
run a complete, maintained MuZero trainer right now and that we have already
proven inside Modal, but the proven run is stock CartPole. LightZero also has
the clearest stock Atari/Pong path. Mctx has the best algorithm surface for our
own implementation, including standard MuZero, Gumbel MuZero, and Stochastic
MuZero search, but it is not a trainer.

Recommendation:

- Use LightZero first for the next dummy Pong MuZero trainer attempt.
- Keep project-owned JAX/Mctx as fallback/comparison if LightZero fights the
  custom env or hides required telemetry.
- Do not use muzero-general, Muax, or EfficientZero as the v0 backbone.

Support matrix:

| Library | Real trainer now | Atari/Pong | Gumbel MuZero | Stochastic MuZero | Custom visual envs | Robustness/randomization hooks | Modal risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightZero | Yes. It has stock trainers, and our Modal CartPole progression smoke already produced logs/checkpoints. | Best stock fit. README has a Pong command, Atari configs use `PongNoFrameskip-v4`, and OpenDILab has a Pong MuZero model card. Our Modal Pong env smoke is blocked at ROM/license setup, not missing algorithm code. | Yes. LightZero exposes Gumbel MuZero policy/model paths and benchmarks Gumbel MuZero on Pong/MsPacman/Gomoku/LunarLander. | Yes, exposed in policy/model/docs; strongest public example/benchmark evidence is 2048 chance modeling, not Pong. | Medium. DI-engine `BaseEnv` custom envs accept observation dicts, action masks, and image observations. Simultaneous multi-agent needs an ego wrapper. | Partial. Atari wrapper exposes frame skip, grayscale, scaling, frame warp, clipping, dynamic seed, and policy configs expose image augmentation. Sticky actions/action noise are best added as explicit Gym/BaseEnv wrappers; `PongNoFrameskip-v4` itself is the non-sticky Gym v4 flavor. | Lowest for stock trainer proof; medium for Curvy/Pong because PyTorch/DI-engine/Atari ROMs and wrapper semantics are heavy. |
| Mctx | No. It is search only: no env, replay, learner, checkpoints, or eval loop. | No built-in envs. We provide Pong/Curvy observations and training code. | Yes. Public API exports `gumbel_muzero_policy`; README recommends it. | Yes. Public API exports `stochastic_muzero_policy` with decision and chance recurrent functions. | Best control, because we own all observation tensors and wrappers. More implementation work. | Best control, but no ready hooks. Sticky actions, action noise, observation transforms, domain randomization, and logging must be project-owned wrappers/metadata. | Good fallback after our GPU search benchmark; not the immediate lane because it still needs a trainer. |
| muzero-general | Yes, educational trainer. | Atari support exists, but listed implemented Atari example is Breakout, not Pong. | No clear upstream Gumbel support. | No clear upstream Stochastic MuZero support. | Medium for simple Gym/custom game files; weaker for visual simultaneous control. | Whatever we write into the game adapter; no obvious first-class robustness profile layer. | Higher: Ray/multithreaded orchestration, older dependencies, unbatched MCTS, and educational-maintenance posture. |
| Muax | Partial. It wraps Mctx into gym-style MuZero training helpers and has CartPole-style examples. | No stock Atari/Pong-first evidence. | Not a strong fit: README centers `muzero_policy`; if Gumbel is needed, use Mctx directly. | No clear exposed trainer path for Stochastic MuZero. | Medium for Gym-style custom envs; visual/simultaneous work is still ours. | Gym wrappers can do it, but the project does not give us a CurvyZero-ready robustness layer. | Medium/high: JAX/Haiku/Optax plus older optional dependency notes; less proven in our Modal stack than direct Mctx. |
| EfficientZero repo | Yes, but it is EfficientZero, not the plainest MuZero baseline. | Strong Atari focus; examples use Breakout and env name is configurable. Pong may work, but it is not the cheapest proof. | No, not the point of this repo. | No, not the point of this repo. | Possible by registering a new config/model/wrapper, but heavy. | Atari wrappers/configs can be changed; robustness is not the reason to adopt it. | Highest: C++/Cython build, GCC, Ray, multi-worker assumptions, and heavier GPU/CPU topology. |

Answers to the specific questions:

- Which library can actually run real MuZero now? LightZero, muzero-general,
  and EfficientZero-family code can run complete trainers. LightZero is the
  only one that is both maintained enough for this audit and already validated
  by our Modal stock trainer smoke. Mctx cannot train by itself.
- Which supports Atari/Pong? LightZero is the clear answer. EfficientZero is
  Atari-capable but heavier and not Pong-proven in our notes. muzero-general
  has Atari support but its named implemented Atari game is Breakout. Mctx and
  Muax do not give us stock Pong training.
- Which exposes Stochastic MuZero or Gumbel MuZero? Mctx exposes both search
  APIs directly. LightZero exposes both at framework level; public benchmark
  evidence is stronger for Gumbel MuZero on Atari/Pong and Stochastic MuZero on
  2048. The other candidates should not be chosen for these variants.
- Which supports custom visual envs? LightZero supports custom image envs
  through DI-engine `BaseEnv`/Gym-style wrappers. Mctx supports any visual env
  we tensorize ourselves. muzero-general and Muax are adaptable but less
  attractive for simultaneous visual control.
- Which has hooks for sticky actions, action noise, observation transforms, or
  domain randomization? None gives us the exact CurvyZero robustness contract
  ready-made. LightZero has Atari preprocessing and image augmentation knobs;
  Gym/ALE provides sticky-action and frame-skip controls, but those must be
  wired deliberately and logged. With Mctx, all robustness hooks are ours, which
  is more work but safer for deterministic replay metadata.
- Which is least risky on Modal? For the next real trainer attempt: LightZero
  custom dummy Pong, because stock CartPole already proved the LightZero
  trainer path on Modal and dummy Pong avoids Atari ROM setup. Project-owned
  Mctx is fallback/comparison if LightZero cannot preserve metadata, artifacts,
  or scorecard telemetry.

## Options

| Option | Use for | Copy/adapt | Avoid |
| --- | --- | --- | --- |
| LightZero | Immediate dummy Pong MuZero trainer attempt. | Ego-agent wrapper, CartPole MLP config patch, trainer/checkpoint/log path, scorecard sidecar. | Forking collector/C++ tree code; joint-action search; hiding nondeterminism inside the wrapper; stock Atari Pong before ROM prep. |
| Project-owned Mctx | Fallback/comparison if LightZero fails or later blocks CurvyTron ownership. | Mctx `gumbel_muzero_policy`/`muzero_policy`, fixed-shape root/recurrent contract, benchmark style, action-weight targets. | Starting this before the LightZero custom-env smoke; writing custom MCTS first; assuming Mctx supplies replay, actors, trainer, or leagues. |
| muzero-general | Educational scaffolding/reference. | File decomposition, `MuZeroConfig` style, replay/target terminology, simple game adapter ideas. | Copying Ray/multithreaded orchestration, old dependency pins, unbatched MCTS limits, assumptions that only single/two-player turn games matter. |
| Muax | JAX/Mctx reference for gym-style MuZero. | Haiku/Optax model signatures, n-step tracer/replay ideas, minimal gym loop, package-level use of Mctx. | Depending on it as a maintained backbone; older release/dependency assumptions; single-agent Gym semantics as CurvyZero architecture. |
| a0-jax | JAX/Mctx AlphaZero example. | Self-play/evaluation loop shape, batched JAX training habits, board-game search-tree diagnostics. | Treating it as MuZero; copying board-game/alternating-player assumptions. |
| turbozero | Fast JAX AlphaZero reference. | Vectorized self-play ideas, subtree-persistence inspiration, hardware-aware batching patterns. | Using it as a MuZero baseline; all-JAX environment commitment; multi-GPU/vectorized complexity in v0. |
| Other cited MuZero repos | Last-resort reference reading. | Isolated implementation details when the main options are unclear. | GPL code paths, C++/Cython build surfaces, Atari-scale assumptions, multi-agent planners such as MAZero before CurvyZero has a tiny learned baseline. |

## What To Copy

- Interfaces, not architecture: `representation`, `dynamics`, `prediction`,
  root output, recurrent output, replay sample, target builder, evaluator.
- A small config object with pinned shapes: observation shape, action count,
  unroll length, support/value transform if used, discount, simulations,
  batch size, and seed.
- Replay fields needed for diagnosis: observation/action/reward/done,
  `action_weights`, raw value, search value, model step, rules hash,
  observation hash, and search config hash.
- A tiny evaluator matrix: random, sticky-random, and one-step-safe heuristic on
  held-out seeds.
- Benchmark reporting: compile time separate from steady-state time; decisions
  per second; simulations per second; memory; action histogram; finite checks.

## What To Avoid

- Copying a whole framework into `src/` before a spike earns it.
- Forking Mctx, LightZero, or any reference repo for v0.
- Adding Ray, DI-engine, C++ tree builds, leagues, PBT, or multi-GPU before a
  single-process run learns anything.
- Joint-action search as the minimal baseline. For 1v1 it changes action
  semantics; for n-player it explodes.
- Vector values, general-sum backups, checkpoint-pool self-play, bonuses, and
  stochastic MuZero until ego-vs-random is reproducible.
- Letting a copied adapter hide CurvyZero's seed/action trace, per-player
  terminal rewards, or deterministic replay metadata.

## Safe Throwaway Spike

Keep the copy disposable and visibly quarantined:

1. Create `spikes/muzero_mctx_minimal/` or a separate worktree/branch. Do not
   import it from core packages.
2. Vendor no third-party source initially. Depend on pinned packages; if code
   is copied later, preserve license headers and record source commit/URL.
3. Use one local adapter from CurvyZero observations to a single ego row. All
   opponent behavior is random or heuristic outside the search tree.
4. Keep one command, one config, and one artifact directory. Every run writes
   config, dependency versions, seed, rules hash, and evaluation summary.
5. Delete or promote deliberately: if the spike passes, move only the small
   project-owned interfaces into neutral packages; if it fails, keep the result
   as an artifact and remove the spike code.

## Minimal Baseline Path

The best first baseline is not "install a MuZero repo and adapt until it
works." It is:

1. Finish the non-MuZero learnability gate.
2. Run the synthetic Mctx benchmark.
3. Build a repo-owned Mctx MuZero toy loop in a quarantined spike.
4. Train ego-vs-random with `num_simulations <= 32`, short unrolls, tiny model,
   and fixed observation/action shapes.
5. Compare wall-clock results against the simple learned baseline, not just
   against random action selection.

If this path is too much code for the first spike, use muzero-general only as a
readable pseudocode companion and LightZero only as a separate PyTorch
throughput/comparison run. Do not merge either shape into the core tree first.

## Rejection Criteria

Reject or pause the MuZero-library baseline if any of these happen:

- The Mctx synthetic benchmark cannot produce stable finite action weights or
  acceptable compile/steady-state timing.
- The minimal loop cannot beat random on held-out seeds after the simpler PPO
  or policy baseline already can.
- Search wall time dominates while win rate does not improve over policy-only
  inference at equal wall-clock budget.
- Integration needs a fork of Mctx/LightZero or invasive changes to CurvyZero's
  simulator contracts.
- Deterministic replay, rules hashes, action traces, or per-player terminal
  details are lost.
- The implementation starts needing joint-action search, vector values, league
  machinery, or custom general-sum MCTS before ego-vs-random works.

## Sources

- Mctx GitHub/PyPI: https://github.com/google-deepmind/mctx and
  https://pypi.org/project/mctx/
- Mctx policy source, including `muzero_policy`, `gumbel_muzero_policy`, and
  `stochastic_muzero_policy`:
  https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py
- LightZero GitHub/docs: https://github.com/opendilab/LightZero and
  https://opendilab.github.io/LightZero/index.html
- LightZero Atari MuZero config:
  https://github.com/opendilab/LightZero/blob/main/zoo/atari/config/atari_muzero_segment_config.py
- LightZero Atari env wrapper:
  https://github.com/opendilab/LightZero/blob/main/zoo/atari/envs/atari_lightzero_env.py
- LightZero custom environment tutorial:
  https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
- OpenDILab Pong MuZero model card:
  https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero
- ALE/Gymnasium Atari docs for sticky actions/frame skip:
  https://ale.farama.org/environments/
- muzero-general: https://github.com/werner-duvaud/muzero-general
- Muax GitHub/PyPI: https://github.com/bwfbowen/muax and
  https://pypi.org/project/muax/
- EfficientZero: https://github.com/YeWR/EfficientZero
- a0-jax: https://github.com/NTT123/a0-jax
- turbozero: https://github.com/lowrollr/turbozero
- Local context: `docs/research/mctx_integration.md`,
  `docs/research/lightzero_integration.md`,
  `docs/research/baseline_learnability.md`, and
  `docs/research/multiplayer_selfplay_muzero.md`.
