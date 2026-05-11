# Training Framework Alternatives - 2026-05-09

Owner: repository/framework alternatives researcher.

Scope: answer whether CurvyZero should move to another repo/framework for
CurvyTron training. This memo is docs-only. No code or tests were run.

## Short Answer

Do not move CurvyZero wholesale to another training repo or framework now.

The practical recommendation is a layered boundary:

1. Keep the CurvyTron simulator core repo-owned, source-faithful, and array-shaped
   around simultaneous `B x P` state.
2. Promote a PettingZoo-style `ParallelEnv` contract as the main public
   multi-agent adapter shape, because it matches simultaneous action dictionaries
   without forcing a training library to own the simulator.
3. Build the first transparent measurement baseline as repo-native PPO in a
   CleanRL-style single-file/readable loop, adapted to our fixed-shape batched
   arrays and project-owned artifacts.
4. Keep LightZero as a contained MuZero-family experiment lane, but make its
   replication/control obligation explicit: before treating alternatives as the
   answer, reproduce a known LightZero Pong-like control closely enough that we
   know whether our failures are framework misuse, under-scale, or custom-env
   semantics. LightZero is not the CurvyTron backbone yet, but it is still the
   MuZero control we owe ourselves.
5. Keep JAX/Mctx as the likely project-owned MuZero search substrate if and when
   MuZero is worth owning. Do not call Mctx a framework migration: it is search
   only, so replay, actors, trainer, eval, checkpointing, and Modal run
   management stay ours.

In one sentence: move the external API toward PettingZoo-style parallel
multi-agent semantics and move baseline learning toward a repo-native
PPO/CleanRL-style loop; keep a concrete LightZero Pong replication lane alive;
do not adopt LightZero, RLlib, Sample Factory, SB3, OpenSpiel, MiniZero, or
muzero-general as the primary CurvyTron framework.

Alternatives clarify tool boundaries. They do not justify skipping LightZero.
LightZero remains the MuZero replication/control obligation; repo-native PPO is
the parallel CurvyTron architecture lane; Mctx is search machinery, not a
trainer.

## Why This Answer Fits CurvyTron `B x P`

CurvyTron is not a normal single-agent Gym task and not a turn-based board game.
The source is held controls over elapsed-ms frames; the trainer hard shape is an
all-live-player wrapper decision:

```text
B = environments in a batch
P = players per environment
actions[B, P]
observations/rewards/done/info with stable per-row, per-player semantics
```

The local vector-state notes already point in this direction: fixed-shape
structure-of-arrays state, reverse source player order, explicit lifecycle and
reset/autoreset metadata, and source-faithful trace parity before speed claims.
That argues against letting any external framework become the semantic owner.

The framework should consume this shape, not define it.

## Local Evidence

Local docs changed the recommendation from "try a MuZero repo" to "own the
training boundary":

- `docs/working/environment/vector_state_schema.md`: the environment target is a
  fixed-shape `B, P, K, ...` array state with source-specific tick ordering and
  explicit event arrays. That shape is a poor fit for framework-owned Python
  env workers as the core authority.
- `docs/design/training_architecture.md`: the stable boundary is that libraries
  adapt to `curvyzero.env`; neither Mctx nor LightZero defines the simulator.
  It also calls for fixed-baseline evals before searched self-play.
- `docs/working/training_state_index_2026-05-09.md`: LightZero can run trainer
  mechanics and checkpoint mirroring on Modal, but dummy Pong still lacks
  reliable held-out improvement; the active blocker is root visit/target
  quality plus action collapse, not more same-config scale.
- `docs/working/training_lessons_for_curvytron_2026-05-09.md`: CurvyTron should
  inherit lane discipline, target sidecars, survival telemetry, support-scale
  checks, independent eval, and Modal artifact patterns; it should not inherit
  Pong result claims.
- `docs/research/simple_training_environment_options.md`: a PettingZoo
  `ParallelEnv` plus optional ego Gymnasium adapter is the right interface split
  for simultaneous games, while the simulator core remains project-owned.
- `docs/research/mctx_integration.md`: Mctx is excellent for batched JAX search
  and fixed `[B, A]` root/recurrent contracts, but it is not a trainer.
- `docs/research/muzero_repo_baseline_options.md`: LightZero is the only full
  MuZero trainer already proven in our Modal path, while Mctx is fallback/search
  infrastructure and other MuZero repos should stay references.

## LightZero Replication Obligation

Framework alternatives must not become an excuse to avoid reproducing
LightZero. The LightZero lane should answer one narrow control question before
we use custom dummy Pong failure as evidence against it:

```text
Can our environment, dependency, artifact, and eval stack reproduce a known
LightZero Pong-like MuZero setup without CurvyZero custom-env semantics?
```

The minimal known setup should be stock LightZero Atari Pong, not custom dummy
Pong:

- env: upstream/stock Atari `PongNoFrameskip-v4` or the current Gym/ALE
  equivalent used by the pinned LightZero config;
- config source: LightZero's stock Atari MuZero config path, with changes limited
  to runtime caps, seed, output paths, and artifact mirroring;
- observation/model: stock visual Pong path, stacked grayscale frames and the
  upstream convolutional MuZero model, not `tabular_ego` or `raster_flat`;
- action space: ALE Pong's native discrete action space, not the project
  three-action dummy Pong space;
- trainer: LightZero's own MuZero trainer entrypoint, not a project trainer and
  not a dry config smoke;
- evaluator: prefer LightZero's stock evaluator once the local `action_mask`
  collation/API issue is fixed; until then, manual eval is acceptable only if it
  is strict checkpoint-load, no-fallback, and records raw observations, policy
  observations, actions, rewards, dones/truncations, and root visit/logit
  summaries;
- artifacts: mirror config, dependency versions, train logs, TensorBoard/log
  files, checkpoints, eval JSON, and action/reward histograms into
  `curvyzero-runs`.

This control does not have to solve Atari Pong from scratch in one heroic run.
It does need to be recognizably LightZero's known Pong setup rather than a
CurvyZero toy wearing a LightZero trainer. A practical minimal ladder:

1. **Strict pretrained/control eval** if an official OpenDILab Pong MuZero
   checkpoint is usable in our pinned image: load it with no fallback and verify
   non-random Pong behavior through our eval/artifact path.
2. **Stock trainer reproduction rung** from scratch using the upstream Atari
   Pong config, with materially closer settings than the smoke runs: stock frame
   stack/conv path, stock reward/action semantics, `num_simulations=50` unless
   a smaller rung is explicitly labeled mechanical-only, nontrivial batch/replay
   settings, and enough env steps to produce multiple meaningful checkpoints.
3. **Checkpoint curve** over `iteration_0`, at least one middle checkpoint,
   final checkpoint, and `ckpt_best`, all evaluated with the same strict path.

The local 4096/sim10 official Atari run is not enough to discharge this
obligation. It proved infrastructure and exposed action collapse, but it was
still far off the upstream recipe: tiny env-step budget, fewer simulations,
small batch/update scale, and eval caps. It is evidence that the wrapper can
run, not evidence that LightZero's known Pong recipe fails.

Moving away from LightZero is justified only by a concrete blocker, for example:

- the stock Pong control cannot be made to load/train/evaluate under pinned
  dependencies without invasive LightZero/DI-engine changes;
- the stock Pong control works, but the custom-env bridge necessarily loses
  seed, action trace, terminal reward, truncation/final-observation, checkpoint,
  root-target, or support-scale evidence;
- matching LightZero's known Pong settings requires runtime or infrastructure
  assumptions incompatible with our Modal budget after one bounded reproduction
  attempt is documented;
- exact upstream-like Pong control reproduces, but CurvyTron/dummy Pong still
  requires simultaneous `[B,P]` semantics that LightZero can only express by
  hiding opponents inside an ego wrapper and corrupting the training/eval
  contract.

Not valid blockers: a custom dummy Pong run fails under tiny simulations; a
manual eval path sees action collapse before the stock evaluator is fixed; a
smoke-scale official Pong run underfits; or PPO looks easier. Those are reasons
to keep the LightZero control lane honest, not reasons to skip it.

## Candidate Comparison

| Candidate | Best use | Fit for simultaneous `B x P` CurvyTron | Main risk | Recommendation |
| --- | --- | --- | --- | --- |
| PettingZoo-style `ParallelEnv` | Public multi-agent env API and compatibility adapter. | Strong API fit: all live agents act together and reset/step return per-agent dictionaries. | It is an API, not a trainer or vector backend; dict wrappers can be too slow if used as the hot loop. | Adopt as external contract; keep the hot core array-native. |
| Repo-native PPO / CleanRL-style | First serious learnability and throughput baseline. | Strong if we write it against `obs[B,P]`, `action[B,P]`, masks, returns, and project-owned eval artifacts. | We must own rollout storage, GAE/returns, self-play pools, and metrics. | Preferred next backbone for CurvyTron baseline learning. |
| LightZero | Required Pong-like MuZero control plus contained custom-env MuZero lane. | Medium/weak for CurvyTron: custom env works through an ego wrapper, but simultaneous opponents are hidden behind `step`. | Under-replicated stock control, target-quality/action-collapse issues, support-scale mismatch risk, DI-engine assumptions, and opaque trainer internals. | Reproduce stock Pong control before moving away; keep contained unless it earns backbone status. |
| Mctx/JAX owned MuZero | Future project-owned MuZero search over learned dynamics. | Strong for fixed batched search rows, `A=3`, JIT, and accelerator use. | It supplies MCTS policy calls only, not training. High implementation bill. | Keep as MuZero substrate after PPO gates or if LightZero provides a concrete blocker. |
| Sample Factory | High-throughput PPO/APPO, multi-agent training, self-play/PBT. | Medium: has multi-agent concepts and custom env support, but expects its env lifecycle and autoreset shape. | Adopting its rollout architecture may fight our terminal traces, final observations, and Modal artifact discipline. | Reference for throughput and multi-policy ideas; not v0 backbone. |
| RLlib | Multi-agent PPO ecosystem, PettingZoo/OpenSpiel wrappers, policy mapping. | Medium API fit but heavyweight. | Ray stack, changing APIs, distributed runtime overhead, and framework-owned sampling/eval complexity. | Reference or later scale candidate only. |
| SB3 | Stable single-agent PPO baseline and wrappers. | Weak for native simultaneous multi-agent; vector envs stack independent envs, not players in one shared game. | Would require ego wrappers and lose the natural `P` axis. | Use as a single-agent reference, not CurvyTron backbone. |
| OpenSpiel | Game-theory environment/algorithm reference. | Conceptual fit for simultaneous, n-player, general-sum games. | Heavy C++/pybind game formalism; adapting CurvyTron to it would distract from source fidelity. | Reference for evaluation/game theory, not implementation path. |
| MiniZero | Complete AlphaZero/MuZero/Gumbel framework with Atari and board games. | Weak for CurvyTron v0; architecture expects game-specific workers/server/storage. | Docker/Linux/GPU-oriented C++ stack and distributed self-play machinery. | Reference for MuZero architecture and results only. |
| muzero-general | Educational MuZero code. | Weak/medium for simple games, weak for simultaneous vector CurvyTron. | Older educational posture, Ray/multithreaded assumptions, unbatched MCTS concerns. | Pseudocode/reference only. |

## Detailed Reads

### LightZero

LightZero is the strongest external MuZero trainer candidate. Its official repo
describes a PyTorch MCTS+RL toolkit with MuZero, Sampled MuZero, Stochastic
MuZero, EfficientZero, and Gumbel MuZero, and its supported table includes
Atari, CartPole, LunarLander, MiniGrid, and board games. Its custom-env docs say
custom environments are based on DI-engine `BaseEnv`, and LightZero expects
observations wrapped as a dict with `observation`, `action_mask`, and `to_play`.
For non-board games, `to_play=-1` and a discrete action mask of ones is the
normal wrapper pattern.

That aligns with our existing dummy Pong adapter plan, but the local lane has
not yet paid the full control debt. Before alternatives become the main answer,
we need a stock Pong replication rung that is recognizably upstream LightZero:
Atari Pong, stock visual observations, native ALE action space, upstream MuZero
config, and strict checkpoint eval/artifacts. Only after that control is clean
should custom dummy Pong failures be interpreted as custom-env or simultaneous
semantics evidence.

LightZero still is not a natural full CurvyTron substrate. The first custom
LightZero formulation is ego-vs-scripted/frozen opponent; it does not preserve
simultaneous multi-agent self-play as a first-class `[B,P]` action tensor. Local
dummy Pong evidence also says we should not "scale the same failure": root
targets and support-scale semantics have to be inspected before quality claims.

Verdict: required Pong-like MuZero control plus useful contained custom-env
lane; not migration target unless it earns backbone status after that control.

### Repo-Native PPO / CleanRL-Style

CleanRL is valuable because its philosophy is readable single-file online RL
implementations, not a framework to import. That is exactly the right reference
posture for CurvyZero: take the PPO mechanics, but keep rollout collection,
seeding, target/eval sidecars, checkpoint manifests, and `[B,P]` state native.

This route does not give MuZero planning, but it does answer the immediate
question the codebase keeps circling: can a simple policy learn anything stable
from the source-like simultaneous simulator under independent scorecards?

The first serious CurvyTron baseline should be:

- shared policy over ego observations for every live player;
- rollout tensors shaped by `B`, `P`, and time, not by opaque framework workers;
- exact terminal/truncation/final-observation handling;
- checkpoint-pool opponents later, fixed baselines first;
- action histograms, terminal causes, survival stats, and held-out eval curves.

Verdict: preferred v0 training backbone.

### PettingZoo-Style Environment API

PettingZoo's Parallel API is explicitly for environments where all agents have
simultaneous actions and observations. Its `parallel_env.reset(seed=...)`
returns observations and infos; each step receives an action dict for live
agents and returns observation, reward, termination, truncation, and info
dictionaries.

That is the right public adapter shape for CurvyTron. It keeps the simultaneous
game honest for RLlib, Sample Factory, Tianshou, CleanRL tutorials, and other
tooling. It should not be the hot internal representation, because CurvyTron
needs fixed arrays and source-parity event rows.

Verdict: adopt as compatibility API, not as internal backend.

### Mctx / JAX Owned MuZero

Mctx is a strong technical match for eventual project-owned MuZero search:
JAX-native MCTS, JIT support, and batched search inputs. The API exposes root
prior/value/embedding and recurrent reward/discount/prior/value. Its
documentation recommends Gumbel MuZero and describes policy outputs with
actions and action weights that can be used as policy targets.

This fits fixed `B` decision batches and an `A=3` CurvyTron action set very
well. But it does not include the rest of training. Choosing Mctx means writing
the model, replay, target builder, actor loop, evaluator, checkpointing,
artifact layout, and Modal orchestration ourselves.

Verdict: best future MuZero substrate if we decide to own MuZero; not a
near-term framework move.

### Sample Factory

Sample Factory is compelling for high-throughput PPO/APPO and multi-agent
training. Its docs emphasize synchronous/asynchronous PPO, multi-agent training,
self-play, multiple policies, PBT, custom metrics, and custom environments. Its
custom multi-agent docs expect a `num_agents` property and list/array/tensor
returns per agent, but also require auto-reset.

The auto-reset and framework-owned rollout lifecycle are exactly where CurvyZero
needs caution: local docs emphasize terminal snapshots, final observations,
source-visible events, durable Modal artifacts, and explicit reset/autoreset
semantics. Those should be proven in repo-native PPO before outsourcing the
rollout engine.

Verdict: strong later throughput candidate/reference, not v0.

### RLlib

RLlib's multi-agent API can express simultaneous, turn-based, or mixed
multi-agent environments, maps `N` agents to `M` policies, and supports
PettingZoo/OpenSpiel wrappers. That is broadly compatible with CurvyTron.

The downside is the Ray stack and API/runtime weight. CurvyZero is still
settling simulator fidelity, target telemetry, and eval discipline. RLlib would
solve scale too early while making it harder to inspect every row.

Verdict: later scale/evaluation comparison only.

### SB3

SB3 is excellent for standard Gymnasium PPO and vectorized single-agent
training. Its vector envs stack independent environments so actions,
observations, rewards, and done flags become vectors over env copies.

That is not the natural CurvyTron shape. We could wrap one ego player and hide
the other players in the environment, but that recreates the LightZero semantic
problem without MuZero benefits.

Verdict: useful single-agent PPO reference; not a simultaneous CurvyTron
framework.

### OpenSpiel

OpenSpiel is the best conceptual reference for game-theoretic breadth. Its docs
explicitly list single/multi-player games, imperfect information, chance nodes,
n-player normal-form games, sequential and simultaneous moves, and zero-sum,
general-sum, and cooperative games.

That makes it a good reference for evaluation concepts, simultaneous-move
search literature, AlphaRank/PSRO-style thinking, and game taxonomy. But its
C++/Python extensive-form formalism is too heavy for a source-fidelity
CurvyTron simulator we are still pinning.

Verdict: research/eval reference, not implementation target.

### MiniZero and muzero-general

MiniZero is a serious AlphaZero/MuZero/Gumbel framework with Go/Othello/Atari
coverage and a server/self-play-worker/optimization-worker/storage architecture.
It is useful to read for large-scale MuZero system design, but adopting it would
pull in a game-specific, distributed, Docker/GPU-oriented stack before CurvyZero
has a stable simple baseline.

muzero-general remains useful as educational pseudocode and terminology. It
should not become the core framework for a source-faithful simultaneous,
batched, artifact-heavy CurvyTron pipeline.

Verdict: references only.

## Recommended Path

### Phase 0: Freeze The Boundary

Keep the simulator core library-independent:

```text
reset_rows(seed, profile) -> state[B, P, ...], obs[B, P, ...], info_arrays
step_rows(state, actions[B, P], rng) -> next_state, obs, reward[B, P], done[B], info_arrays
```

Expose adapters outward:

- PettingZoo `ParallelEnv` for simultaneous multi-agent tools.
- Ego Gymnasium adapter only for single-agent baselines or LightZero-style
  experiments.
- LightZero `BaseEnv` adapter only inside the contained MuZero lane.

### Phase 1A: LightZero Pong Control

Discharge the LightZero replication obligation before declaring that another
framework is the answer:

- fix or route around the stock evaluator `action_mask` issue with a documented
  no-fallback eval path;
- evaluate an official/pretrained Pong checkpoint if one can be used in the
  pinned image;
- run a stock Atari Pong trainer rung with upstream visual/model/action
  semantics and settings materially closer to the recipe than local smoke runs;
- produce a strict checkpoint curve and artifact mirror;
- write down whether the result is mechanical-only, signal-bearing, or a
  concrete blocker.

PPO work can proceed in parallel, but it is not a substitute for this control.

### Phase 1B: Repo-Native PPO Baseline

Implement or refine a PPO baseline in the repo, using CleanRL as a reference
style:

- one readable training entrypoint;
- no Ray or external rollout framework;
- shared policy over ego observations;
- rollout storage with explicit `T x B x P` or flattened live-ego rows plus
  stable mapping back to env/player ids;
- fixed random/scripted baselines first, checkpoint-pool opponents later;
- Modal job writes `summary.json`, `iteration_metrics.jsonl`, checkpoints,
  `episodes.jsonl`, action histograms, terminal causes, and held-out eval.

This is the fastest honest answer to "can CurvyTron learn?"

### Phase 2: LightZero Contained Lane

After the stock Pong control, keep using LightZero only where it earns its keep:

- custom-env import/train smokes;
- root target sidecars;
- support-scale inspection;
- checkpoint mirroring and independent scorecards;
- MuZero target-quality investigations.

Do not promote LightZero to CurvyTron backbone until both are true: the stock
Pong control is reproduced well enough to trust our setup, and the custom-env
lane demonstrates improvement under independent CurvyZero scorecards without
hiding joint action/seed/terminal metadata.

### Phase 3: Mctx Owned MuZero

Start project-owned Mctx MuZero only after one of these is true:

- PPO baseline is stable enough to justify planning comparisons;
- LightZero gives a concrete blocker that Mctx ownership solves;
- fixed-shape vector CurvyTron throughput makes JAX search/training attractive.

The Mctx version should start ego-only, `A=3`, scalar ego value, policy-only
opponents, fixed compiled profiles, and no joint-action search.

## Stop And Reversal Rules

Stop a framework migration if it:

- forces the simulator core away from fixed `B x P` arrays;
- hides seed, joint actions, terminal reward, truncation, final observation, or
  source trace hashes;
- requires auto-reset semantics before terminal snapshots are captured;
- makes independent scorecards harder than trainer-side reward logs;
- requires joint-action search before ego-vs-baseline works;
- turns Modal artifacts/checkpoints into framework-private state.

Reconsider Sample Factory or RLlib only after repo-native PPO has a working
artifact/eval contract and the problem is clearly throughput, not semantics.

Reconsider LightZero as backbone only after it shows reliable custom-env
improvement with target sidecars, correct support scale, and independent eval
after the stock Pong control obligation has been met.

Reconsider MiniZero or another MuZero framework only if CurvyZero decides to
adopt a distributed MuZero system wholesale and is willing to adapt the game to
that system's architecture. That is not the current practical path.

## Source Links

Primary/current web sources:

- LightZero repo and supported algorithm table:
  https://github.com/opendilab/LightZero
- LightZero custom environment docs:
  https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
- Mctx PyPI docs:
  https://pypi.org/project/mctx/
- PettingZoo Parallel API:
  https://pettingzoo.farama.org/api/parallel/
- Sample Factory overview:
  https://www.samplefactory.dev/
- Sample Factory custom multi-agent environments:
  https://www.samplefactory.dev/03-customization/custom-multi-agent-environments/
- Sample Factory PettingZoo integration:
  https://www.samplefactory.dev/09-environment-integrations/pettingzoo/
- RLlib multi-agent environments:
  https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
- Stable-Baselines3 vectorized environments:
  https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- OpenSpiel intro:
  https://openspiel.readthedocs.io/en/stable/intro.html
- MiniZero repo:
  https://github.com/rlglab/minizero
- CleanRL docs:
  https://docs.cleanrl.dev/

Local sources used:

- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/training_lessons_for_curvytron_2026-05-09.md`
- `docs/working/environment/vector_state_schema.md`
- `docs/working/environment/vector_lifecycle_plan_2026-05-09.md`
- `docs/design/training_architecture.md`
- `docs/research/muzero_repo_baseline_options.md`
- `docs/research/simple_training_environment_options.md`
- `docs/research/mctx_integration.md`
- `docs/research/lightzero_feature_fit_for_curvyzero.md`
- `docs/research/multiplayer_selfplay_muzero.md`
