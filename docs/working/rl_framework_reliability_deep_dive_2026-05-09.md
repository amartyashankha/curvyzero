# RL Framework Reliability Deep Dive - 2026-05-09

Owner: RL framework reliability researcher.

Scope: docs-only comparison of alternatives to LightZero for CurvyZero. No code
or pytest was run. This memo uses local project docs plus current primary
sources checked on 2026-05-09. GitHub star counts are rounded snapshots from
repository pages where available; they are popularity signals, not a ranking.

## Short Answer

Do not migrate the CurvyZero training backbone wholesale to another framework
yet.

The robust path is still a split architecture:

1. Keep the simulator, vector state, reset/final-observation semantics,
   scorecards, and Modal artifact layout repo-owned.
2. Use a PettingZoo `ParallelEnv` adapter as the external multi-agent API,
   because it naturally represents wrapper action maps for all live players.
3. Treat repo-native PPO, written in a CleanRL-style readable loop, as the
   first serious learnability baseline over `[T,B,P]` rollouts.
4. Keep LightZero as the contained MuZero control lane until the stock
   Pong/custom-env obligations produce either reliable improvement or a
   documented blocker.
5. Keep Mctx as the likely project-owned MuZero search substrate if MuZero
   earns more investment, but remember that Mctx is search only.

The strongest alternatives to LightZero are not drop-in replacements. They are
role-specific:

- PPO reliability: repo-native PPO informed by CleanRL, TorchRL, and Sample
  Factory.
- API reliability: PettingZoo `ParallelEnv`; optionally SuperSuit only for
  legacy wrappers while noting its maintenance status.
- MuZero reliability: LightZero for immediate full-trainer control, Mctx for
  owned search, MiniZero/EfficientZero/muzero-general/Muax as references rather
  than backbones.

## Project Fit Criteria

CurvyTron's source semantics are held player control state advanced by
elapsed-millisecond server frames. The trainer-facing wrapper requirement is an
auditable all-live-player decision/control snapshot:

```text
actions[B, P]
observations[B, P, ...]
rewards[B, P]
done_or_truncated[B]
info arrays that preserve env id, player id, seed, wrapper joint action/control
snapshot, terminal cause, final observation, trace hash, and scorecard fields
```

Any framework that forces this into hidden single-agent ego wrappers can still
be useful for experiments, but it should not become the semantic owner of the
environment.

Local docs point the same way:

- `docs/working/training_framework_alternatives_2026-05-09.md` recommends a
  layered boundary: PettingZoo-style public API, repo-native PPO baseline,
  LightZero control lane, and Mctx search lane.
- `docs/research/lightzero_feature_fit_for_curvyzero.md` says LightZero is the
  fastest complete MuZero trainer path already proven on Modal, but weak for
  all-live-player CurvyTron wrapper self-play.
- `docs/research/mctx_integration.md` says Mctx fits fixed batched search rows
  and `A=3`, but supplies neither replay nor learner nor evaluator.
- `docs/working/repo_native_ppo_learner_boundary_2026-05-09.md` records that
  the repo-native PPO smoke already preserves `[T,B,P]` rollout fields, masks,
  GAE/returns, checkpoint, and profiling buckets, while still being explicitly
  no-quality.

## Current Signal Matrix

| Candidate | GitHub stars observed | Maintenance/docs signal | Helps PPO? | Helps MuZero? | `[B,P]` simultaneous fit | Reliability read |
| --- | ---: | --- | --- | --- | --- | --- |
| Stable-Baselines3 | 13.2k | Mature, stable, maintained; v2.8.0 Apr 2026; docs and tests are strong. | Strong reference for single-agent PPO; good comparison baseline. | No. | Weak natively; VecEnv stacks independent envs, not players. | Reliable library, wrong core shape. |
| CleanRL | 9.6k | Strong educational/research docs; single-file, benchmarked, seeded implementations. | Very strong reference style for repo-native PPO. | No. | Medium if copied as style into our `[T,B,P]` loop; weak as imported framework. | Best PPO reading material, not a dependency target. |
| TorchRL | 3.4k | Active PyTorch project, modular, documented, tested; multi-agent PPO tutorials exist. | Strong components and ideas; more machinery than v0 needs. | No direct MuZero. | Medium/strong if TensorDict shapes are carefully owned; adoption risk is abstraction weight. | Later component library candidate, not first backbone. |
| Sample Factory | ~1.0k | PPO/APPO throughput focus; tested; docs include multi-agent, multi-policy, self-play. | Strong later throughput candidate. | No. | Medium; multi-agent returns exist, but auto-reset conflicts with terminal evidence discipline. | Use after repo-native PPO proves semantics and throughput is the blocker. |
| RLlib | 42.4k for Ray repo | Large maintained ecosystem; Ray 2.55.1 Apr 2026; multi-agent API plus PettingZoo/OpenSpiel wrappers. | Strong but heavyweight PPO. | No native MuZero path for this need. | Medium; can express multi-agent policies, but Ray sampling/eval becomes semantic owner. | Too heavy while simulator/eval contracts are still settling. |
| PettingZoo | 3.4k | Farama-maintained API; Parallel API explicitly for simultaneous agents. | Indirect: adapter into PPO libraries. | Indirect only. | Strong API fit; not hot-loop backend. | Adopt as compatibility boundary. |
| SuperSuit | star count not reliable from opened page | Primary README says semi-deprecated/unmaintained except compatibility until wrappers merge. | Useful preprocessing wrappers only. | No. | Medium as PettingZoo/Gym wrapper layer, but avoid reliance. | Treat as legacy/transition utility, not core dependency. |
| OpenSpiel | 5.2k | DeepMind-maintained research framework; strong docs; supports simultaneous, stochastic, general-sum games. | Research/eval algorithms, not our PPO engine. | Has AlphaZero materials, not Curvy MuZero trainer. | Conceptually strong; implementation-heavy C++/pybind game formalism. | Great theory/eval reference; poor source-fidelity path. |
| LightZero | 1.6k | Updated Mar 2026; docs; integrated MuZero family table; Modal CartPole already worked locally. | No. | Strongest complete MuZero trainer candidate. | Weak/medium; custom env path is ego-wrapper shaped, not native `[B,P]`. | Keep contained; do not promote before stock/custom controls. |
| Mctx | 2.6k | DeepMind JAX library; latest PyPI 0.0.6 Sep 2025; small API, low churn. | No. | Strong search substrate: MuZero/Gumbel/Stochastic policies. | Strong for fixed batched ego rows, not joint actions. | Best owned-MuZero search primitive, but trainer bill is ours. |
| MiniZero | star count not visible in opened lines | Active enough to advertise development; IEEE ToG 2024; supports AlphaZero/MuZero/Gumbel and Atari. | No. | Strong full-system MuZero reference. | Weak for v0; server/worker/storage architecture wants to own the game pipeline. | Reference architecture, not migration target. |
| muzero-general | 2.8k | Educational, documented; only 132 commits on opened page; Ray-based async features. | No. | Educational MuZero trainer. | Weak/medium for simple games; weak for simultaneous vector CurvyTron. | Pseudocode/reference only. |
| EfficientZero | 933 | Research code, GPL-3.0, no releases; C++/Cython/Ray build surface. | No. | EfficientZero Atari-family reference. | Weak; Atari-heavy distributed pipeline. | Too risky as backbone; useful for target/reanalyze ideas. |
| Muax | GitHub stars unavailable from primary pages; PyPI last release May 2023 | Small JAX library around Mctx for gym-style MuZero; old release cadence. | No. | Thin MuZero helper around Mctx. | Weak/medium; gym-style, not simultaneous `[B,P]`. | Read for JAX/Mctx trainer sketches; do not depend on it. |

## Reliability Beyond Stars

Stars answer "who has heard of it." They do not answer "can we debug a failed
CurvyTron run at 3 a.m. and know whether the environment, learner, search, or
artifact bridge is lying." For this project the more predictive signals are:

| Candidate | Docs quality | Release activity | Known runnable examples | Pretrained/artifact ecosystem | Custom env friction | Exact-problem usefulness |
| --- | --- | --- | --- | --- | --- | --- |
| SB3/RL Zoo | Excellent user docs, algorithm docs, VecEnv caveats, callbacks, env checker, RL Zoo recipes. | Strong: SB3 v2.8.0 in Apr 2026; RL Zoo docs tracking current SB3. | Many single-agent Gymnasium/Atari examples and tuned hyperparameters. | Strongest PPO artifact story: Hugging Face docs say the SB3 team hosts 150+ trained agents with model cards, configs, evals, and videos. | Low for single-agent Gym, high for native simultaneous multi-agent. `terminal_observation` handling is a known footgun. | Great for validating PPO expectations and checkpoint/eval UX; poor as native CurvyTron trainer. |
| CleanRL | Excellent for reading one algorithm end-to-end; explicitly not modular/import-first. | PyPI is stale at 1.2.0 May 2023, but docs/repo remain useful as source examples. | Clean single-file PPO/Atari examples; PettingZoo has a CleanRL Pistonball tutorial. | Has docs/model-zoo references, but less standardized than SB3/RL Zoo. | We avoid friction by not importing it; we translate style into repo-native code. | Best practical blueprint for our current PPO lane because it keeps the loop inspectable. |
| TorchRL | Very good but broad; main docs, knowledge base, collectors, losses, replay, and multi-agent PPO tutorial. | Very strong: PyPI 0.12.0 Apr 2026, with many 2025-2026 releases. | Strong VMAS multi-agent PPO tutorial with explicit agent dimension, shared done, centralized/decentralized critic variants. | We did not find an SB3-like pretrained zoo; examples are more component/tutorial oriented. | Medium: TensorDict can represent `B,P`, but custom collectors/keys must be audited carefully. | Best later component library if repo-native PPO outgrows hand-rolled buffers/losses. |
| Sample Factory | Good throughput docs, examples, env integrations, Hugging Face integration. | Mixed: docs active enough, but PyPI latest is 2.1.1 Jun 2023. | Many environment integrations; docs include custom multi-agent envs, PettingZoo, Atari, VizDoom, IsaacGym. | Good: Hugging Face docs and model listings include Sample Factory models with configs/evals/videos; Atari 2B checkpoints exist. | High for us: custom multi-agent API requires auto-reset and says it has no use for previous final observations. | Useful if throughput dominates later; risky before terminal/final-observation semantics are bulletproof. |
| RLlib | Very broad; excellent distributed docs, multi-agent APIs, PettingZoo/OpenSpiel wrappers, examples. | Very strong: Ray 2.55.1 Apr 2026, frequent releases. | PettingZoo PPO Pistonball tutorial; tuned Atari PPO example claims Pong in roughly 5 minutes on a large multi-GPU node. | Checkpointing is robust, but no simple pretrained zoo comparable to SB3 for our purpose. | High operational friction: Ray runtime, config stack, API-stack transitions, distributed workers. | Good future scale comparison; too much moving machinery for first learnability evidence. |
| PettingZoo | Excellent API docs and custom-env tutorials, including `parallel_api_test`. | Moderate: 1.25.0 Apr 2025, Farama-maintained. | Strong examples across Atari, Butterfly, Classic, MPE, SISL; CleanRL/RLlib/SB3 tutorials. | It is an environment API, not a model artifact ecosystem. | Low as an adapter; moderate if used as hot loop due to dict/agent bookkeeping. | Exact fit for public simultaneous-action API over our repo-owned array core. |
| SuperSuit | Useful wrapper docs, but maintenance warning is central. | Weak/transition: README says semi-deprecated/unmaintained except compatibility. | Common preprocessing examples: color reduction, resize, frame stack, normalize. | None relevant. | Low if used only in tutorials; high if it becomes a required dependency surface. | Avoid as core. Replace with repo-owned preprocessing or current Farama wrappers when needed. |
| OpenSpiel | Excellent research docs for game formalisms, algorithms, AlphaZero, evaluation tools. | Active enough on GitHub; large mature codebase. | Many games/algorithms; AlphaZero docs explicitly describe implementation limits. | Not a practical pretrained-agent source for CurvyZero. | Very high: C++/pybind game formalism and extensive-form representation. | Best for evaluation concepts, simultaneous-game taxonomy, AlphaRank/PSRO; not training backbone. |
| LightZero | Good focused docs for MuZero-family algorithms and custom envs, though DI-engine assumptions leak through. | Good: LightZero PyPI 0.2.0 Apr 2025; README updated Mar 2026 for v0.2.0. | Strongest MuZero examples in this set: CartPole, Atari, LunarLander, board games, Gumbel/Stochastic variants. | Strong but config-version sensitive: OpenDILabCommunity HF Pong includes config, dependency versions, video/eval, and self-reported 20.4 +/- 0.49, but the checkpoint matches older `[4,96,96]`/`downsample=True`, not the current `[4,64,64]` stock config. | Medium/high: `BaseEnv`, `observation/action_mask/to_play`, DI-engine config, evaluator quirks, strict checkpoint/config matching, and ego-wrapper semantics. | Best immediate MuZero control, especially for Pong; still weak as native simultaneous CurvyTron backbone. |
| Mctx | Compact, precise docs/API; easier to audit than a full framework. | Moderate: PyPI 0.0.6 Sep 2025, trusted publishing/provenance visible. | Good search demos and example projects; no training examples at Curvy scale. | No pretrained artifacts; it is a search library. | Low for search API, very high for end-to-end training because we own everything else. | Best search primitive for owned MuZero after PPO/LightZero gates. |
| MiniZero | Good README and paper-backed architecture description. | Some active-development language, but not a simple Python package cadence. | Strong board/Atari full-system examples, server/self-play/optimizer/storage. | Results exist; no simple artifact story for our Modal/Pong control. | High: distributed C++/Python style system wants to own game workers and storage. | Systems reference for MuZero only. |
| muzero-general | Good educational docs and code comments. | Weak/moderate: old educational posture; opened page showed 132 commits. | Board, Gym, Atari examples; single/two-player mode. | No strong current pretrained artifact story. | High: Ray and educational abstractions, no native `[B,P]`. | Use as pseudocode when building targets/replay, not as dependency. |
| EfficientZero | Sparse practical docs, research-code README. | Weak: no releases visible; 18 commits on opened page. | Atari quick start, Breakout-oriented commands, reanalyze/priority knobs. | No robust current artifact ecosystem found. | Very high: GPL-3.0, C++/Cython tree, Ray, multi-GPU/CPU assumptions. | Read only for sample-efficiency mechanisms. |
| Muax | Good PyPI README for a small helper; narrow examples. | Weak: latest PyPI release May 2023. | CartPole notebook/script path using Mctx, Haiku, Optax, tracer, replay. | None meaningful. | Medium/high: gym-style and older JAX stack. | Useful sketch of missing Mctx trainer pieces; avoid dependency. |

## Practical Reliability Ranking By Role

This is not a global ranking. It is a role assignment for our exact failure
surface.

| Role | Best practical choice | Why | Do not confuse it with |
| --- | --- | --- | --- |
| External simultaneous env API | PettingZoo `ParallelEnv` | It matches simultaneous actions and has a `parallel_api_test`. | A high-throughput internal vector backend. |
| First learnability baseline | Repo-native PPO, CleanRL-style | Smallest path that preserves `[T,B,P]`, masks, terminal rows, and artifacts. | Importing CleanRL as a framework. |
| PPO reliability reference | SB3/RL Zoo | Mature docs, tests, releases, pretrained agents, tuned configs. | Native multi-agent CurvyTron support. |
| PPO component library later | TorchRL | Active releases and real multi-agent PPO with agent dimensions. | A no-friction replacement for our current learner smoke. |
| PPO throughput later | Sample Factory | Multi-agent/multi-policy APPO throughput, HF models, serial debug mode. | A safe terminal-evidence collector before we solve auto-reset. |
| Distributed scale later | RLlib | Industrial multi-agent scale and wrappers. | A simple debugging substrate. |
| Immediate full MuZero control | LightZero | Only full MuZero trainer already locally proven on Modal; has Pong pretrained artifact. | Native simultaneous self-play. |
| Owned MuZero search later | Mctx | Clear batched JAX search and policy-improvement targets. | Replay/learner/evaluator/checkpoint framework. |
| MuZero system reference | MiniZero | Serious full-system AlphaZero/MuZero architecture. | A small integration dependency. |
| MuZero pseudocode/reference | muzero-general, Muax, EfficientZero | Useful isolated ideas. | Maintained CurvyZero backbone. |

## Candidate Reads

### Stable-Baselines3

SB3 is the most reliable standard PPO package in this set. Its own description
emphasizes reliable PyTorch RL implementations, docs, custom envs, callbacks,
type hints, and high coverage. The current GitHub page shows v2.8.0 released on
2026-04-01, so the project is alive and conservative.

Its artifact story is better than most RL frameworks: RL Baselines3 Zoo provides
training/eval/tuning/recording scripts plus tuned hyperparameters, and Hugging
Face's RL Zoo docs say the SB3 team hosts 150+ trained agents with model cards,
training configs, evaluation results, and videos. That matters because an SB3
control run is easy to explain and compare.

The mismatch is shape. SB3 `VecEnv` stacks independent environments into one
batch and auto-resets done environments; the docs warn that the observation at a
done index is already the next episode's first observation and that
`terminal_observation` must be carried through `info`. CurvyZero wants
`done[B]` with per-player terminal evidence, not a single-agent VecEnv that
hides the `P` axis.

Verdict: use SB3/RL Zoo as a reliability reference and possibly a single-agent
ego sanity check. Do not make it the CurvyTron backbone.

### CleanRL

CleanRL is the best influence on the repo-native PPO lane. It is explicitly
single-file, readable, benchmarked, seeded, and not meant to be imported as a
modular library. That is almost exactly the posture CurvyZero needs: copy the
clarity, not the framework boundary.

The main caveat is release packaging, not idea quality. PyPI shows the latest
CleanRL package at 1.2.0 from 2023-05-22, so depending on it as a library would
not buy us much. But PettingZoo's CleanRL Pistonball tutorial is exactly the
kind of practical bridge we want: small PPO code over a parallel multi-agent
environment.

Verdict: strongest PPO reference. The implementation should remain repo-native
so rollout storage, masks, GAE, scorecards, and checkpoint artifacts stay shaped
around `[T,B,P]`.

### TorchRL

TorchRL has a healthier long-term component story than SB3 for custom research:
collectors, replay, transforms, losses, modules, TensorDict, multi-agent PPO
tutorials, and PyTorch-native extensibility. It is also active, documented, and
tested.

Release activity is the best in the PPO component group: PyPI lists TorchRL
0.12.0 on 2026-04-27 and many 2025-2026 releases. The multi-agent PPO tutorial
is especially relevant because it shows tensors with environment and agent
dimensions, an `"agents"` nested key, shared `done` outside the agent key, and
per-agent action/reward data. That maps conceptually to CurvyZero's `B,P`
layout better than SB3 or RLlib.

The risk is abstraction weight. If CurvyZero adopts TorchRL too early, we may
spend the same effort proving that TensorDict collectors preserve terminal
snapshots, final observations, player axes, and Modal artifacts that a small
repo-native PPO loop already exposes plainly.

Verdict: good later component library if the repo-native PPO loop grows too much
manual machinery. Not the first reliability move.

### Sample Factory

Sample Factory is attractive if throughput becomes the bottleneck. Its docs
center efficient synchronous/asynchronous PPO/APPO and support multi-agent,
multi-policy, self-play, PBT, serial debugging, and GPU/CPU-heavy envs.

Its example/artifact story is good but older. The docs advertise Hugging Face
integration, multiple tuned environment integrations, and trained models; the
Hugging Face model listings show Sample Factory Atari/Pong and Atari-2B models.
PyPI, however, shows the latest package release as 2.1.1 from 2023-06-19, so
we should treat the current source/docs as the real target if this lane becomes
serious.

Its custom multi-agent docs also state that multi-agent envs should auto-reset
an agent when terminated or truncated, returning the first observation of the
next episode because the last observation is not used for acting. That conflicts
with CurvyZero's current reliability obsession: terminal/final-observation
evidence is not optional until the simulator and scorecards are trusted.

Verdict: keep as later throughput candidate and design reference. Do not adopt
until repo-native PPO proves the semantics and the measured problem is speed.

### RLlib

RLlib can express simultaneous, turn-based, and mixed multi-agent environments,
maps agents to policies, and directly wraps PettingZoo. It is the broadest
industrial option here.

Its current activity is real: Ray's release page shows 2.55.1 in Apr 2026, and
the docs show `pip install -U "ray[rllib]"`, API stability annotations, tuned
examples, PettingZoo wrappers, and OpenSpiel wrappers. The example ecosystem is
strong too: PettingZoo has an RLlib Pistonball PPO tutorial, and RLlib docs cite
a tuned Atari PPO example that solves Pong quickly on a large multi-GPU node.

The liability is the Ray stack. CurvyZero is still pinning source fidelity,
reset/autoreset semantics, artifact manifests, and independent scorecards. A
distributed sampler can solve scale while making every suspicious rollout row
harder to inspect.

Verdict: later scale/eval comparison only. It should not be the first PPO or
MuZero alternative.

### PettingZoo And SuperSuit

PettingZoo is the cleanest API answer. Its Parallel API is explicitly for
environments where all agents have simultaneous actions and observations; `step`
receives an action dict keyed by agent and returns observation, reward,
termination, truncation, and info dictionaries.

Its testing story is also useful: the custom environment tutorial uses
`parallel_api_test(env, num_cycles=1_000_000)`. That is not our pytest lane
today, but it is a concrete adapter-contract check when we build the public
interface.

That matches CurvyTron's public shape. It should be an adapter over the
repo-owned array core, not the internal hot representation.

SuperSuit is different. It is useful for wrappers, but the primary README now
labels it semi-deprecated/unmaintained except for compatibility while wrapper
functionality moves into Gymnasium/PettingZoo wrappers. That makes it a
transition tool, not a reliability foundation.

Verdict: adopt PettingZoo `ParallelEnv` as compatibility API; avoid making
SuperSuit critical.

### OpenSpiel

OpenSpiel is the best conceptual fit for game theory. It supports single and
multi-player games, imperfect information, explicit chance nodes, n-player
normal-form games, sequential and simultaneous moves, and zero-sum, general-sum,
and cooperative games. It also has tools for learning dynamics and evaluation.

The problem is implementation friction. CurvyTron is a source-fidelity,
geometry/tick/trace problem first. Recasting it as an OpenSpiel C++/pybind
extensive-form game would answer a different research question before the
simulator is stable.

Verdict: strong reference for evaluation/game taxonomy and future
AlphaRank/PSRO-style thinking. Not a CurvyTron training framework.

### LightZero

LightZero remains the only complete MuZero-family trainer already validated in
this repo's Modal path, via stock CartPole. Current upstream signals are better
than the local pain might suggest: the README was updated for LightZero v0.2.0
on 2026-03-11, PyPI lists LightZero 0.2.0 from 2025-04-09, and its integrated algorithm table lists MuZero, Sampled MuZero,
Stochastic MuZero, EfficientZero, Gumbel MuZero, ReZero, and UniZero across
several env families. Its custom-env docs explain the DI-engine `BaseEnv`
pattern and the LightZero observation dict with `observation`, `action_mask`,
and `to_play`.

The pretrained-artifact signal is unusually relevant but not plug-and-play:
OpenDILabCommunity's Hugging Face `PongNoFrameskip-v4-MuZero` card includes a
policy config, dependency versions, run/deploy snippets, a video/eval surface,
and a self-reported Pong mean reward of `20.4 +/- 0.49`. That is exactly the
kind of stock control artifact we should try before concluding that LightZero
is unreliable for Pong-like MuZero.

But this is also a reliability trap. Local checkpoint-compatibility notes show
that the HF Pong checkpoint is an older `[4,96,96]` visual contract with
`downsample=True`, while the current stock LightZero Atari config/eval path in
our image is `[4,64,64]`. It should be evaluated only through the matching
downloaded/recreated 96x96 config and wrapper surface. Non-strict-loading it
into the current 64x64 path would convert a useful control into a misleading
adoption shortcut.

The fit problem is not maintenance; it is semantics. LightZero's custom-env path
can host an ego wrapper, but native simultaneous `[B,P]` self-play becomes
hidden inside `step`. Local LightZero dummy Pong evidence also says scale alone
is not enough: target quality, action collapse, support scale, checkpoint
loading, evaluator/API mismatch, and independent scorecards must be proven.

Verdict: keep as contained MuZero control lane. It is still worth discharging
the stock Pong/custom dummy Pong obligations before declaring alternatives the
answer.

### Mctx

Mctx is the best MuZero-family alternative if CurvyZero decides to own the
trainer. Its README says search is JAX-native, JIT-compatible, batched in
parallel, and provides high-level `muzero_policy` and `gumbel_muzero_policy`;
PyPI 0.0.6 also describes Mctx as AlphaZero/MuZero/Gumbel search, shows a
2025-09-02 release with trusted publishing/provenance, and lists Muax
as a gym-style MuZero example. The GitHub README recommends Gumbel MuZero and
documents action weights as policy-improvement targets. The local Mctx notes
already map this cleanly to `RootFnOutput(prior_logits[B,A], value[B],
embedding[B,...])` and `RecurrentFnOutput`.

The gap is everything else: actors, replay, targets, model code, optimizer,
checkpointing, scorecards, Modal run management, and self-play leagues. For
simultaneous CurvyTron, the first sane Mctx formulation is one ego row per live
player with `A=3`, while opponent behavior is learned from replay or supplied
outside the search. Joint-action search should stay out of v0.

Verdict: best search substrate after PPO gates or after a concrete LightZero
blocker. Not a framework replacement by itself.

### MiniZero

MiniZero is a serious AlphaZero/MuZero/Gumbel framework and a better MuZero
systems reference than muzero-general. Its README lists AlphaZero, MuZero,
Gumbel AlphaZero, Gumbel MuZero, many board games, and Atari 57. It also
documents a server, self-play workers, optimization worker, and data storage
architecture with batched GPU inferencing.

That is a useful design reference because it makes the production MuZero shape
explicit: self-play workers, batched GPU inference, optimizer, and durable game
record storage are separate concerns. It also means adopting it would import a
system, not a library.

That architecture is the reason not to adopt it now. CurvyZero does not yet
need a full distributed MuZero stack to own the game loop; it needs transparent
terminal evidence and a small learnability baseline.

Verdict: reference for system design and batched MCTS worker patterns only.

### muzero-general

muzero-general is popular and readable, with a documented PyTorch MuZero
implementation, Ray-based async/cluster support, checkpointing, TensorBoard,
single/two-player modes, and game adapters. Its own README calls it primarily
educational.

For CurvyZero that is the right ceiling. It can teach replay/target
terminology, file decomposition, and simple game adapter patterns. But its age,
Ray shape, small commit count, and lack of native simultaneous vector semantics
make it a poor backbone.

Verdict: pseudocode companion only.

### EfficientZero

EfficientZero is valuable research code for sample-efficient MuZero-style Atari,
but it is the highest-risk adoption target here: GPL-3.0, no releases visible,
18 commits on the opened GitHub page, C++/Cython tree builds, Ray, and
Atari/distributed assumptions. It solves a later sample-efficiency problem, not
the current reliability problem.

Verdict: read for reanalysis, target, and efficient search ideas. Do not adopt.

### Muax

Muax is a small JAX helper library around DeepMind Mctx for gym-style MuZero.
PyPI shows the last release on 2023-05-01. It includes a CartPole-style training
example, episode tracer, replay buffer, Optax optimizer path, and MuZero model
wrapper.

That makes it useful as a sketch of the missing Mctx trainer pieces. Its
release age and gym-style semantics make it a weak dependency for CurvyTron.

Verdict: reference only; use Mctx directly if owning MuZero.

## Recommendation

### Near Term

Proceed with two lanes:

1. **Repo-native PPO lane**: keep the current optional-Torch PPO learner path
   small, explicit, and shaped by `[T,B,P]`. Use CleanRL as style, SB3 as a
   sanity/reference baseline, and TorchRL/Sample Factory only as references for
   later collectors, replay, or throughput.
2. **LightZero control lane**: finish the stock/custom control obligation before
   replacing it. A move away from LightZero should cite a concrete failure:
   hidden seed/action/final-observation metadata, unsalvageable custom env
   wrapper, artifact/checkpoint opacity, or inability to reproduce a stock-like
   Pong control under bounded effort.
   Pretrained controls must be treated as config-pinned artifacts: the
   OpenDILab HF Pong checkpoint is a 96x96/downsample control, not evidence that
   the current 64x64 stock config can strict-load or evaluate it.

Add one no-code acceptance rule to both lanes: every candidate must say exactly
where final observations, joint actions, opponent id, seed, terminal cause,
score return, shaped diagnostic return, checkpoint id, and eval seed land in
its artifacts. If the answer is "inside the framework logs somewhere", it is
not reliable enough yet.

### Medium Term

Expose a PettingZoo `ParallelEnv` adapter over the repo-owned simulator. This
gives RLlib, Sample Factory, CleanRL tutorials, and other tools a standard entry
without letting them define simulator truth.

Start project-owned Mctx MuZero only after either:

- PPO can learn enough to justify planning comparisons;
- LightZero produces a documented blocker that Mctx ownership solves; or
- fixed-shape JAX batching becomes clearly worth the trainer implementation
cost.

### Rejection Rules

Reject or pause any framework path that:

- removes or hides the `P` axis from the core simulator;
- auto-resets before final observations and terminal info are captured;
- hides joint actions, seeds, opponent policy, terminal reward, trace hash, or
  scorecard rows;
- requires joint-action search before ego-vs-baseline works;
- makes independent eval harder than trainer reward logs;
- makes Modal artifacts/checkpoints framework-private;
- adds Ray/C++/Cython/distributed workers before a simple baseline learns.

## Sources

Primary/current web sources:

- Stable-Baselines3 GitHub: https://github.com/DLR-RM/stable-baselines3
- Stable-Baselines3 VecEnv docs:
  https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- CleanRL docs: https://docs.cleanrl.dev/
- CleanRL GitHub: https://github.com/vwxyzjn/cleanrl
- TorchRL docs: https://docs.pytorch.org/rl/main/index.html
- TorchRL GitHub: https://github.com/pytorch/rl
- Sample Factory docs: https://www.samplefactory.dev/
- Sample Factory custom multi-agent docs:
  https://www.samplefactory.dev/03-customization/custom-multi-agent-environments/
- Sample Factory GitHub: https://github.com/alex-petrenko/sample-factory
- RLlib multi-agent docs:
  https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
- Ray GitHub: https://github.com/ray-project/ray
- PettingZoo Parallel API: https://pettingzoo.farama.org/api/parallel/
- PettingZoo GitHub: https://github.com/Farama-Foundation/PettingZoo
- SuperSuit GitHub: https://github.com/Farama-Foundation/SuperSuit
- OpenSpiel docs: https://openspiel.readthedocs.io/en/stable/intro.html
- OpenSpiel GitHub: https://github.com/google-deepmind/open_spiel
- LightZero GitHub: https://github.com/opendilab/LightZero
- LightZero custom env docs:
  https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
- Mctx GitHub: https://github.com/google-deepmind/mctx
- Mctx PyPI: https://pypi.org/project/mctx/
- MiniZero GitHub: https://github.com/rlglab/minizero
- muzero-general GitHub: https://github.com/werner-duvaud/muzero-general
- EfficientZero GitHub: https://github.com/YeWR/EfficientZero
- Muax PyPI: https://pypi.org/project/muax/

Local sources used:

- `docs/working/training_framework_alternatives_2026-05-09.md`
- `docs/research/muzero_repo_baseline_options.md`
- `docs/research/lightzero_feature_fit_for_curvyzero.md`
- `docs/research/mctx_integration.md`
- `docs/design/muzero_modal_architecture.md`
- `docs/working/repo_native_ppo_learner_boundary_2026-05-09.md`
- `docs/working/training_lessons_for_curvytron_2026-05-09.md`
- `docs/working/lightzero_pretrained_checkpoint_compatibility_2026-05-09.md`
- `docs/working/lightzero_exact_upstream_atari_command_2026-05-09.md`
- `docs/working/training_state_index_2026-05-09.md`
