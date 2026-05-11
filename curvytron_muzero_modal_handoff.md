# CurvyTron MuZero on Modal
Expanded repository-starting-point and implementation handoff
Version 2 - May 8, 2026

Audience: an engineer or ML researcher who needs to turn the CurvyTron/MuZero idea into a working repository, runnable Modal jobs, and a debug path toward a strong self-play agent.
# 0. Executive decision: fork CurvyTron, but do not make it the main ML repo
The earlier answer was too compressed. The better answer is: yes, fork CurvyTron, and possibly fork one or two ML libraries later, but do not make a browser-game fork the main training repository. The main repository should be a clean Python/JAX-or-PyTorch training system with a fast deterministic simulator. CurvyTron should be a reference/browser/demo/golden-test fork.
| Repository | Fork? | Role | Why |
| --- | --- | --- | --- |
| curvyzero / curvytron-rl (new repo) | Create fresh | Main ML repo: environment, baselines, MuZero, Modal jobs, tests, benchmarks. | Keeps training code clean and avoids old browser/server assumptions leaking into the RL stack. |
| Curvytron/curvytron | Yes | Reference fork: rules, collision behavior, browser demo, UI, possible golden-test oracle. | The upstream repo is a MIT-licensed web multiplayer Tron-like game with JavaScript/CSS/HTML and release 1.4.5; it is useful, but not shaped like a modern RL training environment. |
| CurvyTron 2 | Maybe / investigate | Modern rules reference only, unless source is found. | The public site documents current rules and bonuses. I did not find an obvious public GitHub source repo for CurvyTron 2. |
| Mctx | No at first | Dependency/reference for JAX batched MCTS/MuZero search. | It is a search library, not a full training stack. Use pinned pip/Git dependency first; fork only if we must patch internals. |
| LightZero | Maybe | PyTorch MuZero-family toolkit candidate. | Good if we prefer PyTorch, but custom-env and multi-agent details still need integration work. Fork only after a smoke test proves we need changes. |
| muzero-general | No production fork | Educational/reference implementation. | Readable and useful, but it explicitly lacks Batch MCTS and >2-player support as future improvements; those are central for CurvyTron. |
| curvytron-client or old RL repos | Maybe archive/reference | Reference for client protocol or prior attempts. | Useful for ideas, but not enough to avoid building a fast step-by-step simulator. |

# 1. Why not just fork CurvyTron and add MuZero inside it?
Because the CurvyTron repo and the training system want opposite shapes.
- **CurvyTron wants browser/server real time. **Training wants headless deterministic simulation that can run faster than real time and be reset/stepped millions of times.
- **CurvyTron wants UI, sockets, rooms, rendering, config, assets, and web dependencies. **Training wants pure state transitions, vectorized batches, reproducible seeds, profileable collision code, and minimal I/O.
- **CurvyTron is JavaScript-era web code. **The raw package metadata shows old Node-era dependencies such as gulp 3.x, express 4.13, faye-websocket, bower, Angular 1.4.3, and createjs-soundjs. That is fine for a reference fork, but not ideal as the root of a 2026 GPU RL stack.
- **RL iteration needs aggressive instrumentation. **We need per-tick collision tests, environment throughput benchmarks, replay integrity checks, model versioning, and Modal job wrappers. Those should live in a repo organized around ML experiments.
The right compromise is to own two layers: a CurvyTron reference fork for source-of-truth behavior and a fresh ML repo that can deliberately copy or reimplement rules into a training-grade simulator. This is also safer legally and technically: the MIT license permits reuse, while a clean reimplementation makes it easier to test and optimize the environment without preserving browser-specific architecture.
# 2. Recommended repo topology
Create a fresh main repository, tentatively named curvyzero or curvytron-rl. Add CurvyTron as a fork/submodule/subtree only as a reference dependency. The exact repo layout should make the environment and throughput benchmarks first-class citizens.
```
curvyzero/
  README.md
  pyproject.toml                  # uv or poetry preferred; exact pins matter
  uv.lock                         # or requirements.lock
  LICENSE

  docs/
    handoff_v2.md
    design_decisions.md
    rules_extraction.md
    modal_runbook.md
    throughput_benchmarks.md
    reward_design.md

  third_party/
    curvytron-reference/           # fork/submodule/subtree of Curvytron/curvytron
    notes/                         # notes from CurvyTron 2, issues, old clients, etc.

  curvyzero_env/
    __init__.py
    config.py
    core.py                        # reset/step state machine
    state.py                       # dataclasses / array state representation
    physics.py                     # motion, heading, velocity, action-repeat
    collisions.py                  # segment/raster/grid collision rules
    scoring.py                     # rank reward, round scoring, match scoring
    bonuses.py                     # off by default; later curriculum
    observations.py                # egocentric raster / ray features
    vector_env.py                  # many envs in one process
    wrappers/
      gymnasium_env.py
      pettingzoo_parallel_env.py
      lightzero_env.py
    render_debug.py                # local debug renderer only, not training path

  curvyzero_agents/
    random_agent.py
    heuristic_agent.py
    ppo_baseline.py
    eval.py

  curvyzero_muzero/
    model.py                       # representation/dynamics/prediction
    mctx_recurrent.py              # JAX recurrent_fn if JAX track wins
    search.py                      # search wrappers, action selection
    replay.py
    selfplay.py
    trainer.py
    evaluator.py
    league.py                      # opponent checkpoint pool
    losses.py
    targets.py

  modal/
    app.py
    images.py
    run_env_benchmark.py
    run_ppo.py
    run_mctx_smoke.py
    run_muzero_train.py
    launch_sweep.py

  tests/
    test_rules_curvytron2.py
    test_collision_edge_cases.py
    test_head_head_collision.py
    test_simultaneous_death.py
    test_observation_perspective.py
    test_determinism.py
    test_vector_env_equivalence.py
    golden_cases/
      case_001_wall_collision.json
      case_002_self_collision.json
      case_003_head_head.json

  scripts/
    inspect_curvytron_js.py
    make_golden_case.py
    benchmark_env.py
    benchmark_mcts.py
    replay_browser_demo.py
```
## 3. Submodule versus subtree versus vendored snapshot
| Option | Use when | Pros | Cons | Recommendation |
| --- | --- | --- | --- | --- |
| Git submodule | We want to track upstream fork separately. | Clear provenance; easy to inspect exact commit. | Submodule friction for contributors and CI. | Acceptable if the team is comfortable with submodules. |
| Git subtree | We want CurvyTron code in the main repo without submodule pain. | Easier clone/CI; still preserves upstream history if imported carefully. | More work to sync updates. | My default choice if the team wants one repo clone. |
| Vendored snapshot | We only need a frozen source reference. | Simplest operationally; no accidental upstream changes. | Less elegant provenance; updates are manual. | Good if CurvyTron is only a rules oracle. |
| Separate fork only | We want clean separation. | No dependency clutter in ML repo. | Engineers must jump between repos. | Good for browser demo, less good for tests. |

Recommended: create a fork of Curvytron/curvytron under the project/org, then import it into the main ML repo as either a subtree or a frozen third_party snapshot. Keep the fork itself available for browser smoke tests and demo work.
# 4. What the CurvyTron fork should be used for
1. Rule mining: inspect JavaScript code for motion, trace creation, collision checks, scoring, powerups, room defaults, and speed/turn-rate parameters.
2. Golden tests: create small deterministic scenarios and verify the Python simulator matches intended rule behavior.
3. Browser demonstration: optionally show a trained policy controlling a live browser/server game later.
4. Regression comparison: keep examples of old edge cases, especially simultaneous collisions and head-to-head scoring.
5. Historical context: document whether we are cloning original CurvyTron v1 or targeting CurvyTron 2 rules.
The fork should not be used for hot-loop training unless a later benchmark proves it can step deterministically and many times faster than real time. That is unlikely without major surgery.
# 5. What we now know about the game target
CurvyTron v1 is explicitly a web multiplayer Tron-like game with curves, licensed MIT. The current public CurvyTron 2 site documents the rules more clearly: players are always moving forward, can only go left or right, die if they touch another player trace, their own trace, or walls, earn one point for each enemy dying before them in a round, and can catch bonuses. The CurvyTron 2 bonus list includes eraser, portal, invincibility, reversed controls, speed-up/slow-down variants, shrink/grow, and right-angle turning for opponents.
This matters because the first RL environment should probably not include the full bonus list. Start with the base game and rank/round scoring, then add bonuses as a curriculum.
## 6. Remaining rule gaps to close from source/gameplay
| Gap | Why it matters | How to close it |
| --- | --- | --- |
| Exact source frame timing and wrapper action repeat | Determines episode length and training horizon. | Inspect JS constants; run browser fork; decide training decision interval independent of render FPS. |
| Turn-rate/curvature dynamics | Defines how actions map to position; affects collision and control. | Extract from code or choose clone parameters and document them. |
| Trace thickness and collision geometry | Naive segment checks are slow and can disagree with original game. | Build tests for wall, own-trail, opponent-trail, and near-miss cases; use occupancy grid or swept-circle approximation. |
| Head-head / simultaneous death scoring | Affects reward fairness and multiplayer policy learning. | Use upstream issue #194 as a warning; choose deterministic tie behavior and test it. |
| Round scoring versus match scoring | CurvyTron 2 awards points per enemy dying before you; MuZero reward can use per-round rank or full-match win. | Decide whether one RL episode is a round or an entire match. Start with one round. |
| Bonuses and powerups | Adds stochasticity, partial observability, and non-stationarity. | Disable in v0; add one bonus at a time after baseline competence. |
| Observation perspective | Must avoid giving unnecessary absolute-coordinate overfitting. | Use egocentric local raster or ray features; test rotation/translation invariance. |
| Number of players | Self-play compute scales with players if each uses search. | Start 1v1; then 3-4 players with policy-only opponents or sampled search. |

# 7. Open-source library decision: where should we start?
There are two credible starting tracks. The decision should be made with a small benchmark, not by ideology.
## 7.1 Track A: JAX + Mctx as the primary MuZero path
Mctx is the strongest match to the bottleneck we care about: batched, JIT-compiled MCTS/MuZero search. Its README says it is JAX-native, supports AlphaZero, MuZero, and Gumbel MuZero-style search, fully supports JIT compilation, and operates on batches of inputs in parallel. That is exactly the design pressure for CurvyTron self-play.
- **Use Mctx if: **we can tolerate JAX/Flax/Optax; we want GPU-batched search; we are willing to implement our own training loop/replay/evaluator.
- **Do not expect Mctx to provide: **a complete end-to-end CurvyTron trainer, replay buffer, Modal orchestration, or custom environment integration.
- **Smoke test required: **run Mctx on Modal with a tiny learned model, batch sizes 64/256/1024, 25/50 simulations, and confirm actual GPU utilization and search throughput.
## 7.2 Track B: PyTorch + LightZero as a practical toolkit path
LightZero is a PyTorch toolkit combining MCTS and deep RL, with MuZero-family algorithms and documentation for custom environments. It may be the faster path to a complete trainer if its collectors, MCTS implementation, and environment abstractions fit CurvyTron. However, LightZero custom environments use a DI-engine-style wrapper and expect observations to include fields such as observation, action_mask, and to_play. That is manageable, but simultaneous multi-agent CurvyTron is not a plain single-player gym env.
- **Use LightZero if: **the team strongly prefers PyTorch; we want an integrated trainer quickly; the custom environment wrapper and MCTS path handle our needs.
- **Risks: **custom environment docs may leave gaps; issue #219 shows at least one user was confused about how to start algorithms on a custom env and how AlphaZero assumes a perfect simulator.
- **Smoke test required: **run LightZero CartPole/Pong on Modal, then a minimal 1v1 CurvyTron wrapper, then inspect whether self-play/search is batched enough. Fork only if we need patches.
## 7.3 Track C: muzero-general as educational reference only
muzero-general is readable and useful for understanding the moving parts. It advertises PyTorch, Ray, multi-GPU support, single/two-player support, and easy game adaptation. But its README lists Batch MCTS and support for more than two-player games under further improvements. A user issue also reports GPU self-play being 2-3 times slower than CPU because MCTS/inference transfer and CPU bottlenecks dominate. That is the exact failure mode we are trying to avoid.
Recommendation: read it, do not build the production CurvyTron system around it.
# 8. The practical recommendation
Start with a fresh main repo and run two short spikes in parallel:
1. Environment spike: build the Python simulator, golden tests, and throughput benchmark. This is mandatory no matter which RL library we choose.
2. JAX/Mctx spike: prove that Modal + JAX + Mctx can run batched search efficiently with our tiny model.
3. PyTorch/LightZero spike: prove that a custom CurvyTron env can be trained without fighting the framework too much.
4. Decision gate: choose JAX/Mctx if search throughput and code control matter most; choose LightZero if the integrated trainer works and throughput is acceptable.
My current default is: build the environment in Python/NumPy first for correctness, expose PettingZoo/Gymnasium wrappers, implement a PPO baseline, then choose Mctx for MuZero if the smoke test shows good GPU batching on Modal. If the team is much stronger in PyTorch, keep LightZero as Plan B.
# 9. Modal/JAX/TPU decision, resolved
We do not need TPUs. JAX does not imply TPUs. JAX has NVIDIA GPU installation paths through CUDA wheels. Modal is a GPU-oriented platform in its public docs: its GPU guide lists T4, L4, A10, L40S, A100, RTX PRO 6000, H100/H200, and B200 GPU options, and its CUDA guide says the NVIDIA driver and CUDA Driver API are already installed for GPU functions. I did not find public first-class Modal TPU hosting docs during this pass.
| Question | Answer |
| --- | --- |
| Does using JAX require TPU? | No. JAX runs on CPU and NVIDIA GPU; JAX docs provide CUDA 12/13 GPU installation commands. |
| Does Modal make TPU easy? | Public Modal docs I found are GPU-focused; they list NVIDIA GPU types, not a TPU product path. |
| Should we use TPUs anyway? | No for this project. CurvyTron MuZero bottlenecks are batching/search/env orchestration, not a need for TPU pods. |
| Which Modal GPU first? | Use L40S/A100/H100 depending availability/cost; start cheaper while debugging, then scale to H100/H200/B200 only after profiling. |
| What matters more than peak FLOPs? | Batched MCTS throughput, GPU utilization, low Python overhead, and avoiding network calls in the per-decision hot loop. |

# 10. Modal architecture: do not put per-step inference through Modal queues
Modal Queues and Dicts are useful for distributed coordination, not for the MCTS inner loop. Modal Dict docs state reads/writes go over the network and have unavoidable latency overhead of a few dozen milliseconds. Queue docs similarly say Queue interactions require network communication and add latency on the order of tens of milliseconds. That is fatal if done per MCTS node or per game tick.
## 10.1 Recommended first Modal shape
```
One Modal GPU function, one process group:

  - Runs many vectorized CurvyTron environments locally.
  - Runs batched MCTS/model inference locally on the same GPU.
  - Writes checkpoints, logs, and replay chunks to a Modal Volume or bucket.
  - Avoids Modal Queue/Dict for every action, every node, or every step.

Later scale-out:

  - Use Modal functions for coarse-grained actor jobs, evaluation jobs, and sweeps.
  - Use Queues for job coordination, not tiny inference messages.
  - Store replay as chunked files in Volume/CloudBucketMount/S3/R2/GCS.
  - Keep model inference close to the tree search that calls it.
```
## 10.2 Modal storage guidance
| Storage primitive | Use it for | Avoid using it for |
| --- | --- | --- |
| Modal Volume | Checkpoints, logs, medium-sized replay chunks, TensorBoard output. | Concurrent many-writer updates to the same file; millions of tiny files. |
| CloudBucketMount / S3 / R2 / GCS | Longer-term replay archives, evaluation videos, experiment artifacts. | Hot inner-loop reads/writes. |
| Modal Dict | Small job metadata, latest checkpoint pointer, counters. | Replay buffer, model weights, per-step data. |
| Modal Queue | Coarse job queue: evaluate checkpoint X, run seed range Y. | Per-node MCTS inference, per-action environment stepping. |

# 11. Environment implementation: the real product
The environment is the main asset. Even if MuZero changes, the environment remains useful for PPO, imitation, heuristic bots, AlphaZero-like search, and evaluation. Treat it like production software.
## 11.1 Core API
```
class CurvyTronEnv:
    def reset(self, seed: int | None = None) -> dict[str, Observation]:
        ...

    def step(self, actions: dict[str, int]) -> StepResult:
        # actions are simultaneous for all alive players
        # returns per-agent observation, reward, termination, truncation, info
        ...

@dataclass
class StepResult:
    observations: dict[str, Observation]
    rewards: dict[str, float]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    infos: dict[str, dict]
```
## 11.2 v0 environment choices
- One round equals one episode; no match-to-score-goal at first.
- 1v1 first; then 3-4 players.
- No bonuses/powerups in v0.
- Discrete actions: left/right if matching CurvyTron 2 exactly; optionally add straight for a training variant, but document that it is a variant.
- Fixed speed and turn rate; no acceleration except future bonuses.
- Wrapper action repeat: one model decision held across a documented source-frame window to reduce horizon and search cost.
- Occupancy-grid or swept-segment collision; no O(history) segment checks in the hot loop.
- Egocentric local raster observation or ray-distance features; start with compact observation before raw pixels.
## 11.3 Collision implementation note
CurvyTron-like games are deceptively sensitive to collision details. Because players move continuously and leave thick trails, a naive point-at-new-position check can miss fast crossings. Use either a swept-circle/line-segment test or a high-resolution occupancy grid updated with rasterized trail thickness. Then create golden tests for tunneling, grazing, head-head collisions, and simultaneous deaths. The original CurvyTron issue about inconsistent head-collision scoring is exactly the kind of ambiguity that should become an explicit test, not an afterthought.
# 12. Multi-agent formulation: choose this before MuZero details
| Formulation | Compute cost | Learning behavior | Recommendation |
| --- | --- | --- | --- |
| All players use MCTS every decision | Highest: players x sims x decisions. | Strong but expensive; can bottleneck quickly. | Use only after batching is working. |
| One searched ego, policy-only opponents | Much lower. | Good data for ego; rotate ego seat across envs. | Best first multiplayer scaling path. |
| Shared policy self-play, policy-only during data collection | Lowest. | PPO-like; less search improvement. | Good baseline and fast smoke test. |
| Opponent league/checkpoint pool | Medium. | Improves robustness; avoids overfitting to latest self. | Add once 1v1 learns. |
| Joint-action search over all players | Branching explodes as actions^players. | Closer to exact game search. | Avoid for 4+ players except tiny experiments. |

For MuZero v0: train a shared policy/value model from the ego perspective. In each environment, pick one or more ego players to search, store their trajectories, and let opponents use current policy or older snapshots. Rotate ego perspective so every seat produces data.
# 13. Reward design: start simple but not naive
For 1v1, terminal win/loss reward is enough if episodes are short and self-play throughput is high. For multiplayer, use rank/round scoring that matches the game more closely: CurvyTron 2 awards one point for each enemy dying before you in each round. A normalized per-round rank reward is easier for neural training and still aligned with survival.
```
For N players, at the end of a round:

  score_i = number of opponents who died before player i
  reward_i = 2 * score_i / (N - 1) - 1

Examples:
  N=2: winner +1, loser -1
  N=4: first +1.00, second +0.33, third -0.33, fourth -1.00

Tie policy must be explicit:
  - If players die in the same tick, give them the same death rank.
  - Or split the corresponding rank rewards.
  - Decide once, test it, and document it.
```
Avoid dense shaping such as reward for open space, reward for distance from walls, or penalty for turning too much until baselines fail. Curriculum is safer than heavy shaping: bigger arena, slower speed, fewer players, no bonuses, then gradually increase difficulty.
# 14. Throughput model and acceptance targets
The practical question is not whether a modern GPU can do neural inference. It can. The practical question is whether the system can batch enough MCTS nodes and environment decisions to keep the GPU doing useful work.
| Metric | Why it matters | Initial target |
| --- | --- | --- |
| Physics ticks/sec/core | Environment speed floor. | >= 100k simple ticks/sec on CPU for v0, higher if vectorized. |
| Episodes/sec random agents | Detects environment overhead and memory leaks. | Thousands/sec for tiny 1v1 settings; exact number depends tick cap. |
| Agent decisions/sec | Input to MCTS cost model. | Enough to generate millions of decisions/day. |
| MCTS node evals/sec | Primary MuZero self-play bottleneck. | Measure at batch 64/256/1024; optimize before scaling. |
| Average inference batch size | Proxy for GPU efficiency. | High enough that GPU utilization is not dominated by launch overhead. |
| GPU utilization and power | Detect CPU/blocking bottlenecks. | Use Modal GPU metrics plus profiler; utilization alone is incomplete. |
| Replay write MB/sec and file count | Avoid storage death by tiny files. | Chunk replay; avoid millions of files. |

# 15. Two-week implementation plan
## Week 1: environment + baselines + Modal smoke
1. Create fresh main repo and fork CurvyTron reference repo.
2. Implement v0 Python simulator: 1v1, no bonuses, deterministic reset/trainer step, wrapper action repeat, terminal rank reward.
3. Add local debug renderer and golden tests for wall, self, opponent, simultaneous collision, and deterministic replay.
4. Add random and simple heuristic agents.
5. Add Gymnasium and PettingZoo Parallel wrappers.
6. Add Modal image with CPU env benchmark function and GPU package smoke tests for JAX/PyTorch.
7. Run PPO or simpler policy-gradient baseline to verify learnability; do not start MuZero until a baseline learns something.
## Week 2: MCTS/MuZero proof
1. Run Mctx smoke test on Modal GPU using a toy model and synthetic root batches.
2. Implement minimal MuZero model: representation, dynamics, reward, value, policy heads.
3. Wire Mctx recurrent_fn and training targets for 1v1 CurvyTron.
4. Generate self-play with 25 simulations first; compare against random/heuristic opponents.
5. Add replay chunks, checkpointing, TensorBoard/W&B logs, and evaluation jobs.
6. Profile MCTS search, environment stepping, replay sampling, and GPU utilization.
7. Decision gate: keep JAX/Mctx, switch to LightZero, or postpone MuZero and improve baseline/env.
# 16. Detailed build-vs-fork matrix
| Component | Build fresh | Reuse/fork | Decision |
| --- | --- | --- | --- |
| CurvyTron simulator | Yes: Python/JAX/NumPy training-grade implementation. | Use CurvyTron fork only as reference. | Build fresh. |
| Browser game/demo | No. | Fork CurvyTron and possibly adapt for bot demo. | Fork reference. |
| Collision engine | Yes. | Could borrow rule constants from JS. | Build and test heavily. |
| Observation encoder | Yes. | No existing library likely fits. | Build fresh. |
| PPO baseline | Mostly no. | Use CleanRL/SB3/RLlib only if convenient; keep wrapper standard. | Reuse optional. |
| MCTS search | No if possible. | Use Mctx or LightZero. | Prefer Mctx after smoke. |
| MuZero trainer | Some. | Mctx requires own trainer; LightZero supplies more. | Build around chosen library. |
| Replay buffer | Yes/simple first. | Can borrow patterns from MuZero repos. | Build fresh. |
| Modal orchestration | Yes. | Use Modal examples/docs. | Build fresh. |
| Experiment tracking | No. | TensorBoard/W&B; choose team preference. | Reuse. |

# 17. If we choose JAX/Mctx: concrete technical path
```
JAX/Mctx spike checklist:

1. Modal GPU image:
   - Python 3.11 or 3.12
   - jax[cuda13] or jax[cuda12] depending Modal image/driver compatibility
   - mctx
   - flax or equinox
   - optax
   - tensorboard or wandb

2. Confirm device:
   import jax
   print(jax.devices())

3. Run synthetic Mctx benchmark:
   - roots: 64, 256, 1024
   - actions: 2 or 3
   - simulations: 25, 50, 100
   - hidden dim: small, e.g. 128
   - record compile time separately from steady-state runtime

4. Implement model functions:
   representation(params, obs) -> hidden
   prediction(params, hidden) -> prior_logits, value
   dynamics(params, hidden, action) -> next_hidden, reward
   recurrent_fn(params, rng_key, action, embedding) -> RecurrentFnOutput

5. JIT/vmap everything stable-shape:
   - fixed max players for a run
   - fixed observation shape
   - fixed action space
   - fixed simulation count per compiled function

6. Integrate environment outside search first:
   - CPU/NumPy vectorized env can generate real transitions
   - Mctx search uses learned latent dynamics, so simulator need not be JAX-native at first

7. Only consider JAX-native env if environment stepping becomes a measured bottleneck.
```
# 18. If we choose PyTorch/LightZero: concrete technical path
```
PyTorch/LightZero spike checklist:

1. Run official quickstart locally and on Modal GPU.
2. Wrap CurvyTron v0 environment in LightZero expected format:
   obs = {
     'observation': obs_array,
     'action_mask': np.ones(action_dim, dtype=np.int8),
     'to_play': -1 or player_id depending mode
   }
3. Start with single-agent ego-versus-scripted-opponent if multi-agent mode is painful.
4. Then implement 1v1 self-play.
5. Measure collector/MCTS throughput and GPU utilization.
6. Inspect whether MCTS inference is batched enough.
7. Fork LightZero only after identifying a specific patch requirement.
```
# 19. What could still kill the project?
| Risk | Symptom | Mitigation |
| --- | --- | --- |
| Environment ambiguity | Agent exploits clone-specific bug; results do not match intended game. | Define v0 rules explicitly; golden tests; compare with reference fork only where useful. |
| Unbatched MCTS | GPU idle, self-play slow, training takes weeks. | Use Mctx or patch batching; profile before scaling. |
| Modal network misuse | Per-step latency explodes. | Keep inner loop inside one container/process; Queues only for coarse jobs. |
| Too much scope | Powerups/multiplayer make learning inscrutable. | 1v1/no-bonus first; add features by curriculum. |
| Reward misspecification | Agent survives but does not win, or learns degenerate behavior. | Use rank/round reward; compare against heuristics; inspect videos. |
| Replay/storage mess | Millions of tiny files, slow attach/commit, lost checkpoints. | Chunk replay; use Volumes/buckets carefully; write checkpoint manifest atomically. |
| Framework lock-in | Library cannot support simultaneous multi-agent or batching needs. | Spike both Mctx and LightZero; keep env independent. |
| No baseline | MuZero bugs are indistinguishable from environment bugs. | Train PPO/heuristic baseline before MuZero. |

# 20. Acceptance gates
## Gate A: environment ready
- 100+ deterministic golden tests pass.
- Random agents can run at least 100k episodes without memory growth or nondeterministic crashes.
- Vectorized env produces same outcomes as single env for identical seeds/actions.
- Debug renderer can replay any stored trajectory.
- Heuristic agent beats random by a statistically meaningful margin.
## Gate B: baseline ready
- PPO or another policy baseline learns to beat random in 1v1 no-bonus CurvyTron.
- Evaluation is against fixed random seeds and held-out random seeds.
- Training curves, videos, and win-rate tables are logged automatically.
## Gate C: MuZero ready
- MCTS synthetic benchmark has acceptable steady-state throughput after compilation.
- Replay buffer stores correct observation/action/reward/search-policy/value-target sequences.
- 1v1 MuZero beats random and heuristic opponents at low simulations before adding complexity.
- GPU utilization/power and profiler traces show the bottleneck is understood, not guessed.
- Ablation exists: raw policy versus MCTS-improved policy.
# 21. Concrete first PRs
| PR | Title | Definition of done |
| --- | --- | --- |
| PR-001 | Repo skeleton + docs + dependency pins | Fresh repo builds locally; docs include decision log and fork plan. |
| PR-002 | CurvyTron reference import | CurvyTron fork is attached as subtree/submodule/snapshot; source hash recorded. |
| PR-003 | CurvyTron v0 simulator | Reset/step works for 1v1, no bonuses; deterministic seed tests pass. |
| PR-004 | Collision golden tests | Wall, own-trail, opponent, simultaneous death tests pass. |
| PR-005 | Debug renderer and trajectory replay | Any test trajectory can be rendered to a short video/gif/local frames. |
| PR-006 | Gymnasium/PettingZoo wrappers | Random-agent smoke tests run through wrappers. |
| PR-007 | Modal CPU/GPU smoke | Modal job prints env speed and GPU device for JAX/PyTorch images. |
| PR-008 | PPO baseline | Learns to beat random in the simplest 1v1 environment. |
| PR-009 | Mctx synthetic benchmark | Reports compile time, steady-state sims/sec, batch-size sweep. |
| PR-010 | Minimal 1v1 MuZero | Beats random or at least shows improving win rate; all metrics logged. |

# 22. Source index
The links below are the sources and references used to create this handoff. Some are official project docs; a few are issue threads or community reports used only as risk signals, not as authoritative documentation.
- **S1. MuZero paper:** https://arxiv.org/abs/1911.08265 - Original MuZero algorithm context.
- **S2. DeepMind MuZero blog:** https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/ - High-level MuZero explanation.
- **S3. Curvytron GitHub:** https://github.com/Curvytron/curvytron - Original open-source CurvyTron repo; MIT; web multiplayer game.
- **S4. CurvyTron v1 public site:** https://www.curvytron.com/ - Public game landing page.
- **S5. CurvyTron 2 public site:** https://curvytron2.com/ - Current public game page.
- **S6. CurvyTron 2 how-to-play:** https://curvytron2.com/how-to-play/ - Rules and bonus list.
- **S7. CurvyTron package.json raw:** https://raw.githubusercontent.com/Curvytron/curvytron/master/package.json - Old Node/gulp/websocket dependency evidence.
- **S8. CurvyTron bower.json raw:** https://raw.githubusercontent.com/Curvytron/curvytron/master/bower.json - Angular/Bower-era dependency evidence.
- **S9. CurvyTron issue #194:** https://github.com/Curvytron/curvytron/issues/194 - Head-collision scoring ambiguity.
- **S10. CurvyTron Docker fork:** https://github.com/cyrale/curvytron - Possible browser smoke-test path.
- **S11. Mctx GitHub:** https://github.com/google-deepmind/mctx - JAX-native batched MCTS/MuZero search.
- **S12. Mctx issue #4:** https://github.com/google-deepmind/mctx/issues/4 - End-to-end Gym example gap signal.
- **S13. JAX installation docs:** https://docs.jax.dev/en/latest/installation.html - CPU/GPU/TPU installation, CUDA wheels.
- **S14. Modal GPU docs:** https://modal.com/docs/guide/gpu - Modal GPU types and multi-GPU single-node notes.
- **S15. Modal CUDA docs:** https://modal.com/docs/guide/cuda - Modal NVIDIA driver/CUDA Driver API details.
- **S16. Modal GPU metrics:** https://modal.com/docs/guide/gpu-metrics - GPU utilization/power metric caveats.
- **S17. Modal Volumes docs:** https://modal.com/docs/guide/volumes - Checkpoint/replay storage considerations.
- **S18. Modal Dicts docs:** https://modal.com/docs/guide/dicts - Distributed key-value store and latency caveat.
- **S19. Modal Queues docs:** https://modal.com/docs/guide/queues - Distributed queue and latency caveat.
- **S20. Modal Queue reference:** https://modal.com/docs/reference/modal.Queue - Queue limits/lifetime reference.
- **S21. Modal CloudBucketMount docs:** https://modal.com/docs/guide/cloud-bucket-mounts - Cloud storage mounting.
- **S22. LightZero GitHub:** https://github.com/opendilab/LightZero - PyTorch MCTS+RL toolkit.
- **S23. LightZero custom env docs:** https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html - Custom environment wrapper requirements.
- **S24. LightZero issue #219:** https://github.com/opendilab/LightZero/issues/219 - Custom env docs gap signal.
- **S25. muzero-general GitHub:** https://github.com/werner-duvaud/muzero-general - Educational MuZero implementation; further improvements list.
- **S26. muzero-general issue #159:** https://github.com/werner-duvaud/muzero-general/issues/159 - Independent report of GPU self-play bottleneck and batching need.
- **S27. PettingZoo Parallel API:** https://pettingzoo.farama.org/api/parallel/ - Simultaneous multi-agent env API target.
- **S28. Gymnasium Env API:** https://gymnasium.farama.org/api/env/ - Single-agent wrapper API target.
- **S29. Gymnasium VectorEnv:** https://gymnasium.farama.org/api/vector/ - Vectorized env API reference.
- **S30. EfficientZero paper:** https://arxiv.org/abs/2111.00210 - Sample-efficient MuZero-family variant.
- **S31. Muax:** https://github.com/bwfbowen/muax - JAX/Mctx Gym-style helper library to inspect.
- **S32. Pgx:** https://github.com/sotetsuk/pgx - JAX-native game simulator reference for batched games.
- **S33. Gymnax:** https://github.com/RobertTLange/gymnax - JAX-native env reference for vectorized RL.
- **S34. a0-jax:** https://github.com/NTT123/a0-jax - JAX/Mctx AlphaZero reference.
- **S35. turbozero:** https://github.com/lowrollr/turbozero - JAX/vectorized AlphaZero-style reference.
# 23. Final recommendation
Create a fresh main ML repo. Fork CurvyTron as a reference/demo/golden-test repo. Build the environment from scratch in the ML repo, with explicit rules, deterministic tests, and throughput benchmarks. Run two library spikes: JAX/Mctx and PyTorch/LightZero. Default to JAX/Mctx if the Modal batched-search benchmark is good, because it directly addresses the likely bottleneck. Use Modal GPUs, not TPUs. Keep Modal Queues/Dicts out of the inner loop. Ship v0 as 1v1 no-bonus CurvyTron with PPO baseline before starting serious MuZero training.
