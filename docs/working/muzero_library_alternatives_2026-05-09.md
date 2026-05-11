# MuZero Library Alternatives Critic - 2026-05-09

Role: MuZero-specific alternatives critic. Scope: compare whether CurvyZero
should keep using LightZero for MuZero replication/control or switch to another
MuZero implementation. No code changes. No pytest.

## Practical Recommendation

Keep LightZero as the external MuZero replication/control lane for now, but
stop treating LightZero dummy Pong scale-up as the path to production until the
checkpoint/eval blockers are resolved.

Do not switch the immediate lane to MiniZero, muzero-general, EfficientZero, or
a random Gumbel/Stochastic MuZero repo. None is a better near-term control than
the LightZero path we have already run on Modal.

The likely long-term training lane is not another full external framework. It
is:

1. finish the repo-native PPO/env learnability gate;
2. use PPO results to validate observation, action, reward, reset, seed, and
   scorecard contracts;
3. only then write the smallest project-owned MuZero loop, preferably reusing
   Mctx for batched Gumbel MuZero search instead of writing MCTS ourselves.

So the answer is:

- keep LightZero for replication/control;
- pause LightZero as a source of learning claims;
- do not switch to another external MuZero framework;
- prepare to switch the production/control surface to project-owned minimal
  MuZero after PPO proves the environment.

## Why This Is Not A Simple "LightZero Failed"

LightZero has already cleared important plumbing gates in this repo:

- stock visual Atari Pong `train_muzero` ran in Modal, collected/evaluated
  through ALE, wrote checkpoints, and returned `ok: true`;
- custom dummy Pong LightZero runs produced real `.pth.tar` checkpoints and
  artifacts;
- the current MCTS loader smoke can strict-load a dummy Pong checkpoint and run
  a LightZero eval-mode forward call.

Those are meaningful replication/control wins. They are not learning-quality
wins.

The current blocker is trust: independent dummy Pong scorecards show degenerate
or weak action use even when trainer-side LightZero telemetry looks strong. The
512/8 checkpoint was effectively constant `up`. The 4096/64 run changed to
`up` plus `stay`, still selected no `down` in the reported scorecard rows, and
still did not beat random or scripted baselines. That means the external
trainer is useful as a control surface, but not yet as a reliable training
claim.

## Comparison Matrix

| Option | What it gives us | Fit for current CurvyZero task | Main blockers | Recommendation |
| --- | --- | --- | --- | --- |
| LightZero | Complete PyTorch MuZero-family framework: env managers, replay, learner, MCTS, logs, checkpoints, stock Atari/Pong configs, MuZero/EfficientZero/Gumbel/Stochastic variants. | Best immediate external control because it has already run in our Modal stack and stock visual Pong now has a positive trainer smoke. | Dummy Pong independent eval does not validate learning; DI-engine/LightZero semantics can hide seed/eval details; custom env path needs sidecar telemetry; trainer-side eval can overstate quality. | Keep for replication/control. Do not scale for quality until MCTS scorecards and action-diversity gates pass. |
| MiniZero | Serious AlphaZero/MuZero/Gumbel AlphaZero/Gumbel MuZero framework with server, self-play workers, optimization worker, data storage, and Atari 57-game support. | Best alternative full framework on paper, especially if we wanted a C++/containerized zero-style trainer comparison. | New build/runtime stack; no local Modal proof; custom Curvy simultaneous env would need a new MiniZero game integration; likely more invasive than fixing LightZero eval gates. | Do not switch now. Consider only as a second external replication benchmark after LightZero scorecards are honest. |
| muzero-general | Readable educational PyTorch MuZero trainer with game-file adaptation, Ray, checkpoints, TensorBoard, single/two-player examples, Gym/Atari Breakout. | Useful reference for replay/target/model structure. | README explicitly frames it as educational; Batch MCTS and >2-player support are listed as future improvements; Atari example is Breakout, not Pong; older Ray/threading stack is a distraction. | Use as pseudocode/reference only. Do not adopt as backbone. |
| EfficientZero repo | Atari-focused EfficientZero implementation from the NeurIPS 2021 paper, with sample-efficiency machinery. | Good paper/repo reference if the question becomes Atari sample efficiency under fixed data budgets. | Not plain MuZero; requires C++/Cython tree build, GCC, Ray, and multi-worker assumptions; default quick start is Breakout; too heavy for dummy Pong or Curvy control debugging. | Do not switch. Reference selectively for sample-efficiency ideas later. |
| Standalone Gumbel/Stochastic MuZero repos | Some repos implement newer variants or offline/stochastic extensions. | Mostly research reading. Mctx/LightZero/MiniZero already cover the useful Gumbel surface more credibly. | Small/stale projects, GPL risk in at least one Stochastic MuZero repo, old dependency pins, no local proof, and usually more framework surface than we want. | Avoid as dependencies. Use primary papers/Mctx/LightZero code paths instead. |
| Mctx-owned implementation | JAX-native batched search library with `muzero_policy` and `gumbel_muzero_policy`; action weights are directly usable as policy targets. | Best long-term search dependency if we own replay, targets, actor loop, checkpoints, eval, and metadata. | It is not a trainer. We must write env batching, model, replay, target construction, optimizer, checkpointing, scorecards, and Modal run management. | Use later for project-owned minimal MuZero after PPO/env gates. Do not call Mctx-only benchmarks "training." |
| Write minimal MuZero after PPO | Maximum control over Curvy simultaneous actions, replay metadata, scorecards, action traces, reset/final-observation handling, and checkpoint format. | Best production fit once PPO proves the env/reward/scorecard contract. | More implementation work; easy to build an unvalidated trainer; search may be slower than PPO at equal wall clock; needs tight scope. | Recommended next ownership path after PPO, not before. Reuse Mctx search rather than hand-rolling MCTS. |

## Candidate Notes

### LightZero

Upstream status is strong enough for a control lane. The LightZero README was
updated for v0.2.0 on 2026-03-11 and describes a PyTorch MCTS+RL toolkit. It
lists MuZero, Sampled MuZero, Stochastic MuZero, EfficientZero, Gumbel MuZero,
ReZero, and UniZero, with Atari support across several variants. It also has
quick-start commands for CartPole and Pong.

Local status is mixed:

- positive: stock visual Pong tiny train smoke passed in Modal with
  `LightZero==0.2.0`;
- positive: custom dummy Pong produced checkpoints and artifact manifests;
- negative: independent dummy Pong eval does not corroborate trainer-side
  wins;
- negative: more dummy Pong training alone did not fix action collapse.

Decision: keep LightZero, but demote it from "learning answer" to "external
MuZero control until scorecard trust is repaired."

Clear LightZero blockers:

- full independent MCTS/eval-mode scorecard across checkpoints and opponents;
- action histogram/entropy gate must show all legal actions are reachable when
  appropriate;
- trainer-side evaluator results must match independent scorecard directionally;
- seed/reset/reward/action trace metadata must remain visible outside
  LightZero internals;
- support/value/reward scale and target semantics must be audited before
  larger runs;
- any CurvyTron wrapper must preserve simultaneous-player terminal data, not
  collapse it into an opaque single-agent story without sidecar traces.

### MiniZero

MiniZero is the strongest external alternative if the only question is "which
other complete MuZero-family framework looks serious?" Its README says it
supports AlphaZero, MuZero, Gumbel AlphaZero, and Gumbel MuZero; supports Go,
Othello, Hex, TicTacToe, and Atari 57 games; and uses a server/self-play
worker/optimization worker/data-storage architecture. It even gives Atari
MuZero and Gumbel MuZero training examples for Ms. Pac-Man.

That said, it is not the cheapest replacement for LightZero. It would introduce
a fresh container/build/runtime stack, and our blocker is not "LightZero cannot
start MuZero." Our blocker is independent eval trust and Curvy semantics.
MiniZero would still need a custom game/env integration and a new artifact
bridge before it could answer the same question.

Decision: do not switch to MiniZero now. It is a possible later second
replication benchmark if we need an external cross-check.

### muzero-general

muzero-general remains valuable as readable MuZero scaffolding. The README says
it is a commented, documented implementation designed to be adaptable by adding
a game file, and primarily educational. It includes single/two-player support,
Ray-based asynchronous/cluster execution, checkpoints, TensorBoard, Gym and
Atari examples.

The same README lists Batch MCTS and support for more than two-player games as
future improvements, and the implemented Atari example called out in the README
is Breakout. That makes it a poor fit for a Curvy simultaneous-control
production lane.

Decision: reference only.

### EfficientZero And Gumbel MuZero Repos

EfficientZero is a real Atari/sample-efficiency implementation, but it is not a
minimal MuZero replication backbone. The upstream README requires building
C++/Cython external packages with GCC and uses Ray. Its quick start is
BreakoutNoFrameskip-v4 with explicit GPU/CPU actor flags. That is the wrong
amount of machinery for our current problem.

For Gumbel MuZero specifically, the strongest practical surfaces are already
LightZero, MiniZero, and Mctx. I do not recommend adopting smaller standalone
Gumbel/Stochastic repositories as dependencies. At least one visible
Stochastic MuZero repo is GPL-3.0, pinned around older Torch/Ray/Gymnasium
versions, and advertises a scheduled PyTorch 2.1 update rather than a current
production-ready surface.

Decision: use EfficientZero/Gumbel repos as reading material, not as a switch
target.

### Mctx-Owned MuZero

Mctx is the cleanest search dependency. Its README says it is a JAX-native MCTS
library for AlphaZero, MuZero, and Gumbel MuZero, supports JIT compilation, and
operates on batched inputs. It exposes `muzero_policy` and
`gumbel_muzero_policy`; the returned action weights are explicitly described as
targets usable to train policy probabilities, and the README recommends
`gumbel_muzero_policy`.

The catch is the whole point: Mctx is search, not training. A project-owned
Mctx path must supply everything around search:

- representation/dynamics/prediction model;
- replay rows and target builder;
- actor loop and env batching;
- optimizer/update loop;
- checkpoints and checkpoint eval;
- seed/rules/observation/action trace manifests;
- scorecards and held-out opponent matrix.

Decision: Mctx is the right dependency for an owned MuZero, but only after PPO
proves the environment and after the owned implementation is capped to a tiny
ego-vs-random scope.

## Blockers Before Any Larger MuZero Claim

These block all candidates, not just LightZero:

- PPO or another simple learned baseline must show that the environment,
  observation, reward, action mask, reset/autoreset, and scorecard can support
  learning.
- Independent eval must be the source of truth; trainer-side collector/evaluator
  returns are not enough.
- Scorecards must include action histograms, entropy/logit spread where
  available, win/loss/truncation, survival/steps, and opponent IDs.
- Checkpoint load paths must be strict enough to catch config/model mismatches.
- The eval horizon, training budget, and environment max-episode length must be
  separate knobs.
- No framework may hide seed, rules hash, terminal player rewards, final
  observations, or joint action traces.
- Search must beat or complement policy-only inference under equal wall-clock
  budget before MuZero earns more complexity.

## Decision Gates

Keep using LightZero if the next diagnostic pass shows:

- `iteration_0`, intermediate, latest, and `ckpt_best` all load through the same
  strict config path;
- LightZero eval-mode/MCTS scorecards are not action-collapsed;
- independent scorecards match trainer-side trends across at least a small seed
  sweep;
- all metadata needed by CurvyZero is present in sidecar artifacts.

Demote or retire LightZero as an active training lane if:

- MCTS eval remains action-collapsed after support/target/seed fixes;
- trainer-side results stay positive while independent held-out scorecards stay
  flat or negative;
- the custom env wrapper needs invasive DI-engine/LightZero forks;
- artifact/seed/action trace recovery remains partial.

Start project-owned Mctx MuZero only when:

- repo-native PPO has a real held-out learning signal;
- replay/export contracts are stable;
- a tiny owned MuZero spike can fit in one quarantined directory/module;
- the first target is ego-vs-random with small observations, `A=3`, short
  unrolls, and low simulation counts.

## Bottom Line

LightZero is still the right external MuZero control. It is not yet the right
source of CurvyZero learning truth.

MiniZero is the only external switch candidate that looks strategically
serious, but switching now would spend effort on another integration before
fixing the core trust problem. muzero-general and EfficientZero are references,
not backbones. Mctx is the best owned-search dependency, but adopting it means
writing the trainer ourselves.

The practical path is therefore: repair LightZero scorecard trust for
replication/control, finish the repo-native PPO learnability gate, then build a
minimal project-owned Mctx MuZero only if PPO says the environment is worth
search.

## Sources Checked

Current upstream sources:

- LightZero GitHub README:
  https://github.com/opendilab/LightZero
- MiniZero GitHub README:
  https://github.com/rlglab/minizero
- muzero-general GitHub README:
  https://github.com/werner-duvaud/muzero-general
- EfficientZero GitHub README:
  https://github.com/YeWR/EfficientZero
- Mctx GitHub README:
  https://github.com/google-deepmind/mctx
- Stochastic MuZero example repo, used only as a cautionary standalone-repo
  check:
  https://github.com/DHDev0/Stochastic-muzero

Local context:

- `docs/research/muzero_repo_baseline_options.md`
- `docs/decisions/0005-main-pong-repository-library-choice.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-tiny-train-smoke.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-longer-run.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md`
- `docs/working/lightzero_pong_scale_performance_critique_2026-05-09.md`
- `docs/working/repo_native_ppo_learner_boundary_2026-05-09.md`
- `docs/working/repo_native_actor_loop_next_step_2026-05-09.md`
