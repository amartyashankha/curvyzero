# MuZero Implementation Reality Check - 2026-05-09

Owner: MuZero implementation reality checker.

Scope: simple-language answer for CurvyZero. No code, no pytest. This note uses
local docs/source evidence plus public primary repo/docs pages.

## Short Answer

Yes. LightZero really does contain a full MuZero implementation in the practical
sense: it has neural network models, MuZero policy/training code, MCTS,
collectors, evaluators, replay/buffer machinery, configs, examples, logging,
and checkpoints.

But "LightZero contains MuZero" is not the same as "LightZero is a finished
CurvyZero trainer." LightZero is a full external MuZero-family framework. It is
not a native CurvyTron simultaneous multi-player `B x P` training system, and it
has not yet produced trustworthy CurvyZero learning evidence under independent
scorecards.

The best practical read is:

- LightZero is real MuZero machinery.
- It is the best external MuZero control lane we have already made run.
- It should stay contained until it proves reliable policy quality and
  CurvyZero metadata preservation.
- It should not silently become the CurvyTron backbone just because it is the
  most complete public MuZero package we tried.

## What LightZero Contains

LightZero is not just pseudocode and not just a search library.

The public LightZero README describes it as a PyTorch toolkit combining MCTS
and reinforcement learning. It names the three core modules as model, policy,
and MCTS. The policy handles learning, collecting, and evaluation; MCTS is part
of the policy/search path; and LightZero ships a supported-algorithm table with
MuZero, Sampled MuZero, Stochastic MuZero, EfficientZero, Gumbel MuZero, ReZero,
and UniZero across environment families including Atari and CartPole.

In this repo, local runs confirm that is not just a README claim:

- Stock LightZero CartPole MuZero ran on Modal and returned a real
  `MuZeroPolicy`, learner/evaluator logs, TensorBoard files, and `.pth.tar`
  checkpoints.
- Custom dummy Pong called LightZero's real `train_muzero`, not a CurvyZero
  imitation loop.
- The dummy Pong smoke wrote LightZero checkpoints including `ckpt_best`,
  `iteration_0`, and later `iteration_8`.
- The checkpoint tensors included the expected MuZero pieces:
  `representation_network`, `dynamics_network`, `prediction_network`, policy
  head, value head, and reward head.
- The custom env path used the LightZero observation contract:
  `observation`, `action_mask`, and `to_play`.

So the honest answer is: LightZero contains a complete MuZero-family trainer.
It is not an empty wrapper around DI-engine and it is not only MCTS.

## What LightZero Does Not Give CurvyZero For Free

LightZero does not give us a finished CurvyZero solution.

It does not natively own our desired CurvyTron shape:

```text
observations[B, P, ...]
actions[B, P]
rewards[B, P]
terminal info per player
simultaneous actions each tick
```

The first working custom dummy Pong adapter made LightZero control one ego
player. The opponent lived inside the environment wrapper as random, scripted,
or frozen-checkpoint behavior. That is acceptable for a contained smoke, but it
is not full simultaneous CurvyTron self-play.

It also does not give us trust by default. Local docs show several adoption
risks:

- The correct evaluator path matters. Generic DI-engine evaluator plumbing can
  call MuZero policies incorrectly unless the MuZero-specific `action_mask`,
  `to_play`, and `timestep` contract is respected.
- Checkpoints are config-sensitive. Stock/pretrained Pong checkpoints can be
  incompatible with current config shapes, such as older `96x96` visual
  checkpoints versus current `64x64` stock config paths.
- Config summaries can be misleading unless we verify compiled LightZero
  internals such as support scale and value/reward support size.
- Trainer-side returns can look good while independent held-out scorecards show
  weak or collapsed action use.
- LightZero policy targets are MCTS root visit distributions, not raw executed
  exploratory actions. So collection can execute all actions while the learned
  policy still collapses if root visits assign little target mass to the useful
  action.

For CurvyZero, independent scorecards and target sidecars are not optional.

## How It Compares To muzero-general

`muzero-general` also contains a real MuZero implementation, but it is a
different kind of project.

Its README calls it a commented, documented implementation based on the
DeepMind paper and pseudocode, designed to be adaptable by adding a game file.
It has PyTorch networks, Ray-based asynchronous/cluster execution,
checkpoints, TensorBoard, single/two-player mode, examples, and pretrained
weights.

The important qualifier is that the README also says it is primarily for
educational purpose. Its own "further improvements" list includes Batch MCTS
and support for more than two-player games. Its listed Atari example is
Breakout, not a current Pong-first control lane.

Practical comparison:

| Question | LightZero | muzero-general |
| --- | --- | --- |
| Full MuZero trainer? | Yes. | Yes, educational. |
| Maintained/current signal? | Better in our notes; README updated for v0.2.0 and broader algorithm table. | Useful but older/smaller; local notes saw 132 commits. |
| Variants | MuZero plus Gumbel, Stochastic, EfficientZero, Sampled, ReZero, UniZero. | Mostly plain MuZero. |
| Public custom-env story | DI-engine `BaseEnv` / LightZero observation dict. | Add a game file/config. |
| CurvyTron fit | Medium/weak: ego wrapper hides simultaneous play. | Weak/medium: educational two-player/simple-game assumptions, no native `B x P`. |
| Best use for us | Contained external MuZero control lane. | Readable pseudocode/reference for replay and targets. |

So if the question is "which public repo looks more like a maintained
MuZero-family framework?", LightZero is stronger. If the question is "which is
easier to read to understand MuZero?", `muzero-general` may be clearer. Neither
should own CurvyZero's simulator contract.

## How It Compares To Other Public MuZero Repos

MiniZero is a serious full-system alternative. Its README says it supports
AlphaZero, MuZero, Gumbel AlphaZero, and Gumbel MuZero, and supports Go,
Othello, Hex, TicTacToe, and Atari 57 games. It also has an explicit system
architecture: server, self-play workers, optimization worker, and data storage.
That makes it useful as a systems reference, but adopting it would mean
adopting another full distributed game/training stack.

Mctx is not a full MuZero trainer. It is a JAX-native MCTS/search library. Its
README says it implements search algorithms such as AlphaZero, MuZero, and
Gumbel MuZero, supports JIT compilation, and operates on batches in parallel.
That is exactly why it is attractive later for a project-owned CurvyZero MuZero
implementation. But it does not supply replay, actors, learner updates,
checkpointing, env adapters, scorecards, or Modal run management.

EfficientZero is a real Atari/sample-efficiency implementation, but it is not
the plainest MuZero baseline for CurvyZero. Its README requires C++/Cython
tree build steps, GCC, Ray, and Atari-oriented commands such as Breakout. It is
better treated as sample-efficiency reading than as our backbone.

Simple role assignment:

| Repo/library | Reality | Best use for CurvyZero |
| --- | --- | --- |
| LightZero | Full MuZero-family PyTorch framework. | External control lane and contained custom-env experiments. |
| muzero-general | Full educational MuZero implementation. | Pseudocode/reference for replay, targets, file shape. |
| MiniZero | Full AlphaZero/MuZero/Gumbel system. | Systems reference, possible later external benchmark. |
| Mctx | Search only, not trainer. | Future project-owned search substrate. |
| EfficientZero | Full EfficientZero/Atari research code. | Read selectively for sample-efficiency/reanalyze ideas. |

## Source Of The Confusion

The confusion comes from overloading the word "implementation."

There are at least five different claims people keep mixing together:

1. "This repo has MuZero code."
2. "This repo has a complete MuZero trainer."
3. "This repo has a MuZero trainer that can run in our Modal image."
4. "This repo can train on our custom dummy Pong wrapper."
5. "This repo is the right long-term CurvyTron backbone."

For LightZero, claims 1-4 are now basically true in bounded form. Claim 5 is
not proven.

There is a second confusion: trainer-side activity is not the same as learned
quality. LightZero can collect episodes, write checkpoints, and report
positive evaluator rows while independent CurvyZero scorecards still show action
collapse or no real improvement. That is not "fake MuZero"; it is a real
trainer producing untrusted or undertrained policy quality for our custom task.

There is a third confusion: Mctx is sometimes compared to LightZero as if both
were full frameworks. They are not. LightZero supplies a trainer. Mctx supplies
search. If we choose Mctx, we are choosing to write the surrounding trainer
ourselves.

## Practical Implications For CurvyZero

Keep LightZero, but keep it in a box.

Use it for:

- stock Pong/ALE reproduction and evaluator parity;
- custom dummy Pong MuZero control runs;
- checkpoint/load/scorecard infrastructure practice;
- target-sidecar audits that tell us what MuZero is actually training toward.

Do not use it as:

- proof that CurvyTron self-play is solved;
- proof that dummy Pong learned just because `train_muzero` ran;
- the owner of the simulator interface;
- a reason to abandon repo-owned `B x P` environment and eval contracts.

The safe CurvyZero posture is:

1. Keep the simulator and serious multi-player environment contract repo-owned.
2. Keep LightZero as the external MuZero control lane.
3. Use independent scorecards as the source of truth, not trainer-side logs.
4. Use `muzero-general`, MiniZero, EfficientZero, and Mctx as references with
   sharply different roles.
5. If CurvyZero eventually owns MuZero, prefer a small project-owned trainer
   around Mctx search after PPO/env learnability gates are solid.

## Bottom Line

LightZero really contains a full MuZero implementation.

The practical mistake would be treating that fact as the end of the CurvyZero
training architecture question. It is only the start: LightZero gives us real
MuZero machinery to test against, but CurvyZero still has to own semantics,
artifacts, independent eval, and eventually the simultaneous multi-player
training shape.

## Sources

Local docs:

- `docs/research/lightzero_feature_fit_for_curvyzero.md`
- `docs/research/muzero_repo_baseline_options.md`
- `docs/research/muzero_reference_examples.md`
- `docs/research/muzero_framework_vs_project_owned.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-longer-run.md`
- `docs/working/lightzero_muzero_target_semantics_2026-05-09.md`
- `docs/working/lightzero_local_adoption_risk_2026-05-09.md`
- `docs/working/muzero_library_alternatives_2026-05-09.md`
- `docs/working/training_framework_alternatives_2026-05-09.md`

Primary public sources:

- LightZero GitHub README:
  https://github.com/opendilab/LightZero
- LightZero docs, custom environments:
  https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
- muzero-general GitHub README:
  https://github.com/werner-duvaud/muzero-general
- MiniZero GitHub README:
  https://github.com/rlglab/minizero
- Mctx GitHub README:
  https://github.com/google-deepmind/mctx
- EfficientZero GitHub README:
  https://github.com/YeWR/EfficientZero
