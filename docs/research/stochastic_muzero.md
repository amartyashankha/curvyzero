# Stochastic MuZero For CurvyTron-Style Games

Status: Research note
Date: 2026-05-09

## Short Answer

Do not implement Stochastic MuZero now.

For the first project-owned MuZero/Mctx trainer, use standard deterministic
MuZero or Gumbel MuZero against a deterministic ruleset: no bonuses, no random
trail holes, fixed action space, explicit seeds, and policy-only opponents.
That already matches the current `curvyzero-v0`, dummy Pong, and Modal/Mctx
architecture notes.

Record Stochastic MuZero as a later branch for rulesets where chance is part of
the game being planned over: random item spawns, random boost/hazard effects,
random trail gaps, hidden/noisy transitions, or opponent actions that cannot be
cleanly represented by the observation and opponent policy metadata.

Plain version: normal MuZero imagines "if I take this action, here is the next
latent state." Stochastic MuZero imagines "if I take this action, the game may
roll one of several outcomes, and I should plan over that distribution."

Training-time noise for robustness is a separate question. Epsilon exploration,
sticky actions, frozen controls, noisy observations, image augmentation, and
domain randomization usually do not require Stochastic MuZero. Start by treating
them as data collection, wrapper, augmentation, or eval-profile choices with
explicit seeds and metadata.

See [robustness_randomization_for_muzero.md](robustness_randomization_for_muzero.md)
for the focused note on sticky actions, action repeat, observation noise,
augmentation, and domain randomization.

## What Standard MuZero Assumes

Standard MuZero learns three functions:

- `representation`: observation history to latent state;
- `dynamics`: latent state plus action to predicted reward and next latent
  state;
- `prediction`: latent state to policy and value.

The important simplifying assumption is in `dynamics`: it is a deterministic
function of the current latent state and chosen action. The original MuZero
paper says the dynamics function is represented deterministically and leaves
stochastic transitions for future work.

That does not mean the real environment must be perfectly simple. Standard
MuZero can be trained from noisy data and may learn average outcomes. The
problem is planning: if one action can lead to several sharply different
futures, a single predicted next latent state can blur them together. Search may
then reason over an average future that is not a real future.

For CurvyZero v0 this is acceptable because the design deliberately removes
chance:

- no bonuses or powerups;
- solid trails;
- deterministic reset for a given seed;
- deterministic transition for a given state/config/action sequence;
- fixed `A=3` ego action space;
- explicit replay metadata for rules, observations, rewards, search, and
  opponent policy versions.

## What Stochastic MuZero Adds

Stochastic MuZero factors one transition into two parts:

```text
state --agent action--> afterstate --chance outcome--> next state
```

It also changes search from one node type to two alternating node types:

- decision nodes, where the agent chooses an action;
- chance nodes, where the model samples or evaluates possible chance outcomes.

The learned model has extra machinery beyond standard MuZero. In the paper this
includes afterstate dynamics, afterstate prediction, a learned discrete chance
outcome model, and a chance loss. In implementation terms, Mctx exposes this as
`stochastic_muzero_policy` with separate `decision_recurrent_fn` and
`chance_recurrent_fn` callbacks instead of one deterministic recurrent
function.

This is the right tool when the tree needs to branch over events the agent does
not control.

## When It Matters For CurvyTron

Use standard MuZero when randomness is outside the search problem:

- item, boost, hazard, or spawn layouts are fixed by the episode seed and fully
  observed before decisions;
- source random streams are recorded and replayed for deterministic training
  fixtures;
- noisy transitions are only domain variation across episodes, not random
  outcomes inside a single planned step;
- opponents are modeled as fixed policy-only agents and their stochasticity can
  be treated as part of the replay distribution for a first smoke.

Revisit Stochastic MuZero when any of these become true:

- Bonuses/items spawn randomly during play and can change the best action now.
- Trail gaps or hazards open/close randomly after an action.
- Boost pickup effects include random type, duration, direction, or placement
  that is not known at decision time.
- Observations are partially aliased, so the same ego observation can hide
  several materially different world states.
- Opponent actions need explicit branches rather than being folded into a
  learned ego-only transition.
- Deterministic ego-only dynamics produces unstable search: good root values in
  training but bad rollouts, high disagreement between model value and realised
  returns, or search preferring actions that only work under an averaged future.

For multiplayer, Stochastic MuZero is one possible answer to opponent
uncertainty, but it is not the only one. Other reserved options are
opponent-conditioned dynamics, focal-ego search with policy-only opponents,
searching both 1v1 players as independent ego rows, or bounded joint-action
experiments. Joint action grows as `3^N`, so it is not the v0 answer either.

## Training-Time Robustness Noise

Robustness noise is randomness we add on purpose so a policy does not become
fragile. It can make training data stochastic without making Stochastic MuZero
the right algorithm.

Use this split:

| Noise kind | First home | Needs chance nodes now? |
| --- | --- | --- |
| Epsilon action exploration | Collector/search policy config. | No. |
| Root Dirichlet noise or visit-count temperature | MuZero search config for self-play. | No. |
| Sticky actions or frozen controls | Environment wrapper and eval profile. | No, unless planning must explicitly reason about the stickiness. |
| Fixed action repeat | Ruleset/action wrapper; include recent actions in observation if needed. | No. |
| Erratic controls, dropped inputs, delayed turn commands | Environment wrapper with explicit random stream. | Usually no. |
| Image noise, color jitter, crop/scale jitter | Observation wrapper or replay-time augmentation. | No. |
| Domain randomization of arena, speed, turn rate, trail width | Named variation profile sampled at reset. | No if sampled config is observed/logged for the episode. |
| Random in-game items, boosts, hazards, trail gaps | Ruleset event system. | Later maybe, if search must plan over future event branches. |

The key distinction is whether randomness is part of the training distribution
or part of the future that search must branch over.

Examples:

- Epsilon exploration says "sometimes collect a deliberately non-greedy action."
  Replay stores the action that actually happened. The model learns from that
  trajectory like any other trajectory.
- Sticky-action eval says "the environment may repeat the last action with some
  probability." This is useful to test robustness against open-loop memorization
  and brittle control. It can live as a wrapper and scorecard row.
- Observation noise says "the raster/rays are slightly perturbed." The real
  state transition has not changed; the representation should become less
  brittle.
- Domain randomization says "sample a game variant at reset." If the sampled
  parameters are logged and, when useful, exposed in the observation, each
  episode can still be deterministic after reset.

Stochastic MuZero becomes more plausible only when the agent needs to compare
actions by looking ahead through unresolved future chance events. A sticky action
wrapper by itself does not prove that. A policy may become robust enough by
training and evaluating under sticky controls, especially when observations
include recent frames/actions.

## Practical CurvyZero Policy

For the next project-owned MuZero/Mctx trainer:

- Keep canonical train/eval deterministic first.
- Use normal MuZero exploration tools for self-play: root noise, visit-count
  temperature, and optionally bounded epsilon/random collection.
- Add robustness noise as named profiles, not as anonymous script flags.
- Store the intended action and the executed action when wrappers can alter
  controls.
- Store `action_repeat`, sticky/frozen-control probability, control-delay
  settings, observation-noise profile, augmentation profile, variation profile,
  and all RNG seeds in replay and eval artifacts.
- Keep canonical eval clean. Add separate sticky-control, noisy-observation, and
  domain-randomized scorecard rows.

For observations:

- Image augmentation can happen at replay sampling time if it does not need to
  be replayed as an environment fact.
- Observation noise used during environment rollout should be seeded and logged
  because it changes the policy's actual input.
- If action effects can be delayed or sticky, include enough history in the
  observation. MuZero's Atari setup used recent frames and recent actions because
  one Atari action may not have an immediate visible effect.

For controls:

- Fixed action repeat is a ruleset/control-frequency choice. It changes the
  decision horizon and should be in the action schema or rules hash.
- Sticky actions, frozen controls, and erratic controls are robustness wrappers
  unless they become the main target game.
- If a wrapper changes the chosen action, train reward/value against the actual
  transition and log both actions. Decide separately whether the policy target is
  the chosen action, the executed action, or the search distribution at the
  pre-wrapper decision point.

## When Robustness Noise Would Escalate

Do not jump straight from "we added noisy training" to Stochastic MuZero.

Escalate only after a deterministic/Gumbel MuZero baseline exists and one of
these failures appears on heldout robustness evals:

- Search repeatedly overvalues actions that are good only if a sticky/frozen
  control outcome breaks favorably.
- The learned deterministic dynamics collapses distinct chance outcomes into an
  average latent state and that average causes bad action choices.
- Increasing history, logging/exposing wrapper state, or training with the same
  noise distribution does not fix the failure.
- The target ruleset itself has future chance events with strategic value, such
  as random item spawn timing, hazard activation, or boost type.

Even then, compare smaller fixes first: add recent action/control-state history,
condition dynamics on sampled variation parameters, train with sticky wrappers,
or evaluate with more seeds. Stochastic MuZero is a later algorithm branch when
the tree truly needs decision/chance structure.

## Recommendation

Keep the current order:

1. Build the first project-owned one-container MuZero/Mctx trainer with
   deterministic or Gumbel MuZero.
2. Use deterministic rulesets for Pong and `curvyzero-v0`.
3. Keep RNG explicit in environment reset, replay, search, and eval metadata.
4. Log item/boost/hazard/trail-gap events even before training on them.
5. Add a later `stochastic-muzero` branch only after a deterministic MuZero
   baseline exists and a stochastic ruleset shows a measured failure.
6. Treat robustness noise as wrappers, augmentation, exploration, and eval
   profiles first; do not make it a separate algorithm.

The practical reason is simple: CurvyZero does not yet have a project-owned
MuZero train loop. Adding afterstates, chance codes, chance-node search, and
new targets before the deterministic trainer exists would increase surface area
without answering the current learnability question.

## Reward Shaping Impact

Stochastic MuZero does not change the reward recommendation.

Keep environment and eval reward tied to the real game outcome:

- Pong: score delta, win/loss, score margin, timeout/truncation rate.
- CurvyTron: paired-seat win/loss/draw or rank payoff, terminal causes, and
  timeout/truncation rate.

Training-only shaping can still exist, but it must stay separate from eval and
must not reward stalling. The same warning applies with or without stochastic
search: if a survival bonus makes timeouts attractive, the algorithm may
optimize the wrong game.

For stochastic rulesets, shaping gets harder to read because a single action may
lead to several legitimate outcomes. If we add shaping later, log it as an
auxiliary target and compare against native-outcome eval. Do not use shaping to
hide a bad observation, missing item state, or broken random-stream contract.

## Eval Impact

The immediate eval protocol does not change for deterministic v0. Keep fixed
seed splits, baselines, checkpoint rows, action histograms, terminal causes,
survival/loss-delay telemetry for Pong, and paired seats for multiplayer.

For stochastic rulesets, add these requirements before making quality claims:

- Record environment seed, random-stream policy, and chance-event log.
- For robustness wrappers, record intended action, executed action,
  sticky/frozen-control probability, action-repeat config, observation-noise
  profile, and augmentation/variation profile.
- Separate train, monitor, selection, heldout, and debug seed/chance splits.
- Run enough episodes that random item/gap/hazard outcomes are not mistaken for
  policy improvement.
- Report confidence intervals or bootstrap intervals for win/rank metrics once
  claims matter.
- Include deterministic-ablation rows, such as fixed item spawn tape or fixed
  trail-gap tape, so regressions can be reproduced.
- Compare deterministic MuZero and Stochastic MuZero on the same stochastic
  ruleset before accepting the added complexity.

Expected promotion signal for a future stochastic branch: same native eval
metric, same budget, same seed/chance split, and Stochastic MuZero improves
heldout outcome or stability over deterministic MuZero, not merely training
loss.

## Implementation Notes For Later

If this branch becomes real, expect at least these changes:

- Replay rows store chance outcomes, chance logits/targets, random-stream ids,
  and event logs for item/boost/hazard/trail-gap transitions.
- Robustness-wrapper replay rows store wrapper settings and, for action noise,
  both requested and executed actions.
- The model grows afterstate dynamics and chance recurrent paths.
- Mctx integration switches from `gumbel_muzero_policy`/`muzero_policy` to
  `stochastic_muzero_policy` for that branch.
- Search configs add `num_chance_outcomes` or equivalent fixed chance-code
  shape. Treat it like action count: changing it creates a new compiled profile.
- Eval artifacts identify whether a policy used deterministic search or
  stochastic search.

Keep these changes out of the deterministic trainer until the branch has a
measured reason to exist.

## Sources

- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with
  a Learned Model" (MuZero): https://arxiv.org/pdf/1911.08265
- Nature version of MuZero: https://www.nature.com/articles/s41586-020-03051-4
- Antonoglou et al., "Planning in Stochastic Environments with a Learned Model"
  (Stochastic MuZero), ICLR 2022: https://openreview.net/forum?id=X6D9bAHhBQ1
- Julian Schrittwieser overview of Stochastic MuZero:
  https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/
- DeepMind Mctx repository and README:
  https://github.com/google-deepmind/mctx
- Mctx `stochastic_muzero_policy` implementation:
  https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py
- LightZero repository, including MuZero-family algorithm support:
  https://github.com/opendilab/LightZero
- Machado et al., "Revisiting the Arcade Learning Environment: Evaluation
  Protocols and Open Problems for General Agents" (sticky actions):
  https://arxiv.org/pdf/1709.06009
- Silver et al., "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm" (AlphaZero exploration noise):
  https://arxiv.org/pdf/1712.01815
- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from
  Simulation to the Real World": https://arxiv.org/abs/1703.06907
- Peng et al., "Sim-to-Real Transfer of Robotic Control with Dynamics
  Randomization": https://arxiv.org/abs/1710.06537
- Laskin et al., "Reinforcement Learning with Augmented Data":
  https://arxiv.org/abs/2004.14990
- Kostrikov et al., "Image Augmentation Is All You Need: Regularizing Deep
  Reinforcement Learning from Pixels": https://arxiv.org/abs/2004.13649
- Stable-Baselines3 Atari wrappers, including sticky actions and action repeat:
  https://stable-baselines3.readthedocs.io/en/v2.4.0/common/atari_wrappers.html
- LightZero tree-search docs showing root Dirichlet noise parameters:
  https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html

## Local Links

- [mctx_integration.md](mctx_integration.md)
- [muzero_architecture_deep_dive.md](muzero_architecture_deep_dive.md)
- [pong_reward_design.md](pong_reward_design.md)
- [training_evaluation.md](training_evaluation.md)
- [training/domain_variation_for_robustness.md](training/domain_variation_for_robustness.md)
- [../design/deterministic_environment.md](../design/deterministic_environment.md)
- [../design/muzero_modal_architecture.md](../design/muzero_modal_architecture.md)
- [../design/training/domain_variation_plan.md](../design/training/domain_variation_plan.md)
