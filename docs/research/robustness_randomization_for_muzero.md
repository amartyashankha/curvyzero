# Robustness Randomization For MuZero-Like Visual Control

Status: Research note
Date: 2026-05-09

## Short Answer

Yes, people intentionally add randomness to reinforcement-learning practice.
For visual control, the common reasons are:

- stop agents from memorizing one deterministic trajectory;
- make policies robust to small control mistakes and sensor variation;
- improve pixel-level generalization and data efficiency;
- train across a family of environment variants instead of one brittle seed.

For CurvyZero, do this after the first deterministic project-owned MuZero/Mctx
trainer exists. Start with named train/eval profiles, explicit seeds, and replay
metadata. Do not jump to Stochastic MuZero just because a wrapper adds noise.

Plain rule:

```text
If randomness only changes the training distribution, standard MuZero is enough.
If the planner must branch over unresolved future chance events, consider
Stochastic MuZero.
```

## What People Add On Purpose

### Sticky actions and control noise

ALE added sticky actions because Atari games are otherwise deterministic enough
that an agent can overfit to action sequences. Sticky actions repeat the
previously executed action with some probability, so the agent's requested
action is not always the action that reaches the emulator. ALE documentation says
the default action-repeat stochasticity is `0.25`, chosen as a level human
play-testers did not notice as delay, and recommends reporting this setting.

Why it helps: it tests robust closed-loop control, not just open-loop trajectory
memorization.

CurvyZero translation: a sticky/frozen-control wrapper is useful as a later
robustness profile. Log both `chosen_action` and `executed_action`, the wrapper
probability, the wrapper RNG seed, and the previous executed action.

### Action repeat and frame skip

Action repeat is standard in Atari-style visual control. It reduces control
frequency, speeds up simulation, and makes the decision horizon shorter. It is
not always "noise": fixed action repeat is part of the rules/control interface.
Random frame skip is stochastic and should be reported separately.

Why it helps: repeated actions reduce per-frame twitchiness and computation, but
too much repeat can make fine steering impossible.

CurvyZero translation: fixed action repeat is a ruleset/action-schema choice.
Use a small fixed value only if the game tick is too fine for meaningful
decisions. Do not mix random frame skip into the canonical score unless it is
the target game profile.

### Observation noise and image augmentation

Image augmentation is common for pixel RL. RAD studied random translation, crop,
color jitter, patch cutout, random convolutions, and amplitude scaling, and
reported better data efficiency and generalization on common control benchmarks.

Why it helps: the representation learns that small pixel shifts, crops, colors,
and sensor artifacts are not strategic facts.

CurvyZero translation: replay-time augmentation is the first choice for raster
Pong/Curvy observations. Environment-time observation noise is stronger because
it changes what the policy sees during rollout; seed and log it.

### Domain randomization

Domain randomization deliberately samples simulator parameters. The robotics
version randomizes visuals, lighting, camera, object placement, and other
simulation details so the real world looks like another variation. Robust RL
work also trains over ensembles of source domains to handle simulator mismatch
and unmodeled effects.

Why it helps: the policy learns a family of tasks instead of one exact
configuration.

CurvyZero translation: later variation profiles can sample arena size, speed,
turn rate, trail width, colors, raster scale, spawn geometry, opponent family,
and camera/crop settings at reset. Keep the sampled parameters in replay and
eval summaries. If the sampled parameters change the rules, include them in the
rules hash or variation profile hash.

## What Is Too Much

Randomization is too much when it hides the learning signal or changes the task
without a label.

Practical warning signs:

- Score on clean eval drops while noisy-train score rises.
- The agent learns to stall, hedge, or freeze instead of winning.
- The same checkpoint has high variance but no improvement in mean score,
  survival, p90 survival, or rare win count.
- The policy cannot infer the current control state because wrappers changed
  actions but observations omit recent actions.
- Different randomizations are mixed into one metric, so no one can tell what
  improved.
- The trainer spends capacity modeling nuisance variation before it has learned
  basic paddle/trail geometry.

For the first MuZero/Mctx smoke, "too much" means almost any extra randomness.
Use deterministic Pong/Curvy first. Add robustness rows only after the clean
scoreboard moves.

Suggested starting levels once a clean baseline exists:

| Profile | First level | Why this level |
| --- | --- | --- |
| Sticky action eval | `0.05` then `0.10`; keep `0.25` as a hard ALE-like row. | Small enough to expose brittleness before it dominates steering. |
| Frozen control | one-step freeze with low probability. | Tests recovery without making turns feel broken. |
| Fixed action repeat | `1` canonical, maybe `2` as a separate control-frequency profile. | Curvy steering is sensitive; repeat changes the game. |
| Random frame skip | avoid in canonical. | It changes timing and can obscure whether the policy learned control. |
| Pixel augmentation | small translations/crops/color jitter at replay sample time. | Helps visual invariance without changing the environment trajectory. |
| Domain randomization | one parameter family at a time. | Keeps failures explainable. |

Do not tune all noise knobs together. Add one profile, score it separately, and
keep the clean eval row as the promotion gate.

## Standard MuZero Versus Stochastic MuZero

Standard MuZero can handle a lot of randomness as ordinary experience:

- exploration noise in self-play;
- root Dirichlet noise and visit temperature;
- epsilon/random action collection;
- sticky actions, if replay records the transition that actually happened;
- observation augmentation, if it is applied consistently to training inputs;
- reset-time domain randomization, if the sampled variant is logged and exposed
  when needed.

In these cases the deterministic dynamics model may learn the expected value of
actions under the data distribution. That is often fine. The planner does not
need to explicitly branch over every wrapper outcome if the policy can become
robust through closed-loop feedback.

Planning needs stochastic/chance-node dynamics when averaging becomes a lie:

- one action can lead to several sharply different next states;
- those next states require different follow-up actions;
- the chance event is unresolved at planning time;
- the event has strategic value, such as a random item spawn, random boost type,
  random hazard activation, dice/card-like outcome, or hidden opponent state;
- deterministic MuZero search repeatedly prefers actions that only work under an
  averaged future that never really occurs.

This is the Stochastic MuZero case. It factors planning into an action-created
afterstate followed by a chance outcome and searches over both decision nodes and
chance nodes.

Plain example:

- Sticky control at `0.05`: train standard MuZero with the wrapper and log it.
  The agent can learn "keep enough margin that a missed turn is survivable."
- Random item spawn that may appear left or right next tick and changes the best
  turn now: consider stochastic planning later, because the tree may need to
  compare branches.

## CurvyZero Recommendation

Use three lanes, in this order:

1. Canonical deterministic lane: no sticky actions, no random frame skip, no
   random trail gaps, no random items, fixed action schema, fixed rules hash.
2. Robustness scorecard lane: same trained checkpoint, evaluated under named
   sticky-control, observation-noise, and domain-randomized profiles.
3. Robustness training lane: train with one named noise profile only after clean
   learning exists and the profile exposes a real weakness.

Only open a Stochastic MuZero implementation lane after the deterministic
MuZero/Mctx trainer exists and one of these is measured:

- a stochastic ruleset is the actual target game;
- deterministic dynamics collapses real branches into harmful averages;
- extra history, wrapper-state exposure, and profile-specific training do not
  fix the failure.

Metadata to store in replay/eval:

- `chosen_action` and `executed_action`;
- previous executed actions when controls can stick or freeze;
- `action_repeat` and random-frame-skip settings;
- sticky/frozen/delay/drop probabilities;
- observation-noise and augmentation profile names;
- domain-randomization profile name plus sampled parameters;
- all RNG seeds and random stream names;
- clean-vs-noisy eval row name.

## Sources

- Silver et al., "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm" (AlphaZero), arXiv:1712.01815:
  https://arxiv.org/abs/1712.01815
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a
  Learned Model" (MuZero), arXiv:1911.08265:
  https://arxiv.org/abs/1911.08265
- Antonoglou et al., "Planning in Stochastic Environments with a Learned Model"
  (Stochastic MuZero), OpenReview ICLR 2022:
  https://openreview.net/forum?id=X6D9bAHhBQ1
- Machado et al., "Revisiting the Arcade Learning Environment: Evaluation
  Protocols and Open Problems for General Agents", JAIR 2018:
  https://jair.org/index.php/jair/article/view/11182
- ALE environment specification for action-repeat stochasticity:
  https://ale.farama.org/env-spec/
- ALE environment docs for sticky actions, frame skip, and
  `repeat_action_probability`:
  https://ale.farama.org/environments/
- Laskin et al., "Reinforcement Learning with Augmented Data" (RAD),
  arXiv:2004.14990:
  https://arxiv.org/abs/2004.14990
- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from
  Simulation to the Real World", arXiv:1703.06907:
  https://arxiv.org/abs/1703.06907
- Rajeswaran et al., "EPOpt: Learning Robust Neural Network Policies Using Model
  Ensembles", arXiv:1610.01283:
  https://arxiv.org/abs/1610.01283
