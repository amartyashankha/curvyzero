# Domain Variation For Robustness

Status: Research note
Date: 2026-05-08

## Short Answer

Domain variation is useful later, after the source-fidelity and fixed-rules
learning lanes are stable. It should not block reconstruction work.

For CurvyZero, the best first robustness recipe is narrow:

- Keep `curvyzero-v0` fixed for debug, golden tests, and baseline learning.
- Add a separate `curvyzero-robust-*` training family that samples controlled
  changes at reset.
- Evaluate every robust policy on both the canonical ruleset and held-out
  variation ranges.
- Log the sampled config with every episode, replay, checkpoint, and eval row.

This keeps two facts separate:

- Source-fidelity asks: does our simulator match the chosen CurvyTron source
  behavior?
- Robust training asks: can a later agent handle small, named changes around a
  target game?

## What The Literature Suggests

Domain randomization trains on many simulated variants so the target setting
looks like one more sampled case. Tobin et al. used visual randomization for
sim-to-real object localization and grasping. Peng et al. used dynamics
randomization so robot policies handled different physical parameters.

Observation augmentation is the lighter version for learned observations. RAD
and DrQ show that pixel and state augmentations can improve data efficiency and
test generalization in RL. For CurvyZero this means crop/scale/color/noise
changes should live near the observation wrapper, not inside the source physics.

Procedural generation work is the strongest warning against fixed-seed comfort.
CoinRun and Procgen show that RL agents can overfit surprisingly large training
sets, and that train/test splits over environment seeds matter.

Robust parameter training can also target hard cases. EPOpt trains on ensembles
and emphasizes poor-performing parameter samples. That maps to later CurvyZero
stress cases such as cramped arenas, fast speeds, wide trails, and high player
counts.

Multi-agent self-play adds another kind of variation: other policies. Competitive
self-play can create a natural curriculum, but MARL policies can overfit to the
opponents seen during training. OpenAI Five and AlphaStar-style systems used
large-scale self-play and diverse opponents/leagues; PSRO makes the same point
more directly for policy mixtures.

## CurvyZero Variation Knobs

Use these only in an explicit robust ruleset, not in the current fidelity lane.

| Knob | Why it helps | First safe range |
| --- | --- | --- |
| Arena size | Prevents spawn/layout overfit and tests horizon scaling. | Small symmetric sweep around the canonical size, for example `0.8x` to `1.25x`. |
| Speed | Tests timing and collision margins. | Mild multiplier, for example `0.9x` to `1.1x`. |
| Turn rate | Tests control robustness and curve geometry. | Mild multiplier, for example `0.9x` to `1.1x`. |
| Trail width | Tests clearance estimates and raster/ray scale. | Mild multiplier, for example `0.8x` to `1.2x`. |
| Trail gaps | Tests memory and partial observability when holes exist. | Off at first; later vary gap period/length within source-inspired ranges. |
| Colors | Prevents visual identity and team/color shortcuts. | Random palette or channel permutation for visual observations only. |
| Observation noise | Tests sensor-like uncertainty without changing physics. | Small ray distance noise, head jitter, or raster dropout. |
| Camera/raster scale | Prevents overfit to one crop resolution or zoom. | Small zoom/translation jitter after observation generation. |
| Bonus frequency | Tests stochastic events and changing incentives. | Off for v0; later low/medium/high named buckets. |
| Player count | Tests multiplayer density and joint interaction. | Fixed 1v1 first; later train/eval separate 2, 3, 4+ player curricula. |

## Recommended Order

1. Finish source-fidelity reconstruction and fixed `curvyzero-v0` baseline gates.
2. Add observation-only augmentation first: colors, raster crop jitter, ray noise,
   and scale jitter. This has low risk because it does not change game rules.
3. Add mild environment parameter sweeps: arena size, speed, turn rate, and trail
   width. Keep one sampled config for the full episode.
4. Add harder events: trail gaps, bonuses, and player-count curricula.
5. Add opponent diversity: random, sticky random, heuristic, recent checkpoints,
   older strong checkpoints, and held-out evaluation opponents.

## Guardrails

- Never mix robustness samples into source-fidelity pass/fail tests.
- Never claim a robust policy is better from training reward alone. Use held-out
  config ranges and fixed canonical evaluation.
- Keep variation ranges small until the fixed environment is learnable.
- Store `rules_hash`, `variation_profile`, sampled parameter values, observation
  schema, reward schema, seed, and opponent ids in replay.
- Treat catastrophic held-out failures as useful. They should tighten the
  variation plan, not silently widen randomization.

## Primary Sources

- Tobin et al., ["Domain Randomization for Transferring Deep Neural Networks from
  Simulation to the Real World"](https://arxiv.org/abs/1703.06907).
- Peng et al., ["Sim-to-Real Transfer of Robotic Control with Dynamics
  Randomization"](https://arxiv.org/abs/1710.06537).
- Rajeswaran et al., ["EPOpt: Learning Robust Neural Network Policies Using Model
  Ensembles"](https://arxiv.org/abs/1610.01283).
- OpenAI et al., ["Solving Rubik's Cube with a Robot Hand"](https://arxiv.org/abs/1910.07113).
- Laskin et al., ["Reinforcement Learning with Augmented Data"](https://arxiv.org/abs/2004.14990).
- Kostrikov et al., ["Image Augmentation Is All You Need: Regularizing Deep
  Reinforcement Learning from Pixels"](https://arxiv.org/abs/2004.13649).
- Cobbe et al., ["Quantifying Generalization in Reinforcement
  Learning"](https://arxiv.org/abs/1812.02341).
- Cobbe et al., ["Leveraging Procedural Generation to Benchmark Reinforcement
  Learning"](https://arxiv.org/abs/1912.01588).
- Bansal et al., ["Emergent Complexity via Multi-Agent
  Competition"](https://arxiv.org/abs/1710.03748).
- Berner et al., ["Dota 2 with Large Scale Deep Reinforcement
  Learning"](https://arxiv.org/abs/1912.06680).
- Lanctot et al., ["A Unified Game-Theoretic Approach to Multiagent
  Reinforcement Learning"](https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html).
- Vinyals et al., ["Grandmaster level in StarCraft II using multi-agent
  reinforcement learning"](https://www.nature.com/articles/s41586-019-1724-z).
