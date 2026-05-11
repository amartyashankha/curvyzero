# Research Index

Research notes are evidence-gathering documents. They may be exploratory, opinionated, or incomplete. Stable conclusions should be promoted into decision records or design docs.

## Active Notes

- [baseline_learnability.md](baseline_learnability.md) - minimal baselines and evaluation before MuZero.
- [curvytron_reference_notes.md](curvytron_reference_notes.md) - original CurvyTron repository setup and rule mining.
- [curvytron_survival_reward_design_2026-05-11.md](curvytron_survival_reward_design_2026-05-11.md) - simple reward variants for two-player CurvyTron survival self-play and why unbounded shared survival reward is risky.
- [env_fidelity_curriculum.md](env_fidelity_curriculum.md) - fidelity, rule variants, and curriculum/domain randomization.
- [lightzero_feature_fit_for_curvyzero.md](lightzero_feature_fit_for_curvyzero.md) - skeptical feature-gap audit for trusting LightZero on dummy Pong and later Curvy.
- [lightzero_integration.md](lightzero_integration.md) - PyTorch/LightZero alternative critique.
- [mctx_integration.md](mctx_integration.md) - JAX/Mctx integration plan and benchmark shape.
- [modal_example_patterns.md](modal_example_patterns.md) - concrete Modal patterns extracted from local examples and official docs.
- [modal_patterns.md](modal_patterns.md) - Modal primitives and patterns for this project.
- [modal_training_patterns.md](modal_training_patterns.md) - Modal training examples and sequencing for CurvyZero GPU runs.
- [multiplayer_selfplay_muzero.md](multiplayer_selfplay_muzero.md) - prior work and practical formulations for multiplayer self-play with search.
- [muzero_architecture_deep_dive.md](muzero_architecture_deep_dive.md) - MuZero model/search/self-play architecture in implementation terms.
- [muzero_framework_vs_project_owned.md](muzero_framework_vs_project_owned.md) - current LightZero-first adapter plan and feature-gap checklist, with project-owned Mctx as fallback.
- [muzero_reference_examples.md](muzero_reference_examples.md) - stock MuZero/LightZero examples to use as local smoke references.
- [muzero_repo_baseline_options.md](muzero_repo_baseline_options.md) - off-the-shelf MuZero repo/library baseline options.
- [observation_reward_design.md](observation_reward_design.md) - observation and reward schemas for baseline and MuZero.
- [performance_vectorization.md](performance_vectorization.md) - path from Python stepping to large-scale rollout throughput.
- [pong_reward_design.md](pong_reward_design.md) - reward rule for dummy Pong and why rally length stays a log, not a reward.
- [repo_structure_critique.md](repo_structure_critique.md) - critique of repo topology and sequencing.
- [reward_shaping_for_pong_curvy_muzero.md](reward_shaping_for_pong_curvy_muzero.md) - practical recommendation for keeping Pong/Curvy rewards game-true while using loss-delay shaping only as telemetry or temporary target.
- [robustness_randomization_for_muzero.md](robustness_randomization_for_muzero.md) - sticky actions, control noise, action repeat, visual augmentation, and domain randomization for MuZero-style visual control.
- [simple_training_environment_options.md](simple_training_environment_options.md) - simplest toy environment for validating a MuZero-shaped loop before CurvyTron.
- [stochastic_muzero.md](stochastic_muzero.md) - whether Stochastic MuZero is needed for items, boosts, hazards, trail gaps, noisy transitions, and later stochastic rulesets.
- [training_loop_bottlenecks_amdhals_law_2026-05-09.md](training_loop_bottlenecks_amdhals_law_2026-05-09.md) - whole-loop MuZero/LightZero/Mctx bottlenecks, Amdahl's law, and what to measure before env/GPU rewrites.
- [training_evaluation.md](training_evaluation.md) - staged evaluation plan for dummy survival, Tiny Line Duel, current Pong checkpoints, and later CurvyTron training progress.
- [training_architecture_notes.md](training_architecture_notes.md) - training stack, repo structure, and implementation slices.
- [wiki_architecture.md](wiki_architecture.md) - documentation hierarchy and maintenance rules.

## Promoted Or Related Design

- [../design/deterministic_environment.md](../design/deterministic_environment.md) - current simulator contract distilled from environment research.
- [../design/repository_hierarchy.md](../design/repository_hierarchy.md) - current hierarchy proposal distilled from repo and wiki critique.
- [../design/training_architecture.md](../design/training_architecture.md) - current training sequence distilled from baseline, MuZero, and library research.

## Research Note Format

Use this shape unless a note has a better natural structure:

```md
# Topic

## Short Answer

## Evidence

## Recommendations

## Risks

## Open Questions

## Sources
```
