# LightZero Train Refactor, 2026-05-13

Purpose: plan and guard a cleanup of the CurvyTron stock LightZero training
lane. This directory is the working memory for the refactor. It is separate
from the experiment-matrix and tournament notes on purpose.

## Current Target

Keep the trusted training path close to stock LightZero:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
```

The trainer should call stock `lzero.entry.train_muzero`. CurvyZero code should
only provide the environment, configuration, checkpoint/artifact plumbing, and
external observability.

## First Rule

Tests first. No broad refactor until the current contracts are pinned down with
focused regression tests.

## Read First

- [current_source_of_truth.md](current_source_of_truth.md)
- [review_index.md](review_index.md)
- [high_level_goal.md](high_level_goal.md)
- [operating_patterns.md](operating_patterns.md)
- [delegation_orchestration.md](delegation_orchestration.md)
- [regression_test_lockdown_plan.md](regression_test_lockdown_plan.md)
- [refactor_targets.md](refactor_targets.md)
- [checkpoint_helper_contract.md](checkpoint_helper_contract.md)
- [opponent_registry_design.md](opponent_registry_design.md)
- [opponent_leaderboard_interface.md](opponent_leaderboard_interface.md)
- [opponent_leaderboard_interface_second_critique_2026-05-13.md](opponent_leaderboard_interface_second_critique_2026-05-13.md)
- [bug_registry.md](bug_registry.md)
- [decision_log.md](decision_log.md)
- [source_file_inventory.md](source_file_inventory.md)
- [test_inventory.md](test_inventory.md)
- [refactor_sequence.md](refactor_sequence.md)
- [next_refactor_cut.md](next_refactor_cut.md)
- [stock_lightzero_parity_audit.md](stock_lightzero_parity_audit.md)
- [side_hook_refactor_critique_2026-05-13.md](side_hook_refactor_critique_2026-05-13.md)
- [granular_action_cadence_plan_2026-05-13.md](granular_action_cadence_plan_2026-05-13.md)
- [cleanup_policy.md](cleanup_policy.md)
- [subagent_briefs.md](subagent_briefs.md)
- [glossary.md](glossary.md)
- [todo.md](todo.md)
- [trainer_surface_map.md](trainer_surface_map.md)

## Neighboring Evidence

- [../curvytron_architecture_research_2026-05-12/current_source_of_truth.md](../curvytron_architecture_research_2026-05-12/current_source_of_truth.md)
- [../curvytron_architecture_research_2026-05-12/checkpoint_discovery_audit_2026-05-13.md](../curvytron_architecture_research_2026-05-12/checkpoint_discovery_audit_2026-05-13.md)
- [../checkpoint_tournament_checkpoint_discovery_handoff_2026-05-13.md](../checkpoint_tournament_checkpoint_discovery_handoff_2026-05-13.md)
