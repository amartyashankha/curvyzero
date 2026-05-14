# Delegation And Orchestration

Purpose: keep subagent work visible and bounded.

## Active Agents

| Agent | Lane | Output | Status |
| --- | --- | --- | --- |
| Pauli/Zeno/Ohm/Dalton/Hegel | First audit wave from before context transition. | intended separate audit docs | handles lost; no docs written |
| Archimedes | Critique taxonomy, naming, and test cleanup risk. | `naming_and_test_cleanup_critique_2026-05-13.md` | complete |
| Confucius | Find exact regression tests needed before refactor. | `test_lockdown_audit_2026-05-13.md` | complete |
| Aquinas | Trace checkpoint discovery/status/poller/resume/GIF paths and propose tested patch. | `checkpoint_bug_patch_design_2026-05-13.md` | complete |
| Turing | Map current file responsibilities and extraction order. | `architecture_boundary_audit_2026-05-13.md` | complete |
| Carver | Dirty-state risk plus local E2E contract design. | `dirty_state_and_e2e_contract_audit_2026-05-13.md` | complete |
| Kant | Re-audit remaining fixed `lightzero_exp` path assumptions after the Bug 1 patch. | final message | complete |
| Carson | Recommend the next smallest test-backed refactor cut after `lightzero_checkpoints.py`. | final message | complete |
| Chandrasekhar | Critique future Modal Dict / opponent assignment design. | final message | complete |
| Herschel | Audit active hard-coded frozen opponent refs. | final message | complete |
| McClintock | Critique why the trainer file grew and guardrails for the registry feature. | final message | complete |
| Anscombe | Stock LightZero parity audit for `--mode train`. | final message folded into `stock_lightzero_parity_audit.md` | complete |
| Gibbs | Trainer/env boundary audit for simultaneous action and opponent mixtures. | final message folded into `stock_lightzero_parity_audit.md` | complete |
| Arendt | Validation coverage audit after the refactor. | final message folded into `stock_lightzero_parity_audit.md` and `todo.md` | complete |
| Banach | Re-audit whether fresh `--mode train` still delegates collector/search/replay/learner to LightZero. | final message folded into parity audit and experiment log | complete |
| Bernoulli | Audit whether CurvyZero side hooks are passive for fresh train. | final message folded into bug registry and experiment log | complete |
| Helmholtz | Audit what regression tests now lock and what remains untested. | final message folded into test inventory and todo | complete |
| Volta | Design and critique the future opponent leaderboard / assignment interface. | `opponent_leaderboard_interface.md` | complete; handle already closed/gone |
| Halley | Read-only critique of remaining stock-LightZero refactor risks. | final message folded into test inventory and next-refactor plan | complete; handle already closed/gone |
| Euler | Second critique of opponent leaderboard / assignment interface before wiring. | `opponent_leaderboard_interface_second_critique_2026-05-13.md` | complete; closed |
| Dalton | Critique next refactor cut for side hooks and checkpoint scaffolding. | `side_hook_refactor_critique_2026-05-13.md` | complete; closed |
| Sagan/Hilbert | Audit granular action cadence from launcher defaults through env stepping. | `granular_action_cadence_audit_2026-05-13.md` | complete |
| Noether | Critique action-cadence redesign architecture and semantic risks. | `granular_action_cadence_design_critique_2026-05-13.md` | complete |
| Ampere | Critique regression-test coverage and 300ms/12-frame assumptions. | `granular_action_cadence_regression_critique_2026-05-13.md` | complete |
| Parfit | Critique remaining stale `decision_ms` foot-guns after the default change. | `granular_action_cadence_footgun_critique_2026-05-13.md` | complete |
| Raman | Audit trainer-loop side effects of one-frame cadence. | `granular_action_cadence_trainer_loop_side_effects_2026-05-13.md` | complete |
| Euclid | Audit env/reward/source-step side effects. | `granular_action_cadence_env_reward_side_effects_2026-05-13.md` | complete |
| Dirac | Audit background eval, GIF, website, and artifact side effects. | `granular_action_cadence_eval_gif_site_side_effects_2026-05-13.md` | complete |
| Dewey | Audit launch scripts, manifests, Modal entrypoints, and stale CLI values. | `granular_action_cadence_launch_manifest_side_effects_2026-05-13.md` | complete |
| Lovelace | Audit downstream checkpoint consumers, tournaments, and opponent policies. | `granular_action_cadence_downstream_consumers_side_effects_2026-05-13.md` | complete |
| Darwin | Find the smallest honest post-patch train-loop smoke. | `granular_action_cadence_e2e_smoke_plan_2026-05-13.md` | complete |
| Russell | Audit post-smoke artifact visibility and Volume commit semantics. | `cadence_smoke_artifact_visibility_audit_2026-05-13.md` | active |
| Nash | Plan eval/GIF cadence pass-through patch and tests. | `eval_gif_cadence_passthrough_patch_plan_2026-05-13.md` | active |

## Orchestration Notes

- Main thread owns this file.
- Agents should not launch Modal jobs.
- Agents should not edit source unless explicitly reassigned as workers later.
- First wave was read-only and sharpened the test plan.
- Latest read-only wave confirmed fresh `--mode train` is stock-owned in the
  important training-loop sense. Remaining validation gates are background
  eval/GIF, live timestamped `lightzero_exp_*`, and any future resume
  equivalence claim.
- Follow-up action from latest wave: added a direct local fake-`train_muzero`
  regression and made resume-sidecar write failures non-fatal.
- Latest parallel split closed: Volta wrote the hybrid leaderboard interface
  recommendation, and Halley identified hook-passivity tests as the best next
  guardrail before moving hook code.
- Follow-up action from Halley: added and ran a focused fresh-resume hook
  pass-through regression. Fresh runs now have local coverage proving
  `call_hook`, `eval`, and `random_collect` still return stock results when
  resume is inactive.
- Latest parallel split closed: Sagan/Hilbert, Noether, Ampere, and Parfit all
  found the same core risk. The one-frame default was necessary but not
  sufficient because stale `decision_ms` values could still hide action repeat.
  The trainer now rejects stale multi-frame `decision_ms` values in trusted
  train/dry mode, passes `decision_source_frames=1` explicitly, and active
  manifest builders emit the one-frame value.
- Current parallel split: side-effect audit after the cadence patch. The local
  contract tests are green, and Darwin identified the smallest honest waited CPU
  Modal `--mode train` smoke. Main thread owns running it and recording the
  result because agents should not launch Modal jobs.
- Fresh waited CPU smoke returned `ok=true` and `called_train_muzero=true`, but
  Volume listing did not show final summary/checkpoint artifacts. Treat this as
  a trainer scaffolding artifact-visibility bug until Russell's follow-up says
  otherwise.
- Dalton result folded in: do not start by extracting hook installers. First
  extract pure resume-sidecar selection, then auto-resume checkpoint selection,
  then progress payload construction. Only consider shared hook mechanics after
  those smaller contracts are green.
- Euler result folded in: for opponent assignment wiring, add pure
  `opponent_registry.py` schema/hash helpers first, then trainer plumbing tests.
  Do not wire live Modal Dict reads into the trainer in this cut.
