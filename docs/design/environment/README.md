# Environment Fidelity

Status: Draft

This is the stable map for environment-fidelity work. Research notes can stay
messy; design docs should say what we plan to build.

## Current Short Answer

- `curvyzero-v0` is a simple training ruleset, not an exact CurvyTron clone.
- Fidelity work still defines the source guardrail. The core scalar/source slice
  is now strong enough that speed work is active too.
- The verified source-fidelity slice now covers source-style kinematics plus the
  narrow wall/border event contract for normal-wall death, same-frame normal-wall
  draw, plain source borderless wrap, borderless PrintManager wrap,
  destination-body skip, and exact-edge/corner-axis behavior.
- The 3P/4P normal-wall scoring, death-order, and terminal-draw canaries are now
  verified through the JS/Python common-trace batch.
- The `source-body-canary` checkpoint is verified for six narrow fixtures:
  opponent tangent-safe strict overlap, opponent overlap-kill, own delta `3`
  safe, own delta `4` kill, same-frame point kill, and same-frame point control.
  A separate old-body metadata batch verifies one `old:true` body-death event.
- The `source-print-manager-canary` checkpoint is verified for the stable
  eight-case batch: print-to-hole, hole-to-print, exact-zero toggle, active
  no-toggle control, delayed start, active printing/already-hole
  stop-on-death, and body-collision stop-on-death. The random-tape call-order
  fixture is verified in a separate one-case batch.
- The `source-trail-cadence-canary` checkpoint is verified for normal point
  insertion and below-radius no-point behavior.
- The `source-trail-gap-canary` checkpoint is verified for forced hole-space
  safety, stored-body-in-visual-hole kill, print-to-hole boundary kill, and
  hole-to-print same-update emitted-body kill.
- Next source-fidelity slice should be chosen deliberately from the remaining
  narrow gaps: longer real PrintManager cadence, round lifecycle, bonuses,
  observation/reward, emitted-trail bodies, or a borderless corner follow-up if
  needed.
- Current speed work should stay fixture-backed: fixed event arrays,
  `B>1` timing, and Modal/JAX/Mctx sweeps are active, but they are not source-
  fidelity proof by themselves. The goal is one fast faithful environment, not a
  slow reference and a separate fast game.
- Shared scenario schema, toy runner, and source-fidelity runner ownership is
  split: `curvyzero.env.scenario_schema`, `curvyzero.env.toy_runner`, and
  `curvyzero.fidelity.source_runners` respectively. `curvyzero.env.scenarios`
  remains the compatibility facade/CLI.
- Trainer-facing code should still import only `curvyzero.env`; source runners,
  scenario schema helpers, trace normalization, and diff tooling are evidence
  machinery.
- The raw CurvyTron browser app still does not run from this checkout because the
  old generated build outputs are missing.
- Browser hosting and trainer-facing APIs are still parked. Performance work is
  active only where it stays tied to verified fixtures, honest timing, and clear
  unsupported-semantics labels.

## North Star

Source fidelity remains the guardrail; speed work is now active because
the core state slice is trustworthy enough to compare. Keep public training
interfaces separate from the fidelity machinery: reconstruction evidence can
inform the interface, but it should not be constrained by the first coach-facing
API. Defer browser hosting and trainer-facing API expansion until state,
observation, reward, and batch contracts are explicit.

## Reading Path

1. Start with the [active work tracker](../../working/environment/active_lanes.md).
2. Use the [full environment spec](../../working/environment/full_environment_spec_2026-05-09.md)
   for the source rules we are rebuilding.
3. Use the [fast environment migration plan](../../working/environment/fast_environment_migration_plan_2026-05-09.md)
   for the current move from proof tools into the one fast runtime.
4. Use the [working coverage tracker](../../working/environment/coverage_tracker.md)
   for the current verified/open slice map.
5. Use the [working question map](../../working/environment_questions.md) for open
   questions and plain-language concerns.
6. Read [deterministic environment](../deterministic_environment.md) and
   [rulesets](../rulesets.md) to understand the current simulator contract.
7. Use [fidelity checklist](fidelity_checklist.md) and
   [fidelity comparison](fidelity_comparison.md) for the comparison ladder.
8. Use [multiplayer canaries](multiplayer_canaries.md) for verified 3P/4P
   normal-wall scoring and death-order targets.
9. Use [reference oracle](reference_oracle.md) for the headless JS oracle design.
10. Use [observability plan](observability_plan.md) for the small artifact layer
   that makes first mismatches, event timelines, and state context easy to read.
11. Use [Modal fidelity jobs](modal_fidelity_jobs.md) and the
   [Modal fidelity runbook](../../runbooks/modal_environment_fidelity.md) for
   remote batch shape and artifacts.
12. Use [training interface contract](training_interface_contract.md) for the
   boundary between reconstruction evidence and coach/training consumption.
13. Use [EnvironmentTransitionV0](environment_transition_v0.md) for the
   transition contract that migrates source semantics into the fast
   implementation and gates optimized-path claims.
14. Use the [environment performance/vectorization plan](../../research/environment/performance_vectorization_plan.md)
   for the active fixture-backed speed lane and benchmark TODOs.
15. Use the [source feature inventory](../../working/environment/source_feature_inventory.md)
   as the current rebuild map with per-mechanic status labels.
16. Treat the older [May 8 handoff](../../handoffs/2026-05-08-environment-fidelity-handoff.md)
   as history only. It is not the current status source.

## Design Docs In This Folder

- [fidelity_checklist.md](fidelity_checklist.md) - plain checklist of source rules
  that must either match or be named as differences.
- [fidelity_comparison.md](fidelity_comparison.md) - trace, golden, replay, event,
  server-message, and pixel comparison policy.
- [trace_loop_contract.md](trace_loop_contract.md) - scenario, trace, diff, and
  artifact contract for the local loop.
- [reconstruction_workflow.md](reconstruction_workflow.md) - short workflow for
  moving from source facts to matched Python behavior.
- [probe_automation_plan.md](probe_automation_plan.md) - next local automation
  steps for scenario batches and summaries.
- [observability_plan.md](observability_plan.md) - low-cost debug artifacts for
  common traces, first mismatches, event timelines, and small state tables.
- [reference_oracle.md](reference_oracle.md) - plan for a small headless JS oracle
  that emits state traces from the original source.
- [modal_fidelity_jobs.md](modal_fidelity_jobs.md) - Modal job, artifact, and
  manifest design for environment fidelity batches.
- [modal_reference_hosting.md](modal_reference_hosting.md) - plan and caveats for
  hosting or probing the old CurvyTron reference on Modal.
- [multiplayer_canaries.md](multiplayer_canaries.md) - pending and verified
  multiplayer source-fidelity canaries for map size, reverse update order,
  scoring, death order, and round end.
- [training_interface_contract.md](training_interface_contract.md) - concise
  boundary for what training can rely on, what is unstable, and which interface
  decisions must be made before training consumes the environment.
- [environment_transition_v0.md](environment_transition_v0.md) - transition
  contract, trace schema, and migration plan from executable source spec to the
  fast production environment.

## Related Stable Docs

- [deterministic_environment.md](../deterministic_environment.md) - current Python
  simulator scope and known `curvyzero-v0` deviations.
- [rulesets.md](../rulesets.md) - names and provenance rules for `curvyzero-v0`,
  `curvytron-v1-reference`, and later variants.
- [modal_architecture.md](../modal_architecture.md) - project-wide Modal hot-loop
  rule and compute/storage shape.
- [local_development.md](../../runbooks/local_development.md) - local tests,
  benchmark smoke, and current Modal smoke commands.

## Evidence Index

Use [research/environment/README.md](../../research/environment/README.md) for the
research notes behind these design docs.
