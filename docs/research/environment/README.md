# Environment Research Index

Status: Working

This page is the evidence map for the environment-fidelity lane. Stable plans live
in [design/environment](../../design/environment/README.md). The compact current
handoff is [2026-05-08 environment fidelity handoff](../../handoffs/2026-05-08-environment-fidelity-handoff.md).

## Current Evidence

- [CurvyTron reference notes](../curvytron_reference_notes.md) - source-mined
  timing, movement, trail gaps, collision, scoring, bonuses, and golden-test ideas.
- [CurvyTron raw run probe](curvytron_raw_run_probe.md) - why the raw reference
  app does not run from this checkout yet, and how to try it safely.
- [CurvyTron JS state oracle notes](curvytron_js_state_oracle.md) - practical plan
  for direct headless JS object stepping and state trace output.
- [Fidelity comparison options](fidelity_comparison_options.md) - ranking of
  source facts, state traces, goldens, events, replays, server messages, pixels,
  and videos.
- [Environment fidelity plan critique](fidelity_plan_critique.md) - risks in the
  plan, especially multiplayer, scoring, update order, and weak comparisons.
- [Environment performance/vectorization plan](performance_vectorization_plan.md) -
  deferred speed plan: fidelity first, stable API boundary, measure before
  optimizing, then local Python to NumPy batch to later framework/GPU experiments.
- [Modal CurvyTron hosting research](modal_curvytron_hosting.md) - feasibility and
  risks of using Modal for the old reference server or a headless probe.
- [Environment fidelity curriculum](../env_fidelity_curriculum.md) - how to name
  source-derived rules, v0 choices, curriculum variants, and randomized variants.

## Promoted Design

- [Environment design index](../../design/environment/README.md) - stable reading
  path for this lane.
- [Deterministic environment](../../design/deterministic_environment.md) - current
  simulator contract and known `curvyzero-v0` deviations.
- [Rulesets](../../design/rulesets.md) - named rule targets and provenance labels.
- [Fidelity checklist](../../design/environment/fidelity_checklist.md) - source
  behavior checklist.
- [Fidelity comparison](../../design/environment/fidelity_comparison.md) - chosen
  comparison ladder and artifact contract.
- [Reference oracle](../../design/environment/reference_oracle.md) - design for the
  headless JS oracle.
- [Modal fidelity jobs](../../design/environment/modal_fidelity_jobs.md) - remote
  batch and artifact design.
- [Modal fidelity runbook](../../runbooks/modal_environment_fidelity.md) - command
  shape for future fidelity runs.

## Current Answers

### Can We Run CurvyTron?

Not yet from this checkout. The old app needs generated build files and old
dependencies. The next safe test is a disposable copy or pinned Modal image.

Useful docs:

- [CurvyTron raw run probe](curvytron_raw_run_probe.md)
- [Modal CurvyTron hosting research](modal_curvytron_hosting.md)
- [Modal reference hosting design](../../design/environment/modal_reference_hosting.md)

### How Do We Compare?

Use source facts and state traces first. Use screenshots and videos later for
human review.

Useful docs:

- [Fidelity comparison options](fidelity_comparison_options.md)
- [Fidelity comparison design](../../design/environment/fidelity_comparison.md)
- [Reference oracle design](../../design/environment/reference_oracle.md)

### What About Pixels?

Pixels are useful for demos and render checks, but they are a weak first oracle.
The simulator truth should come from state, events, and golden outcomes.

Useful docs:

- [Observation and reward design](../observation_reward_design.md)
- [Fidelity comparison design](../../design/environment/fidelity_comparison.md)

### What About Modal?

Use Modal for coarse jobs and artifact storage. Keep tick loops and trace diffs
inside one process/container.

Useful docs:

- [Modal CurvyTron hosting research](modal_curvytron_hosting.md)
- [Modal fidelity jobs](../../design/environment/modal_fidelity_jobs.md)
- [Modal fidelity runbook](../../runbooks/modal_environment_fidelity.md)
- [Modal patterns](../modal_patterns.md)

## Open Threads

- Build the original reference in a disposable copy or pinned image.
- Prove the headless JS oracle can step 2-player, then 3/4-player cases.
- Choose the first trace schema and tolerances.
- Decide whether first source-fidelity traces use elapsed ms or fixed `1000 / 60` ms.
- Decide which v0 differences are allowed before serious learning starts.
