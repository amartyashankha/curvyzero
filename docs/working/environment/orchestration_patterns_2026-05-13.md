# Environment Orchestration Patterns - 2026-05-13

Status: orchestration memory for current environment work.
Scope: how to sequence the investigation, not a completion claim.

The main thread should behave like a control room: keep the map current, split
bounded work, compare findings, and decide the next smallest source-backed
promotion. Follow-up threads should be narrow probes, not new centers of
gravity.

North star: faithful multiplayer CurvyTron environment first, then speed and
training integration. LightZero/training plumbing is a guarded downstream
interface only.

## Main-Thread Pattern

1. Re-read the current front-door docs before changing direction.
2. State the active lane and the evidence surface before delegating.
3. Delegate bounded probes with a concrete output format: finding, files read,
   tests or commands, risk, and next step.
4. Merge results into a short orchestration note or the relevant focused doc.
5. Close stale agents once their finding is recorded or superseded.
6. Reorder the queue when new source truth or product-route evidence changes
   risk.

## Delegation Rules

- Give each subagent one question and one boundary. Examples:
  hit-owner order, bonus probability semantics, trainer controls, replay/final
  observation, or stale-doc cleanup.
- Ask for citations to source files, tests, fixtures, or docs. Avoid summary-only
  answers.
- Use follow-ups for missing evidence instead of widening the original probe.
- Do not let a probe rewrite the product direction. The destination remains
  `VectorMultiplayerEnv` unless the main thread explicitly changes that.
- Bring findings back to the front-door docs when they affect future work.

## Spec-First Worker Pattern

Before a multi-file or cross-surface change, run one spec/catalog worker first.
That worker should produce a concrete map of exactly what must change, what must
not change, and how the claim will be proven. Only then split implementation
and test workers.

Required handoff shape:

- Target surface: source truth, product runtime, trainer/replay, renderer, or a
  named combination.
- Existing source and code anchors, with stale/toy paths called out.
- Target contract, field names, metadata, render mode rules, and no-overclaim
  language.
- Bounded implementation slices and focused validation commands.
- Explicit non-claims: route smoke, metadata-only replay, approximate renderer,
  or fixed/frozen opponent control must stay labeled as such.

For the current trainer work, the spec is
[source_state_multiplayer_trainer_surface_spec_2026-05-13.md](source_state_multiplayer_trainer_surface_spec_2026-05-13.md).
Implementation workers should build only the v0 surface described there; test
workers should attack final visual observations, render-mode guards, live-seat
flattening, and metadata honesty.

## Current Orchestration Plan

The authoritative short queue lives in
[current_queue_2026-05-13.md](current_queue_2026-05-13.md). The current
orchestration order is:

1. Multiplayer fidelity gaps.
   Audit full-game multiplayer behavior through source truth and
   `VectorMultiplayerEnv`: lifecycle, presence/leave, scoring, match-end,
   replay/final observations, bonus stack/death stress, and 3P/4P breadth.
2. Controls fidelity.
   Trace held controls from wrapper-facing actions through source-frame
   advancement. Confirm turn direction, decision-window hold behavior, masks,
   terminal padding, and sidecar action metadata without collapsing source
   frames into one physics step.
3. Renderer/fast-path boundary.
   Keep source-state/native fidelity claims separate from optimized or
   approximate render paths. `body_circles_fast` and similar modes stay
   labeled as approximate speed/profiling paths.
4. Docs/orchestration rhythm.
   Keep docs as working memory, keep main-thread planning/delegation/synthesis
   explicit, and use subagents only for bounded audits, tests, and docs.
5. Downstream training interface.
   Preserve trainer/replay/target-row/sample-batch work and Hume's opt-in
   real-LightZero construction helper, but treat it as downstream
   construction-smoke only. Real buffer sampled-target parity remains
   unproven.

## Queue Discipline

- Keep at most one main active implementation lane and one docs cleanup lane.
- Record blocked items with the missing proof, not just the symptom.
- Prefer small source-backed promotions over broad "fidelity" language.
- After every meaningful finding, ask whether the active queue order still
  matches risk.
- Run the environment doc guard after front-door doc edits, and run
  `git diff --check` before handing off.
- Preserve no-death/profile/training-helper additions as project features, not
  source-fidelity proof.
