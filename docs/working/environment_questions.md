# Environment Questions

Status: Working map

This is the messy question map for the environment-fidelity lane. The goal is to keep the user's concerns visible in simple language while the stable docs stay short.

Current compact handoff: [2026-05-08 environment fidelity handoff](../handoffs/2026-05-08-environment-fidelity-handoff.md).

## North Star

Rebuild the CurvyTron environment faithfully first, then optimize or train
against a chosen interface. Public training APIs are separate from the fidelity
machinery: traces prove behavior, while later wrappers decide what coaches and
models consume. Performance, Modal scale-out, and browser hosting stay deferred
until state traces and core rules are stable.

## Current Progress

- The source-style kinematics path matches JS through the local common-trace
  loop for the current movement fixtures.
- The local fidelity loop can select that path with `--python-runner source-kinematics`.
- Source says there are two border behaviors: normal-wall death and source
  borderless wrap from a timed bonus.
- Source borderless is not a clean torus: margin `0`, exact opposite edge, lost
  overshoot, strict edge equality is safe, first border axis only, and no body
  collision on the wrap frame.
- Current matched state traces: simple kinematics, normal-wall death,
  same-frame normal-wall draw, source borderless wrap, and the 3P/4P
  normal-wall scoring/death-order/terminal-draw canaries. The narrow
  `source-body-canary` runner is also verified for six body/trail fixtures:
  opponent tangent-safe strict overlap, opponent overlap-kill, own delta `3`
  safe, own delta `4` kill, same-frame point kill, and same-frame point
  control. The narrow `source-print-manager-canary` runner is verified for the
  stable eight-case PrintManager batch plus the separate random-tape call-order
  fixture. The narrow `source-trail-cadence-canary` runner is verified for
  normal point insertion and below-radius no-point behavior. The narrow
  `source-trail-gap-canary` runner is verified for forced hole-space safety,
  stored-body-in-visual-hole kill, and print-to-hole boundary kill. The narrow
  wall/border event contract is verified for two normal-wall fixtures, plain
  source borderless wrap,
  borderless PrintManager wrap, destination-body skip, exact-edge/corner-axis
  behavior, and the three multiplayer canaries.
- Current next target: preserve the verified batches, then pick the next longer
  PrintManager cadence edge, normal-wall edge controls, bonuses, or observation
  checks one small source claim at a time. Optimization stays later.

## Plain Questions From The User

### Can We Run CurvyTron?

Concern: before training anything, we need to know whether we can run the game or at least reproduce the game behavior.

Break it into two questions:

- Can we run or observe the original CurvyTron reference?
- Can we run the CurvyZero Python environment today?

Current answer:

- Original reference: not yet from this checkout. The generated `bin/curvytron.js` and generated browser files are missing. The next run attempt should use a disposable copy or pinned Modal image.
- Python environment: yes for current scaffold smokes. Local tests and Modal smokes
  have run, but they prove only that the simplified environment runs.
- Source kinematics: current movement fixtures match through common-trace diff
  with `--python-runner source-kinematics`.
- Wall/border events: the mixed source-border batch passed on 2026-05-09 with
  `6` pass, `0` fail, `0` blocked using `comparison.include_events: true`.
- 3P/4P normal-wall canaries: the multiplayer batch passed on 2026-05-08 with
  `3` pass, `0` fail, `0` blocked using `source-border-rules` and
  `comparison.include_events: true`, including the 4P same-frame terminal draw.
- Source-body canaries: the body batch passed with `6` pass, `0` fail, `0`
  blocked using `source-body-canary`, covering opponent tangent-safe strict
  overlap, opponent overlap-kill, own delta `3` safe, own delta `4` kill,
  same-frame point kill, and same-frame point control.
- Source trail cadence: the trail batch passed with `2` pass, `0` fail, `0`
  blocked using `source-trail-cadence-canary`, covering one normal point/body
  insertion and one below-radius no-point/no-body control.
- Source trail gaps: the gap batch passed with `2` pass, `0` fail, `0` blocked
  using `source-trail-gap-canary`, covering forced hole-space safety and a
  stored-body-in-visual-hole kill.
- Bonuses: narrow JS/Python source-env parity exists for active
  `BonusSelfSmall` catch/no-catch/death-order, one natural one-type
  spawn/type/position RNG fixture, one natural game-world spawn retry fixture,
  and one timed `BonusSelfSmall` expiry/restore fixture. Caps/default weights,
  other effects, other bonus types, and vector/runtime support remain open.

Proof still needed:

- Command or runbook for the original reference.
- Command or test output for the Python environment.
- A tiny deterministic reset/step example.
- Broader event checks beyond the narrow source-body canary, plus broader
  deterministic print-manager behavior, trails, bonus effects/spawn edges,
  observations, and full replay
  messages.

Current links:

- [Environment fidelity handoff](../handoffs/2026-05-08-environment-fidelity-handoff.md)
- [CurvyTron raw run probe](../research/environment/curvytron_raw_run_probe.md)
- [CurvyTron JS state oracle notes](../research/environment/curvytron_js_state_oracle.md)
- [Fidelity comparison](../design/environment/fidelity_comparison.md)
- [Fidelity comparison options](../research/environment/fidelity_comparison_options.md)
- [CurvyTron reference notes](../research/curvytron_reference_notes.md)
- [Local development runbook](../runbooks/local_development.md)
- [Modal environment fidelity runbook](../runbooks/modal_environment_fidelity.md)
- [Deterministic environment design](../design/deterministic_environment.md)

### How Do We Compare?

Concern: "close enough" needs a test, not just vibes.

Compare in layers:

- Constants: speed, turn rate, radius, trail gap, normal-wall rule,
  source borderless rule, scoring rule.
- Golden cases: small scripted situations with known outcomes.
- Rollout fingerprints: same seed plus same actions gives the same trace.
- Videos or screenshots: good for humans, but secondary to data.

Open decision:

- What is the first canonical trace format?
- Which source behaviors must match before the first learning run, and which
  belong only behind a later public training interface?
- Which differences are deliberate `curvyzero-v0` simplifications?
- Should source-fidelity traces use elapsed milliseconds or fixed `1000 / 60` ms steps?
- What should the runner option be called for source-style kinematics?

Current links:

- [Fidelity comparison design](../design/environment/fidelity_comparison.md)
- [Environment fidelity checklist](../design/environment/fidelity_checklist.md)
- [Reference oracle design](../design/environment/reference_oracle.md)
- [Ruleset world model](../design/environment/ruleset_world_model.md)

### Pixels Vs State

Concern: should the agent learn from rendered pixels, or from simulator state?

Current stance:

- Use state-derived observations first.
- Start with rays plus scalar features and an action mask.
- Move to a local raster when CNN/MuZero work needs spatial structure.
- Keep pixels for rendering, debugging, demos, and possible later experiments.

Why:

- State is easier to test.
- Render pixels can change because of browser, canvas, antialiasing, or capture details.
- The simulator should define truth; the renderer should show truth.

### How Do We Single-Step?

Concern: when something goes wrong, we need to pause one tick and see what happened.

Useful single-step output:

- Tick number and seed.
- Ruleset id and config.
- Border mode: fixed toy-v0 wall, normal-wall source mode, or source borderless wrap.
- Before position, heading, alive flag, and trail state.
- Joint action.
- After position, heading, alive flag, and trail state.
- Collision details.
- Reward, done, truncated, and terminal reason.

Possible first form:

- A CLI helper that prints JSON.
- A notebook helper that shows tables and a small render.
- Later, a small web view if that becomes worth it.

Current links:

- [Reference oracle design](../design/environment/reference_oracle.md)
- [Deterministic environment design](../design/deterministic_environment.md)

### How Do We Host On Modal?

Concern: Modal should help us run bigger work, but it should not slow down every game tick.

Current target: no new Modal work for the next local mechanic slice. Keep the
next checks local and state/event based until another coarse batch job actually
needs Modal.

Current stance:

- Local machine: tight edit loop, unit tests, tiny benchmarks.
- Modal CPU: remote smoke tests and environment benchmarks.
- Modal GPU: model, MCTS, training, and evaluation smokes.
- Modal storage: checkpoints, replay chunks, logs, and videos.

Avoid:

- One Modal call per environment step.
- Modal Queue or Dict inside the hot loop.
- Replay layouts with huge numbers of tiny files.

Current links:

- [Modal fidelity jobs](../design/environment/modal_fidelity_jobs.md)
- [Modal reference hosting plan](../design/environment/modal_reference_hosting.md)
- [Modal environment fidelity runbook](../runbooks/modal_environment_fidelity.md)
- [Modal architecture](../design/modal_architecture.md)
- [Modal patterns](../research/modal_patterns.md)
- [Local development runbook](../runbooks/local_development.md)

### How Do We Iterate?

Concern: the work needs a loop that lets us learn without turning the docs into a maze.

Simple loop:

1. Pick one question.
2. Write the smallest test or trace that answers it.
3. Fix the simulator or label the difference honestly.
4. Run local tests.
5. Use Modal only later, when remote behavior matters for a coarse scenario job.
6. Promote the stable answer into design docs.

Do not promote rough notes too early. Keep messy thinking here or in research until the answer is boring enough to become design.

Self-reflection: memory and handoffs are useful starting points, not proof. Source
and probe output win on rule details. If the source/probe result is not ready,
write `pending`.

## Navigation

- Stable map: [design/environment/README.md](../design/environment/README.md)
- Research index: [research/environment/README.md](../research/environment/README.md)
- Handoff: [handoffs/2026-05-08-environment-fidelity-handoff.md](../handoffs/2026-05-08-environment-fidelity-handoff.md)
- Fidelity comparison: [design/environment/fidelity_comparison.md](../design/environment/fidelity_comparison.md)
- Reference oracle: [design/environment/reference_oracle.md](../design/environment/reference_oracle.md)
- Ruleset world model: [design/environment/ruleset_world_model.md](../design/environment/ruleset_world_model.md)
- Modal fidelity jobs: [design/environment/modal_fidelity_jobs.md](../design/environment/modal_fidelity_jobs.md)
- Current simulator contract: [design/deterministic_environment.md](../design/deterministic_environment.md)
- Observation and reward choices: [research/observation_reward_design.md](../research/observation_reward_design.md)

## Current Homes

Use these current docs instead of adding more placeholders:

- `docs/research/environment/curvytron_raw_run_probe.md` - current raw reference run status.
- `docs/design/environment/reference_oracle.md` - first source oracle design.
- `docs/design/environment/ruleset_world_model.md` - normal-wall, fixed toy-v0 wall,
  and source borderless distinction.
- `docs/design/environment/fidelity_comparison.md` - trace and comparison policy.
- `docs/design/environment/modal_fidelity_jobs.md` - Modal batch and artifact design.
- `docs/runbooks/modal_environment_fidelity.md` - future Modal command shape.
