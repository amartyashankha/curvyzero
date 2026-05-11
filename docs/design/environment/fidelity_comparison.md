# Fidelity Comparison

Status: Draft

This page defines how we compare the Python reconstruction to CurvyTron. The goal is to
catch rule drift early, keep fixes small, and avoid treating visual similarity as proof
that the simulator is correct.

Supporting research: `docs/research/environment/fidelity_comparison_options.md`.

## Recommendation

Use this order of truth:

1. Source facts for constants, formulas, update order, and rules.
2. Deterministic state traces for per-tick behavior, when the reference can be run in a
   controlled way.
3. Behavior and golden tests for collision, scoring, and other small cases.
4. Replay files for regression, debugging, and artifact review.
5. Server-message checks when browser interoperability matters.
6. Browser screenshots and videos for visual and human-feel review.

Pixel comparison may be useful later, but it should not be the first source of truth if
state traces are possible. A pixel diff can fail because rendering changed while game
state is still correct. A state trace shows the rule difference directly.

## Fidelity Levels

| Level | Name | Meaning | Exit check |
| --- | --- | --- | --- |
| 1 | Source-fact match | Python config and docs match facts mined from CurvyTron source. | Constants, formulas, update order, and rule labels are recorded with source links. |
| 2 | Deterministic state trace match | Same scenario, seed, actions, and time steps produce the same state over time within agreed tolerances. | First N canonical traces match for position, angle, trail counts, collisions, deaths, and scores. |
| 3 | Behavior / golden-test match | Important edge cases produce the expected outcome even if internals differ. | Golden tests pass for wall, self, opponent, head-head, same-tick death, scoring, spawn, trail gaps, and bonus cases in scope. |
| 4 | Browser pixel match | Browser output is visually close for trusted scenarios. | Fixed screenshots or short frame sequences match within a visual tolerance. |
| 5 | Human-feel match | The game feels right to a person playing or watching it. | Reviewers accept speed, turning, trail behavior, crash timing, scoring rhythm, and overall play feel. |

These levels stack. A human-feel match does not replace trace or golden-test evidence.

## What To Compare

| Question | First comparison | Backup comparison |
| --- | --- | --- |
| Are constants and formulas copied correctly? | Source-fact checklist. | Small unit tests for each formula. |
| Does movement match over time? | Deterministic state trace. | Fixed action rollout fingerprint. |
| Does trail printing match? | State trace plus event log. | Golden cases for print and hole boundaries. |
| Does collision match? | Collision outcome goldens plus trace. | Rendered replay for inspection. |
| Does scoring match? | Golden scoring tables and state trace. | Replay outcome summary. |
| Does the browser receive the same state? | Server-message diff. | Browser screenshot after trusted state diff passes. |
| Does it look right? | Screenshot diff. | Short video review. |
| Does it feel right? | Human play/watch review. | Episode metrics: length, death causes, win rates, and trail density. |

## Repeatable Iteration Loop

1. Choose one scenario.
   Keep it small. Name the ruleset, source target, player count, seed, initial state,
   action script, time-step policy, and expected comparison method.

2. Run the reference.
   Record source commit, Node/browser version if used, config, seed, actions, sampled
   parameters, trace schema, event log schema, and artifact paths.

3. Run the Python clone.
   Use the same scenario data. Emit the same trace fields and event names where possible.

4. Diff in a fixed order.
   Start with source facts and config, then state trace, then event log, then collision
   and scoring outcomes, then replay summary, then server messages, then pixels or video
   if the scenario needs them.

5. Fix the first real mismatch.
   Prefer the smallest rule or config fix. If the mismatch is a deliberate v0 choice,
   mark it as `v0-choice` instead of forcing source behavior.

6. Add a test or fixture.
   Store the scenario as a golden test or replay fixture. Include provenance, ruleset id,
   config hash, action script, expected outcome, and any tolerance.

7. Repeat.
   Promote stable scenarios into the regression set. Keep failed scenarios easy to render
   or inspect.

## Diff Policy

- Numeric traces should use explicit tolerances for position, angle, and time. The
  tolerance belongs in the fixture, not only in test code.
- Exact fields should stay exact: alive flags, action inputs, death causes, killer ids,
  score events, terminal flags, and event order.
- The first divergent tick is the main debug clue. Reports should show that tick and the
  previous tick.
- If source behavior is ambiguous, write the ambiguity down and choose a named policy for
  the active ruleset.
- If Python intentionally differs from CurvyTron, label the fixture as `v0-choice` or
  `source-inspired`.

## First Scenario Set

Start with these:

| Scenario | Main level | Why |
| --- | --- | --- |
| Fixed straight movement for 60 ticks | State trace | Checks speed, timestep, and position integration. |
| Constant left turn for 60 ticks | State trace | Checks action mapping and turn rate. |
| Wall hit from a known position | Golden test | Checks radius and boundary policy. |
| Equal-distance body touch | Golden test | Checks strict collision overlap. |
| Self-collision latency threshold | Golden test | Catches off-by-one errors. |
| Opponent trail hit | Golden test | Checks cross-player body matching. |
| Same-tick double death | Golden test | Checks update order and scoring. |
| Trail print/hole boundary | Trace plus event log | Checks distance-based printing once source gaps are in scope. |
| One full fixed replay | Replay fixture | Checks that the whole path stays stable. |

## Artifact Contract

Each comparison artifact should include:

- Scenario id.
- Ruleset id and version.
- Source target and source commit.
- Python implementation commit when available.
- Config hash, observation hash, reward hash, and trace schema version.
- Seed streams and sampled parameters.
- Source control script.
- Expected outcome and allowed tolerances.
- Provenance label: `source-derived`, `source-inspired`, `v0-choice`, or `unresolved`.

## Holes And Questions

- Can we run the CurvyTron reference with fixed time and fixed randomness without changing
  behavior?
- Can the reference accept forced initial state, or do we need a JS test harness that
  builds game objects directly?
- Which target gets priority first: `curvytron-v1-reference`, `curvytron2-reference`, or
  `curvyzero-v0`?
- Do we compare elapsed-millisecond reference traces directly, or compare against a fixed
  60 Hz source-fidelity mode?
- What is the accepted tolerance for JavaScript-vs-Python floating point drift?
- Which collision cases are source facts, and which are project choices made for a faster
  training backend?
- How much server protocol fidelity is required before the browser demo work starts?
- What is the smallest screenshot/video suite that helps humans without slowing CI?
