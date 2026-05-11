# Fidelity Comparison Options

Status: Proposed research note

This note lists ways to compare the Python simulator to CurvyTron. It ranks them by
usefulness and effort, and it calls out where each method is a good source of truth.

## Short Answer

Use source facts and deterministic state traces first. They are the cleanest way to
find real rule drift.

Use event logs, collision goldens, and replay files as the day-to-day regression
tools. They make failures small and easy to fix.

Use server-message checks if the Python code needs to drive or match the browser
protocol.

Use browser pixels, screenshots, and videos later. They are useful for visual
acceptance and human review, but they should not be the first source of truth if
state traces are possible. Pixels can change because of rendering, browser timing,
font smoothing, antialiasing, and interpolation even when game state is correct.

## Ranking

| Rank | Method | Usefulness | Effort | Best use | Main limit |
| --- | --- | --- | --- | --- | --- |
| 1 | Deterministic state traces | Very high | Medium-high | Prove per-tick motion, trail, collision, death, and score behavior. | Needs a reference harness with fixed seed, fixed starts, fixed actions, and fixed time steps. |
| 2 | Code constants and source facts | High | Low-medium | Pin constants, formulas, update order, action mapping, scoring rules, and known source behavior. | Does not prove full runtime behavior by itself. |
| 3 | Collision outcome goldens | Very high | Medium | Lock down wall, self, trail, head-head, tie, grazing, and tunneling cases. | Only covers cases we write down. |
| 4 | Event logs | High | Medium | Explain why traces diverge by showing semantic events. | Needs stable event names and enough detail. |
| 5 | Replay files | High | Medium | Build a corpus of seeds/actions/outcomes for regression and debugging. | A replay is only as good as the trace and metadata stored with it. |
| 6 | Server messages / wire events | Medium-high | Medium-high | Check browser interoperability and client-visible behavior. | Transport shape can match while internal game rules still drift. |
| 7 | Browser pixels / screenshots | Medium later | High | Catch renderer, camera, color, trail, and UI drift after state is trusted. | Noisy as a rule oracle. Should not lead if state traces are available. |
| 8 | Videos | Medium for review, low as oracle | Medium-high | Judge feel, communicate failures, and review long episodes. | Hard to diff, large artifacts, weak exactness. |

## Comparison Methods

### 1. Code Constants And Source Facts

Compare facts mined from CurvyTron source to Python config and docs.

Useful facts include:

- Motion constants: base speed, turn rate, radius, speed floor, and timestep model.
- Action mapping: left, right, neither, both, inverse controls, and straight-angle mode.
- Trail rules: print distance, hole distance, trail point spacing, delayed printing, and
  self-collision latency.
- Collision rules: strict circle overlap, wall boundary, borderless wrap, update order,
  and invincibility.
- Spawn rules: arena size, spawn margins, heading margins, and overlap checks.
- Scoring rules: death count at frame start, same-frame deaths, survivor score, max
  score, and game end.
- Bonus rules: enabled types, spawn cap, pop timing, probability, duration, and effect
  stacking.

This is low effort and should happen first. It gives a checklist and catches many
obvious mistakes. It cannot prove that a rollout is right, because code paths and timing
can still differ.

### 2. Deterministic State Traces

Run the same scenario in CurvyTron and in Python, then diff state at each tick.

A good trace row should include:

- Scenario id, ruleset id, source commit, simulator commit, config hash, and seed.
- Tick index and elapsed time in milliseconds.
- For each player: alive flag, x, y, angle, speed, turn input, angular velocity, radius,
  printing flag, trail point count, death cause, killer, score, and round score.
- Global state: active bonuses, border mode, round phase, death count, winner, and
  terminal flag.
- Semantic events emitted during the tick.

This is the best source of truth if we can make the reference deterministic. It shows
the first exact place where the Python clone diverges. The main work is building a
reference runner that can force initial state, seed/random stream, actions, and time
steps. If the reference cannot be made fully deterministic, traces can still help on
scripted scenarios where randomness is removed.

### 3. Collision Outcome Goldens

Write small focused cases that ask, "Who dies, why, and when?"

Important cases:

- Wall collision at each side, including radius boundary behavior.
- Equal-distance circle contact stays alive; epsilon overlap kills.
- Self-collision latency threshold.
- Own trail, opponent trail, and old head/body collision.
- Head-head collision and same-tick deaths.
- Grazing and near-miss cases.
- Fast movement that might tunnel through a trail.
- Borderless wrap, once that mode is in scope.
- Invincible and bonus-driven size/speed changes, once bonuses are in scope.

These tests are high value because collision details decide the game. They should use
small fixtures with explicit provenance labels: `source-derived`, `source-inspired`,
`v0-choice`, or `unresolved`.

### 4. Event Logs

Event logs record what happened, not every numeric field.

Useful events:

- `spawn`
- `input_applied`
- `position_updated`
- `trail_point_added`
- `print_started`
- `print_stopped`
- `hole_started`
- `hole_ended`
- `collision_checked`
- `collision_hit`
- `wall_hit`
- `bonus_spawned`
- `bonus_caught`
- `bonus_started`
- `bonus_ended`
- `death`
- `round_ended`
- `score_resolved`
- `server_message_sent`

Event logs are not as exact as state traces, but they make diffs easier to read. They
are especially useful when a trace first diverges and we need to know whether the cause
was input, motion, trail printing, collision, or scoring.

### 5. Replay Files

A replay file should be a portable scenario plus enough output to make it useful.

Minimum fields:

- Ruleset id and version.
- Source commit and Python implementation commit.
- Config hash, observation hash, reward hash, and trace schema version.
- Seed streams and sampled episode parameters.
- Initial state or spawn fixture.
- Per-tick action inputs.
- Terminal outcome, scores, death causes, and a small trace fingerprint.
- Optional full state trace, event log, screenshots, or video references.

Replay files are good regression artifacts. They let us rerun a known bug, make a
rendered debug view, and check that future changes did not move the outcome. They should
not hide the exact ruleset and hashes, because old replay data can become misleading
after rule changes.

### 6. Server Messages / Wire Events

Compare the messages that CurvyTron sends to the browser, and later what the Python
server or bridge sends.

Good checks:

- Message names and order for game start, round start, input, avatar update, death,
  score, bonus, round end, and game end.
- Numeric compression and rounding, if the Python side mirrors the protocol.
- Client-visible fields after one tick, one death, one score update, and one bonus.
- Reconnect or replay message shape, if used.

This is useful if browser demos need to attach to the Python simulator or if we want a
browser-side oracle. It is less useful for training-only correctness, because protocol
compatibility is not the same thing as rule fidelity.

### 7. Browser Pixels And Screenshots

Compare screenshots from CurvyTron and from the Python-driven renderer.

Good uses:

- Check that trail thickness, colors, camera scale, arena size, bonus sprites, and death
  markers look right.
- Catch UI or protocol drift after game state already matches.
- Produce visual artifacts for failed goldens.

This should be later-stage evidence. Pixel diffs are noisy and can fail for reasons that
do not matter to the simulator: antialiasing, frame timing, browser version, canvas
scaling, interpolation, fonts, or CSS. Pixel comparison may become valuable after state
and event traces are trusted, but it should not be the first source of truth if state
traces are possible.

### 8. Videos

Record short videos of reference and Python runs.

Good uses:

- Review human feel: speed, turn response, trail gaps, crash timing, and round rhythm.
- Share bug reports in a way humans can understand quickly.
- Spot large visual or timing problems that exact tests missed.

Videos are weak exact tests. Use them as review artifacts, not as the main oracle.

## Best Order To Build

1. Source-fact checklist from the reference code.
2. Python trace schema and trace emitter.
3. Small reference trace runner for fixed starts, fixed actions, and fixed `dt`.
4. Collision golden fixtures.
5. Replay fixture format.
6. Event-log diff view.
7. Server-message checks, if browser interoperability is needed.
8. Screenshot and video capture for accepted scenarios.

## Holes And Questions

- Can the CurvyTron reference run in a deterministic headless mode, or do we need to patch
  time and randomness?
- Can we force exact initial positions, headings, alive flags, trail bodies, and bonus
  state in the reference?
- Which reference target is first: original CurvyTron v1 source, CurvyTron 2 public
  behavior, or a named `curvyzero-v0` training ruleset?
- Should source-fidelity traces use elapsed milliseconds like the reference, or fixed
  `1000 / 60` ms steps for deterministic comparison?
- How close must Python floating point be to JavaScript floating point before we call a
  trace row a match?
- Which collision backend is allowed for a source-inspired ruleset: endpoint circles,
  swept circles, occupancy grid, or more than one backend with parity checks?
- What trace fields are required before a fixture can be promoted to a golden test?
- How much of the wire protocol matters for training, and how much only matters for a
  browser demo?
- Which browser, viewport, device scale, and rendering options should be fixed before
  screenshot diffs are meaningful?
- Where should golden fixtures, replay files, screenshots, and videos live so CI stays
  small and fast?
