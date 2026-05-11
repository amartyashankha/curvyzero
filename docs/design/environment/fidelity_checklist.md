# Environment Fidelity Checklist

This is the plain checklist for reconstructing CurvyTron. If our Python environment does not match one of these on purpose, the difference must be named and tested.

## What Must Match

### Game Setup

- Player count rules.
- Multiplayer map-size changes.
- Player ordering.
- Map size formula.
- Spawn positions.
- Spawn headings.
- Random seed behavior, if we can control it.
- Round warmup and start timing.
- Round end timing.

### Controls

- What inputs exist: left, right, straight, no input.
- How input state is stored.
- When input changes take effect.
- Whether controls are continuous between ticks or sampled once per tick.
- Native source CurvyTron advances elapsed-ms frames with held control state; it
  is not a discrete simultaneous trainer-action transition by default.
- Reversed controls and other bonus-modified controls later.

### Motion

- Coordinate system.
- Units.
- Speed.
- Turn rate.
- Time delta handling.
- Tick rate target.
- What happens when a frame is late or early.

### Trails

- Trail width/radius.
- Trail start delay.
- Trail gaps/holes.
- Whether dead players write trail on the death frame.
- Whether spawn area starts occupied.
- How trails are stored internally.

### Collision

- Normal border collision: source says a wall/border hit kills when the avatar
  body crosses the map edge with avatar-radius margin.
- Source borderless border behavior: source says the timed `borderless` bonus
  uses margin `0` and wraps to the opposite edge instead of killing.
- Source borderless caveats: it is not a clean torus. It loses overshoot, uses
  exact opposite-edge placement, handles only the first border axis found, and
  skips body collision on the wrap frame. Exact edge equality is safe.
- Own trail collision.
- Other trail collision.
- Head-to-head collision.
- Three-or-more-player same-frame collisions.
- Same-frame collisions.
- Self-collision delay.
- Exact comparison rule, such as strict `<` versus `<=`.
- Whether collision checks use only the new head position or a swept path.

### Scoring

- Score per death.
- Same-frame death score.
- Winner score.
- Draw score.
- Multiplayer rank/score behavior.
- Whether score depends on frame-start death count.
- Match score versus round score.

### Bonuses

- Which bonuses exist.
- Spawn timing.
- Spawn position rules.
- Catch radius.
- Duration.
- Stacking rules.
- Effect on movement, controls, trail, and collision.

### Server And Client Boundary

- Server state fields.
- Network messages.
- Message timing.
- Per-player messages and player perspective.
- Client interpolation, if any.
- Rendering scale and camera behavior.
- UI-visible state.

## Ways To Compare

### Best First: State Traces

Run the original JS logic and Python logic from the same start state and same inputs. Compare after every step:

- position
- heading
- alive/dead
- score
- trail/print state
- bonus state
- emitted events

This is the best first target because it checks the game, not the drawing.

### Good Early: Golden Outcomes

Small scenarios with known results:

- normal wall hit kills
- source borderless edge crossing wraps with exact source caveats
- hit own trail
- hit other trail
- two players die same frame
- trail gap avoids collision
- trail gap does not avoid collision
- bonus catch changes speed

### Useful Later: Server Messages

Compare the messages the original server sends to clients against messages or trace events from our clone.

### Useful Later: Pixels

Render the original browser and our debug renderer, then compare screenshots or videos. This catches display bugs, but it is a weak first test because tiny drawing differences can hide or exaggerate game differences.

## Iteration Loop

1. Pick one tiny scenario.
2. Run it in the JS reference.
3. Save a state trace.
4. Run the same scenario in Python.
5. Diff the traces.
6. Fix one mismatch.
7. Add a Python test.
8. Repeat.

## Early Canaries

Do these early, even before the full clone is fast:

- A 1v1 scripted normal-wall death or trail collision.
- A 1v1 scripted source borderless wrap.
- A 2-player forced movement step from the JS oracle.
- A 2-player same-frame death.
- A 3-player round where two players die together and one survives.
- A 4-player setup check for map size, spawn order, and player ordering.
- A whole-round fingerprint: hash the important state after every tick for a scripted input sequence.

## Fidelity Levels

- Level 0: constants documented.
- Level 1: simple golden outcomes match.
- Level 2: state traces match for scripted scenarios.
- Level 3: server events/messages match.
- Level 4: browser pixels/videos match closely enough.
- Level 5: human play feels the same.

Training should not wait for Level 5. But source-faithful reconstruction work should aim for Level 2 before serious learning claims.

## Multiplayer Warning

CurvyTron is multiplayer. A 1v1 clone can be useful for training v0, but it is not enough to prove the game was reconstructed. Fidelity tests need at least some 3-player and 4-player scenarios because map size, spawn layout, same-frame deaths, scoring, player order, and per-player messages can behave differently once more than two players exist.

## Honesty Rule

Current tests on the Python toy environment prove only that the toy environment is deterministic. They do not prove CurvyTron fidelity. A test should say what it proves: toy-v0 behavior, source-derived behavior, or browser/demo behavior.

Self-reflection: do not trust memory or a handoff for rule details. Treat source
and probe output as the winner. If a source/probe result is missing, say
`pending` instead of filling the gap from memory.

## Current Oracle Status

We are not at full fidelity. The target is one fast faithful environment.
`CurvyTronSourceEnv` and the original JS source path are proof tools while
source rules move into that fast path.

Source mining says normal mode uses walls. Source `borderless` is a timed bonus
with margin `0`, exact opposite-edge wrap, lost overshoot, one-axis-first border
handling, and skipped body collision on the wrap frame. The narrow normal-wall,
source borderless, 3P/4P normal-wall, and six-case `source-body-canary` traces
are now verified through the local JS/Python common-trace loop. The body canary
is limited to strict opponent overlap, own-body point delay, and two direct
same-frame point materialization fixtures.

Lifecycle/spawn/RNG has pinned lifecycle fixtures through focused 4P all-dead
warmdown/next-round and focused present/non-present survivor scoring,
including one focused all-present 3P `max_score: 2` match-end path and one
focused all-present 3P multi-round match-end path. The active
bonus proof covers seeded `BonusSelfSmall` catch, strict-overlap no-catch, and
same-tick wall-death no-catch/death-order in Python/source-env. The natural
one-type bonus spawn/type/position RNG fixture also has JS/Python source-env
parity.

Next mismatch order: broader 4P match lifecycle beyond the focused all-dead and
survivor-next-round cases, broader present/non-present variants beyond the
focused first-round, survivor-scoring, and next-round cases, vector lifecycle, trainer/replay/final
observation, broader trail/gap cases, more collision shapes, bonus
retries/caps/effects for spawned multi-type bonuses/expiry, observations, and replay/message
behavior. Speed/vector work should not distract from reproduction; it can
continue only when directly tied to promoted source claims. The short plan is in
[next_mismatch_plan.md](../../research/environment/next_mismatch_plan.md), and
the compact current proof map is in
[coverage_tracker.md](../../working/environment/coverage_tracker.md).
