# Fidelity Loop Self-Critique

Status: Draft critique
Date: 2026-05-08

## Read

- `docs/design/environment/README.md`
- `docs/design/environment/fidelity_checklist.md`
- `docs/design/environment/fidelity_comparison.md`
- `docs/design/environment/reference_oracle.md`
- `docs/design/environment/scenario_schema.md`
- `docs/design/environment/trace_loop_contract.md`
- `docs/design/environment/trace_schema.md`
- `docs/design/environment/modal_fidelity_jobs.md`
- `docs/design/environment/multiplayer_canaries.md`
- `docs/design/deterministic_environment.md`
- `docs/design/rulesets.md`
- `docs/experiments/environment/2026-05-08-js-scenario-runner.md`
- `docs/experiments/environment/2026-05-08-python-scenario-trace-and-diff.md`
- `docs/experiments/environment/2026-05-08-local-fidelity-loop.md`
- `docs/experiments/environment/2026-05-08-trace-normalization.md`
- `docs/experiments/environment/2026-05-08-common-trace-diff.md`
- `docs/experiments/environment/2026-05-08-source-kinematics-first-match.md`
- `docs/research/observation_reward_design.md`

## Short Answer

The core loop is not too complicated:

```text
scenario JSON -> JS trace -> Python trace -> common trace -> diff
```

That is the right shape. The risk is around it: too many schema names, too many
artifact layouts, raw diff mode, several future Modal commands, and too much
discussion of browser/pixel/server work before the state loop is useful.

The simplest useful path is:

1. Keep the local loop as the main tool.
2. Compare common traces only.
3. Fix one source mechanic at a time.
4. Add one tiny scenario for each new mechanic.
5. Use Modal only when the same local batch needs a remote runtime or shared
   artifact storage.

Do not build a larger framework yet.

## What Is Working

- The headless JS runner is the right first oracle. It calls source JS objects
  and avoids the old browser/server build.
- The Python runner is honest. `curvyzero-v0` is labeled as a toy training
  ruleset, not source fidelity.
- The source-kinematics runner is scoped well. It matches the current one-step
  movement fixtures without pretending to cover collisions, trails, or scoring.
- The common trace projection is the right comparison layer. Raw JS and Python
  payloads should remain debug artifacts.
- `pass`, `fail`, and `blocked` are better than one boolean, because some runs
  cannot make a valid source-fidelity claim.
- The Modal rule is good: batch-level only, no per-tick calls.

## Where We Are Overcomplicating

### Too many scenario shapes

The docs mention both the older shape:

```text
schema_version, id, source_setup, steps[].tick, moves[]
```

and the newer shape:

```text
schema, scenario_id, world, time, steps[].step_index, moves{}
```

The code still accepts aliases. That is fine for migration, but the design docs
should pick one shape and call the rest compatibility. Otherwise every runner,
normalizer, and artifact reader becomes a schema translator.

Update: the active contract now treats the accepted `environment-scenario-v0`
shape as the current write shape. A future `environment_scenario/v1` migration
should be a separate, tested change, not half-applied in docs.

### Raw diff should stop being a normal path

Raw diff was useful to prove plumbing. It now mostly reports runner metadata
differences, such as JS `loadedSources`.

Recommendation: keep raw trace files for debugging, but make common-trace diff
the default and only meaningful fidelity result.

### The artifact plan is ahead of the evidence

The stable target layout with `input.json`, `js/common_trace.json`,
`python/common_trace.json`, `diff/report.json`, and `manifest.json` is enough.
The larger Modal tree with many nested manifests, JSONL streams, summaries, and
exports can wait until traces are actually large.

Recommendation: for local and first Modal batches, write a small scenario
folder and one final manifest. Add more structure only after a real batch hurts.

### Modal has too many planned entry points

`js-probe`, `python-probe`, `diff`, `batch`, deploy commands, and fetch scripts
are useful later. For now, one local command and one future Modal batch command
are enough.

Recommendation: make `fidelity_batch` the first remote shape. Split stages only
after rerunning one stage saves real time.

### Browser, server, and pixels are too early for the main loop

Browser hosting, websocket payloads, screenshots, videos, and pixel diffs are
valuable later. They are weak first proof for source rules.

Recommendation: state and events first. Pixels only after the state trace can
explain what should be on screen.

## Simplest Path To Useful Source Fidelity

### Step 1: Keep the target honest

Keep two names separate:

- `curvyzero-v0`: small training ruleset.
- `curvytron-v1-reference`: source-derived behavior.

Do not silently move `curvyzero-v0` toward the source. Add a source-fidelity
mode or ruleset for source-derived checks.

### Step 2: Lock movement first

The source-kinematics runner has already shown the right pattern. Expand from
one forced step to:

- straight movement for 60 ticks
- constant left turn for 60 ticks
- constant right turn for 60 ticks
- mixed left, straight, right script for 60 ticks

Compare only:

- `x`
- `y`
- `angle`
- `alive`
- `step_ms`

### Step 3: Add collisions without trail gaps

Next scenarios:

- wall hit
- equal-distance body touch, where strict overlap should be safe
- opponent trail hit from a manually seeded body
- same-frame 2-player wall death

Add common trace fields only when needed:

- `death_tick`
- `death_cause`
- maybe `hit_owner`

### Step 4: Add trail printing and holes

Trail holes are source behavior, not a bonus. Still, they can wait until basic
movement and collision pass.

First fields:

- `printing`
- `trailPointCount`
- `lastTrailPoint`
- `bodyCount`

Do not model every print-manager internal in the common trace unless a mismatch
needs it.

### Step 5: Add multiplayer scoring canaries

Use the existing 2-player, 3-player, and 4-player canaries. They are cheap and
catch bugs a 1v1 loop hides.

First checks:

- source map size for 2, 3, and 4 players
- reverse JS update order where source behavior exposes it
- same-frame death order
- same-frame score capture
- final survivor score

### Step 6: Add randomness last

Forced state should stay the default for physics. Random spawn and trail-hole
randomness should become separate scenarios after deterministic forced cases
pass.

When randomness enters, record:

- seed or random stream name
- sampled spawn positions and headings
- sampled trail print/hole lengths
- sampled bonus state, if bonuses are enabled later

## Coverage Check

| Surface | Current coverage | Practical next move |
| --- | --- | --- |
| State | Good first slice. Common trace covers step, player id, position, angle, alive, score when present. | Keep it small. Add fields only for the next failing mechanic. |
| Events | JS runner records source events. Common diff does not compare them yet. | Add event comparison after state movement/collision passes. Start with `position`, `angle`, `die`, and `score:round`. |
| Observations | Current learned observation work is separate from source-fidelity traces. `CurvyTronEnv._observations()` is a global debug vector. | Keep observation fidelity as its own test lane. Do not block state-source parity on learned observations. |
| Image/raster observations | Local raster is planned for learning; browser pixels are deferred. | Test simulator-native rasters before browser screenshots. Use pixels only for render/demo checks later. |
| Multiplayer | Correctly called out in docs and canaries. Not yet part of the one-command loop. | Add 3-player and 4-player forced canaries before claiming source fidelity. |
| Scoring | Source facts are documented; canaries define good first checks. | Make scoring table tests for 2, 3, and 4 players. Compare events and final state. |
| Randomness | Mostly deferred. Forced state is used for first scenarios. | Keep forced state for mechanics. Add seeded random tests only for spawn, trail holes, and bonuses. |
| Final training observations | Replay metadata plans mention observation hashes, masks, terminal observations, and joint actions. | Before training claims, store observation schema id/hash, terminal observation, legal mask, joint action, reward-next, and ego id. |

## What Not To Build Yet

- A full browser/server host for the first fidelity loop.
- Pixel-perfect screenshot comparison as a source oracle.
- A large event bus or trace database.
- A generic multi-run artifact framework.
- Per-tick Modal calls.
- Full bonus semantics before no-bonus movement, collision, trail, and scoring
  checks pass.
- A single schema that tries to describe state traces, replay, observations,
  browser messages, and training data at once.

## Top Recommendations

1. Make common-trace diff the default result.
2. Freeze one scenario write shape and treat old fields as read-only
   compatibility.
3. Keep source-fidelity work mechanic-by-mechanic: movement, collision, trail,
   scoring, randomness, bonuses.
4. Add 3-player and 4-player canaries early, even if training remains 1v1.
5. Keep observation fidelity separate from source state fidelity, but require
   observation schema hashes and terminal observations before training claims.
6. Use Modal only for whole local-loop batches and artifact storage.
