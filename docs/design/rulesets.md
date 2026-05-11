# Rulesets

Status: Draft

The project needs two related but distinct ideas:

- A target ruleset that defines what game the agent is meant to master.
- Training variants that intentionally perturb the target ruleset for curriculum or robustness.

## Proposed Names

- `curvyzero-v0`: first explicit 1v1 no-bonus training ruleset.
- `curvytron-v1-reference`: behavior derived from the original CurvyTron repo.
- `curvytron2-reference`: behavior derived from public CurvyTron 2 rules or gameplay notes.
- `curvyzero-robust-*`: later domain-randomized variants.

## `curvyzero-v0`

`curvyzero-v0` is intentionally a simplified training environment, not a claim of exact
CurvyTron source fidelity.

- `v0-choice`: 2 players only.
- `v0-choice`: 64x64 grid arena by default.
- `v0-choice`: one episode is one round, with immediate movement after reset.
- `v0-choice`: fixed per-tick speed and turn rate.
- `v0-choice`: occupancy-grid trail collision with solid trails.
- `v0-choice`: no bonuses, no borderless mode, no warmup/warmdown, and no match-level max
  score ladder.
- `wrapper-choice`: a trainer decision supplies public action ids for all live players,
  then the wrapper maps them to held source-style controls before resolving rewards.
  A strict public env should record the native control model id, trainer wrapper id, and
  decision window (`decision_ms`) for replay/profile metadata.

The config may expose source-reference metadata for comparison, but that metadata is not
part of the active `curvyzero-v0` rule semantics.

## `curvytron-v1-reference`

The local CurvyTron v1 source points to these source-derived target semantics for a future
fidelity ruleset:

- 60 Hz target server loop with elapsed wall-clock milliseconds as the physics step.
- Native control is held player input state over elapsed-ms server frames, not
  `step(joint_action)` with discrete simultaneous actions. A fixed decision cadence is a
  CurvyZero wrapper/ruleset choice.
- Map size formula: `round(sqrt(80^2 + ((players - 1) * 80^2 / 5)))`.
- Avatar defaults: velocity `16`, angular velocity base `2.8 / 1000` radians/ms,
  radius `0.6`, velocity floor `8`, and self-collision latency `3`.
- Controls: left-only `-1`, right-only `1`, neither or both `0`; inverse controls flip
  the sign.
- Trail holes: distance-based print manager with base print distance `60` and base hole
  distance `5`; randomized ranges are applied by the reference implementation.
- Round timing: `3000` ms warmup, `5000` ms warmdown, and another `3000` ms delay before
  trail printing starts after `game:start`.
- Collision: circular body overlap is strict (`distance < radius_a + radius_b`), with
  endpoint checks after movement and before print-manager/bonus processing.
- Scoring: same-frame deaths share the same frame-start death-count score; the round winner
  receives `max(players - 1, 1)` extra score; tied leaders at max score continue.
- Bonuses: default enabled set includes self, enemy, all-color, game-borderless, and
  game-clear effects; base bonus radius is `3`, default duration is `5000` ms, spawn cap is
  `20`, and base pop time is `3000` ms adjusted by `bonusRate`.

## Versioning Rule

Changing behavior that can affect trajectories or rewards should create a new ruleset version or an explicit config migration note.

`rules_hash` should cover behavior-affecting config fields for the active ruleset. Source
reference metadata that is present only for documentation or future fidelity work should not
change the hash.

## Parameters Likely To Vary Later

- Arena size.
- Speed.
- Turn rate.
- Trail gap period and gap length.
- Trail width.
- Trainer decision cadence or action repeat.
- Spawn spacing and heading.
- Player count.
- Bonus availability.
- Observation noise or partial observability.

## Guardrails

- Robustness variants must not silently replace the target ruleset.
- Evaluation should include fixed canonical rules and held-out randomized variants.
- Golden tests should pin target behavior even if training randomizes around it.
- Replay and checkpoints should eventually store a rules hash, observation schema hash, reward schema hash, and implementation metadata.

## Known Fidelity Notes

- CurvyTron reference behavior appears to include trail gaps/holes via `PrintManager`; a solid-trail v0 should be labeled as a deliberate simplified ruleset, not silently treated as exact reference behavior.
- The reference uses elapsed-millisecond integration, while `curvyzero-v0` currently uses
  fixed per-tick movement. A source-fidelity ruleset should make that difference explicit.
- The reference collision model is circular body/island lookup with strict overlap and
  self-collision latency. The current grid collision backend is a v0 implementation choice.
- `BonusSelfGodzilla` exists in source but is absent from default room config, server mapping,
  and client sprites; treat it as unreachable unless a hidden config source is found.
