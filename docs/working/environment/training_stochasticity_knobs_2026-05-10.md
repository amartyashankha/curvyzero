# CurvyTron Training Stochasticity Knobs - 2026-05-10

Status: current Coach training note.

Latest proof, 2026-05-11 20:10 EDT: Modal smoke
`curvytron-two-seat-repeat-smoke-s6101-20260511` exercised the two-seat
policy-action-repeat path with `policy_action_repeat_max=3` and
`policy_action_repeat_extra_probability=0.5`. It passed, changed weights, saved
checkpoints, and logged `8` fresh policy decisions plus `8` reused seat actions
across `16` active seat rows. This proves the knob runs; it is not a learning
claim.

This note keeps source-environment fidelity separate from training-time
stochasticity. The source env should stay source-faithful and replayable. Extra
training noise belongs in the trainer, collector, or wrapper layer unless we are
explicitly changing CurvyTron rules.

## Current Coach Decision

- CurvyTron Hail Mary runs should use the current-policy two-seat self-play
  path: one current policy chooses for both seats, the trainer builds the joint
  action, and replay records both seats.
- Do not use the native fixed/frozen-opponent path for that Hail Mary unless the
  run is explicitly a fixed-opponent control.
- The native source-state fixed-opponent path is still useful. It exercises
  stock LightZero `train_muzero`, checkpointing, renderer/device plumbing, and
  control baselines. It is not current-policy two-player self-play.
- Start CurvyTron reward simple: survival time. Do not add reward shaping until
  the simple survival signal and reporting are understood.

## Simple Split

RNG provenance for fidelity and replay is not the same thing as training noise.

For fidelity, RNG provenance means:

- what seed or random tape produced a reset, spawn, bonus, or lifecycle event;
- which row/player consumed which random value;
- enough metadata to replay or compare the same source-backed case later.

That is bookkeeping for truth. It should make tests and replay more exact, not
more random.

For training, stochasticity has two layers:

- source-like randomness that belongs to CurvyTron itself, such as reset/spawn,
  timers, bonuses, and opponent variation;
- extra robustness variation added by us, such as visual jitter, arena jitter,
  or curriculum noise.

Fidelity checks should be controlled and replayable, not because the game is
deterministic, but because debugging parity needs the same random choices on
both sides. A failing parity test should not depend on today's lucky random
draw. If a test needs a seeded random tape, the tape should be named, stored,
and replayable.

## Knob We Want Now

The active stochasticity knob is policy-action-repeat/dropout at the
trainer/wrapper layer, not inside source env mechanics.

Plain behavior:

- The policy chooses an action for a seat.
- The wrapper may execute that chosen action now, or hold/repeat that seat's
  last executed action for a small number of physical steps.
- After the repeat window ends, the policy gets another decision for that seat.
- The source env only receives the executed joint action. It should not hide
  this policy dropout inside CurvyTron mechanics.

For two-player self-play, schedules must be per env row and per seat. Do not
use one shared global repeat/dropout schedule across all players. Each row/seat
needs its own seed or salt so decorrelation is visible and replayable.

Log at least:

- policy-selected action;
- executed action;
- repeat/dropout count and remaining repeat steps;
- env row, seat id, seed, and salt.

Keep no-op or action-override noise off for learning claims unless it has been
audited. Overrides can make the searched action differ from the executed action,
which can poison value/policy attribution and make a run look less honest than
it is.

Implementation note for the current two-seat path:

- default repeat is off: min `1`, max `1`, extra probability `0`;
- repeat draws are per active env row and seat;
- a repeated seat reuses its last executed action without a fresh policy search;
- replay rows are written for fresh policy decisions, and physical action counts
  are logged separately so repeated behavior remains visible.

## Later Knobs

Possible later domain-randomization or training-noise knobs:

- reset seed variation across episodes or curriculum phases;
- spawn jitter for position, heading, or retry policy within source-valid
  bounds;
- arena size variation;
- speed, turn-rate, and radius jitter;
- decision cadence and action-repeat jitter, if implemented outside source env
  mechanics and logged per row/seat;
- bonus spawn/type probability schedules;
- opponent policy mixtures for ego-row training;
- observation noise, color jitter, and scale jitter for visual training;
- frame stack variation or controlled frame dropout;
- curriculum schedules that widen any of the above only after baseline fidelity
  and learning checks are stable.

These knobs should not silently change source rules. If they become useful, add
them behind explicit ruleset/training configuration fields, record the active
values in replay or run metadata, and keep a controlled source-fidelity proof
path beside them.

## Current Stance

Do not block current environment work on extra knobs. The base no-repeat
two-seat run is still the clean first Hail Mary. Repeat/dropout variants are
now available for robustness sweeps after that clean run is alive.
