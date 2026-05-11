# Pong State Critique - 2026-05-09

Status: historical critique. The active Pong spine is
`docs/working/pong_selfplay_training_plan_2026-05-09.md` plus
`docs/working/pong_training_critique_wave_2026-05-09.md`; self-play is under
critique after the gen2 failure.

Scope: critique the current Pong docs and recent experiment notes. Source only:
the working docs, training smoke runbook, and
`docs/experiments/2026-05-09-dummy-pong*`.

## Current State

Pong is now a good small visual toy, not just an eval stub. It has raster
observations, fixed baseline eval, trace artifacts, imitation replay/training,
learned-checkpoint eval, scoring replay, all-ego positive and negative reward
rows, a tiny value-target trainer, explicit paddle off-center bounce metadata,
and a self-play hypothesis now under critique.

The honest result is still modest. The learned raster policy is runnable and
beats random, but it is weaker than scripted `track_ball`. The value-target
smoke proves target construction and checkpoint writing, not a better policy.

## Stale Claims

- `docs/working/pong_training_plan.md` still says:
  "Add a value/reward-target smoke from all-ego scoring replay."
  This is stale. `docs/experiments/2026-05-09-dummy-pong-scoring-all-ego-value-train-smoke.md`
  says that smoke ran and wrote `summary.json` plus `checkpoint.npz`.
- `docs/working/pong_training_plan.md` also says:
  "The next move is a small value/reward-target smoke from all-ego scoring
  replay." Same stale point.
- `docs/working/training_loop_agenda.md` says E8 has "value/reward-target smoke
  next." That is now behind the experiment state.
- `docs/working/training_coach_packet.md` lists next action 1 as:
  "Add a Pong value/reward-target smoke using all-ego scoring replay." That is
  now done once.

I did not find a stale claim that Pong has no scoring replay, no all-ego rows,
or no off-center bounce metadata. Those are represented correctly in the newer
working docs.

## Missing Next Steps

- The docs do not yet name the next step after the value-target smoke. They
  should move from "can we build value targets?" to "can a policy use those
  targets to choose better returns?"
- The runbook has no Pong smoke section, so a reader has to hunt through
  experiment notes to know which commands are canonical.
- The docs mention the off-center return north star, but they do not define the
  first tiny test for it. The paddle-angle smoke proves the mechanics exist; it
  does not prove a policy can use them.

## Overcomplication

- The Pong path is getting mixed with MuZero, Modal, survival, Mctx, and
  LightZero planning language. That is useful background, but it makes the next
  Pong move harder to see.
- More replay formats are not the bottleneck right now. The useful split is
  simple: expert-only rows for action copying, all-ego rows for value targets.
- Do not add reward shaping to environment or eval. The current self-play
  hypothesis uses a separate shaped episode return for training.

## Historical Recommended Actions

These actions have been superseded by the self-play smoke, gen2 smoke, and
critique wave:

1. Add a Pong self-play replay builder and shaped-return summary. Done.
2. Add a tiny self-play policy/value trainer and score periodic checkpoints.
   Done.
3. Do not add more generations by default; decide repair-self-play versus a
   simpler known baseline/curriculum.
