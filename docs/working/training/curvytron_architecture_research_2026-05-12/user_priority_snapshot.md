# User Priority Snapshot

Purpose: preserve the current priorities in plain language so the coach can
review them without asking the user to repeat them.

## Operating Pattern

- Main thread plans, delegates, synthesizes, and decides.
- Subagents own bounded work: code edits, tooling, critiques, docs cleanup,
  matrix prep, and focused bug searches.
- Use follow-ups when assumptions change.
- Do not sit idle while agents, evals, or training runs are running.
- Keep every active thread visible in [delegation_log.md](delegation_log.md).
- Keep the active worldview in [current_source_of_truth.md](current_source_of_truth.md).
- Keep wording simple: say which metric is moving.

## Current Training Goal

The next real target is not "beat the frozen opponent." It is:

```text
learn to survive longer from visual CurvyTron observations
```

Frozen opponents may be too weak. If they die quickly, outcome reward can
saturate while the learned policy is still bad.

Scale note: the user is comfortable launching a large batch. The job is to make
it large in organized blocks, not small by default and not blindly crossed.

## Current Reward Direction

For the next diagnostic lane:

- training reward should be survival plus bonus pickup;
- outcome reward should be zero/off;
- outcome should remain an eval/readout metric only;
- bonus pickup reward should be immediate on the step where the bonus is hit;
- old v1d runs should be judged by score/outcome curves, not by the next
  survival objective.

## Current Opponent Direction

Add or use a diagnostic opponent that cannot cheaply end the game for the
learner:

- `opponent_death_mode=immortal`: player 0 can die, player 1 cannot die. Today
  this is passive death immunity only; player 1 can move out of bounds and still
  leave trail.
- `opponent_trail_mode=none`: possible later/canary lane where player 1 leaves
  no collision trail.

The simple first diagnostic is close to one-player wall avoidance while keeping
the stock two-player training plumbing.

Additional opponent directions:

- repeated copies matter: important stochastic/random-opponent setups should
  run several times with different seeds;
- blank canvas matters: a no-trail fake opponent is the cleanest wall-avoidance
  sanity check;
- random-policy frozen opponents should use different opponent policy seeds;
- random-policy frozen opponents are not wired into the stock source-state env
  yet; decide between generated random checkpoints and an explicit random
  opponent kind;
- ancestor-checkpoint opponents are useful controls but may need fewer copies;
- immortal behavior must stay pinned down before any future row treats it as a
  serious opponent lane, especially wall and trail collisions;
- a scripted wall-avoidant opponent may be a useful stronger baseline.
- reactive/reflection-style wall avoidance should be iterated before dismissal;
  if it fails, record why in data, not vibes.

## Axis Decisions

| Axis | Current decision |
| --- | --- |
| Episode cap | Do not sweep. Use a high cap such as `65536`. |
| Render | Run matched `body_circles_fast` and `browser_lines` rows for serious cells. |
| Stochasticity | Sweep no, low, medium, and high levels. This was underswept before. |
| Search | Default to sim8; use sim16 only as a small sentinel. |
| Collector count | Default to C32 unless new evidence says otherwise. |
| Learner batch | Default to B32 unless new evidence says otherwise. |
| Opponent age | Keep fixed/old/recent as controls, not the main success claim. |
| Copies | Repeat important random/stochastic rows, roughly five copies when needed. |

Large-matrix stance: about 50 runs can test the clean lane, about 100 can add
opponent-family comparison, 200+ is reasonable when the extra rows are repeat
groups or confirmation blocks, and about 300 can be justified if the extra rows
mainly widen repeats on top cells or random-opponent families.

Current anchor: blank canvas is the cleanest first claim. If that does not
learn wall survival, richer opponent-family results will be hard to interpret.

## Analysis Tooling Need

Build reusable curve tooling:

- fetch or read run artifacts;
- extract curves for outcome now, survival/reward later;
- compute simple shape metrics: latest, best, delta, early/late slope,
  late-bloomer, peak-then-crash, flat/collapse;
- export compact snapshots for subagents;
- be cautious about false negatives.

## Immediate Launch/Monitor Rule

The first 50-row survivaldiag batch has launched. For this batch, monitor rather
than duplicate it. For future large batches, do not launch until the gates are
checked:

- stock `train_muzero` path;
- immortal opponent canary;
- survival plus bonus reward canary with outcome reward off;
- strict-stop high cap canary; do not trust `max_train_iter` alone for tiny
  checks;
- checkpoint/eval/GIF artifacts discoverable;
- live status checked against real artifacts and Modal app state; heartbeat and
  poller JSON may be stale after a forced stop;
- matrix names encode the important axes.
