# CurvyTron Inspector Operating Loop - 2026-05-11

Purpose: keep the inspector work simple, active, and restartable.

## Job

The inspector does not train CurvyTron.

The inspector reads run artifacts and turns them into plain evidence:

- what was run
- what checkpoints were evaluated
- whether survival changed
- whether actions collapsed
- how episodes ended, when the data can say
- what the data cannot explain yet

The broader product goal is to make CurvyTron training easier to debug. If an
agent dies after a short run, the team should quickly know whether it hit a wall,
hit a trail, timed out, used a broken eval, or just behaved like a weak baseline.

## Loop

1. Reorient.
   Read the current docs, local artifacts, and code paths before making claims.

2. Delegate.
   Keep the main thread for planning, review, and integration. Send bounded
   artifact reads, code reviews, tests, and experiments to subagents when they
   can run in parallel.

3. Inspect.
   Build or run the smallest report that answers one real question.

4. Run it.
   Use real CurvyTron artifacts, not only toy examples.

5. Ask why.
   If survival is short, ask what ended the episodes. If the report cannot answer,
   record that as missing data.

6. Critique.
   Look for fake confidence: mixed settings, missing checkpoints, action collapse,
   fixed-opponent evals described as self-play, or death reasons that are only
   round outcomes.

7. Update memory.
   Keep this doc family current so the next loop starts with facts, not vibes.

8. Choose the next small improvement.
   Prefer the simplest addition that removes the biggest source of confusion.

## Replay Before Rerun

When an old visual-survival eval has a nested per-episode JSON:

1. Check whether it has ordered `episode.actions`.
2. Replay those actions locally in the matching source-backed visual env.
3. Compare the replay final `trace_hash` to the stored final telemetry hash.
4. Trust the replay only if the hash matches.
5. Fetch or rerun old checkpoints only when action replay is missing or cannot
   match the artifact.

This is simpler than pulling checkpoint weights for a question the stored action
trace can already answer.

The main inspector is allowed to do this local replay automatically when the
nested files are already on disk. It is still read-only. It must not fetch
Modal data or spend remote compute.

## Current First Gate

A useful first report lets a human answer these in under a minute:

- Which run is this?
- Which checkpoints exist in the eval?
- Did mean survival improve, get worse, or stay flat?
- Did one action dominate?
- Are the eval settings comparable?
- Do we know how deaths happened?
- What should be inspected next?

## Simplicity Rule

Always ask: can this be simpler?

For now, the report should be read-only and local. It should not fetch Modal data
or rerun training. Modal fetch/rerun is an operational step and belongs in a
separate runbook. The inspector can read fetched files, but should not quietly
spend remote compute or mutate remote state.

## Trust Rule

The inspector should downgrade its verdict before it invents confidence.

Call the learning claim incomplete or unknown if:

- checkpoint rows do not use the same seed set
- caps or timesteps differ
- strict checkpoint loads are missing or false
- the latest checkpoint was not evaluated
- action collapse is present
- the report lacks baseline comparisons
- death cause is unavailable
- fixed or frozen opponent results are being used as self-play proof

## Tournament Addendum - 2026-05-13

The checkpoint tournament lane is now part of inspection. Its job is to turn
many CurvyTron checkpoints into useful evidence for the coach.

Operating loop for this lane:

1. Re-read the tournament orchestration, active-thread, todo, validation, and
   critique docs before changing code.
2. Keep the main thread on the map: what is being built, why it matters, what is
   blocked, what can run in parallel.
3. Delegate independent work: website critique, Modal ops, scheduler research,
   refactor review, and validation can run in parallel.
4. Make one small behavior-preserving cleanup cut at a time.
5. Test locally after each cut. Smoke Modal only after the local contract is
   stable.
6. Record evidence immediately, including what is still not proven.
7. Do not launch a large tournament until checkpoint discovery, observation
   contract, scheduler bounds, and website scale are all explicit.

Plain North Star: the coach should be able to see which checkpoints are
improving, why that belief is supported, and where the evidence is still weak.
