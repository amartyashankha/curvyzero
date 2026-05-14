# Subagent Briefs

Use these briefs when launching or following up with agents.

## Shared Context For All Agents

- Work in `/Users/shankha/curvy`.
- Scope is training-code scaffolding only.
- Trusted lane is stock LightZero `train_muzero` through `--mode train`.
- Do not redesign environment mechanics.
- Do not revive old custom `two-seat-selfplay` as a learning path.
- Prefer exact file/function refs.
- Separate facts, hypotheses, and recommendations.
- Write one concise doc if assigned.

## Test Lockdown Brief

Find the exact regression tests needed before refactor. Focus on checkpoint
discovery, status, resume, poller, eval/GIF, and manifest refs.

## Architecture Boundary Brief

Map the big trainer file into responsibility groups and propose safe extraction
cuts after tests.

## Checkpoint Bug Brief

Trace fixed-path assumptions and design the smallest tested patch.

## Dirty State Brief

Classify dirty files by lane and identify which files this refactor should
avoid touching.

## E2E Contract Brief

Design local temp-dir tests that simulate a run root with timestamped
`lightzero_exp*` checkpoint dirs and no Modal dependency.

