# Lessons Learned

## Stock Path Discipline

The CurvyTron path is easiest to reason about when stock LightZero owns the
training loop and CurvyZero owns only the env adapter and artifact plumbing.

## Fixed Paths Are Dangerous

LightZero/DI-engine may create timestamped experiment directories. Code that
assumes one fixed checkpoint directory can silently report stale status.

## Docs Need A Front Door

When the thread gets long, agents need one directory with a front door, active
facts, todo list, and delegation file. Otherwise old docs and stale launches
pollute the current decision.

