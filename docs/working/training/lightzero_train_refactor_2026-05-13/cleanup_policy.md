# Cleanup Policy

## Source Cleanup

Clean only after tests cover the behavior. Prefer small moves that preserve old
public wrappers during the first pass.

## Test Cleanup

Do not delete tests because they look ugly. Delete a test only when:

- it covers a dead path;
- replacement tests cover the real contract;
- the deletion is recorded here or in the decision log.

## Docs Cleanup

Do not rewrite history during active debugging. Instead:

- mark stale docs as stale;
- link to the current source of truth;
- move noisy background evidence only after the active phase is calm.

## Naming Cleanup

Names should explain the human question:

- checkpoint discovery;
- resume selection;
- progress payload;
- poller candidate;
- eval/GIF request.

Avoid names that only encode internal accidents.

