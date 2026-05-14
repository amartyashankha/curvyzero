# Todo

## Now

- [x] Create a separate refactor planning directory.
- [x] Record current source of truth.
- [x] Narrow scope to training-code scaffolding; environment is an interface
  contract only.
- [x] Add first-pass trainer surface map.
- [x] Add review index and first-pass refactor targets.
- [x] Add full skeleton docs for goal, bugs, decisions, source inventory, test
  inventory, refactor sequence, cleanup policy, subagent briefs, and glossary.
- [x] Draft checkpoint helper contract.
- [x] Launch parallel audits for tests, architecture boundaries, checkpoint
  discovery patch design, dirty-state risk, and end-to-end contracts.
- [x] Fold first returned audit outputs into these docs.
- [x] Decide the exact first test files and test names for Phase 1.
- [x] Add first red regression tests for timestamped `lightzero_exp*`
  discovery.
- [x] Run focused tests and fix only the behavior they expose.
- [x] Patch `lightzero_exp*` checkpoint discovery in place.
- [x] Run focused tests again.
- [x] Extract pure checkpoint path/parsing helpers to
  `src/curvyzero/training/lightzero_checkpoints.py`.
- [x] Extract checkpoint candidate collection and latest-selection helpers.
- [x] Add and fix regression coverage for timestamped LightZero resume-sidecar
  saving.
- [x] Add opponent assignment snapshot design note.
- [x] Guard top-level frozen opponent refs with the same immutable
  `iteration_N.pth.tar` rule as mixture refs.
- [x] Add first pure opponent assignment snapshot parser and tests.
- [x] Pin down the future opponent leaderboard / assignment interface:
  `opponent_leaderboard_interface.md`.
- [x] Run parallel stock-parity and trainer/env-boundary audits.
- [x] Record the stock LightZero parity discrepancy table.
- [x] Add a trainer-config-to-registered-env boundary test for scalar action
  stepping with opponent mixture/no-op behavior hidden inside the env.

## Next

- [x] Add regression tests before touching source behavior.
- [x] Do the first extraction only after the bugfix is green.
- [x] Run the focused local gate after the boundary test.
- [x] Run a tiny Modal background eval/GIF artifact smoke.
- [x] Add a local stock-entrypoint regression proving `--mode train` calls
  `lzero.entry.train_muzero`.
- [x] Make resume-sidecar save failures non-fatal after stock checkpoint save.
- [x] Add a fresh-run hook pass-through regression for `call_hook`, `eval`, and
  `random_collect`.
- [ ] After the bugfix, refactor the trainer scaffolding in small tested cuts.
- [x] Add exact resume-sidecar candidate-selection tests before extracting that
  pure helper.
- [x] Extract exact resume-sidecar candidate selection after tests are green.
- [x] Add auto-resume checkpoint candidate-selection tests before extracting
  that pure helper.
- [x] Reuse the shared checkpoint candidate helper in auto-resume selection.
- [x] Add progress payload construction tests before splitting payload building
  from JSON writing.
- [x] Split checkpoint progress payload construction from JSON writing.
- [ ] Extract checkpoint poller candidate/stability helpers.
- [ ] Audit active scripts that still default to fixed
  `train/lightzero_exp/ckpt` refs.
- [x] Add strict assignment schema-id and canonical-hash tests in
  `tests/test_opponent_registry.py`.
- [ ] Audit granular action cadence from launcher config to env step.
- [ ] Add tests proving trusted `--mode train` defaults do not bundle multiple
  game ticks behind one policy action.
- [ ] Patch trainer/config cadence plumbing if the audit finds hidden action
  repeat or multi-tick stepping.
- [ ] Decide whether manifest builders should read assignment snapshots or a
  registry source directly.
- [ ] Add pure validators/builders for leaderboard snapshots, live Dict
  pointers, and assignment audit records before wiring Modal.
- [ ] Wire assignment snapshots into the trainer only after adding trainer
  plumbing tests.

## Later

- [ ] Extract status/resume/poller payload helpers.
- [ ] Leave Modal functions as thin wrappers.
- [ ] Delete or archive stale tests only after the new tests own the contract.
- [ ] Update neighboring experiment/tournament docs with final file locations.
