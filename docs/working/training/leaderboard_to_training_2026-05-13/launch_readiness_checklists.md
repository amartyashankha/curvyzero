# Launch Readiness Checklists

## Static Overnight Training Manifest

Use this if launching before full leaderboard-to-training wiring.

- [ ] Manifest uses trusted `--mode train`.
- [ ] Cadence is one-frame: `decision_source_frames=1`.
- [ ] No stale `decision_ms=200` or `300` in train kwargs.
- [ ] Frozen checkpoint refs are exact `iteration_N.pth.tar`.
- [ ] Frozen refs were discovered with broad `lightzero_exp*/ckpt` scan.
- [ ] Run names encode objective, opponent family, render, stochasticity, sim,
  collector, batch, repeat/copy, and seed.
- [ ] Browser/fast pairing is intentional for important cells.
- [ ] `batch64` is not scaled.
- [ ] `sim16` appears only as a sentinel, if at all.
- [ ] `collector64` appears only as a bounded probe.
- [ ] Scripted/blank/passive entries use currently supported trainer/env flags.
- [ ] Expected eval/GIF behavior is documented.
- [ ] Launch artifact is saved and linked from the current decision doc.

## Leaderboard-Derived Training

Use this only when assignment plumbing exists or when static manifest mirrors the
assignment manually.

- [ ] Source leaderboard/rating snapshot identified.
- [ ] Snapshot is final or explicitly marked provisional.
- [ ] Rows have games, distinct opponents, status, context hash, and checkpoint refs.
- [ ] Assignment strategy is documented.
- [ ] Assignment JSON is immutable and hashable.
- [ ] Assignment audit records selected rows, source snapshot, strategy id, seed,
  and fallback reasons.
- [ ] Assignment artifact was produced by
  `scripts/materialize_curvytron_leaderboard_assignment.py` or by the future
  Modal publisher/selector path.
- [ ] Trainer launch metadata records assignment ref and sha256.
- [ ] Resume behavior is defined: reuse vs explicit refresh.
- [ ] Eval/GIF consumes same assignment metadata.

## New Public Leaderboard / Tournament

- [ ] Rating context uses current one-frame semantics.
- [ ] `decision_source_frames=1` recorded in rating spec and game summaries.
- [ ] Context hash changes if cadence changes.
- [ ] Checkpoint discovery uses broad glob.
- [ ] Roster labels include run id and iteration.
- [ ] Seat fairness decision documented.
- [ ] Games per pair odd and documented.
- [ ] Active-status threshold documented.
- [ ] Website can show progress/rankings without request-time huge scans.
- [ ] GIF sample budget bounded.

## Intake / Online Elo

- [ ] Intake manifest active key exists.
- [ ] Subscriber tick discovers new checkpoints.
- [ ] Queue events include stable event ids.
- [ ] `queued_checkpoint_refs` and `seen_checkpoint_refs` are separate.
- [ ] Drain claim prevents duplicate rating spawn.
- [ ] Existing rating run behavior is explicit: reject or continue.
- [ ] Continuation from `latest.json` tested before calling this online.
- [ ] Stale-claim reset path exists or manual operator procedure documented.

## Scripted / Seeded Roster

- [ ] Decide whether scripted policies are leaderboard players or assignment-only.
- [ ] If leaderboard players, define `player_kind`, stable id, label, and context fields.
- [ ] If assignment-only, record entries in assignment audit.
- [ ] Invincible behavior is a modifier, not silently a policy identity.
- [ ] Passive/immortal controls are labeled dirty diagnostics.
- [ ] Hand-coded policy parameters are hashed into context/assignment audit.

## Optimizer / Speed Axis

- [ ] Current optimizer source docs were read.
- [ ] Chosen render surface preserves the intended observation contract.
- [ ] Chosen collector count is separated from learning-quality claims.
- [ ] GPU render is not used unless parity and handoff are proven.
- [ ] Speed profile names all buckets: env, render, opponent, search, replay,
  learner, checkpoints, eval/GIF, artifact I/O.
- [ ] No old `fast_gray64_direct` or `body_circles_fast` recommendation leaks into
  current stock-path production commands unless explicitly labeled historical.
