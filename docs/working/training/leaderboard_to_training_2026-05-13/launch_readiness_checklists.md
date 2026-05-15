# Launch Readiness Checklists

## Current Restart Gate

Do not launch new training until all boxes in this section are checked.

- [x] All-v2 reset is complete: exact v2 Volumes/Dicts/Queue were
  delete/recreated, v2 apps redeployed, and `modal.Volume.from_name(...,
  version=2).info()` passed for `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`.
- [x] Current v2 real18 run is stopped or explicitly archived as diagnostic
  smoke; no restart assignment is published from it.
- [x] Learner seat/perspective fix is implemented and tested, with the next
  manifest set to `random_per_episode`.
- [x] No-op/straight action semantics are checked and documented for both
  training and tournament eval.
- [x] Tournament eval parity with the trainer observation/action surface is
  tested.
- [x] Stale detached Modal apps and confusing old dashboards are cleaned up or
  clearly labeled before restart.
- [x] Next manifest globally includes at least about `20%` blank/immortal
  pressure, plus some higher-immortal variants.
- [x] The previous weak `5%` wall-avoidant immortal recipes are not reused as
  the main restart pressure plan.
- [x] Storage namespace is explicitly chosen for the launch: the current lane is
  all-v2 and must use the names in `src/curvyzero/contracts/curvytron.py`.
- [x] Run a fresh all-v2 canary before a larger batch. The recreated all-v2
  canary `curvy-e2e-allv2-canary-20260515a` proved wiring at canary scale.
- [x] Launch builder fails closed instead of silently using stale defaults:
  explicit ratings snapshot required, assignment mode/control-volume targets
  are default, refresh pointers are default, and coarse assignment refresh is
  nonzero by default.
- [x] Submitter rejects app-name mismatches so a v2 manifest cannot be launched
  into an old trainer app by stale guards or a bad `--app-name`.
- [ ] Build and audit the exact larger launch manifest for stale non-v2
  checkpoint refs, fresh leaderboard snapshot metadata, fresh tournament/rating
  ids, and resolved shared-contract defaults.
- [ ] Run the manifest checkpoint-ref existence audit against the active
  all-v2 runs volume, not only local JSON syntax checks.
- [ ] Identify or run a production-shaped bounded leaderboard/assignment
  validation with real active-row gates. The all-v2 canary's provisional
  relaxations do not prove production-quality ranking.

## Static Overnight Training Manifest

Use this if launching before full leaderboard-to-training wiring.

- [ ] Manifest uses trusted `--mode train`.
- [ ] Cadence is one-frame: `decision_source_frames=1`.
- [ ] No stale `decision_ms=200` or `300` in train kwargs.
- [ ] Frozen checkpoint refs are exact `iteration_N.pth.tar`.
- [ ] Frozen refs were discovered with broad `lightzero_exp*/ckpt` scan.
- [ ] Reward variant exists in both env reward logic and Modal trainer config.
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
- [ ] Snapshot path is explicit in the build command; no default `/private/tmp`
  snapshot path is accepted.
- [ ] Rows have games, distinct opponents, status, context hash, and checkpoint refs.
- [ ] Assignment strategy is documented.
- [ ] Assignment JSON is immutable and hashable.
- [ ] Assignment artifacts are written to `control:` for current all-v2
  launches.
- [ ] Per-recipe refresh pointers are written to `control:` and each training
  row has a nonzero `opponent_assignment_refresh_interval_train_iter`.
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

- [ ] Fresh restart uses new tournament/rating ids, or the exact old intake
  queue partition is drained/reset before reuse.
- [ ] Intake manifest active key exists.
- [ ] Subscriber tick discovers new checkpoints.
- [ ] Queue events include stable event ids.
- [ ] `queued_checkpoint_refs` and `seen_checkpoint_refs` are separate.
- [ ] Drain claim prevents duplicate rating spawn.
- [ ] Existing rating run behavior is explicit: reject or continue.
- [ ] Continuation from `latest.json` tested before calling this online.
- [ ] Stale-claim reset path exists or manual operator procedure documented.
- [ ] Any `modal run` command that spawns background game/rating workers uses
  `--detach`, or waits for the child work to finish before returning.
- [ ] Success is verified from durable artifacts: `latest.json` advanced and
  completed game summaries exist. "Round scheduled" alone does not pass.

## Scripted / Seeded Roster

- [ ] Decide whether scripted policies are leaderboard players or assignment-only.
- [ ] If leaderboard players, define `player_kind`, stable id, label, and context fields.
- [ ] If assignment-only, record entries in assignment audit.
- [ ] Invincible behavior is a modifier, not silently a policy identity.
- [ ] Passive/immortal controls are labeled dirty diagnostics.
- [ ] Hand-coded policy parameters are hashed into context/assignment audit.

## Optimizer / Speed Axis

- [ ] Current optimizer source docs were read.
- [ ] Production policy observation backend is named as CPU `cpu_oracle`
  `browser_lines + simple_symbols`.
- [ ] GPU `browser_lines + simple_symbols` is labeled lab/profile-only until
  trainer-visible contract parity passes.
- [ ] Chosen render surface preserves the intended observation contract.
- [ ] Chosen collector count is separated from learning-quality claims.
- [ ] H100 learner/search compute is not described as GPU rendering.
- [ ] GPU render is not used unless parity and handoff are proven.
- [ ] Speed profile names all buckets: env, render, opponent, search, replay,
  learner, checkpoints, eval/GIF, artifact I/O.
- [ ] No old `fast_gray64_direct` or `body_circles_fast` recommendation leaks
  into current stock-path production commands unless explicitly labeled
  historical/control.
