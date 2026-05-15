# TRAINING_CONTROL

This doc owns trainer-side decisions: manifests, rewards, slots, assignment
refresh, champion bootstrap, and live interventions.

## 2026-05-15 Current Storage/App Contract

- Active trainer app: `curvyzero-lightzero-curvytron-visual-survival-train-v2`.
- Active training/checkpoint Volume: `curvyzero-runs-v2`, opened as Modal
  VolumeFS v2.
- Active assignment/control Volume: `curvyzero-curvytron-control-v2`, opened as
  Modal VolumeFS v2.
- Active leaderboard source for assignments:
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Do not launch trainer rows that point at old non-v2/hybrid assignment refs
  unless those refs are first rematerialized into the all-v2 lane.

## 2026-05-15 09:40 EDT P0 Player Perspective Risk

Current stop rule:

- The current v2 real18 run is invalid enough to stop. Keep artifacts for smoke
  and diagnosis, but do not spend more training budget treating it as a
  candidate production run.
- Do not launch the next training batch until all restart blockers below have
  tests: randomized learner seat/perspective, no-op/straight action semantics,
  tournament eval parity, and cleanup of stale Modal apps/artifacts that can
  confuse evidence.

Open question:

- The trainer may only be teaching the learned policy from seat 0 / player 1
  perspective.
- Tournament games may be evaluating a checkpoint in both seat 0 and seat 1
  positions. If seat 1 observations/actions are not trained or normalized the
  same way, rankings can be misleading.

Why this matters:

- If the policy only learned the player-0 view, then a tournament where the same
  policy must act from player-1 view may punish or reward checkpoints for a
  distribution they never trained on.
- A clean long-term fix is probably to train from both seats, likely by choosing
  the learner seat at reset/episode boundaries and keeping the opponent in the
  other seat.
- A tempting evaluation-only fix, giving both policies player-0 observations,
  may break action semantics if actions are relative to each player's own
  heading. This must be audited before changing it.

Current rule:

- Do not call the current rerate final or relaunch the real batch until the
  training and tournament perspective fixes land with tests.
- If a relaunch is needed before the full fix, it must be explicitly named a
  smoke/probe and must not replace the restart plan.

Main-thread first read:

- Current `CurvyZeroSourceStateVisualSurvivalLightZeroEnv` rejects old
  `ego_player_index` config and uses `learner_seat_mode` instead. Fresh
  training defaults to `random_per_episode`; `fixed_player_0` and
  `fixed_player_1` are explicit diagnostics only.
- Tournament games build one observation per seat from `SourceStateGray64Stack4`
  and pass `observation[0, seat]` to the policy controlling that seat. That
  means old seat-0-only checkpoints could be rated in a seat they did not learn;
  fresh restart checkpoints should avoid that mismatch by using
  `learner_seat_mode=random_per_episode`.
- The live action space is currently `left`, `straight`, `right`. `straight`
  is the available no-turn action. There is not a separate live-player inert
  no-op that skips movement; inactive slots can use `-1` internally.
- The likely real fix is not to force both tournament policies to see seat-0
  observations. The safer direction is to train with randomized learner seat at
  episode/reset boundaries, then make tournament eval remain per-seat.
- Restart target: use `random_per_episode` learner seat/perspective selection
  and prove that seat/perspective metadata, action semantics, and tournament
  observation/action parity are recorded in the manifest and covered by tests.

## Next-Batch Opponent Pressure

Updated user direction, 2026-05-15 09:50 EDT:

- Do not do the five-row-only weak-run intervention on the current live rows.
- In the next clean manifest, raise blank/immortal exposure for all rows to at
  least roughly `20%`.
- Include some variants with higher immortal exposure, but keep most probability
  on leaderboard checkpoints so the run still learns against real policies.
- This is a manifest design change after the player-perspective fix, not a live
  mutation of the three current shared control pointers.
- Do not carry forward the previous weak main recipes that only had `5%`
  wall-avoidant immortal pressure. Some rows may still use low pressure as an
  intentional control, but the global next-manifest baseline must be stronger.

Restart manifest requirements:

| Requirement | Rule |
| --- | --- |
| Learner perspective | `random_per_episode` |
| Blank/immortal baseline | At least about `20%` globally |
| Higher-pressure variants | Include some rows above the baseline immortal/blank mix |
| Leaderboard exposure | Keep enough ranked-checkpoint probability for real-policy learning |
| Old weak recipes | Do not reuse `blank5-wall5-*` as the default restart shape |

## Current Batch

Historical note: this table describes the invalidated `v2real18` batch. It is
not the current launch target after the all-v2 reset. The current active next
step is the one-row all-v2 canary described in `TODO.md` and
`FULL_LOOP_PROOF.md`.

| Field | Value |
| --- | --- |
| Batch prefix | `curvy-v2real18-` |
| Rows | 18 |
| Compute | H100 learner/search compute target, not render proof |
| Observation render | CPU `body_circles_fast + simple_symbols` historical batch surface only, not restart target |
| Search | `num_simulations=8` |
| Batch size | `32` |
| Checkpoint cadence | `save_ckpt_after_iter=10000` |
| Max steps target | `1_048_576` |
| Source leaderboard | `v2refresh18p-live-r3-20260515a` local snapshot, 41 active rows |
| Initial policy checkpoint | rank 1 checkpoint from that snapshot |
| Assignment storage | v2 control Volume (`control:` refs) |
| Refresh storage | three per-recipe v2 control pointers |
| Refresh cadence | every `2000` learner train iterations |

Fresh launch candidate, 2026-05-15:

- manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2real18-20260515a/curvy-v2real18-20260515a.json`;
- dry run:
  `submission-dryrun-shankha-dev.json`, `18` rows, `3` assignment writes,
  `3` refresh pointer writes;
- tests:
  `tests/test_curvytron_tonight18_manifest.py`,
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py`, and
  `tests/test_promote_curvytron_rating_round.py` -> `98 passed, 3 skipped`.
- launch:
  `submission-full.json`, status `submitted`, Modal env `shankha-dev`,
  `18` train calls and `18` poller calls spawned on
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`.
  The submitter wrote `3` assignments and `3` refresh pointers to
  `curvyzero-curvytron-control-v2` before spawning the rows.
- immediate health:
  `18/18` rows have attempt manifests and `progress_latest.json` on
  `curvyzero-runs-v2`; first progress timestamps are around
  `2026-05-15T10:08:33Z` to `2026-05-15T10:09:01Z`, all at learner iter `0`.

## Current Coach Handoff

Use the trusted stock LightZero path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
```

Recommended next run settings:

| Setting | Current recommendation |
| --- | --- |
| reward | `survival_plus_bonus_no_outcome` |
| simulations | `num_simulations=8` |
| learner batch | `batch_size=32` |
| collectors | `collector_env_num=32` main, `64` bounded probe |
| production policy observation | CPU `browser_lines + simple_symbols` through `cpu_oracle` |
| GPU observation lane | lab/profile-only `browser_lines + simple_symbols` until trainer-visible contract parity passes |
| historical CPU control | `body_circles_fast + simple_symbols` only when explicitly labeled ablation/control |
| checkpoint cadence | `save_ckpt_after_iter=5000-10000`, lower for canaries |
| avoid | `batch64`, multi-GPU, broad `sim16` |

Speed probes:

- practical aggressive compute probe: `C256/H100/sim8` with the target
  observation surface;
- cheaper wide compute probe: `C384/L4/sim8` with the target observation
  surface;
- old `fast` body-circles rows are historical CPU controls, not a
  learning default or GPU-render proof.

## Reward Variants

The current batch intentionally spans:

- outcome only;
- survival + bonus, no outcome;
- survival + bonus + outcome.

Current read:

- outcome-only latest eval is up in all 6 rows;
- survival+bonus-only latest eval is weak, up in only 2/6 rows;
- survival+bonus+outcome is currently strongest by best-seen improvement.

Do not collapse reward conclusions from one noisy latest eval. Track both
latest and best-seen values.

## Slot Recipes

The intended slot model:

- A training run consumes immutable `assignment.json` files.
- Slot recipes can be operator intent, but training truth is the immutable
  assignment actually used.
- Slots can point to leaderboard-ranked checkpoints or scripted/blank opponents.
- Some checkpoint slots may have an immortal/invincible overlay.
- Blank-canvas/no-op opponent is also invincible by nature and leaves no trail.

Important boundary:

- Trainer must not read live tournament state inside env steps or learner
  updates.
- Any refresh must happen at a clean boundary and write telemetry/audit.

## Weak-Run Immortal Intervention

User request:

- Find the five current runs that are improving least.
- First inspect their current slot probabilities.
- Then increase total blank/immortal exposure to about `50%` for only those
  five runs.
- The 50% can include:
  - blank-canvas/no-op;
  - frozen checkpoint opponent with immortal/invincible death mode;
  - hand-coded wall-avoidance immortal opponent if wired and safe.
- Keep some leaderboard checkpoint exposure; do not turn the whole batch into
  all immortal opponents.
- If these five runs get worse, that is acceptable signal.

Implementation guard:

- Do not mutate live assignments until the exact current control path is known:
  Modal Dict recipe, refresh pointer, assignment bank, or direct assignment ref.
- Any live mutation must write an audit record with old mix, new mix, affected
  run ids, timestamp, and reason.

2026-05-15 read-only update:

- Current control path for the v2 refresh batch is a shared control-volume
  `refresh_pointer.json` plus immutable assignment files, checked by trainers
  every `50` learner iterations.
- Fresh eval-summary read over the 18 submitted run ids identified these five
  weakest rows by latest-minus-first survival, with best-minus-first as the
  secondary sanity check:

| Run label | Latest - first | Best - first | Current assignment mix |
| --- | ---: | ---: | --- |
| `survbonusnoout-blank5-wall5-rank2_25-rank1_65-so10rep10` | `-60.0` | `+0.0` | `blank 5%`, `wall_avoidant_immortal 5%`, `rank2 25%`, `rank1 65%` |
| `survbonusnoout-blank5-wall5-rank2_25-rank1_65-clean` | `-12.0` | `+94.8` | `blank 5%`, `wall_avoidant_immortal 5%`, `rank2 25%`, `rank1 65%` |
| `survbonusout-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-so10rep10` | `-11.8` | `+23.3` | `blank 10%`, `wall_avoidant_immortal 5%`, `rank4 10%`, `rank3 20%`, `rank2 20%`, `rank1 35%` |
| `survbonusnoout-blank20-wall5-rank1_75-clean` | `-5.3` | `+14.0` | `blank 20%`, `wall_avoidant_immortal 5%`, `rank1 75%` |
| `survbonusnoout-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-clean` | `-4.3` | `+60.0` | `blank 10%`, `wall_avoidant_immortal 5%`, `rank4 10%`, `rank3 20%`, `rank2 20%`, `rank1 35%` |

- Proposed intervention, not applied in this pass: write new immutable
  assignments for only these five run labels with about `50%` combined
  blank/immortal exposure, preserve at least `50%` leaderboard-checkpoint
  exposure, then atomically update the shared control pointer with an audit
  record naming old assignment sha, new assignment sha, affected run labels, and
  reason.
- Blocker: do not update the live pointer until the assignment-writer command
  and audit path are selected for this exact v2 refresh control pointer. A bad
  pointer would affect running trainers.

## Champion Bootstrap

Next production-shaped launches should start fresh learners from the current
trusted tournament winner.

Rules:

- This is model-only initialization.
- It is separate from opponent assignment.
- It must not silently restore same-run optimizer/replay/collector state.
- The exact checkpoint ref must be recorded in the manifest and launch summary.

## Refresh Cadence

Current local regenerated manifest uses refresh interval `50` learner
iterations. This may be too chatty.

Do not change to `1000` or `2000` until resume/refresh safety is known:

- same-run auto-resume behavior;
- assignment pointer behavior;
- whether existing trainers pick up the change;
- telemetry proof that refresh happened.

## Render Path

Production lane:

- stock LightZero with CPU `cpu_oracle` `browser_lines + simple_symbols`
  policy observations.

GPU lab lane:

- `browser_lines + simple_symbols` only in profiling/lab harnesses until
  trainer-visible contract parity passes.

Historical CPU control only:

- `body_circles_fast + simple_symbols`.

Do not describe H100 learner/search compute or CPU body-circles rendering as
GPU observation rendering.

## Run Naming

Future run names should be readable. Include only the human-critical knobs:

- opponent source/slot family;
- reward family;
- stochasticity family;
- render lane;
- seed or replicate id.

Avoid opaque names like long date/hash/knob soup unless they are in metadata
rather than the visible run name.
