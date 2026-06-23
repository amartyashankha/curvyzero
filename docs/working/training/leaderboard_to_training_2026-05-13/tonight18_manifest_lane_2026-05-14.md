# Tonight 18-Run Manifest Lane - 2026-05-14

## Status

2026-05-16 current lane: use
`artifacts/local/curvytron_tonight18_manifests/curvy-r18v2-bootstrap-20260516a/curvy-r18v2-bootstrap-20260516a.json`
for the fresh all-v2 bootstrap batch. That manifest has been launched against
`curvyzero-lightzero-curvytron-visual-survival-train-v2` and is the active
large-batch lane to monitor. Older `curvy-night18-*` and `curvy-v2real18-*`
sections below are historical evidence only.

2026-05-15 supersession: this page is historical launch evidence only. Do not
copy its `body_circles_fast + simple_symbols` fixed knobs into fresh production
runs. Current production policy observations are CPU `cpu_oracle`
`browser_lines + simple_symbols`; GPU rendering is lab/profiling-only until
trainer-visible contract parity passes.

Live launch is in progress from the top-10 fallback snapshot. The top-100 gate
is still a useful tournament-debug lane, but it is not blocking tonight's
training launch.

Current live manifest:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-night18-top10r1-20260514a/curvy-night18-top10r1-20260514a.json
```

Current live app:

```text
curvyzero-lightzero-curvytron-visual-survival-train
```

2026-05-14 launch facts:

- `r001` and `r013` launched first as canaries and both wrote real train
  progress plus `iteration_0.pth.tar`.
- The remaining rows were submitted through the deployed app spawn path, not one
  ephemeral app per row.
- Five initial rows (`r004`, `r009`, `r010`, `r015`, `r016`) exposed a concrete
  Modal/run-management bug: generated `run_id`/`attempt_id` strings exceeded
  the 96-character limit enforced by the trainer's run-management layer.
- `scripts/build_curvytron_tonight18_manifest.py` now truncates generated ids to
  96 characters with a hash suffix, and
  `scripts/submit_curvytron_survivaldiag_manifest.py` now refuses any selected
  row with an overlong `run_id` or `attempt_id` before remote launch.
- Replacement rows for `r004`, `r009`, `r010`, `r015`, and `r016` were submitted
  from the corrected manifest. Treat the earlier overlong submissions as failed
  launch attempts, not part of the intended 18-run batch.
- A later poll showed 10 of 18 corrected rows had visible training artifacts,
  and at least three rows had already reached `iteration_10000`.
- The background checkpoint eval/inspect lane had a real signature bug:
  `_run_checkpoint_eval_and_inspect` did not accept `opponent_assignment_ref`
  even though the poller passed it. Local code now accepts the argument, targeted
  tests pass, and the trainer app was redeployed at about 04:40 ET.

Live intake wiring, 2026-05-14:

- This was not automatic. The manifest submitter launched training and pollers
  only; it did not seed tournament intake.
- Intake was explicitly seeded for the narrow prefix `curvy-night18top10r1`.
  Do not use the broader `curvy-night18` prefix for this batch.
- Live intake ids:
  `curvy-night18-top10r1-20260514a` /
  `elo-night18-top10r1-20260514a`.
- Seed discovered 10 current checkpoints and queued all 10. Current discovered
  pool includes a mix of `iteration_10000` and `iteration_0` refs, because only
  10 of the 18 training rows had produced checkpoints at seed time.
- The initial night18 rating config has 10 checkpoint players, 45 pairs, 945
  games, one-frame timing, 21 games per pair, GIFs on, and 5 GIF samples per
  pair.
- Maturity was set to the current top10-compatible gate:
  `placement_min_opponents=9`, `placement_min_games=189`. This lets the current
  10-player pool become active instead of waiting for all 18 rows to start.
- A detached `intake-drain` saw `rating_run_claim_exists` after the seed. The
  durable rating artifacts already exist, so treat that as a claim/ownership
  state to monitor before spawning any duplicate writer.
- Follow-up progress proved the rating is doing real work: exact summary count
  saw 84 completed games, 4 completed pairs, 0 failed games, and no missing
  game-output ambiguity.
- The intake manifest then advanced from 10 to 11 seen checkpoints after a
  training row produced a fresh `iteration_10000` checkpoint. This is the first
  concrete proof that the live prefix watch is picking up new night18
  checkpoints after the initial seed.
- A later training status poll showed five rows at `iteration_10000`, ten rows
  with visible train artifacts, and eight rows still `train_root_absent`
  (probably not started/queued yet unless logs prove otherwise).
- Historical note, superseded by the current v2 publisher gate. Round 0
  completed cleanly after the seed/first continuation:
  11 checkpoint players, 55 pairs, 1155 games, 0 failed games,
  `ratings_written=true`, final `latest.json` present, one-frame settings, GIFs
  on, and 11 active rating rows. Under the current publisher contract this
  `stable=false` round is not publishable/materializable as a non-diagnostic
  training source. Diagnostic-only publication may record evidence, but must
  not move the latest training pointer.
- Round 0 was published successfully as
  `curvy-night18-top10r1-20260514a-elo-night18-top10r1-20260514a` with snapshot
  id `round0-20260514`: 11 active rows, 0 provisional rows, pointer published,
  latest ref
  `tournaments/curvytron/leaderboards/curvy-night18-top10r1-20260514a-elo-night18-top10r1-20260514a/latest.json`.
- The intake manifest later advanced to 16+ seen checkpoints, but `latest.json`
  still had 11 players. A round-1 continuation was staged with 66 pairs / 1386
  games, but its progress stayed at `game_map_started` with 0 started pairs.
  Treat this as the current continuation bug to debug, not as a failure of the
  valid round-0 leaderboard.
- Modal emitted the concrete warning that `.remote()` and `.map()` calls in
  detached apps may be canceled when the local caller disconnects. This matches
  the round-1 symptom: input/progress exists, but no child game workers started.
  The likely fix path is to make continuation launches use a fully safe
  spawn/get or waited path, then rerun a tiny continuation smoke before trusting
  automatic continuation.
- Manual continuation repair with `--wait` reclaimed the stuck claim, rebuilt
  the queued continuation work, and started real round-1 game workers. Progress
  reached `games_running`, with visible game results in logs. An exact progress
  poll later saw 42 completed games, 2 completed pairs, and 0 failures in
  round 1. Local code now changes the rating loop's round hop from `.remote()`
  to `spawn().get()`, and the deployed tournament app has been refreshed with
  that safer path.
- The current 18 training jobs were launched with static inline
  `opponent_mixture_spec`; they do not prove live assignment refresh from the
  public leaderboard. To prove that path, publish the night18 leaderboard,
  materialize a `stable_slots_v1` assignment, then launch a tiny new smoke with
  `opponent_assignment_ref` or explicit refresh args.
- The published round-0 leaderboard was materialized locally into a
  `stable_slots_v1` assignment at
  `/private/tmp/curvy-night18-stable-slots-assignment`. The three slots are
  champion, recent-strong, and blank-canvas sentinel. The next missing proof is
  a tiny trainer smoke that consumes this assignment artifact.

Manifest path smoke: passed against a local active-leaderboard-derived rating
fixture. The builder produced 18 rows, submitter dry-run accepted them, and every
row carries `commit_on_checkpoint=true`.

Stale artifact warning, 2026-05-14: rebuild the final
`curvy-tonight18-20260514a` manifest after fetching the final top-100 gate
snapshot. Older generated JSON may predate the corrected opponent mix.

Current gate, 2026-05-14: do not let the exact-212 stress lane block the real
training launch. Use a practical one-frame gate leaderboard, currently planned
as `curvy-oneframe-top100-gate-20260514a` /
`elo-oneframe-top100-gate-20260514a`, then build the 18-run batch from active
top rows only.

Important correction: `curvy-oneframe-visual-exact212-final-20260514b` was
contaminated by cleanup racing with rating output. Treat it as trash.

The old 212-checkpoint injection is useful as a stress test, but it is not the
main launch gate. The gate should use one-frame settings, 21 games per pair,
GIFs on, 5 sampled GIFs per pair, and 80 fps GIF output. It only needs enough
evidence to select top training opponents without provisional rows.

The builder now refuses top-4 rows unless they are `status=active`. This blocks
the old `curvy-oneframe-visual-main-20260514a` provisional snapshot by design.

## Connected Assignment-Backed Batch

## Clean3 Assignment-Backed Loop18 Candidate - 2026-05-14

Current candidate:

```text
matrix: curvy-loop18-clean3-20260514e
run prefix: curvy-loop18c3
assignment bank: curvy-loop18c3-assignments / try-loop18c3-assignments
manifest: artifacts/local/curvytron_tonight18_manifests/curvy-loop18-clean3-20260514e/curvy-loop18-clean3-20260514e.json
```

Source leaderboard:

```text
leaderboard: curvy-night18-connected-clean3-20260514e-elo-night18-connected-clean3-20260514e
snapshot: round0-clean3-20260514e
snapshot ref: tournaments/curvytron/leaderboards/curvy-night18-connected-clean3-20260514e-elo-night18-connected-clean3-20260514e/snapshots/round0-clean3-20260514e.json
snapshot sha256: ea25396e15712775f4ed23f00596fb2832b6f4595dbd6a08c9282d7d7319cd26
```

What is already true:

- Clean3 round 0 completed as an exact 18-player all-pairs one-frame tournament:
  153 pairs, 3213 games, 0 failed games, GIFs on, 5 GIF samples per pair, and
  80 fps GIF output.
- The public leaderboard publish succeeded from the immutable clean3 round-0
  snapshot with 18 active rows and 0 provisional rows.
- The loop18c3 manifest has 18 rows: 3 reward variants x 3 opponent recipes x 2
  noise modes.
- All 18 rows use `opponent_assignment_ref`; none use inline
  `opponent_mixture_spec`.
- The three assignment artifacts were written to the training Volume:
  `blank5-wall5-rank2_25-rank1_65`,
  `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35`, and
  `blank20-wall5-rank1_75`.
- Tiny CPU trainer smokes passed for all three assignment recipes:
  `curvy-loop18c3-smoke-blank5-20260514a`,
  `curvy-loop18c3-smoke-blank10-20260514a`, and
  `curvy-loop18c3-smoke-blank20-20260514a`.
- Smoke telemetry proves the assignment refs reached the environment. The
  blank10 and blank20 smokes loaded a real frozen leaderboard checkpoint with
  `opponent_provider_load_ok=true`. The blank5 smoke sampled its blank and
  immortal wall-avoidant scripted slots.
- Direct provider smokes passed for rank2, rank3, and rank4 checkpoint refs:
  each exact checkpoint was found on the Volume, loaded strictly with no missing
  or unexpected model keys, and called through the frozen-opponent wrapper.

Immediate launch gate:

1. Submit the full 18-row manifest through the deployed trainer
   app spawn path. Done at `2026-05-14 13:29:22 EDT`.
2. Monitor until all 18 jobs have visible training artifacts and at least
   `iteration_0.pth.tar`.
3. Seed or verify the live intake watch for prefix `curvy-loop18c3` only after
   the jobs exist. The watch should use one-frame settings and the same
   training-facing tournament lane, then long-sleep until the first natural
   `iteration_10000` checkpoints are expected.
4. After waking, verify the full chain with artifacts: trainer checkpoint refs,
   intake manifest/progress, rating progress/latest, public leaderboard pointer,
   and later trainer assignment telemetry.

Launch record:

```text
submission: artifacts/local/curvytron_tonight18_manifests/curvy-loop18-clean3-20260514e/submission.json
status: submitted
rows spawned: 18 / 18
train function: lightzero_curvytron_visual_survival_h100_cpu40
poller function: lightzero_curvytron_visual_survival_checkpoint_eval_poller
```

First immediate post-launch poll:

- All 18 rows have Modal train and poller call ids in the submission record.
- The first Volume status read was only seconds after launch. It showed a
  partial startup wave: some train roots, some pollers, and a few
  `status_heartbeat.json` files, but no batch-wide `progress_latest.json` yet.
  This is too early to judge health; keep polling until all rows either write
  progress/checkpoints or expose a real failure.

Current health poll, 2026-05-14 13:55 EDT:

- All 18 rows still have status heartbeats.
- All 18 rows have at least one checkpoint.
- 13 rows have reached `iteration_10000`; 5 rows are still at `iteration_0`.
- All 18 background checkpoint eval/GIF pollers report `running`.
- Eval/GIF artifacts exist for the rows that have been sampled so far. Some
  policies already show collapsed action distributions; treat that as training
  signal, not as a launch-health failure.

Live tournament lane:

```text
tournament: curvy-loop18-live-clean3-20260514e
rating: elo-loop18-live-clean3-20260514e
rating call: fc-01KRKRVBGAVDR6NHP6XF5XBM65
```

- Seeded with the 18 exact source checkpoints from the clean3 public
  leaderboard snapshot.
- Seed command used `modal run --detach`, one-frame settings, `all_pairs`,
  21 games per pair, GIFs on, 5 GIF samples per pair, and the default 80 fps
  GIF output.
- The seed accepted all 18 source checkpoint refs, queued all 18, committed the
  intake manifest, and spawned the first rating loop.
- Next: when the loop18c3 trainers have real checkpoints, re-seed this same
  tournament/rating id with `run_id_prefix=curvy-loop18c3`. The existing 18
  source checkpoints should remain in `seen_checkpoint_refs`, while the live
  prefix becomes the active watch for new loop18 checkpoints.

Current live proof lane:

```text
tournament: curvy-loop18-live-mixed-20260514g
rating: elo-loop18-live-mixed-20260514g
```

- Seeded first with the clean3 source refs plus the exact loop18c3 run ids.
- A bug was found where mixed explicit refs plus run ids could collapse back to
  source-only discovery. Local code now merges fixed refs with live run
  discovery, treats any scan with run selectors as a live watch, and preserves
  fixed refs when run ids are submitted.
- The tournament app was redeployed after the patch.
- The manifest was then re-seeded with the 18 exact run ids only. The durable
  `scan_spec` now has `checkpoint_refs=""` and the exact 18 `run_ids`, while
  `seen_checkpoint_refs` keeps the source refs plus newly discovered loop18
  checkpoints.
- The current manifest has 44 seen checkpoint refs and 27 queued events.
- The active rating round was already running from the earlier 35-player pool:
  595 pairs / 12495 games, all-pairs, one-frame, 21 games per pair, GIFs on, 5
  sampled GIFs per pair.
- Lightweight progress at 13:56 EDT showed 143 started pairs, about 3003 seen
  games, and 0 failed games. Wait for final reducer output before publishing.
- Follow-up progress at 14:10 EDT showed 423 started pairs, about 8883 seen
  games by shard progress, and 0 failed games. The round is healthy but not
  done. Do not publish or build the next assignment from this arena until the
  reducer writes a final, internally consistent `latest.json`.
- Exact per-game diagnostic progress at 14:14 EDT showed 436 completed pairs,
  9168 completed games, 2 partial pairs, 438 started pairs, and 0 failed games.
  This is the best current health signal. It still does not replace final
  reducer output.
- Lightweight progress at 14:17 EDT showed 574 started pairs, about 12054 seen
  games, and 0 failed games. Intake now sees 67 total checkpoint refs and 51
  queued events. The next gate is still final round-0 reducer output, then a
  continuation drain.
- Round 0 then completed cleanly: 35 active rows, 595 pairs, 12495 games, 0
  failed games, one-frame settings. A continuation claim already existed, and
  `round-000001` started. First round-1 progress showed about 82 players, 3321
  pairs, 69741 games, 9 started pairs, and 0 failed games.
- Current intake status sees all 18 loop18c3 latest checkpoint refs with no
  missing run ids. The queue has newer events waiting behind the active rating
  writer; drain them only after this round clears, using continuation from the
  final latest rating.

Current connected batch:

```text
matrix: curvy-night18-connected-20260514d
run prefix: curvy-n18conn
tournament/rating: curvy-night18-connected-20260514d / elo-night18-connected-20260514d
manifest: artifacts/local/curvytron_tonight18_manifests/curvy-night18-connected-20260514d/curvy-night18-connected-20260514d.json
```

What is proven:

- All 18 manifest rows use `opponent_assignment_ref`; none use inline
  `opponent_mixture_spec`.
- Sampled remote trainer telemetry proves the assignment-backed opponent path
  is not just config plumbing. `env_steps.jsonl` rows contain assignment
  ref/hash, `rank1` or `rank2`, exact frozen checkpoint refs from the tdfix
  leaderboard, and `opponent_provider_load_ok=true`.
- The trainers are producing later checkpoints. Prefix discovery now sees
  multiple `iteration_10000.pth.tar` refs.
- A manual intake tick found 10 new `iteration_10000` refs and enqueued them for
  the connected tournament.
- A manual drain spawned tournament rating work over a 28-ref pool. A later
  recheck showed this arena is contaminated: `round-000000/input.json` says
  300 pairs / 6300 games / 28-ref roster hash `8d8ed01185038cdb`, while
  `progress.json`, `results.json`, `ratings.json`, and `latest.json` say
  153 pairs / 3213 games / 18-ref roster hash `b60b8c1af0c48199`.
- A connected leaderboard had already been published as
  `curvy-night18-connected-20260514d-elo-night18-connected-20260514d`, snapshot
  `round0-connected-20260514a`, with 28 active rows and 0 provisional rows. Do
  not trust it as final proof because its stored source ref now points at a
  mutable rating file whose current sha no longer matches the snapshot's source
  sha.
- A new `stable_slots_v1` assignment was materialized from that connected
  leaderboard and written to the training Volume as
  `curvy-n18conn-round0-stable-slots`.
- Tiny trainer smoke
  `curvy-n18conn-assignment-smoke-20260514c /
  try-curvy-n18conn-assignment-smoke-20260514c` consumed that assignment,
  completed with `ok=true`, and wrote checkpoints `iteration_0.pth.tar` through
  `iteration_11.pth.tar`. This remains a trainer assignment-consumption proof,
  but no longer closes the connected arena proof.
- Its telemetry proves the new assignment reached self-play: assignment sha
  `5a5009b79855882ab347e22aeaa689f6f313f51c0bb771bc35cb084b6c42e2b5`,
  `slot_champion`, the connected checkpoint selected from the new leaderboard,
  and `opponent_provider_load_ok=true`.

Caveats:

- Relaunch a clean connected tournament with a fresh id before using this lane
  as final loop evidence. For the small connected pool, prefer `all_pairs` so
  everyone plays everyone and artifact counts are simple.
- This proves assignment-at-launch for the connected leaderboard. Direct
  in-running refresh has a separate tiny proof, but the final run-control /
  `ready.json` path is still future work.

Builder:

```bash
uv run python scripts/build_curvytron_tonight18_manifest.py
```

Default input after the tournament has a usable final snapshot:

```text
/private/tmp/curvy-oneframe-top100-gate-20260514a-latest.json
```

Default output:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-tonight18-20260514a/curvy-tonight18-20260514a.json
```

The live top-10 fallback output uses the `curvy-night18-top10r1-20260514a`
directory shown above. Do not rebuild over it while the current launch is being
monitored.

Dry-run submitter validation:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/curvy-tonight18-20260514a/curvy-tonight18-20260514a.json
```

Canary/full launch command shape. Use the deployed app spawn path; do not fan
out 18 independent ephemeral Apps:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/curvy-tonight18-20260514a/curvy-tonight18-20260514a.json --allow-launch
```

## Matrix

Shape: 3 reward variants x 3 opponent recipes x 2 noise modes = 18 rows.

Reward variants:

- `sparse_outcome`
- `survival_plus_bonus_no_outcome`
- `survival_plus_bonus_plus_outcome`

Meaning of `survival_plus_bonus_plus_outcome`: survival + same-step bonus +
terminal outcome scaled by the episode source-step count. A win roughly doubles
survival; a loss roughly cancels it.

Opponent recipes:

- `blank5-wall5-rank2_25-rank1_65`: historical recipe with blank 5, immortal wall-avoidant scripted opponent 5, rank2 25, rank1 65
- `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35`: historical recipe with blank 10, immortal wall-avoidant scripted opponent 5, rank4 10, rank3 20, rank2 20, rank1 35
- `blank20-wall5-rank1_75`: historical recipe with blank 20, immortal wall-avoidant scripted opponent 5, rank1 75

Current restart guidance supersedes these recipes: blank and hard-coded sentinel
slots should be immortal always. Frozen checkpoint/leaderboard slots should be
mostly mortal, with only small explicit immortal slices, and total immortal
exposure should stay around 20-30%, generally not above 30%.

The policy already has an explicit no-turn/no-op choice: action id `1`, named
`straight`, maps to source move `0`. Do not add a fourth action for tonight; that
would change the model action head.

Historical correction: the clean3/loop18c3 batch did not initialize learners
from the leaderboard champion. That was acceptable only as a static diagnostic
batch. It is not acceptable for the next production-shaped feedback batch.

Optional quality input: a fresh learner may receive `initial_policy_checkpoint_ref`
from a trusted rank-1 active checkpoint using model-only load semantics, but this
is not required for bootstrap or loop proof. Do not use same-run auto-resume for
this; auto-resume restores learner progress/optimizer state and is a different
contract.

Noise modes:

- `clean`: no straight override, no extra action repeat
- `straight_override_p10_repeat_p10`: `ego_action_straight_override_probability=0.10`, `policy_action_repeat_min=1`, `policy_action_repeat_max=2`, `policy_action_repeat_extra_probability=0.10`

Fixed knobs:

- `compute=gpu-h100-cpu40`
- `collector_env_num=256`
- `num_simulations=8`
- `batch_size=32`
- `source_state_trail_render_mode=body_circles_fast`
- `source_state_bonus_render_mode=simple_symbols`
- `save_ckpt_after_iter=10000`
- `commit_on_checkpoint=true`

The commit flag matters: the subscriber/intake lane can miss new checkpoints
until the final volume commit if the trainer only writes local checkpoint files.

The builder embeds direct `opponent_mixture_spec` JSON into both `train_kwargs`
and `poller_kwargs`, avoiding a separate assignment artifact.

## Top-4 Checkpoint Refs

Source: the final active rows from
`/private/tmp/curvy-oneframe-top100-gate-20260514a-latest.json`.

Do not reuse the old `/private/tmp/curvy-oneframe-visual-main-20260514a-latest.json`
source. Its top rows are still `status=provisional`, and the builder now rejects
it. Also do not block tonight's launch on the stale exact-212 stress lane.

## Current Operating Pattern

This is the shape to keep following until the loop is proven:

1. Relaunch the practical gate tournament with `modal run --detach` and a clean
   arena name.
2. Prove the gate with fetched artifacts, not Modal intuition: rating config,
   progress, completed game summaries, and a usable `latest.json`.
3. Wait until it writes a usable `latest.json` with active rows, not only
   provisional output.
4. Publish the public leaderboard only from that final snapshot.
5. Build the 18 training jobs from active leaderboard rows only.
6. Launch a one-row or few-row canary through the same deployed spawn path and
   prove start/artifacts before full launch.
7. Launch the remaining 18-run batch only after the canary is healthy.
8. Monitor the jobs until they start and write normal artifacts.
9. Use long sleeps between expected checkpoint drops. Estimate the wait from
   `save_ckpt_after_iter=10000` and observed train speed; 30 minutes to 2 hours
   is acceptable if that is the real checkpoint cadence.
10. After waking, verify that new training checkpoints were discovered by the
   subscriber, added to the tournament, rated, and reflected in the public
   leaderboard.
11. After promotion, verify either a refresh-enabled trainer applies the new
   assignment at a coarse boundary, or a controller launches a fresh attempt
   that consumes the new immutable assignment. Static batches cannot prove this.
12. In parallel, run or design a tiny dummy closed-loop test with very frequent
   checkpoints so the full path can be observed quickly without waiting for the
   real H100 runs.
13. If the top-100 gate is blocked or slow, start a smaller honest fallback
    gate, such as a top-10 one-frame tournament, and use it to launch a limited
    training path once it publishes active rows. Keep debugging the top-100 gate
    in parallel.

If progress appears stuck, run the explicit game-summary diagnostic/count path
before concluding Modal scheduling failed. A round-scheduled artifact is not
enough; completed `summary.json` files and advanced `latest.json` are the proof.

If the dummy path exposes a real bug, kill and relaunch the 18-run batch from the
manifest after the fix. The main point is to keep forward motion while keeping
the real batch recoverable.

Fallback rule: a smaller gate is not final policy-strength evidence, but it is
valid engineering evidence if it uses the same real handoffs: exact checkpoint
refs, one-frame games, final `latest.json` rows, manifest build, trainer launch,
checkpoint discovery, tournament continuation, and public leaderboard refresh.

Top-10 fallback caveat: a 10-player all-pairs tournament can give each player at
most 9 opponents and 189 games. That is enough for a fallback rating snapshot if
the rating rows become `active`, but public leaderboard publishing must use
scaled thresholds (`active_min_valid_games<=189`,
`active_min_distinct_opponents<=9`). Do not apply the top100 maturity thresholds
to a top10 fallback.

## Pre-Launch Gate Checks

Before launching the 18 H100 jobs, refresh the local snapshot from the tournament
Volume and verify it is the intended gate, not a stale file:

```bash
T=curvy-oneframe-top100-gate-20260514a
R=elo-oneframe-top100-gate-20260514a
SNAP=/private/tmp/${T}-latest.json
MAN=artifacts/local/curvytron_tonight18_manifests/curvy-tonight18-20260514a/curvy-tonight18-20260514a.json

uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments \
  tournaments/curvytron/$T/ratings/$R/latest.json \
  "$SNAP"

jq -e --arg T "$T" --arg R "$R" '
  .tournament_id == $T and
  .rating_run_id == $R and
  (.ratings | length) >= 4 and
  all(.ratings[:4][]; .status == "active") and
  all(.ratings[:4][]; (.checkpoint_ref | test("iteration_[0-9]+\\.pth\\.tar$"))) and
  all(.ratings[:4][]; ((.checkpoint_ref | contains("latest")) | not) and ((.checkpoint_ref | contains("ckpt_best")) | not))
' "$SNAP"
```

Then build and dry-run the manifest:

```bash
uv run python scripts/build_curvytron_tonight18_manifest.py --ratings-snapshot "$SNAP"
uv run python scripts/submit_curvytron_survivaldiag_manifest.py "$MAN"

jq -e '
  (.rows | length) == 18 and
  all(.rows[];
    .train_kwargs.commit_on_checkpoint == true and
    .train_kwargs.save_ckpt_after_iter == 10000 and
    .train_kwargs.opponent_mixture_spec == .poller_kwargs.opponent_mixture_spec
  )
' "$MAN"
```

Only then run:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py "$MAN" --allow-launch
```

Prefer a manifest slice for the first canary, then the full manifest after the
canary writes normal run artifacts. Both should use the same deployed spawn
path.

Also seed the live intake watch for the 18-run prefix, so new checkpoints from
the launched jobs can enter the same tournament later:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode intake-seed \
  --tournament-id "$T" \
  --rating-run-id "$R" \
  --run-id-prefix curvy-night18 \
  --checkpoint-selection latest \
  --round-count 1 \
  --continue-from-latest \
  --pair-selection adaptive_v0 \
  --pairs-per-round 300 \
  --placement-min-opponents 20 \
  --placement-min-games 420 \
  --games-per-pair 21 \
  --games-per-shard 1 \
  --reuse-policies-per-shard \
  --active-pool-limit 100 \
  --decision-source-frames 1 \
  --decision-ms 16.666666666666668 \
  --source-physics-step-ms 16.666666666666668 \
  --policy-mode eval \
  --max-steps 8000 \
  --num-simulations 8 \
  --save-gif \
  --gif-sample-games-per-pair 5 \
  --gif-sample-strategy evenly_spaced \
  --intake-active
```

The prefix was checked before launch and had zero existing runs, so this watch
should discover only the new 18-run batch.

If the gate drags too long, first run exact progress with
`--progress-read-summaries`. If enough game summaries exist but no final snapshot
does, try the reduce path or partial reduce, but still launch only if the top
rows are actually `active`.

Race caveat found during night18 debugging:

- do not publish from `curvy-night18-top10r1-20260514a` without a fresh
  consistency check;
- it produced a mixed round-1 artifact once: input/progress from a larger pool
  and results/latest from a smaller pool;
- the local code now prevents the root cause, but the old arena remains
  diagnostic unless a fresh post-fix continuation produces matching
  input/results/latest pool sizes.

## Operator Instruction Block

Verbatim instruction summary from the main thread, kept here so future context
loss does not erase it:

> Once you're done you can keep going and just make sure it stabilizes, it
> publishes to the leaderboard, and once that happens let's launch all 18
> training jobs and make sure that everything is connected.
>
> Once you launch all the training jobs the main goal will be to make sure this
> whole flow is working. The training jobs will start spilling off checkpoints
> but of course you will have to do a very long sleep.
>
> Once you sleep and wake up and see new checkpoints, make sure those checkpoints
> were actually picked up by the subscriber and the subscriber actually added
> them to the tournament. Once it gets into the tournament, make sure it runs,
> make sure it gets promoted correctly, and make sure the leaderboard gets
> updated.
>
> If you're stuck waiting and you're about to do a long sleep, another thing you
> can do is just do a separate dummy experiment. Maybe you should do one where
> you spin off a different arena, a dummy test arena, and just with five or ten
> policies. Then you start some other dummy training jobs and those dummy
> training jobs will be very fast, very small batches, dumping checkpoints every
> iteration so you can rapidly check.
>
> Let's launch all the training jobs for us just to be safe, the main ones, but
> then we should go into testing. If the testing fails we should figure out why
> it fails. We should iterate. We should figure out exactly what went wrong and
> fix it and then try again.

## Fixed Top10 Fallback Batch Status - 2026-05-14 10:29 EDT

Current fixed manifest:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-night18-top10fallback-fixed-20260514a/curvy-night18-top10fallback-fixed-20260514a.json
```

This is the current live 18-row batch. The older `top10r1` manifest remains
useful history, but it is not the clean fixed launch.

Status summary from Volume artifacts:

| Row | Status | Latest checkpoint | Reward | Noise |
| --- | --- | --- | --- | --- |
| r001 | running | `iteration_210000` | `sparse_outcome` | clean |
| r002 | running | `iteration_160000` | `sparse_outcome` | stochastic |
| r003 | failed | `iteration_130000` | `sparse_outcome` | clean |
| r004 | failed | `iteration_20000` | `sparse_outcome` | stochastic |
| r005 | running | `iteration_160000` | `sparse_outcome` | clean |
| r006 | running | `iteration_170000` | `sparse_outcome` | stochastic |
| r007 | running | `iteration_150000` | `survival_plus_bonus_no_outcome` | clean |
| r008 | running | `iteration_150000` | `survival_plus_bonus_no_outcome` | stochastic |
| r009 | running | `iteration_150000` | `survival_plus_bonus_no_outcome` | clean |
| r010 | running | `iteration_200000` | `survival_plus_bonus_no_outcome` | stochastic |
| r011 | failed | `iteration_0` only | `survival_plus_bonus_no_outcome` | clean |
| r012 | running | `iteration_200000` | `survival_plus_bonus_no_outcome` | stochastic |
| r013 | running | `iteration_110000` | `survival_plus_bonus_plus_outcome` | clean |
| r014 | running | `iteration_120000` | `survival_plus_bonus_plus_outcome` | stochastic |
| r015 | running | `iteration_120000` | `survival_plus_bonus_plus_outcome` | clean |
| r016 | running | `iteration_120000` | `survival_plus_bonus_plus_outcome` | stochastic |
| r017 | failed | `iteration_20000` | `survival_plus_bonus_plus_outcome` | clean |
| r018 | running | `iteration_120000` | `survival_plus_bonus_plus_outcome` | stochastic |

All four failures ended with the same LightZero replay-buffer error:
`ValueError: 'a' and 'p' must have same size`. Treat this as one replay
bookkeeping bug until proven otherwise, not as a reward-specific result.

Immediate next step:

1. Keep monitoring the 14 running rows.
2. Use the new replay invariant audit to make the next failure explain which
   replay-buffer list diverged.
3. Do not relaunch the failed rows until the replay-buffer mismatch is pinned
   down or a deliberate defensive repair is added.

## Replay Diagnostic Version - 2026-05-14 10:35 EDT

The trainer app was redeployed with replay-buffer invariant diagnostics. The
current `curvy-n18fb` batch was not killed.

New manifest:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-night18-replaydiag-20260514a/curvy-night18-replaydiag-20260514a.json
```

New run prefix:

```text
curvy-n18diag
```

Launched rows:

| Row | Reason |
| --- | --- |
| r003 | Sparse clean row that failed after `iteration_130000` in the fixed batch. |
| r004 | Sparse stochastic row that failed after `iteration_20000` in the fixed batch. |
| r011 | Survival+bonus/no-outcome clean row that failed almost immediately in the fixed batch. |
| r017 | Survival+bonus+outcome clean row that failed after `iteration_20000` in the fixed batch. |

These rows cap at `max_train_iter=80000`; their purpose is to expose the replay
invariant if the same failure recurs, not to replace the whole 18-run batch.

## Replay Fix And Fresh Full Version - 2026-05-14 10:42 EDT

The diagnostic row `r017` reproduced the replay crash and captured the exact
invariant:

```text
num_transitions=42014
game_segment_game_pos_lookup_len=42014
game_pos_priorities_len=107155
```

Cause: the trainer was setting LightZero `td_steps` to `source_max_steps`
(`65536`). For unfinished stock LightZero replay chunks, that made
`td_steps + num_unroll_steps` much larger than `game_segment_length`, so
LightZero appended far more priority entries than transition lookup entries.

Fix: stop overriding `policy.td_steps` from `source_max_steps` and add a guard
that fails config construction if `td_steps + num_unroll_steps >
game_segment_length`.

Verification:

- `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py -q -k 'survival_plus_bonus or replay_td_window or target_audit'`
  passed: `5 passed`.
- `uv run ruff check src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_curvytron_live_checkpoint_eval_plumbing.py`
  passed.

The trainer app was redeployed after this fix.

Fresh full manifest:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-night18-tdfix-20260514c/curvy-night18-tdfix-20260514c.json
```

Run prefix:

```text
curvy-n18tdfix
```

Launch status:

- All 18 rows submitted through the deployed app at about `2026-05-14 10:43 EDT`.
- Submission record:
  `artifacts/local/curvytron_tonight18_manifests/curvy-night18-tdfix-20260514c/submission.json`.
- The earlier `curvy-n18fb`, `curvy-n18diag`, and `curvy-n18new` rows were not
  killed during this launch.

First status check:

| Check | Result |
| --- | --- |
| Rows spawned | `18 / 18` |
| Rows running | `18 / 18` |
| Visible iteration range | `10000` to `30000` |
| Immediate replay crash | none seen |
| Checkpoint artifacts | present on sampled rows |

Use compact status reads from the Modal Python Volume API for this batch. The
plain CLI loop is too slow, and bulk checkpoint listing can hit Modal Volume
rate limits.

Second status check after a short wait:

| Check | Result |
| --- | --- |
| Rows running | `18 / 18` |
| Visible iteration range | `10000` to `30000` |
| Replay priority/count crash | none seen |

Current interpretation: `curvy-n18tdfix` is the trusted live version. The
pre-fix rows are useful only as failure evidence for the `td_steps` bug.

## Connected Assignment Batch - 2026-05-14 11:58 EDT

The first fixed 18-row batch is healthy but static. It does not read the
published leaderboard assignment path.

Built and launched a fresh connected copy:

| Field | Value |
| --- | --- |
| matrix | `curvy-night18-connected-20260514d` |
| manifest | `artifacts/local/curvytron_tonight18_manifests/curvy-night18-connected-20260514d/curvy-night18-connected-20260514d.json` |
| submission | `artifacts/local/curvytron_tonight18_manifests/curvy-night18-connected-20260514d/submission.json` |
| run prefix | `curvy-n18conn` |
| app | `curvyzero-lightzero-curvytron-visual-survival-train` |
| rows | `18` |

The builder now has an assignment-backed mode. It keeps the same human run
shape:

- reward variants: `sparse_outcome`,
  `survival_plus_bonus_no_outcome`,
  `survival_plus_bonus_plus_outcome`;
- superseded seat recipes from the first connected batch were
  blank5/wall5/top2-top1, blank10/wall5/top4-top3-top2-top1, and
  blank20/wall5/top1;
- noise: clean and `straight_override_p10_repeat_p10`.

The connected rows use `opponent_assignment_ref` and do not include inline
`opponent_mixture_spec`. The three assignment files were written through the
deployed trainer app to:

`training/lightzero-curvytron-visual-survival/curvy-n18conn-assignments-20260514d/attempts/try-n18conn-assignments-20260514d/opponents/assignments/.../assignment.json`.

Those historical assignments preserved the intended scripted seats:

- blank canvas no-op;
- 5% proactive wall-avoidant immortal opponent.

Current restart guidance has changed. Do not reuse the weak 5% immortal-pressure
recipes as the next default. The next manifest should use blank and hard-coded
sentinel opponents as immortal, allow only small explicit immortal frozen
checkpoint slices, and keep total immortal opponent pressure around `20-30%`.
Bootstrap can use curated exact checkpoint refs; it does not need a perfect
starting ranking.

Pre-launch checks:

- builder regression tests passed;
- assignment volume readback succeeded for all three recipes;
- dry-run submission accepted all 18 rows.

First post-launch check:

- `5 / 18` rows had reached `running`;
- `13 / 18` rows were still missing `latest_attempt.json`;
- no trainer crash was visible yet.

Second post-launch check:

- `18 / 18` rows wrote `latest_attempt.json`;
- `18 / 18` command heartbeats include `opponent_assignment_ref`;
- `15 / 18` rows wrote `progress_latest.json`;
- no failed status or replay priority/count crash is visible;
- `summary.json` is not written yet.

Tournament plan for connected checkpoints:

- Do not mutate `curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c`;
  that lane is historical evidence, not a current bootstrap blocker.
- When `curvy-n18conn` has at least two visible checkpoints, create a fresh
  connected tournament/rating lane:
  `curvy-night18-connected-20260514d / elo-night18-connected-20260514d`.
- Use `modal run --detach` for the intake seed with
  `run_id_prefix=curvy-n18conn`, `max_runs=18`, one-frame settings,
  `games_per_pair=21`, GIFs on, 5 sampled GIFs, and `intake_spawn_rating`.

Connected tournament seed result:

- Tournament/rating:
  `curvy-night18-connected-20260514d / elo-night18-connected-20260514d`.
- Seed used `modal run --detach`.
- Discovery found `18 / 18` connected checkpoint refs, all currently
  `iteration_0.pth.tar`.
- Seed enqueued 18 events and spawned rating call
  `fc-01KRKKZQ4WP06H48JVXHHXK9P1`.
- First progress check: `round-000000`, `153` pairs, `3213` games,
  phase `games_running`, `started_pair_count=1`, `failed_game_count=0`.

This proves the connected training batch can feed the tournament runner. It
does not yet prove trained checkpoints, promotion, public leaderboard publish,
or trainer refresh from the new connected leaderboard.

## Tdfix Tournament Intake - 2026-05-14 11:30 EDT

The trusted fixed training batch was not being watched by the old intake lane.
The active top10r1 watch still used `run_id_prefix=curvy-night18top10r1`, so it
would never discover `curvy-n18tdfix` checkpoints.

Started a clean tdfix intake/rating lane instead of mixing these rows into the
old top10r1 arena:

| Field | Value |
| --- | --- |
| tournament_id | `curvy-night18-tdfix-20260514c` |
| rating_run_id | `elo-night18-tdfix-20260514c` |
| run_id_prefix | `curvy-n18tdfix` |
| app run | `ap-NvvwpyJpNpD3WmUdIymEGb` |
| rating call | `fc-01KRKHR965EHQDQXWK4FBCS3J2` |

Seed result:

- found `18 / 18` fixed training runs;
- selected each run's latest visible checkpoint;
- enqueued 18 existing checkpoint events;
- wrote intake manifest
  `tournaments/curvytron/curvy-night18-tdfix-20260514c/intake/elo-night18-tdfix-20260514c/config.json`;
- spawned the rating loop.

Rating config:

| Setting | Value |
| --- | --- |
| checkpoint count | `18` |
| pair selection | `adaptive_v0` |
| pairs per round | `300` |
| games per pair | `21` |
| games per shard | `1` |
| one-frame setting | `decision_source_frames=1` |
| GIFs | `save_gif=true`, `gif_sample_games_per_pair=5` |
| active pool limit | `100` |
| placement gate | `20 opponents`, `420 games` |

Initial progress:

- `round-000000`;
- `pair_count=153`;
- `game_count=3213`;
- `phase=game_map_started`;
- no `latest.json` yet at the first post-seed check.

Follow-up progress scan:

- the compact `progress.json` was stale at `game_map_started`, but explicit
  progress with game-summary counting showed real work;
- `started_pair_count=90`;
- `completed_pair_count=51`;
- `completed_game_count=1611 / 3213`;
- `failed_game_count=0`;
- phase `games_running`.

So the tdfix tournament lane is alive. Continue monitoring until
`latest.json` exists, then publish only after checking active rows and pool
consistency.

Operational note: Modal warned the runs volume is using about 84% of available
inodes. Cleanup is now a real follow-up, but the immediate priority remains
verifying this tdfix rating round starts games and writes `latest.json`.

## Clean Exact Connected Proof - 2026-05-14 12:56 EDT

The first connected arena,
`curvy-night18-connected-20260514d / elo-night18-connected-20260514d`, is
contaminated and must not be used as final proof. A later broad-prefix relaunch,
`curvy-night18-connected-clean-20260514e`, was also diagnostic only because the
prefix could include smoke runs.

Current proof lane history:

| Field | Value |
| --- | --- |
| tournament_id | `curvy-night18-connected-clean2-20260514e` |
| rating_run_id | `elo-night18-connected-clean2-20260514e` |
| app run | `ap-n4CKU8j0jjgwskSKGe1voV` |
| rating call | `fc-01KRKPNVFBERMFS1T01AEZ3ZBA` |
| seed source | exact 18 manifest run IDs, not prefix |
| checkpoint count | `18 / 18`, no missing |
| pair plan | all-pairs, `153` pairs, `3213` games |
| game settings | one-frame, 21 games per pair, 5 GIF samples per pair |
| GIF speed | `gif_fps=80.0` |

Clean2 completed but had one failed pair (`21` failed games) from a checkpoint
zip-read error. Do not publish clean2 unless explicitly accepting that failure.
A new exact-ID arena was launched after rediscovery saw all 18 checkpoints at
normal size:

| Field | Value |
| --- | --- |
| tournament_id | `curvy-night18-connected-clean3-20260514e` |
| rating_run_id | `elo-night18-connected-clean3-20260514e` |
| app run | `ap-REwRdoHopbtosjBETcu6An` |
| rating call | `fc-01KRKQ8MDG4N52ZS5QGBXN7WQR` |
| seed source | exact 18 manifest run IDs, not prefix |
| checkpoint count | `18 / 18`, no missing |
| pool hash | `d3bdf59a46ac828a` |
| pair plan | all-pairs, `153` pairs, `3213` games |
| game settings | one-frame, 21 games per pair, 5 GIF samples per pair |
| GIF speed | `gif_fps=80.0` |

Use clean3 for the next publish/assignment proof only after:

- `results.json`, `ratings.json`, and `latest.json` exist;
- round input, progress, results, ratings, and latest agree on checkpoint count,
  pair count, game count, pool hash, and roster hash;
- failure count is zero if possible. A nonzero failure count must be explicitly
  justified before publishing;
- public leaderboard publish writes a durable snapshot and Dict pointer;
- a trainer smoke consumes the clean assignment and logs
  `opponent_provider_load_ok=true`.

Cleanup note:

- Stale detached tournament apps from earlier stress tests were stopped while
  clean2 was running, because they had hundreds of tasks and were no longer
  part of the proof lane.
- Preserve the deployed tournament app, deployed trainer app, all
  `curvy-n18conn-*` training artifacts, the tdfix lane, the clean2 diagnostic
  artifacts, and the clean3 app until the proof is complete.

## Loop18 Clean3 Training Batch - 2026-05-14 13:48 EDT

Training batch:

| Field | Value |
| --- | --- |
| matrix_id | `curvy-loop18-clean3-20260514e` |
| run prefix | `curvy-loop18c3` |
| manifest | `artifacts/local/curvytron_tonight18_manifests/curvy-loop18-clean3-20260514e/curvy-loop18-clean3-20260514e.json` |
| submission | `artifacts/local/curvytron_tonight18_manifests/curvy-loop18-clean3-20260514e/submission.json` |
| row count | `18` |
| launched | about `2026-05-14 13:29 EDT` |

First health read:

- all 18 train calls and all 18 poller calls are still running at the Modal
  call layer;
- 12 rows have written `iteration_0.pth.tar`;
- 6 rows have not yet written a checkpoint or are still in startup;
- no row-specific crash was found;
- `/runs` inode use is high, about 89%.

Live tournament watch issue:

- `curvy-loop18-live-clean3-20260514e` is diagnostic only because a broad
  `run_id_prefix=curvy-loop18c3` matched smoke runs.
- `curvy-loop18-live-main-20260514f` avoided smoke refs by using exact run IDs,
  but exposed a deeper contract issue: the intake scan did not support fixed
  source checkpoint refs and live run IDs in the same durable scan. A later
  scheduled tick could restore the source-only scan spec.

Patch status:

- mixed fixed-source-plus-live-run discovery is implemented;
- live-watch detection now ignores whether fixed source refs are present;
- submit no longer deletes fixed checkpoint refs when adding run IDs;
- focused and broader intake/tournament tests pass.

Next action:

- redeploy tournament app;
- create a new clean live arena, not reusing `clean3` or `main`;
- seed it in one call with both the clean3 source checkpoint refs and the exact
  18 loop18 run IDs;
- monitor until the manifest proves both sides are durable and no smoke refs are
  present.

Fresh mixed live arena:

| Field | Value |
| --- | --- |
| tournament_id | `curvy-loop18-live-mixed-20260514g` |
| rating_run_id | `elo-loop18-live-mixed-20260514g` |
| app run | `ap-uUOT68w7OQrOuG46mdI940` |
| rating call | `fc-01KRKSP9MZEQT3WQYPM1MH6NHM` |
| seed mode | one mixed `intake-seed`, not two racing seeds |
| source refs | 18 clean3 checkpoint refs |
| live watch | exact 18 loop18 run IDs |

Mixed seed result:

- `35` checkpoint refs found at seed time:
  18 clean3 source refs plus 17 live loop18 refs;
- one loop18 row was missing during the seed, but the next training status scan
  showed all 18 rows alive with at least `iteration_0`;
- several live rows had already advanced to `iteration_10000`;
- rating plan is all-pairs over 35 refs: `595` pairs and `12495` games;
- GIFs remain on: `save_gif=true`, `gif_sample_games_per_pair=5`,
  `gif_fps=80.0`.

Current proof after manual tick:

- A current-code `intake-tick` on the mixed arena found `36` refs with
  `missing_count=0`.
- It queued two new `iteration_10000` refs from loop18 runs:
  `curvy-loop18c3-sparse-blank5...so10rep10...iteration_10000` and
  `curvy-loop18c3-survbonusnoout-blank5...clean...iteration_10000`.
- `intake-drain` saw the first mixed rating round still running and correctly
  skipped a continuation spawn because the active rating claim had not finished.
- Progress at the last check: `38` started pairs, `11` completed pairs,
  `524 / 12495` games complete, `0` failed games.

Next proof gates:

- finish mixed rating round with zero or explicitly accepted failures;
- drain queued live checkpoint events into a continuation after the current
  writer finishes;
- publish public leaderboard;
- materialize/write an immutable `stable_slots_v1` assignment;
- run a trainer smoke that consumes that fresh assignment and logs loaded
  checkpoint opponents;
- for a leaderboard-derived quality batch, optionally set
  `initial_policy_checkpoint_ref` from a trusted active checkpoint.

## 2026-05-16 Current Bootstrap Review Artifact

The old loop18/clean3 material above is historical. The current prepared
bootstrap artifact is:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-r18v2-bootstrap-20260516a/curvy-r18v2-bootstrap-20260516a.json
```

Status:

- built from
  `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt`;
- not launched;
- `18` rows;
- all-v2 trainer app and Volumes from `src/curvyzero/contracts/curvytron.py`;
- `random_per_episode`;
- `save_ckpt_after_iter=10000`;
- `browser_lines + simple_symbols`;
- control-volume immutable assignments and refresh pointers;
- recipe immortal pressure totals: `20%`, `25%`, and `30%`;
- blank and hard-coded sentinels are always `opponent_immortal=true`;
- frozen checkpoint slots are mortal except explicit small `_immortal` slices.

Verification:

- manifest syntax audit: `ok=true`, `4` exact checkpoint refs, `0` bad refs;
- Modal ref audit: `ok=true`, all `4/4` referenced checkpoint files exist in
  `curvyzero-runs-v2`;
- grouped submitter dry-run: selected all `18` rows, would write `3`
  assignments and `3` refresh pointers, and did not spawn jobs;
- focused tests after the cleanup:
  `tests/test_opponent_registry.py tests/test_opponent_mixture.py tests/test_opponent_leaderboard.py`
  -> `53 passed`;
  `tests/test_curvytron_tonight18_manifest.py tests/test_env_contract.py tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`
  -> `69 passed`;
  `tests/test_curvytron_checkpoint_tournament.py tests/test_curvytron_live_checkpoint_eval_plumbing.py`
  -> `226 passed, 14 skipped`.

Plain warning: this artifact does not need a high-quality starting leaderboard.
It uses exact refs plus explicit immortal sentinels. A ranked source leaderboard
is only optional quality input for future leaderboard-derived slot choices.
