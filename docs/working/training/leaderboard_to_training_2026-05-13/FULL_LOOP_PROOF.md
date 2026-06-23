# FULL_LOOP_PROOF

This doc proves or disproves the full feedback loop. A link is not proven unless
it has concrete artifact refs or timestamps.

## Target Loop

```text
trainer writes checkpoint
-> live run-id intake/subscriber sees checkpoint
-> tournament rates checkpoint
-> public leaderboard snapshot + Modal Dict pointer update
-> Coach materializes immutable assignment + control pointer
-> same running trainer refreshes at a safe boundary
-> env telemetry shows the new assignment sha with provider-ok rows
```

Survival improvement is a separate learning-quality claim. It is important, but
it is not part of the wiring proof.

Correction, 2026-05-16: a stable Elo snapshot is not required for the training
loop to keep learning. `stable=true` only means the rating numbers have mostly
settled. The production loop still needs auditability, immutable assignments,
zero silent pointer rewrites, and visible labels, but it can consume a moving
training-candidate leaderboard. The missing large-batch proof is therefore not
"wait for stable"; it is "the controller automatically turns current tournament
output into per-recipe assignment pointers, and running trainers actually use
them."

Local toy proof added, 2026-05-16:
`tests/test_curvytron_training_candidate_controller_local.py` proves the
dangerous controller boundary without Modal remote compute. It uses fake local
tournament/runs/control mounts and a fake Dict. Covered cases: writes a
training-candidate leaderboard snapshot/latest and Dict pointer; rewrites a
control refresh pointer to a new immutable assignment; preserves blank/wall and
rank-slot recipe shape; rejects corrupted input pointer sha; aborts on reload
failure; aborts on tournament or control commit failure before trainer-facing
pointer rewrite; rejects stale pointer races; rejects missing active ranks; and
the trainer resolver can read the rewritten `control:` pointer and resolve the
new frozen checkpoint refs. This proves the controller shape locally. It does
not prove the deployed large 18-run lane until a remote controller call rewrites
the recipe pointers and the same running trainers log `decision=applied` plus
provider-ok env rows.

Corrected current live proof lane, 2026-05-16 05:43 EDT:
`curvy-r18fresh-live-bounded-dsf1-20260516b` /
`elo-r18fresh-live-bounded-dsf1-20260516b`. This is now the Tournament Arena
default because it carries explicit one-frame metadata. It was seeded from the
same 18 r18fresh run ids and found `287` checkpoints. Persisted
`round-000000/input.json` has `decision_source_frames=1`, `300` adaptive pairs,
`6,300` games, `21` games per pair, `games_per_shard=21`, `active_pool_limit=100`,
and `save_gif=false`. Rating call `fc-01KRR2PG4NN24PH9Q1B0QTB90V` continued
through multiple bounded rounds.

Corrected live loop proof closed, 2026-05-16 06:56 EDT:
`round-000002` wrote a one-frame `latest.json` with `304` rated rows and max
checkpoint iteration `220000`. The controller wrote training-candidate snapshot
`r18fresh-dsf1-r2-training-g9-20260516b`, published the Modal Dict pointer, and
rewrote the three live recipe control pointers to generation-9 assignment shas
`bb33908f...`, `03b97093...`, and `c6b5d111...`. After the long wait, every
one of the `18` running trainers had env telemetry rows using one of those
assignment shas. Remote tail proof across the 18 env logs found `177,382`
target-sha rows, `89,934` frozen-checkpoint opponent rows with
`opponent_provider_load_ok=true`, and `0` provider-load false rows. Sample rows
show `decision_source_frames=1`, `frozen_lightzero_checkpoint` opponents, and
control-volume frozen checkpoint refs copied from the tournament-derived
training-candidate assignments. This closes the live wiring proof:
checkpoints reached the tournament, tournament output became assignment
pointers, the same running trainers refreshed, and self-play actually loaded
the new frozen opponents.

Follow-up long-running proof turn, 2026-05-16 07:30 EDT:
the tournament kept advancing after generation 9. Current latest is
`round-000008` with `365` rated rows, max checkpoint iteration `260000`,
`decision_source_frames=1`, `300` pairs, and `6,300` games. Controller
generation 10 was published from `round-000006` as
`r18fresh-dsf1-r6-training-g10-20260516b`, with assignment shas
`ffc5bc3f...`, `c25af087...`, and `673381d4...`. After a 30-minute sleep, every
one of the `18` running trainers had env telemetry rows using the gen-10 shas.
Remote tail proof across the 18 env logs found `177,203` target-sha rows,
`93,427` frozen-checkpoint opponent rows with `opponent_provider_load_ok=true`,
and `0` provider-load false rows. This proves the loop keeps working across a
later tournament snapshot, not only the first refresh. The controller call
itself is still operator-triggered in this lane; full autonomy needs a durable
periodic controller.

Automation turn, 2026-05-16 11:55 EDT:
`curvytron_training_candidate_refresh_tick` is deployed in
`curvyzero-checkpoint-tournament-v2` with a 30-minute schedule. A manual smoke
of that same scheduled function published generation 11 from tournament
`round-000009` as `auto-r000009-g11-f2805622`; assignment shas were
`2f33dab8...`, `8e6f41c1...`, and `0564089c...`. After the shorter post-smoke
wait, the tournament latest had advanced again to `round-000011` with `398`
rated rows and max checkpoint iteration `290000`. Remote env-tail proof showed
partial gen-11 trainer uptake: `14/18` trainers had gen-11 rows, with
`137,019` target-sha env rows, `83,945` frozen-checkpoint provider-ok rows, and
`0` provider-load false rows. Four trainers were still on gen-10, so gen-11 is
not counted as a closed 18/18 proof yet. Next proof action is a full 30-minute
sleep followed by another tournament/latest and assignment-uptake check.

Scheduled controller proof, 2026-05-16 12:40 EDT:
after the requested full 30-minute sleep, the deployed scheduled controller had
published generation 12 from the live tournament `round-000011` as
`auto-r000011-g12-65387f58`, with assignment shas `899ed9c6...`,
`1d9c86ff...`, and `2af7c6ff...`. The tournament latest at that point had
`398` rated rows, max checkpoint iteration `290000`, one-frame metadata, `300`
pairs, and `6,300` games. Status showed `15/18` trainers still running and
`3/18` completed. Remote env-tail proof over the still-running trainers found
`15/15` with gen-12 target-sha rows, `146,976` total target rows, `84,578`
frozen-checkpoint provider-ok rows, and `0` provider-load false rows. Plain
meaning: new checkpoints reached the tournament, the scheduled controller
turned a current tournament snapshot into fresh assignments, and every trainer
that was still alive consumed those tournament-derived frozen opponents.

First bounded live proof lane, 2026-05-16 05:26 EDT:
`curvy-r18fresh-live-bounded-20260516a` /
`elo-r18fresh-live-bounded-20260516a`. It completed and wrote final
`latest.json` with `272` rated rows through max iteration `190000`, `300` rated
pairs, and `6,300` games. It is not publishable as trainer source because
`rating_spec.decision_source_frames` was `null`; the training-candidate
controller correctly refused it. The code default was fixed after this failure
so future rating snapshots carry explicit `decision_source_frames=1`.

Remote r18fresh controller proof, 2026-05-16: subagent artifact audit found the
generation-8 training-candidate snapshot
`r18fresh-round-000007-training-g8-20260516a` with `71` rows. Three gen-8
assignments were materialized and installed into the three live control
pointers, with shas beginning `de7b812c`, `d5c73188`, and `ac99e534`. Same
running trainers consumed those assignments: `18/18` trainer refresh logs had
`decision=applied` and `env_ready_report.ok=true` between
`2026-05-16T06:47:22Z` and `2026-05-16T07:09:37Z`. Env rows with those
assignment shas and provider-loaded checkpoint opponents exist for all three
recipes. Plain meaning: r18fresh wiring is proven at live-batch scale once.
The freshness gap remains: that controller generation was based on an older
completed snapshot, while newer checkpoints now reach `iteration_190000`.

Scheduler bug fixed, 2026-05-16: live run-id intake previously inherited
unbounded all-pairs behavior. The dirty live lane proved the failure mode when
`round-000013` admitted `261` refs and scheduled `33,930` pairs / `712,530`
games. Current code now makes live intake continuation default to
`adaptive_v0`, `pairs_per_round=300`, and `active_pool_limit=100`; exact-ref
stress/validation lanes can still use all-pairs.

## Proof Table

Current reset warning, 2026-05-15: the active launch surface has been moved to
fresh all-v2 storage. The required current objects are `curvyzero-runs-v2`,
`curvyzero-curvytron-tournaments-v2`, `curvyzero-curvytron-control-v2`,
`curvyzero-curvytron-checkpoint-intake-v2`,
`curvyzero-curvytron-checkpoint-events-v2`, and
`curvyzero-curvytron-opponent-leaderboard-live-v2`. Old/hybrid proof remains
valid as a mechanics proof only. The all-v2 lane needed its own fresh canary
after object recreation and redeploy, and that canary has now passed.

Current all-v2 proof result: `curvy-e2e-allv2-canary-20260515a` /
`try-e2e-allv2-canary-20260515a`, with fresh tournament
`curvy-e2e-allv2-canary-live-20260515a` and rating
`elo-e2e-allv2-canary-live-20260515a`, passed the full same-trainer loop on the
recreated all-v2 objects: checkpoint -> v2 intake -> v2 tournament -> v2
leaderboard/promotion -> v2 control pointer -> same running trainer applies the
promoted assignment and records env rows using it.

Current 2026-05-16 all-205 validation truth: the clean validation tournament
`curvy-r18fresh-validate-all205-20260516a` has a persisted all-pairs input with
`205` checkpoint refs, max iteration `140000`, `20,910` pairs, and `439,110`
games. Logs prove games are running with balanced randomized seats and
`max_steps=1048576`. It is not yet a full-loop proof because no completed
`ratings.json`/`latest.json` exists for this validation rating, and no
controller-produced assignment from this validation leaderboard has been
observed in the same running trainers. A 30-minute observation poll is in
progress to check whether new checkpoints complete the path.

Current larger-batch proof status, 2026-05-15:
`curvy-r18bootfix-20260515a` is the active 18-row bootstrap batch. It was
launched from curated exact checkpoint refs, not from a trusted ranked
leaderboard. The three assignment recipes use immortal blank/hard-coded
sentinel pressure plus mostly mortal frozen checkpoint slots, with total
immortal exposure of `20%`, `25%`, and `30%`. The first failed bootstrap launch
(`curvy-r18bootrefs-20260515a`) exposed a control-volume visibility race; the
fresh launch after the fix showed all `18` trainers and pollers alive and later
all `18` rows had at least `iteration_0` checkpoints. A later current-prefix
discovery found `13/18` latest refs at `iteration_10000.pth.tar`, `5/18` latest
refs still at `iteration_0.pth.tar`, `0` missing refs, and expected checkpoint
sidecar metadata. Live tournament intake then seeded `32` current-batch
checkpoints into `curvy-r18bootfix-live-20260515a` /
`elo-r18bootfix-live-20260515a`; `round-000000` completed all-pairs with
`496` pairs, `10,416` games, `0` failed games, and `ratings_written=true`.
The round is `stable=false` with `max_abs_delta=117.51743133112922`. Plain
meaning: the tournament ran and wrote ratings, but one pass moved ratings a lot,
so this snapshot is useful live-system proof, not final ranking truth. This
proves launch, checkpoint writing, checkpoint discoverability, intake seeding,
and active tournament play. It does not yet prove the larger-batch full loop.
The next required proof is public leaderboard publication,
tournament-derived assignment/pointer materialization, and then use by a running
patched trainer with provider-ok env telemetry.

Plain correction: a strong starting leaderboard is not required for this proof.
It only improves the quality of starting frozen opponents. The loop can and
should be proved with exact refs, immortal blank/hard-coded sentinels, and the
new checkpoints produced by the trainers.

Current larger-batch proof status, 2026-05-16:
`curvy-r18v2-bootstrap-20260516a` is the fresh all-v2 bootstrap batch. It was
submitted to `curvyzero-lightzero-curvytron-visual-survival-train-v2` after the
latest hardening and redeploy checks. Submission wrote `3` control assignments,
`3` control refresh pointers, and spawned `18` trainers plus `18` pollers. First
status showed all `18` rows have `iteration_0` checkpoints and poller/eval/GIF
artifacts. Live run-id tournament intake was seeded as
`curvy-r18v2-bootstrap-live-20260516a` /
`elo-r18v2-bootstrap-live-20260516a` with `checkpoint_selection=all`; it found
`18/18` startup checkpoints, missed `0`, enqueued `18`, drained queue length to
`0`, and spawned rating call `fc-01KRQ78KSP09GH0R4K5B4YH3QE`. Rating
`round-000000` has `18` players, `153` all-pairs battles, and `3213` games,
with one-frame timing, `max_steps=1048576`, eval mode, and the trained
`browser_lines + simple_symbols` policy surface. This proves launch,
checkpoint writing, intake seeding, and tournament rating startup for the fresh
large batch. It does not yet prove completed ratings, public leaderboard
publication, assignment materialization, same-trainer refresh, provider-ok use,
or survival improvement.

Follow-up in the same fresh batch: every row reached at least `iteration_10000`,
many reached `iteration_20000`, and the live intake manifest grew from `18`
startup refs to `47` refs with `0` missing. This proves new numbered trainer
checkpoints are visible to the live intake. The first rating round has started
all `153` pairs and planned `3213` games; wait for final pair summaries/latest
before publication or continuation.

Current large-batch check, 2026-05-16: the same `18` trainers are still alive
and still writing/evaluating checkpoints. Latest status showed every row at
least `iteration_10000`, most rows at `iteration_20000` or `iteration_30000`,
and `2-4` checkpoints per row. The live intake manifest now has `64` refs, so
numbered checkpoints from the running trainers are continuing to enter the
watch state. Durable rating output exists through `round-000002` with `37`
checkpoints, `666` pairs, `13,986` games, and `stable=false`
(`max_abs_delta=137.8483556121352`). `round-000003` is running with `64`
checkpoints, `1,596` all-pairs matchups, and `33,516` planned games. This
proves the large-batch trainer -> intake -> tournament path is functioning
under load. It still does not prove a production public leaderboard snapshot,
large-batch assignment materialization, same-trainer refresh from that large
leaderboard, or survival improvement.

Large-batch live-watch repair, 2026-05-16: an exact-ref submission had made the
large intake scan explicit-ref only. That was wrong for a live batch because it
would keep the submitted files but stop watching the 18 run ids for future
checkpoints. The running intake was re-seeded with the 18 run ids and
`checkpoint_selection=all`; direct Volume config now shows `18` run ids,
`83` seen refs, and `continue_from_latest=true`. Local code now preserves an
existing live watch when exact refs are pinned into the same manifest, and
drain now refuses to spawn a continuation while a newer rating round is still
missing its `ratings.json`. Focused tests and ruff passed, and the v2
tournament app was redeployed. This repair supports future discovery and avoids
duplicate/overlapping continuations; it does not by itself prove the large
leaderboard has been published or consumed by trainers.

Live large-batch continuation repair, later 2026-05-16: the previous paragraph
was not the whole story. `round-000003` finished and wrote durable ratings, but
the snapshot was still unstable with `57` rated checkpoints. The live intake
then held `92` checkpoint refs. Trying to continue exposed two practical bugs:
first, reading progress before a round starts can write a zero-work
`round-000004/progress.json`, and the round writer incorrectly treated that as
a real artifact; second, a detached rating loop started child rounds with
`.remote()`, which Modal can cancel when the local caller disconnects. Both
bugs now have focused tests and deployed fixes. Current proof state:
`round-000004/input.json` exists, games are running, logs show balanced random
seat order and `max_steps=1048576`, and the large lane is again moving.
Progress read at `2026-05-16T03:02Z`: `4,186` pairs, `87,906` games planned,
`886` pairs started, about `18,606` games seen by shard-summary estimate, and
`0` estimated failures. Logs from active detached worker
`ap-t8dhK6PpMxvqhyGo6hMRrG` show successful game summaries, balanced random
seat order, current r18v2 checkpoint ids, and `max_steps=1048576`. `latest.json`
still points at unstable `round-000003`, so no public leaderboard or assignment
should be made from this large lane yet. Broad tournament/intake regression
passes (`162 passed, 11 skipped`) and ruff passes for touched files. It is still
not full large-loop proof until latest becomes stable, is published, is
materialized into an assignment, and is consumed by the same running trainers.

Do not confuse this with the sparse canary proof above. The canary already
proved the full wiring path at small scale. The big batch is now proving scale
and learning signal. Because its latest large snapshot is still unstable, the
correct action is to keep rating/monitoring, not to force a non-diagnostic
publish.

Slot correction: blank and hard-coded sentinel opponents should be immortal
every time they are sampled. Frozen checkpoint slots should usually remain
mortal, with explicit small immortal slices where a recipe asks for harder
pressure. Keep total immortal exposure around `20-30%`; do not block launch on
perfect slot composition if the loop proof itself is still pending.

Local and remote hardening note, 2026-05-16: the trainer resolver no longer
puts internal checkpoint Volume reload warnings inside `opponent_mixture.entries`.
Those entries are env-facing gameplay slots and must stay clean. The warning is
now resolved-assignment metadata. Regression and ruff passed locally; the v2
trainer app was redeployed.

Fresh warningfix canary proof, 2026-05-16:
`curvy-e2e-warningfix-canary-20260516a` launched from the redeployed trainer
image. It wrote checkpoints through at least `iteration_5600`. A tiny frozen
tournament `curvy-e2e-warningfix-live-20260516a` /
`elo-e2e-warningfix-live-20260516a` rated `iteration_0` against
`iteration_1000` with `21` games, `0` failures, `stable=true`, and
`ratings_written=true`. Promotion published public leaderboard snapshot
`e2e-warningfix-live-r0-20260516a`, wrote assignment sha
`8b171c177c401b886a5658fafc1c16076b5797c640b6d6a689003575e6d46208`, and
rewrote the running trainer's control pointer. The same trainer then logged
`decision=applied` for that sha at train iter `5693` with
`env_ready_report.ok=true`. Its env-step log contained `87` rows using the new
assignment sha; all `87` rows had `opponent_provider_load_ok=true`, and no
provider-load-false row was observed. This closes the current canary-scale
refresh proof after the warning-leak fix.

Current-code live-watch canary, 2026-05-16:
`curvy-e2e-currentlive-canary-20260516a` launched after the latest v2 trainer
and tournament redeploys. Submission spawned trainer
`fc-01KRQ4B3MMWSX44VS8ENSDAQM5` and poller
`fc-01KRQ4B3FKT7VS3QZA2AFQQHY7`; starter assignment sha:
`af4167dd73868e5d7444b4b40b7ef28c86f6cfdc71c41f62a5cda368e02df81f`.
First status showed checkpoints through at least `iteration_1250` and refresh
events still `unchanged`, which is expected before promotion. Live-watch intake
was seeded from the run id, not explicit checkpoint refs:
`curvy-e2e-currentlive-live-20260516a` /
`elo-e2e-currentlive-live-20260516a`. The seed discovered `34` current
checkpoints, enqueued `34` checkpoint events, preserved checkpoint sidecar
metadata, and spawned rating call `fc-01KRQ4E0P30S5N6HECHXYQ8VPJ`. Round 0 has
`630` all-pairs battles and `13,230` planned games. This proves current-code
run-id intake can see fresh checkpoints and hand them to a tournament. It does
not yet prove public leaderboard publication, assignment materialization,
same-trainer refresh, or provider-ok env use for this current-code canary.
Those remain the live proof gate.

Parallel sparse canary for a faster live proof:
`curvy-e2e-currentlive-sparse-canary-20260516a` was launched with checkpoint
cadence `1000` and max train iter `8000`. Dry-run and Modal ref audit passed;
trainer `fc-01KRQ4K9RGG7P5VAQBS9H19Z3W`, poller
`fc-01KRQ4K9N4M9WMA0SGWF5D56GW`, starter assignment sha
`4b7261e7a795da17517360a768b581879dd20ffa034d30f8cc7540858a731f4b`.
First status showed the process alive before any checkpoint was written. The
planned proof is to seed live run-id intake after `iteration_1000` exists so
the tournament has only `iteration_0` and `iteration_1000`, then require the
same publish/materialize/pointer/apply/provider-ok chain.
That live seed is now done: intake found exactly the two sparse checkpoints,
queued two events, preserved sidecar metadata, and spawned rating call
`fc-01KRQ4VESF8BRCWYY763KDG09S`; round 0 is `1` pair and `21` games.
The sparse rating completed final `round-000007` with `1` pair, `21` games,
`0` failures, `stable=true`, and `max_abs_delta=4.060196778430208`.
Promotion published leaderboard snapshot sha
`a1a003523adde3f3fc273ecba9b825da60f3c75f0b7ff23064e2db34ea24a79b`,
materialized assignment sha
`774b70dd15fa71bc59a92819f3d417c9025184d6a24634ad4dbebe490dbb1009`, and
rewrote the sparse trainer's control pointer. The same running trainer then
emitted `decision=applied` for that sha at train iter `5373` with
`env_ready_report.ok=true`. A later `env_steps.jsonl` fetch had `357` rows with
the promoted sha, including `312` rows with `opponent_provider_load_ok=true` and
`0` observed rows with `opponent_provider_load_ok=false`. This closes the
current-code live run-id proof at canary scale: trainer checkpoint -> live
run-id intake -> tournament -> public leaderboard -> assignment/pointer -> same
trainer refresh -> provider-ok env use.

Follow-up status: the sparse watch later contained `6` checkpoints through
`iteration_5000` with queue length `0`, and the trainer remained alive on the
promoted assignment. The larger high-frequency current-live canary is recorded
only as stress evidence: it produced many checkpoints and evals, but never
applied a promoted assignment, and its failed status was caused by a wrapper
artifact-scan path bug rather than a clear LightZero train failure. Local fix:
scan `RUNS_MOUNT / exp_name` for relative `exp_name` values; validation:
`tests/test_curvytron_live_checkpoint_eval_plumbing.py` -> `88 passed, 3
skipped`.

False-proof warning: the active r18bootfix trainers were launched before the
latest deployed refresh-reload patch. Their checkpoints can prove the
tournament lane, but they cannot close the patched-trainer refresh proof. Close
that separately with a fresh patched canary or a relaunch.

2026-05-15 patched refresh canary, current run:
`curvy-e2e-patched-refresh-canary-20260515a` /
`try-e2e-patched-refresh-canary-20260515a` launched after the trainer
refresh-reload patch. Starter assignment sha:
`9ebcba51ffa02e7ef39efb06cd7feebaae457901465c47807d17712601ee0ea7`.
The trainer reached at least iteration `1500` on the first status read and
checkpoint discovery has already found numbered checkpoints through at least
`iteration_2700.pth.tar`. Live canary tournament intake:
`curvy-e2e-patched-refresh-live-20260515a` /
`elo-e2e-patched-refresh-live-20260515a`. The first seed admitted `22`
checkpoints with `0` missing and correct sidecar metadata
(`cpu_oracle`, `browser_lines + simple_symbols`, `random_per_episode`, one
source frame). A later status showed the manifest at `28` seen checkpoints.
The active rating round `round-000000` started with `26` checkpoints,
`325` all-pairs battles, and `975` planned games; progress at
`2026-05-16T00:02:09Z` showed `270/975` estimated games seen. Round 0 later
completed with `975/975` games, `0` failed games, and ratings written, but
promotion correctly refused because `stable=false`
(`max_abs_delta=90.66666666666667`). Round 1 is running with the growing live
pool, so it is useful live-watch evidence but not the clean promotion gate.
Caveat: the first non-detached seed hit a Modal app-lifecycle conflict while
spawning a background rating worker, but the durable rating artifacts exist and
the live-watch round did run.

To close the promotion gate without a moving pool, a frozen two-checkpoint
canary was seeded from exact refs produced by the same patched trainer:
`curvy-e2e-patched-refresh-frozen2-20260515a` /
`elo-e2e-patched-refresh-frozen2-20260515a`. It uses `iteration_0.pth.tar` and
`iteration_1000.pth.tar`, explicit-ref intake, `21` games per pair,
`round_count=8`, `stop_when_stable=true`, one-frame timing, eval mode, and
`browser_lines + simple_symbols`. Frozen2 reached final `round-000007` with
`1` pair, `21` games, `0` failed games, `stable=true`, and
`max_abs_delta=1.1749968212540947`. Promotion succeeded at
`2026-05-16T00:13:44Z`: public leaderboard snapshot
`tournaments/curvytron/leaderboards/e2e-patched-refresh-frozen2-20260515a/snapshots/e2e-patched-refresh-frozen2-r7-20260515a.json`,
snapshot sha `9e162f96d1f7fa7ef2ef8204df488862ef0a57ad63f839f5366790c00f39eae9`,
assignment ref
`control:training/lightzero-curvytron-visual-survival/e2e-patched-refresh-promotion-bank-20260515a/attempts/try-e2e-patched-refresh-promotion-bank-20260515a/opponents/assignments/e2e-patched-refresh-frozen2-r7-assignment-20260515a/assignment.json`,
and assignment sha
`f8a469b5ff8598fe64bd42906de64fb68d06a8aa75f6f4a2c20be82fa4c8eedc`. The
canary refresh pointer was rewritten to this assignment. Required close:
confirm the same running patched trainer emits `decision=applied` and later env
rows with this assignment sha and `opponent_provider_load_ok=true`.

2026-05-15 all-v2 launch note: before launching this canary, the submitter was
patched so direct refresh-pointer writes use
`modal_volume_kwargs_for_name(...)`; this makes `curvyzero-curvytron-control-v2`
open with `version=2` instead of relying on Modal defaults. Focused validation:
`tests/test_curvytron_tonight18_manifest.py`,
`tests/test_curvytron_survivaldiag_submitter.py`, and
`tests/test_curvytron_shared_contracts.py` -> `15 passed`; ruff passed for the
submitter and touched tests. One historical seed checkpoint was explicitly
copied from `curvyzero-runs` into `curvyzero-runs-v2`:
`training/lightzero-curvytron-visual-survival/curvy-looplive-proof-controlrun2-20260515f/attempts/try-looplive-proof-controlrun2-20260515f/train/lightzero_exp/ckpt/iteration_8019.pth.tar`.

2026-05-15 all-v2 canary launch: manifest written to
`artifacts/local/curvytron_e2e_canary/curvy-e2e-allv2-canary-20260515a/manifest.json`.
Submission wrote starter assignment sha
`117226d3a761cdddff995b62f9196aeca47d6a711f8048e3d07e77f8bc81cf9a`, wrote the
control refresh pointer in `curvyzero-curvytron-control-v2`, and spawned poller
`fc-01KRPDXH9FW4DNNMW41G9XC3EY` plus trainer
`fc-01KRPDXHDF4C5RS17DR1Z18EVX` on
`curvyzero-lightzero-curvytron-visual-survival-train-v2`.

2026-05-15 all-v2 proof close:

- Trainer wrote checkpoints through at least
  `training/lightzero-curvytron-visual-survival/curvy-e2e-allv2-canary-20260515a/attempts/try-e2e-allv2-canary-20260515a/train/lightzero_exp/ckpt/iteration_5300.pth.tar`
  by `2026-05-15T18:39:01Z`.
- Live v2 intake admitted checkpoints from the run and the v2 tournament rating
  `round-000003` completed with `checkpoint_count=25`, `6` pairs, `18` games,
  `0` failures, `ratings_written=true`, and `stable=true`.
- Promotion published leaderboard snapshot
  `tournaments/curvytron/leaderboards/e2e-allv2-canary-live-r3-20260515a/snapshots/e2e-allv2-canary-live-r3-20260515a.json`
  with sha `c7ac3d894b3780b29d1e99572c5e5c91b5cd3008ff0f68308101510f15319ed1`.
- Promotion materialized assignment
  `control:training/lightzero-curvytron-visual-survival/e2e-allv2-canary-promotion-bank-20260515a/attempts/try-e2e-allv2-canary-promotion-bank-20260515a/opponents/assignments/e2e-allv2-canary-live-r3-assignment-20260515a/assignment.json`
  with sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`.
- Champion checkpoint for that assignment:
  `training/lightzero-curvytron-visual-survival/curvy-e2e-allv2-canary-20260515a/attempts/try-e2e-allv2-canary-20260515a/train/lightzero_exp/ckpt/iteration_0.pth.tar`.
- The mutable v2 control pointer was rewritten at
  `control:training/lightzero-curvytron-visual-survival/e2e-allv2-canary-control-20260515a/attempts/try-e2e-allv2-canary-control-20260515a/opponents/current_assignment_pointer.json`.
- The same running trainer emitted `decision=applied` at train iter `5061`,
  `refresh_index=1`, with all `8` envs ready and assignment sha
  `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`.
- A later refresh event at train iter `5372` stayed on the promoted sha.
- `env_steps.jsonl` contained `22572` rows on the latest post-refresh fetch;
  `1836` rows used the promoted assignment sha with
  `opponent_provider_load_ok=true`. Those rows also show
  `browser_lines + simple_symbols`, one source frame, and alternating
  controlled-player seats in telemetry.
- Caveat: this was a tiny canary promotion, so `--allow-recent-provisional` and
  relaxed active-row thresholds were intentionally used. That is acceptable for
  wiring proof only, not as a production leaderboard-quality gate.
- Caveat: the champion checkpoint in this tiny proof was
  `iteration_0.pth.tar`; a later `recent_strong` slot reached a nonzero
  checkpoint such as `iteration_2400.pth.tar`. This proves the refreshed
  assignment can load/use current-run checkpoints, not that a learned nonzero
  checkpoint was production-quality champion.

Historical storage warning, 2026-05-15 13:57 EDT: before the all-v2 reset, the
trainer was temporarily hybrid. That is no longer the current launch state. The
long clean canary below remains valid proof of the loop in its original storage
lane, but those assignment refs are not valid smoke inputs for the recreated
all-v2 deployment unless explicitly rematerialized.

| Link | Status | Required proof | Current evidence |
| --- | --- | --- | --- |
| Trainer writes checkpoints | Proven for all-v2 canary | Exact checkpoint refs from current-code run | `curvy-e2e-allv2-canary-20260515a` wrote through at least `iteration_5300.pth.tar` |
| Subscriber sees checkpoints | Proven for all-v2 canary | Durable intake manifest count and refs | v2 intake manifest discovered `27` refs through `iteration_2600.pth.tar`; rating latest covered `25` refs |
| Tournament rates checkpoints | Proven for all-v2 canary | Completed `latest.json` covering intended tiny roster with zero failures | `elo-e2e-allv2-canary-live-20260515a`, `round-000003`, `6` pairs / `18` games / `0` failures, `stable=true` |
| Public leaderboard updates | Proven for all-v2 canary | Immutable public snapshot sha/ref from clean current rating | `e2e-allv2-canary-live-r3-20260515a`, snapshot sha `c7ac3d894b3780b29d1e99572c5e5c91b5cd3008ff0f68308101510f15319ed1` |
| Coach materializes assignment | Proven for all-v2 canary | Assignment ref + audit generated from public snapshot | `e2e-allv2-canary-live-r3-assignment-20260515a`, sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0` |
| Trainer consumes assignment at launch | Proven for clean bounded round | Env telemetry rows with assignment sha and opponent loads | Smoke `loop18-main-adaptive417-consume-smoke-20260515e` verified provider/env rows |
| Trainer starts from tournament winner | Proven for clean bounded round | Initial model load from exact champion checkpoint, fresh optimizer preserved | Smoke `loop18-main-adaptive417-consume-smoke-20260515e` loaded rank-1 `iteration_240000.pth.tar` with `matching_shape` |
| Running trainer refreshes assignment | Proven for all-v2 canary | Refresh JSONL plus post-refresh env telemetry | All-v2 canary applied sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0` at train iter `5061`, stayed on it at train iter `5372`, and env telemetry had `1836` provider-ok rows with that sha on the latest fetch |
| Survival improves | Partial and noisy | Quantified eval/collector progression | Latest `curvy-n18conn-*` eval-summary: best mean improves `+49.8`, latest mean improves only `+9.2`, latest improves `10/18` rows |

## 2026-05-15 Historical Post-Cleanup Old-Lane Canary

This is historical old-lane proof after the seat/slot cleanup. The current
active proof is the all-v2 canary above; older `controlrun2` and
`clean-canary-long` evidence are templates, not current launch inputs.

- Trainer row:
  `curvy-e2e-clean-canary-20260515a` /
  `try-e2e-clean-canary-20260515a`.
- Starter assignment:
  `control:training/lightzero-curvytron-visual-survival/e2e-clean-canary-assignment-bank-20260515a/attempts/try-e2e-clean-canary-assignment-bank-20260515a/opponents/assignments/e2e-clean-canary-initial-blank-20260515a/assignment.json`.
- Starter assignment sha:
  `a34450084bdb0ef771e0d8ebc53e362322117308e75ee9d7ee5e2f6686316f69`.
- Mutable refresh pointer:
  `control:training/lightzero-curvytron-visual-survival/e2e-clean-canary-control-20260515a/attempts/try-e2e-clean-canary-control-20260515a/opponents/current_assignment_pointer.json`.
- Initial model checkpoint:
  `training/lightzero-curvytron-visual-survival/curvy-looplive-proof-controlrun2-20260515f/attempts/try-looplive-proof-controlrun2-20260515f/train/lightzero_exp/ckpt/iteration_8019.pth.tar`.
- Fresh tournament ids:
  `curvy-e2e-clean-canary-20260515a` /
  `elo-e2e-clean-canary-20260515a`.
- Generated manifest:
  `/private/tmp/curvy-e2e-clean-canary-20260515a/manifest.json`.
- Submission result:
  `/private/tmp/curvy-e2e-clean-canary-20260515a/submission.json`.
- Deployed trainer calls:
  train `fc-01KRP5T71HDVF5TSRMDYXNP9CB`, poller
  `fc-01KRP5T6WZYBA4C24VGJ6S42NM`.

Pass condition: a new checkpoint from this trainer is admitted to the fresh
tournament, rating `round-000000` completes with zero failed games, promotion
writes a different assignment sha to the pointer above, and later
`opponent_assignment_refresh_events.jsonl` plus `env_steps.jsonl` prove the same
running trainer applied and used that new sha.

2026-05-15 12:12 EDT update:

- First canary `curvy-e2e-clean-canary-20260515a` produced checkpoints through
  `iteration_3950.pth.tar` and then ended at `max_train_iter=4000`.
- A clean live-watch tournament was seeded as
  `curvy-e2e-clean-canary-live-20260515b` /
  `elo-e2e-clean-canary-live-20260515b` using the trainer `run_id` only and
  `checkpoint_selection=all`. It discovered `48` checkpoints from that run and
  enqueued them.
- The drain/rating round completed `3` pairs / `9` games / `0` failed games
  with `stable=true`. Game rows show `seat_order.mode=balanced_random`.
- Promotion published leaderboard snapshot
  `e2e-clean-canary-live-r0-20260515b`, wrote assignment
  `control:training/lightzero-curvytron-visual-survival/e2e-clean-canary-promotion-bank-20260515b/attempts/try-e2e-clean-canary-promotion-bank-20260515b/opponents/assignments/e2e-clean-canary-live-r0-assignment-20260515b/assignment.json`,
  assignment sha
  `5c3f3b67057132d699f76b960fd345239ff9a3af6e2e2dfe5f15737d8fa6e8af`, and
  rewrote the original control pointer.
- This proves checkpoint production -> live run-id intake -> tournament rating
  -> public leaderboard -> assignment materialization -> pointer rewrite on
  current code.
- It does not prove same-process refresh: the pointer was rewritten at
  `16:09:21Z`; the last refresh event was still the starter sha at
  `16:09:04Z`, train iter `3781`, and the trainer ended before another refresh.
- Next validation action: launch a longer second canary and repeat the same
  front-half proof with enough post-promotion runtime to observe
  `decision=applied` and env rows using the promoted sha.

Long canary launched:

- Trainer row:
  `curvy-e2e-clean-canary-long-20260515c` /
  `try-e2e-clean-canary-long-20260515c`.
- Deployed train call: `fc-01KRP6J8GDR05P6MYRZKXVQ9QK`.
- Initial model checkpoint: first canary champion
  `training/lightzero-curvytron-visual-survival/curvy-e2e-clean-canary-20260515a/attempts/try-e2e-clean-canary-20260515a/train/lightzero_exp/ckpt/iteration_1000.pth.tar`.
- Starter assignment sha:
  `2eedcdd422bef01baab22b7c9f2aa1c742310c9626dbb81b3110428e98e99a0a`.
- Refresh pointer:
  `control:training/lightzero-curvytron-visual-survival/e2e-clean-canary-long-control-20260515c/attempts/try-e2e-clean-canary-long-control-20260515c/opponents/current_assignment_pointer.json`.
- Config changes relative to the short canary: `max_train_iter=20000`,
  `max_env_step=1000000`, `save_ckpt_after_iter=100`, refresh interval still
  `25` train iterations.
- Next step: wait for numbered checkpoints, seed a live-watch tournament from
  this run id, promote the result to the long canary pointer, then verify
  refresh/env telemetry.

Long canary tournament/promotion:

- Live-watch tournament:
  `curvy-e2e-clean-canary-long-live-20260515c` /
  `elo-e2e-clean-canary-long-live-20260515c`.
- Seed discovered `5` checkpoints by run id, through
  `iteration_400.pth.tar`, with zero missing refs.
- Drain/rating completed `2` pairs / `6` games / `0` failed games and
  `stable=true`.
- Promotion wrote assignment sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30`
  from champion
  `training/lightzero-curvytron-visual-survival/curvy-e2e-clean-canary-long-20260515c/attempts/try-e2e-clean-canary-long-20260515c/train/lightzero_exp/ckpt/iteration_0.pth.tar`.
- Bug found: `scripts/promote_curvytron_rating_round.py` passed the literal
  `control:` prefix to `modal volume put`, which wrote the pointer under
  `control:training/...` instead of overwriting `training/...`. The running
  trainer correctly kept the starter sha after a refresh at train iter `1951`.
- Fix landed locally: promotion now strips `control:`/`runs:` before volume
  put. Ruff passed for the script.
- Manual repair for the live canary: uploaded the generated
  `refresh_pointer.json` to the correct control-volume path without the prefix.
  Next proof point is a refresh event applying sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30` and env
  rows using it.
- Full loop proof after repair: the same running trainer emitted
  `decision=applied` at train iter `2750`, `refresh_index=1`, with
  `env_ready_report.ok=true` across `8` envs and assignment sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30`.
- Later env telemetry already contains `1107` rows with that sha,
  `opponent_assignment_refresh_index=1`, and
  `opponent_provider_load_ok=true`.
- Pre-all-v2 deployed E2E proof passed for this historical canary path:
  trainer checkpoint -> live run-id intake -> tournament rating -> public
  leaderboard -> immutable assignment -> repaired control pointer -> same
  trainer refresh -> env use.
- Regression added: `tests/test_promote_curvytron_rating_round.py` now checks
  that prefixed pointer refs are stripped before `modal volume put`.
  Validation: `uv run pytest tests/test_promote_curvytron_rating_round.py -q`
  -> `8 passed`; ruff passed for the promotion script and test.

2026-05-15 12:35 EDT hardening update:

- Post-refresh canary continued well beyond the apply event: latest fetched
  progress before cleanup stop was `iteration_4500.pth.tar`, learner train iter
  `4564`, timestamp `2026-05-15T16:31:50Z`.
- Refresh events after the apply event stayed on sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30` through
  train iter `4565`.
- Broad local E2E-adjacent regression now passes:
  `385 passed, 14 skipped, 16 warnings`.
- A second live-log bug was found and patched locally: checkpoint eval poller
  command resolution now reloads the relevant Volume before reading `control:`
  or `runs:` assignment refs. This keeps background eval/GIF pollers from
  missing newly written control-volume assignment files. Local regression is
  included in `tests/test_curvytron_live_checkpoint_eval_plumbing.py`.
- Remote smoke after that patch found one more poller-only gap: the poller got
  past the `control:` assignment read, but failed when resolving the assignment's
  frozen checkpoint under `/runs`. The same-trainer refresh proof remains
  passed; background eval/GIF pollers still need either a nested checkpoint
  volume reload fix or an explicit disabled-for-launch decision.
- Local patch after the remote smoke adds that nested checkpoint reload only for
  poller/assignment resolution callers that opt into it. The running trainer's
  assignment refresh still avoids broad `/runs` reloads. Focused validation:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py -q`
  -> `85 passed, 3 skipped`; broad E2E-adjacent slice ->
  `386 passed, 14 skipped`; ruff passed for touched files. Remote redeploy and
  poller smoke are complete for the current namespace.
- Remote smoke correction: using the old non-v2 long-canary assignment under the
  current `curvyzero-runs-v2` deployment failed because the assignment points at
  checkpoints in `curvyzero-runs`. A current-namespace smoke then wrote control
  assignment `poller-v2-namespace-smoke-20260515a` pointing at v2 checkpoint
  `training/lightzero-curvytron-visual-survival/optimizer-gpuobs-canary-20260515/attempts/cpu-oracle-c1-sim2-steps64/train/lightzero_exp/ckpt/iteration_0.pth.tar`.
  Poller run `curvy-e2e-poller-v2-namespace-smoke-20260515a` completed with
  `status=completed`, `seen_count=0`, and no assignment/checkpoint resolution
  error. This proves the poller resolution path, not full current-namespace
  training feedback.
- Validation after that patch:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py
  tests/test_curvytron_survivaldiag_submitter.py
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -q`
  -> `131 passed, 3 skipped`;
  broad slice -> `385 passed, 14 skipped`;
  ruff passed for touched trainer, manifest, and test files.
- Cleanup: stopped old deployed training app
  `ap-OstFBxJk7XFsuzQKeJgAxE` after proof; redeployed patched trainer app as
  `ap-XpJMQzldf1KM1fIDiwqpZ1` with `0` tasks. Tournament service remains
  deployed as `ap-WHh65eiJOGazSsyKsq1021`.

## 2026-05-15 V2 Real18 Proof State

2026-05-15 09:26 EDT correction:

- The existing real18 feedback mechanics are partially proven, but the current
  67-ref rerate is not valid final tournament evidence because it used 20ms
  tournament source ticks against 16.6667ms trainer/checkpoint runtime.
- The corrected detached rerate for this historical diagnostic lane needed
  explicit 16.6667ms source timing and explicit
  `body_circles_fast + simple_symbols` policy observation surface. Fresh
  production proof steps should use CPU `cpu_oracle`
  `browser_lines + simple_symbols`.
- Do not use the wrong-tick rerate to update training opponents. Use it only as
  evidence that game workers are live.

| Link | Status | Evidence |
| --- | --- | --- |
| Trainer writes checkpoints | Proven for most original real18 rows | Status snapshot after launch showed `15/18` original rows still running; most had `iteration_10000`, several had `iteration_20000`. |
| Tournament rates historical v2real18 checkpoints | Diagnostic only | `curvy-v2real18-live-20260515a` / `elo-v2real18-live-20260515a`, `round-000001`, completed `231` pairs / `4,851` games / `0` failures. This is not current restart source evidence. |
| Rows are usable for training | Proven for this v2 round | Latest rating snapshot has `22` active rows and `0` provisional rows. Earlier `status=provisional` was caused by an impossible `placement_min_games=420` gate for a 17-player pool. |
| Coach materializes recipe assignments | Proven for this v2 round | Refreshed manifest `curvy-v2real18-refresh-r1-20260515a` wrote three control-volume assignments and three recipe refresh pointers without spawning duplicate rows. |
| Running trainers consume refreshed assignments | Partly proven for real18 | `run_status_after_refresh.json` showed `4/18` rows had `decision=applied` with one of the new assignment hashes: `9717c8...` or `e34871...`. Continue monitoring until all live rows either apply or fail clearly. |
| Failed rows diagnosed | Proven | Failed-row summaries show old deployed code used `td_steps=1048576` and died in LightZero replay sampling with `ValueError: 'a' and 'p' must have same size`. |
| Failed rows relaunched with fix | Launched, not yet proven | Rows `r008`, `r009`, `r011` relaunched from refreshed manifest under new run ids after deploying the `td_steps` fix. Need verify they start, keep stock `td_steps=5`, checkpoint, and join the tournament. |
| Survival improves | Not proven for real18 yet | Background eval summaries are still empty for these rows. Checkpoint production and tournament feedback are working, but no clean survival curve exists yet. |

2026-05-15 07:20 EDT update:

| Link | Status | Evidence |
| --- | --- | --- |
| Trainer writes checkpoints | Proven for most tracked rows | Current status over 21 tracked rows shows `90` durable checkpoint files, with max latest checkpoint `iteration_60000`. |
| Running trainers consume refreshed assignments | Stronger partial proof | `15/21` rows have `assignment_refresh_applied_count > 0`, split evenly across the three recipe assignment shas. |
| Tournament rates historical v2real18 checkpoints | Running diagnostic, not launch evidence | Fresh clean all-pairs rerate `elo-v2real18-rerate67-allpairs-20260515a` is running; logs show successful games with `max_steps=1048576`, but `latest.json` is not written yet. Do not publish/materialize restart assignments from it. |
| Survival improves | Still not proven | `eval_manifest_count=0` for the tracked real18 rows; background eval completions total only `7`. |

2026-05-15 update:

| Link | Status | Evidence from this pass |
| --- | --- | --- |
| Tournament observation contract | Locally tested | `uv run pytest tests/test_curvytron_checkpoint_tournament.py -q` -> `136 passed, 11 skipped`; added active-bonus parity and 51-row scheduler guard. |
| 51-row tournament explanation | Explained locally | Current completed `latest.json` remains old; synthetic guard confirms `193 * 192 / 2 = 18,528` all-pairs and `389,088` games, while bounded `adaptive_v0` gives `300` pairs / `6,300` games. |
| Current survival quantification | Updated | Read-only Modal eval-summary over 18 submitted `curvy-v2refresh18p-*` run ids completed 2026-05-15; weak rows and mixes are recorded in `TRAINING_CONTROL.md`. |
| Weak-run assignment intervention | Blocked | Current mixes are known, but live pointer mutation needs the exact assignment-writer/audit command path for the shared v2 refresh control pointer. |

2026-05-15 consume-smoke update:

| Link | Status | Evidence from this pass |
| --- | --- | --- |
| Clean rating to public snapshot | Proven for bounded first round | `elo-loop18-live-main-adaptive417-20260515b` completed `300` pairs / `6,300` games with `zero_failed_games=true`; `stable=false` only means not final convergence. |
| Public snapshot to assignment | Proven | Assignment `loop18-main-adaptive417-r0-assignment-20260515b` written under `loop18-main-assignment-bank-20260515b`. |
| Assignment consumed by trainer launch | Proven | Smoke `loop18-main-adaptive417-consume-smoke-20260515e` reports `smoke_passed=true`, `smoke_artifacts_verified=true`, `provider_ok_row_count=359`, `env_step_row_count=359`. |
| Tournament winner used as initial checkpoint | Proven | Same smoke loaded the rank-1 checkpoint `iteration_240000.pth.tar`, matched `171/175` initial tensors where the fresh model shape differed, then loaded full prepared model state during LightZero's normal load path. |
| Fresh optimizer preserved | Proven | Same smoke reports `fresh_optimizer_preserved=true`; optimizer load was skipped for the marked model-only seed checkpoint. |
| Continuous live feedback loop | Proven on small deployed loop | `controlrun2` followed a fresh trainer checkpoint through intake, tournament, promotion, pointer update, and same-trainer refresh. |
| V2 storage live refresh | Proven on small deployed loop | `curvy-v2-looplive-proof3-20260515a` wrote v2 checkpoints; v2 intake-spawned rating `elo-v2-looplive-proof3-r0-20260515a` completed; direct v2 rating also completed; promotion wrote sha `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770`; the same running v2 trainer applied it at train iter `1904` and later env rows used it with provider load OK. |
| Survival after current connected batch | Mixed | 2026-05-15 read-only eval-summary over 18 `curvy-n18conn-*` rows: all-row first/latest/best means are `132.5` / `141.7` / `182.3`; latest is up in `10/18` rows. V2 canary post-refresh return rose `118.24 -> 134.72`, but mean length fell `159.20 -> 144.96` on only `50` old and `25` new terminal samples. |
| Existing 18-run automatic refresh | Not enabled | Manifest inspection should show refresh settings | `curvy-night18-connected-20260514d` rows have fixed `opponent_assignment_ref` only; `initial_policy_checkpoint_ref`, `opponent_assignment_refresh_interval_train_iter`, and `opponent_assignment_refresh_ref` are all null/missing. |
| Trainer refresh pointer support | Proven remotely | Trainer can read mutable pointer file that names immutable assignment + sha | Historical non-v2 proof passed, and recreated all-v2 canary proved same-process refresh with provider-ok env rows. |
| Refresh pointer Volume visibility | Locally proven and deployed | Running trainer reloads the Volume before reading pointer file | Focused tests now pass (`79 passed, 3 skipped`); patch avoids stale pointer reads inside already-running trainers. |

2026-05-15 live fast proof update:

| Link | Status | Evidence from this pass |
| --- | --- | --- |
| Trainer writes checkpoints | Proven for fast proof | `curvy-looplive-proof-fast-20260515a` wrote numbered checkpoints at least through `iteration_930.pth.tar`. |
| Subscriber/intake sees checkpoints | Proven for fast proof | Intake manifest `curvy-looplive-fast-proof-20260515a` grew to `129` refs with queue length `0`. |
| Tournament rates checkpoints | Proven for fast proof | `elo-looplive-fast-proof-20260515a` produced `stable=true` latest round over `60` checkpoints; top row was fast-run `iteration_10.pth.tar`. |
| Public leaderboard and assignment | Proven for fast proof | Published `looplive-fast-proof-r1-20260515a`, materialized `looplive-fast-proof-r1-assignment-20260515a`, and wrote refresh pointer sha `b8f66ff55c00002f5de626f68b9e5515e487ff756c7f262fca3b36b7835f6d0c`. |
| Running trainer consumes refreshed assignment | Failed, patching | Refresh events stayed `kept_previous` because `runs_volume.reload()` failed with `cwd is inside volume`. The next patch must move cwd outside `/runs` for the reload, then rerun this proof. |

2026-05-15 follow-up:

- The cwd-only fix was insufficient. A redeployed fast proof
  `curvy-looplive-proof-fast-20260515b` failed refresh because Modal refused to
  reload `/runs` while LightZero had an event file open under
  `train/lightzero_exp/log/`.
- Corrected design: mutable assignment pointer and assignment files move to the
  separate `/control` Volume and use explicit `control:` refs. Training
  checkpoints/logs remain on `/runs`.
- Local focused tests for this patch pass with `83 passed, 3 skipped`.
- Remote full-loop proof must be rerun after redeploy using a `control:...`
  refresh pointer and a control-volume assignment write.

2026-05-15 control-volume proof in progress:

| Link | Status | Evidence from this pass |
| --- | --- | --- |
| Trainer launch with control pointer | Running | `curvy-looplive-proof-controlfast-20260515c` / `try-looplive-proof-controlfast-20260515c`, function call `fc-01KRNCJ74TH3AKEE7AAYXYA993`, refresh pointer `control:training/lightzero-curvytron-visual-survival/looplive-proof-control-20260515c/attempts/try-looplive-proof-control-20260515c/opponents/current_assignment_pointer.json`. |
| Trainer writes checkpoints | Proven for this proof | The run has written numbered checkpoints at least through `iteration_635.pth.tar`; seed saw refs through `iteration_565.pth.tar`. |
| Intake sees checkpoints | Proven for this proof | `intake-seed --detach` app `ap-nv50D3kdiYT0JLBaBIpQmy` accepted the champion anchor plus live run refs. `intake-status` showed `checkpoint_count=129` and `queue_len=0`. |
| Tournament rates checkpoints | Running | Rating call `fc-01KRNCT4GNV0493G5YSM7B9ZX8`; progress exists under `tournaments/curvytron/curvy-looplive-controlfast-proof-20260515c/ratings/elo-looplive-controlfast-proof-20260515c/progress.json`. First poll: `status=running`, `pair_count=10`, `completed_pair_count=0`. |
| Tournament rates checkpoints | Proven for this proof | Rating call `fc-01KRNCT4GNV0493G5YSM7B9ZX8`; round `round-000000` completed `10` pairs / `210` games / `0` failures over `115` rows. |
| Control assignment promotion | Proven for this proof | Promotion wrote `control:training/lightzero-curvytron-visual-survival/looplive-controlfast-assignment-bank-20260515c/attempts/try-looplive-controlfast-assignment-bank-20260515c/opponents/assignments/looplive-controlfast-proof-r0-assignment-20260515c/assignment.json`, sha `4fbc8ef9d621ed5848a474d63f0cec900d6a97dd7827b5c1e81e17de0e12d462`, and rewrote the `/control` refresh pointer. |
| Running trainer applies refresh | Proven for this proof | `opponent_assignment_refresh_events.jsonl` shows `decision=applied` at train iter `1130`, `refresh_index=1`, `env_ready_report.ok=true`, and the new sha `4fbc8ef9d621ed5848a474d63f0cec900d6a97dd7827b5c1e81e17de0e12d462` instead of the initial adaptive417 sha `b333431382eb8f5f846c445da4d96af6a7bac190f3471de661a53fbed455cf71`. |
| Trainer uses refreshed assignment | Proven for this proof | Later `env_steps.jsonl` rows contain the same sha with `opponent_assignment_refresh_index=1`; sampled rows report `opponent_provider_load_ok=true`. |

Follow-up behavior proof:

- A longer proof run was launched as
  `curvy-looplive-proof-controllong-20260515d` /
  `try-looplive-proof-controllong-20260515d`, function call
  `fc-01KRND94WW84K58S2AWHRCEJPM`, starting from proof champion
  `iteration_135.pth.tar`.
- It uses the control assignment sha
  `4fbc8ef9d621ed5848a474d63f0cec900d6a97dd7827b5c1e81e17de0e12d462` and
  refresh pointer
  `control:training/lightzero-curvytron-visual-survival/looplive-proof-control-20260515d/attempts/try-looplive-proof-control-20260515d/opponents/current_assignment_pointer.json`.
- Caveat: this launch shows `commit_on_checkpoint=false`, so it may not expose
  new checkpoints to tournament intake during the run. If visible checkpoints
  do not advance beyond startup, relaunch a corrected behavior proof with
  `commit_on_checkpoint=true` and a modest checkpoint cadence.
- The caveat was checked: visible checkpoints did advance through at least
  `iteration_700.pth.tar`, so tournament intake can see the long proof.
- Long proof intake:
  `curvy-looplive-controllong-proof-20260515d` /
  `elo-looplive-controllong-proof-20260515d` accepted `9` refs, including the
  controlfast champion anchor plus controllong iterations `0,100,...,700`, and
  spawned rating call `fc-01KRNDHPX02DSBMF7BXHPQ4MEA`.
- That first long-proof rating stalled at `game_map_started` with zero started
  pairs because a rating claim existed around half-written round artifacts.
  The replacement rating id is
  `elo-looplive-controllong-proof-fresh-20260515e`; it accepted `19` refs
  through controllong `iteration_1700.pth.tar` and spawned rating call
  `fc-01KRNDRVRRKS8CNG9TVQYZGXBH`.
- The replacement rating initially looked stalled at `game_map_started` with
  zero started pairs, but that was a stale progress snapshot. Fresh progress
  shows `elo-looplive-controllong-proof-fresh-20260515e` completed `10/10`
  pairs and `210/210` games with `0` failures. The first small controlfast
  proof remains the completed end-to-end loop proof.
- Modal app list at `2026-05-15 05:09 EDT` showed the longer trainer app
  `ap-ciAzi7ByfRueLxZLtqxuEf` stopped at `2026-05-15 05:05:35 EDT`. This means
  the long behavior proof cannot produce more refresh evidence unless relaunched.
  Relaunch only after the tournament worker stall has a clear diagnosis.
- Direct rating separator:
  `curvy-looplive-directrating-smoke-20260515a` /
  `elo-looplive-directrating-smoke-20260515a` rated two checkpoints from
  `curvy-looplive-proof-controllong-20260515d` (`iteration_0` and
  `iteration_3000`) with `3` games, `0` failed games, and `stable=true`. Direct
  rating/game workers are healthy.
- Intake separator:
  `curvy-looplive-intake-smoke-20260515a` /
  `elo-looplive-intake-smoke-20260515a` completed through
  `intake-seed`/`intake-drain`, `1/1` pair, `3/3` games, `stable=true`.
  Intake is not generally dead.

Namespace caveat:

- The first clean adaptive417/controlrun2 proof artifacts are in the non-v2
  storage lane. That still matters when reading old evidence.
- V2 storage now has its own small refresh proof:
  `curvy-v2-looplive-proof3-20260515a` -> v2 intake-spawned rating
  `elo-v2-looplive-proof3-r0-20260515a` completed `1` pair / `3` games / `0`
  failures -> direct v2 rating
  `elo-v2-looplive-proof3-direct-r0-20260515a` also completed -> promoted v2
  assignment sha
  `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770` ->
  same running v2 trainer applied at train iter `1904`.

## What Does Not Count

- A queued checkpoint does not prove tournament rating.
- A round input file does not prove games completed.
- A public pointer does not prove assignment consumption.
- A static assignment batch does not prove running refresh.
- A tournament winner as opponent does not prove champion start weights.
- Monitoring a manifest that did not enable a link does not prove that link.

## Current Blockers

1. Survival improvement still needs quantified follow-up after refreshed
   opponents are used for a meaningful window.
2. The current proof roster was intentionally tiny. It proves wiring, not
   tournament ranking quality at production scale.
3. Production-quality promotion is still open. The canary deliberately allowed
   provisional rows and used relaxed active-row thresholds.
4. A learned nonzero checkpoint as champion is not proven by this canary:
   `iteration_0.pth.tar` was the champion, while nonzero checkpoints appeared in
   non-champion slots.
5. Background eval/GIF poller behavior is separate. The canary proves the
   training refresh path, not the broad eval/GIF path.

## Current Remote Proof Plan

The recreated all-v2 deployed proof has passed. Do not repeat the same tiny
wiring proof unless storage namespace, assignment format, refresh mechanics, or
Modal VolumeFS handling changes.

For the next real run, carry forward these requirements:

- use only all-v2 Volumes, Dicts, Queue, and app names from
  `src/curvyzero/contracts/curvytron.py`;
- audit the exact launch manifest for stale non-v2 checkpoint or assignment
  refs before launch;
- start from the selected tournament winner only if that checkpoint exists in
  `curvyzero-runs-v2`;
- start with an immutable control-volume assignment and a mutable `control:`
  refresh pointer;
- checkpoint frequently enough for subscriber intake to see new files quickly;
- keep the run alive long enough for a new checkpoint to be rated, promoted,
  written as a new assignment, and picked up by the trainer refresh hook;
- after launch, prove the same gates again at larger scale: `decision=applied`
  plus env telemetry rows with the new assignment sha and
  `opponent_provider_load_ok=true`.

Historical remote-proof notes below explain how the earlier non-v2 and pre-reset
v2 proofs led to the current all-v2 lane.

2026-05-15 controlrun2 behavior proof:

- Trainer spawned:
  `curvy-looplive-proof-controlrun2-20260515f` /
  `try-looplive-proof-controlrun2-20260515f`, function call
  `fc-01KRNF20A3YCKANZEH8PV0G33F`.
- It uses the deployed trainer app, explicit non-v2 proof storage, control
  assignment sha `4fbc8ef9d621ed5848a474d63f0cec900d6a97dd7827b5c1e81e17de0e12d462`,
  and refresh pointer
  `control:training/lightzero-curvytron-visual-survival/looplive-proof-control-20260515f/attempts/try-looplive-proof-control-20260515f/opponents/current_assignment_pointer.json`.
- Settings chosen to keep the loop observable:
  `commit_on_checkpoint=true`, `save_ckpt_after_iter=50`,
  `opponent_assignment_refresh_interval_train_iter=25`, `max_train_iter=8000`.
- Pass condition: a fresh `controlrun2` checkpoint gets rated, promoted to a
  new assignment, the pointer above is rewritten, and this same running trainer
  logs `decision=applied` for the new assignment sha.
- Result: passed at 2026-05-15 05:30 EDT.
  - Fresh checkpoint:
    `training/lightzero-curvytron-visual-survival/curvy-looplive-proof-controlrun2-20260515f/attempts/try-looplive-proof-controlrun2-20260515f/train/lightzero_exp/ckpt/iteration_400.pth.tar`.
  - Intake/tournament:
    `curvy-looplive-controlrun2-proof-20260515f` /
    `elo-looplive-controlrun2-proof-r0-20260515f`, `round-000000`,
    `1` pair / `3` games / `0` failures, `stable=true`.
  - Promotion:
    assignment
    `control:training/lightzero-curvytron-visual-survival/looplive-controlrun2-assignment-bank-20260515f/attempts/try-looplive-controlrun2-assignment-bank-20260515f/opponents/assignments/looplive-controlrun2-proof-r0-assignment-20260515f/assignment.json`,
    sha `3ff1af447117e4e90cd1e82277530063d20ba14086d180df5474e7d5309dfa9d`.
  - Running trainer refresh:
    `opponent_assignment_refresh_events.jsonl` shows `decision=applied` at
    train iter `1798`, `refresh_index=1`, `env_ready_report.ok=true`.
  - Trainer use:
    later `env_steps.jsonl` rows contain the new sha with
    `opponent_assignment_refresh_index=1`, `opponent_provider_load_ok=true`,
    and slots selecting both `slot_recent_strong` (`controlrun2` iteration
    `400`) and `slot_champion` (controlfast iteration `135`).
  - First behavior read after refresh:
    `315` env telemetry rows, `67` terminal rows total. Old assignment sha
    `4fbc8...`: `49` terminal rows, mean return `162.92`, median return `206`,
    win/loss `29/20`. New assignment sha `3ff1...`: `18` terminal rows, mean
    return `212.44`, median return `270`, win/loss `13/5`. This is encouraging
    but still a small canary window, not a final survival-improvement claim.

2026-05-15 05:31 EDT update: this pass condition is satisfied. Do not rerun
the same tiny proof unless changing storage namespace, assignment format, or
refresh mechanics. Next evidence target is survival/progress after the refresh.

2026-05-15 v2 storage proof:

- Trainer spawned:
  `curvy-v2-looplive-proof3-20260515a` /
  `try-v2-looplive-proof3-20260515a`, function call
  `fc-01KRNGBHYW3S0J3EJXN17H6164`.
- Storage names were the v2 names:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`,
  `curvyzero-curvytron-control-v2`.
- Direct v2 rating:
  `curvy-v2-looplive-proof3-direct-20260515a` /
  `elo-v2-looplive-proof3-direct-r0-20260515a` rated the champion anchor
  against proof checkpoint `iteration_300`, completed `1` pair / `3` games /
  `0` failures, and wrote ratings.
- Promotion:
  assignment
  `control:training/lightzero-curvytron-visual-survival/v2-looplive-proof3-assignment-bank-20260515a/attempts/try-v2-looplive-proof3-assignment-bank-20260515a/opponents/assignments/v2-looplive-proof3-direct-r0-assignment-20260515a/assignment.json`,
  sha `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770`.
- Running trainer refresh:
  `opponent_assignment_refresh_events.jsonl` shows `decision=applied` at train
  iter `1904`, `refresh_index=1`, `env_ready_report.ok=true`.
- Trainer use:
  later `env_steps.jsonl` rows contain the new sha with
  `opponent_assignment_refresh_index=1`, `opponent_provider_load_ok=true`, and
  slots selecting both the champion anchor and the fresh proof checkpoint
  `iteration_300`.
- Intake recheck:
  the original v2 intake-spawned rating
  `elo-v2-looplive-proof3-r0-20260515a` later showed `status=complete`,
  `1` pair / `3` games / `0` failures, `ratings_written=true`, `stable=true`.
  The earlier stuck read was stale progress, not a live blocker.
- First behavior read:
  old assignment sha `d881...` had `50` terminal samples, mean return `118.24`,
  mean length `159.20`; refreshed sha `adb04...` had `25` terminal samples,
  mean return `134.72`, mean length `144.96`. This is useful telemetry but not
  enough to claim survival is improving.

## Minimum Honest Proof After Fixes

1. Use a clean tournament rating over a sane roster.
2. Verify completed games and zero/accepted failures.
3. Publish immutable public leaderboard snapshot.
4. Materialize immutable assignment.
5. Rewrite the intended control pointer.
6. Show the same running trainer logs `decision=applied` for that assignment at
   a clean refresh boundary.
7. Verify later env telemetry shows:
   - assignment sha;
   - opponent checkpoint refs;
   - `opponent_provider_load_ok=true`;
   - correct slot probabilities.
8. Separately quantify survival before/after for the affected run or smoke.

## Parallel Fallback Rule

If the large current tournament is blocked, run a smaller honest proof in
parallel:

- 5-10 checkpoints;
- same one-frame timing;
- same CPU `cpu_oracle` `browser_lines + simple_symbols` production
  observation surface;
- same assignment materializer;
- one tiny trainer consuming the result.

This is allowed because it tests the same contract faster.
