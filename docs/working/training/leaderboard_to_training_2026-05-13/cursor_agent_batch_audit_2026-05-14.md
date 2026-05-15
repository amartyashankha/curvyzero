# Cursor Agent Batch Audit - 2026-05-14

Scope: read-only audit of recently launched CurvyTron training batches and
checkpoint tournament lanes after the leaderboard-to-training work. This audit
only edited this document.

Last checked: 2026-05-14 about 12:02 EDT, using local docs/artifacts plus
read-only Modal status/Volume reads.

## Current Answer In Plain Language

The full loop is partly set up, but do not call it production-validated for the
18-row batches yet.

What is working at production-ish scale:

- `curvy-n18tdfix` proves fixed training rows can emit checkpoints, the intake
  subscriber can discover those checkpoints, a clean tournament can rate them,
  and a public leaderboard snapshot can be published from that rating.
- `curvy-night18-connected-20260514d` proves the next trainer launch shape can
  consume immutable `opponent_assignment_ref` files derived from that tdfix
  leaderboard. All 18 connected rows launched and reached `running`.

What is not yet proven end to end for these production batches:

- The connected rows have not yet produced enough nontrivial checkpoints to seed
  their own connected tournament.
- No connected tournament exists yet at
  `tournaments/curvytron/curvy-night18-connected-20260514d`.
- The loop has not yet gone all the way through connected training checkpoint ->
  connected intake/subscriber -> connected tournament/rating -> connected
  leaderboard/pointer -> next trainer assignment.
- Live in-run assignment refresh is not proven by these batches. The connected
  batch reads an assignment at launch; it is not evidence that running trainers
  refresh from new leaderboard state.

So the honest answer is: manual/tiny smokes and the tdfix production batch prove
large parts of the loop, and the connected batch is the first real
assignment-backed launch. The whole production loop is in progress, not done.

## Executive Summary

There are multiple overlapping "18-row" training generations. The currently
trusted generation is `curvy-night18-tdfix-20260514c` / run prefix
`curvy-n18tdfix`: all 18 rows were still `running` in the live run-status read,
with visible checkpoints between roughly `iteration_30000` and
`iteration_60000`.

A newer assignment-backed generation launched after the previous audit:
`curvy-night18-connected-20260514d` / run prefix `curvy-n18conn`. It has 18
running rows and uses `opponent_assignment_ref` files materialized from the
published `curvy-night18-tdfix` leaderboard. It is the latest and most
connected trainer launch, but it is still early: most rows are at startup or
`iteration_0`, no summary files are written yet, and no connected tournament has
been seeded.

Two older full 18-row generations are still live/noisy: `curvy-n18fb`
(`curvy-night18-top10fallback-fixed-20260514a`) has 13 running rows and 5 failed
rows as of the latest read; `curvy-n18new`
(`curvy-night18-fullfresh-20260514b`) has 14 running rows and 4 failed rows. The
four-row replay diagnostic lane (`curvy-n18diag`) was a pin-down tool, not a
replacement batch: only the failure-shaped rows were intended; the manifest has
18 rows but 14 are absent.

The trusted `tdfix` tournament lane is healthy and ahead of the older notes:
`curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c` completed
`round-000001` with 300/300 pairs, 6300/6300 games, 0 failed games, and
`latest.json` containing 32 active rows. Its intake manifest has advanced to 49
seen checkpoints and queue length 0. The older `top10r1` tournament lane is
still active but should be treated as dirty continuation evidence: `latest.json`
is at `round-000004` with 87 rows, while progress points at `round-000005`
running with 0 counted starts/games.

The batches do not directly trample Coach semantics: they use distinct run
prefixes and mostly static inline opponent mixtures. They can still trample
Coach's work operationally by consuming H100 capacity/inodes, confusing intake
with multiple live prefixes, and publishing from stale/dirty arenas if someone
uses the wrong tournament id.

## Batch Inventory

| Batch / id | Approx rows | Launch/source | Trainer/tournament target | Current status | Known failures |
| --- | ---: | --- | --- | --- | --- |
| `curvy-night18-top10r1-20260514a`, prefix `curvy-night18top10r1` | 18 intended plus replacements | Generated `2026-05-14T08:31Z`; canaries first, then remaining rows; manifest under `artifacts/local/curvytron_tonight18_manifests/curvy-night18-top10r1-20260514a/` | Deployed trainer app; intake/tournament `curvy-night18-top10r1-20260514a / elo-night18-top10r1-20260514a` | Historical/noisy. Earlier checks saw only 10/18 with visible artifacts; old intake still active. | Five initial rows (`r004`, `r009`, `r010`, `r015`, `r016`) had overlong `run_id`/`attempt_id` and were replaced. Later continuation artifacts are dirty. |
| `curvy-night18-top10fallback-fixed-20260514a`, prefix `curvy-n18fb` | 18 | Generated `2026-05-14T09:36Z`; remote attempt starts about `09:38Z` | Deployed trainer app; no clean dedicated final intake initially | Live but pre-fix/noisy: 13 running, 5 failed in latest status read. Running rows have reached about `iteration_150000` to `iteration_270000`. | Replay priority/count mismatch. Docs originally recorded 4 failures; latest status shows 5 failed rows: `r003`, `r004`, `r011`, `r013`, `r017`. |
| `curvy-night18-replaydiag-20260514a`, prefix `curvy-n18diag` | 4 intended from an 18-row manifest | Generated `2026-05-14T14:31Z`; launched only failure-shaped rows `r003`, `r004`, `r011`, `r017` | Deployed trainer app; diagnostic only, `max_train_iter=80000` | Latest read: 14 absent, 3 running, 1 failed. | `r017` reproduced the replay invariant and exposed the exact mismatch. |
| `curvy-night18-fullfresh-20260514b`, prefix `curvy-n18new` | 18 | Generated `2026-05-14T14:36Z`; submitted through deployed app | Deployed trainer app; pre-tdfix full relaunch | Live but superseded: 14 running, 4 failed. Running rows mostly at `iteration_30000` to `iteration_50000`; one row had no latest checkpoint in the status read. | Four failures by latest read: `r005`, `r006`, `r015`, `r017`. This was launched before the final `td_steps` fix, so treat as pre-fix/noisy. |
| `curvy-night18-tdfix-20260514c`, prefix `curvy-n18tdfix` | 18 | Generated `2026-05-14T14:42Z`; submitted about 10:43 EDT; submission record `artifacts/local/curvytron_tonight18_manifests/curvy-night18-tdfix-20260514c/submission.json` | Deployed trainer app; clean intake/rating `curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c` | Trusted live batch: 18 running, visible checkpoints about `iteration_30000` to `iteration_60000`; no replay crash in latest read. | None seen in latest status read. |
| `curvy-night18-connected-20260514d`, prefix `curvy-n18conn` | 18 | Generated `2026-05-14T15:54Z`; submitted around 11:56 EDT; submission record `artifacts/local/curvytron_tonight18_manifests/curvy-night18-connected-20260514d/submission.json` | Deployed trainer app with assignment refs from the tdfix leaderboard; planned tournament `curvy-night18-connected-20260514d / elo-night18-connected-20260514d` does not exist yet | Latest assignment-backed batch: 18 running, no failed rows in status read, mostly startup through `iteration_0` and a few learner iters. | None seen yet. Too early: no `summary.json`; no connected tournament seeded. |
| `curvy-oneframe-top100-gate-20260514a / elo-oneframe-top100-gate-20260514a` | 100 checkpoint players | Detached practical top100 gate, launched around 03:14 EDT | Tournament/rating gate used for H100 manifest source | Complete: 1000/1000 pairs, 21000/21000 games, 0 failures, `latest.json` exists. | Not all rows active: 62 active, 38 provisional. Good health evidence, not a reason to publish all rows. |
| `curvy-night18-top10r1-20260514a / elo-night18-top10r1-20260514a` | Grew from 10/11 to 87+ rows | Old top10r1 intake/rating lane | Tournament/rating and published fallback leaderboard source | Dirty active lane: latest round 4 has 87 rows, 63 active; progress is on round 5 with 300 pairs / 6300 games and 0 counted starts. | Earlier round artifact rewrites and overlapping claims; do not use as clean continuation evidence without a fresh consistency check. |
| `curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c` | Started at 18, now 49 seen checkpoints, latest rating has 32 rows | Clean intake seeded at 11:30 EDT for prefix `curvy-n18tdfix` | Tournament/rating for trusted `tdfix` training batch | Healthy: round 1 complete, 300/300 pairs, 6300/6300 games, 0 failures, 32 active rows, queue len 0. | No failed games. Still needs publish decision and consistency check before promotion. |

## Tournament Mapping

| Training source | Tournament/rating target | Same or separate? | Notes |
| --- | --- | --- | --- |
| `curvy-night18-top10r1-20260514a` | `curvy-night18-top10r1-20260514a / elo-night18-top10r1-20260514a` | Same named lane | Old fallback intake prefix was `curvy-night18top10r1`. It does not discover `curvy-n18tdfix`. |
| `curvy-n18fb` fixed fallback batch | Initially no clean separate current tournament | Separate from tdfix | Its checkpoints may be visible to broad scans, but they are pre-fix/noisy and should not be mixed into the trusted tdfix rating. |
| `curvy-n18diag` diagnostic rows | No production tournament target | Diagnostic only | Purpose was reproducing replay-buffer invariant, not rating. |
| `curvy-n18new` fullfresh batch | No trusted tournament target found in current docs | Separate/noisy | Submitted before the final td-window fix and now has 4 failures. Do not use for clean leaderboard progression. |
| `curvy-n18tdfix` trusted batch | `curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c` | Clean separate lane | This is the right mapping for current fixed training checkpoints. Intake status shows prefix `curvy-n18tdfix`, 18 base checkpoint refs, 49 seen refs, queue 0. |
| `curvy-n18conn` connected batch | Planned `curvy-night18-connected-20260514d / elo-night18-connected-20260514d` | Not created yet | The batch consumes assignments from the tdfix leaderboard. It is not yet running against its own tournament; `modal volume ls` for that tournament path returned missing. |
| Top100 gate | `curvy-oneframe-top100-gate-20260514a / elo-oneframe-top100-gate-20260514a` | Separate launch gate | Completed 100-player practical gate. It supplied active rows for manifest building but is not the same as the tdfix continuation tournament. |
| Older top100 stress | `arena-oneframe-top100-plus-latest-20260514a / elo-oneframe-top100-plus-latest-20260514a` | Separate stress lane | Completed a 100-player / 300-game stress proof, all provisional. Do not confuse with the 1000-pair top100 gate. |

## Failure And Bug-Fix Timeline

| Time / order | Symptom | Fix or state change | Affected later batches |
| --- | --- | --- | --- |
| Early top10r1 launch | Five rows had generated `run_id` / `attempt_id` longer than the 96-character trainer run-management limit. | `scripts/build_curvytron_tonight18_manifest.py` now truncates ids with hash suffix; `scripts/submit_curvytron_survivaldiag_manifest.py` refuses overlong selected rows. | Later manifests have safe ids, including `curvy-n18fb`, `curvy-n18new`, and `curvy-n18tdfix`. |
| Early top10r1 background eval | `_run_checkpoint_eval_and_inspect` did not accept `opponent_assignment_ref` passed by the poller. | Local code accepted the argument; targeted tests passed; trainer app redeployed around 04:40 EDT. | Later rows should not fail this background eval signature path. |
| Night18 tournament continuation | Detached/local `.remote()` child work could be canceled when the local caller disconnected; round input/progress existed with no child workers. | Tournament rating loop later changed round hop to `spawn().get()` and app was redeployed. | `tdfix` tournament round 0/1 completed; old top10r1 remains dirty evidence. |
| `curvy-n18fb` fixed batch | Replay crash: `ValueError: 'a' and 'p' must have same size` from LightZero replay sampling. | Diagnostic hooks added to capture replay invariants. | `curvy-n18diag` launched only the failure-shaped cells. |
| `curvy-n18diag` | Diagnostic `r017` captured `num_transitions=42014`, lookup length `42014`, priorities length `107155`. | Root cause pinned: trainer set `td_steps=source_max_steps` (`65536`), far larger than stock LightZero chunk window. Fix stopped overriding `policy.td_steps` and added config guard for `td_steps + num_unroll_steps > game_segment_length`. | `curvy-n18tdfix` is the first trusted full 18-row generation after this fix. |
| `curvy-n18new` fullfresh | Submitted before final td-window fix; latest read shows 4 failures. | Superseded by `curvy-n18tdfix`; not killed. | Treat as noisy/pre-fix. |
| Tdfix intake | Old active watch was still `curvy-night18top10r1`, so it would never discover `curvy-n18tdfix`. | Seeded clean `curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c` with prefix `curvy-n18tdfix`. | Tdfix checkpoints now feed the clean tdfix tournament, not the old top10r1 arena. |
| Connected assignment batch | `tdfix` was healthy but static, using inline `opponent_mixture_spec`. | Published tdfix leaderboard `curvy-night18-tdfix-20260514c-elo-night18-tdfix-20260514c`, snapshot `round1-active-20260514a`, wrote three assignment artifacts, and launched `curvy-n18conn` with `opponent_assignment_ref`. | This is the first real assignment-ref launch. It is not production-shaped under the current contract because it lacks champion bootstrap via `initial_policy_checkpoint_ref` and has not produced the next tournament/leaderboard cycle. |

## Progress Snapshot

| Area | Latest observed progress |
| --- | --- |
| `curvy-n18tdfix` training | 18/18 running. Latest visible checkpoint refs in status read ranged from about `iteration_30000` to `iteration_60000`; no `ValueError: 'a' and 'p' must have same size` observed. |
| `curvy-n18fb` training | 13 running, 5 failed. Running rows reached as high as `iteration_270000`; failed rows include `r003`, `r004`, `r011`, `r013`, `r017`. |
| `curvy-n18new` training | 14 running, 4 failed. Most running rows are around `iteration_30000` to `iteration_50000`; this generation is superseded. |
| `curvy-n18diag` diagnostic | 14 absent, 3 running, 1 failed. It should remain diagnostic only. |
| `curvy-n18conn` connected training | 18/18 running. Latest status read showed no failed rows, several rows still at train iter 0, and visible `iteration_0.pth.tar` on many rows. This is too early to expect `iteration_10000` checkpoints. |
| Tdfix intake | Active manifest key present, `checkpoint_count=18`, `seen_checkpoint_count=49`, `run_id_prefix=curvy-n18tdfix`, `queue_len=0`, updated `2026-05-14T15:52:37Z`. |
| Tdfix rating | `progress.json`: status `complete`, phase `reduced`, `round-000001`, 300 pairs, 6300 games, 0 failed games, `ratings_written=true`, reduced `2026-05-14T15:50:56Z`. `latest.json`: 32 rows, all active, one-frame, 21 games/pair, GIFs on. |
| Tdfix leaderboard | `tournaments/curvytron/leaderboards/curvy-night18-tdfix-20260514c-elo-night18-tdfix-20260514c/latest.json` exists. Snapshot `round1-active-20260514a` has 32 active rows and is the assignment source for `curvy-n18conn`. |
| Connected tournament | Not created yet. `modal volume ls` for `tournaments/curvytron/curvy-night18-connected-20260514d` returned missing. |
| Top100 gate rating | Complete: `round-000000`, 1000 pairs, 21000 games, 0 failures, 100 rows. Status counts: 62 active, 38 provisional. |
| Old top10r1 rating | Latest final snapshot: `round-000004`, 87 rows, 63 active, 24 provisional. Current progress: `round-000005`, status `running`, phase `game_map_started`, 300 pairs, 6300 games, but 0 counted starts/games in the cheap progress artifact. |

## Opponent Source Wiring

| Batch | Opponent source | Assignment refs? | Tournament-fed at launch? | Notes |
| --- | --- | --- | --- | --- |
| `curvy-night18-top10r1-20260514a` | Static inline `opponent_mixture_spec` built from top10 fallback rows | No | Indirectly: frozen checkpoint refs came from a tournament snapshot, but the trainer does not read leaderboard state | Useful as first plumbing, not full assignment path. |
| `curvy-night18-top10fallback-fixed-20260514a` / `curvy-n18fb` | Static inline `opponent_mixture_spec` | No | Indirectly only | Meaningful for replay failure evidence; not connected assignment evidence. |
| `curvy-night18-replaydiag-20260514a` / `curvy-n18diag` | Same shape as fixed batch, diagnostic subset | No | No production tournament target | Diagnostic only. |
| `curvy-night18-fullfresh-20260514b` / `curvy-n18new` | Static inline `opponent_mixture_spec` | No | Indirectly only | Superseded pre-fix generation. |
| `curvy-night18-tdfix-20260514c` / `curvy-n18tdfix` | Static inline `opponent_mixture_spec` | No | Its checkpoints feed a tournament after launch | This is the clean source for tdfix rating/leaderboard, but it did not itself consume assignment refs. |
| `curvy-night18-connected-20260514d` / `curvy-n18conn` | Immutable assignment files from tdfix leaderboard `round1-active-20260514a` | Yes, 18/18 manifest rows | Yes at launch: assignment files are derived from tdfix tournament/leaderboard output | This is the first real assignment-ref batch. It has not yet fed its own tournament. |

## Original / First Batch Read

The original `curvy-night18-top10r1-20260514a` batch was partially meaningful,
but broken/noisy as production evidence.

Meaningful parts:

- It proved the deployed trainer spawn path could launch canaries and full rows.
- It wrote real training progress and checkpoints on some rows.
- Its old top10r1 tournament produced publishable fallback leaderboard evidence
  at least once.
- It exposed important launch-management bugs early.

Broken or noisy parts:

- Five first submissions had overlong run/attempt ids and should be counted as
  failed launch attempts, not real training cells.
- Background eval had an argument signature bug.
- The old top10r1 intake/tournament later showed dirty continuation artifacts,
  overlapping claims, and progress pointing at later rounds while older latest
  snapshots existed.
- It used static inline opponents, not the assignment-ref path.

Conclusion: keep it as a plumbing artifact and bug-finding record. Do not treat
it as clean end-to-end proof or as the current source for Coach decisions.

## Per-Batch Next Signals

| Batch | Launched? | Running? | Failed count | Checkpoint signal | Tournament target | Subscriber/intake target | Assignment source | Expected next signal |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `top10r1` | Yes | Historical/dirty | At least 5 launch failures | Some real checkpoints and many later tournament rows | `curvy-night18-top10r1-20260514a / elo-night18-top10r1-20260514a` | `run_id_prefix=curvy-night18top10r1` | None; static manifest opponents | Only review for history unless debugging old artifact corruption. |
| `curvy-n18fb` | Yes | 13 running in latest read | 5 | Up to about `iteration_270000` | None clean/current | None clean/current | None; static manifest opponents | Decide whether to stop/ignore; do not feed clean leaderboard. |
| `curvy-n18diag` | Four rows intended | 3 running, 1 failed in latest read | 1 | Diagnostic rows only | None | None | None | No production next signal; it already found the replay invariant. |
| `curvy-n18new` | Yes | 14 running in latest read | 4 | Up to about `iteration_50000` | None trusted | None trusted | None; static manifest opponents | Decide whether to stop/ignore; superseded by tdfix. |
| `curvy-n18tdfix` | Yes | 18 running in latest read | 0 | About `iteration_30000` to `iteration_60000` at last read | `curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c` | `run_id_prefix=curvy-n18tdfix`, 49 seen checkpoints | None; static manifest opponents | Continue monitoring and preserve as clean source leaderboard. |
| `curvy-n18conn` | Yes | 18 running in latest read | 0 | Mostly startup/`iteration_0`; no `summary.json` yet | Planned `curvy-night18-connected-20260514d / elo-night18-connected-20260514d`; missing now | Not seeded yet; planned `run_id_prefix=curvy-n18conn` | Tdfix leaderboard `round1-active-20260514a` assignment refs | Wait for at least `iteration_10000` checkpoints, then seed connected intake/tournament. |

## What To Review Next

Concrete read-only checks:

1. Confirm connected trainer assignment consumption from local manifest:
   `python - <<'PY' ...` over
   `artifacts/local/curvytron_tonight18_manifests/curvy-night18-connected-20260514d/curvy-night18-connected-20260514d.json`
   and assert 18 `train_kwargs.opponent_assignment_ref` values and 0
   `opponent_mixture_spec` values.
2. Inspect connected run progress after a real checkpoint interval:
   read `latest_attempt.json` and `progress_latest.json` under
   `training/lightzero-curvytron-visual-survival/curvy-n18conn-*/...` on the
   runs Volume. Look for 18 running rows, `iteration_10000.pth.tar`, and no
   `summary_ok=false`.
3. Check whether connected tournament exists before anyone assumes it is wired:
   `uv run --extra modal modal volume ls --json curvyzero-curvytron-tournaments tournaments/curvytron/curvy-night18-connected-20260514d`.
4. Once connected checkpoints exist, run an intake-status read after seeding
   (not before) for `curvy-night18-connected-20260514d /
   elo-night18-connected-20260514d` and confirm `run_id_prefix=curvy-n18conn`.
5. Re-fetch tdfix leaderboard source if assignment source is questioned:
   `uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments tournaments/curvytron/leaderboards/curvy-night18-tdfix-20260514c-elo-night18-tdfix-20260514c/latest.json /private/tmp/audit-tdfix-leaderboard-latest.json`.
6. Compare connected assignment audits:
   `artifacts/local/curvytron_tonight18_manifests/curvy-night18-connected-20260514d/assignments/*/audit.json`
   and verify the source leaderboard id, snapshot id, and exact checkpoint refs.
7. If capacity becomes an issue, review old live batches before acting:
   `curvy-n18fb`, `curvy-n18new`, and `curvy-n18diag` are the candidates to
   ignore or stop, but this audit did not stop anything.

## Do Not Overclaim

- Do not claim the production loop is fully validated. The connected batch has
  not yet fed its own tournament or produced a refreshed leaderboard.
- Do not claim live in-run assignment refresh is proven by `curvy-n18conn`; it
  only proves assignment-ref launch consumption so far.
- Do not claim old `top10r1` is clean. It is useful history and bug evidence,
  but the continuation artifacts are dirty.
- Do not claim `curvy-n18fb` or `curvy-n18new` are valid comparison rows for
  learning quality. They are pre-fix/noisy and still have failures.
- Do not publish or consume connected tournament results before the connected
  tournament directory exists, writes progress/latest, and passes pool
  consistency checks.
- Do not infer that background eval/GIF is healthy for every connected row until
  those row artifacts are sampled.

## Risks And Caveats

- Capacity/inode risk is real. Modal warned the runs Volume was around 84-85% of
  available inodes, and several old/new 18-row generations are still running.
- The audit used status/Volume reads and local manifests; it did not stop,
  relaunch, purge, or publish anything.
- Some status reads are point-in-time and expensive. The slow all-batch run
  status read completed after several minutes; values can be stale by the time
  this doc is read.
- The old top10r1 tournament has live progress but known artifact-rewrite and
  overlapping-claim concerns. Do not use it as clean evidence without checking
  round input/results/latest pool consistency.
- `curvy-n18fb` and `curvy-n18new` are still live enough to consume resources
  and produce checkpoints. They should not be mixed into the trusted tdfix
  tournament unless someone explicitly wants noisy/pre-fix comparison data.
- Tdfix training uses static inline `opponent_mixture_spec`. It proves fixed
  launch-time leaderboard-derived opponents, not live public-leaderboard
  assignment refresh inside a running trainer.
- Coach boundary: the current architecture still says Coach materializes
  immutable assignments and decides promotion. Tournament publisher output must
  not silently mutate running trainers.

## Recommended Next Actions

Read-only review:

1. Re-read `curvy-night18-tdfix-20260514c / elo-night18-tdfix-20260514c`
   `latest.json` and confirm `ratings`, `results`, `progress`, and round inputs
   all agree on the 32-row pool before publishing.
2. Inspect current `curvy-n18tdfix` training status again after a long enough
   checkpoint interval; confirm all 18 remain running and no replay invariant
   error appears.
3. Check tdfix background eval/GIF artifacts for a sample of rows, especially
   rows at the low end of the checkpoint range.
4. Keep old top10r1 and top100 lanes labeled as historical/diagnostic unless a
   fresh consistency check says otherwise.

Possible future operations, not performed here:

1. Decide whether to stop or leave `curvy-n18fb`, `curvy-n18new`, and lingering
   diagnostic rows. This is an operator/Coach capacity decision, not a docs
   audit action.
2. Publish the tdfix leaderboard only after pool consistency and active-row
   checks pass.
3. If publishing succeeds, materialize a frozen `stable_slots_v1` assignment and
   run a tiny trainer smoke that consumes it. That is the missing proof for
   leaderboard-to-assignment consumption.
4. Plan Volume/inode cleanup with exact keep/delete sets. Do not use broad
   wildcard deletion while active tournament writers exist.
