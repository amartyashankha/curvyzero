# TODO: Coach / Tournament Task Board

Use this as the live checklist. Every task needs an owner, a status, and a next
action. Do not leave important state only in chat.

Current architecture entry point:
`CURRENT_TOURNAMENT_PIPELINE_2026-05-16.md`. Older proof-lane rows in this file
are historical unless they are explicitly re-listed there as current.

## P0

Current restart snapshot:

- Current cleanup state, 2026-05-16: `CURRENT_TOURNAMENT_PIPELINE_2026-05-16.md`
  is the current architecture entry point. Tournament CLI current-lane modes
  default to the shared current arena/rating ids when `--tournament-id` is
  omitted. `--mode current` prints the static current-lane automation contract
  without spawning Modal work. Older proof-lane rows remain audit history, not
  launch instructions.
- Current source-of-truth framing, 2026-05-16 11:35 EDT: the active tournament
  is the best-policy source of truth, while the public/trainer leaderboard is a
  controller export from that truth. The active arena has advanced to
  `round-000029` with `549` rated rows / `531` nonzero rows. The trainer-facing
  Dict export is generation `18` from `round-000027`
  (`auto-r000027-g18-f8a118b4`). This is normal lag if the controller only
  exports periodically. Do not confuse "tournament truth is maintained" with
  "the tournament has found an impressive champion"; the current champion set is
  still modest mid-run r18fresh checkpoints.
- Current trainer-export proof, 2026-05-16 11:40 EDT:
  `training-candidate-auto-refresh` refreshed from `round-000029` and rewrote
  all three training recipe pointers. The Modal Dict pointer then read as
  generation `21`, snapshot `auto-r000029-g21-65ecaffa`, published at
  `2026-05-16T15:39:55Z`. This confirms tournament -> trainer-facing export is
  alive; it does not prove the exported policies are strong.
- Tournament GIF visibility, 2026-05-16 11:47 EDT: active arena
  `curvy-r18fresh-live-bounded-dsf1-20260516b` has persisted no-GIF settings
  from its original launch: `save_gif=false` in both intake rating defaults and
  `round-000029/input.json`.
  `gif_sample_games_per_pair=5` and `gif_fps=800.0` are present but inert. Do
  not expect Tournament Arena GIF samples from this lane unless a future
  continuation is explicitly submitted with `save_gif=true`. Root cause is not
  source-default regression: code still has `DEFAULT_SAVE_GIF=True` and
  `DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR=5`. The current intake config was
  corrected for future rating work; old completed no-GIF rounds remain no-GIF.
  Add a preflight gate for future visual arenas that checks the actual intake
  config and latest round input.
- Own-latest control proof, 2026-05-16 11:43 EDT: selected3 lane
  `curvy-ownlatest-staticmix-20260516b` has one nonzero row so far. Row `r011`
  wrote `iteration_10000`, applied own-checkpoint assignment sha
  `23ebfe2d...` at train iter `10789`, and env-tail proof found
  `19,240/19,240` target rows loading that checkpoint with provider OK and
  `0` provider-load false rows. The other two rows are still before nonzero
  checkpoint and correctly kept previous assignment because their run-local
  own-latest pointer does not exist yet. The active tournament subscriber does
  not automatically include ownlatestb because its scan is scoped to `r18fresh`
  run IDs; if wanted, submit exact nonzero ownlatest refs later.
- Portfolio rule: this multi-run setup only needs some runs to work. Track
  top-ranked tournament checkpoints and best-so-far survival, not just latest
  survival per run. If the top tournament policies remain unimpressive, the next
  intervention should target learning stability/retention or a better control
  recipe, not another proof of the wiring.
- Survival-quality investigation, 2026-05-16: corrected aggregate plus fresher
  full-status pull says `18/18` rows improved at least once, but only `10/18`
  are improved at latest; latest mean survival is only `+15.5` steps over first
  while best-so-far is `+86.1`. Historical collapse flags appear in `13/18`
  ladders, but latest eval checkpoints are not currently collapsed by the 0.95
  top-action threshold. Current diagnosis target is late regression and weak
  retention, not total absence of learning. Active docs:
  `survival_stagnation_investigation_2026-05-16.md`,
  `survival_static_control_run_plan_2026-05-16.md`, and
  `survival_mechanics_reward_audit_2026-05-16.md`.
- Current active feedback lane, 2026-05-16 12:40 EDT:
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`.
  Generation 9 is proven end-to-end for the 18 running trainers:
  tournament `round-000002` -> training-candidate snapshot
  `r18fresh-dsf1-r2-training-g9-20260516b` -> three control refresh pointers
  -> same-running trainers -> env rows. Env-tail proof found all `18/18` runs
  using the gen-9 shas, `89,934` frozen-checkpoint provider-ok rows, and `0`
  provider-load false rows.
- Current long-running follow-up: generation 10 is also proven after a
  30-minute wait. It was published from `round-000006` as
  `r18fresh-dsf1-r6-training-g10-20260516b`; all `18/18` trainers had gen-10
  env rows, with `93,427` frozen-checkpoint provider-ok rows and `0`
  provider-load false rows.
- Automation follow-up: `curvytron_training_candidate_refresh_tick` is deployed
  on a 30-minute schedule in `curvyzero-checkpoint-tournament-v2`. Manual smoke
  of the same function published gen-11 from `round-000009`. After the full
  30-minute wait, the scheduled tick published gen-12 from `round-000011` as
  `auto-r000011-g12-65387f58`; shas are `899ed9c6...`, `1d9c86ff...`, and
  `2af7c6ff...`.
- Current tournament latest checked by live-state audit is `round-000023` with
  `511` rated rows, `493` nonzero-iteration rows, max checkpoint iteration
  `310893`, explicit `decision_source_frames=1`, `300` pairs, and `6,300`
  games. The active/provisional/retired split is `72/132/307`; this is coherent
  live standing data, not final-stable truth (`stable=false`,
  `max_abs_delta=15.46`).
- Current live proof status: `15/18` trainers are still running and `3/18`
  completed. Gen-12 proof over the still-running trainers found `15/15` with
  target-sha env rows, `146,976` target rows, `84,578` provider-ok
  frozen-checkpoint rows, and `0` provider-load false rows. This closes the
  active long-term loop proof. Remaining watch item: future fresh batches should
  launch long enough that scheduled refreshes happen comfortably before end of
  run.
- Previous bounded tournament, 2026-05-16 05:26 EDT:
  `curvy-r18fresh-live-bounded-20260516a` /
  `elo-r18fresh-live-bounded-20260516a`.
  URL:
  `https://modal-labs-shankha-dev--curvyzero-checkpoint-tournament--93d419.modal.run/?tournament_id=curvy-r18fresh-live-bounded-20260516a&rating_run_id=elo-r18fresh-live-bounded-20260516a`.
  It was seeded from the same 18 r18fresh run ids and found `272` checkpoints,
  max iteration `190000`. Round 0 is bounded/adaptive: `300` pairs, `6,300`
  games, `21` games per pair, `games_per_shard=21`, `save_gif=false` for fast
  proof. It completed but is not publishable because the rating snapshot had
  `decision_source_frames=null`. The corrected current website/default lane is
  the `dsf1` arena above.
- Live scheduler fix deployed: live run-id intake now defaults to
  `adaptive_v0`, `pairs_per_round=300`, and `active_pool_limit=100`.
  Regression tests passed. Exact-ref validation/stress lanes can still use
  all-pairs. This fixes the previous live continuation explosion where
  `round-000013` scheduled `261` refs as `33,930` pairs / `712,530` games.
- Fresh clean all-205 validation lane, 2026-05-16 04:17 EDT: tournament
  `curvy-r18fresh-validate-all205-20260516a`, rating
  `elo-r18fresh-validate-all205-20260516a`, detached app
  `ap-VQZzMzRPLR5ZojFpN1iHbR`, rating call
  `fc-01KRQXFK7WSS1ZEEMAW5GYAHFK`. It accepted exactly `205/205` refs and
  wrote `round-000000/input.json` with `20,910` all-pairs battles and
  `439,110` planned games. Logs prove successful running games with balanced
  randomized seats and `max_steps=1048576`. It is not complete proof until
  `round-000000/ratings.json` and root `latest.json` exist with `205` rows,
  `completed_game_count=439110`, and non-initial ratings.
- Watch rule for this validation lane: root/round progress can remain at
  `completed_game_count=0` while games are running, because the current
  `games_per_shard=1` code path writes progress only at map start and after the
  full game map returns. Use Modal logs for liveness, but use
  `latest.json`/`ratings.json` for final proof.
- 2026-05-16 04:30 EDT validation poll: app still alive with `505` tasks; logs
  reached pair indices around `1975..2034`; parsed recent games were `175/175`
  successful with `0` errors. Final `ratings.json`/`latest.json` is still
  pending.
- Superseded 2026-05-16 website/current fix: the website briefly pointed at the
  all-205 validation lane. That is no longer current. Current website/default
  should be the corrected `dsf1` lane above; the all-205 lane is validation
  only.
- 2026-05-16 recovery hardening deployed: skipped stale rounds now consume their
  round index for continuation, explicit calls to a skipped existing round
  return `status=skipped`, and legacy shard tallies missing
  `wins_by_checkpoint` are rebuilt from shard games. Targeted regression slice:
  `4 passed`; ruff clean. `curvyzero-checkpoint-tournament-v2` redeployed after
  the guard.
- Visibility truth: completed `latest.json` for the old live lane is stale at
  `98` rows / max iteration `70000`. Running validation input has `205` refs /
  max iteration `140000`. Running dirty/live `round-000012` input has `180` refs
  / max iteration `130000`. So new checkpoints are reaching running tournament
  rounds, but completed leaderboard snapshots have not caught up yet.
- Current tournament catch-up repair, 2026-05-16 04:03 EDT: active dirty lane is
  `curvy-r18fresh-live-20260516a` / `elo-r18fresh-live-20260516a`. Do not call
  it clean production proof. `latest.json` is now `round-000008` with `98`
  rated checkpoints, `4,753` pairs, `99,813` games, `58` failed games from
  progress, `stable=false`, and `max_abs_delta=318.85214603560865`. Root
  `progress.json` still points at old `round-000010` as running, while
  `round-000011` and `round-000012` folders also exist. The artifact tree is
  polluted by overlapping old workers.
- Current implemented repair bundle is deployed to
  `curvyzero-checkpoint-tournament-v2`: no-output skip requires the real stale
  age floor; public pointers are monotonic; `continue_from_latest` uses one
  active claim instead of a pool-hash-per-growth claim; and intake refuses to
  spawn reducer recovery for an unfinished running round. Focused
  intake/recovery/pointer tests and ruff passed before deploy.
- Current intake read: manifest has `196` seen checkpoint refs and
  `updated_at=2026-05-16T08:02:52.922179Z`. Queue length reports `0`; manifest
  bookkeeping still says `queued_checkpoint_count=196`, so queue bookkeeping
  must be re-proven on a clean run.
- App cleanup status: deployed v2 trainer/tournament/GIF browser apps are up.
  Detached old tournament apps `ap-ily3OHjnYXnun9616HKYGb` and
  `ap-svRQc9Y8SyxPu5wbJZ9fTT` are still `stopping...`. Recheck until gone
  before starting another large proof.
- Next action: update docs, confirm old detached apps are stopped, then choose
  one clean path: either fresh tournament/rating ids using current code, or a
  deliberate purge/repair of dirty artifacts before continuing this id. The
  proof gate is not "scheduled"; it is clean latest -> controller writes
  immutable assignments/control pointers -> same running trainers apply and use
  them.
- 2026-05-16 00:54 EDT reset: all CurvyZero v2 apps were stopped, then
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`,
  `curvyzero-curvytron-control-v2`,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-opponent-leaderboard-live-v2`, and
  `curvyzero-curvytron-checkpoint-events-v2` were deleted and recreated empty.
  The v2 apps were redeployed afterward.
- Any TODO row below that says a `curvy-r18v2-*`, `curvy-r18scratch-*`, or older
  tournament is live should be read as historical until regenerated after the
  00:54 reset.
- Hard boundary: stale rows that mention `92` refs, `round-000004`,
  `round-000005`, or `curvy-r18v2-bootstrap-*` are pre-purge evidence only.
  They must not be used as current launch truth. The current live lane is
  `curvy-r18fresh-live-20260516a` / `elo-r18fresh-live-20260516a`.
- Historical dirty-lane launch checklist status: redeploy v2 apps done; rebuilt
  fresh scratch
  `real18` as `curvy-r18fresh-allv2-20260516a`; submitted all `18` trainers;
  attached run-id based tournament intake as `curvy-r18fresh-live-20260516a` /
  `elo-r18fresh-live-20260516a`. Startup proof is complete through
  `iteration_0` checkpoint -> intake -> tournament rating, but the larger live
  tournament lane became dirty from overlapping old workers and has not closed
  the full feedback loop cleanly.
- Survival sanity check: latest survival is mixed but slightly improved from
  the previous read. `10/18` runs improved at latest vs `iteration_0`; `15/18`
  improved at least once; aggregate latest mean changed `159.2 -> 164.5`
  steps (`+5.3`). Keep monitoring, but do not
  describe this batch as a clean learning win yet.
- Do not use `stable=true` as the training-loop gate. It is a final-rating
  confidence label, not proof that the trainer can or cannot learn. The live
  training loop should publish a clearly labeled current training-candidate
  snapshot and rewrite immutable assignment pointers automatically. The
  controller must preserve the existing recipe shapes instead of silently
  collapsing all runs to one generic mix.
- Latest decision: the active lane is all-v2. Delete/recreate exact v2 objects
  only, update code/docs to point at them, redeploy v2 app names, then validate
  before any real training launch.
- Current all-v2 objects:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`,
  `curvyzero-curvytron-control-v2`,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-checkpoint-events-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Current app names:
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`,
  `curvyzero-checkpoint-tournament-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- Local code contracts are green for seat randomization, no-op/straight action,
  public slot immortality, trainer/tournament observation surface metadata, and
  balanced tournament seating.
- Current Modal app cleanup is not fully done: the all-v2 trainer, tournament,
  and GIF browser apps are deployed, but two old detached tournament apps are
  still `stopping...`. Do not start another large proof until they disappear or
  the next proof uses an isolated fresh id.
- Current storage cleanup is complete for the active lane. Old non-v2 canary
  assignments that point at `curvyzero-runs` are not valid inputs for this
  deployment unless rematerialized into v2 storage.
- Verified Modal VolumeFS versions: `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2` all open with `version=2`. Code must use
  explicit volume-version mapping, not suffix guesses.
- The fresh deployed E2E canary after the latest seat/slot cleanup passed after
  one pointer repair in the old lane, and the recreated all-v2 canary has now
  passed without manual pointer repair: checkpoint -> intake/subscriber ->
  tournament -> leaderboard -> assignment refresh -> trainer used the promoted
  opponent.
- Historical pre-launch discipline: before the fresh `r18fresh` launch, the
  rule was to avoid launching a larger batch from vibes. The larger fresh batch
  is now launched, so this row is no longer a blocker; use it as a reminder to
  keep proof and docs explicit.
- Latest local hardening bundle: `386 passed, 14 skipped`; ruff passed for
  touched trainer and poller tests. This found and fixed stale blank-slot
  fixtures plus eval-poller control-ref/nested-checkpoint reload issues.

| Task | Owner | Status | Next action |
| --- | --- | --- | --- |
| Survival stagnation root-cause pass | Main + subagents | Active | Current facts: `18/18` rows improved at best, only `10/18` improved at latest, latest mean `175.4` vs first `159.9` and best `246.0`. Mechanics/reward audit did not find wrong-seat bonus/reward or wrong tournament observation. Latest baseline-control critique says the best historical CurvyTron signal was static matched frozen-opponent training (`s92`, roughly `151.8 -> 417.0 -> 500.4`), not live tournament refresh. Current top suspects for the completed r18fresh batch are live opponent nonstationarity, dense target/support cap, weak `num_simulations=8`, huge collect waves, and historical r18fresh `batch_size=32`. Current broad L4 launch default is now `batch_size=64`. Next action: monitor `curvy-r18nofb-*` to nonzero checkpoints, then decide whether to patch support/cadence before any new broad launch. |
| Atomic latest-file writes | Main | Deployed to trainer app | Early no-feedback polling exposed `progress_latest.json` mid-write reads. `run_management.write_json` now writes non-exclusive JSON through temp file plus atomic replace; focused test and ruff pass. Trainer app `curvyzero-lightzero-curvytron-visual-survival-train-v2` was redeployed at 2026-05-16 11:03 EDT. Existing running jobs keep their old image; new jobs use this. |
| Explicit trainer diagnostic knobs | Main | Deployed to trainer app | `scripts/build_curvytron_tonight18_manifest.py` now exposes `collector_env_num`, `n_episode`, `num_simulations`, `batch_size`, `model_support_cap`, `td_steps`, and background eval sizing. The trainer accepts support-cap/`td_steps` overrides and records them in command/target config. Focused bundle: `146 passed, 3 skipped`; ruff clean. Smoke build with `64/25/128/support2048/td50` produced the expected fixed knobs. Next action: choose a small clean diagnostic shape after the no-feedback rows have enough signal. |
| Static/no-tournament control | Wegener + Main | Running; mixed early read | Fresh 11:02 EDT status: refresh events/applied remain `0` for all selected rows. `r007` has latest checkpoint/eval at `iteration_20000` with latest eval `138.125`; `r009` has `iteration_10000` with latest eval `167.625`; `r011` has checkpoint `iteration_30000`, latest completed eval `iteration_20000` at `171.5`. Subagent aggregate over the selected three says first/best/latest `156.4 / 160.4 / 151.5`; this is early but not an obvious clean win. Keep watching and compare to any target/support control before another full launch. |
| Own-latest no-tournament control | Copernicus + Main | Fresh selected3 launched | Trainer can publish a same-run nonzero exact checkpoint as an immutable opponent assignment and update a run-local `runs:` refresh pointer, then consume it through the existing refresh machinery. Manifest builder has `--own-checkpoint-opponent-refresh`; smoke manifest confirmed no assignment bank/control pointer. The first selected3 launch (`curvy-ownlatest-staticmix-20260516a`) is correctly shaped but not proof: subagent status found all three rows still at iteration 0, `assignment_refresh_applied_count=0`, and one `kept_previous` refresh event per row caused by a runs-volume reload failure with an open TensorBoard event file. Those stale pre-fix train/poller calls were cancelled after the fresh lane launched. Fix deployed at 2026-05-16 11:23 EDT. Fresh proof lane launched as `curvy-ownlatest-staticmix-20260516b` selected rows `r007/r009/r011`; launch record `artifacts/local/curvytron_tonight18_manifests/curvy-ownlatest-staticmix-20260516b/curvy-ownlatest-staticmix-20260516b.selected3.launch.json`; train calls `fc-01KRRP92QZBVWY8AWP9F65A7HD`, `fc-01KRRP92ZXNV5EZABBC30HE81E`, `fc-01KRRP936TR1GQD6XDBQSM0FA4`. Startup poll: no checkpoints/progress yet; pollers running; no refresh applied. Next action: poll after first `iteration_10000` and prove own pointer write -> applied refresh -> provider-ok env rows -> learner metrics visible in status. |
| Learner metrics observability | Main | Deployed to trainer app | Added passive `BaseLearner.train` recorder: first few calls and every 1000th call write `learner_metrics.jsonl` and `learner_metrics_latest.json`; checkpoint progress embeds `last_learner`; run-status falls back to `learner_metrics_latest.json` and reports call index, train iter before/after, collector envstep, elapsed time, and numeric learner metrics. This is non-blocking and preserves original result/exception behavior. Next action: use it only on fresh jobs launched after the 11:23 deploy. |
| Target/support scale control | Main | Open | The dense rows imply million-scale returns while the LightZero model support is capped at `300`. Decide whether to add launch-time overrides for model support cap, shorter `source_max_steps`, or reward scaling before another real 18-row launch. This is a higher-value patch than another broad noisy sweep. |
| Modal/RL debugging research | Huygens | Done | Wrote `modal_debugging_patterns_2026-05-16.md` using official Modal docs. Operational rule: prove autonomy from durable artifacts, not submitted logs; use deployed/detached apps for background work; use Volume artifacts as durable truth; make queue/dict gaps non-blocking and recoverable. |
| Original-vs-current delta audit | Popper | Done | Wrote `original_vs_current_deep_compare_2026-05-16.md`. Top suspects: reward/value support saturation, live refreshed-opponent nonstationarity, huge 256-env collect waves with batch 32, shallow `num_simulations=8`, possible selected-action vs executed-action noise mismatch, sparse outcome credit weakness, and tournament feedback differing from the older matched-opponent setup. |
| Corrected bounded live feedback proof | Main | Gen9 and Gen10 closed | Current arena is `curvy-r18fresh-live-bounded-dsf1-20260516b` / `elo-r18fresh-live-bounded-dsf1-20260516b`. Gen9 proof: `18/18` runs, `177,382` target-sha rows, `89,934` frozen-checkpoint provider-ok rows, `0` provider-load false. Gen10 proof after 30-minute wait: `18/18` runs, `177,203` target-sha rows, `93,427` frozen-checkpoint provider-ok rows, `0` provider-load false. Tournament latest kept advancing to `round-000008` / `365` rows / max iteration `260000`. Next action: make the controller trigger durable/periodic instead of operator-run. |
| First bounded live feedback proof | Main | Completed but not publishable | `curvy-r18fresh-live-bounded-20260516a` / `elo-r18fresh-live-bounded-20260516a` completed `272` rated rows through max iteration `190000`, but `rating_spec.decision_source_frames=null`; the controller correctly refused it. Keep as evidence that bounded scheduling works, not as trainer source. |
| Clean all-205 tournament validation | Main | Running, not final | This is no longer the website default; the current/default arena moved to the bounded live proof lane. Latest artifact check: `205` refs, max iteration `140000`, `20,910` pairs, `439,110` games, and running Modal game logs; final `ratings.json`/`latest.json` is still pending. Final gate: all games complete, `ratings_written=true`, root `latest.json` points at `round-000000`, exactly `205` ranked rows exist, and ratings/ranks are no longer all initial. |
| Live tournament orphaned-round recovery | Main | Deployed; dirty lane not proof | Current deployed code has the stale-skip, monotonic-pointer, active-continue-claim, and no-premature-reducer fixes. The live lane itself is polluted: `latest.json` is `round-000008` with `98` rows / `99,813` games / `58` failures / `stable=false`, while root progress still shows old `round-000010` and later running artifacts exist. Next action: either start a fresh clean tournament/rating id with current code, or explicitly purge/repair this lane before using it for proof. |
| All-v2 namespace reset | Main | Done | Contract points every current CurvyTron app/Volume/Dict/Queue at `-v2` names. Exact v2 Volumes/Dicts/Queue were deleted and recreated at about 2026-05-15 14:08 EDT; the three Volumes pass `Volume.from_name(..., version=2).info()`; v2 trainer/tournament/GIF apps were redeployed at about 14:09 EDT; old non-v2 Curvy trainer/tournament deployments were stopped at about 14:10 EDT. |
| Fresh all-v2 deployed canary | Main | Passed | `curvy-e2e-allv2-canary-20260515a` launched on the v2 trainer app after copying one historical seed checkpoint into `curvyzero-runs-v2`; v2 intake/tournament `curvy-e2e-allv2-canary-live-20260515a` / `elo-e2e-allv2-canary-live-20260515a` completed `round-000003` with `18/18` games, `0` failures, and `stable=true`; promotion wrote assignment sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`; same running trainer applied it at train iter `5061`; later refresh stayed on that sha at train iter `5372`; env telemetry had `1836` provider-ok rows using it on the latest fetch. |
| Checkpoint policy metadata sidecars | Main | Deployed to v2 tournament; monitor live proof | Fresh checkpoint progress hooks now write `iteration_N.pth.tar.metadata.json` beside each `.pth.tar`, with policy observation surface, runtime timing, model env/reward variants, and learner seat mode. Tournament discovery, direct rating refs, and intake rating specs preserve discovered sidecar fields before falling back to run/attempt metadata or defaults. Focused tests pass and `curvyzero-checkpoint-tournament-v2` has been redeployed after this patch. Still keep the live batch proof honest: do not claim tournament/trainer surface parity for a new checkpoint until a current-batch checkpoint has been admitted and rated with the expected metadata visible in durable artifacts. |
| Fresh all-v2 canary launch | Main | Passed | Poller `fc-01KRPDXH9FW4DNNMW41G9XC3EY`, trainer `fc-01KRPDXHDF4C5RS17DR1Z18EVX`. Keep this as the current launch template: all-v2 volumes/dicts/queue, `control:` refresh pointer, live run-id intake, explicit canary relaxations only for tiny proof promotion, and provider-ok env telemetry as the final proof. |
| Submitter v2 control-volume open | Main | Done | `scripts/submit_curvytron_survivaldiag_manifest.py` now uses `modal_volume_kwargs_for_name(...)` for direct refresh-pointer writes, so `curvyzero-curvytron-control-v2` is opened with `version=2`. Focused submitter/manifest/shared-contract launch bundle passed (`26 passed`) and ruff is clean. |
| Restart18 manifest fail-closed defaults | Main | Done locally | `scripts/build_curvytron_tonight18_manifest.py` now requires either an explicit ranked snapshot or an explicit checkpoint refs file, defaults to assignment mode, writes assignments and refresh pointers to `control:`, uses default refresh interval `2000`, and names fresh `curvy-r18v2-*` rows. Bootstrap can use `--checkpoint-refs-file` so it does not pretend a trusted ranking exists. Inline mixture mode now needs an explicit diagnostic refresh-off flag. Covered by focused launch tests and E2E-adjacent slices; rerun after each launch patch. |
| Submitter app-name guard | Main | Done locally | `scripts/submit_curvytron_survivaldiag_manifest.py` rejects stale or overridden app names unless they match each row's manifest app and the current shared-contract all-v2 trainer app. This blocks accidental submission of v2 rows into old apps. Covered by focused launch tests and ruff. |
| Restart18 diagnostic manifest dry-run | Main | Done; do not launch | Built `curvy-r18v2-dryrun-canarysource-20260515a` from the all-v2 canary leaderboard and dry-ran the submitter. It has the right control/refresh/app/surface/cadence fields, but the source has only `4` active rows and rank 1 is `iteration_0.pth.tar`, so it is wiring evidence only. |
| All-v2 ranked-source inventory | Plato + Main | Done | Current v2 tournament storage has only `curvy-e2e-allv2-canary-live-20260515a` plus public leaderboard `e2e-allv2-canary-live-r3-20260515a`. No production-quality leaderboard-derived opponent source exists in recreated v2 storage. Do not build leaderboard-derived restart18 opponent assignments from existing v2 storage until a real ranked snapshot is created or old strong refs are rematerialized/rerated into v2. Bootstrap/static launch may use curated assignments and exact refs. |
| Manifest checkpoint-ref existence audit | Main | Done locally | Added `scripts/audit_curvytron_launch_manifest_refs.py` plus tests. It collects initial-policy refs and frozen assignment/mixture refs, rejects mutable/control-prefixed checkpoint refs, and can check local or Modal existence. It now also accepts `--refs-file` for source/rematerialization gates. Focused audit tests passed, the canary dry-run manifest passed `--check-modal` against `curvyzero-runs-v2` with `4/4` refs present, and the restart source refs passed syntax plus old-source Modal existence with `100/100` present. |
| Fresh deployed E2E canary after cleanup | Main | Passed after repair | Short canary proved checkpoint -> live run-id intake -> rating -> leaderboard -> assignment -> pointer rewrite, but ended before same-trainer refresh. Long canary `curvy-e2e-clean-canary-long-20260515c` then proved the full loop: live tournament `curvy-e2e-clean-canary-long-live-20260515c` completed `2` pairs / `6` games / `0` failures; promotion sha `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30`; same trainer applied at train iter `2750`; `env_steps.jsonl` had `1107` rows using that sha with provider load OK. Post-refresh events stayed on the sha through train iter `4565`; old canary app is now stopped. |
| Stop invalidated v2 run | Main | Done | Invalidated v2real18 evidence is diagnostic only. The v2 durable objects have now been recreated fresh, so old v2real18 refs are gone from the active lane. Do not publish its rerate as restart evidence. |
| Bootstrap restart gate for next training | Main | Live batch and tournament running | The recreated all-v2 loop is proven at canary scale, and the larger launch does not need a perfect historical leaderboard. Fresh review artifact `curvy-r18v2-bootstrap-20260516a` uses curated exact refs, immortal blank/hard-coded pressure, small explicit immortal checkpoint slices, mostly mortal frozen checkpoint slots, `random_per_episode`, all-v2 storage, checkpoint metadata sidecars, control assignments/pointers, and `save_ckpt_after_iter=10000`. Syntax audit, Modal ref audit (`4/4` refs present in `curvyzero-runs-v2`), dry-run grouped submission, focused tests, and trainer/tournament redeploy checks passed. Submission wrote `3` assignments, `3` refresh pointers, and spawned `18` trainers plus `18` pollers. Current status: all `18` rows are alive with numbered checkpoints, the live tournament watch is preserved as run-id based, and `round-000004` is the active large rating continuation with `92` refs in the intake watch. Durable latest still points at unstable `round-000003` (`57` rated checkpoints), so the large lane is not publishable yet. Next action: keep monitoring `round-000004`; only publish/materialize from a `stable=true` latest snapshot that passes gates, then rewrite the recipe control pointers and verify same-trainer refresh/provider-ok rows. |
| Live-watch collapse repair | Main | Round 5 running after repair | Exact-ref submissions into an existing live intake must not turn the intake into a frozen explicit-ref-only scan. Patch preserves existing run-id/prefix watches while pinning exact refs. Two follow-up Modal bugs were fixed after live testing: a zero-work `waiting_for_round_input` progress stub no longer blocks the round writer, and the rating loop now uses spawned child round workers instead of `.remote()` from a detached app. Broad tournament/intake regression passes (`162 passed, 11 skipped`) and ruff passes; `curvyzero-checkpoint-tournament-v2` was redeployed. Active large manifest is live-watching `18` run ids with `checkpoint_selection=all` and now has `92` refs. `round-000004` was recovered by detached reduce after transient Modal Volume I/O and parent-tail trouble; it is complete with `92` checkpoints, `4186` rated pairs, `87906` games, `failed_game_count=1`, `stable=false`, and `max_abs_delta=401.45`, so do not publish it. Same-pool continuation bug fixed: CLI passed `spawn_if_existing`, but drain only honored `spawn_if_empty`; focused regression passes and app was redeployed. Patched drain spawned `round-000005` as app `ap-CHJuWMB4lb4qfENQqv86N3`, function call `fc-01KRQEPEAAFCC595YRMP0R7MB0`; `round-000005/input.json` exists with `92` checkpoints, `active_pool_limit=100`, all-pairs, `4186` pairs, `87906` games, and `previous_round_id=round-000004`. Next action: monitor `round-000005`; promote/materialize only from a `stable=true`, zero-failure snapshot that passes gates; separately fix or replace stale progress writer/reducer observability. |
| Automatic training-candidate controller | Main | Local controller implemented; large remote proof open | The large lane should not depend on manual promotion. Current code has `curvytron_training_candidate_refresh`: it reads a tournament rating snapshot, writes a clearly marked training-candidate leaderboard, materializes immutable per-recipe assignments, stages assignment/audit writes before pointer rewrites, rejects stale pointer races, fails closed on reload/commit errors, and publishes the Dict pointer only after trainer-facing pointers are committed. The remote canary proved this shape earlier; the dirty 18-run lane has not cleanly proven it after the latest tournament fixes. |
| Training-candidate toy experiments | Main + subagents | Passed locally | Fast local controller proof added in `tests/test_curvytron_training_candidate_controller_local.py`: `8 passed`. The toy uses fake local mounts, not remote Modal games. Covered cases: fake tournament `latest.json`, leaderboard snapshot/Dict pointer, control assignment files, refresh pointer rewrite, rank-slot replacement, recipe preservation, input pointer sha validation, tournament/control commit failure abort, reload failure abort, stale-pointer/CAS rejection, missing-rank behavior, and trainer-visible pointer resolution. This proves the missing controller bridge locally, not remote deployment. |
| Large leaderboard promotion plan | Main + McClintock | Prepared, not safe on dirty lane | Preconditions are clear: future clean `latest.json` must advance to the intended round; `ratings.json` and `latest.json` must agree; input/progress/results/ratings/latest must agree on tournament/rating/round; one-frame contract and expected failure policy must hold; no diagnostic/provisional relaxations for production. `stable=true` is a final-rating quality label, not a bootstrap-training gate, but a dirty overlapping round tree is still not safe to publish as proof. |
| Survival improvement evidence | Main | Mixed, keep monitoring | Current `/private/tmp/curvy-r18v2-eval.clean.json` summary: 18 rows alive, latest checkpoint range roughly `iteration_40000` to `iteration_80000`; overall latest eval mean is up `+9.17` steps, best-so-far mean is up `+39.67`, latest improves in `9/18` rows, best-so-far improves in `16/18`, and `1/18` latest rows is collapsed. Reward split: sparse outcome is strongest (`+41.46` latest mean), survival+bonus without outcome is nearly flat (`+2.23`), survival+bonus+outcome is down (`-16.19`). This proves policies are changing and sometimes improving, not that the whole batch is steadily getting better. |
| Bootstrap refs manifest | Main | Tournament round complete; not full loop | `curvy-r18bootrefs-20260515a` exposed a warm-container control-volume visibility race and is historical/failed. Fix landed: trainer startup reloads assignment/checkpoint volumes before first assignment read. Fresh launch `curvy-r18bootfix-20260515a` is built from the same audited refs file, uses 18 rows, control-volume immutable assignments, control-volume refresh pointers, shared initial checkpoint `iteration_240000.pth.tar`, `random_per_episode`, and 20/25/30% immortal pressure. Dry-run and Modal ref audit passed (`4/4` exact refs present). Submission wrote `3` assignments, `3` refresh pointers, and spawned `18` trainers plus `18` pollers. Status proved all `18` rows launched and wrote at least `iteration_0`; discovery before intake found `13/18` latest refs at `iteration_10000` and `5/18` still at `iteration_0`, with expected policy metadata sidecars. Live tournament `curvy-r18bootfix-live-20260515a` / `elo-r18bootfix-live-20260515a` completed all-pairs `round-000000`: `32` checkpoints, `496` pairs, `10,416` games, `0` failed games, `ratings_written=true`, `stable=false`, `max_abs_delta=117.51743133112922`. Treat this as tournament/intake proof, not final ranking truth. Next action: use the patched canary to close trainer refresh proof. |
| Assignment refresh runs-volume reload conflict | Main | Deployed; needs remote proof | Live `curvy-r18bootfix` status exposed a non-fatal but important refresh problem: refresh checks can fail before reading the pending assignment because reloading the runs Volume is blocked by open TensorBoard files. Local fix makes checkpoint-volume reload during assignment refresh best-effort and then tries the already-mounted checkpoint path; control-volume pointer reload remains strong. Focused tests and ruff pass. `curvyzero-lightzero-curvytron-visual-survival-train-v2` has been redeployed. Existing running jobs are still on the old deployed code, so relaunch or a fresh canary is needed to prove this fix remotely. |
| Assignment refresh warning leakage | Main | Remote canary proof passed | A non-fatal checkpoint Volume reload warning must never be inserted into `opponent_mixture.entries`; the env validates those entries as gameplay slots. Local patch stores reload warnings as resolved-assignment metadata instead. Regression passed: `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_resolve_opponent_assignment_checkpoint_reload_failure_can_be_nonfatal`; broader opponent/plumbing slice passed `106 passed, 3 skipped`; ruff passed for touched files. `curvyzero-lightzero-curvytron-visual-survival-train-v2` was redeployed after the patch. Fresh canary proof then applied a promoted assignment in the same running trainer and used it in env rows with provider load OK. |
| Warningfix deployed canary | Main | Passed canary-scale full refresh proof | Fresh one-row canary `curvy-e2e-warningfix-canary-20260516a` launched after the warning-leak deploy. Tiny tournament `curvy-e2e-warningfix-live-20260516a` / `elo-e2e-warningfix-live-20260516a` rated `iteration_0` vs `iteration_1000` with `21` games, `0` failures, and `stable=true`, then published leaderboard snapshot `e2e-warningfix-live-r0-20260516a`. Promotion wrote assignment sha `8b171c177c401b886a5658fafc1c16076b5797c640b6d6a689003575e6d46208` and rewrote the canary pointer. The same trainer applied it at train iter `5693` with `env_ready_report.ok=true`; `env_steps.jsonl` had `87` rows using the new sha, all `87` with `opponent_provider_load_ok=true`, and no observed provider-load failures. This proves the refresh loop at canary scale, not survival improvement or large-batch stability. |
| Current-code high-frequency stress canary | Main | Stress-only; not launch proof | `curvy-e2e-currentlive-canary-20260516a` launched from the current redeployed trainer app with starter assignment sha `af4167dd73868e5d7444b4b40b7ef28c86f6cfdc71c41f62a5cda368e02df81f`. It produced `77` checkpoints and `65` eval manifests; live run-id intake `curvy-e2e-currentlive-live-20260516a` / `elo-e2e-currentlive-live-20260516a` later saw `67` checkpoints. It never applied a promoted assignment, so it is not full-loop proof. The final failed status appears to be a post-run artifact-scanner bug: LightZero returned `ok=true` and checkpoints existed, but the scanner looked outside the mounted runs Volume. Local scanner fix and regression passed. |
| Sparse current-code live proof | Main | Passed canary-scale full live proof | `curvy-e2e-currentlive-sparse-canary-20260516a` launched after dry-run and Modal ref audit passed. Train call `fc-01KRQ4K9RGG7P5VAQBS9H19Z3W`, poller call `fc-01KRQ4K9N4M9WMA0SGWF5D56GW`, starter sha `4b7261e7a795da17517360a768b581879dd20ffa034d30f8cc7540858a731f4b`. Live run-id intake `curvy-e2e-currentlive-sparse-live-20260516a` / `elo-e2e-currentlive-sparse-live-20260516a` found exactly `2` checkpoints and ran through stable `round-000007`: `1` pair / `21` games / `0` failures / `stable=true`. Promotion published snapshot sha `a1a003523adde3f3fc273ecba9b825da60f3c75f0b7ff23064e2db34ea24a79b`, wrote assignment sha `774b70dd15fa71bc59a92819f3d417c9025184d6a24634ad4dbebe490dbb1009`, and rewrote the sparse canary pointer. The same trainer applied that sha at train iter `5373` with `env_ready_report.ok=true`; later `env_steps.jsonl` fetch had `357` rows with that sha, `312` provider-ok rows, and `0` observed provider-load-false rows. This proves the current-code live run-id feedback loop at canary scale; it does not prove survival improvement or production-scale ranking quality. |
| Patched trainer refresh proof | Main | Pointer rewritten; wait for trainer apply | Do not count the current r18bootfix tournament as patched-trainer refresh proof because those trainers started before the latest redeploy. Fresh patched canary `curvy-e2e-patched-refresh-canary-20260515a` is running after the trainer redeploy and reached numbered checkpoints. Live watch tournament `curvy-e2e-patched-refresh-live-20260515a` / `elo-e2e-patched-refresh-live-20260515a` completed round 0 with `26` checkpoints, `325` all-pairs battles, `975` games, and `0` failures, but promotion correctly refused because `stable=false`. Frozen-pool intake `curvy-e2e-patched-refresh-frozen2-20260515a` / `elo-e2e-patched-refresh-frozen2-20260515a` then seeded exact refs `iteration_0` and `iteration_1000`; final `round-000007` was stable with `1` pair, `21` games, `0` failures, and `max_abs_delta=1.1749968212540947`. Promotion published snapshot sha `9e162f96d1f7fa7ef2ef8204df488862ef0a57ad63f839f5366790c00f39eae9`, wrote assignment sha `f8a469b5ff8598fe64bd42906de64fb68d06a8aa75f6f4a2c20be82fa4c8eedc`, and rewrote the canary control pointer. Next action: require one `decision=applied` refresh event in the same running patched trainer plus later env rows with that assignment sha and `opponent_provider_load_ok=true`. |
| Optional ranked-source opponent lane | Main | Diagnostic only | The 100-ref rerate is diagnostic only because iteration-zero rows rose to the top. The current ranked-source candidate is the 96-ref nonzero fallback `curvy-restart18-source-rerate-nonzero-20260515a` / `elo-restart18-source-rerate-nonzero-20260515a`; round 6 completed with `96` active rows and `0` failures but worsened to `stable=false`, `max_abs_delta=25.199213332028748` after round 5 had `15.636412948237727`. Do not use this rerate for leaderboard-derived restart18 opponent assignment/promotion until it is coverage-mature, `stable=true`, hash-guarded, and audited. This is useful later, but it must not block bootstrap/static training. |
| Source leaderboard wording | Main | Corrected | A ranked/source leaderboard means only "a ranked list used to choose better starting checkpoint opponents." It is optional quality input. It is not a bootstrap blocker and it is not required to prove the loop. Bootstrap can run with exact refs plus immortal sentinels while the tournament catches up. |
| Hard-coded slot immortality cleanup | Main | Tested and deployed | Generic opponent mixtures now reject hard-coded `fixed_straight` / `proactive_wall_avoidant` entries unless `opponent_immortal=true`. The old stable-slot selector rejects bulk immortal checkpoint slots; use explicit modern mixture recipes for small frozen-immortal slices. Latest focused validation passed: `131 passed, 3 skipped` for opponent/plumbing/publisher slices; `26 passed` for learner-seat/no-op/manifest slices; ruff clean. The v2 trainer and v2 tournament apps were redeployed from this code on 2026-05-16. |
| Intake claim false-proof risk | Main | Watching | A claim can exist before the operator sees a rating call id. For this live run, the rating artifacts and logs now prove work is running, so the claim is not dead. Future proof must check Volume artifacts, not just dict claims. |
| Production-quality promotion gate | Main | Open | Canary promotion used relaxed/provisional gates and picked `iteration_0.pth.tar` as champion. A real leaderboard-derived opponent source must come from a production-shaped all-v2 rerate with active-row gates satisfied, `stable=true`, zero unexpected failures, and non-diagnostic assignment materialization guarded by expected round/context/roster/snapshot hashes. |
| Reject unstable training publishes | Main | Deployed | `curvytron_opponent_leaderboard_publish` now refuses `stable=false` rating snapshots unless `diagnostic_only=True`; diagnostic publishes do not update latest/pointer. Focused tests passed, and `curvyzero-checkpoint-tournament-v2` was redeployed after the guard change. |
| Production source strategy | Rawls + Main | Recommended | Use `loop18-main-adaptive417` only as a candidate-selection artifact: top 100 active exact checkpoint refs, copy/rematerialize those files from old runs storage into `curvyzero-runs-v2`, then rerate under fresh v2 tournament/rating ids. Do not copy the old leaderboard as launch truth; do not start from canary as production source. |
| Restart source-ref plan artifact | Main | Done locally | Added `scripts/prepare_curvytron_restart_source_refs.py` and generated `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top100-20260515a/`. It has `100` refs, `4` iteration-zero candidates, max iteration `306755`, and command text with source audit, copy, target audit, then rerate. Latest focused guardrail tests after CLI correction: `13 passed`; ruff clean. |
| Rematerialize restart source refs into v2 | Main | Done | Source audit passed: `100/100` refs exist in old `curvyzero-runs`. Target-before-copy audit showed `100/100` missing from `curvyzero-runs-v2`, as expected after the v2 reset. `scripts/rematerialize_curvytron_checkpoint_refs.py` copied `100/100` refs into `curvyzero-runs-v2`; target-after-copy audit passed with `100/100` present. |
| Fresh v2 source rerate | Main | Diagnostic only now | Rounds 0-6 each completed `300` pairs / `6,300` games with `stable=false`; latest `max_abs_delta=18.39723682286698`. Coverage is mature, but `iteration_0` rows occupy ranks `1`, `2`, `7`, and `100`, so the 100-ref lane is not a restart source even if later rounds cross the numeric stability threshold. Focus on the nonzero fallback. |
| Nonzero source fallback | Main + Sagan | Round 6 complete; diagnose before more rounds | Built `restart18-source-loop18-top96-nonzero-20260515a` with `96` historical active refs after excluding `iteration_0`; old-source and v2-target audits both pass `96/96`. Round 0 completed `stable=false`, `max_abs_delta=34.07017967162989`, `0` active rows. Round 1 completed `stable=false`, `max_abs_delta=21.880940181012807`, `0` active rows. Round 2 completed `stable=false`, `max_abs_delta=22.572625403714373`, `15` active rows. Round 3 completed with all `96` rows active but `stable=false`, `max_abs_delta=39.7420779825474`. Round 4 completed `stable=false`, `max_abs_delta=17.371056613899057`. Round 5 completed `stable=false`, `max_abs_delta=15.636412948237727`. Round 6 completed `300` pairs / `6300` games, `0` failures, `96` active rows, `stable=false`, `max_abs_delta=25.199213332028748`; the max mover was `ckpt-079` after mostly `random_bridge` exposure. Do not publish. Diagnose scheduler/exposure before continuing. |
| Optional ranked-source publish/materialize path | Helmholtz + Main | Prepared; waiting on quality signal | This is optional quality work, not a bootstrap blocker. Once a rerate is good enough (`stable=true`, coverage-mature, expected hashes, zero unexpected failures), use `scripts/promote_curvytron_rating_round.py` with expected round/context/roster/snapshot hashes, explicit `--assignment-target-volume control`, then build leaderboard-derived opponent assignments from the fetched public snapshot, dry-run, audit refs against `curvyzero-runs-v2`, and publish assignments before trainer launch. Bootstrap can proceed from exact refs plus immortal sentinels without this. |
| Survival improvement evidence | Main | Open | The all-v2 canary proves wiring only. Current survival evidence remains partial/noisy; do not claim learning improvement until eval/collector survival metrics improve cleanly over a meaningful window. |
| Background eval/GIF proof | Main | Open | The all-v2 canary proves training refresh, not broad background eval/GIF poller behavior. Keep that path disabled or separately smoke it before depending on it for a large launch. |
| Background eval poller with control refs | Main | Fixed and smoke-tested for current namespace | Live logs showed a poller could miss a newly written `control:` assignment file. Command resolution now reloads `/control` or `/runs` before reading prefixed assignment refs, and poller assignment resolution can reload nested checkpoint volumes without changing the active trainer refresh path. Focused validation: `85 passed, 3 skipped`; broad slice: `386 passed, 14 skipped`; ruff passed. Remote smoke against an old non-v2 canary assignment failed correctly because current trainer defaults read checkpoints from `curvyzero-runs-v2`; a current-namespace smoke `curvy-e2e-poller-v2-namespace-smoke-20260515a` then completed without assignment/checkpoint resolution errors. |
| Next manifest design | Main | Locally prepared and audited | Current tonight18 builder uses `random_per_episode` by default, exposes fixed seats only as diagnostics, uses public `opponent_immortal`, keeps blank and hard-coded sentinel entries immortal, allows small explicit immortal frozen-checkpoint slices, and bounds total immortal pressure at `20-30%` across recipes. Current recipes are `20%`, `25%`, and `30%` total immortal pressure. Focused manifest/env/tournament/plumbing tests passed after the latest cleanup. |
| Audit player perspective in training | Averroes + Main | Locally implemented | Trainer default is `learner_seat_mode=random_per_episode`; old `ego_player_index` config is rejected; focused regression covers both seats over reset. Rerun focused tests after the current slot cleanup. |
| Audit player perspective in tournament eval | McClintock + Main | Locally implemented | Tournament game specs default to balanced physical seating and per-checkpoint policy observation surfaces; focused tests cover balanced seats, seat-aware win counting, eval policy mode, and policy surface rejection. Rerun focused tests after the current slot cleanup. |
| Confirm live-policy no-op semantics | Main + perspective agents | Locally documented | Current action space is `left`, `straight`, `right`; `straight` is the explicit no-turn action. Treat this as satisfying no-op-in-turn-space unless the engine contract changes. |
| Purge hidden compatibility defaults | Main + Pascal | Locally fixed | Fresh training defaults to `random_per_episode`, old `ego_player_index` config is rejected, fresh mixture/assignment recipes use public `opponent_immortal`, blank/no-op entries must explicitly be immortal, episode selection derives runtime `opponent_death_mode`, and current launch/manifest/tournament defaults now read from `src/curvyzero/contracts/curvytron.py`. The old `curvytron_volume_names.py` shim is deleted; the tournament web default no longer points at invalid v2; fresh tournament specs no longer accept observation/source/generic render aliases. The shared Modal Volume helper now rejects old Curvy non-v2 volume names unless an explicit migration/audit tool is used. Latest focused validation for the current guardrail patch: `14 passed`; ruff clean. |
| Tournament balanced physical seats | Zeno + Main | Locally done | Zeno implemented default `balanced_random` tournament seat ordering and tests. Integrate this as a restart requirement and keep it separate from trainer `learner_seat_mode`. |
| Recheck stale local red reports | Main | Done for non-render blockers | Rechecked current code after side-agent reports: `tests/test_curvytron_checkpoint_intake_repair.py` -> `17 passed`; `tests/test_multiplayer_source_state_trainer_surface.py` -> `11 passed`. Render pixel-perfect parity remains deferred to optimizer unless it blocks the current policy surface contract. |
| Inventory live metrics and decide relaunch | Rawls + Main | Done enough for decision | Audit found `21` tracked rows, `14` running, `7` failed, `215` checkpoints, max `iteration_160000`, old live leaderboard only `53` rows / max admitted `30000`, no reward/private fields. Decision: fix seat training before trusting/relaunching; keep current rows as smoke/data unless they waste compute or block capacity. |
| Workspace cleanup inventory | Volta + Main | Active lane reset done | Exact v2 Curvy volumes, dicts, and queue were deleted, recreated, and verified. Non-v2 Curvy storage remains as historical evidence only. See `cleanup_lane_2026-05-15.md`. |
| Non-v2 intake queue stale items | Main | Historical foot gun | Queue `curvyzero-curvytron-checkpoint-events-v0` has old items, but the active v2 queue was recreated empty. Do not use non-v2 intake for the next launch. |
| Intake drain duplicate rating race | Main later | Observed, not blocking canary | Both fresh canary drains completed the intended rating round, but parent logs also showed a second spawned rating call raising `FileExistsError` because `round-000000` artifacts already existed. Keep `--wait` and inspect durable artifacts for now; later tighten claim/spawn behavior so one drain cannot spawn duplicate round workers for the same claim. |
| Weak-run immortal intervention | Godel + Main | Dropped for current live rows | Audit proved the five-row intervention was not applied and shared pointers would overreach. User no longer wants this as a live intervention. Fold the lesson into the next manifest instead: use at least about `20%` blank/immortal exposure globally, with some higher-immortal variants, after the seat-perspective fix. |
| Correct v2real18 tournament runtime contract | Main | Deployed and launched | Patch rejects timing mismatch and propagates `policy_bonus_render_mode`; `154 passed, 21 skipped`. Corrected diagnostic rerate `elo-v2real18-rerate67-allpairs-16ms-20260515a` is running under app `ap-MKU8vQNXqZWCqX6Dle0ztG`; persisted input verified exact 16.6667ms and historical CPU-control `body_circles_fast + simple_symbols`, not the restart target. |
| Stop v2 real18 loop cleanly | Main | Superseded by stop decision | Current tournament `curvy-v2real18-live-20260515a` / `elo-v2real18-live-20260515a` remains useful for forensic evidence only. Stop or let stale compute drain through the cleanup lane; do not keep refreshing assignments from this run as a candidate training source. |
| Monitor replacement rows for replay crash fix | Main | Launched | Replacements for failed rows `r008`, `r009`, `r011` are spawned from `curvy-v2real18-refresh-r1-20260515a`; confirm their summaries do not contain `td_steps=1048576` and they pass the old `a/p size` failure point. |
| Catch current tournament up to current checkpoints | Main | Active | Discovery over the 21 tracked real18/replacement run ids found 67 exact checkpoint refs, while latest tournament ranks only 40 rows. Submit refs in batches of 10, then run detached intake drain/continuation and confirm row count grows without provisional rows. |
| Fresh 67-ref all-pairs rerate | Main | Corrected run active | Wrong-tick smoke app `ap-uIXpEjsU0Iy0lM0NHs8qEk` was stopped. Monitor corrected app `ap-MKU8vQNXqZWCqX6Dle0ztG` until `latest.json` exists or a concrete failure appears. |
| Archive corrected fresh rerate when complete | Main | Diagnostic only | After corrected `latest.json` exists, record it as timing/render smoke and perspective-failure evidence. Do not publish/materialize new training assignments from this rerate for the restart. |
| Quantify real18 survival trend | Main | Not yet proven | Current status over 21 rows shows `90` checkpoints and `15/21` applied refreshed assignments, but `eval_manifest_count=0`. Use raw progress only as a liveness signal. Wait for eval artifacts or add a compact survival telemetry reader before claiming improvement. |
| Finish tournament observation-surface patch | Main | Local tests pass | Latest local run: `uv run pytest tests/test_curvytron_checkpoint_tournament.py -q` passed with `123 passed, 11 skipped`; deploy remains separate. |
| Prove tournament eval matches trainer observation lane | Main + parity audit | Locally covered for CPU diagnostic lane | Added active-bonus parity coverage for historical CPU-control `body_circles_fast + simple_symbols`. Restart target is CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU `browser_lines + simple_symbols` remains lab/profiling-only until trainer-visible contract parity passes. |
| Explain and fix 51-row standings | Main + tournament debug lane | Historical/forensic | The old 51-row symptom belonged to invalidated v2refresh/loop18 lanes. Do not spend more launch energy there unless extracting old champion anchors. The next real batch must use fresh all-v2 tournament/rating ids and a clean dashboard default. |
| Clean rerate after patch | Main | Historical bounded first round | `elo-loop18-live-main-adaptive417-20260515b` completed 300 pairs / 6,300 games and wrote `latest.json`, but it is non-v2/historical and `stable=false`. Do not use it as current launch truth. |
| Full-loop proof | Main + validation lane | Proven on all-v2 canary | Current proof is `curvy-e2e-allv2-canary-20260515a`: trainer wrote checkpoints, v2 intake/tournament completed `round-000003`, promotion wrote sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`, same trainer applied it at train iter `5061`, and env rows used it with provider load OK. `controlrun2` is historical pre-reset template evidence. |
| Launch v2 real18 batch | Main | Superseded / diagnostic only | Old `submission-full.json` evidence is not a current launch lane. Do not keep refreshing assignments from this run as a candidate training source; use it only for forensic/survival-signal history. |
| Launch real refresh-enabled proof run | Main | Done as all-v2 canary | The current deployed-function proof is `curvy-e2e-allv2-canary-20260515a`. Future runs must use shared all-v2 defaults, a control-volume pointer, `commit_on_checkpoint=true` when live intake matters, and clear deployed-vs-ephemeral wording. |
| Trainer refresh pointer support | Main | Proven remotely | Trainer supports `/control` mounted Volume and `control:` assignment refs; all-v2 canary proved remote same-process refresh and provider-ok env use. Keep local tests, but no rerun is needed unless this contract changes. |
| Control-volume live proof | Main | Proven in all-v2 lane | `controlfast` and `controlrun2` are historical. The current proof is the recreated all-v2 canary. Next: scale/survival readiness, not another storage migration proof. |
| Second behavior proof | Main | Stopped; rating completed after stale read | Trainer `curvy-looplive-proof-controllong-20260515d` wrote checkpoints through `iteration_3021` and refresh checks fired, but Modal shows its app stopped at `2026-05-15 05:05:35 EDT`. Fresh progress shows `elo-looplive-controllong-proof-fresh-20260515e` completed `10/10` pairs and `210/210` games. We missed the chance to refresh that same running trainer. |
| Resolve storage namespace | Main | Proven for active all-v2 lane | Exact v2 objects were recreated and proven by `curvy-e2e-allv2-canary-20260515a`. Older `curvy-v2-looplive-proof3-20260515a` is pre-reset v2 evidence only. |
| Trainer initial policy checkpoint | Main | Done and smoke-verified | `initial_policy_checkpoint_ref` works for an immutable tournament winner with model-only `matching_shape` load and fresh optimizer preservation. Latest proof: `loop18-main-adaptive417-consume-smoke-20260515e`. |
| Quantify survival | Main | Updated again | Fresh read-only eval-summary over 18 `curvy-n18conn-*` rows completed 2026-05-15: best mean +49.8, latest mean +9.2, latest up 10/18. V2 canary post-refresh return rose `118.24 -> 134.72`, but mean length fell `159.20 -> 144.96` on a small sample. Keep monitoring because improvement is noisy, not monotonic. |
| Identify five weak runs | Weak-run lane | Done | Use the five rows listed in `TRAINING_CONTROL.md` for any intervention. |
| Inspect five weak-run slot mixes | Weak-run lane | Done locally | Current mixes are `10%`, `15%`, or `25%` combined blank/immortal exposure depending on recipe; exact rows recorded in `TRAINING_CONTROL.md`. |
| Bump weak-run immortal/blank exposure | Main after inspection | Blocked on live pointer audit | Do not mutate the shared control pointer until the exact assignment-writer/audit command path is selected for this v2 refresh control pointer. |

## P1

| Task | Owner | Status | Next action |
| --- | --- | --- | --- |
| Current arena UI marker | Main | Needs redeploy after next ids | Current code defaults to the all-v2 canary proof ids, not old loop18/v2real18. After choosing real restart tournament/rating ids, update shared contract and redeploy the Tournament Arena. |
| Fix Tournament Arena default | Main | Needs fresh-id redeploy | Old deployed verification for `curvy-loop18-live-main-20260514f` is historical. The next deployment should point only at the fresh all-v2 real tournament. |
| Fix GIF browser current batch | Main | Done and live-checked | Current code now uses `curvy-r18fresh-*` as the current-batch prefix, archives older run ids by default, and renames the destructive-looking UI action from `Delete` to `Archive`. The Modal image now installs `numpy` so the app can import CurvyZero env contracts. Live checks passed: root redirects to `category=current`, `/api/summaries?limit=5` returns `reload_error=null`, and rendered HTML lists current `curvy-r18fresh-*` runs. |
| Render path decision | Main + optimizer notes | Corrected | Current production target is CPU `cpu_oracle` `browser_lines + simple_symbols` policy observations. GPU `browser_lines + simple_symbols` is lab/profiling-only until trainer-visible contract parity passes; CPU `body_circles_fast + simple_symbols` is historical ablation/control only. H100 compute is not GPU observation rendering. |
| GIF safety policy | Main | Partly fixed | Tournament GIF generation defaults are now fast for new artifacts: `800` fps with `1ms` minimum frame duration, while keeping the existing `5` sampled GIFs per battle default. This does not rewrite already-generated GIF files; regenerate or rerate if old files must play faster. Still decide frame stride/sample caps before million-step tournament GIFs. |
| Old champion anchors | Old champion lane | Open | Find top prior full-sweep winners, verify refs still exist, inject into clean tournament after P0 rerate path is fixed. |
| Cleanup old arenas/apps/artifacts | Cleanup lane | Still active | Keep current deployed v2 trainer/tournament/GIF services. Old detached tournament apps `ap-ily3OHjnYXnun9616HKYGb` and `ap-svRQc9Y8SyxPu5wbJZ9fTT` were asked to stop and are still `stopping...` as of 04:03 EDT. Recheck and only start clean proof work once stale writers are gone or deliberately isolated by a new id. |
| Stop stale proof tournament workers | Main / cleanup | Needs finer target | Stale `curvy-looplive-fast-proof-20260515a` failures share app ids with later/current activity. Do not blindly `modal app stop`; find finer-grained cancellation or wait for tasks to drain, then recheck. |
| Direct tournament rating smoke | Main | Done | `curvy-looplive-directrating-smoke-20260515a` completed `3` games and wrote `latest.json` with `stable=true`. Game workers are alive. |
| Tiny intake scheduling smoke | Main | Done | `curvy-looplive-intake-smoke-20260515a` completed through intake: `1/1` pair, `3/3` games, `stable=true`. Intake is not generally dead. |
| Next behavior proof / launch | Main | Passed | Current all-v2 canary applied promoted sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0` at train iter `5061` with provider-ok env rows. Next: quantify survival and inspect the larger manifest; storage namespace choice is all-v2. |
| Refresh interval | Resume lane | Open | Do not change `50` to `1000/2000` until resume/refresh safety is proven. |

## P2

| Task | Owner | Status | Next action |
| --- | --- | --- | --- |
| Contract cleanup | Main later | Active | Shared CurvyTron constants are centralized in `src/curvyzero/contracts/curvytron.py`; old volume-name compatibility imports are purged. Continue removing local default copies when touching nearby code, but do not expand tests for trivia. |
| Better run names | Main later | Pending | New launches should use short human-readable names: opponent source, reward, noise, seed family, render lane. |
| App/dashboard hygiene | Cleanup lane | Pending | Keep only necessary apps and current arenas visible. |

## Rules For This Board

- If blocked, start a smaller honest proof in parallel.
- If a live experiment might be invalid, keep it only as smoke until the
  invalidating question is answered.
- If a task mutates live state, first write the exact intended change here.
- If a subagent returns a result, summarize it here or in the owning doc.
- If a task stops mattering, mark it dropped with the reason.
