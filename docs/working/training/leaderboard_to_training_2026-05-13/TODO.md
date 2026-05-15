# TODO: Coach / Tournament Task Board

Use this as the live checklist. Every task needs an owner, a status, and a next
action. Do not leave important state only in chat.

## P0

Current restart snapshot:

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
- Current Modal app cleanup is good enough for compute: the all-v2 trainer,
  tournament, and GIF browser apps are deployed; old non-v2 Curvy
  trainer/tournament deployments are stopped.
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
- We are now in the hardening phase. Do not launch a larger batch from vibes:
  rerun the broad local E2E-adjacent tests, keep the exact canary artifacts in
  `FULL_LOOP_PROOF.md`, and close or explicitly accept the remaining cleanup
  risks before a real batch.
- Latest local hardening bundle: `386 passed, 14 skipped`; ruff passed for
  touched trainer and poller tests. This found and fixed stale blank-slot
  fixtures plus eval-poller control-ref/nested-checkpoint reload issues.

| Task | Owner | Status | Next action |
| --- | --- | --- | --- |
| All-v2 namespace reset | Main | Done | Contract points every current CurvyTron app/Volume/Dict/Queue at `-v2` names. Exact v2 Volumes/Dicts/Queue were deleted and recreated at about 2026-05-15 14:08 EDT; the three Volumes pass `Volume.from_name(..., version=2).info()`; v2 trainer/tournament/GIF apps were redeployed at about 14:09 EDT; old non-v2 Curvy trainer/tournament deployments were stopped at about 14:10 EDT. |
| Fresh all-v2 deployed canary | Main | Passed | `curvy-e2e-allv2-canary-20260515a` launched on the v2 trainer app after copying one historical seed checkpoint into `curvyzero-runs-v2`; v2 intake/tournament `curvy-e2e-allv2-canary-live-20260515a` / `elo-e2e-allv2-canary-live-20260515a` completed `round-000003` with `18/18` games, `0` failures, and `stable=true`; promotion wrote assignment sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`; same running trainer applied it at train iter `5061`; later refresh stayed on that sha at train iter `5372`; env telemetry had `1836` provider-ok rows using it on the latest fetch. |
| Fresh all-v2 canary launch | Main | Passed | Poller `fc-01KRPDXH9FW4DNNMW41G9XC3EY`, trainer `fc-01KRPDXHDF4C5RS17DR1Z18EVX`. Keep this as the current launch template: all-v2 volumes/dicts/queue, `control:` refresh pointer, live run-id intake, explicit canary relaxations only for tiny proof promotion, and provider-ok env telemetry as the final proof. |
| Submitter v2 control-volume open | Main | Done | `scripts/submit_curvytron_survivaldiag_manifest.py` now uses `modal_volume_kwargs_for_name(...)` for direct refresh-pointer writes, so `curvyzero-curvytron-control-v2` is opened with `version=2`. Focused submitter/manifest/shared-contract launch bundle passed (`26 passed`) and ruff is clean. |
| Restart18 manifest fail-closed defaults | Main | Done locally | `scripts/build_curvytron_tonight18_manifest.py` now requires an explicit ratings snapshot, defaults to assignment mode, writes assignments and refresh pointers to `control:`, uses default refresh interval `2000`, and names fresh `curvy-r18v2-*` rows. Inline mixture mode now needs an explicit diagnostic refresh-off flag. Covered by the `26 passed` launch bundle and the `343 passed, 24 skipped` E2E-adjacent slice. |
| Submitter app-name guard | Main | Done locally | `scripts/submit_curvytron_survivaldiag_manifest.py` rejects stale or overridden app names unless they match each row's manifest app and the current shared-contract all-v2 trainer app. This blocks accidental submission of v2 rows into old apps. Covered by focused launch tests and ruff. |
| Restart18 diagnostic manifest dry-run | Main | Done; do not launch | Built `curvy-r18v2-dryrun-canarysource-20260515a` from the all-v2 canary leaderboard and dry-ran the submitter. It has the right control/refresh/app/surface/cadence fields, but the source has only `4` active rows and rank 1 is `iteration_0.pth.tar`, so it is wiring evidence only. |
| All-v2 source leaderboard inventory | Plato + Main | Done | Current v2 tournament storage has only `curvy-e2e-allv2-canary-live-20260515a` plus public leaderboard `e2e-allv2-canary-live-r3-20260515a`. No production-quality source snapshot exists in recreated v2 storage. Do not launch restart18 from existing v2 storage until a real source snapshot is created or old strong refs are rematerialized/rerated into v2. |
| Manifest checkpoint-ref existence audit | Main | Done locally | Added `scripts/audit_curvytron_launch_manifest_refs.py` plus tests. It collects initial-policy refs and frozen assignment/mixture refs, rejects mutable/control-prefixed checkpoint refs, and can check local or Modal existence. Focused audit tests passed and the canary dry-run manifest passed `--check-modal` against `curvyzero-runs-v2` with `4/4` refs present. |
| Fresh deployed E2E canary after cleanup | Main | Passed after repair | Short canary proved checkpoint -> live run-id intake -> rating -> leaderboard -> assignment -> pointer rewrite, but ended before same-trainer refresh. Long canary `curvy-e2e-clean-canary-long-20260515c` then proved the full loop: live tournament `curvy-e2e-clean-canary-long-live-20260515c` completed `2` pairs / `6` games / `0` failures; promotion sha `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30`; same trainer applied at train iter `2750`; `env_steps.jsonl` had `1107` rows using that sha with provider load OK. Post-refresh events stayed on the sha through train iter `4565`; old canary app is now stopped. |
| Stop invalidated v2 run | Main | Done | Invalidated v2real18 evidence is diagnostic only. The v2 durable objects have now been recreated fresh, so old v2real18 refs are gone from the active lane. Do not publish its rerate as restart evidence. |
| Restart gate for next training | Main | Unblocked by wiring proof; source snapshot still blocked | The recreated all-v2 loop is proven at canary scale, the focused launch/promotion/ref-audit regression bundle passed (`30 passed`), and the broader E2E-adjacent slice passed (`343 passed, 24 skipped`). Before a real batch, create a production-quality all-v2 source snapshot, then build/audit the exact restart18 manifest and choose fresh tournament/rating ids. |
| Production-quality promotion gate | Main | Open | Canary promotion used relaxed/provisional gates and picked `iteration_0.pth.tar` as champion. Before a real batch relies on a leaderboard, run or identify a production-shaped bounded rating with real active-row gates and a non-diagnostic assignment materialization. |
| Production source strategy | Rawls + Main | Recommended | Use `loop18-main-adaptive417` only as a candidate-selection artifact: top 100 active exact checkpoint refs, copy/rematerialize those files from old runs storage into `curvyzero-runs-v2`, then rerate under fresh v2 tournament/rating ids. Do not copy the old leaderboard as launch truth; do not start from canary as production source. |
| Survival improvement evidence | Main | Open | The all-v2 canary proves wiring only. Current survival evidence remains partial/noisy; do not claim learning improvement until eval/collector survival metrics improve cleanly over a meaningful window. |
| Background eval/GIF proof | Main | Open | The all-v2 canary proves training refresh, not broad background eval/GIF poller behavior. Keep that path disabled or separately smoke it before depending on it for a large launch. |
| Background eval poller with control refs | Main | Fixed and smoke-tested for current namespace | Live logs showed a poller could miss a newly written `control:` assignment file. Command resolution now reloads `/control` or `/runs` before reading prefixed assignment refs, and poller assignment resolution can reload nested checkpoint volumes without changing the active trainer refresh path. Focused validation: `85 passed, 3 skipped`; broad slice: `386 passed, 14 skipped`; ruff passed. Remote smoke against an old non-v2 canary assignment failed correctly because current trainer defaults read checkpoints from `curvyzero-runs-v2`; a current-namespace smoke `curvy-e2e-poller-v2-namespace-smoke-20260515a` then completed without assignment/checkpoint resolution errors. |
| Next manifest design | Main | Locally prepared | Current tonight18 builder uses `random_per_episode` by default, exposes fixed seats only as diagnostics, uses public `opponent_immortal`, gives every recipe at least `20%` blank/immortal pressure, includes higher-pressure variants, and does not reuse the old `5%` wall recipes. Keep this checked before launch. |
| Audit player perspective in training | Averroes + Main | Locally implemented | Trainer default is `learner_seat_mode=random_per_episode`; old `ego_player_index` config is rejected; focused regression covers both seats over reset. Rerun focused tests after the current slot cleanup. |
| Audit player perspective in tournament eval | McClintock + Main | Locally implemented | Tournament game specs default to balanced physical seating and per-checkpoint policy observation surfaces; focused tests cover balanced seats, seat-aware win counting, eval policy mode, and policy surface rejection. Rerun focused tests after the current slot cleanup. |
| Confirm live-policy no-op semantics | Main + perspective agents | Locally documented | Current action space is `left`, `straight`, `right`; `straight` is the explicit no-turn action. Treat this as satisfying no-op-in-turn-space unless the engine contract changes. |
| Purge hidden compatibility defaults | Main + Pascal | Locally fixed | Fresh training defaults to `random_per_episode`, old `ego_player_index` config is rejected, fresh mixture/assignment recipes use public `opponent_immortal`, blank/no-op entries must explicitly be immortal, episode selection derives runtime `opponent_death_mode`, and current launch/manifest/tournament defaults now read from `src/curvyzero/contracts/curvytron.py`. The old `curvytron_volume_names.py` shim is deleted; the tournament web default no longer points at invalid v2; fresh tournament specs no longer accept observation/source/generic render aliases. Latest focused validation: `79 passed`; ruff clean. |
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
| Fix GIF browser current batch | Main | Needs redeploy after launch prefix | Current code uses restart prefix `curvy-r18v2-*`, not invalidated `curvy-v2real18-*`. Redeploy once the fresh rows exist. |
| Render path decision | Main + optimizer notes | Corrected | Current production target is CPU `cpu_oracle` `browser_lines + simple_symbols` policy observations. GPU `browser_lines + simple_symbols` is lab/profiling-only until trainer-visible contract parity passes; CPU `body_circles_fast + simple_symbols` is historical ablation/control only. H100 compute is not GPU observation rendering. |
| GIF safety policy | Main | Open | Decide frame stride/sample cap before million-step tournament GIFs. |
| Old champion anchors | Old champion lane | Open | Find top prior full-sweep winners, verify refs still exist, inject into clean tournament after P0 rerate path is fixed. |
| Cleanup old arenas/apps/artifacts | Cleanup lane | Cataloged; no stop | Noether cataloged live apps and stopped none. Mixed app ids had both stale and current tournament evidence, so app-level stopping was unsafe. See `modal_stale_proof_cleanup_2026-05-15.md`. |
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
