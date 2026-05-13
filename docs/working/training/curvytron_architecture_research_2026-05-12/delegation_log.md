# Delegation Log

Purpose: keep parallel work visible. The main thread plans, synthesizes, and
decides; agents own bounded lanes and report concrete outputs.

## 2026-05-13 Gate-Cleanup Batch

Status: completed/superseded by the launch-readiness batch below. This batch
closed several traps: docs no longer present stale row lists as runnable, the
old stock manifest generator is blocked by default, and the status/export bridge
now has local coverage for the survivaldiag fields.

| Lane | Owner | Scope | Output Needed |
| --- | --- | --- | --- |
| Rich eval/status export | Parfit | patch status/export so `eval_checkpoints` carries reward components, bonus counts, terminal causes, action summaries, entropy, and eval health when available | done: local status/export tests passed; live snapshot still needed |
| Matrix docs cleanup | Kuhn | remove or quarantine stale launch-looking matrix rows and stale reward/opponent names | done: launch docs now mark stale sketches archival/non-runnable |
| Validation/observability audit | Sagan | read-only critique of which launch gates are cleared, partial, or blocked; verify action-observability artifacts | done: launch blocked; tiny canaries clear plumbing, not large matrix |
| Manifest generator audit | Mencius | read-only check of stock manifest generator against current reward, opponent, seed, render, and stochasticity truth | done: generator stale; main thread blocked it by default |

## 2026-05-13 Launch-Readiness Batch

Current phase: the ugly `v1b` first wave was stopped and preserved. The first
clean 300-row grouped launch, `curvy-survive-bonus-large-20260513a`, failed
because the grouped submitter omitted required trainer kwargs. The fixed
300-row launch, `curvy-survive-bonus-large-20260513b`, is now submitted through
one deployed Modal trainer app. Rows 1, 2, and 50 have trainer heartbeats,
progress files, iteration 0 evals, and GIF refs. Later sampled rows are still
startup/queued until proven otherwise.

| Lane | Owner | Scope | Output Needed |
| --- | --- | --- | --- |
| Local validation round 2 | Cicero | run focused and broad tests over reward, blank canvas, action repeat, status/export, eval curves, and manifest guard | exact commands, pass/fail, gates cleared or still partial |
| Survivaldiag manifest generator | Goodall | implement or precisely specify a new dry-run-only current manifest generator | schema, example rows, tests, remaining decisions |
| Matrix critique | Rawls | critique the staged overnight tensor and copy counts | staged block table at 50/100/200/300 scale and open questions |
| Feature-gap audit | Carson | inspect code/config/docs for remaining missing launch features | readiness table with file refs and next validation commands |
| Research-doc synthesis | McClintock | summarize historical gotchas that should shape the launch | concise facts/risks from the docs with refs |

Returned so far:

- Rawls: next matrix should be blank-canvas heavy, render-paired, stochasticity
  laddered, and repeat-heavy; scripted/random/frozen families stay gated until
  first-class and canaried.
- Carson: core reward/env/stochasticity are mostly ready; large-run blockers
  remain high-cap runtime canary, current manifest schema, live rich-status
  snapshot, and excluding non-wired opponent families.
- McClintock: historical lessons reinforce stock `train_muzero`, explicit
  opponent semantics, survival-over-outcome, Modal code snapshot awareness, and
  stale-tool cleanup.
- Cicero: local validation round 2 passed `526 passed, 14 skipped`; old
  manifest generator fails closed; eval tooling can read synthetic rich
  survivaldiag status fields.
- Goodall: new survivaldiag dry-run manifest generator exists with command
  artifacts and focused tests.
- Mill: manifest critique found that ancestor rows were too early, passive
  immortal was too large, names missed seed/compute detail, and preflight did
  not match the main stochastic lane closely enough. Main thread patched the
  generator to make ancestor rows gated only, shrink passive immortal, add
  blank-canvas extra repeats, encode training seed/compute in run ids, and
  record richer background eval/GIF fields.
- Nietzsche: opponent-feature continuity audit confirmed that blank canvas is
  the only clean first-wave opponent anchor; passive immortal is a dirty
  control; scripted wall-avoidant and random/checkpoint opponents remain
  second-wave until wired and canaried.
- Dirac: matrix critique agreed with the earlier blank-canvas-heavy 50-row
  shape and flagged stale docs that still treated ancestor rows as executable;
  main thread patched the active docs. This was superseded by the clean 300-row
  large-ready manifest.
- Singer: manifest/launcher audit found no hard flag/compute/background-eval
  blocker in the default generated commands. It flagged ambiguous
  `expansion_gate` metadata on executable sentinel rows; main thread renamed
  that field to `row_note` and regenerated artifacts.
- Einstein: feature-status audit confirmed the clean implemented core is
  blank-canvas no-op, survival-plus-bonus/no-outcome reward, held-action repeat,
  and local eval/GIF/status plumbing. Passive immortal, moving no-trail,
  scripted wall-avoidant, and random/checkpoint expansions remain gated.
- Main thread: local validation round 3 passed after the manifest/docs cleanup:
  `20 passed`, `209 passed`, `67 passed, 10 skipped`, and ruff clean.
- Volta: scripted `proactive_force_field` is a clean later-wave plan, but it is
  tooling-only today. It needs first-class source-state trainer wiring,
  metadata, tests, and canaries. It should not block the clean 300-row
  blank-canvas matrix.
- Schrodinger: doc-consistency pass found stale active-doc wording around the
  exact manifest, reward name, ancestor/frozen controls, and blank-canvas
  status. Main thread patched those docs.
- Kepler: minimal remaining live gates are one matched blank-canvas high-cap
  pair plus a real live status snapshot; if passive dirty rows stay, they need
  their own matched exact-lane canary or removal. Main thread launched blank,
  passive dirty, and sim16 matched pairs as tiny live canaries.
- Main thread: those first high-cap canaries were manually stopped. Modal app
  state is stopped, but volume heartbeat/poller files can still say `running`
  after a forced stop. The useful finding is that `max_train_iter` is not a
  strict tiny-canary cap because LightZero can run many learner updates inside
  one collect/update block before checking it.
- Darwin: active read-only critique lane for the canary cap semantics and any
  missing code/doc patches.
- Russell: active read-only doc audit lane for stale wording caused by the
  canary cap finding.
- Main thread: patched the launcher so `--stop-after-learner-train-calls`
  installs the existing learner-call cap in train mode, then launched a matched
  blank-canvas fast/browser strict-stop retry.
- Main thread: strict-stop live canaries now pass for passive immortal
  fast/browser and sim16 fast/browser. Combined with earlier blank-canvas
  fast/browser strict-stop canaries, the first-wave exact-lane mechanics gate is
  cleared: completed train status, one checkpoint, completed poller, one eval,
  one GIF, reward components, bonus fields, terminal cause, action histogram and
  entropy, eval health, and GIF health. Passive immortal remains a dirty
  control, not a clean opponent design.
- Epicurus: Modal dashboard cleanup is possible with one deployed app and many
  `Function.from_name(...).spawn(...)` calls, but the submitter must preserve
  the local entrypoint's checkpoint-poller spawn. Recommendation: do not switch
  cold for the overnight batch; canary a grouped submitter separately.
- Lovelace: active follow-up lane to independently verify the single-app Modal
  pattern, local examples, and any tiny safe proof. Do not touch the live
  survivaldiag batch while this lane runs.
- Main thread: fixed survivaldiag manifest attempt IDs to stay under the
  run-management 96-character limit. A partial failed dispatch from the old
  long-attempt-id commands was stopped and should not be used for learning
  claims.
- Main thread: launched `survivaldiag-v1b-20260513h` as a 50-row stock
  `train_muzero` batch after patching train/poller timeouts for overnight use.
  Early status checks show all 50 rows `train_status=running`; this is liveness
  only, not learning signal.
- Main thread: later live status/curve sweeps still show all 50 rows running.
  By 2026-05-13 03:13 EDT, 18 rows had written `iteration_5000` checkpoints and
  at least 12 had `iteration_5000` eval manifests. Row 1 live manifest directly
  verified `eval_reward_variant`, `model_reward_variant`, and
  `env_reward_variant` are all `survival_plus_bonus_no_outcome`; the old
  critique about eval defaulting to `auto` is resolved for this launched batch.
- Main thread: final early sweep at 2026-05-13 03:22 EDT still showed all 50
  rows running. Forty-eight rows had reached at least `iteration_5000`, twelve
  had reached `iteration_10000`, and the checkpoint pollers were still running.
  Stop intervening unless a later sweep shows concrete failures.
- Pascal: speed display bug is plain. The GIF browser shows `speed unknown`
  when `train/progress_latest.json` is absent. The trainer should write that
  file on each LightZero checkpoint save with at least `iteration` and
  `elapsed_sec`.
- Dewey: earlier matrix critique proposed a 50-row clean repeat wave. That was
  superseded by the clean 300-row large-ready manifest.
- Ramanujan: frozen checkpoint opponents exist in stock `--mode train` and the
  known old/mid/recent refs exist. Treat them as a small side control after one
  exact canary, not as a main learning lane.
- Main thread: active replacement worker is checking whether scripted
  wall-avoidant opponent variants can be wired as a small first-class trainer
  lane. This must not block the clean 300-row launch.
- Huygens: naming critique completed. Adopted plain names:
  `curvy-survive-bonus-<opponent>-<render>-<stochasticity>-<compute>-rNNN-sSEED`.
  Dropped opaque tags like `survbonusnoout`, `blanknoop`, `armed`, and
  `l4t4c40` from run IDs.
- Hilbert: grouped Modal launch critique completed. The clean path is to deploy
  one trainer app and use `modal.Function.from_name(...).spawn(...)` for many
  function calls. Important detail: spawn the checkpoint poller and then the
  train function for each row.
- Chandrasekhar: large-matrix critique completed. Main trusted ready lane is a
  large blank-canvas batch with passive dirty controls and small compute
  sentinels. Non-ready lanes stay gated.
- Meitner: proactive wall-avoidant opponent was implemented and locally tested,
  but it is still out of the 300-row launch until remote e2e canary.
- Feynman: checkpoint cadence readout completed. `save_ckpt_after_iter=5000`
  gave roughly one checkpoint every 10 minutes; next batch uses `15000` for
  roughly 30 minutes.
- Main thread: stopped the ugly 50-row batch, generated
  `curvy-survive-bonus-large-20260513a` with 300 rows, added grouped submitter,
  added `progress_latest.json` checkpoint-speed writer, and paused before
  launch per user instruction.
- Erdos: completed Modal volume cleanup after the failed 300a launch. Preserved
  the agreed 50-row v1b artifacts and deleted the broken 300a direct run roots.
  Report:
  `artifacts/local/curvytron_modal_cleanup_reports/modal_volume_cleanup_20260513_v1b_only_final.md`.
- Main thread: patched grouped train kwargs, added submitter preflight tests,
  redeployed the trainer app and GIF browser, launched
  `curvy-survive-bonus-large-20260513b`, and verified early trainer/eval/GIF
  artifacts.
- Euler: cadence recheck corrected the working assumption. `15000` produced
  checkpoints in about 28-31 minutes on sampled rows that reached
  `iteration_15000`; missing later checkpoints on some rows are a liveness or
  row-speed question, not enough evidence by themselves. Next mixture working
  default is `save_ckpt_after_iter=10000`.
- Leibniz: audited recent-checkpoint mixture support. Current trusted path has
  one static frozen checkpoint opponent only; it does not have a checkpoint
  pool, per-episode weighted mixture, live refresh, or same-run lagged opponent.
- Linnaeus: active implementation lane for episode-level opponent mixture in
  the trusted stock path. Must keep it separate from the running 300b batch.
- Main thread: mixture plan critique updated after the latest canary failure.
  The first full `curvy-mix2` launch should not use all 228 review rows by
  default. Keep the core sim8/C32/B32 rows with paired fast/browser render and
  rep0/repM/repH; hold sim16/C64/B64 sentinels until the corrected canary
  reaches `train_muzero`. The canary must prove the readiness relation is
  `learner_vs_weighted_episode_opponent_mixture`, not
  `learner_vs_fixed_straight`.
- Main thread: `curvy-mix2-canary-20260513a` failed before `train_muzero`
  because command metadata still said fixed-straight while env config said
  weighted episode opponent mixture. Patched the trainer so command and env
  config use one helper for the relation, and patched the readiness summary to
  allow mixture runs. Focused pytest, ruff, py_compile, and `git diff --check`
  passed. The trainer was redeployed, and fresh canary
  `curvy-mix2-canary-20260513b` was launched with six `curvy-mix2b-*` rows.
  Early status shows train roots and pollers exist. A browser scripted row has
  `iteration_0`, `progress_latest.json`, raw/collect GIFs, and selected mixture
  fields in the GIF summary: `scripted`, weight `50`, index `1`, age label
  `scripted_wall_avoidant`. Keep monitoring the remaining rows and first `k10`
  checkpoint before full launch.
- Main thread: later status for `curvy-mix2-canary-20260513b` shows five of six
  rows have reached `iteration_0`; two rows have eval summaries; multiple rows
  have raw and `collect_t1` GIFs. One row still had only a train root in the
  latest sample. The 180-row `curvy-mix2-core-20260513a` grouped-submit dry-run
  passed and targets one deployed trainer app. New follow-up lanes: Gibbs owns
  fidelity/cadence audit; Lorentz owns mixture-batch critique. Neither should
  launch or kill runs.
- Gibbs: cadence/fidelity audit found no big render-only split in sampled 300b
  rows. Browser was about 0-2 minutes slower per 15000 iterations than matched
  fast rows; batch64/heavier rows and trainer-liveness ambiguity are larger
  suspects. No docs edited.
- Lorentz: mixture-batch critique recommended not launching the 180-row core as
  is. Passive rows should remain canary/dirty-control evidence only. Main
  thread added a recipe allow-list to the manifest builder, tested it, and
  generated the pruned 156-row `curvy-mix2-clean-20260513a` launch candidate.
  Grouped-submit dry-run passed; no runs launched yet.
- Main thread: `curvy-mix2-canary-20260513b` reached `iteration_10000` on all
  six rows. All six `iteration_10000` selfplay summaries were `ok=true`, had
  raw and `collect_t1` GIF variants, and recorded selected mixture component
  fields. Observed selected entries included `blank`, `mid`, and `recent`.
  First `k10` cadence was about 31-38 minutes. Local focused tests, ruff,
  py_compile, grouped-submit dry-run, and `git diff --check` passed for the
  pruned launch candidate.
- Main thread: launched `curvy-mix2-clean-20260513a` at 2026-05-13 07:49 EDT.
  It has 156 rows, no passive recipes, no heavy sentinels, paired fast/browser
  render, repeat levels `rep0`/`repM`/`repH`, and `k10` checkpoints. The launch
  artifact records all 156 train call IDs and all 156 poller call IDs in the
  single deployed trainer app.
- Main thread: launched `curvy-mix3-currentckpt-20260513a` at about 2026-05-13
  09:31 EDT. It has 300 rows, uses fresh recent/mid/old refs from
  `curvy-survive-bonus-large-20260513b`, keeps passive out, pairs
  `body_circles_fast` and `browser_lines`, and alternates render launch lead.
  Fresh 10:13 EDT startup read: 187 train roots, 180 running pollers, 36 live
  trainer heartbeats, 35 checkpoint rows, 3 rows at `iteration_10000`, 8 eval
  rows, and 26 GIF rows. This is liveness/startup evidence, not a learning
  ranking. Follow-up 10:22 EDT read: 190 train roots, 186 running pollers, 33
  rows at `iteration_10000`, 1 row at `iteration_20000`, 28 eval rows, and 34
  GIF rows.
- Main thread: found a checkpoint eval/GIF Modal volume commit storm. Patched
  the train/eval code to reduce checkpoint eval commits and to use retry/backoff
  for trainer, eval, and GIF volume commits. Trainer and eval apps were
  redeployed around 10:02-10:03 EDT. New workers emit labelled retry logs;
  pre-redeploy workers can still fail at old direct commit lines until they
  drain.

## 2026-05-13 Opponent Mixture Batch

Purpose: prepare the next launch lane while the 300b batch runs.

Source doc:
[opponent_mixture_batch_plan_2026-05-13.md](opponent_mixture_batch_plan_2026-05-13.md).

Main rules:

- compact base grid, not one fixed base;
- first draft was 100 rows, 20 recipes x 5 seeds, but it is now stale;
- main recipes use a recent frozen checkpoint for about 50% of episodes;
- the remaining 50% varies older checkpoints, scripted opponent, passive
  dirty control, and blank canvas;
- include paired fast/browser render rows for core recipes and a few named
  search/stochasticity probes;
- opponent component is chosen once per reset/episode, not per step;
- exact checkpoint refs go into the manifest; no `latest` pointer inside the
  trainer;
- do not launch until local tests, grouped dry-run, corrected cadence, and
  remote canaries pass.
- relation-mismatch failures before `train_muzero` are launch blockers, even if
  poller artifacts exist.

## 2026-05-12/13 Exploration Batch

Status: completed/superseded by the gate-cleanup batch above. Keep this as the
paper trail for why the current matrix direction changed.

| Lane | Owner | Scope | Output Needed |
| --- | --- | --- | --- |
| Manual old-run analysis | agent lane | inspect old run artifacts by hand; identify signal, gaps, and misleading takeaways | concise readout with file/run refs |
| Tooling validation on old runs | agent lane | run/check eval tooling against old artifacts; no trainer edits | pass/fail notes, mismatches, limitations |
| Future tensor critique | agent lane | critique whether future tensor/feature logging is needed; no implementation | must-have vs nice-to-have signals |
| Immortal-opponent code check | agent lane | inspect env/config path for immortal opponent; no code changes | file refs, risks, fastest safe path |

## Main Thread Owns

- Keep [current_source_of_truth.md](current_source_of_truth.md) accurate.
- Keep [todo.md](todo.md) current.
- Merge subagent results into the docs.
- Decide which old-run conclusions are trustworthy.
- Decide whether any later matrix work is justified after old-run analysis.
- Avoid letting stale docs or old custom two-seat runs drive decisions.
- Keep lane boundaries tight and reject scope creep.
- Record final decisions; agents provide inputs, not launch authority.

## Expected Follow-Ups

- If manual old-run analysis finds a real signal, cross-check it with tooling
  before promoting it.
- If tooling disagrees with hand inspection, document the mismatch before
  using the summary.
- If tensor critique finds a must-have missing signal, keep it as future
  instrumentation work, not a launch prerequisite.
- If immortal-opponent code check finds trainer coupling, stop at the risk note
  and do not patch code in this phase.
- Send follow-ups when the main worldview changes. Do not let workers finish
  against stale assumptions.
- Follow-ups should be narrow: one correction, one priority update, or one
  explicit blocker question. Avoid dumping the whole main thread into a worker.

## Recent Follow-Ups

- Hypatia: keep eval checks metric-agnostic; old v1d uses outcome curves, next
  survivaldiag uses survival/reward curves; be cautious about false negatives.
- Poincare: keep immortal-opponent work at code-check scope in this phase; no
  trainer edits and no no-trail scope creep.
- Main thread: old-run analysis and tooling validation come before any renewed
  matrix discussion.

## 2026-05-13 Opponent Design Lanes

| Lane | Owner | Scope | Status |
| --- | --- | --- | --- |
| Immortal behavior audit | Faraday the 2nd | pin down wall/trail/body behavior for `opponent_death_mode=immortal` | done: current behavior is passive death immunity; no reflection/containment/no-trail |
| Future tensor seed/copy cleanup | Gibbs the 2nd | document repeat-copy axis and separate seed meanings | done: docs updated |
| Scripted wall-avoidant opponent | Maxwell the 2nd | design/test simple wall-avoidant policy through real env where feasible | done: proactive M=20 survived 1024-step real-env probes; reactive contact proxy failed; follow-up iteration assigned to Anscombe |
| Blank/no-trail opponent lane | Meitner the 2nd | inspect minimal fake opponent with no movement/trail/death pressure | done: added `blank_canvas_noop_opponent_lane.md`; recommends wrapper runtime mode |
| Reward wiring audit | Popper the 2nd | check whether survival-plus-bonus/no-outcome exists and what is needed | done: initially found missing variant; later implemented and e2e-gated |
| Random opponent wiring audit | Mencius the 2nd | check accepted opponent kinds, random learned opponent options, seed fields | done: use immutable random-init/iteration_0 checkpoints first; random learned source-state opponent is not first-class |
| Aggressive matrix critique | Linnaeus the 2nd | critique scale and block structure for the next batch | done: large staged blocks recommended; see `aggressive_matrix_scale_plan.md` |
| Wall-avoidant/reflection iteration | Anscombe the 2nd | improve probe/tooling and test corrected reflection/avoidance variants | done: proactive force-field M=20 recommended; reflection-like variants remain secondary |
| Old-run projection refinement | Sartre the 2nd | project old v1d runs along important knobs with simple tables | done: `v1d_axis_projection.md` updated |
| Reward-gate implementation | Huygens the 2nd | add `survival_plus_bonus_no_outcome` reward path/tests; no opponent edits | done: local tests pass; e2e support mismatch found and fixed by main thread |
| Matrix adversarial critique | Pasteur the 2nd | challenge 100/200/300-run design and missing measurements | done: blank canvas should anchor; repeats/action telemetry under-scaled; passive immortal/ancestor/compute over-sweep risk |
| Blank-canvas implementation design | Epicurus the 2nd | read-only exact implementation plan for `blank_canvas_noop` | done: recommends `disabled_player_mask`, render masking, and focused wrapper tests |
| Eval-curve tooling | Harvey the 2nd | extend local curve parser/schema for future survivaldiag metrics | done: parser now accepts reward/bonus/terminal/action/eval-health fields if upstream JSON carries them |
| Blank-canvas implementation | Feynman the 2nd | implement `opponent_runtime_mode=blank_canvas_noop`, disabled player mask, render masking, Modal plumbing, and tests | done: local tests pass; e2e blank-canvas canary passes after shared support fix |
| Hand-designed opponent variants | Aristotle the 2nd | extend bounded probe/design lane for legal-action wall-avoidant opponents | done: proactive force-field, lazy weave, jitter force-field recommended first; wall follower later stress row |
| Prelaunch validation audit | Singer the 2nd | bounded audit of reward, blank canvas, eval curves, wall probe | done: audit doc added; recommended canaries now passed for tiny e2e scope |
| Canary artifact inspection | Cicero the 2nd | inspect successful e2e canary artifacts on Modal volume | done: blank-canvas body-circles artifacts validated; matched canaries also passed |

## Main Thread Runtime Update

- 2026-05-13: patched dry-run surface validation to normalize JSON list/tuple
  support ranges.
- Modal dry run now returns `ok=true` for
  `source_state_fixed_opponent` plus `survival_plus_bonus_no_outcome` at
  `source_max_steps=65536`.
- Focused plumbing tests pass:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py`
  -> `38 passed, 1 skipped`.
- Local process sweep found no visible wall-probe or Python simulation process.
  Active worker was told not to run broad local simulation sweeps.
- 2026-05-13: first real e2e canary failed on incoherent LightZero support
  sizing (`5` reward bins vs `601` target bins). Main thread changed to the
  shared capped `model.support_scale=300` required by LightZero v0.2.0, so both
  reward and value heads are effective `601` bins.
- 2026-05-13: patched e2e blank-canvas canary
  `curvytron-prelaunch-e2e-blank-20260513b` completed with stock
  `train_muzero`, telemetry rows, checkpoints, eval/inspection, and GIF jobs.
- Tiny stock `train_muzero` e2e canaries have passed for
  `blank_canvas_noop/body_circles_fast`, `blank_canvas_noop/browser_lines`, and
  `normal/body_circles_fast`.

## Thread Limit Note

Subagent spawn hit the thread limit, so current work is assigned to existing
agents rather than new threads. Close or reuse completed agents before opening
more.

## 2026-05-13 Opponent Mixture Runtime Update

- Linnaeus finished local mixture eval/GIF wiring. Background checkpoint eval
  and GIF jobs now carry `opponent_mixture_spec` and record selected component
  fields.
- Main thread added
  `scripts/build_curvytron_opponent_mixture_manifest.py`.
- Current artifacts:
  - canary:
    `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix-recent-canary-20260513a.json`
  - 100-row batch:
    `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix-recent-20260513a.json`
- The old dense-run checkpoint refs were missing on the Modal volume. The
  manifests now use preserved v1b immutable refs:
  `iteration_20000.pth.tar`, `iteration_10000.pth.tar`, and
  `iteration_0.pth.tar`.
- Local gates passed:
  - `uv run pytest tests/test_curvytron_opponent_mixture_manifest.py tests/test_opponent_mixture.py tests/test_curvytron_survivaldiag_submitter.py -q`
    -> `19 passed`
  - progress/speed writer regression test after `SaveCkptHook` patch:
    `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py -q`
    -> `45 passed, 1 skipped`
- Planck: speed UI root cause is stale `progress_latest.json`. The trainer now
  writes `event="checkpoint"` and refreshes progress from the LightZero
  `SaveCkptHook` checkpoint path. Redeploy is required before future runs pick
  this up.
- Galileo: fixed-base 100-row mixture draft is too narrow. Next manifest should
  use `curvy-mix2`, `save_ckpt_after_iter=10000`, a 228-row grid, and readable
  baseline tokens such as `rf-s8-c32-l32-repM-k10`.
- Ptolemy: cadence/fidelity audit agrees on one global `k10`; no separate
  browser cadence. It cautions against broad sim16/C64/B64 sentinel rows for the
  next mixture launch, especially browser sentinels. Treat iteration-0-only rows
  as a liveness class: queued, crashed, or slow/stalled must be distinguished.
- Jason: implemented `curvy-mix2` manifest builder and tests. The generated
  batch has 228 rows and the generated canary has 6 rows. Local checks and
  grouped submitter dry-runs passed.
- Main thread checked the first mixture canary FunctionCall IDs. All three
  train calls failed immediately with `_run_visual_survival_train() got an
  unexpected keyword argument 'opponent_mixture_spec'`; the pollers kept running,
  which is why only poller roots existed. Corrective action: redeploy trainer
  after the current local patches, then launch a corrected `curvy-mix2` canary.
  The three stale canary poller calls were cancelled.
  - focused mixture/env tests earlier -> `51 passed`
  - ruff, py_compile, and `git diff --check`
  - grouped submitter dry-run for canary and first 5 batch rows
- Next action: monitor the fresh 6-row `curvy-mix2b` canary for trainer files,
  `called_train_muzero=true`, checkpoint poller files, eval/GIF artifacts, and
  selected mixture component fields.
- Main thread critique of the 180-row core manifest was superseded by Lorentz
  and the pruned manifest. After the canary clears, the recommended first full
  mixture launch is now `curvy-mix2-clean-20260513a`, not the full 180-row core.
  Keep 50%-recent non-passive main recipes, keep non-passive controls, keep
  paired fast/browser render, and use `save_ckpt_after_iter=10000`.

## 2026-05-13 Mixture Launch Recommendation Update

- Current recommendation after the canary clears: launch
  `curvy-mix2-clean-20260513a`, not the default `full` scope and not the
  passive-containing 180-row core artifact.
- Exact shape: 156 rows = 7 main recipes x 2 render modes x 3 repeat settings
  x 3 seeds, plus 5 controls x 2 render modes x 3 repeat settings x 1 seed.
- Keep paired `body_circles_fast` and `browser_lines` rows. Fast is the speed
  lane; browser is the higher-fidelity anchor. Use one `k10`
  `save_ckpt_after_iter=10000` cadence for both.
- Drop passive rows from the first launch: `r50-pass50` and `pass100` remain
  canary/dirty-control evidence only.
- Hold sim16/C64/B64 sentinels for a second wave. The first large mixture batch
  should keep sim8/C32/B32 fixed.
- The manifest builder currently defaults to `full`; pass `--batch-scope core`
  and the recipe allow-list explicitly when preparing the launch artifact.

## 2026-05-13 Post-Launch Monitoring And Critique

- `curvy-mix2-clean-20260513a` launched at 2026-05-13 07:49 EDT with 156 train
  call IDs and 156 poller call IDs in the single deployed trainer app.
- First all-row status sweep: 56 rows had `iteration_0`, 83 had trainer
  heartbeat, 128 pollers were running, and 17 train roots were still absent.
- Render split from that sweep: 34/78 fast rows and 22/78 browser rows had
  `iteration_0`. This is not clean render-speed evidence because the manifest
  orders fast rows before browser rows inside each recipe.
- Main-thread rule for the next check: compare matched fast/browser rows at the
  first `k10` checkpoint, not just startup status.
- Pasteur: investigate whether checkpoint cadence is actually split by render
  fidelity or by startup order / other knobs. Output should be facts and a
  small doc note if useful.
- Bacon: critique the next mixture matrix. Output should be a simple proposed
  grid over baseline knobs, mixture distribution, render split, repeats, seeds,
  and cadence, plus riskiest assumptions.
- Pasteur returned: first render split is launch-order confounded. Valid read
  must use matched fast/browser pairs at `k10`; future manifests should
  interleave or shuffle render order.
- Bacon returned: candidate next matrix is 300 rows: 180 main mixture rows,
  60 pure controls, 60 narrow compute probes. Keep passive out, keep paired
  renders, use readable names, and decide `k7500` versus `k10000` from matched
  checkpoint cadence.
- Main thread added `scripts/analyze_curvytron_mixture_status.py` to turn a
  manifest plus run-status JSON into matched fast/browser cadence summaries.
  First JSON read showed 25 rows at `iteration_10000` and 6 matched pairs at
  `iteration_10000`; browser was not obviously slower in those six pairs.
- Main thread corrected the analyzer: it now uses checkpoint file mtimes rather
  than latest progress elapsed seconds. Latest true `k0 -> k10` read:
  38 matched pairs, fast median about 1285 sec, browser median about 1395 sec,
  browser-minus-fast median about 117 sec.
- Main thread patched run-status JSON to expose latest eval fields, checkpoint
  mtimes, and GIF selected mixture component fields. Focused status/analyzer/
  manifest tests pass.
- Turing: owns a draft next-wave manifest profile in
  `scripts/build_curvytron_opponent_mixture_manifest.py` and
  `tests/test_curvytron_opponent_mixture_manifest.py`. Goal is 300 rows with
  balanced render launch order and no Modal launch.
- Turing returned: implemented `--profile next-wave`, default
  `curvy-mix3-nextwave-20260513a`, 300 dry-run rows, no passive rows, matched
  fast/browser pairs with alternating launch lead. Main thread generated the
  artifact and grouped-submit dry-run passed for all 300 rows.
- Archimedes returned: eval/GIF artifacts are appearing. Latest JSON showed
  39 rows with eval manifests and 100 rows with at least one GIF artifact.
  `background_poller_completed_count=0` is expected while pollers are still
  running because that count is written when the poller exits and joins spawned
  jobs. Use artifact counts and latest GIF/eval checkpoint fields for live
  health.

## 2026-05-13 Current-Checkpoint Mix3 Launch

- Gauss audited GIF browser plumbing. Findings: the browser lists per-row
  `run_id`s, not matrix names; `curvy-survive-bonus-large-20260513b` and
  `curvy-mix2-clean-20260513a` have picker flags and GIF summaries; exact API
  checks show both `raw.gif` and `collect_t1.gif`. `speed unknown` means missing
  or unreadable `progress_latest.json`, not a GIF failure.
- Anscombe audited `curvy-mix3-nextwave-20260513a`. Findings: the 300-row
  shape is good and names are readable, but the stale v1b checkpoint refs should
  be replaced if the current survival batch has healthy checkpoints.
- Main thread closed that gate: all 300 `curvy-survive-bonus-large-20260513b`
  rows are running with progress, checkpoints, evals, and GIFs. The next wave
  now uses current checkpoint refs from
  `curvy-survive-bonus-blank-fast-light-base-r063-s1111121`:
  `iteration_105000` as recent, `iteration_60000` as mid, and `iteration_0` as
  old.
- Generated `curvy-mix3-currentckpt-20260513a`: 300 rows, 180 main rows,
  60 controls, 60 compute probes, no passive rows, paired fast/browser renders,
  alternating render launch lead, and substantial recent-checkpoint weight in
  every main recipe.
- Dry-run and focused tests passed, then the grouped launch was submitted at
  about 2026-05-13 09:31 EDT through
  `curvyzero-lightzero-curvytron-visual-survival-train`.
  Launch artifact:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_launch.json`.
