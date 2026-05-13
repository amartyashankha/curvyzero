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

Current phase: the ugly `v1b` first wave was stopped. The active launch target
is the clean 300-row `curvy-survive-bonus-large-20260513a` manifest, submitted
through one deployed Modal app. Launch is paused until the required 30-minute
prelaunch sleep completes.

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
