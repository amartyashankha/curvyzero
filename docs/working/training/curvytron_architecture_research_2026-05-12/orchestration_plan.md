# Orchestration Plan

Purpose: keep the CurvyTron exploration work split into clean lanes. The main
thread plans, synthesizes, and decides. Agents own bounded lanes with narrow
outputs and clear stop points.

Current phase: monitor the live batches and keep the paper trail current. The
ugly 50-row batch was stopped and preserved. The first clean 300-row launch,
`curvy-survive-bonus-large-20260513a`, failed because grouped `train_kwargs`
omitted required trainer settings. That bug is patched, tested, and guarded by
submitter preflight.

The current run catalog lives in
[run_inventory_2026-05-13.md](run_inventory_2026-05-13.md). Read that before
paring down or deleting anything; it separates current batches, stale canaries,
preserved old rows, and dry-run-only manifests.

The fixed 300-row batch is `curvy-survive-bonus-large-20260513b`. It is healthy
background training now: all rows have trainer-owned files, checkpoints, evals,
and GIF artifacts. Keep it running as a survival diagnostic and as the source
of frozen checkpoints for mixture runs.

The first full mixture batch, `curvy-mix2-clean-20260513a`, is also healthy
enough to monitor in the background. It has reached many `k10` checkpoints and
has eval/GIF artifacts, but recipe ranking should wait for more eval coverage.

The current next wave is `curvy-mix3-currentckpt-20260513a`. It was launched
through the same single trainer app using recent/mid/old checkpoint refs from
`curvy-survive-bonus-large-20260513b`. The active gate is monitoring startup:
trainer roots, `iteration_0`, first eval/GIF artifacts, and first `k10`
checkpoints. Fresh 2026-05-13 10:13 EDT read shows 187 train roots, 36 live
trainer heartbeats, 35 checkpoint rows, 3 `k10` rows, 8 eval rows, and 26 GIF
rows. Follow-up 10:22 EDT read shows 190 train roots, 33 `k10` rows, 1 `k20`
row, 28 eval rows, and 34 GIF rows. Treat this as moving startup, not full
maturity.

## Active Lanes

| Lane | Owner | Output | Status |
| --- | --- | --- | --- |
| Feature status audit | Einstein | read-only table of actual implemented opponent/reward/stochasticity/observability features, tests, launch stance, and gaps | done |
| Manifest/launcher compatibility audit | Singer | verify generated command flags, compute values, dry-run gating, background eval/GIF flags, and row counts against the launcher | done |
| Matrix critique | Dirac | critique whether the current staged batch aligns with user priorities and avoids stale or over-wide axes | done |
| Speed display bug | Pascal | why the web UI says `speed unknown` and smallest fix | done: trainer must write `train/progress_latest.json` on checkpoint save |
| Clean 300-row manifest | main thread | patch full grouped train kwargs, regenerate, test, and relaunch fixed manifest | done: `curvy-survive-bonus-large-20260513b` submitted |
| Grouped Modal submitter | main thread | submit rows into one deployed app with poller+train calls, not one app per row; preflight must catch missing train kwargs before launch | done for 300b, mix2-clean, and mix3-currentckpt |
| Opponent variant wiring | worker lane | make scripted wall-avoidant opponent first-class if small; otherwise report blocker | done for current proactive wall-avoidant use; later variants optional |
| Ancestor controls | Ramanujan | check whether frozen checkpoint controls are mechanically ready | done: path exists; use only after exact tiny canary |
| Recent checkpoint mixture audit | Leibniz | read-only map of current and historical support for recent/frozen/checkpoint mixture opponents | done: current trusted path has one static frozen checkpoint only |
| Opponent mixture implementation | Linnaeus | implement episode-level mixture support in the trusted stock path; tests/docs only until canaries pass | done; mix2 canary and clean launch proved remote plumbing |
| Opponent mixture manifest/canary | main thread | corrected canary and full batch manifest with cadence 10000 and compact base grid | done for mix2-clean; mix3-currentckpt launched |
| Checkpoint cadence/fidelity audit | Euler/Ptolemy | split checkpoint timing by fast/browser and by heavy rows; recommend cadence | active: use k10 for mixtures; browser is modestly slower, not a separate-cadence blocker |
| Next mixture grid critique | Galileo | propose readable mixture/baseline grid and naming before launch | done for mix3-currentckpt; reopen after readout |
| Mix3 manifest prep | Turing | draft 300-row next-wave manifest profile with balanced fast/browser ordering; no launch | done; launched as `curvy-mix3-currentckpt-20260513a` |
| Eval/GIF status semantics | Archimedes | check whether current artifacts are healthy and whether status-reader fields are misleading | active: use eval/GIF counts and checkpoints, not poller completion count |
| Speed display bug | Planck | why progress/speed stays unknown or stale | done for future runs; keep monitoring row-level `progress_latest.json` |
| Volume cleanup | Erdos | delete old broken direct-run roots, preserve only the agreed v1b batch and current live batch | done for broken 300a/old direct roots |
| Artifact commit storm | main thread | reduce Modal volume commit pressure from checkpoint eval/GIF workers and redeploy | patched and redeployed; old workers may still log old direct-commit `DataLossError` until they drain |

Recently closed:

- Parfit: local status/export bridge landed for reward, bonus, terminal,
  action, entropy, failure-rate, and eval-health fields in `eval_checkpoints`.
- Kuhn: launch-facing matrix docs were cleaned so stale rows are marked
  archival/non-runnable.
- Sagan: launch readiness remains blocked; tiny canaries clear plumbing, not
  the large matrix.
- Mencius: old stock manifest generator was stale; main thread patched it to
  fail closed unless `--allow-historical-matrix` is passed.
- Cicero: local validation round 2 passed `526 passed, 14 skipped`.
- Goodall/Mill: current dry-run survivaldiag manifest generator exists and was
  corrected after critique; it emits 50 executable review rows and 10 gated
  specs, with `current_launch_approved=false`.
- Rawls/Carson/McClintock and later critiques: blank canvas remains the anchor;
  passive immortal is a dirty control; scripted/random/checkpoint opponents
  stay gated until wired and canaried.
- Einstein/Singer/Dirac: feature status, launcher compatibility, and matrix
  shape were independently re-audited. No default manifest/launcher blocker was
  found; docs and manifest metadata were patched to remove row-shape drift and
  executable-row gate ambiguity.

## Main-Thread Jobs

1. Decide lane boundaries and stop conditions.
2. Merge returned findings into the current source of truth.
3. Keep old-run conclusions in the background: they inform axis choices, but
   they are no longer the active lane.
4. Keep [open_questions_and_hypotheses.md](open_questions_and_hypotheses.md)
   and [known_wrong.md](known_wrong.md) current.
5. Promote only stable conclusions into design docs and the Coach index.

## Investigation DAG

```text
local validation round 2
        -> tells us whether the feature surface is internally coherent
        -> updates launch gates with tested facts

survivaldiag manifest generator
        -> turns the matrix design into reviewable commands without launching
        -> prevents stale reward/opponent/seed schemas from coming back

matrix critique
        -> chooses staged blocks and copy counts before any big spend
        -> decides which axes are main, paired, sentinel, or gated

feature-gap audit
        -> lists exactly what is still missing before overnight launch
        -> produces the next canary commands

research-doc synthesis
        -> keeps historical failures from polluting the current path
        -> updates gotchas and simple rules for future agents
```

## Near-Term Stopping Condition

This rescue relaunch is done when:

- the clean 300-row manifest generation, tests, row-name checks, and train
  kwargs completeness checks pass; **done for 300b**
- the grouped submitter dry run proves rows target one deployed app and include
  both poller and train calls with a call shape accepted by the trainer; **done
  for 300b**
- the speed display fix has local test coverage and future trainers write
  `progress_latest.json`;
- the broken poller-only app is stopped;
- the trainer app is redeployed with the fixed code; **done**
- the full 300 rows are relaunched from the fixed manifest; **done**
- sampled rows have real trainer files, not only poller files:
  `run.json`, `latest_attempt.json`, `attempt.json`, `status_heartbeat.json`,
  and later `progress_latest.json`;
- late sampled rows have trainer-owned files, not only poller roots;
- the GIF browser has been redeployed or confirmed current, and new run markers
  are visible after trainer startup;
- surviving subagent results have been folded into the docs.

Current blockers by wave:

- Clean 300-row ready wave: healthy background training. Remaining work is
  learning/eval monitoring, not launch plumbing.
- Mix2 clean wave: healthy enough for background monitoring. Remaining work is
  more `k10`/`k20` eval coverage before recipe ranking.
- Mix3 current-checkpoint wave: launched. Remaining work is startup and first
  checkpoint/eval/GIF monitoring.
- Scripted wall-avoidant wave: current proactive wall-avoidant component is
  wired and has at least one clean remote GIF summary; broader scripted variants
  remain later work.
- Ancestor control wave: mechanically possible, but blocked on one exact tiny
  canary with current reward/env/GIF settings.
- Random-init frozen wave: blocked on immutable generated checkpoint refs and
  explicit opponent seed fields.
- Recent-checkpoint mixture wave: current trusted stock path now supports a
  static weighted episode-level mixture. The launched surface is
  `curvy-mix3-currentckpt-20260513a`: recent/mid/old refs from 300b, paired
  fast/browser render rows, and small compute probes.

## Subagent Output Protocol

Every subagent should keep output narrow:

- state whether files were edited;
- write or propose one target doc when useful;
- include exact file/line refs for repo claims;
- avoid broad recommendations unless tied to evidence;
- separate facts, hypotheses, and next checks;
- avoid launching runs unless explicitly assigned;
- avoid calling a path "self-play" without naming the opponent source and
  action semantics.

Preferred landing docs:

- active coordination: `delegation_log.md`;
- current plan: `current_source_of_truth.md`, `todo.md`;
- matrix plan: `next_overnight_matrix_plan.md`;
- known mistakes: `known_wrong.md`;
- open questions: `open_questions_and_hypotheses.md`;
- cleanup: `cleanup_targets.md`;
- background references: existing history/path/dataflow notes.
