# Orchestration Plan

Purpose: keep the CurvyTron exploration work split into clean lanes. The main
thread plans, synthesizes, and decides. Agents own bounded lanes with narrow
outputs and clear stop points.

Current phase: rescue relaunch. The ugly 50-row batch was stopped. The clean
300-row `curvy-survive-bonus-large-20260513a` manifest was submitted into one
deployed Modal app, but trainer calls crashed immediately because grouped
`train_kwargs` omitted required trainer settings after recent launcher edits.
Poller calls did write `checkpoint_eval_poller.json`, so the dashboard showed
poller activity without real trainer artifacts. Fix the kwargs shape, test it,
stop the broken poller-only app, redeploy, relaunch the full 300 rows, then
monitor real trainer heartbeats, progress files, browser markers, checkpoints,
and GIF/eval artifacts. Scripted wall-avoidant, random-init frozen opponents,
ancestor checkpoint controls, and recent-checkpoint mixture opponents are
separate gated waves.

## Active Lanes

| Lane | Owner | Output | Status |
| --- | --- | --- | --- |
| Feature status audit | Einstein | read-only table of actual implemented opponent/reward/stochasticity/observability features, tests, launch stance, and gaps | done |
| Manifest/launcher compatibility audit | Singer | verify generated command flags, compute values, dry-run gating, background eval/GIF flags, and row counts against the launcher | done |
| Matrix critique | Dirac | critique whether the current staged batch aligns with user priorities and avoids stale or over-wide axes | done |
| Speed display bug | Pascal | why the web UI says `speed unknown` and smallest fix | done: trainer must write `train/progress_latest.json` on checkpoint save |
| Clean 300-row manifest | main thread | patch full grouped train kwargs, regenerate, test, and relaunch `curvy-survive-bonus-large-20260513a` | active |
| Grouped Modal submitter | main thread | submit rows into one deployed app with poller+train calls, not one app per row; preflight must catch missing train kwargs before launch | active |
| Opponent variant wiring | worker lane | make scripted wall-avoidant opponent first-class if small; otherwise report blocker | active |
| Ancestor controls | Ramanujan | check whether frozen checkpoint controls are mechanically ready | done: path exists; use only after exact tiny canary |
| Recent checkpoint mixture audit | Leibniz | read-only map of current and historical support for recent/frozen/checkpoint mixture opponents | done: current trusted path has one static frozen checkpoint only |
| Volume cleanup | Erdos | dry-run old artifact cleanup; preserve current `curvy-survive-bonus-*` rows until relaunch is healthy | active |

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
  kwargs completeness checks pass;
- the grouped submitter dry run proves rows target one deployed app and include
  both poller and train calls with a call shape accepted by the trainer;
- the speed display fix has local test coverage and future trainers write
  `progress_latest.json`;
- the broken poller-only app is stopped;
- the trainer app is redeployed with the fixed code;
- the full 300 rows are relaunched from the fixed manifest;
- sampled rows have real trainer files, not only poller files:
  `run.json`, `latest_attempt.json`, `attempt.json`, `status_heartbeat.json`,
  and later `progress_latest.json`;
- the GIF browser has been redeployed or confirmed current, and new run markers
  are visible after trainer startup;
- surviving subagent results have been folded into the docs.

Current blockers by wave:

- Clean 300-row ready wave: blocked only by the grouped kwargs bug and relaunch
  validation. It is not blocked by scripted, random, ancestor, or mixture work.
- Scripted wall-avoidant wave: blocked until first-class trainer wiring and
  exact tiny canary exist.
- Ancestor control wave: mechanically possible, but blocked on one exact tiny
  canary with current reward/env/GIF settings.
- Random-init frozen wave: blocked on immutable generated checkpoint refs and
  explicit opponent seed fields.
- Recent-checkpoint mixture wave: current trusted stock path supports a single
  static frozen checkpoint opponent, not a pool or rolling recent-checkpoint
  sampler. Treat this as the next batch design/implementation lane after the
  rescue relaunch is healthy.

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
