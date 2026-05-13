# Raw Instruction Digest - 2026-05-13

Purpose: preserve the recent large user instructions recovered from local raw
Codex JSONL before trusting derived planning docs.

## Raw Files Inspected First

These were inspected before reading project docs:

- `/Users/shankha/.codex/history.jsonl`
  - `:9368`, `:9369`, `:9379`: invincible/blank opponent, survival reward,
    matrix, eval-curve tooling, planning docs.
  - `:9392`: not ready for matrix; old runs need manual and tooling analysis.
  - `:9395`, `:9399`, `:9400`: repeated-run/copy axis, blank canvas,
    invincible/random/ancestor opponents, wall-avoidant scripted opponent.
  - `:9404`: operating-pattern doc must describe planning/delegation docs.
  - `:9423`, `:9426`, `:9431`, `:9434`: prelaunch validation,
    self-critical loop, aggressive parallelism, raw-transcript recovery.
- `/Users/shankha/.codex/sessions/2026/05/08/rollout-2026-05-08T19-36-31-019e09f3-29f3-70d2-bcf6-434ef853f539.jsonl`
  - `:52350` / duplicate event `:52351`: full combined large instruction on
    weak frozen opponents, survival-plus-bonus/no-outcome reward, high episode
    cap, render twins, stochasticity, old-run projection, eval-curve tooling,
    massive tensor planning, docs, and parallel orchestration.
  - `:52609` / `:52610`: matrix is not ready; analyze old runs manually and
    with tooling before recommending the next tensor.
  - `:52903`, `:53017` and event duplicates: repeated copies, blank-canvas
    fake opponent, invincible random/ancestor opponents, and wall-avoidant
    scripted opponent behavior.
  - `:54100`, `:54130`, `:54299`, `:54407`: validate new features before the
    overnight batch; add self-critical loop; stay aggressively parallel; recover
    raw instructions.
- `/Users/shankha/.codex/sessions/2026/05/11/rollout-2026-05-11T11-01-43-019e178e-ee85-75a1-b469-f1082e75e55c.jsonl`
  - `:417` / `:418`: never stop after one run; implement, test, run, inspect,
    ask why, improve observability, update docs, repeat.
  - `:1946` / `:1947`: checkpoint eval/inspection should connect to training
    so new checkpoints trigger background Modal eval/GIF work.
  - `:5526` / `:5527`, `:8590` / `:8591`, `:11203` / `:11204`, `:13605` /
    `:13606`: clean old run visibility, purge stale runs when safe, keep the
    browser/tooling focused on live useful runs.
- `/Users/shankha/.codex/sessions/2026/05/01/rollout-2026-05-01T11-04-38-019de412-030e-71e1-8ec8-d931ed76b589.jsonl`
  - `:102555`, `:102771`, `:102903`, `:103553`: planning docs, working memory,
    todo/gates, operating patterns, and self-criticality loop are part of the
    work, not optional decoration.

## Plain Priorities

- Use stock LightZero `train_muzero` through `--mode train` as the trusted
  learning lane. Treat old `two-seat-selfplay` as historical unless native
  replay/target parity is proven.
- The immediate goal is visual CurvyTron survival, not beating a weak opponent.
  Outcome/win rate is telemetry; it can saturate while the policy is bad.
- Main reward for the next diagnostic lane should be survival plus bonus pickup
  with terminal outcome reward off/zero.
- Use a high episode cap such as `65536`; do not sweep cap unless a concrete
  technical reason appears.
- Start with clean blank-canvas/no-op wall survival: keep the two-player shape,
  but make player 1 inert, hidden, no-trail, no-collision, no-bonus, and
  irrelevant to trainer reward.
- Passive immortal is only a dirty canary unless it is deliberately labeled as
  such. It must not be confused with blank canvas or contained trail pressure.
- Keep scripted wall-avoidant opponents alive as an important trail-maker lane,
  but do not treat them as launchable until trainer plumbing and canaries exist.
- Use matched fast/browser render twins for serious cells; keep search,
  collector, and learner-batch sweeps as small sentinels.
- Sweep stochasticity meaningfully. v1d only tested a tiny level.
- Run repeated copies, about five where it matters, for random/stochastic rows.
  Separate `training_seed`, `reset_seed`, `opponent_policy_seed`,
  `opponent_behavior_seed`, `eval_seed`, and `copy_id`.
- Analyze old v1d by the metrics it actually trained: outcome/score first.
  Do not use old survival readouts to justify the new survival objective.
- Build eval-curve tooling that reads artifacts, creates per-run curves, and
  compares outcome, survival, reward, bonus, terminal causes, action collapse,
  and eval health without mixing them into one magic number.
- Operate as an orchestrator: plan, delegate, follow up, update docs, stay
  self-critical, and keep the main thread clear for synthesis.

## Raw Instructions Vs Current Docs

Aligned:

- `current_source_of_truth.md`, `user_priority_snapshot.md`,
  `hypotheses_and_evidence.md`, and `operating_patterns.md` now carry the main
  worldview: stock LightZero, survival-first, outcome as telemetry, old v1d
  saturation, blank-canvas anchor, repeat-copy groups, and parallel docs-first
  orchestration.
- `v1d_axis_projection.md` and `v1d_fresh_eval_summary_2026-05-13.md` correctly
  treat old runs as outcome/score evidence and warn against overclaiming
  survival learning from them.
- `eval_curve_tooling_plan.md` matches the raw tooling request: metric-agnostic
  curves, shape filters, false-negative caution, local snapshots, and future
  survivaldiag fields.
- `opponent_diagnostic_design.md`,
  `blank_canvas_noop_opponent_lane.md`, and
  `scripted_wall_avoidant_opponent_baseline_2026-05-13.md` correctly separate
  blank canvas, passive immortal, and scripted wall-avoidant opponents.
- `aggressive_matrix_scale_plan.md` captures the organized-scale idea: 50/100/
  200+ staged blocks, with extra rows spent on repeats and confirmation rather
  than a blind Cartesian product.

Gaps and drift:

- This digest previously cited derived May 13 subagent rollout files instead of
  the raw `history.jsonl` and named rollout records above. That made the
  provenance look cleaner than it was.
- Launch-facing docs still contain stale exact names in places. In particular,
  `next_matrix_manifest_design.md` still has old `survival_plus_bonus` and
  `survival_only` row sketches while the validated reward is
  `survival_plus_bonus_no_outcome`.
- `next_matrix_manifest_design.md` keeps a stale 48-row runnable-looking draft.
  It has warnings, but it is still easy for a launcher/generator agent to copy
  the wrong shape.
- `launch_gate_checklist.md` still has unchecked gates that are partly cleared
  elsewhere, and also gates that remain genuinely open. It needs a final
  source-of-truth pass before any matrix launch.
- Rich eval/status export is still a real blocker: tooling can parse reward
  components, bonus counts, terminal causes, action entropy, and eval health
  once present, but the upstream status path still must carry them reliably.
- Random-init / `iteration_0` frozen opponent generation is still not a
  manifest-addressable implementation path. Do not include random learned
  opponents until this is made immutable and explicit.
- Scripted wall-avoidant opponents are well probed, but not yet a stock trainer
  opponent lane. They should stay out of the main batch until wired and
  canaried.
- The raw instruction sometimes says render twins for every run; current docs
  narrow that to serious cells. That is a reasonable pruning choice, but it
  should be explicit in the final matrix so it does not look accidental.

## Highest-Priority Missing To-Dos

1. Finalize the exact next matrix manifest/generator after replacing stale
   reward names, stale 48-row rows, and ambiguous opponent labels.
2. Make eval/status export carry the survivaldiag fields the tooling already
   knows how to parse.
3. Reconcile `launch_gate_checklist.md` with the latest e2e canary state and
   mark which gates are cleared versus still open.
4. Decide whether the first launch is blank-canvas only or blank-canvas plus a
   small dirty passive-immortal control. Keep scripted/random opponents out
   until their gates pass.
5. If random frozen opponents enter the matrix, generate immutable
   random-init/`iteration_0` checkpoint refs and expose opponent seeds/copy ids
   in the manifest.
