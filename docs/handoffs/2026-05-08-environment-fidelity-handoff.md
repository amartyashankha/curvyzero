# 2026-05-08 Environment Fidelity Handoff

## Scope

This handoff is for the environment-fidelity lane. It covers the current CurvyTron
reference evidence, the CurvyZero simulator status, the comparison plan, and the
Modal handoff for future fidelity jobs. Treat it as the current-memory packet for
the next environment turn, with `docs/working/environment/active_lanes.md` as the
short live map.

## Current Checkpoints

- Source-body-canary batch is verified for six narrow body/trail fixtures:
  opponent tangent-safe strict overlap, opponent overlap-kill, own delta `3`
  safe, own delta `4` kill, same-frame point kill, and same-frame point control.
- Source-print-manager-canary batch is verified for three narrow toggle
  fixtures: print-to-hole, hole-to-print, and active no-toggle control.
- Wall/border and 3P/4P normal-wall batches remain the regression base for
  border, scoring, same-frame death, and death-order behavior.
- Benchmark scaffold has been added at `scripts/benchmark_env.py`; it is
  simplified toy-v0 measurement scaffolding, not source-fidelity or optimization
  proof.
- Observability plan exists at `docs/design/environment/observability_plan.md`.
  The local loop now writes explicit common trace artifacts and compact timeline
  artifacts in common-trace mode.
- Production optimization backend rewrites remain deferred until source-fidelity
  fixtures and the single-env contract are stable enough to compare backends.
- Next fidelity target: trail cadence and trail-gap/collision behavior, followed
  by broader collisions, bonuses, and observation checks.
- Next observability target: add bounded first-mismatch context around the first
  differing common-trace step.

## What We Know

- The original CurvyTron source is available at `third_party/curvytron-reference`.
- The raw reference app does not run from this checkout yet. `bin/curvytron.js` and
  generated browser files are missing. They are produced by the old Gulp/Bower build.
- The local machine has modern Node/npm. The reference project is from the Node 0.10
  era, so the next build attempt should happen in a disposable copy or pinned image.
- Source mining already found the important server rules: elapsed-ms movement, 60 Hz
  target loop, 3000 ms warmup, 5000 ms warmdown, delayed trail printing, source map
  size formula, speed `16`, turn base `2.8 / 1000`, radius `0.6`, trail latency `3`,
  distance-based trail holes, strict circle overlap, normal-wall death, borderless
  timed-bonus edge wrap, and same-frame score behavior.
- Source borderless is not a clean torus: it uses margin `0`, places the avatar at
  the exact opposite edge, loses overshoot, handles the first border axis found, and
  skips body collision on the wrap frame.
- The current `curvyzero-v0` Python environment is a simplified training scaffold. It
  uses fixed ticks, a 64x64 grid, solid trails, grid collision, 1v1 episodes, no
  bonuses, fixed out-of-bounds walls, no source borderless wrap, and no source
  round lifecycle.
- Current Python and Modal smokes prove the scaffold can run. They do not prove
  CurvyTron source fidelity.
- The best first reference path is a headless Node state oracle, not the browser UI.
  The oracle should load server-side JS objects, force a small state, apply scripted
  actions, call `game.update(step_ms)`, and emit JSON state traces.
- The repo contains a headless Node oracle probe at
  `tools/reference_oracle/headless_probe.js`. It loads original source files in
  a Node `vm`, creates fake players, applies `avatar.updateAngularVelocity(move)`,
  calls `game.update(1000 / 60)`, and emits state/events.
- The raw server dependency path is still blocked without old packages. The first
  missing module is `faye-websocket`.
- The repo contains Python toy-v0 trace/fingerprint helpers in
  `src/curvyzero/env/tracing.py`. They are useful for trace plumbing but do not
  prove CurvyTron fidelity.
- Scenario tooling is split by ownership: shared schema/parsing lives in
  `curvyzero.env.scenario_schema`, the toy-v0 scenario runner lives in
  `curvyzero.env.toy_runner`, and source-fidelity implementation now lives in
  `curvyzero.fidelity.source_runners`. `curvyzero.env.scenarios` remains a
  compatibility facade and `python -m curvyzero.env.scenarios` CLI entry point,
  not the home for new source-runner implementation.
- Post-refactor guard fix: source kinematics and borderless runners validate
  forced positions/headings/alive lengths against `player_count`, and forced
  alive extraction supports nested `players[].initial.alive`.
- The repo contains a Python-only Modal trace smoke in
  `src/curvyzero/infra/modal/fidelity_smoke.py`. The last recorded smoke wrote
  `trace_fingerprint.json` to the `curvyzero-runs` Volume.
- A shared scenario fixture exists at
  `scenarios/environment/forced_two_player_turn_step.json`.
- The JS scenario runner works:
  `node tools/reference_oracle/scenario_runner.js scenarios/environment/forced_two_player_turn_step.json`.
- The Python scenario runner works:
  `uv run python -m curvyzero.env.scenarios scenarios/environment/forced_two_player_turn_step.json --compact`.
- The one-command local loop works: `uv run python tools/run_fidelity_loop.py`.
  It writes JS, Python, diff, and summary artifacts under
  `artifacts/local/fidelity/forced_two_player_turn_step/`. Common-trace diff is
  now the default; `--raw-diff` is only for debugging raw runner output.
- Common trace diff now works. The default Python runner still shows the expected
  toy-v0 mismatch at player 0 angle: JS `-0.046667`, Python toy-v0 about `-0.08`.
- The loop also supports `--python-runner source-kinematics`. Last recorded
  checks matched the JS common trace for the current four one-step movement
  scenarios.
- The loop supports `--python-runner source-normal-wall`. Last recorded checks
  matched the JS common trace for `source_normal_wall_death_step` and
  `source_normal_wall_same_frame_draw_step`. It is scoped to movement plus
  normal-wall death state/events only.
- The `source-border-rules` runner now verifies the narrow 3P/4P normal-wall
  scoring and death-order canaries, including the harder 4P terminal draw:
  `source_normal_wall_3p_two_die_one_survivor_step`,
  `source_normal_wall_4p_ordered_deaths_survivor_score`, and
  `source_normal_wall_4p_two_prior_then_same_frame_terminal_draw`.
- The `source-body-canary` runner now verifies the narrow body/trail canaries:
  `source_body_opponent_tangent_safe_step`,
  `source_body_opponent_overlap_kills_step`,
  `source_body_own_delta3_safe_step`,
  `source_body_own_delta4_kills_step`,
  `source_body_same_frame_point_kills_step`, and
  `source_body_same_frame_point_control_safe_step`. This only covers strict
  opponent stored-body overlap/safe tangent behavior, own-body latency at the
  `> 3` point-number gate, and the two direct same-frame point materialization
- The `source-print-manager-canary` runner verifies deterministic toggle basics:
  print-to-hole, hole-to-print, and active no-toggle control. This only covers
  forced active manager state, distance subtraction, property payloads, important
  point side effects, and final trail/body counters; broader trail cadence/gaps,
  delayed start, bonuses, browser messages, and replay payloads remain pending.
- The loop supports `--python-runner source-borderless-wrap`. Last recorded
  checks matched `source_borderless_wrap_step`. Use `--python-runner
  source-border-rules` for the mixed wall/border batch.
- Event comparison is opt-in through `comparison.include_events: true`. Without
  that flag, common-trace diffs are state-only. The current wall/border source
  fixtures opt in to the narrow event contract.
- The narrow event contract is ordered per-step events with only these event
  names and fields: `position` (`player_id`, `x`, `y`), `point` (`player_id`,
  `x`, `y`, `important`), `die` (`player_id`, `killer_id`, `old`),
  `score:round` and `score` (`player_id`, `score`, `roundScore`), and
  `round:end` (`winner_id`).
- The current reconstruction loop should stay simple:
  `source map -> probe -> scenario -> common trace -> diff -> implement -> test`.
  Raw traces are debug artifacts; common-trace diff is the meaningful comparison.
- A compact source-map index now exists at
  `docs/research/curvytron_source_map/facts_index.md`. Use it as the fast
  orientation page, then read the deeper subsystem notes when changing behavior.
- A thin local scenario batch runner exists at `tools/run_fidelity_batch.py`.
  Modal batch should come after the local movement and wall/death batches are
  useful.
- Forced state/event traces now cover normal-wall single death, normal-wall
  same-frame draw, source borderless wrap, and the narrow 3P/4P normal-wall
  multiplayer canaries. Narrow wall/border event fidelity is verified only for
  those fixtures.

## North Star

Full source fidelity comes first, then optimization. Performance, training
throughput, browser hosting, and Modal scale-out stay out of the current lane
until state traces and core game behavior are trustworthy. The public training
interface is a separate contract from the fidelity machinery: use reconstruction
evidence to inform it, but do not let interface convenience hide source behavior.

## What We Decided

- Keep `curvyzero-v0` honest: it is a deliberate training ruleset, not an exact clone.
- Compare in this order: source facts, deterministic state traces, golden outcomes,
  replay/event logs, server messages, then screenshots or videos.
- Use state and then opted-in events as the first proof. Pixel or video checks
  are later human review.
- Use Modal only around whole scenarios, batches, benchmarks, and artifact storage.
  Do not call Modal per environment tick or inside MCTS/search hot loops.
- Full browser/server hosting remains deferred. Keep it out of the first trace loop.
- Keep canonical contracts separate from support notes. Handoffs, experiment logs,
  and critiques can point at the contract docs, but should not redefine schemas.
- Add multiplayer canaries early. Even if first training is 1v1, source fidelity needs
  3-player and 4-player checks for map size, scoring, same-frame deaths, and ordering.
- Do not say plain "wall collision" when the rule matters. Use normal-wall death,
  fixed toy-v0 wall, or source borderless wrap.
- The 3P/4P normal-wall scoring, death-order, and 4P terminal draw canary slice
  is verified locally through common-trace state/event comparison. Keep the
  result narrow: it does not cover body collisions, trails, bonuses, browser
  messages, or replay payloads.
- The source-body-canary slice is verified locally for six narrow fixtures only.
  Keep the result narrow: it proves opponent strict overlap/tangent, own latency
  at the `> 3` point-number gate, and the two direct same-frame point
  materialization fixtures, not print-manager holes, broader trail storage,
  bonuses, browser messages, or replay payloads.
- Benchmark scaffold is allowed as measurement groundwork only. It must not pull
  optimization, vectorization, or backend rewrites into the fidelity lane.
- Observability should stay artifact-first and local: materialize common traces,
  then timeline summaries. Do not turn this into dashboards or browser replay.

## Self-Reflection

Do not trust memory or old handoff wording for rule details. Source and probe output
win. If a source/probe result is not ready, mark it pending instead of guessing.

## What Is Still Unknown

- Whether the old reference app can build cleanly in a disposable copy or Modal image.
- Which first target wins: `curvytron-v1-reference`, `curvytron2-reference`, or only
  the named `curvyzero-v0` training ruleset.
- Whether source-fidelity traces should use elapsed milliseconds or fixed `1000 / 60`
  ms steps.
- Numeric tolerances for longer elapsed-ms traces.
- Which `curvyzero-v0` differences must close before the first serious learning run.
- How to handle head-head and update-order behavior when source behavior is awkward.
- Broader event fidelity beyond the narrow wall/border, 3P/4P normal-wall, and
  source-body canary contracts, including print-manager holes, broader trails,
  bonuses, observations, and full replay/server messages.

## Next Actions

1. Keep the wall/border and 3P/4P normal-wall multiplayer batches in regression
   before changing source border or scoring behavior.
2. Keep the scenario split intact: shared schema in `curvyzero.env.scenario_schema`,
   toy-v0 runner in `curvyzero.env.toy_runner`, source fidelity in
   `curvyzero.fidelity.source_runners`, and `curvyzero.env.scenarios` only as a
   compatibility facade/CLI.
3. Keep trainer-facing code importing only `curvyzero.env`; source runners,
   scenario schema helpers, trace projectors, and diff tools remain offline
   evidence machinery.
4. Keep Python same-frame point materialization scoped to the two verified direct
   fixtures unless new source evidence expands it.
5. Keep Python print-manager support scoped to the three verified direct toggle
   fixtures unless new source evidence expands it.
6. Broaden next to trail storage/gap canaries,
   bonuses/replay messages, and observation checks.
7. Use the existing common trace and compact timeline sidecars when reading
   mismatches; add bounded first-mismatch context only when a failing scenario
   needs it.
8. Keep Modal, browser hosting, and performance work out of the next local
   mechanic slice unless a coarse batch job specifically needs them.

## Later, Not Now

- Browser/server hosting for the raw reference app.
- Modal batch plumbing beyond coarse scenario jobs.
- Hot-loop optimization, vector backends, production data-layout rewrites, or
  performance rabbit holes. Benchmark scaffolding may continue only as deferred
  measurement groundwork.
- Public training-interface changes unless a docs link or boundary clarification
  is needed.

## Last Recorded Checks

- `uv run --extra dev pytest`: 98 passed.
- `uv run --extra dev ruff check .`: all checks passed.
- `node tools/reference_oracle/headless_probe.js`: passed and emitted a 2-player
  forced-step trace.
- `uv run python tools/run_fidelity_loop.py`: passed and wrote JS/Python/diff/
  summary artifacts under `artifacts/local/fidelity/forced_two_player_turn_step/`.
- `uv run python tools/run_fidelity_loop.py scenarios/environment/forced_two_player_turn_step.json --python-runner source-kinematics --artifact-root /private/tmp/curvy-single-default-common-trace`:
  passed with `match: true`; the diff command included `--common-trace` by default.
- `uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --artifact-root /private/tmp/curvy-source-kinematics-batch`:
  passed with `4` pass, `0` fail, `0` blocked.
- `uv run python tools/run_fidelity_loop.py scenarios/environment/source_normal_wall_death_step.json --python-runner source-normal-wall --artifact-root /private/tmp/curvy-normal-wall-death-loop-source-runner`:
  passed with `match: true`, `diff_status: pass`, and `first_mismatch: null`.
- `uv run python tools/run_fidelity_loop.py scenarios/environment/source_normal_wall_same_frame_draw_step.json --python-runner source-normal-wall --artifact-root /private/tmp/curvy-normal-wall-draw-loop-source-runner`:
  passed with `match: true`, `diff_status: pass`, and `first_mismatch: null`.
- `uv run python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_batch.json --python-runner source-normal-wall --artifact-root /private/tmp/curvy-source-normal-wall-batch`:
  passed with `2` pass, `0` fail, `0` blocked.
- `uv run python tools/run_fidelity_loop.py scenarios/environment/source_borderless_wrap_step.json --python-runner source-borderless-wrap --artifact-root /private/tmp/curvy-borderless-wrap-loop-source-runner`:
  passed with `match: true`, `diff_status: pass`, and `first_mismatch: null`.
- `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-border-batch-post-refactor-guard`:
  passed with `3` pass, `0` fail, `0` blocked; `diff_mode: common-trace`;
  `source_normal_wall_death_step`, `source_normal_wall_same_frame_draw_step`,
  and `source_borderless_wrap_step` each had `match: true`, `diff_status: pass`,
  and `first_mismatch: null`.
- `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-post-refactor-guard`:
  passed with `3` pass, `0` fail, `0` blocked; `diff_mode: common-trace`;
  `source_normal_wall_3p_two_die_one_survivor_step`,
  `source_normal_wall_4p_ordered_deaths_survivor_score`, and
  `source_normal_wall_4p_two_prior_then_same_frame_terminal_draw` each had
  `first_mismatch: null`.
- `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-same-frame`:
  passed with `6` pass, `0` fail, `0` blocked; `diff_mode: common-trace`;
  `source_body_opponent_tangent_safe_step`,
  `source_body_opponent_overlap_kills_step`,
  `source_body_own_delta3_safe_step`,
  `source_body_own_delta4_kills_step`,
  `source_body_same_frame_point_kills_step`, and
  `source_body_same_frame_point_control_safe_step` each had
  `first_mismatch: null`.
- Modal Python trace smoke: run `ap-muerWyWt71VmTbAhIc5DLL`, artifact
  `experiments/env-fidelity-smoke-20260508-python-trace/attempts/attempt-20260508T181410Z-486f8f56551c/fidelity-smoke/trace_fingerprint.json`.

## Main Docs

- [Environment design index](../design/environment/README.md)
- [Environment research index](../research/environment/README.md)
- [Working question map](../working/environment_questions.md)
- [Fidelity comparison](../design/environment/fidelity_comparison.md)
- [Reconstruction workflow](../design/environment/reconstruction_workflow.md)
- [Trace loop contract](../design/environment/trace_loop_contract.md)
- [Probe automation plan](../design/environment/probe_automation_plan.md)
- [CurvyTron source facts index](../research/curvytron_source_map/facts_index.md)
- [Reference oracle design](../design/environment/reference_oracle.md)
- [Modal fidelity jobs](../design/environment/modal_fidelity_jobs.md)
- [Modal environment fidelity runbook](../runbooks/modal_environment_fidelity.md)
- [CurvyTron raw run probe](../research/environment/curvytron_raw_run_probe.md)
- [CurvyTron JS state oracle notes](../research/environment/curvytron_js_state_oracle.md)
