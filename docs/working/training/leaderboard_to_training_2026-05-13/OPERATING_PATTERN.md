# OPERATING_PATTERN

This is the working rhythm. Review it before long work sessions and whenever
the thread feels confused.

## Main Pattern

- Keep docs current. If a fact changes, write it down before moving on.
- Keep the main thread for planning, integration, direct verification, and user
  status.
- Use subagents for bounded audits, critiques, cleanup, metrics, and side lanes.
- Use follow-ups when a subagent result is incomplete.
- Do not wait passively. If blocked, start a smaller honest version of the same
  proof in parallel.
- Think a few steps ahead: if steps 1-5 might all be needed, consider starting
  probes for later steps early, knowing some may fail.
- Be aggressive with parallelism, but keep a small written task board so the
  lanes do not disappear.
- Every time a "blocker" appears, ask whether it blocks all progress or only
  one quality path. Do not turn an optional quality improvement, like a better
  historical leaderboard, into a fake launch blocker for bootstrap training.
- Use plain names for proof gates. A "source leaderboard" is only a ranked list
  for choosing starting checkpoint opponents. It is optional. The actual gate is
  whether the loop works: trainer checkpoint -> intake -> tournament ->
  assignment -> trainer use.
- If work starts orbiting one issue for too long, write the question in plain
  language, name the smaller honest experiment, and run that experiment while
  the deeper diagnosis continues.

## Reporting Pattern

- Use simple words.
- Quantify instead of hand-waving.
- Keep names readable.
- Explain what is proven, what is inferred, and what is unknown.
- Do not claim success from partial artifacts.

## Code Pattern

- Prefer the existing LightZero path and minimal scaffolding.
- Keep environment behavior in the environment.
- Keep trainer behavior in the trainer.
- Keep tournament/Inspector behavior in the tournament lane.
- Avoid hidden fallbacks in new contracts. Fresh research paths should fail
  loudly when a required field is missing. If old artifacts need repair, put
  that in an explicit migration/normalization tool, not inside the default
  trainer/env path.
- Do not keep backwards compatibility just because it is easy. This is a
  research repo; the current correct path should be obvious in code and docs.
- Current CurvyTron defaults live in `src/curvyzero/contracts/curvytron.py`.
  Do not add new local copies of app names, volume names, source max steps,
  checkpoint cadence, learner-seat defaults, or policy render modes in
  launchers and manifest builders.
- Tournament UI current ids also belong in the shared contract. Do not hard-code
  stale arena/rating ids directly in the web app.
- Do not restore old compatibility shims such as `curvytron_volume_names.py`.
  If old artifacts need handling, write an explicit migration or repair tool.
- Fresh tournament/rating specs should use `policy_trail_render_mode` and
  `policy_bonus_render_mode`. Do not add new `observation_*`, `source_state_*`,
  or generic `trail_render_mode` aliases to the fresh spec path.
- After emergency patches, schedule cleanup if the patch adds fallback soup.
- Keep public intent fields clean. Example: slot recipes should say
  `opponent_immortal=true/false`; lower-level runtime switches such as
  `opponent_death_mode` can be derived at the env boundary.
- `blank_canvas_noop` is an inert/immune opponent shape in practice. Public
  slot recipes must say `opponent_immortal=true` for it; do not rely on the env
  quietly making it immune.
- Hard-coded opponents are also immortal by contract when used as training
  pressure. If a run needs mortal opposition, use frozen checkpoint opponents
  from exact refs; do not create a mortal fixed-straight or wall-avoidant
  sentinel.

## Live-System Pattern

- Use deployed Modal apps for durable services, fanout, and repeated
  coordination. Treat overlapping ephemeral `modal run` calls as a development
  surface, not production proof.
- After contract cleanup, immediately run a fresh deployed end-to-end canary.
  Old remote proofs are templates, not proof that the current code still works.
- Use `modal run --detach` only when a long development/proof job must outlive
  the local command, and then track and stop stale detached apps deliberately.
- Verify child work completed; do not trust "scheduled" as success.
- Treat Volume JSON as the durable truth for manifests, checkpoints, ratings,
  assignments, audits, and debug bundles.
- If old workers have already polluted a live artifact tree, do not explain
  around it. Name it dirty, stop stale apps, and either use a fresh id or run an
  explicit repair/purge. A dirty root `progress.json` is not proof even if some
  later code is fixed.
- After an interruption, context compaction, or long pause, restart from durable
  state: read `NOW.md`/`TODO.md`/`FULL_LOOP_PROOF.md`, re-query Modal Volumes
  for the current artifacts, then update docs before taking the next mutating
  action. Do not trust old terminal sessions as the source of truth.
- Treat Modal Dict and Queue as coordination only: wakeups, leases, pointers,
  and operator intent. They are not the historical record.
- Write immutable per-round/per-shard files first, then move one small pointer
  last.
- Avoid broad reload-dependent behavior during active file reads. Design readers
  and websites so they can recover from explicit Volume commit/reload semantics
  instead of depending on lucky cache state.
- Do not reload the same Modal Volume from a long-running trainer when the
  trainer itself may have open files on that Volume. For run-local handoffs
  written by the same process, read the local path directly; reserve
  `Volume.reload()` for external control/tournament state that must cross
  container boundaries.
- Preserve current live lanes while cleaning old garbage.
- Clean dashboards matter: hide/purge old arenas and apps once they are not
  needed for current proof or old champion extraction.
- Use long sleeps only when the expected next artifact genuinely takes time.
  While sleeping is not needed, keep working other lanes.
- Do not turn a large stress proof into the only path. If an all-pairs live
  tournament is useful but slow, start a smaller honest live proof in parallel
  and keep the large one as stress evidence.
- Exact checkpoint refs are frozen seeds. Run ids or run-id prefixes are live
  watches. If a live service needs both, preserve the run watch and pin the
  exact refs beside it; do not collapse the service back to explicit refs only.
- Progress reads must not poison future work. A zero-work
  `waiting_for_round_input` progress file is only an observation marker, not an
  active rating artifact. A real round input or non-empty progress can block
  overlap; an empty marker must not.
- In detached Modal flows, child work that must outlive the local caller should
  be started with `.spawn()`, not `.remote()`. Then verify durable Volume
  artifacts, not just function call ids.

## Validation Phase Pattern

- After a broad cleanup/refactor, move quickly from design to proof. The order is
  local focused tests, broad E2E-adjacent tests, deployed canary, then larger
  run.
- Keep one plain proof doc with artifact ids and one plain task board with the
  next risk. Do not let stale older docs be the first thing a new agent reads.
- A deployed canary is enough to prove the path exists, but not enough to prove
  scale, survival improvement, or cleanup. Write those as separate claims.
- A full-loop proof is not closed until the same running trainer has a
  `decision=applied` refresh event and later env telemetry rows using the new
  assignment sha with `opponent_provider_load_ok=true`.
- When a proof finds a bug, add one regression test for the bug and record the
  exact broken artifact. Do not only patch the live artifact.
- When a live-feedback path is suspect, run no-tournament controls in parallel:
  static fixed opponents and an own-latest frozen-checkpoint control. These
  answer different questions and should not be merged into one blurry
  "self-play" claim.
- Before a larger launch, verify the launch manifest names the current contract:
  `random_per_episode`, control-volume assignment pointer, fresh tournament and
  rating ids, policy observation surface, checkpoint cadence, and public
  `opponent_immortal` slot intent.
- Leaderboard-derived launch builders should fail closed: require the source
  leaderboard snapshot explicitly, default to immutable assignments plus
  refresh pointers, and reject stale app/Volume names instead of relying on
  operator memory. Bootstrap/static launch builders may use curated assignments
  and exact checkpoint refs without claiming a trusted top leaderboard.
- A manifest is not launchable just because its refs are syntactically exact.
  Before launch, audit that every initial/frozen checkpoint ref exists in the
  active runs Volume and that the assignment/control refs point at the active
  control Volume. Old leaderboard snapshots can contain perfectly shaped refs
  that point at files outside the current storage lane.
- A rating snapshot is not a production-quality opponent source just because
  games completed. For leaderboard-derived restart opponents, require the
  latest source rerate to be coverage-mature, `stable=true`, and published with
  expected round/context/roster/snapshot hashes. Treat `stable=false` as a hard
  blocker for opponent-source publish/materialization, not for bootstrap
  training from curated/static assignments.
- Do not confuse opponent-source quality with system proof. If the ranked
  source is weak, stale, or unstable, bootstrap can still proceed from exact
  checkpoint refs and immortal sentinels while the tournament learns a better
  public ordering over time.
- Do not say "stable source leaderboard" as if it is one required object. Say
  the plain claim instead: either "this ranked source is good enough to choose
  leaderboard-derived slots" or "bootstrap does not need it." The system proof
  is the feedback loop, not the starting rank quality.
- When the user challenges a premise, stop and rewrite the premise in plain
  language before adding more machinery. If the premise is fake, delete it from
  the docs/tests instead of explaining around it.
- For long tournament/rating checks, prefer direct Volume artifacts over a
  browser/progress endpoint when the endpoint is slow or stale:
  `ratings/<rating_run_id>/latest.json`,
  `ratings/<rating_run_id>/results.json`, and
  `ratings/<rating_run_id>/rounds/<round_id>/{input,progress,ratings}.json`.
  The persisted round input is the truth about roster size and previous-round
  linkage.
- If the source is being rematerialized from old storage, audit both sides:
  first prove the selected old refs exist in the historical source, then after
  copying prove the same refs exist in the all-v2 target. The fresh rerate only
  starts after the v2 target audit passes.
- Current launch helpers should reject old Curvy Volume names. Historical
  storage reads belong in explicit migration/audit tools with loud names, not in
  the default trainer, tournament, submitter, or dashboard paths.

## Tournament Pattern

- Ratings should use eval/greedy policy mode unless explicitly running a noisy
  diagnostic.
- Tournament observations must match the checkpoint's training observation
  surface.
- The current policy observation surface is `browser_lines + simple_symbols`.
  Optimizer work may make that GPU-backed, but the launcher contract should
  still name this surface.
- Perspective is a Coach/training contract: the policy receives the
  controlled-player view for the physical seat it controls. Optimizer must not
  choose or randomize seats inside the renderer backend.
- CPU `browser_lines + simple_symbols` is the parity oracle/fallback while the
  optimizer lane improves the fast implementation.
- `body_circles_fast + simple_symbols` is historical CPU ablation/control
  evidence only, not the destination for new runs.
- Tournament timing is one source frame per policy action. The 16.6667ms
  decision interval should come from the shared source-frame contract, not a
  scattered literal.
- Large tournaments must use sane scheduling. If all-pairs is chosen, it must be
  deliberate and documented.
- Public leaderboard snapshots should come only from validated rating artifacts.

## Training Pattern

- Rewards and slot recipes are experimental knobs; record exact settings.
- Assignment files are immutable truth.
- Modal Dict can hold operator intent or pointers, not hidden training truth.
- Running refresh must happen only at a clean boundary and write telemetry.
- Champion start weights are separate from opponent assignment.
- Learner seat/perspective belongs in the training config. The current default
  is `random_per_episode`; fixed-seat modes are diagnostics, not the restart
  default.
- Slot intent is simple: blank/hard-coded sentinel opponents are immortal all
  the time; frozen checkpoint slots are mortal most of the time, with explicit
  small immortal slices only when a recipe asks for them. Keep total immortal
  exposure around `20-30%` unless a diagnostic says otherwise.
- Keep internal debug/audit fields out of env-facing slot dictionaries. If
  Modal reloads produce warnings, record them beside the resolved assignment or
  in refresh events, never inside `opponent_mixture.entries`.

## Self-Critique Loop

Regularly ask:

- Am I proving the actual requested loop, or only an adjacent piece?
- Am I waiting when I could run a smaller honest proof in parallel?
- Did I update the docs with the latest fact?
- Did I leave a side lane without an owner or next action?
- Did I hide a behavior behind fallback code?
- Are the names understandable to a tired human looking at a dashboard?
