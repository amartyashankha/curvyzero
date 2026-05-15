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
- Preserve current live lanes while cleaning old garbage.
- Clean dashboards matter: hide/purge old arenas and apps once they are not
  needed for current proof or old champion extraction.
- Use long sleeps only when the expected next artifact genuinely takes time.
  While sleeping is not needed, keep working other lanes.

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
- Before a larger launch, verify the launch manifest names the current contract:
  `random_per_episode`, control-volume assignment pointer, fresh tournament and
  rating ids, policy observation surface, checkpoint cadence, and public
  `opponent_immortal` slot intent.
- Real launch builders should fail closed: require the source leaderboard
  snapshot explicitly, default to immutable assignments plus refresh pointers,
  and reject stale app/Volume names instead of relying on operator memory.
- A manifest is not launchable just because its refs are syntactically exact.
  Before launch, audit that every initial/frozen checkpoint ref exists in the
  active all-v2 runs volume. Old leaderboard snapshots can contain perfectly
  shaped refs that point at files outside the current storage lane.

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

## Self-Critique Loop

Regularly ask:

- Am I proving the actual requested loop, or only an adjacent piece?
- Am I waiting when I could run a smaller honest proof in parallel?
- Did I update the docs with the latest fact?
- Did I leave a side lane without an owner or next action?
- Did I hide a behavior behind fallback code?
- Are the names understandable to a tired human looking at a dashboard?
