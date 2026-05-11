# User Instruction Packet

Date: 2026-05-09

Scope: distilled project instructions recovered from local Codex session state,
current CurvyZero repo docs, and current-session activity. This is a working
packet for agents operating in `/Users/shankha/curvy`.

Privacy rule: this packet intentionally summarizes. It does not copy raw
transcripts, secrets, credentials, unrelated personal content, or private text
outside the CurvyZero/CurvyTron effort.

## Sources Reviewed

- 2026-05-09 raw-session refresh: read `/Users/shankha/.codex/sessions`,
  `history.jsonl`, `state_5.sqlite`, `logs_2.sqlite` schema/counts, selected
  `.codex/memories`, and current repo docs. Did not inspect `auth.json`, did
  not dump raw conversations, and treated the session store as sensitive.
- Latest scan shape: 1,610 session files, 27,528 session user-message events,
  7,184 history lines, 8,190 exact-unique user messages. Current state index
  showed 3,209 threads, 395 in `/Users/shankha/curvy`.
- Current repo docs under `docs/`, especially handoffs, design docs, research
  notes, experiment logs, and working notes.
- `/Users/shankha/.codex/state_5.sqlite`: thread index and Curvy repo session
  metadata. Earlier 2026-05-08 pass showed 2,852 total threads, 75 threads in
  `/Users/shankha/curvy`, and 74 Curvy subagent threads; the 2026-05-09 refresh
  found the larger counts above.
- `/Users/shankha/.codex/logs_2.sqlite`: log shape and counts. It is mostly
  telemetry/tool trace data; useful for activity scale, not user intent.
- `/Users/shankha/.codex/history.jsonl`: keyword-filtered review for
  CurvyZero/CurvyTron and durable workflow instructions.
- `/Users/shankha/.codex/sessions/2026/05/08/`: main Curvy session and spawned
  worker prompts/results.
- `/Users/shankha/.codex/config.toml`: non-secret configuration only.
- Selected `/Users/shankha/.codex/memories/*` continuity notes where they
  explained durable working style. These are mostly cross-project guardrails,
  so only project-relevant patterns are included here.
- Existing `docs/working/user_message_memory_review.md`, which remains useful
  as an audit trail and raw-source summary.

Not reviewed: `/Users/shankha/.codex/auth.json` and any source that looked like
credential or account material.

## Highest Priorities

1. Faithful environment reconstruction is the current north star.
2. Keep the main thread clear: plan, delegate, verify, synthesize, and decide.
3. Use many parallel subagents, but make each one bounded, useful, and explicit.
4. Keep docs current because chat memory is short and workers are concurrent.
5. Keep the loop simple: source map -> probe -> scenario -> common trace -> diff
   -> implement -> test.
6. Close one real mismatch at a time. Do not build a large framework around a
   mismatch before proving the next small behavior.
7. Source and probe output beat memory, handoff text, and intuition.
8. Do not call `curvyzero-v0` source-faithful. It is a simplified training
   ruleset unless a specific behavior is explicitly matched.
9. Use Modal early for coarse smokes, benchmarks, GPU checks, and artifacts, but
   never inside per-step or per-node hot loops.
10. Use plain language. Avoid invented terminology, unclear taxonomies, and
    jargon that hides what is happening.

## 2026-05-09 Reorientation

Stable user instructions from the latest raw-session review:

- Keep docs as working memory. Durable artifacts beat chat-only analysis.
- Use aggressive parallelism, but keep it bounded: precise scope, owned files,
  required final shape, and durable output.
- The main thread should plan, delegate, synthesize, verify, and decide. It
  should not become the worker.
- Do not revert or overwrite concurrent edits. The repo may be mostly
  untracked, so `git diff` alone is a weak integration view.
- Use plain language. Avoid taxonomy bloat, clever architecture, and vague
  progress claims.
- Validate hard: source output, command output, recorded eval waves, artifact refs,
  and reproducible logs beat memory.
- If a result is too clean, check leakage, target scoreability, seed luck,
  selection bias, and artifact mismatch.
- Keep Modal as coarse compute/artifact infrastructure, not per-step runtime.
- Never expose secrets or raw transcripts in docs.

Current project north star:

```text
source map -> probe -> scenario -> common trace -> diff -> implement -> test
```

Training is a parallel lane, not the source of truth. Its honest near-term goal
is a small checkpointed loop that separates learned behavior from scripts,
planner rules, fixed-seed luck, and best-checkpoint luck. Pong is a visual
learning probe; Tiny Line Duel is a multiplayer shape check; survival is a
diagnostic. CEM/MLP/Pong baselines are useful, but they are not MuZero progress.

Current failure modes to watch:

- Orchestration can become the product: many workers, many docs, many status
  surfaces, not enough ownership of the next gate.
- Training docs can become a thicket. Label baselines plainly and stop calling
  scaffolding "MuZero" progress.
- Front-door docs must stay current. Stale active paths are a serious bug.
- Meta-reorientation is useful only if it leads to one concrete gate.

Aggressive parallel runbook:

- Use one captain plus 5 to 7 bounded lanes when parallelism helps.
- Good lanes: instruction auditor, environment worker, training worker, Modal
  worker, docs worker, critic worker, integration worker.
- Launch with exact file ownership. Collect after 30 to 60 minutes. Close stale
  workers. Synthesize into a single next gate.
- Avoid vague swarms.

Docs that must stay current:

- Front door: `README.md`, `docs/README.md`.
- User/project instructions: this file and
  `docs/working/user_message_memory_review.md`.
- Environment truth: `docs/design/environment/README.md`,
  `docs/design/environment/reconstruction_workflow.md`,
  `docs/design/environment/trace_loop_contract.md`,
  `docs/design/environment/scenario_schema.md`,
  `docs/design/environment/trace_schema.md`,
  `docs/design/environment/rulesets.md`,
  `docs/design/environment/deterministic_environment.md`, and
  `docs/working/environment/coverage_tracker.md`.
- Training truth: `docs/working/training_coach_packet.md`,
  `docs/working/training_loop_agenda.md`, `training_eval_protocol.md`,
  `training_smokes.md`, and `training_experiment_backlog.md`.
- Evidence index: `docs/experiments/README.md` plus dated experiment logs.
- Modal truth: `docs/design/modal_architecture.md`,
  `docs/runbooks/modal_environment_fidelity.md`, and
  `docs/research/modal_training_execution_plan.md`.

## Non-Negotiable Working Style

- Use simple language and explain clearly.
- Keep the main thread focused on orchestration and high-level synthesis.
- Delegate concrete research, critique, code edits, tests, and doc updates to
  scoped workers when parallelism helps.
- Send follow-ups to workers when new facts appear. Do not launch workers and
  forget them.
- Give every worker a narrow scope, owned files, enough context, and a required
  final answer shape.
- Require durable artifacts from workers: docs, tests, experiment logs, or exact
  command results. Invisible analysis is not enough.
- Respect concurrent work. Do not revert or overwrite edits made by others.
- Keep file ownership tight. If another worker owns a file, do not edit it from
  the main thread unless coordination requires it.
- Be skeptical of clean results. Check uniqueness, leakage, sampling bias,
  schema drift, seed policy, and artifact integrity.
- Prefer conservative, boring, inspectable architecture over clever control
  planes or taxonomy growth.
- If a result looks too clean, validate harder.
- If confused or drifting, stop briefly, reread current durable docs, state the
  north star and next concrete step, then do the step.
- Do not spend a turn on meta-memory recovery unless it directly unblocks the
  project.
- Never expose secrets or irrelevant personal data in project notes.

## Project Goals

- Build CurvyZero as a fresh ML/investigation repo for training strong agents in
  a CurvyTron-like game.
- Treat the original CurvyTron repo as reference evidence, provenance, possible
  demos, and golden-test source, not as the training hot loop.
- Build a deterministic Python simulator as the core asset.
- Start with a narrow, boring vertical slice: 1v1, no bonuses, deterministic
  reset/step, explicit rules, focused tests.
- Validate the simulator against source-derived behavior where useful.
- Use baseline learnability checks before serious MuZero work: random stress,
  heuristic advantage, and a simple policy baseline.
- Keep later training robust to variation, but do not let robustness variants
  silently change the source-fidelity target.
- Remember this is a multiplayer game. Even if training starts 1v1, source
  fidelity needs 3-player and 4-player canaries for map size, scoring, update
  order, and same-frame deaths.

## Architecture Decisions To Obey

- The simulator boundary is protected. Training libraries and adapters wrap the
  simulator; they do not define its core API.
- Keep package naming neutral under `src/curvyzero/`. Do not make MuZero, Mctx,
  LightZero, or PPO the permanent center too early.
- Keep `curvyzero-v0` and source-fidelity rulesets distinct:
  - `curvyzero-v0`: deliberate simplified training ruleset.
  - `curvytron-v1-reference`: behavior derived from the local original source.
  - `curvytron2-reference`: public CurvyTron2 behavior if needed later.
- Use decision records for implementation-shaping choices. Include evidence and
  reversal conditions.
- Keep current truth in `docs/design`, `docs/decisions`, and `docs/runbooks`.
- Keep evidence and exploration in `docs/research`, `docs/experiments`,
  `docs/sources`, `docs/handoffs`, and `docs/working`.
- Do not let experiment notes, handoffs, or critiques redefine schemas. They may
  point to canonical contracts.
- Prefer local single-scenario and local batch fidelity loops before Modal batch
  wrappers.
- Use common-trace diff as the normal comparison layer. Raw traces are debug
  artifacts.
- Use pass/fail/blocked distinctions. A blocked comparison is not a physics
  mismatch.

## Documentation Rules

- Top-level docs must be concise and human-readable.
- Deeper docs may be verbose, messy, and evidence-heavy.
- Promote stable conclusions upward. Do not let working notes become permanent
  junk drawers.
- Keep front-door docs current. Stale active paths are a serious failure mode.
- Write experiment logs with setup, exact commands, outputs/artifacts, result,
  and interpretation. Negative results count.
- Use names based on questions, mechanics, decisions, or artifacts, not agent
  names.
- Do not paste raw transcript dumps into docs. Summarize project-relevant
  instructions only.
- Mark uncertainty plainly. If source/probe evidence is not ready, say
  `pending` instead of guessing.
- Separate canonical contracts from support notes:
  - Canonical: reconstruction workflow, trace loop contract, trace schema,
    scenario schema, rulesets, deterministic environment, Modal architecture.
  - Support: handoffs, experiments, critiques, working notes.
- Keep `docs/working/user_message_memory_review.md` as a useful audit log. This
  packet supersedes it as the active instruction checklist, but not as evidence.

## Modal Rules

- Use Modal from the beginning for remote smokes, benchmarks, GPU checks, and
  artifact storage when feasible.
- Use Modal to keep the local machine clean for compute-heavy work.
- Keep Modal as a coarse job and artifact layer.
- Never put Modal calls inside `env.step()`, JS ticks, MCTS node expansion,
  action selection, normalization loops, or diff loops.
- Use Modal Functions for whole jobs: tests, benchmarks, self-play shards,
  training runs, evaluation runs, replay conversion, artifact packaging, and
  fidelity batches.
- Use Volumes or buckets for chunked artifacts: checkpoints, replay chunks,
  logs, profiles, videos, trace folders, and manifests.
- Use Queues and Dicts only for coarse coordination or tiny metadata. They are
  not per-action, per-tick, per-node, or per-inference tools.
- Make long Modal runs resumable, checkpointed, idempotent, and retry-safe.
- Use explicit Modal Images and deliberate source-copy patterns. Do not copy
  replay/checkpoints/runtime output into Images.
- Use deployed apps and `Function.from_name` only after smoke jobs are stable.
- Return structured run summaries and artifact refs, not giant logs.
- Use clearly named secrets. Document secret names, never secret values.
- Do cost checks before large GPU/MCTS/self-play runs.

## Environment Fidelity Rules

- Current north star: faithful environment reconstruction before performance,
  training throughput, or model work.
- Browser hosting and screenshot/pixel checks are later. State and events are
  the first proof.
- The reference source map is the fast orientation page. Read subsystem notes
  before changing behavior.
- The headless JS oracle is the first reference path. It should call original
  source objects, not reimplement physics in new JS.
- The reconstruction loop is:

```text
source map -> probe -> scenario -> common trace -> diff -> implement -> test
```

- Compare in this order: source facts, deterministic state traces, golden
  outcomes, replay/event logs, server messages, then screenshots/videos.
- Use common traces for meaningful comparison. Raw JS and Python traces can
  differ in runner metadata and shape.
- Movement comes first, then wall/border behavior, collision, trails, scoring,
  randomness, bonuses, multiplayer, observations, browser protocol, and pixels.
- `source-kinematics` is calibration for movement only. Do not let it grow into
  a parallel source clone for collisions, trails, scoring, or bonuses.
- Do not say plain `wall collision` when the distinction matters. Use:
  - normal-wall death,
  - fixed toy-v0 wall,
  - source borderless wrap.
- Source borderless is not a clean mathematical torus. It is source-specific
  edge teleporting from a timed bonus.
- Add multiplayer canaries early, especially 3-player and 4-player map size,
  scoring, update order, and same-frame death cases.
- Keep observation fidelity separate from source state fidelity. Do not block
  state parity on learned observation design, but do test observation fidelity
  before serious training claims.

## Training And Model Rules

- Baselines gate MuZero. Do not jump to MuZero before random stress,
  heuristic-vs-random, and one simple policy baseline exist.
- Prefer sparse terminal rewards at first. Add shaping only with evidence.
- Start with compact egocentric rays for baselines; consider heading-aligned
  local rasters for CNN/MuZero later.
- Use fixed debug seeds only for diagnosis. Each eval wave should sample fresh
  pseudo-random eval seeds, record the generator seed/list, and use many starts
  for claims. Do not tune against one reused eval seed list.
- Multi-agent self-play should start from ego-perspective shared policy and
  checkpoint/opponent pools. Avoid full joint-action search early.
- Mctx/JAX is the likely first MuZero search spike because batching and JIT
  matter, but it is not a full trainer. Prove it with a synthetic benchmark.
- LightZero is a contained PyTorch/MuZero fallback until wrapper and collector
  overhead are measured.
- `muzero-general` is reference/education, not the production base unless
  evidence changes.
- Performance matters later, potentially thousands of rollouts at once, but
  optimize only after measurement.

## Delegation And Subagent Rules

- The user wants aggressive parallelism, but not a vague swarm.
- Main thread job: keep the big picture, plan a few steps ahead, delegate,
  monitor, send follow-ups, synthesize, verify, and keep docs coherent.
- Worker job: own a narrow scope, inspect evidence, edit only allowed files,
  run scoped tests/experiments, and report exact paths/results.
- Critique lanes are valuable. Use them for repo structure, architecture,
  Modal design, environment fidelity, measurement, and simplification.
- If capacity is blocked by stale agents, close or replace them cleanly.
- Avoid `fork_context` when it creates bloated or brittle worker launches. Give
  workers exact local context instead.
- Use follow-ups to cut scope when workers take too long.
- Do not broaden a worker from docs into code, or from one module into a
  cross-repo cleanup, unless explicitly directed.
- Durable worker output matters more than status chatter.

## Open Concerns

- Scenario schema drift remains a real risk. Pick one current write shape or
  explicitly mark the documented future shape as future.
- Some docs may still describe completed work as future work. Keep front-door
  pages aligned with code.
- The whole repo is currently untracked in git, so `git diff` is not a reliable
  integration view. Read files directly when needed and avoid destructive
  assumptions.
- The original CurvyTron app has not been fully built from this checkout. Old
  Node/Gulp/Bower dependencies remain a blocker; use disposable or pinned
  environments for build attempts.
- Exact source/probe goldens are still pending for normal-wall death,
  wall-death scoring, source borderless wrap, trail behavior, self-collision
  latency, and same-frame multiplayer deaths.
- The batch/local fidelity artifact shape is intentionally small. Do not expand
  it until real remote or CI needs appear.
- Cost discipline is not yet first-class enough in experiment templates. Add
  cost fields before larger Modal GPU/MCTS/self-play sweeps.
- `docs/working/agent_roster.md` does not list every later wave. Treat it as
  lightweight, not authoritative.

## Main Agent Drift Audit

This audit covers the current main Curvy session activity visible in local
Codex state. It is concrete by design: where the main thread did useful work,
where it polluted itself, and what to do next.

### Where The Main Thread Got Polluted Or Rabbit-Holed

- The main thread became an orchestration log. The session trace shows heavy
  control activity: 87 subagent spawns, 73 waits, and 72 closes, plus many
  status updates. That is not automatically bad, but it means the main thread
  spent a lot of attention managing workers instead of holding a small agenda.
- It repeatedly broadened the surface before the local fidelity loop was stable:
  Modal hosting, browser/pixel checks, source-map waves, batch artifacts,
  observation fidelity, robustness/domain variation, and MuZero research all
  appeared while the core loop was still forming.
- It allowed documentation and schema vocabulary to multiply. Several docs
  described future scenario/common-trace shapes while fixtures and code used
  older or different field names.
- It used too much current-turn memory and raw state inspection after the user
  asked for reorientation. The instruction-review task was valid, but the next
  hour should not become another meta-memory workstream.
- It sometimes reacted to uncertainty by launching more workers instead of
  writing one short agenda and doing the next concrete local check.
- It almost grew `source-kinematics` into a tempting second implementation path.
  That runner is useful calibration, but it must stay movement-only.

### Where It Correctly Used Delegation, Docs, And Tests

- It correctly treated CurvyTron as a reference source and built CurvyZero as a
  fresh repo with a protected simulator boundary.
- It used workers with named scopes and owned files for source mining, Modal
  research, docs cleanup, JS oracle work, Python tracing, batch runner work, and
  critique.
- It repeatedly ran local verification and kept results visible: pytest, Ruff,
  JS oracle probes, single fidelity loop, source-kinematics loop, and local
  movement batch.
- It converted vague fidelity concerns into a concrete workflow:
  source map -> probe -> scenario -> common trace -> diff -> implement -> test.
- It corrected the wall-vs-torus assumption with source review and probe output,
  then updated docs to distinguish normal-wall death from source borderless
  wrap.
- It added useful durable docs: handoffs, source-map notes, trace-loop contract,
  probe backlog, Modal architecture, and experiment logs.
- It listened to critique workers and simplified some paths: common trace became
  default, batch reused the single loop, and Modal was pushed back to coarse
  batches.

### Under-Obeyed User Instructions

- "Keep the main thread clear" was only partly obeyed. The main thread did
  orchestration, but also became a dense stream of worker management, doc
  alignment, code review, and meta-recovery.
- "Use simple language" was mostly obeyed in final docs, but the system still
  accumulated schema names and compatibility terms that make the path harder to
  scan.
- "Do not complicate code while thinking clearly" was under pressure. The
  common loop is simple now, but compatibility fallbacks and multiple schema
  shapes could hide drift.
- "Keep docs up to date" was obeyed in volume, but not always in active truth.
  Some docs lagged behind code and had to be corrected later.
- "Do not get distracted" was under-obeyed around later lanes. Robustness,
  browser hosting, Modal wrappers, and MuZero research are useful, but should
  stay secondary until the local fidelity loop is boring.
- "Be comprehensive about CurvyTron source details" was obeyed through
  source-map workers, but the source-map wave should now feed the next scenario,
  not become endless reading.

### Guardrails For The Next Hour

1. Do not launch another broad wave. Use at most one worker, and only for a
   narrow source/probe question that directly supports the next scenario.
2. Pick one next local fidelity target: normal-wall death is the best candidate.
3. Convert the existing wall/border probe knowledge into one shared scenario
   fixture before adding any new architecture.
4. Make code and docs agree on the current scenario schema. Either migrate the
   fixture shape now or clearly mark the newer doc shape as future.
5. Run only the relevant local checks: focused tests, the single scenario loop,
   and the current batch if touched.
6. Do not touch Modal, browser hosting, MuZero, robustness variation, or
   observation fidelity during this hour unless the user redirects.
7. Keep `source-kinematics` movement-only.
8. Update only the canonical docs affected by the next step and one handoff if
   the current state changes.
9. End with exact changed paths and exact commands/results.
10. If the work starts to sprawl, write a three-line reset:
    north star, current gate, next concrete action.
