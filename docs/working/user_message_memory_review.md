# User Message Memory Review

Date: 2026-05-08

Latest refresh note, 2026-05-09: the active instruction checklist is now
[user_instruction_packet.md](user_instruction_packet.md). A newer raw-session
review found 1,610 session files, 27,528 session user-message events, 7,184
history lines, 8,190 exact-unique user messages, and 3,209 indexed threads with
395 in `/Users/shankha/curvy`. This file remains an audit trail, not the
front-door instruction source.

Scope: local Codex/session state review for durable user instructions,
preferences, and CurvyZero ideas that may not yet be documented. This note
summarizes user intent and redacts by omission: no raw transcript dumps, no API
keys, no credentials, and no irrelevant personal content.

## Privacy Notes

- Did not inspect `/Users/shankha/.codex/auth.json`.
- Did not copy raw transcript text into this document.
- Treated session JSONL and history as sensitive local state; extracted only
  durable patterns and CurvyZero-relevant ideas.
- Ignored plugin cache content except where tooling paths were listed by file
  discovery.

## Local State Inspected

- `/Users/shankha/.codex/state_5.sqlite`
  - Schema inspected.
  - `threads` queried for CurvyZero sessions and rollout paths.
  - Found 2,752 indexed threads, 24 with `cwd=/Users/shankha/curvy`.
  - `stage1_outputs` existed but had 0 rows.
- `/Users/shankha/.codex/logs_2.sqlite`
  - Schema/count inspected.
  - Found 1,148,869 log rows; sampled only enough to determine it is mostly
    Codex telemetry/tool traces, not a better source of user intent.
- `/Users/shankha/.codex/state.db`
  - Checked for tables; none were returned.
- `/Users/shankha/.codex/session_index.jsonl`
  - Used as an index signal only.
- `/Users/shankha/.codex/history.jsonl`
  - Sampled beginning and end; used for cross-project durable preference
    patterns and CurvyZero messages.
- `/Users/shankha/.codex/config.toml`
  - Inspected for non-secret Codex configuration, trusted project paths, enabled
    plugins, and MCP names.
- `/Users/shankha/.codex/sessions/...`
  - Parsed user-message fields from local rollout JSONL paths referenced by
    `state_5.sqlite`.
  - Read CurvyZero session messages under
    `/Users/shankha/.codex/sessions/2026/05/08/`.
  - Ran a keyword-filtered all-session user-message scan for durable workflow
    instructions.
- `/Users/shankha/.codex/memories/watchdog_role_brief.md`
- `/Users/shankha/.codex/memories/captain_overnight_reanchor.md`
- `/Users/shankha/.codex/memories/codex_watchdog_morning_checklist.md`
- `/Users/shankha/.codex/memories/captain_sequential_gate_nudge_v1.txt`
- Current repo docs inspected for overlap:
  - `README.md`
  - `curvytron_muzero_modal_handoff.md`
  - `docs/README.md`
  - `docs/working/README.md`
  - `docs/working/ideas_inbox.md`
  - `docs/working/agent_roster.md`
  - `docs/design/modal_architecture.md`
  - `docs/design/repository_hierarchy.md`
  - `docs/research/wiki_architecture.md`
  - `docs/research/modal_patterns.md`
  - `docs/research/modal_example_patterns.md`
  - `docs/decisions/0001-investigation-first-repo-structure.md`
  - `docs/decisions/0002-modal-hot-loop-locality.md`

## Enduring Workflow Preferences

- Read the existing docs and code deeply enough to build a holistic picture
  before making confident claims.
- Keep the main thread clean for high-level planning, synthesis, delegation,
  and user-facing decisions.
- Use parallel subagents aggressively for bounded research, critique, source
  mining, experiments, and docs work, then synthesize their results.
- Give subagents precise scope, enough context, owned files, and explicit final
  answer requirements.
- Use follow-ups with subagents as new information arrives; do not launch once
  and forget them.
- Require subagents to leave durable artifacts. The user dislikes invisible
  analysis that must be redone from memory.
- Respect concurrent work. Repeated instruction: do not revert or overwrite
  others' edits, and keep owned write scopes narrow.
- Prefer plain language. Avoid jargon, "bullshit terms", hidden behavior,
  vague claims, and elaborate terminology unless it pays for itself.
- Keep docs and planning files current as working memory. The user explicitly
  treats documentation as compensation for short context windows.
- Be critical, including of prior docs and agent outputs. Clean results should
  invite validation, not complacency.
- Prefer simple, clean, separated architecture over clever control planes,
  taxonomy bloat, or framework churn.
- Use current repo state and current durable docs as authority over stale
  thread memory. Use session-history recovery when drift or ambiguity appears,
  but do not let meta-memory become the workstream.
- When scaling anything, first check uniqueness, sampling bias, cost, and
  measurement truth. A prior repeated-run failure made this a strong durable
  warning.

## Repo And Docs Hierarchy Preferences

- Top-level docs should be concise, human-readable maps.
- Deeper layers should hold verbose evidence, failed attempts, experiment logs,
  raw handoffs, scratch notes, and messy working memory.
- Stable current truth belongs in `docs/design`, `docs/decisions`, and
  `docs/runbooks`.
- Evidence and exploration belong in `docs/research`, `docs/experiments`,
  `docs/sources`, `docs/handoffs`, and `docs/working`.
- Promote useful findings upward; do not let `working/` and `research/` become
  permanent junk drawers.
- Use decision records for implementation-shaping choices, with evidence and
  reversal conditions.
- Experiment logs should include setup, exact commands, results, artifacts, and
  interpretation. Negative results are useful evidence.
- Prefer names based on questions or decisions, not on agent names.
- Keep code package names neutral while the training method is not settled.
  Avoid making MuZero, Mctx, LightZero, or PPO the permanent center too early.
- Protect the simulator boundary. Training frameworks and adapters should wrap
  the environment, not define its core reset/step/state contract.
- Avoid speculative empty directories. Add structure when real files or repeated
  concepts justify it.
- Do not move files while other agents are still writing to fixed paths.

## Modal Preferences

- Use Modal from the beginning for remote smoke tests, benchmarks, and GPU
  checks when feasible, partly to keep the local machine clean.
- Treat Modal as the coarse compute and artifact layer, not the per-step runtime.
- Keep environment stepping, MCTS expansion, inference batches, replay sampling,
  and training updates inside one process/container hot loop whenever possible.
- Use Modal Functions for coarse jobs such as train runs, self-play shards,
  evaluation shards, benchmarks, replay conversion, and artifact packaging.
- Use Volumes and buckets for chunked artifacts: checkpoints, replay, logs,
  profiles, videos, and manifests.
- Use Queues and Dicts only for coarse coordination and tiny metadata. They are
  not per-action, per-tick, per-node, or per-inference hot-loop tools.
- Make long Modal runs resumable, checkpointed, idempotent, and retry-safe.
- Use explicit Modal Images and source-copy patterns. Avoid accidental giant
  images or copying replay/checkpoint/runtime output into Images.
- Use deployed apps and `Function.from_name` after smoke jobs are stable.
- Prefer structured run summaries and exact artifact refs over downloading or
  pasting giant logs.
- Modal secrets should be named clearly and specifically. If an OpenAI key is
  needed in Modal, use a clearly named OpenAI secret rather than a generic one.
  Do not document secret values.
- Do cost checks before large runs. A medium sample with real API usage is
  preferred over speculative cost math when possible.

## Subagent Orchestration Preferences

- The user wants many parallel subagents, but bounded and purposeful, not a
  vague swarm.
- Main thread responsibilities: keep the big picture, plan a few steps ahead,
  delegate, monitor, send follow-ups, synthesize, and keep docs coherent.
- Worker responsibilities: own a narrow scope, inspect evidence, edit only
  allowed files, run scoped tests/experiments, and report exact paths/results.
- Critique lanes are valuable. The user repeatedly asks for skeptical review of
  repo structure, training assumptions, Modal design, and failure patterns.
- If capacity is blocked by stale agents, shut them down or replace them with a
  cleaner wave.
- Existing role pattern from local memory:
  - Captain owns the core agenda and next concrete unblock.
  - Quartermaster handles bounded support or larger well-defined tasks.
  - Watchdog preserves continuity and re-anchors from current durable docs; it
    should not become the captain.
- Durable agent artifacts matter. `docs/working/agent_roster.md` exists, but the
  user preference is stronger: agent outputs should leave reusable notes, not
  just status chatter.

## CurvyZero Project Ideas Recovered

- CurvyZero should be a fresh ML/investigation repo. CurvyTron source is a
  reference, not the main training runtime.
- Clone or vendor CurvyTron for rule mining, provenance, possible demos, and
  golden-test candidates. Running the original may not be necessary if source
  inspection answers the important questions.
- The environment is the core asset. Build a deterministic Python simulator with
  explicit seed handling, simultaneous actions, action repeat, collision
  semantics, rank/tie rewards, replayability, and golden tests.
- Start simple: 1v1, no bonuses, one round per episode, fixed speed/turn rate,
  explicit v0 rule choices.
- Source fidelity matters for movement, turn dynamics, collision scale, wall
  rules, spawn constraints, scoring/ties, trail gaps, bonuses, timing, and
  defaults. Explicitly label any v0 inventions.
- Later training should vary environment parameters after the clone is faithful
  enough: arena size, speed, turn rate, action repeat, trail gaps, collision
  backend, spawn sampler, scoring, observations, and bonuses.
- Performance will matter because training should run far faster than real time,
  potentially thousands of rollouts at once.
- Start with readable Python, then use measured pressure to move toward NumPy,
  Numba, PyTorch tensor environments, JAX-native environments, or native
  extensions.
- Use occupancy-grid or swept collision paths carefully; naive endpoint checks
  and continuous trails create edge cases.
- Baseline learnability should gate MuZero. Prove random stress, heuristic
  advantage, and PPO or imitation/PPO against random before serious MuZero.
- Use fixed debug seeds only for diagnosis. Each eval wave should sample fresh
  pseudo-random eval seeds, record the generator seed/list, and use many starts
  for claims. Do not tune against one reused eval seed list.
- Avoid dense reward shaping at v0. Prefer curriculum or clearer observations
  before adding shaping that MuZero would inherit.
- Observation path: start with compact egocentric rays for baselines; consider
  heading-aligned local rasters for CNN/MuZero.
- Multi-agent self-play should start from an ego-perspective shared policy and
  checkpoint/opponent pools. Avoid full joint-action search early.
- MuZero research should be implementation-oriented: model signatures,
  representation/dynamics/prediction, action encoding, reward/value/policy
  heads, replay fields, target construction, weight handoff, actor/trainer/
  evaluator loops, inference batching, and MCTS integration.
- Mctx is the likely JAX search spike because batching/JIT matters, but it is
  not a full trainer. It needs a synthetic benchmark before commitment.
- LightZero should remain a contained PyTorch/MuZero alternative, not a default
  backbone, until wrapper and collector overhead are measured.
- `muzero-general` is educational/reference material, not the production base
  unless evidence changes.

## Warnings And Constraints

- Do not treat current raw session memory as more authoritative than current
  repo docs and artifacts.
- Do not paste raw user messages, credentials, private paths with secrets, or
  large transcript fragments into project docs.
- Do not broaden scoped workers into unrelated edits.
- Do not call a result trustworthy without checking uniqueness, held-out
  separation, seed policy, artifact integrity, and measurement bugs.
- Do not let docs imply an old path is active. Stale front-door docs are a
  serious failure mode in this user's workflows.
- Do not overfit architecture to an early transport shape, row taxonomy,
  framework API, or one clean experiment.
- Do not commit to JAX/Mctx, LightZero, Modal distribution patterns, replay
  storage, or vectorized backends before small measured spikes.
- Keep local-vs-Modal parity visible: exact commands, environment/package
  versions, seed ranges, artifacts, and where results live.
- Keep secret handling boring and explicit. Values should live in secrets or
  local env files, never in notes or run summaries.

## Not Yet Fully Reflected In Current Docs

- The CurvyZero docs capture most technical ideas, but the explicit
  "main thread as planner/delegator, subagents as bounded artifact-producing
  workers" protocol is only partially captured in `docs/working/agent_roster.md`.
- The user's cross-project preference for plain-language active-state docs and
  stale-doc purges is not yet stated as a CurvyZero documentation rule.
- "Run on Modal from the get-go to keep the local machine clean" is stronger
  than the current docs' more balanced local-plus-Modal wording. Decide whether
  this should become a runbook default or remain a preference.
- Cost discipline before scale is not yet a CurvyZero first-class rule. It
  should probably be added before GPU/MCTS sweeps or large self-play jobs.
- The local Codex memory review itself was not documented before this file.
- The first wave intended notes such as `docs/research/deterministic_env_notes.md`
  and `docs/research/modal_repro_notes.md`; later docs cover much of this, but
  a future cleanup should check whether any intended findings were never
  promoted.
- `docs/research/wiki_architecture.md` includes some proposed filenames that do
  not match the current tree exactly. Treat it as a proposal, not current truth.
- `docs/working/agent_roster.md` does not yet list all later CurvyZero worker
  waves and status results. It may need a refresh after active workers settle.

## Suggested Promotions

- Add a short "Agent Orchestration" note under `docs/working/` or
  `docs/handoffs/` that records the CurvyZero-specific worker protocol.
- Add a plain-language "Documentation Hygiene" rule to `docs/README.md` or
  `docs/working/README.md`: current front-door docs must not point to stale
  active paths.
- Add cost/accounting fields to future experiment-log templates before Modal
  GPU sweeps.
- Promote remote-first Modal command examples into `docs/runbooks/` after the
  current smoke/benchmark work stabilizes.
