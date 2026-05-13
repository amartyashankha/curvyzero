# Environment Operating Patterns - 2026-05-13

Status: working memory for environment investigations.
Scope: how to work, not a fidelity claim.

Use this page when an environment thread needs to reorient quickly. The goal is
to turn docs into working memory: short enough to reread, concrete enough to
change behavior.

North star: faithful multiplayer CurvyTron environment first, then speed and
training integration. LightZero/training plumbing is a guarded downstream
interface, not the main focus.

## Reorientation Loop

1. Read the front-door docs before starting: `active_lanes.md`,
   `reorientation_packet.md`, and the newest focused reorientation note.
2. Name the surface being touched before reading code:
   source truth, product runtime, trainer/replay, or renderer.
3. Read source/docs/code in that order when the claim is semantic. Read
   runtime/tests/docs in that order when the claim is about existing product
   behavior.
4. Keep notes close to the evidence. Update the focused working doc when a
   finding changes the plan, boundary, or next test.
5. Validate the docs and the touched claim. For doc-only edits, run the
   environment doc guard and `git diff --check`.

Current active threads are multiplayer fidelity gaps, controls fidelity,
renderer/fast-path boundary, and docs/orchestration rhythm. Use those lanes to
decide what to delegate next.

## Spec-First Worker Pattern

Use this for any change that crosses source truth, product runtime,
trainer/replay, or renderer boundaries.

1. Main thread owns planning, delegation, orchestration, synthesis, and
   reorientation. It should keep docs as working memory and keep the route to
   faithful `VectorMultiplayerEnv` multiplayer behavior visible.
2. First worker catalogs exactly what must change before implementation starts:
   current behavior, target behavior, files likely touched, proof surface,
   validation commands, non-claims, and stale paths that must stay quarantined.
3. Implementation workers make bounded changes against that catalog. Keep code
   simple, clean, minimal, robust, and modular; do not invent fake observation
   paths, silent fallbacks, or parallel product runtimes.
4. Test/critic workers prove or criticize the exact surface changed. They
   should say whether evidence is source truth, product runtime,
   trainer/replay, renderer, or only route smoke.
5. Docs are updated last with what changed, what is proven, what is still open,
   and which focused note future agents should read first. Do not let docs
   claim more fidelity than the evidence surface proves.

## Truth Boundaries

- Source truth: original JS, oracle/probe output, and source-env parity. This
  answers what CurvyTron should do.
- Product runtime: `VectorMultiplayerEnv` and its runtime kernel behavior. This
  answers what the intended environment does.
- Trainer/replay: wrapper observations, rewards, masks, final observations,
  sidecars, replay records, and autoreset behavior. This answers what training
  and later analysis will actually receive. It is downstream of environment
  reconstruction.
- Renderer: source-state/native render modes, approximate speed modes, gray64,
  bonus diagnostics, and eventual browser/canvas pixels. This answers what a
  visual claim actually proves.

Do not close a gap by mixing surfaces. A source-state visual pass is not
trainer replay proof. A metadata replay row is not full replay arrays. A route
smoke is not environment fidelity. A fast approximate render mode is not the
default source-state visual gate.
No-death/profile/training-helper modes are project features to preserve with
explicit metadata, not source-fidelity evidence.

## Working Habits

- Keep the main thread focused on plan, delegation, orchestration, synthesis,
  and hard decisions.
- Use subagents or follow-up threads for bounded work with a named question,
  files to inspect, and expected output.
- Start each bounded investigation with a short to-do list and end it with a
  finding, evidence links, remaining risk, and exact next step.
- Maintain lightweight experiment and to-do logs. Delete or close stale agents
  and stale branches of thought when they stop affecting the plan.
- Avoid rabbit holes by asking: which surface is this about, what would change
  if it is true, and what command or source read would settle it?
- Prefer one source-backed claim at a time over broad cleanup unless stale docs
  are actively misleading orchestration.
- Keep old toy/debug paths quarantined as historical smoke/interface evidence
  unless a current doc explicitly promotes them for a narrow proof surface.
- Keep optimized and approximate render paths labeled and separate from engine
  rule fidelity.

## Documentation Rule

Docs should say what is known, what is not known, and what evidence would move a
claim. Avoid broad status dashboards. Link to focused evidence instead of
copying logs. When a finding changes the plan, update the front-door docs or add
a narrow note with a link from the front door.

## Validation

For doc-only environment edits:

```bash
python3 scripts/check_environment_doc_status.py docs/working/environment
git diff --check
```

For code or behavior changes, add the focused source/product/trainer/renderer
test that matches the surface being changed, then record the command in the
focused doc.
