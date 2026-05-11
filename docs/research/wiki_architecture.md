# Wiki Architecture

## Short Answer

Keep the documentation tree layered by stability and audience:

1. Top-level maps stay concise and tell a reader where to go next.
2. Design docs, decision records, and runbooks hold current operating truth.
3. Research notes and experiment logs hold detailed reasoning and reproducible evidence.
4. Sources, raw handoffs, and working memory preserve provenance without bloating the first reading path.

The project should optimize for fast agent handoff and later auditability: every important claim should either be promoted into a stable doc or linked to evidence deeper in the tree.

## Proposed Hierarchy

```text
docs/
  README.md                         # five-minute map; no long arguments
  design/                           # current intended architecture
    README.md
    deterministic_env.md
    modal_training.md
  decisions/                        # short ADRs with evidence and reversal conditions
    README.md
    0001-investigation-first-repo-structure.md
  runbooks/                         # commands and procedures that should work now
    README.md
    local_smoke.md
    modal_gpu_smoke.md
  research/                         # focused investigations and synthesis
    README.md
    wiki_architecture.md
    training_architecture_notes.md
  experiments/                      # plans, logs, benchmark reports, and result summaries
    README.md
    2026-05-08-mctx-synthetic-benchmark.md
  sources/                          # source index and evidence packets
    README.md
    index.md
    2026-05-08-curvytron-source-index.md
  handoffs/                         # compact context packets for humans/agents
    README.md
    2026-05-08-initial-handoff.md
  working/                          # scratch notes, raw handoffs, temporary memory
    README.md
    raw-handoffs/
    scratch/
```

Top-level files should answer "what is true now?" and "where do I look next?" They should not preserve all evidence inline. Deeper files should preserve the full trail: source notes, experiment output summaries, caveats, failed attempts, raw handoffs, and agent working memory.

## Layer Rules

| Layer | Folder | Reader Promise | Max Shape |
| --- | --- | --- | --- |
| Map | `docs/README.md`, section READMEs | Fast orientation and links. | Short, stable, skimmable. |
| Current truth | `design/`, `decisions/`, `runbooks/` | Implement from this without reading every old note. | Curated, dated where useful. |
| Investigation | `research/` | Explain options, evidence, risks, and recommendations. | Verbose enough to defend the conclusion. |
| Reproducibility | `experiments/` | Rerun or critique a claim. | Append-only logs plus concise interpretation. |
| Provenance | `sources/` | Find source material and know how trustworthy it is. | Indexed, source-by-source. |
| Transfer | `handoffs/` | Resume work quickly after context loss. | Compressed, link-heavy. |
| Scratch | `working/` | Do messy thinking without polluting stable docs. | Temporary; promote or delete. |

## Naming Conventions

- Use lowercase kebab-case for topic docs: `training-architecture.md`, `modal-storage.md`.
- Use numbered ADRs: `0001-short-decision-title.md`.
- Use dated experiment logs: `YYYY-MM-DD-short-experiment-name.md`.
- Use dated handoffs: `YYYY-MM-DD-short-context-label.md`.
- Use raw imports under `working/raw-handoffs/YYYY-MM-DD-source-slug.md` until distilled.
- Use source IDs in `docs/sources/index.md`: `S001`, `S002`, and so on.
- Prefer names that describe the question or decision, not the author or agent.
- Rename or supersede stale docs rather than letting conflicting truths share equal status.

## Decision Record Rules

Decision records should remain short and reversible. One ADR should capture one implementation-shaping choice.

Required fields:

- `Date`: the date the decision was written or updated.
- `Status`: `Proposed`, `Accepted`, `Superseded`, or `Rejected`.
- `Context`: the pressure that made the decision necessary.
- `Decision`: the chosen path in direct language.
- `Evidence`: links to research notes, experiment logs, source IDs, or handoffs.
- `Consequences`: what this makes easier and harder.
- `Reversal Conditions`: what evidence would cause the team to change course.
- `Links`: related docs, issues, PRs, or artifacts.

Rules:

- Do not paste raw experiment output into ADRs; link to the experiment log.
- Do not silently edit history when a decision changes; mark the old ADR `Superseded` and link the replacement.
- Promote a research conclusion into an ADR when it affects repo layout, public APIs, training architecture, storage, reproducibility, evaluation, or cost.
- Keep uncertainty visible. A `Proposed` ADR is useful when the team needs a provisional default before experiments finish.

## Experiment Log Rules

Experiment logs should be reproducible enough for another person or future agent to rerun, inspect, or challenge.

Required fields:

- `Question`: the claim being tested.
- `Setup`: hardware, software versions, config, seed policy, data, and relevant commit or snapshot.
- `Command`: exact commands or Modal entry points.
- `Results`: concise tables or summaries; link large artifacts instead of pasting them.
- `Interpretation`: what changed in project belief.
- `Artifacts`: checkpoints, plots, logs, videos, profiler traces, or storage paths.
- `Follow-ups`: next run, design change, ADR candidate, or dead end.

Rules:

- Write the log on the day of the run, even if the result is messy.
- Separate compile/setup time from steady-state runtime for JAX, Modal, and GPU experiments.
- Record negative results; they are often the best evidence for architecture choices.
- Keep large raw logs outside top-level docs and link them from `Artifacts`.
- Promote stable conclusions to `research/`, `design/`, or `decisions/`.

## Source And Evidence Index

`docs/sources/index.md` should become the canonical source ledger. Each source entry should include:

- Source ID, such as `S001`.
- URL or local path.
- Access date.
- Source type: `primary`, `official-doc`, `paper`, `issue`, `community-report`, `local-handoff`, or `experiment-artifact`.
- Authority grade: `authoritative`, `suggestive`, or `risk-signal`.
- Short summary of the relevant claim.
- Supported docs or decisions.
- Snapshot or archival note when the source may change.

Evidence should be cited by ID from ADRs and design docs. Research notes may cite direct links while drafting, but any source that becomes important to implementation should be promoted to the index.

Initial local sources to index:

- `curvytron_muzero_modal_handoff.md`: broad initial handoff, source list, repo topology, Modal guidance, and implementation gates.
- `docs/README.md`: current documentation map and promotion rule.
- `docs/decisions/0001-investigation-first-repo-structure.md`: provisional repository-structure decision.

## Compaction And Handoff Strategy

Handoffs should be compact packets, not full transcripts. They exist to survive context loss and let the next agent continue without rereading every raw note.

Each handoff should include:

- Current state in ten bullets or fewer.
- Accepted and proposed decisions since the last handoff.
- Commands, tests, and experiments run, with links to logs.
- Files changed and why.
- Open questions, blockers, and the next three actions.
- Links to research notes, ADRs, source IDs, and artifacts instead of pasted evidence.

Raw handoffs, long transcripts, and uncompressed working memory should land in `docs/working/raw-handoffs/` first. When they become useful, distill them into:

- `docs/handoffs/` for a compact resumable packet.
- `docs/research/` for an investigation.
- `docs/decisions/` for a choice.
- `docs/sources/` for provenance.

At each major milestone, create a new handoff and prune or promote stale `working/` files. The goal is not to keep the tree tidy for its own sake; it is to make the next context window start with the right facts.

## Promotion Workflow

1. Capture messy thinking in `working/` or `research/`.
2. Run experiments and store logs in `experiments/`.
3. Index important sources in `sources/`.
4. Distill current truth into `design/` or `runbooks/`.
5. Record implementation-shaping choices in `decisions/`.
6. Write a compact handoff when context is likely to be lost.

## Risks

- If top-level docs become too detailed, future readers will stop trusting them as maps.
- If raw notes never get promoted, the project will accumulate evidence but lose decisions.
- If experiments omit setup details, benchmark results will be hard to compare.
- If handoffs paste too much history, they will fail at their main job: quick resumption.

## Open Questions

- Should `docs/sources/index.md` be one global file, or should high-volume areas such as Modal, CurvyTron rules, and MuZero libraries get separate source ledgers?
- Should experiment artifacts live only in external storage, or should small canonical artifacts be checked into the repo?
- How often should `working/` be pruned: weekly, per milestone, or before every handoff?

## Sources

- `docs/README.md`
- `docs/experiments/README.md`
- `docs/handoffs/README.md`
- `docs/sources/README.md`
- `docs/working/README.md`
- `docs/decisions/README.md`
- `docs/decisions/0001-investigation-first-repo-structure.md`
- `curvytron_muzero_modal_handoff.md`
