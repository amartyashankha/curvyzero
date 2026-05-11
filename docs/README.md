# Documentation Map

This documentation tree is intentionally hierarchical.

Top-level pages should stay short, stable, and readable by a human who has five minutes. Deeper pages can be more detailed, experimental, and messy. Raw notes, handoffs, logs, and scratch reasoning belong lower in the tree so the project can move fast without losing the plot.

For the proposed docs/code hierarchy this tree should grow toward, see [design/repository_hierarchy.md](design/repository_hierarchy.md).

## Main Sections

- [design/](design/) - current architecture, API design, and repository hierarchy.
- [decisions/](decisions/) - decision records with status, evidence, and reversal conditions.
- [research/](research/) - focused investigations and critiques.
- [runbooks/](runbooks/) - how to run local, Modal, training, evaluation, and debugging flows.
- [experiments/](experiments/) - experiment plans, logs, results, and benchmark reports.
- [sources/](sources/) - source index, provenance, and external references.
- [handoffs/](handoffs/) - compressed context packets for future agents or humans.
- [working/](working/) - messy temporary working memory that should later be promoted or deleted.

## Stability Layers

- Stable maps: this page and section READMEs should stay short and link-heavy.
- Current truth: `design/`, `decisions/`, and `runbooks/` should be coherent enough to implement or operate from.
- Evidence: `research/`, `experiments/`, and `sources/` can be deeper, more detailed, and occasionally contradicted.
- Transfer and scratch: `handoffs/` and `working/` preserve context without crowding the stable reading path.

## Current Investigation Lanes

- Game fidelity: source-derived CurvyTron behavior, explicit v0 rule choices, golden tests.
- Environment fidelity: [design/environment/](design/environment/) maps run/compare/single-step/Modal/iteration questions; details live in [research/environment/](research/environment/) and [working/environment_questions.md](working/environment_questions.md).
- Deterministic simulator: reset/step API, collision representation, rewards, vectorization hooks.
- Training architecture: baselines, replay, MuZero/MCTS, evaluation, league/self-play.
- Modal architecture: GPU jobs, storage, snapshots, sandboxes, reproducibility, cost.
- Performance path: Python first, then measured migration to NumPy/Numba/JAX/PyTorch/native code.
- Repository hierarchy: concise overview first, deep evidence and logs below, neutral code packages under `src/curvyzero/`.

## Promotion Rule

When a research note becomes important enough to guide implementation, promote the distilled decision into [decisions/](decisions/) or [design/](design/), and leave the raw note as supporting evidence.
