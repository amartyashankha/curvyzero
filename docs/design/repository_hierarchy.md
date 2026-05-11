# Repository Hierarchy

Status: Draft

This page proposes the shape the repository should grow toward. It is intentionally
deeper than the current tree, but the first reading path should stay short:
top-level maps explain where to go, while detailed evidence and messy work live
lower in the tree.

## Principles

- Keep stable maps short. `docs/README.md` and section READMEs should orient a
  reader in minutes, not preserve every argument.
- Promote conclusions upward. Research, experiments, and working notes can be
  noisy; decisions, design docs, and runbooks should describe current intent.
- Keep package names neutral. Long-lived code should not assume MuZero, Mctx,
  LightZero, PPO, or any other method is the permanent center of the project.
- Protect the simulator boundary. Training libraries and framework adapters
  should wrap `curvyzero.env`; they should not define the core reset/step/state
  contract.
- Avoid speculative empty folders. Add deeper folders when there are enough real
  files or duplicated concepts to justify the extra navigation.

## Documentation Layers

The current top-level documentation folders are good. The main cleanup is to make
the stability level explicit:

```text
docs/
  README.md          stable map of the docs tree
  design/            current architecture and implementation contracts
  decisions/         ADRs for choices with consequences and reversal conditions
  runbooks/          commands and procedures expected to work now
  experiments/       dated runs, benchmarks, and measured results
  sources/           source ledger and provenance
  research/          focused investigations, critiques, and synthesis
  handoffs/          compact context packets for people or agents
  working/           scratch space, inboxes, and raw temporary memory
```

The deeper, messier layers are `research/`, `experiments/`, `sources/`,
`handoffs/`, and `working/`. Those folders may contain false starts, contradicted
ideas, raw source notes, and detailed logs. Stable pages should link to them as
evidence instead of duplicating their contents.

If `docs/research/` becomes hard to scan, split it by question family rather than
by author or agent:

```text
docs/research/
  rules/             CurvyTron source behavior, v0 rule choices, fidelity gaps
  simulator/         environment API, collision, observations, rewards
  training/          baselines, MuZero, self-play, replay, evaluation
  infra/             Modal, storage, reproducibility, job orchestration
  performance/       vectorization, throughput, profiling, acceleration
  docs/              documentation and repository-structure critiques
```

Do not move current research files while other agents may still be writing to
fixed paths. Treat this split as a future migration once active work is quiet, or
use it only for new files after coordination.

## Code Package Map

The current package starts in the right place:

```text
src/curvyzero/
  env/               deterministic simulator core
  infra/modal/       Modal smoke tests and remote entrypoints
```

Future package subdomains should appear when implementation pressure makes them
useful:

```text
src/curvyzero/
  env/               reset/step API, config, state, rules, collision, obs, rewards
  agents/            random, heuristic, scripted, frozen, and opponent policies
  eval/              seed sets, matches, leagues, metrics, regression gates
  replay/            trajectory schema, replay chunks, sampling, storage IO
  models/            neural network definitions and framework-specific modules
  search/            MCTS/MuZero search interfaces and library adapters
  training/          learner loops, target construction, optimization, checkpoints
  adapters/          Gymnasium, PettingZoo, LightZero, Mctx, or demo wrappers
  infra/             Modal, storage, images, secrets, cloud job utilities
  viz/               debug renderers, replay viewers, and inspection utilities
```

`env/` should remain the most stable package. A good future internal split is:

```text
src/curvyzero/env/
  config.py          behavior-affecting rules and hashable configuration
  state.py           fixed-shape state containers
  core.py            reference single-environment reset/step implementation
  collision.py       collision backends and equivalence tests
  observations.py    ego features, rasters, and observation schemas
  rewards.py         terminal and shaped reward variants
  vectorized.py      reset_many/step_many once single-env semantics are proven
  rulesets.py        named v0/v1 configs with provenance labels
```

The first version does not need all of these files. Add them when `core.py`
starts mixing unrelated responsibilities or when tests need a clearer contract.

## What Stays Out Of `src`

- One-off benchmark drivers belong in `scripts/` or dated experiment logs until
  they become reusable package code.
- Raw source mining, handoffs, and critique belong under `docs/`.
- Vendored or pinned reference material belongs under `third_party/`, with
  provenance documented in `docs/sources/`.
- Framework spikes should be thin adapters first. Only promote them into stable
  packages after measured smoke tests justify the dependency and API shape.

## Promotion Gates

Promote code into a stable package when at least one of these is true:

- Two real callers need the same behavior.
- Tests need a named boundary to protect semantics.
- A runbook or Modal job depends on it.
- An experiment result says the boundary is hot, fragile, or worth optimizing.

Promote documentation upward when it starts guiding implementation:

- `working/` item becomes a research question, source note, design doc, ADR, or
  experiment.
- `research/` conclusion becomes current intent in `design/` or a choice in
  `decisions/`.
- `experiments/` result changes a default, benchmark target, or reversal
  condition.

## Near-Term Cleanup

- Keep the existing top-level docs hierarchy.
- Link this page from `docs/README.md` and `docs/design/README.md`.
- Keep `docs/research/` flat for now, but make its index match the files that
  actually exist.
- Continue using neutral package names under `src/curvyzero/` while the training
  approach is still being measured.
