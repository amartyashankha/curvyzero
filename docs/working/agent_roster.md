# Agent Roster

This page tracks parallel subagents and the notes they are expected to produce. It is intentionally lightweight; promote durable findings elsewhere.

## Wave 1

These agents were started, then shut down to free capacity for a cleaner second wave:

- CurvyTron reference investigation -> intended `docs/research/curvytron_reference_notes.md`
- Training architecture -> wrote/started `docs/research/training_architecture_notes.md`
- Deterministic environment -> intended `docs/research/deterministic_env_notes.md`
- Repo structure critique -> intended `docs/research/repo_structure_critique.md`
- Modal reproducibility -> intended `docs/research/modal_repro_notes.md`
- Modal patterns -> intended `docs/research/modal_patterns.md`

## Wave 2

- `019e0891-6a00-7d13-bf6e-4916b9ea5605` / Lagrange: repo structure critique. Completed `docs/research/repo_structure_critique.md`.
- `019e0891-6aea-7d01-bde1-4a32b7e76b35` / Heisenberg: Modal patterns.
- `019e0891-6bea-7ab2-b1fa-d37853d951b0` / Epicurus: environment fidelity and curriculum.
- `019e0891-6cfe-7760-b82d-8196d1af24a7` / Hegel: performance/vectorization path.
- `019e0891-6e34-7c53-814d-c59b2fcd5ee2` / Halley: wiki architecture.

## Wave 3

- `019e0892-de43-7b62-bd4b-c093afd5ee0b` / Averroes: multiplayer self-play and MuZero/MCTS prior work.
- `019e0893-c723-7850-abbf-08b2ed568c04` / Nash: CurvyTron source mining.
- `019e0894-0979-7e43-b855-05e14131a2cc` / Dalton: baseline learnability before MuZero.
- `019e0894-f42a-7280-8f55-fb3d07c6e0b5` / Linnaeus: deep MuZero architecture.
- `019e0899-ef05-70b3-984a-f63189960c1b` / Aristotle: JAX/Mctx integration.
- `019e0899-f084-7e33-9947-a36d65f8791d` / Hubble: LightZero integration critique.
- `019e0899-f194-7293-a6ac-a48fdd250759` / Mill: observation/reward design.
- `019e089d-076e-7502-b597-513cedd7c40c` / Cicero: Modal example pattern extraction.

## Current Wave

Started 2026-05-09 after the smoke matrix passed with vector pytest `56`,
mixed comparator `20`, batch quick `19`, and actor quick `19`.

- `019e0d42-5078-75b3-b797-74870e7c87e8` / Hilbert the 3rd:
  trainer observation contract implementation. Owns trainer contract code,
  focused tests, and trainer-contract docs.
- `019e0d42-518f-78b2-a1a6-c3171132c737` / Maxwell the 3rd:
  actor bridge, policy-row mapping, and local replay writer wiring. Owns
  training helper code/tests and LightZero handoff docs.
- `019e0d42-52b6-7681-b893-6f2978e052ea` / Anscombe the 3rd:
  speed and parallelism measurements. Owns speed docs and, if useful, one
  small benchmark script or test.
- `019e0d42-afaf-7b73-9304-96c5fe83a3ce` / Nietzsche the 3rd:
  critique and reorientation. Owns `reorientation_packet.md` only; no
  implementation code.
- `019e0d47-5998-7370-a51e-568165fe65a6` / Nash the 3rd:
  stale-status guard for working docs. Owns one new script and one new test.

## Current Wave 2

Started after full pytest was green at `271 passed in 4.81s`.

- `019e0d4e-ccc6-7272-9a3a-1b505b0491fb` / Lagrange the 3rd:
  reset/autoreset production-helper slice. Owns vector reset helper code,
  focused vector tests, and the reset plan.
- `019e0d4e-cede-78b1-8039-569b46335340` / Beauvoir the 3rd:
  no-train LightZero-shaped CurvyZero adapter smoke. Owns new training adapter
  code/tests and LightZero handoff docs.
- `019e0d4e-d019-7fc0-80d4-63768db9410e` / Mill the 3rd:
  observation fidelity fixture/manifest work. Owns observation docs and focused
  trainer observation tests.

## Current Training Lanes

Updated 2026-05-09. Docs-only memory sync; promote durable findings into the
handoff/backlog.

- Post-seed-fix experiment worker: active on the LightZero dummy Pong CPU
  1024/16 dynamic-seed train plus held-out independent MCTS scorecard.
- Frozen-checkpoint self-play worker: active on minimal support for learner
  versus frozen older LightZero checkpoint opponent. Scout result:
  `docs/working/lightzero_pong_frozen_checkpoint_selfplay_plan_2026-05-09.md`.
- Performance probe lane: active stance is scale after trust. Do seed sweep,
  search-cost, and collector probes before GPU/vectorization or larger jobs.
