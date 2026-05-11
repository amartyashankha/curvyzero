# Investigation Plan

This project should move fast, but not by pretending the core uncertainties are already solved. The early goal is to build a narrow, working vertical slice while keeping major choices reversible until we have evidence.

## North Star

Train strong agents for a CurvyTron-like game by building a fast deterministic simulator, validating it against source-derived behavior where useful, then layering baselines, search, self-play, and Modal orchestration on top.

## Operating Principles

1. The simulator is the product. Training libraries can change; deterministic game stepping, tests, and benchmarks should survive.
2. CurvyTron source is a reference, not the hot loop, unless measurements prove otherwise.
3. Major decisions need evidence gates: source inspection, local tests, benchmarks, and small Modal smokes.
4. Keep rules explicit. If a behavior is cloned from source, cite it. If invented for v0, label it.
5. Optimize only after measuring. Design for vectorization now, but do not start with exotic infrastructure.
6. Documentation has layers: concise overview first, detailed evidence deeper, raw working memory deepest.

## Investigation Lanes

### 1. CurvyTron Reference

Questions:

- How is the original game structured?
- Where are motion, turning, trail creation, collision, scoring, bonuses, and timing implemented?
- Which constants and edge cases should become simulator tests?
- Can the original repo run locally enough to generate golden references?

Outputs:

- `third_party/curvytron-reference`
- `docs/research/curvytron_reference_notes.md`
- `docs/sources/curvytron_reference.md`

### 2. Deterministic Simulator

Questions:

- What is the smallest correct 1v1 no-bonus trainer `step` API?
- How do wrapper actions, held control state, deaths, ties, decision cadence, and rewards
  work?
- Should v0 use occupancy grids, swept geometry, or both?
- How do we keep vectorization possible without overbuilding?

Outputs:

- `src/curvyzero/env/`
- `tests/test_*.py`
- `docs/design/deterministic_environment.md`
- `docs/research/deterministic_env_notes.md`

### 3. Training Architecture

Questions:

- What baseline proves learnability before MuZero?
- What replay/checkpoint/evaluation contracts are needed?
- What does a minimal MuZero/MCTS vertical slice require?
- Which library decisions can wait until after simulator and baseline gates?

Outputs:

- `curvyzero_agents/`
- `curvyzero_muzero/`
- `docs/design/training_architecture.md`
- `docs/research/training_architecture_notes.md`
- `docs/research/baseline_learnability.md`

### 4. Modal Architecture

Questions:

- Which Modal primitives belong in the project from day one?
- What is the clean split between local, Modal CPU, and Modal GPU runs?
- How do Volumes, CloudBucketMounts, Queues, Dicts, Sandboxes, and snapshots fit?
- What must never enter the per-step or per-node hot loop?

Outputs:

- `modal/`
- `docs/design/modal_architecture.md`
- `docs/runbooks/modal_*.md`
- `docs/research/modal_patterns.md`

### 5. Performance Path

Questions:

- What throughput is enough for each stage?
- How far can Python/NumPy go?
- When should we consider Numba, JAX-native envs, PyTorch tensor envs, or native extensions?
- Which data structures make migration easier later?

Outputs:

- `scripts/benchmark_env.py`
- `docs/experiments/*benchmark*`
- `docs/research/performance_vectorization.md`

### 6. Fidelity, Variation, And Curriculum

Questions:

- What does it mean to be “CurvyTron-like” versus an exact clone?
- How are rule versions named and tested?
- Which environment parameters can vary during training?
- How do we prevent robustness work from silently changing the target game?

Outputs:

- `curvyzero_env/config.py`
- `docs/design/rulesets.md`
- `docs/research/env_fidelity_curriculum.md`

## Evidence Gates

### Gate 0: Repository Ready

- Documentation hierarchy exists.
- CurvyTron reference repo is available or blocked with a clear reason.
- First decision records describe current repo and investigation posture.

### Gate 1: Rule Understanding

- Motion/collision/scoring source locations are documented.
- v0 cloned versus invented rule choices are listed.
- At least ten golden-test candidates are written down.

### Gate 2: Simulator Vertical Slice

- `reset(seed)` and trainer-wrapper `step(action_by_player)` work for 1v1 no-bonus.
- Determinism tests pass.
- Wall, self, opponent, and simultaneous-death tests exist.
- A random-agent throughput benchmark exists.

### Gate 3: Baseline Learnability

- A simple heuristic beats random.
- PPO or another policy baseline learns above random in the simplest setting.
- Evaluation uses fixed and held-out seeds.
- MuZero work stays gated until random stress, heuristic-vs-random, and one policy baseline give us a debugging baseline.

### Gate 4: Modal Smoke

- Modal CPU job runs environment tests or benchmark.
- Modal GPU job prints JAX and/or PyTorch device info.
- Checkpoint/log storage path is tested without creating many tiny files.

### Gate 5: MuZero/MCTS Decision

- Mctx synthetic benchmark has steady-state results.
- LightZero or PyTorch alternative has a small custom-env smoke if still in contention.
- We choose a training path with reversal conditions documented.

## First Implementation Slice

The first slice should be deliberately boring:

- `pyproject.toml` with local package metadata and test dependencies.
- `curvyzero_env` with config/state/core modules.
- One deterministic 1v1 no-bonus environment.
- Focused tests for reset, stepping, walls, ties, and deterministic replay.
- Documentation that labels every rule as source-derived, v0-invented, or unresolved.

This avoids locking the project to a training library before we know whether the environment is correct and fast.
