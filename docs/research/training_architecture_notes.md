# CurvyTron RL/MuZero Research Wiki Draft

Source read: `curvytron_muzero_modal_handoff.md`, Version 2, May 8, 2026.

## Purpose

This note broadens the handoff into a research wiki structure for the CurvyTron RL/MuZero repository. It should help the team separate durable conclusions from hypotheses, preserve evidence, keep early choices reversible, and define concrete gates before building an expensive training system.

The strongest conclusion from the handoff still holds: build a fresh Python ML repo and keep CurvyTron browser code as a reference/demo/golden-test source, not as the main training codebase. The simulator is the core asset. Wrappers, baselines, MCTS, MuZero, Modal orchestration, replay, and checkpoints should remain layered around it.

## Wiki Structure

Proposed `docs/research/` outline:

```text
docs/research/
  index.md
  00_summary.md
  01_source_index.md
  02_rules_evidence.md
  03_repo_topology_critique.md
  04_environment_design.md
  05_collision_model.md
  06_observation_reward_design.md
  07_baselines.md
  08_muzero_and_mcts.md
  09_modal_architecture.md
  10_replay_checkpoints.md
  11_experiment_registry.md
  12_acceptance_gates.md
  decisions/
    DR-000-template.md
    DR-001-main-repo-vs-curvytron-fork.md
    DR-002-v0-rules-scope.md
    DR-003-simulator-core-representation.md
    DR-004-wrapper-apis.md
    DR-005-first-baseline.md
    DR-006-mctx-vs-lightzero.md
    DR-007-modal-storage-and-job-layout.md
    DR-008-replay-checkpoint-schema.md
  experiments/
    EXP-000-template.md
    EXP-001-env-throughput.md
    EXP-002-collision-fidelity.md
    EXP-003-wrapper-smoke.md
    EXP-004-baseline-learnability.md
    EXP-005-modal-jax-device-smoke.md
    EXP-006-mctx-synthetic-throughput.md
    EXP-007-replay-resume-integrity.md
  gates/
    GATE-A-simulator.md
    GATE-B-wrappers-baselines.md
    GATE-C-modal-mctx-smoke.md
    GATE-D-muzero-v0.md
```

In this workspace, keep the actual write scope to this file unless asked to split it out. The outline above is the intended wiki map once the repo is created.

## Source And Evidence Model

The research wiki should treat every architectural claim as one of four types:

| Type | Meaning | Example |
| --- | --- | --- |
| Evidence | Observed fact from source, benchmark, issue, docs, or experiment. | CurvyTron browser repo is web-oriented and MIT-licensed. |
| Decision | Choice the project is currently making. | v0 disables bonuses. |
| Hypothesis | Belief that needs a spike or benchmark. | Mctx will provide enough batched search throughput on Modal. |
| Gate | Condition that must pass before expanding scope. | PPO or a simple baseline beats random before MuZero integration. |

Every decision record should include status: `proposed`, `accepted`, `superseded`, or `reversed`. That matters here because several early choices are useful defaults but would be dangerous if they ossify too soon.

## Critique Of The Handoff Repo Topology

The handoff topology is directionally strong: it separates a clean ML repo from the CurvyTron browser fork, makes the environment first-class, includes standard wrappers, gives Modal explicit ownership, and names replay/checkpoints early. That is the right center of gravity.

The main weakness is that it reads as if every later package already deserves a permanent home. `curvyzero_muzero/`, `league.py`, `lightzero_env.py`, and a full set of Modal training launchers are plausible, but some should wait until spikes justify them. Early topology should optimize for replacing algorithms without rewriting the simulator.

Recommended adjustment:

```text
curvyzero/
  curvyzero_env/          # stable core; no trainer dependencies
  curvyzero_wrappers/     # Gymnasium, PettingZoo, LightZero adapters
  curvyzero_agents/       # random, heuristic, PPO/simple baselines
  curvyzero_search/       # Mctx smoke/search adapters if JAX path wins
  curvyzero_training/     # replay, checkpoints, targets, trainer loops
  curvyzero_modal/        # Modal apps and run entrypoints
  tests/
  docs/research/
```

Why this shape is better:

- It keeps wrappers out of the simulator package, so trainer dependencies do not leak into core environment tests.
- It avoids naming the search/training package `muzero` before the team has proven whether the first production path is JAX/Mctx, LightZero, PPO-first, or something else.
- It groups replay/checkpoints with training infrastructure, not a specific algorithm.
- It gives Modal a Python package if jobs need shared imports, rather than only loose scripts.
- It leaves room for multiple research tracks without creating parallel repos.

The handoff's `third_party/curvytron-reference/` recommendation is reasonable, but the import mechanism should remain undecided until the team chooses between subtree, submodule, vendored snapshot, or separate fork plus source-hash notes. For v0, a frozen snapshot or separate fork with recorded commit may be lower friction than a submodule.

## Premature Decisions

These choices are good hypotheses but should not be frozen before evidence:

| Decision | Why premature | Reversible path |
| --- | --- | --- |
| JAX/Mctx as primary MuZero implementation | Search throughput is likely the bottleneck, but Modal compile/runtime behavior must be measured. | Keep simulator NumPy/Python and wrappers algorithm-neutral; run Mctx synthetic smoke before integration. |
| LightZero wrapper file in initial skeleton | Useful fallback, but it may bring dependency and abstraction weight before needed. | Document expected adapter contract first; add concrete wrapper only after LightZero smoke. |
| Full `curvyzero_muzero/` package | Assumes serious MuZero work starts before baseline learnability is proven. | Use `curvyzero_search/` and `curvyzero_training/` until the algorithm choice is accepted. |
| Action repeat of 3 to 5 ticks | Sensible, but it changes control feel, horizon, and collision frequency. | Make it config-driven; record chosen value in env config hash. |
| Occupancy grid as collision model | Likely fastest and easiest to test, but may drift from continuous trace behavior. | Define a collision interface and benchmark occupancy grid against swept-segment golden cases. |
| Modal Volume as main artifact store | Good first choice, but replay scale or multi-writer patterns may push storage to bucket-backed artifacts. | Hide storage behind manifest paths and chunk readers/writers. |
| PPO baseline | Good learnability test, but a simpler policy-gradient or imitation-style baseline may be enough for first signal. | Define baseline gate by behavior, not by algorithm name. |

## Missing Decisions

The handoff should also call out these decisions explicitly:

| Missing area | Decision needed | Why it matters |
| --- | --- | --- |
| Environment versioning | How configs, rules, and observation schemas get hashed/versioned. | Replay and checkpoints are invalid across silent env changes. |
| Coordinate and units convention | Continuous coordinates, grid resolution, turn units, trail thickness, wall boundary inclusivity. | Collision bugs often hide in unit and boundary ambiguity. |
| Randomness contract | What is seeded: spawn positions, any gaps/bonuses, tie-breakers, opponent sampling. | Determinism tests need exact seed scope. |
| Test fixture schema | JSON or npz format for golden trajectories and collision cases. | Golden tests should be inspectable and stable across languages. |
| Evaluation protocol | Fixed seeds, held-out seeds, opponent set, confidence intervals, sample size. | "Beats random" needs statistical meaning. |
| Experiment tracking | TensorBoard, W&B, local CSV/JSONL, or all of the above. | Modal runs must be comparable and resumable. |
| Config system | Dataclasses plus YAML, Hydra/OmegaConf, or simple TOML/JSON. | Reproducibility depends on full config capture. |
| CI boundary | Which tests run on every PR versus nightly/Modal-only benchmarks. | Throughput tests can be too slow for normal CI. |
| Licensing/provenance | How copied constants/rules from CurvyTron are attributed. | MIT reuse is permissive, but provenance should be clean. |
| Human debugging UX | Minimal trajectory viewer, renderer, videos, and failure artifacts. | RL failures need visual inspection early. |

## Decisions That Should Stay Reversible

Keep these behind interfaces or manifests:

- Main algorithm: PPO/simple baseline, JAX/Mctx MuZero, LightZero, or later AlphaZero-style variants.
- Simulator implementation: Python/NumPy first, optional Numba/JAX/C++ only after profiling.
- Collision backend: occupancy grid, swept segment, or hybrid.
- Observation family: egocentric raster, rays, full grid, or mixed features.
- Wrapper APIs: Gymnasium, PettingZoo Parallel, LightZero, custom batched API.
- Storage backend: Modal Volume, bucket mount, S3/R2/GCS, or local filesystem.
- CurvyTron reference import: separate fork, submodule, subtree, or vendored snapshot.
- Experiment tracker: local logs, TensorBoard, W&B, or another project standard.

The practical rule: stable code should depend on small project-owned interfaces, not directly on a research library's preferred shape.

## Research Threads

| Thread | Research question | First artifact | Gate |
| --- | --- | --- | --- |
| Rules extraction | What exactly is v0 CurvyTron? | `rules_v0.md`, evidence notes, golden case list. | No ambiguous collision/scoring behavior remains for v0. |
| Simulator correctness | Can we step deterministic rounds and replay them? | Core env plus trajectory fixture schema. | Same seed/actions produce identical trajectory summaries. |
| Collision fidelity | Which collision model is fast and faithful enough? | Occupancy vs swept benchmark and golden cases. | Edge cases pass and throughput target is met. |
| Wrapper compatibility | Can standard RL APIs express all-live-player wrapper actions cleanly? | Gymnasium and PettingZoo smoke results. | Random-agent wrapper tests pass without simulator-specific hacks. |
| Learnability | Is the v0 game learnable before MuZero? | Heuristic and baseline training report. | Baseline beats random on held-out seeds. |
| Search throughput | Can Mctx batch enough work on Modal GPUs? | Synthetic Mctx benchmark. | Steady-state throughput and utilization justify integration. |
| Replay integrity | Can data survive long runs and resume safely? | Replay/checkpoint schema plus resume test. | Resume reproduces expected model/replay state. |

## First Implementation Slices

| Slice | Files/modules | Build | Acceptance |
| --- | --- | --- | --- |
| 1. Wiki and decisions | `docs/research/`, decision template | Source index, decision records, experiment templates, gates. | Every major default has a status and evidence link. |
| 2. Rules/config | `rules_v0.md`, env config dataclasses | 1v1, no bonuses, action set, units, seed contract, scoring/ties. | Rules are implementable without reading browser code. |
| 3. Deterministic simulator | `curvyzero_env/state.py`, `core.py`, `physics.py`, `scoring.py` | Reset/trainer step, wrapper action maps, terminal rewards. | Byte-stable trajectory summaries under fixed seed/action trace. |
| 4. Collision goldens | `collisions.py`, `tests/golden_cases/` | Wall, self, opponent, head-head, simultaneous death, grazing/tunneling. | Golden suite passes and failure artifacts are inspectable. |
| 5. Observations/replay viewer | `observations.py`, `trajectory.py`, `render_debug.py` | Egocentric observation and replayable trajectories. | Any failed test can produce a human-readable replay artifact. |
| 6. Wrappers/baselines | wrapper package, agents package | Gymnasium/PettingZoo adapters, random, heuristic, first learned baseline. | Heuristic and learned baseline beat random under evaluation protocol. |
| 7. Modal/Mctx smoke | Modal package, search smoke | JAX device check, synthetic Mctx benchmark, throughput report. | Compile and steady-state timings are recorded for fixed sweeps. |
| 8. Replay/checkpoints | training package | Chunk format, manifests, checkpoint save/load, resume test. | Resume and replay integrity tests pass before MuZero consumes data. |

## Acceptance Gates

Gate A: research readiness

- Source index exists with links, provenance notes, and confidence levels.
- Decision records exist for repo split, v0 rules scope, simulator representation, wrapper APIs, and first algorithm spike.
- Experiment templates define hypothesis, setup, metrics, artifacts, result, and next decision.

Gate B: simulator readiness

- Deterministic reset/trainer step under fixed seed and wrapper action trace.
- Golden tests cover wall, self, opponent, head-head, tunneling/grazing, and same-tick deaths.
- Vectorized env matches single env for identical seeds/actions.
- Debug renderer or textual replay can inspect any stored trajectory.
- Long random-agent stress test has no nondeterministic crashes or memory growth.

Gate C: wrapper and baseline readiness

- Gymnasium and PettingZoo random-agent smoke tests pass.
- Evaluation protocol has fixed seeds, held-out seeds, opponent definitions, and confidence reporting.
- Heuristic beats random.
- PPO or simpler learned baseline beats random in 1v1 no-bonus mode.

Gate D: Modal and Mctx readiness

- Modal GPU job confirms JAX sees the expected device.
- Synthetic Mctx benchmark reports compile time separately from steady-state throughput.
- Root batches 64, 256, and 1024 run without shape-recompile churn.
- Profiling identifies whether search, env stepping, replay I/O, or training is the bottleneck.

Gate E: MuZero v0 readiness

- Replay sequences align observations, actions, rewards, search policy targets, and value targets.
- Checkpoint save/resume works from the chosen storage backend.
- 1v1 MuZero improves against random and then challenges the heuristic.
- Evaluation compares raw policy versus MCTS-improved action selection.

## Recommendation

Implement the research wiki and simulator slices first. Do not make the initial repo topology algorithm-shaped beyond what smoke tests justify. Keep the simulator, wrappers, search, training, and Modal orchestration separated by small project-owned interfaces.

The first real code PRs should be:

1. Research docs, decision templates, source index, and v0 rules decision.
2. Environment config/versioning plus deterministic 1v1 no-bonus simulator.
3. Collision golden tests and trajectory replay artifacts.
4. Wrappers, random/heuristic agents, and evaluation protocol.
5. Local/Modal throughput benchmark.
6. Baseline learnability experiment.
7. Standalone Modal JAX/Mctx smoke.
8. Replay/checkpoint schema and resume integrity test.
9. Minimal MuZero only after the baseline and Mctx gates pass.
