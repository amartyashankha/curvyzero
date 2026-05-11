# Repo Structure Critique

## Short Answer

The proposed structure is directionally right: keep the CurvyTron browser code out of the ML hot path, make the deterministic simulator the core asset, and keep research notes below stable docs. The main risk is committing too early to repo boundaries, framework-specific folders, and "future complete" abstractions before the project has evidence from rule mining, environment throughput, Modal smoke tests, and baseline learning.

Start as one investigation-first ML repo with a pinned CurvyTron reference import or adjacent fork. Split later only when there is a real release, permission, dependency, or ownership boundary. Treat the docs tree as a promotion pipeline: raw evidence in research/experiments, distilled commitments in decisions/design.

## Separate Repos vs Folders

The handoff's "fresh ML repo plus CurvyTron reference fork" is the right default. A browser game repo and an RL training repo want different dependencies, runtimes, tests, CI, and performance assumptions. Forcing MuZero, Modal jobs, replay, and vectorized simulation into the web game would make the training system inherit the wrong shape.

Within the ML effort, folders are better than repos until interfaces harden. `env`, `agents`, `training`, `modal`, `docs`, and `experiments` should live together while the team is still discovering tick semantics, collision rules, observations, and search throughput. Cross-repo development would add coordination cost before there is a stable package boundary.

The CurvyTron source should be separate in provenance, not necessarily separate in daily workflow. A fork plus pinned `third_party/curvytron-reference` snapshot/subtree is easier to audit and test against than an unpinned external checkout. A submodule is defensible only if the team already tolerates submodule friction.

Split into more repos later only if one of these becomes true:

- The simulator has external consumers and needs independent versioning.
- The browser demo needs its own deployment and release cadence.
- Modal infrastructure needs separate secrets, access control, or operational ownership.
- Dependency conflicts become real rather than anticipated.
- The CurvyTron reference import creates clone, CI, or licensing friction.

## Delay These Commitments

Delay a hard repo split beyond "ML repo plus reference fork." It is easier to split a cohesive repo with clean package boundaries than to stitch together premature repos with unstable APIs.

Delay library-specific architecture until smoke tests pick a lane. Folders such as `curvyzero_muzero`, `lightzero_env.py`, or Mctx-specific wrappers should stay thin or experimental until Modal benchmarks show whether JAX/Mctx or PyTorch/LightZero is viable.

Delay full multi-agent generality. Start with 1v1, no bonuses, explicit simultaneous actions, and one episode equals one round. General N-player league abstractions will be easier to design after the 1v1 simulator, reward, and evaluation loop are measured.

Delay complex Modal orchestration. The first Modal code should prove image reproducibility, GPU visibility, environment throughput, MCTS throughput, checkpoint writes, and replay chunk shape. Queues, Dicts, sweeps, actor fleets, and distributed replay should follow measured bottlenecks.

Delay polished documentation taxonomy below the current top-level map. The existing `docs/README.md` hierarchy is enough. Add notes as evidence appears, then promote distilled decisions. Avoid creating empty pages for every future subsystem.

Delay forks of ML libraries. Use pinned dependencies first. Fork Mctx, LightZero, or other RL code only after a smoke test identifies a specific missing feature or patch.

## Abstractions That May Age Badly

Algorithm-branded package names can trap the project. If PPO, AlphaZero-style search, EfficientZero, or a policy-only baseline becomes the practical path, a repo centered on `curvyzero_muzero` will feel stale. Prefer neutral core names for long-lived surfaces, with algorithm-specific modules below them.

Framework wrappers can leak into the environment. Gymnasium, PettingZoo, LightZero, and Mctx adapters should wrap a small simulator core rather than define it. The stable contract should be reset/step/state/replay, not a third-party library's current environment shape.

Over-general "league" and "self-play" abstractions can become speculative architecture. Build the first loop in a boring, inspectable way; abstract once there are two or three real training modes with duplicated code.

`third_party/notes` and catch-all research folders can become junk drawers. Each note should have an owner-like purpose: rule evidence, benchmark result, design critique, or open question. Otherwise future readers cannot tell whether a note is current, contradicted, or just old.

Decision documents can conflict with narrative handoffs. The handoff is valuable context, but stable commitments should live in ADRs or design docs with evidence and reversal conditions. Do not let `handoff_v2.md`, `design_decisions.md`, and ADRs become three competing sources of truth.

Golden tests can accidentally encode an arbitrary clone rather than the intended game. Every golden case should say whether it came from CurvyTron source behavior, CurvyTron 2 public rules, an explicit v0 simplification, or a training convenience.

## Evidence Gates Before Commitment

Commit to the simulator core after rule evidence is good enough: tick/update semantics, turn rate, collision thickness, wall/self/opponent collisions, simultaneous death behavior, scoring, seed determinism, and action repeat are documented and covered by golden tests.

Commit to performance-sensitive implementation choices after benchmarks: random-agent episodes/sec, physics ticks/sec/core, vectorized equivalence, memory growth over long runs, and debug replay from stored trajectories.

Commit to an RL framework after smoke tests, not preference. Mctx needs a synthetic Modal benchmark with compile time separated from steady-state search throughput. LightZero needs a custom-env smoke with measured collector/MCTS throughput and no hidden single-agent assumptions.

Commit to MuZero only after a baseline learns. A PPO or simple policy-gradient baseline beating random/heuristic opponents is the best evidence that the environment, observations, rewards, and evaluation are sane enough for a search-heavy method.

Commit to distributed Modal architecture only after profiling a single-container loop. The first bottleneck should be observed in traces or metrics before adding queues, actor pools, shared replay services, or storage complexity.

Commit to a repo split only after an actual boundary appears: independent release, dependency conflict, external user, separate deployment, or access-control need.

## Keeping Research Useful

Use research notes for evidence and critique, not hidden decisions. Each note should end with open questions, candidate gates, and the specific decision/design doc it could promote into.

Keep top-level docs short. `docs/README.md` already sets a good rule: stable overview first, messy detail lower down. Preserve that discipline so future contributors can get oriented in five minutes.

Make status visible in each research note: proposed, contradicted, superseded, accepted, or needs measurement. A technically correct note with no status becomes dangerous once the project moves on.

Separate source-derived facts from project choices. "CurvyTron does X" and "v0 will do Y" should be visibly different statements, especially around collisions, bonuses, scoring, and action space.

Prefer small promotion steps. A research note should become one ADR, one design page section, one benchmark report, or one issue. Large handoffs are useful snapshots, but they should not be the maintenance unit.

## Open Questions

- Should the CurvyTron reference be a subtree, submodule, vendored snapshot, or adjacent fork for this specific team?
- What package names stay neutral if MuZero is delayed or replaced?
- Which evidence gate is required before declaring the simulator API stable?
- How often should research notes be pruned, superseded, or promoted?

## Sources

- `curvytron_muzero_modal_handoff.md`
- `docs/README.md`
