# Ideas Inbox

This is a low-friction capture page for user ideas, half-formed questions, and future investigation branches. Promote useful items into research notes, design docs, decisions, or experiments.

## 2026-05-08

- Use many parallel subagents for critique and evidence gathering, while keeping the main thread focused on high-level structure and synthesis.
- Treat documentation as durable memory because context is short. Keep top-level docs concise and human-readable; push messy details, experiment logs, old handoffs, and raw notes deeper.
- Clone CurvyTron as reference material, but investigate whether running it is necessary or whether source inspection is enough.
- Deterministic stepping is central. Need explicit seed handling, simultaneous actions, collision semantics, rewards, replay, and golden tests.
- Research Modal patterns deeply. Modal primitives may include functions, images, volumes, cloud bucket mounts, queues, dicts, sandboxes, snapshots, GPU access, and secrets.
- Be careful with environment fidelity. First clone the game well enough; later vary parameters during training for robustness.
- Expect the simulator to run far faster than real time, potentially thousands of rollouts at once. Start in Python, but design so the path to NumPy, Numba, PyTorch, JAX, or native extensions stays open.
- Investigate prior work on MuZero/MCTS/self-play in multiplayer games. Confirm how multi-agent self-play is usually formulated and what breaks when moving beyond two-player zero-sum games.
- Deeply understand MuZero itself: representation/world model, dynamics, prediction heads, value/reward/policy targets, action representation, MCTS integration, self-play actors, replay, latest weight handoff, and inference batching.
- Keep revisiting repo/docs structure early. This is the best time to reorganize before implementation calcifies around accidental structure.
- Promoted repo/docs hierarchy into `docs/design/repository_hierarchy.md`: keep top-level maps stable and readable, keep research/experiments/working deeper and messier, and do not move active research files while agents may still depend on fixed paths.
- Future `src/curvyzero` domains to add only when justified: `env`, `agents`, `eval`, `replay`, `models`, `search`, `training`, `adapters`, `infra`, and `viz`.
