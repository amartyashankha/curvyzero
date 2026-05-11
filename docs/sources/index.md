# Source Index

This is the source ledger for implementation-shaping evidence. Research notes may contain direct links while drafting, but sources that support durable decisions should be promoted here.

| ID | Source | Type | Authority | Relevant Claim | Supports |
| --- | --- | --- | --- | --- | --- |
| S001 | `curvytron_muzero_modal_handoff.md` | local-handoff | suggestive | Initial repo topology, Modal cautions, MuZero/Mctx/LightZero plan, and acceptance gates. | ADR-0001, ADR-0002, ADR-0003 |
| S002 | https://github.com/Curvytron/curvytron / `third_party/curvytron-reference` commit `8fec14c` | primary source | authoritative for original repo | Original CurvyTron source and MIT license for rule mining and demo/reference work. | `docs/sources/curvytron_reference.md`, rulesets |
| S003 | `docs/research/modal_patterns.md` | research synthesis | suggestive | Modal primitives should coordinate coarse work; hot-loop stepping/search/inference should remain local to one process/container. | ADR-0002 |
| S004 | `docs/research/performance_vectorization.md` | research synthesis | suggestive | Start with readable Python/NumPy, fixed-shape state, occupancy grid, and benchmarks before Numba/JAX/PyTorch/native backends. | deterministic environment design |
| S005 | `docs/research/multiplayer_selfplay_muzero.md` | research synthesis | suggestive | Use ego-perspective shared-policy self-play first; defer joint-action search and general-sum value complexity. | ADR-0003 |

