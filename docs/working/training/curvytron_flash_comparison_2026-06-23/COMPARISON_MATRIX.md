# Comparison Matrix

Status: source-read draft. Keep algorithmically non-comparable claims explicit.

## Summary Table

| Axis | Curvytron Flash | CurvyZero | Comparison Verdict |
| --- | --- | --- | --- |
| Primary use | Playable bot lab plus PPO training | Research-grade CurvyTron environment/training/optimizer workspace | Different centers of gravity. |
| Main algorithm | PPO | LightZero/MuZero plus compact owner-search speed path | Not algorithmically apples-to-apples. |
| Runtime target | Modal-only app, Flash/web server, GPU accelerated env | Modal H100 training/eval/profile/tournament, local tests | Both Modal-heavy, different granularity. |
| Environment model | Direct Curvytron app/reference plus accelerated GPU env with compact/raycast observations | Source-state/vectorized envs, visual observations, compact owner/search profile surfaces | Compare raw env/observation costs only under a separate denominator. |
| Speed evidence | Fresh raw H100 controls; one-update diagnostic PPO profile; recovered-doc H100:8 DDP no-sync headline, not packet-run evidence | Extensive same-work H100 speed ledger; OPT-104 baseline `12689.38` env/s; 2x target, current evidence below target | Flash speed is useful calibration only; raw, PPO, and DDP rows must not share a denominator. |
| Learning evidence | PPO checkpoint exports exist for live bots; policy quality still needs eval evidence | Reward-axis H100 campaign, tournaments, eval/GIF/control plane | Compare policy behavior/eval outcomes, not framework training rates. |
| Organization | Small, Modal-only, few entrypoints | Large, heavily documented, many active/historical lanes | Flash is easier to scan; CurvyZero is more auditable but drift-prone. |
| Product loop | Train/export feeds WebSocket bot roster directly | Eval/GIF/tournament/control-plane is rich but less first-glance direct | Flash has a better playable feedback loop. |

## Algorithmic Non-Comparables

- PPO policy-gradient training versus MuZero search/model training.
- Direct accelerated self-play mechanics versus source-state visual LightZero
  wrapper and compact-owned replay/search loops.
- WebSocket bot playability versus tournament/eval training-quality evidence.

## Potentially Comparable Surfaces

- Raw environment throughput.
- Observation extraction cost.
- Reward shaping and retention diagnostics.
- Modal app/run ergonomics.
- Checkpoint export and playable policy bridge.
- File/doc structure for keeping experiments legible.

## Denominators

| Label | Meaning |
| --- | --- |
| `flash_raw_env_control` | Flash `modal_accelerated.py::benchmark`; env/agent steps only. |
| `flash_ppo_profile_control` | Flash one-update PPO timing buckets. |
| `flash_ddp_headline` | Flash H100:8 PPO DDP worklog/docs claim. |
| `curvyzero_lightzero_profile` | CurvyZero stock-ish LightZero/MuZero profile. |
| `curvyzero_compact_profile_only` | CurvyZero owner/search/replay profile harness, no promotion. |
| `curvyzero_same_work_h100_full_loop` | Only valid CurvyZero speed-claim currency. |

These labels should appear in future artifact names or JSON metadata.

## Speed Readout

| Denominator | Fresh Evidence | Speed | Apples-To-Apples Status |
| --- | --- | ---: | --- |
| `flash_raw_env_control` | `h100_grid_baseline_20260623.txt` | `161.55M env/s` | Mechanics ceiling only. |
| `flash_raw_env_control` | `h100_grid_raycast_v1_20260623.txt` | `15.65M env/s` | Observation-cost control only. |
| `flash_ppo_profile_control` | `h100_ppo_profile_diagnostic_20260623.txt` | `438k agent_steps/s` | PPO profile, not MuZero full-loop. |
| `curvyzero_same_work_h100_full_loop` | `goal.md` OPT-104 | `12,689 env/s` | Accepted CurvyZero baseline. |
| `curvyzero_same_work_h100_full_loop` | Current support row in retrospective | `15,853 env/s` | Support evidence, not repeatable 2x. |

The only apples-to-apples CurvyZero speed proof is another same-work H100
full-loop row. Flash rows are controls: they quantify possible ceiling and
bucket costs, not promotion-grade CurvyZero speed.

## Implementation Disparities

| Axis | Flash | CurvyZero | Comparison Read |
| --- | --- | --- | --- |
| GPU residency | Env state and PPO rollout buffers live on CUDA; raw benchmark can keep mechanics and observations in one tight loop. | Production path still goes through LightZero/MuZero and compact owner/search/replay boundaries. Some compact paths are device-first, but full training remains boundary-heavy. | Flash is a better ceiling control for resident mechanics; CurvyZero is doing more whole-loop work. |
| CPU syncs | PPO profile still syncs on done/reset and scalar summaries; reset was the largest fresh bucket at `40.96%`. Its `env_step_s` bucket includes observation generation because `step_batch(..., return_observations=True)` is timed there. | Compact and LightZero paths have action/result/materialization, replay, learner, and profiling sync surfaces depending on mode. | Both have sync cost; CurvyZero's syncs are more structural because search/replay/learner ownership is split. |
| CUDA graph | Flash raw benchmark supports CUDA graph capture only for fixed-shape raw step replay; the fresh straight-grid graph row was slower than eager (`152.01M` vs `161.55M env/s`). | CurvyZero speed proof is not presently a single CUDA-graphable tight env loop. | Graph capture is not the first CurvyZero answer; ownership and fixed-shape handoffs matter more. |
| Multi-GPU | Flash has explicit PPO DDP with `H100:8` in recovered code/docs, but this packet did not rerun it. In DDP, `num_envs` is per rank, not one sharded env batch. | CurvyZero exposes H100 multi-GPU surfaces mostly through Modal resources and LightZero flags; same-work speed gate is single accepted denominator. | Do not compare Flash DDP headline to CurvyZero unless both rows share work and proof scope and include runtime device evidence. |

## Decisions

| Decision | Item | Action |
| --- | --- | --- |
| Benchmark only | Flash accelerated env | Run as raw-env control. Do not promote to CurvyZero speed evidence. |
| Adopt concept | Playable/latest export loop | Define a CurvyZero latest eval/GIF/playable-checkpoint front door. |
| Adopt concept | First-five-minute map | Add this shape to active CurvyZero training/optimizer docs. |
| Study selectively | Flash kernels/ABI | Raw benchmark/parity outputs now exist; study for ceiling and sync ideas, not direct porting. |
| Do not compare | Flash PPO/DDP agent steps vs CurvyZero MuZero rows | Different algorithm, hardware, denominator. |
| Keep | CurvyZero fail-closed speed bar | Same-work H100 full-loop remains the promotion gate. |

## Candidate Guardrails From CurvyZero For Flash Ideas

- **Do not promote raw env throughput as whole-loop speed.** CurvyZero's
  `goal.md` is right to require same-work H100 full-loop rows for speed claims.
- **Keep reward semantics explicit.** CurvyZero's reward contracts make schemas,
  support bounds, and non-claims visible. Flash's dense reward coefficients are
  practical but easier to blur into "policy quality" without a contract.
- **Fail closed before H100 spend.** Flash has Modal-only checks; CurvyZero has
  local proof gates and stronger "do not launch H100 for architecture questions"
  language.
- **Separate learning hypotheses.** PPO, stock-ish LightZero, RND, compact speed,
  and raw env benchmark rows need different ledgers.

## Organization Verdicts

- Flash has the better first-five-minute map.
- CurvyZero has the better proof discipline.
- CurvyZero needs fewer active-looking stale docs and more decisive front doors.
- Flash-derived outputs belong under `artifacts/local/flash_controls/`, not
  optimizer speed ledgers.
