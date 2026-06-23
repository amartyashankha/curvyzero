# Speed Ledger Recheck

Date: 2026-05-23

This is the clean speed/bottleneck ledger after rereading the current reorientation docs, nearby optimizer notes, and the r18fresh postmortem. The short version is: there are real training speeds, there are stock `train_muzero` profile speeds, and there are compact/search profile-only speeds. They are useful, but they are not interchangeable.

## One-line Read

- The trusted Coach path is still stock LightZero `train_muzero`.
- No compact/direct CTree/MCTX/sample-gate work has proven a material real Coach-loop speedup yet.
- The latest matched `train_muzero` H100 profile rows support about 1.28x-1.30x on the profile denominator, not 5x-10x on the real training denominator.
- Compact and MCTX rows show important search/dataflow headroom, but they are profile-only unless promoted through the Coach gates.
- Amdahl now points less at one magic kernel and more at ownership boundaries: scalar object fanout, root/leaf preparation, env/observation handoff, replay/RND materialization, and learner cadence.

Sources: `docs/working/optimizer/reorientation_2026-05-23/CURRENT_STATE.md`, `TODO.md`, `EXPERIMENT_QUEUE.md`, `ORCHESTRATION.md`, `MEASUREMENT_LEDGER.md`, `SPEED_TIMELINE.md`, `BOTTLENECK_MODEL.md`, `FAILURE_ANALYSIS.md`, `COMPACT_OWNERSHIP_PLAN.md`, plus the local artifacts cited below.

## Speed Currencies

| Currency | Unit | What it means | Safe use |
| --- | --- | --- | --- |
| Real Coach-loop speed | learner iterations/hour, sometimes checkpoint cadence | Actual stock training runs through `train_muzero`, including the real learner loop and run overheads | Use for Coach recommendations and wall-clock training claims |
| Stock `train_muzero` profile speed | env steps/sec or roots/sec inside a short instrumented profile | Calls the real trainer, but uses short controlled profiles, often with sidecars/eval/checkpoints disabled | Use for matched A/B speed comparisons only |
| Compact/search profile-only speed | steps/sec, slab roots/sec, roots/sec | Exercises compact bridge/search/dataflow probes outside live Coach | Use for architecture evidence, not training claims |
| Synthetic ceiling | steps/sec or roots/sec under mocked or stripped services | Removes or replaces parts of the real path | Use only to identify possible upper bounds |

Do not compare these directly. A 40k roots/sec compact ceiling is not the same claim as 40k learner iterations/hour.

## Known Real Coach-loop Speeds

| Run family | Denominator | Known speed | What was included | Notes |
| --- | --- | --- | --- | --- |
| CZ26 136-run batch | learner iterations/hour | Roughly 15k learner iters/hour from completed rows | Stock LightZero `train_muzero`, L4/T4-class `gpu-l4-t4-cpu40`, C256/N256, batch64, sim8, `browser_lines` + `simple_symbols`, sidecars on | Real Coach run, but the speed was reconstructed and was not an official optimizer speed table. Sources: `docs/working/optimizer/reorientation_2026-05-23/SPEED_TIMELINE.md`, `artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md` |
| RND blank sweep, 2026-05-19 | learner iterations/hour | Mean about 18.4k, median about 19.7k, range about 14.4k-23.0k learner iters/hour | Stock `train_muzero`, L4/T4-class `gpu-l4-t4-cpu40`, C256/N256, batch64, sim8, `browser_lines` + `simple_symbols` + CPU oracle, GIF/eval sidecars on | Best recent real L4-style Coach speed read. Source: `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/actual_training_speed_read_20260521.md` |
| r18fresh H100, older setup | learner iterations/hour | Average about 31.5k learner iters/hour; fastest sparse row about 37.9k | Stock `train_muzero`, `gpu-h100-cpu40`, C256, batch32, sim8, source-state fixed opponent, CPU oracle, checkpoints every 10k | Real training, but older setup and not a direct comparator for current L4/CZ26. Source: `docs/working/training/r18fresh_postmortem_2026-05-16/H100_L4_OPTIMIZER_HANDOFF.md` |

Current real Coach-loop optimizer speedup: unproven. The current docs correctly treat it as 0x proven actual Coach speedup over the trusted stock path until a promoted candidate wins on learner iterations/hour.

## Stock Trainer Profile Speeds

These call `train_muzero`, but the denominator is still profile env steps/sec, not learner iterations/hour.

| Profile | Denominator | Stock | Candidate | Ratio | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-05-22 no-RND stock vs direct | profile env steps/sec | 433.17 | 566.19 | 1.31x | Short profile, not full Coach speed. Sources: `docs/working/optimizer/reorientation_2026-05-23/SPEED_TIMELINE.md`, `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/current_hot_path_bottleneck_map_20260522.md` |
| 2026-05-22 RND stock vs direct | profile env steps/sec | 351.02 | 448.52 | 1.28x | Same warning. RND did not disappear as a concern, but direct output helped the profile wall. Sources: same as above |
| Latest local Gate A H100 no-RND warm | profile env steps/sec | 928.91 | 1203.60 | 1.30x | `called_train_muzero=true`, H100, C512, batch64, sim4, no-death, 32 learner calls, env_steps 262144, sidecars/eval/checkpoints off. Source: `artifacts/local/curvytron_optimizer_profile_results/gate-a-directctree-h100-rndoff-warm-20260523/collected_results.json` |
| Latest local Gate A H100 RND-meter warm | fallback profile env steps/sec | 992.60 | 1270.13 | 1.28x | `called_train_muzero=true`, same broad H100 shape, RND meter path, but strict Gate A rejects raw collector-step attestation and falls back to `mcts_search_root_sum_profile_fallback`. Source: `artifacts/local/curvytron_optimizer_profile_results/gate-a-directctree-h100-rndmeter-warm-20260523/collected_results.json` |
| Latest local Gate A L4 no-RND warm | profile env steps/sec | 591.04 | 846.96 | 1.43x | Strict Gate A accepted. Same profile shape on L4-class hardware. Source: `artifacts/local/curvytron_optimizer_profile_results/gate-a-directctree-l4-rndoff-warm-20260523/collected_results.json` |
| Latest local Gate A L4 RND-meter warm | fallback profile env steps/sec | 577.71 | 903.82 | 1.56x | Same RND fallback caveat as H100. Treat as sidecar until raw collector-step telemetry is fixed. Source: `artifacts/local/curvytron_optimizer_profile_results/gate-a-directctree-l4-rndmeter-warm-20260523/collected_results.json` |
| L4/C256 batched GPU observation profile | profile env steps/sec | CPU oracle 586.14 | batched GPU obs 893.55 | 1.52x | Useful observation-path evidence, not actual training speed. Source: `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/actual_training_speed_read_20260521.md` |

Plain read: direct CTree GPU-latent is a real matched-profile cleanup on both
H100 and L4. It is still not a 5x-10x architecture answer and not yet a proven
real Coach-loop learner-iterations/hour win.

## Compact/Search Profile-only Speeds

These are not Coach speeds. In the current artifacts they do not promote a backend, and the compact/MCTX rows do not call live `train_muzero`.

| Probe | Denominator | Representative speed | What it says | Source |
| --- | --- | ---: | --- | --- |
| Direct CTree compact slab, H100 | profile steps/sec and slab roots/sec | B1024/A16/sim8 around 8291 steps/sec and 12937 slab roots/sec | Direct compact slabs are much faster than old object-heavy paths, but still profile-only | `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/compact_slab_h100_profile_summary_20260523d.md` |
| Clean falsifier direct, scalar off | profile steps/sec and slab roots/sec | 7191 steps/sec, 12870 slab roots/sec | Good compact baseline, no sample gate | `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-clean-direct-20260523/collected_results.json` |
| Clean falsifier direct, scalar on | profile steps/sec and slab roots/sec | 4652 steps/sec, 11163 slab roots/sec | Scalar/object materialization is still expensive in this denominator | Same source as above |
| Clean service-tax ceiling | profile steps/sec and slab roots/sec | 5174 steps/sec, 32000 slab roots/sec | Search/service can become cheap, but total wall does not scale with roots/sec | `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-clean-service-tax-20260523/collected_results.json` |
| Clean mock ceiling | profile steps/sec and slab roots/sec | 5263 steps/sec, 43154 slab roots/sec | Higher root ceiling, still not a Coach claim | `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-clean-mock-20260523/collected_results.json` |
| Clean dense Torch profile | profile steps/sec and slab roots/sec | 7898 steps/sec, 15715 slab roots/sec | Interesting dense-search evidence, but semantic mismatch with LightZero CTree remains | `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-clean-dense-torch-20260523/collected_results.json` |
| MCTX strict H100 80/20 | profile steps/sec and roots/sec | B512/s16 MCTX 11826 steps/sec vs direct 5864; B1024/s32 MCTX 13964 vs direct 4400 | MCTX can make the search sub-bucket much cheaper and full profile 1.8x-3.2x faster in this probe, but it is profile-only and semantically different | `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/mctx_scaling_grid_summary_20260523e.md`, `artifacts/local/curvytron_hybrid_observation_profile_results/opt-mctx-strict-h100-8020-20260523f/collected_results.json`, `artifacts/local/curvytron_hybrid_observation_profile_results/opt-direct-strict-h100-8020-20260523f/collected_results.json` |

The quick sample-gate wave also showed that forced sample materialization can dominate a short toy denominator: sample-gate time was about 6.8s-20.8s in the quick rows. That justified the clean no-sample-gate wave, but it does not by itself prove replay sampling is the main real Coach wall. Sources: `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-20260523-direct/collected_results.json`, `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-20260523-service-tax/collected_results.json`, `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-20260523-mock/collected_results.json`, `artifacts/local/curvytron_hybrid_observation_profile_results/opt-c-falsifier-20260523-dense-torch/collected_results.json`.

## Amdahl Points Now

- In latest H100 Gate A profiles, direct CTree/output work cuts the policy/search part hard, but total `train_wall_time` only moves about 1.28x-1.30x. The remaining wall is collector/env/control flow, learner/replay cadence, and trainer overhead.
- In the clean compact falsifier, service-tax/mock rows make roots/sec look huge, but total steps/sec does not explode. That points at non-search ownership costs: actor/env/observation handoff, slab assembly, device transfer, and result materialization.
- Scalar materialization is not a rounding error. In the clean direct probe, scalar-off is about 7191 steps/sec while scalar-on is about 4652 steps/sec; precomputed recurrent scalar-off is about 7188 while scalar-on is about 5179.
- MCTX makes the search bucket much cheaper in strict profiles. Once search is cheaper, Amdahl shifts to observation/env/handoff and to the semantic cost of joining the real LightZero trainer.
- Replay/RND are not dismissed. They are currently measured as smaller than search/control in some stock trainer profiles, but forced sample-gate probes show materialization can dominate when sampled too eagerly or measured under a toy cadence.

## Claims Still Unproven

- Any material actual Coach-loop speedup over the trusted stock LightZero path on learner iterations/hour.
- A 5x-10x real Coach training speedup.
- That direct CTree, compact slabs, MCTX, or dense Torch are Coach-ready training backends.
- That MCTX is semantically equivalent to the LightZero CTree path, or that an algorithmic change should be accepted for Coach.
- That L4 Gate A speed automatically predicts real L4 Coach-loop speed. The
  latest L4 profile rows are collected and accepted for no-RND (`1.43x`), but
  they are still short `train_muzero` profile rows, not learner-iterations/hour
  training runs.
- That compact/profile-only wins survive learner updates, RND, replay insertion/sampling, terminal/autoreset behavior, sidecars, and checkpoints.
- That faster samples/sec improves learning per wall-clock under the same game/training semantics.
- That replay sampling is negligible once attached to the real learner at realistic cadence.

## Why Previous Optimizer Work Felt Like Flailing

1. Speed currencies were mixed. Learner iterations/hour, stock profile env steps/sec, compact roots/sec, and synthetic ceilings were allowed to sit in the same mental bucket. That made real progress feel both larger and less trustworthy than it was.
2. Work moved across lanes faster than promotion gates. There were good compact proofs, sidecars, MCTX rows, direct-output rows, and observation rows, but too few were forced through one matched Coach denominator before the next promising lane appeared.
3. The bottleneck moved, but the story lagged. Render/observation used to be a big visible wall; after fixes, LightZero scalar object work, root/leaf prep, search/control handoff, replay/RND materialization, and learner cadence became the limiting surface. Celebrating sub-loop wins before rerunning the whole denominator made the effort feel slippery.

## Practical Recheck Recommendation

Treat the latest Gate A direct rows as useful `train_muzero` profile wins:
about `1.30x` on H100 and `1.43x` on L4 without RND. Treat compact/MCTX rows as
architecture evidence only. The next Coach-facing claim should wait for a
promoted full-loop comparison that reports learner iterations/hour, env
steps/sec, replay/RND metrics, sidecar settings, and `called_train_muzero=true`
in the same table.
