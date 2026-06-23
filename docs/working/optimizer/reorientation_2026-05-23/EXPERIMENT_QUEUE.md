# Experiment Queue

Date: 2026-05-23

Archive. Do not launch rows from this file unless `TASK_BOARD.md` names them.
Active optimizer work lives in `goal.md` and `TASK_BOARD.md`; do not infer it
from this archive.

This is the queue after the local compact-contract gate passed. Do not launch
live Coach training from this file. This is optimizer profiling only unless a
separate launch doc says otherwise.

## Historical Fresh Gate, 2026-05-26

Status: parked. This queue records the older compact-vs-stock profiling wave.
Active speed work now lives in `goal.md`, `TASK_BOARD.md`, and
`FOLLOWUPS.md`. `model_compile_mode=default` reduced the
model/service bucket in an older lane, but the formal decision was
`park_model_compile_default_speed_unapproved` because end-to-end wall speed did
not repeatably improve. Do not launch these old rows as "current" work without
rewriting their denominator and claims.

Purpose: answer the user's direct question without mixing currencies.

Launched from the same repo state:

- Stock Coach-compatible profile:
  `artifacts/local/curvytron_optimizer_profile_manifests/optimizer-fresh-stock-gate-20260526a.json`
  - H100, `collector_env_num=512`, `batch_size=64`, `num_simulations=8`;
  - rows: `exploration_bonus_mode=none` and `rnd_meter_v0`;
  - `called_train_muzero=true` required;
  - eval/GIF/checkpoint sidecars off;
  - stop after `32` learner train calls.
- Compact candidate host-stack profile:
  `artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-fresh-compact-b1024-host-rndprobe-20260526a/manifest.json`
  - H100, B1024/A16/sim8, 100 warmup, 120 measured steps;
  - search feedback, replay sample gate, learner gate, RND-style input;
  - scalar timesteps off;
  - `calls_train_muzero=false`.
- Compact candidate resident-stack profile:
  `artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-fresh-compact-b1024-resident-rndprobe-20260526a/manifest.json`
  - same compact denominator;
  - resident observation search on;
  - hidden host fallback must be zero;
  - `calls_train_muzero=false`.

Completed compact rows:

| Row | Result | Wall sec | Steps/sec | Important checks |
| --- | --- | ---: | ---: | --- |
| host-stack compact | `artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-fresh-compact-b1024-host-rndprobe-20260526a/row_001_result.json` | `15.829` | `15526.3` | `calls_train_muzero=false`, scalar rows off |
| resident-stack compact | `artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-fresh-compact-b1024-resident-rndprobe-20260526a/row_001_result.json` | `12.230` | `20095.0` | `calls_train_muzero=false`, resident obs used, hidden host fallback `0` |

Plain read so far: the resident observation/render boundary is `1.29x` faster
than the same compact candidate with host-stack observation movement. This is
real for the compact profiler, but it is not yet a stock Coach training speedup.
The stock `train_muzero` rows are still the pending denominator.

Completed stock rows:

| Row | Result | Wall sec | Env/profile steps/sec | Important checks |
| --- | --- | ---: | ---: | --- |
| stock no RND | `artifacts/local/curvytron_optimizer_profile_results/optimizer-fresh-stock-gate-20260526a/row_001_result.json` | `274.668` | `954.4` | `called_train_muzero=true`, `env_steps_collected=262144`, H100 util max `3%` |
| stock RND-meter weight 0 | `artifacts/local/curvytron_optimizer_profile_results/optimizer-fresh-stock-gate-20260526a/row_002_result.json` | `322.745` | `812.2` | `called_train_muzero=true`, `train_muzero_with_reward_model`, RND trained `100` predictor steps |

Plain read after stock rows: the real Coach-compatible loop is still dominated
by stock collection/search/env-manager/object plumbing. The compact resident row
is far faster in its own currency. The compact loop now owns local device replay
targets and a local `compact_muzero` learner-gate update, but it is not a real
Coach speedup until the compact path is run through the Modal/grid gate and later
gets checkpoint/eval hooks and policy handoff.

Update: the first Modal/grid `compact_muzero` smoke row passed on H100
(`optimizer-compact-muzero-gate-smoke-20260526`, row `001`). It is still
profile-only, but the next compact-candidate rows can now use the stricter
learner denominator instead of the old `toy_probe`.

Update: the first strict compact MuZero scale wave also passed:
`optimizer-compact-muzero-scale-20260526`.

| Row | Hardware | Shape | Wall sec | Steps/sec | Plain read |
| --- | --- | --- | ---: | ---: | --- |
| 001 | H100 | B512/A16/sim8 | `7.710` | `10625.5` | good larger-row baseline |
| 002 | H100 | B1024/A16/sim8 | `10.899` | `15032.9` | best strict compact row so far |
| 003 | L4/T4 | B512/A16/sim8 | `11.824` | `6928.0` | slower than H100 but usable |
| 004 | L4/T4 | B1024/A16/sim8 | `23.698` | `6913.7` | no throughput gain from B1024 |

All four rows used `compact_muzero`, support scale `300`, explicit successor
targets, resident observations, device replay rows, learner input H2D `0`,
observation H2D `0`, committed replay payload D2H `0`, and
`calls_train_muzero=false`.

Next strict compact wave:

- H100 B1024, sample batch `512`, interval `4`;
- H100 B1024, sample batch `256`, interval `8`;
- H100 B1024, sample batch `512`, interval `8`;
- H100 B2048, sample batch `256`, interval `4`, if memory allows;
- optional L4 B512 repeat only if we need a cheap sanity row.

Do not use these rows as Coach launch promises. Use them to decide whether the
compact-owned candidate is worth the next promotion step.

Status: completed.

| Row | Hardware | Shape | Sample gate | Wall sec | Steps/sec | Plain read |
| --- | --- | --- | --- | ---: | ---: | --- |
| baseline repeat | H100 | B1024/A16/sim8 | B256 every 4 | `16.835` | `14598.5` | slower repeat, same bottleneck split |
| larger batch | H100 | B2048/A16/sim8 | B256 every 4 | `24.810` | `19811.4` | best raw throughput in this wave |
| persistent render state | H100 | B1024/A16/sim8 | B256 every 4 | `17.192` | `14294.7` | not useful here |
| larger learner sample | H100 | B1024/A16/sim8 | B512 every 4 | `14.580` | `16855.9` | better than baseline |
| lower learner cadence | H100 | B1024/A16/sim8 | B256 every 8 | `15.932` | `15425.3` | sample gate cheaper, total only mildly better |
| larger sample, lower cadence | H100 | B1024/A16/sim8 | B512 every 8 | `14.371` | `17100.7` | best B1024 row |

Next experiment queue is no longer sample-gate-first. The next measured rows
should isolate the actor/search step boundary and remaining synchronization:
action D2H, mask H2D, observation refresh, and per-step Python ownership.

Read rule:

- Stock rows answer what Coach can trust today.
- Compact rows answer whether the resident compact candidate is worth promoting.
- Do not divide compact `steps/sec` by stock `env steps/sec` and call that a
  training speedup.

## Wave 1: Matched Full-Loop Gate A

Purpose: answer whether a candidate improves the stock `train_muzero` profile
denominator, not just a compact sub-loop.

- Stock baseline: current stock `train_muzero` profile row.
- Candidate: same command shape with the candidate search/observation backend.
- Required match: hardware, C/N/batch/sims, RND, death/noise, checkpoint/eval/GIF
  side effects, observation contract, learner calls, replay sample calls.
- Required output: `speed_currency=stock_train_muzero_profile_env_steps_per_sec`.

Promotion rule: ignore any row the Gate A comparator rejects.

## Wave 2: Compact C0-C8 Falsifier

Purpose: identify the wall inside the compact/search/dataflow loop.

2026-05-23 active quick wave:

- `opt-c-falsifier-20260523-direct`: direct CTree GPU-latent and precomputed
  recurrent outputs, H100, B512, A16, sim8, 80 steps, 20 warmup.
- `opt-c-falsifier-20260523-service-tax`: service-tax search, same denominator.
- `opt-c-falsifier-20260523-mock`: mock search service, same denominator.
- `opt-c-falsifier-20260523-mock-public`: mock search plus public-output
  materialization, same denominator.
- `opt-c-falsifier-20260523-dense-torch`: dense Torch MCTS prototype, same
  denominator.

All are profile-only hybrid rows. They do not call `train_muzero` and are not
Coach speed claims.

2026-05-23 active clean follow-up wave:

- `opt-c-falsifier-clean-direct-20260523`: direct CTree GPU-latent and
  precomputed recurrent, H100, B512, A16, sim8, 200 steps, 50 warmup, no sample
  gate. Includes scalar off/on rows for C4.
- `opt-c-falsifier-clean-service-tax-20260523`: service-tax search, same
  denominator, no sample gate.
- `opt-c-falsifier-clean-mock-20260523`: mock/no-search ceiling, same
  denominator, no sample gate.
- `opt-c-falsifier-clean-dense-torch-20260523`: dense Torch MCTS prototype,
  same denominator, no sample gate.

Reason: the quick wave showed sample-gate materialization dominating measured
loop time, so this wave isolates search/dataflow from replay-materialization
tax.

Clean compact follow-up results, 2026-05-23:

| row | compact currency | steps/sec | roots/sec | useful read |
| --- | ---: | ---: | ---: | --- |
| direct CTree GPU-latent, scalar off | compact profile | `7191.39` | `12870.03` | current compact direct baseline |
| direct CTree GPU-latent precomputed recurrent, scalar off | compact profile | `7188.08` | `14113.03` | model/recurrent saving improves roots/sec but not wall steps/sec |
| direct CTree GPU-latent, scalar on | compact profile | `4651.58` | `11162.63` | scalar materialization costs about `35%` in this denominator |
| direct CTree GPU-latent precomputed recurrent, scalar on | compact profile | `5178.76` | `13612.35` | precompute helps more when scalar path is on |
| service-tax fake tree | compact profile | `5174.37` | `32000.27` | high roots/sec does not automatically mean high loop steps/sec |
| mock/no-search | compact profile | `5263.10` | `43154.44` | no-search ceiling is not faster than direct loop wall here |
| dense Torch MCTS prototype | compact profile | `7898.04` | `15715.21` | fastest compact wall row, but not yet CTree semantics/trainer-ready |

Plain interpretation: compact rows are useful for locating the wall, not for
claiming Coach training speed. The odd result is that fake/no-search rows have
high roots/sec but worse wall steps/sec than direct/dense rows. That means the
closed-loop denominator is still dominated by other fixed costs and needs a
fixed-shape owner/validation pass before we promote any search backend.

2026-05-23 fixed-shape owner local gate:

- `fixed_shape_search_owner` is now a profile-only array-ceiling mode and a
  compact rollout slab search service.
- It does not run LightZero CTree and does not run real MCTS yet; it picks first
  legal actions to test fixed `R`, `A=3`, active-root masks, action feedback,
  replay identity, and no CTree/list/per-sim host round-trip counters.
- Local tests passed (`161 passed, 2 warnings`) for the owner, boundary adapter,
  compact slab route, and grid builder.

Fixed-shape comparison wave, 2026-05-23:

All rows were H100, B512/A16/sim8, 200 measured steps, 50 warmup, compact slab
on, scalar materialization off, profile-only.

| row | steps/sec | roots/sec | search sec | env runtime sec | status |
| --- | ---: | ---: | ---: | ---: | --- |
| direct CTree GPU-latent | `7165.96` | `12510.68` | `6.92` | `4.10` | trusted compact comparator |
| direct CTree precomputed recurrent | `9386.40` | `17546.66` | `3.62` | `3.36` | fastest compact wall row, profile-only precompute |
| dense Torch MCTS | `7717.58` | `15916.25` | `2.27` | `4.47` | profile-only, not CTree semantics |
| compact Torch fixed-shape | `9136.79` | `20431.41` | `7.28` | `4.01` | profile-only, valid adapter after local fix |
| fixed-shape first-legal owner | `5152.73` | `518934.96` | `0.03` | `31.97` | boundary proof only |
| service-tax probe | `4248.20` | `25956.73` | `0.35` | `32.18` | action-confounded ceiling |
| mock/no-search | `4666.46` | `38281.90` | `0.00` | `31.40` | action-confounded ceiling |

Interpretation: this wave did not fairly answer closed-loop search speed because
different search lanes drove different actions. The first-legal/mock/service-tax
rows made env runtime about `8x-9x` higher than direct rows. Keep the result as
a boundary proof: search-owner overhead can be tiny, but the closed-loop wall
comparison needs controlled actions.

Next compact rows to run, profile-only:

- Rerun direct CTree, direct precomputed recurrent, compact Torch fixed-shape,
  fixed-shape first-legal, service-tax, and mock/no-search under that controlled
  action stream.
- Report both search/service timing and closed-loop wall timing, with a clear
  label that replay correctness is not claimed if actions are overridden.

Controlled-action mode is now available as
`hybrid_compact_rollout_slab_action_mode=scripted_random`. Required result
checks for the rerun:

- `env_action_checksum_total` matches across all rows.
- `env_trajectory_checksum_total` matches across all rows.
- `compact_rollout_slab_action_override_drop_count` is nonzero.
- `compact_rollout_slab_committed_index_row_count` is zero.
- search/service timing is still reported.

Launched controlled H100 wave, 2026-05-23:

- `opt-controlled-direct-slab-h100-b512-s8-20260523`: direct CTree GPU-latent
  and direct CTree precomputed recurrent.
- `opt-controlled-compacttorch-slab-h100-b512-s8-20260523`.
- `opt-controlled-fixedshape-slab-h100-b512-s8-20260523`.
- `opt-controlled-densetorch-slab-h100-b512-s8-20260523`.
- `opt-controlled-servicetax-slab-h100-b512-s8-20260523`.
- `opt-controlled-mock-slab-h100-b512-s8-20260523`.

All are profile-only, H100, B512/A16/sim8, 200 measured steps, 50 warmup,
scalar materialization off, compact slab on, `scripted_random`.

Controlled H100 wave results:

| row | steps/sec | search sec | slab sec | measured sec |
| --- | ---: | ---: | ---: | ---: |
| direct CTree GPU-latent | `6823.53` | `8.23` | `17.32` | `30.01` |
| direct CTree precomputed recurrent | `8840.06` | `4.39` | `12.47` | `23.17` |
| compact Torch fixed-shape | `6924.39` | `8.44` | `12.54` | `29.58` |
| dense Torch MCTS | `9326.97` | `1.78` | `10.84` | `21.96` |
| service-tax probe | `11375.15` | `0.31` | `6.97` | `18.00` |
| mock/no-search | `14019.12` | `0.00` | `4.77` | `14.61` |
| fixed-shape first-legal | `19000.09` | `0.04` | `0.35` | `10.78` |

All rows matched:

```text
env_action_checksum_total = 105265044
env_trajectory_checksum_total = 491942602644
```

Next fair-wave candidates:

- direct CTree/dense Torch/mock/fixed-shape at B1024 and maybe B2048 to see
  whether the non-search floor amortizes.
- sim16/sim32 for direct CTree, dense Torch, and mock to see how search cost
  scales once the trajectory is fixed.
- explicit non-search floor rows: observation off, mechanics no-op/frozen
  advance, compact batch build off if a clean switch exists.

Launched controlled scaling wave:

- `opt-controlled-scale-direct-h100-20260523`: direct CTree GPU-latent and
  precomputed recurrent, B512/B1024 x sim8/sim16.
- `opt-controlled-scale-dense-h100-20260523`: dense Torch MCTS,
  B512/B1024 x sim8/sim16.
- `opt-controlled-scale-mock-h100-20260523`: mock/no-search,
  B512/B1024 x sim8/sim16.
- `opt-controlled-scale-fixedshape-h100-20260523`: fixed-shape first-legal,
  B512/B1024 x sim8/sim16.

All are profile-only, H100, compact slab on, `scripted_random`, 120 measured
steps, 50 warmup.

Controlled scaling wave results:

| batch | sims | row | steps/sec | search sec | slab sec | measured sec | read |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 512 | 8 | direct CTree GPU-latent | `7448.40` | `4.11` | `9.36` | `16.50` | current real CTree comparator |
| 512 | 8 | direct CTree precomputed recurrent | `7959.32` | `2.61` | `7.91` | `15.44` | recurrent/model ceiling |
| 512 | 8 | dense Torch MCTS | `9457.66` | `1.14` | `6.65` | `12.99` | best real-search-ish row at B512 |
| 512 | 8 | mock/no-search | `11572.89` | `0.00` | `3.10` | `10.62` | no-search floor with model-ish edge |
| 512 | 8 | fixed-shape first-legal | `16324.18` | `0.02` | `0.20` | `7.53` | owner overhead floor only |
| 1024 | 8 | direct CTree GPU-latent | `9341.27` | `5.55` | `15.90` | `26.31` | B1024 improves direct by `1.25x` |
| 1024 | 8 | direct CTree precomputed recurrent | `10182.98` | `4.93` | `14.03` | `24.13` | precompute still helps |
| 1024 | 8 | dense Torch MCTS | `11700.40` | `1.72` | `9.97` | `21.00` | strongest real-search-ish profile row |
| 1024 | 8 | mock/no-search | `15189.51` | `0.00` | `5.22` | `16.18` | non-search floor amortizes with batch |
| 1024 | 8 | fixed-shape first-legal | `22180.86` | `0.03` | `0.28` | `11.08` | fixed owner boundary ceiling |
| 512 | 16 | direct CTree GPU-latent | `6093.39` | `7.83` | `13.03` | `20.17` | sim16 worsens direct wall |
| 512 | 16 | dense Torch MCTS | `5459.24` | `3.72` | `15.10` | `22.51` | sim16 worsens dense at B512 |
| 1024 | 16 | direct CTree GPU-latent | `6469.55` | `13.51` | `25.73` | `37.99` | direct CTree search scales poorly here |
| 1024 | 16 | dense Torch MCTS | `8857.01` | `4.24` | `16.36` | `27.75` | B1024 helps but still worse than sim8 |

Result: keep B1024/sim8 as the best compact-profile setting for the next
controlled rows. Do not treat sim16 as a default just because it does more
search; in this denominator it burns wall time faster than it buys throughput.

Kill rule: do not promote fixed-shape ownership until it either runs real MCTS
or proves a large fair win on the controlled-action compact denominator. Do not
use first-legal action rows as Coach speed claims.

## Completed Non-Search Floor Rows

These rows are done. Keep them as constraints while implementing compact
ownership; do not treat them as the current critical path.

- `nonsearch-floor-fixedshape-baseline`: B1024/sim8, fixed-shape first-legal,
  current observation/actor settings.
- `nonsearch-floor-fixedshape-no-refresh`: same row with observation-stack
  refresh disabled, profile-only label.
- `nonsearch-floor-fixedshape-device-only-stack`: same row with host stack
  update disabled, profile-only label.
- `nonsearch-floor-fixedshape-native-buffer`: same row with native actor buffer
  enabled independently.
- `nonsearch-floor-directctree-input-mode`: direct CTree B1024/sim8, compare
  `host_uint8_pinned` versus current input mode if supported.

## Next Immediate Implementation

- Make the compact slab call a two-phase search owner when available.
- Env-critical step returns selected actions plus stable identity only.
- Replay/search payload flushes later by handle when commit actually needs it.
- Keep this profile-only until identity, RND, replay, and denominator gates pass.

This has now been implemented for compact Torch and profiled against direct
CTree. See `COMPACT_TWO_PHASE_PROFILE_2026-05-24.md`.

Latest matched result:

| batch | direct CTree steps/sec | compact Torch two-phase steps/sec | speedup | floor steps/sec |
| ---: | ---: | ---: | ---: | ---: |
| 512 | `6103.25` | `8785.72` | `1.44x` | `18343.48` |
| 1024 | `8756.62` | `12040.17` | `1.38x` | `16548.13` |
| 2048 | `7463.65` | `11998.74` | `1.61x` | `19816.19` |

Next queue:

- Optimize the sample gate path. The replay-flush honesty row completed and
  found `sample_gate_sec=12.432s` out of `24.260s`; the learner gate itself was
  only `0.343s`.
- Trainer-like compact denominator: compact replay rows plus RND/latest-frame
  data plus learner/replay sample proof, still no live Coach run.

These rows are not learning claims. They are to identify which part of the
remaining `10-16s` floor is observation, actor/parent exchange, device transfer,
or compact batch construction.

Launched, 2026-05-23, all profile-only H100 B1024/A16/sim8, 200 measured
steps, 80 warmup, compact slab on, `scripted_random`:

- `opt-nonsearch-floor-fixed-baseline-h100-b1024-s8-20260523`
- `opt-nonsearch-floor-fixed-norefresh-h100-b1024-s8-20260523`
- `opt-nonsearch-floor-fixed-devicestack-h100-b1024-s8-20260523`
- `opt-nonsearch-floor-fixed-nativeactor-h100-b1024-s8-20260523`
- `opt-nonsearch-floor-fixed-persiststate-h100-b1024-s8-20260523`
- `opt-directctree-input-hostuint8-h100-b1024-s8-20260523`
- `opt-directctree-input-pinned-h100-b1024-s8-20260523`

Interpret these carefully: no-refresh and device-only rows are floor ablations,
not semantic training candidates. Direct input-mode rows are closer to real
direct CTree timing and should keep matching action/trajectory checksums before
their host-transfer timing is compared.

Initial results, same checksums for all rows:

```text
env_action_checksum_total = 420223838
env_trajectory_checksum_total = 1967449471838
```

| row | steps/sec | measured sec | actor wall | observation | compact batch | probe/slab | h2d | model | search | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed-shape baseline | `17396.93` | `23.54` | `10.40` | `9.22` | `2.94` | `0.59` | `0.00` | `0.00` | `0.05` | real no-search/no-model floor with current observation path |
| fixed-shape no refresh | `39172.24` | `10.46` | `5.72` | `2.05` | `2.00` | `0.45` | `0.00` | `0.00` | `0.04` | floor collapses when renderer/stack refresh is removed |
| fixed-shape device-only stack | `20252.46` | `20.22` | `9.61` | `6.65` | `3.02` | `0.57` | `0.00` | `0.00` | `0.05` | skipping host update alone helps only a little |
| fixed-shape native actor | `21176.78` | `19.34` | `8.96` | `7.21` | `2.43` | `0.54` | `0.00` | `0.00` | `0.05` | native buffer helps but does not remove the main floor |
| fixed-shape persistent state | `25519.63` | `16.05` | `7.55` | `5.81` | `2.04` | `0.48` | `0.00` | `0.00` | `0.04` | persistent render-state buffer is a real medium win |
| fixed-shape no slab | `13988.97` | `29.28` | `9.82` | `8.91` | `4.60` | `4.83` | `0.00` | `0.00` | `0.07` | removing slab is slower; slab path is not the floor |
| mock host uint8 | `9721.83` | `42.13` | `9.83` | `12.42` | `6.23` | `12.29` | `9.65` | `1.71` | `0.00` | mock is no-tree, not no-model; H2D dominates |
| mock pinned uint8 | `18859.01` | `21.72` | `8.27` | `8.06` | `2.03` | `3.00` | `0.57` | `1.70` | `0.00` | pinning matters when H2D is the visible wall |
| direct CTree host uint8 | `7440.08` | `55.05` | `10.06` | `9.04` | `2.87` | `32.68` | `7.75` | `4.64` | `12.40` | direct CTree remains search/model/control heavy |
| direct CTree pinned uint8 | `7312.99` | `56.01` | `10.74` | `11.68` | `4.60` | `28.24` | `0.29` | `4.64` | `13.12` | pinning removes H2D but does not improve total wall here |

Plain read: the current no-search floor is not slab/search. It is mostly
observation/render-stack work plus actor/env and compact batch construction.
The best floor row so far is no-refresh at `39k` steps/sec, but that is not a
semantic training candidate because it uses stale/zero observation. The best
closer-to-real floor row is persistent render state at `25.5k` steps/sec.
Pinned host input is valuable for mock/no-tree rows but not sufficient to move
the direct CTree total wall.

Follow-up rows launched to see whether the wins stack:

- `opt-nonsearch-floor-fixed-norefresh-nativeactor-h100-b1024-s8-20260523`
- `opt-nonsearch-floor-fixed-devicestack-persiststate-h100-b1024-s8-20260523`
- `opt-mock-input-pinned-norefresh-h100-b1024-s8-20260523`
- `opt-directctree-input-pinned-norefresh-h100-b1024-s8-20260523`

Follow-up results, same checksums:

| row | steps/sec | measured sec | actor wall | observation | compact batch | probe/slab | h2d | model | search | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed-shape no refresh + native actor | `29267.28` | `14.00` | `7.39` | `2.99` | `2.84` | `0.55` | `0.00` | `0.00` | `0.05` | worse than no-refresh alone; native actor does not stack cleanly |
| fixed-shape device-only + persistent state | `23214.77` | `17.64` | `8.80` | `5.12` | `2.93` | `0.53` | `0.00` | `0.00` | `0.05` | worse than persistent state alone |
| mock pinned + no refresh | `26363.76` | `15.54` | `7.10` | `2.55` | `2.54` | `3.01` | `0.58` | `1.69` | `0.00` | pinning and no-refresh stack for the no-tree model path |
| direct CTree pinned + no refresh | `10333.14` | `39.64` | `7.87` | `3.18` | `2.87` | `25.35` | `0.29` | `4.63` | `12.51` | a real profile win over direct baseline, but still CTree/search heavy |

Conclusion for next wave:

- Do not spend more time on `hybrid_device_only_stack` as currently wired; it
  does not beat persistent state or no-refresh.
- Keep persistent render-state buffers as the practical observation cleanup
  candidate.
- Use pinned input for mock/direct rows when H2D is a suspected wall, but do not
  expect it to fix direct CTree wall time by itself.
- The highest-value implementation lane is still two-phase real fixed-shape
  search plus delayed replay payloads. The highest-value runtime lane is
  resident/compact observation ownership so fresh policy observations do not
  require the current host stack refresh cost.

## Active Gate A Rows

2026-05-23 active warm matched stock-vs-candidate rows:

- `gate-a-directctree-h100-rndoff-warm-20260523`
- `gate-a-directctree-h100-rndmeter-warm-20260523`
- `gate-a-directctree-l4-rndoff-warm-20260523`
- `gate-a-directctree-l4-rndmeter-warm-20260523`

Each directory has two rows: stock collect search and
`direct_ctree_gpu_latent`. Denominator: subprocess env manager, C512, batch 64,
sim4, death disabled for profile, 32 learner calls, eval/GIF/checkpoint side
effects off. RND off and RND meter are intentionally separate comparisons
because RND changes denominator metadata. These call `train_muzero` in profile
mode only.

Interim H100 read, 2026-05-23:

- no-RND row is attested by the Gate A summarizer:
  - stock: `928.91` env steps/sec;
  - `direct_ctree_gpu_latent`: `1203.60` env steps/sec;
  - ratio: `1.30x`.
- RND-meter row has the same shape but strict Gate A attestation currently
  rejects it because raw collector-step telemetry is missing and the profile
  falls back to `mcts_search_root_sum_profile_fallback`:
  - stock: `992.60` fallback-profile steps/sec;
  - `direct_ctree_gpu_latent`: `1270.13` fallback-profile steps/sec;
  - ratio: `1.28x`;
  - treat this as a sidecar signal until RND profile telemetry is fixed or
    intentionally attested.

Plain interpretation: direct CTree GPU-latent is a real matched-profile cleanup
on H100, roughly `1.28x-1.30x`. It is not the `5x-10x` architecture move.

Interim L4 read, 2026-05-23:

- no-RND row is attested by the Gate A summarizer:
  - stock: `591.04` env steps/sec;
  - `direct_ctree_gpu_latent`: `846.96` env steps/sec;
  - ratio: `1.43x`.
- RND-meter row has the same fallback telemetry caveat as H100:
  - stock: `577.71` fallback-profile steps/sec;
  - `direct_ctree_gpu_latent`: `903.82` fallback-profile steps/sec;
  - ratio: `1.56x`;
  - treat as sidecar until RND profile collector-step telemetry is fixed.

Plain interpretation: L4 benefits more from direct CTree GPU-latent than H100,
but still only as a cleanup/profile win. It does not remove the deeper
LightZero object/search ownership wall.

- C0: current compact direct baseline.
- C1: observation refresh off.
- C2: mechanics no-op.
- C3: scalar materialization on/off boundary.
- C4: replay full-payload materialization on/off boundary.
- C5: precomputed recurrent outputs.
- C6: service-tax fake tree.
- C7: no-search/mock action.
- C8: largest clean batch/root count that fits.

Promotion rule: a row matters only if the expected bucket collapses and the
denominator is unchanged.

## Wave 3: Trainer T0-T5 Falsifier

Purpose: see whether compact wins can reach the real trainer.

- T0: full trainer profile with RND on.
- T1: replay no-op.
- T2: learner no-op.
- T3: replay real, learner no-op.
- T4: learner real, replay minimal.
- T5: RND update/estimate cadence isolated.

Promotion rule: if compact rows get faster but trainer rows do not, the speedup
is trapped outside Coach and should not be sold as training speed.

## Wave 4: Architecture Probe

Purpose: test the big move.

- Fixed-shape dense search prototype with `R=B*P`, `A=3`, padded sim/node
  tensors, and preallocated outputs.
- Resident observation/search-input prototype: avoid CPU readback until a coarse
  edge.
- Compact replay owner prototype: store index rows first, materialize only at
  sampler/learner edge.

Target: prove whether the realistic path is `1.5x-2.5x` cleanup or a larger
`5x+` architecture rewrite.
