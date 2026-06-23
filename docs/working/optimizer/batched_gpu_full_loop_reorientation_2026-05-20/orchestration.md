# Orchestration

Date: 2026-05-20

Purpose: keep the optimizer lane focused on the next real integration proof:
preserve batched GPU observations through a LightZero-shaped full-loop profile.

## Latest, 2026-05-22

Current center:

```text
HybridCompactBatch is the boundary of truth.
```

The compact sidecar now has three checked consumers:

- real LightZero direct CTree profile hook, still profile-only;
- RND latest-frame input adapter, input-shape proof only;
- target-row policy-record adapter, active-root ordered output proof only.

The new target-row bridge proves:

```text
selected_action[active_root]
visit_policy[active_root,3]
root_value[active_root]
-> PolicyRowRecordV0
-> build_source_state_multiplayer_target_rows_v0(...)
```

It validates row/player ids, `to_play=-1`, active-root masks, binary legal
masks, legal selected actions, legal visit-policy mass, P4 active-root order,
and mixed active/done rows. It does not claim native LightZero replay,
`GameSegment`, learner update, policy improvement, or Coach-facing speed.

The combined local edge now also passes:

```text
HybridCompactBatch
-> direct CTree compact profile hook
-> compact action/visit/value arrays
-> target-row adapter
-> checked source-state target rows
```

So the main lane should become more aggressive. Stop spending primary effort on
small compatibility edges unless a new semantic hole appears.

Next main lane:

1. Keep direct CTree compact parity/statistical gates green.
2. Build the closed compact batch consumer falsifier:
   search output + RND latest-frame input + target rows from one compact
   sidecar path.
3. If this cannot plausibly deliver a `3x` class profile win, escalate to the
   search-service/native vector-buffer architecture instead of polishing
   direct CTree wrappers.
4. Run matched full-loop A/B before any Coach advice.

Update after the first closed compact falsifier:

```text
B512/A16 closed compact arrays + native actor buffer:
  before RND latest-only fix: about 26.5k timesteps/sec
  after RND latest-only fix:  about 57.9k-62.8k timesteps/sec

B2048/A16 closed compact arrays + native actor buffer:
  closed compact:      about 71.6k timesteps/sec
  native-vector mock:  about 80.4k timesteps/sec
```

Plain read:

```text
The compact sidecar/RND/target adapter path is now cheap in the local
profile-only denominator. Do not confuse this with real MCTS or train_muzero.
The next aggressive target is the real collect/search/replay boundary:
per-simulation CPU/list CTree contracts, model-output D2H/listifying, and
per-env output fanout.
```

Current critical path:

1. Keep the compact sidecar as the contract.
2. Prototype compact search/replay output that avoids public per-env collect
   fanout.
3. Use B2048/A16 and B512/A16 as local shape controls; avoid A32 unless fresh
   evidence says the partition overhead is fixed.
4. Promote nothing to Coach until matched stock-loop A/B with RND/death gates
   passes.

Hard guardrail stays unchanged: no live training interference.

## Latest, 2026-05-21

Reorientation: the active problem is the whole-loop denominator, not another
render-only debate. Use
[whole_loop_denominator_ledger_20260521.md](whole_loop_denominator_ledger_20260521.md)
before making any speed claim. Actual training, stock full-loop profiles, and
profile-only boundary probes are different currencies.

Active lane: compact LightZero MCTS arrays boundary. The renderer/vector bridge
work is still important background, but the current Amdahl wall is around
LightZero collect/search.

Current direct probe:

```text
direct_ctree_arrays = real LightZero MuZero model + real CTree MCTS + compact arrays out
```

Latest H100 medium result:

```text
stock facade:                       2419.81 roots/sec
direct_ctree_arrays old host uint8: 3859.44 roots/sec
direct_ctree_arrays host uint8:     5247.95 roots/sec
direct_ctree_arrays pinned uint8:   4678.23 roots/sec
resident reuse ceiling:             5820.96 roots/sec
```

Late P2 refresh on the same H100 B512/A16/sim8 60/15 shape:

```text
stock facade host uint8:              2670.68 roots/sec
direct_ctree_arrays host uint8:       4764.06 roots/sec
direct_ctree_arrays pinned uint8:     3689.15 roots/sec
direct_ctree_arrays resident ceiling: 3069.08 roots/sec
```

Decision read: direct CTree over the public facade is still the active speed
signal. Pinned/resident input is not the active win; H2D shrank, but total wall
moved into search/root/model-output/variance buckets.

This is profile-only. It does not touch live runs or trainer defaults. Next
main-thread work is parity gates plus splitting the remaining direct path wall:
MCTS search/root prep/model/output/observation. Longer repeats now say pinned
input is a modest stable win (`4513.15` versus `4111.80` roots/sec), but not
the main wall. Resident reuse remains stale-input ceiling evidence only.

2026-05-21 latest validation note: the first parity/debug gate is in place, but
it deliberately does not claim exact action/visit parity yet. It proves
searched-value agreement, legal/normalized visit outputs, and illegal-action
guarding on real CPU LightZero sim1/sim2/sim8. Exact visit/action equality is
still open because separate real CTree calls can tie-break differently even with
Python/NumPy/Torch seeds fixed and root noise disabled. Keep
`direct_ctree_arrays` profile-only.

## Current Priority

Main lane: validate and split the direct CTree arrays boundary. The bridge must
keep the batch compact through observation and policy/search-shaped work, then
return compact action/value/visit arrays without public per-root fanout.

Background reconnection lane: keep the vector/full-loop bridge available for
matched stock `train_muzero` A/B/C after the direct boundary passes parity. The
bridge still matters, but it is not the current Amdahl wall.

2026-05-21 reorientation: the resident canary now shows that this sentence is
the real center of the optimizer lane. GPU rendering alone is no longer the
question. The question is whether we can avoid destroying the batch before
policy/search/replay work can use it.

Background diagnostic: duplicate-seed and runtime variance. Keep the repeated
stock CPU-oracle rows as anchor checks, but do not make variance investigation
the main lane unless workload counts or reset traces stop matching.

Separate profile axis: RND cadence. Treat `rnd_update_per_collect`,
CPU-vs-CUDA RND, and no-RND controls as independent profile settings. Do not
fold RND cadence wins or losses into renderer claims.

Hard guardrail: no live training interference. Use profile-only run ids,
profile-only flags, no default trainer changes, no tournament/checkpoint
promotion, and no touching active live runs.

## 2026-05-21 Superseded Renderer Background

Current main-thread state:

- Persistent GPU policy-space framebuffer is implemented as a profile-only
  backend: `jax_gpu_persistent_policy_framebuffer_profile`.
- It updates newly appended trail deltas instead of redrawing all historical
  trail slots every step.
- B512/100 improved from about `81ms` to about `45ms` per surface step.
- B512/500 before the env patch spent about `16ms` in render, about `39ms` in
  env step, and about `39ms` in host stack update.
- Cursor-bound body collision scanning reduced that B512/500 env step to about
  `14ms` and total surface step from about `79ms` to about `66ms`.

Interpretation:

```text
Renderer optimization worked.
Cursor-bound collision scanning helped.
The next wall is host stack/materialization, with env still worth monitoring.
```

Current correction: this section is background. The active optimizer lane has
since moved to profile-only direct CTree arrays, fixed-seed parity gates, and
the remaining LightZero collect/search boundary split.

Current parallel lanes:

- Main thread: patched and profiled cursor-bound body collision scanning.
- Lovelace subagent: advised against a tactical host-only ring buffer because it
  likely moves the copy into packaging unless the downstream policy consumes the
  ring/device contract.
- Ampere subagent: found and helped close the cursor-bound hit-attribution
  hole.
- Rawls subagent: identified the hybrid pre-scalar stack probe as the clean
  device-stack seam.
- Main thread then ran the first H100 hybrid grid:
  `uint8/no-scalar ~16.3k`, `uint8/scalar ~9.8k`, `float32/no-scalar ~9.2k`
  scalar timesteps/sec.
- Main thread then repeated the resident canary at sim8 on both H100 and L4:
  H100 scalar-off `~13.8k`, H100 scalar-on `~6.5k`, L4/T4 scalar-off `~9.0k`,
  L4/T4 scalar-on `~4.2k`, H100 device-latest scalar-off `~9.8k`.

Decision rule:

- Keep the cursor-bound collision patch only with broad collision/fidelity tests.
- If host stack update remains near `40ms`, do not guess: either build a
  profile-only device/uint8 stack contract or prove a ring buffer helps without
  materializing the same float32 tensor immediately afterward.
- Do not promote persistent renderer defaults until partial-row/autoreset,
  prefix mutation/map-clear, and backend-label gates are closed.
- Next experiment should use the compact `uint8` batched stack as the input to a
  GPU consumer before scalar materialization. Keep scalar LightZero payload as
  an optional measured edge, not as the hot-loop assumption.
- Device-latest is not a win yet because the host stack still exists. Do not
  call it an optimization until the no-scalar path can bypass host stack update
  rather than maintaining two stacks.

## 2026-05-21 Active Wave: Where Does The Batch Die?

Plain goal:

```text
Find the first place where a fast batched resident observation path turns back
into slow scalar Python/NumPy LightZero rows.
```

Active sub-agents:

- Peirce the 2nd: completed stock-boundary batch-death audit. Key finding:
  trusted stock path kills residency at scalar env observation rows and again
  around MuZero collect/search CPU NumPy conversion.
- Franklin the 2nd: completed resident-chunk prototype plan. The profile-only
  resident probe is now implemented.
- Plato the 2nd: completed RND/death guardrail critique. Keep RND and terminal
  semantics separate from speed-only rows.
- Carson the 2nd: active next real-consumer canary plan.
- Hooke the 2nd: active materialization byte/count instrumentation lane.
- Hooke the 2nd: completed materialization byte/count instrumentation.
- Bacon the 2nd: completed JAX/MCTX spike critique; recommends a tiny
  scratch-only spike after the LightZero real-consumer canary is unblocked.

Main-thread duties:

- keep docs current;
- do not touch live Coach runs;
- keep production defaults unchanged;
- use raw boundary-profile Modal commands for hybrid canaries because the
  optimizer manifest runner is stock-train-profile-specific;
- only patch shared tooling if it removes a real repeated source of confusion.

Current falsifiers:

- If scalar-off resident throughput does not stay clearly above the stock
  zero-observation ceiling once a real consumer is added, resident batching is
  not yet the next launch path.
- If scalar-on collapses near stock-boundary throughput, scalar LightZero
  materialization is the wall and a stock bridge is premature. Current
  profile-only scalar-on rows do not totally collapse, but they still pay a
  large tax and create `61,440` host timesteps in the medium B512/A16/sim8 row.
- If an RND or normal-death guardrail breaks, keep the work profile-only even if
  no-death/no-RND speed looks good.
- If B1024 does not beat B512, do not keep widening the synthetic resident
  probe. Current B1024 H100 stayed flat scalar-off and worsened scalar-on, so
  B512 is the default shape until a real consumer changes the curve.

2026-05-21 real-consumer update:

- The `--hybrid-lightzero-collect-forward-probe` gate now exists and passed a
  tiny L4/T4 remote smoke.
- It does not call `train_muzero`, does not touch live runs, and keeps
  `materialized_timestep_count=0` in the primary row.
- It flattens `[B,2,4,64,64]` to `[B*2,4,64,64]`, passes real action masks and
  player ids, calls actual `MuZeroPolicy.collect_mode.forward`, and decodes
  every root.
- Two image mistakes were found and fixed during the smoke: floating Torch
  installed CUDA13, and Torch `2.5.1` brought cuDNN too old for JAX. The profile
  image now pins `torch==2.8.0` with CUDA12/cuDNN `9.10`.
- Next rows: B512/A16/sim8 on H100 and L4, scalar edge off/on, compared to the
  synthetic resident anchor.
- Those rows are now measured. H100 scalar-off `2669.32` roots/sec, H100
  scalar-on `2100.31`, L4/T4 scalar-off `2159.35`, L4/T4 scalar-on `2053.57`.
  Compared with the synthetic resident probe, real collect-forward keeps only
  about `24-50%` of the synthetic throughput. The next wall is the actual
  LightZero collect/search path, not the renderer.
- Run-state lesson: PTY handles disappeared after compaction and Modal app logs
  did not retain the JSON payload for three rows. Any future detached or long
  profile wave must have a durable run registry plus artifact/result capture,
  not only local terminal output.

Next orchestration move:

```text
Split the real collect-forward bucket.
1. Measure real MuZero model initial inference only over the same pre-scalar
   [B,2,4,64,64] stack.
2. Compare that against full collect-forward.
3. If initial inference is cheap, the wall is LightZero CPU tree/MCTS/output
   handling. If initial inference is expensive, the wall is model/GPU batch
   topology.
```

Run registry rule:

```text
For every Modal profile wave, record the app/run id in experiment_log before
or immediately after launch, plus the exact compact table row when it returns.
Do not rely on local PTY session ids; they can disappear after compaction.
```

2026-05-22 correction:

```text
For hybrid boundary profile rows, use
scripts/run_curvytron_hybrid_observation_profile_manifest.py.
```

Reason:

```text
The stock train-profile manifest runner already has structured collection, but
the hybrid boundary profiler previously emitted raw shell commands only. That
made profile rows too easy to lose after compaction. The new runner is boring
on purpose: blocking Modal commands in, local JSON files out.
```

Active durable wave:

```text
mock_search_service sim8/sim16
direct_ctree_gpu_latent sim8/sim16
recurrent_toy sim8/sim16
```

All are H100 B512/A16, 60 measured steps, 15 warmup steps, profile-only,
scalar materialization off, root noise `0.0`, and no live-run side effects.

Result:

```text
mock_search_service sim16:       11648.29 roots/sec
direct_ctree_gpu_latent sim16:    5303.97 roots/sec
recurrent_toy sim16:              8512.57 roots/sec
```

Decision read:

```text
Search-service boundary ownership is worth investigating, but the measured
ceiling is about 2.2x over current direct, not 10x. The next radical plan must
combine search-service ideas with native/vector environment buffer ownership
or another larger topology change.
```

2026-05-21 critique fix status:

- Implemented fail-closed contract for real LightZero collect-forward rows:
  `uint8`, `direct_gray64`, persistent GPU profile backend.
- Switched collect-forward `to_play` to fixed-opponent `-1` to match the scalar
  source-state env convention.
- Added zero-action-mask filtering before LightZero.
- Added telemetry flags for CPU-tree-inclusive collect-forward timing.
- Focused tests and ruff passed.

2026-05-21 split result:

- Initial-inference-only rows passed on H100 and L4.
- Corrected collect-forward sim8 and sim1 rows passed on H100 and L4.
- H100 roots/sec:
  - initial inference `9238.85`
  - collect-forward sim1 `3296.02`
  - collect-forward sim8 `2693.10`
- L4 roots/sec:
  - initial inference `6790.63`
  - collect-forward sim1 `1687.81`
  - collect-forward sim8 `1381.35`
- Interpretation: the model root pass is not the current wall. The public
  LightZero collect/search path is.

2026-05-21 deeper-search split plan:

- Done: model-call timer inside `collect_mode.forward`.
- Done: subagent critique. Both review lanes said the next move is a narrow
  MCTS/search split, not another broad full-loop A/B.
- Done in code: the boundary collect-forward probe now times
  `policy._mcts_collect.search`, ctree `batch_traverse`, ctree
  `batch_backpropagate`, and residuals outside MCTS.
- Running: H100 and cheap-pool rows on the same B512/A16/sim8 shape.

Decision after rows return:

```text
MCTS search dominates:
  inspect or replace the search boundary next.
ctree traverse/backpropagate small but MCTS search large:
  focus on CPU/GPU conversion, NumPy/list glue, latent/logit movement, and
  root/output handling around C++.
outside-MCTS dominates:
  focus on root setup, legal-action preparation, action sampling, and output
  dict assembly.
```

Still not full-training truth: this is a profile-only boundary canary, not
`train_muzero`, replay, learner, RND reward, checkpoint, eval, GIF, or
tournament proof.

2026-05-21 pure-policy follow-up:

- MCTS collect sim8 H100: `2572.12` roots/sec, `35.36s` collect-forward.
- Pure-policy collect H100: `6286.61` roots/sec, `4.88s` collect-forward.
- Raw ctree traverse/backprop in the MCTS row: only `0.98s`.

Decision:

```text
The next optimizer lane is not another render pass and not just bigger GPU.
The target is the MCTS branch representation path: root setup, CPU/list
conversion, ctree wrapper/search result handling, and per-root output fanout.
```

Next subagent wave:

- Dewey the 2nd: design a small replacement-ceiling toy, roots in / compact
  arrays out, no production changes.
- Pascal the 2nd: source-audit `_forward_collect` and MCTS `search` to identify
  the exact CPU/list/dict-heavy operations and next timing hook.

Subagent results:

- Pascal confirmed the likely heavy pieces: root latent/logit CPU readback,
  legal-action lists, Dirichlet lists, root construction/prep, distribution
  list extraction, `select_action`, and one output dict per root. Inside MCTS
  search, each sim also does CPU latent gathering, tensor conversion,
  recurrent output CPU conversion, `.tolist()`, and latent history storage.
- Dewey proposed the next profile-only toy:
  - `policy_arrays`: initial inference plus masked compact arrays out;
  - `recurrent_toy`: initial inference plus batched recurrent calls and compact
    array outputs;
  - no `collect_mode.forward`, no `_mcts_collect.search`, no per-root dict/list
    output.

Main-thread decision:

```text
Build the toy as a profile-only HybridBatchedStackProbe, not as a trainer patch.
It is a ceiling test, not a learning algorithm change.
```

Active validation wave:

- Bohr the 2nd: coverage map, existing tests versus missing proof gates.
- Parfit the 2nd: smallest GPU-observation test additions if clear gaps exist.
- Gauss the 2nd: LightZero boundary and profile-semantics critique.
- Mendel the 2nd: RND/reset/death proof-gate critique.

Main-thread rule for this wave:

```text
Do not launch more broad scaling rows until the test/semantic matrix is clear
enough that a fast row cannot silently mean the wrong thing.
```

2026-05-21 boundary update:

- Summary-side speed-row attestation now exists in
  `scripts/summarize_curvytron_optimizer_profile_results.py`.
- The summarizer can fail under-labeled rows with `--require-attestation` and
  no longer overloads one `render` column for both render mode and render time.
- LightZero output decode tests now cover string-keyed rows, list outputs,
  nested wrappers, missing actions, and illegal decoded actions.
- Producer-side compact outputs now carry a `semantic_identity` block with
  explicit dtype, scalar materialization, `to_play`, zero-mask/action-mask, and
  consumer identity fields. Full `--require-attestation` now requires this
  block. Fresh stock profile smoke `opt-semantic-identity-smoke-20260521a`
  passes the guard; old artifacts can be summarized but are not fully current
  evidence.
- The attestation smoke also found two toolchain issues that are now fixed:
  the local runner now prefers the top-level compact JSON over nested schema
  objects, and the stock trainer image pins Torch `2.8.0` to avoid cuDNN
  mismatch with the CUDA12 stack.

Minimum promotion rows before Coach recommendations:

- no-RND matched A/B/C: CPU oracle, resident candidate, zero observation;
- `rnd_meter_v0` matched row with cadence and hash/reward checks;
- normal-death/autoreset semantic row;
- long no-death row for trail growth;
- real or realistic policy/search pressure row.

2026-05-21 reorientation correction:

- The stock-profile `rnd_meter_v0` and normal-death/autoreset gates have
  already passed as profile gates.
- They are not launch advice by themselves:
  - normal-death rows are semantic gates because live root batches collapse;
  - RND meter rows are overhead/safety gates, not positive-RND learning proof.
- Do not keep re-planning those gates as if they are absent. The current open
  optimizer work is:
  1. speed-row semantic attestation;
  2. LightZero decode/output edge cases;
  3. deeper collect/search instrumentation or toy comparison;
  4. positive-RND normalization later, separately.

2026-05-21 current status:

- Speed-row semantic attestation is no longer open for the stock smoke shape.
- The active optimizer lane is now deeper collect/search instrumentation. The
  H100/L4 split refresh repeated the earlier result: model initial inference is
  much faster than public LightZero collect-forward, so the next wall is inside
  collect/search/output handling.
- The first internal H100 timer row sharpened this: collect-forward spent
  roughly `69.8s` in its forward bucket over the measured row, but only about
  `2.7s` was timed model initial/recurrent inference. Treat the residual as the
  next target: LightZero tree/search/output handling around the model calls.

## Active Wave: Vector/Full-Loop Bridge

Question: can the fast batched direct GPU observation surface survive the
LightZero collection loop without collapsing back to scalar rendering or losing
too much wall time to render/stack work?

2026-05-20 update: zero-observation stock manager rows passed at C64 and C128.
That is important because it shows the one-process batched manager can beat the
subprocess CPU-oracle path when renderer pixels are removed. The current Amdahl
question is therefore narrower and cleaner:

```text
Can we make real browser_lines + simple_symbols observations cheap enough to
approach the zero-observation ceiling?
```

2026-05-21 update: post-patch C512 real GPU render reached about `1439.84`
steps/s, while the existing C512 zero-observation ceiling is about `1805.22`.
That says the observation path is still worth cleaning up, but the current
maximum remaining win from perfect observation removal is only about `1.25x` at
C512. The next phase should stop treating renderer-only work as the whole
answer and should compare:

```text
targeted renderer/stack cleanup
vs
manager/policy/search architecture work
vs
hybrid actor-parallel + batched GPU render
```

2026-05-21 gate update: normal-death/autoreset and RND meter now both pass as
profile gates. C768 did not scale cleanly: real render was flat versus C512 and
zero-observation was slower, which points at topology/scheduling rather than a
simple render wall. A new detached saturation grid is running:

```text
C512 x real/zero x sim2/sim4
C768 x real/zero x sim2/sim4
```

The purpose is not to find a launch recommendation for Coach. The purpose is to
decide whether the one-process batched manager is saturated and whether sim4
moves the wall to policy/search.

Concrete work:

- Wire a profile-only vector facade around `SourceStateMultiplayerTrainerSurface`
  and `direct_gray64`.
- Preserve `[B, players, stack, 64, 64]` inside the surface until the scalar
  LightZero payload boundary.
- Emit row/player scalar timesteps with correct live ids, rewards, dones,
  actions, and terminal `final_observation`.
- Run base-manager first; use subprocess rows only after the base bridge is
  semantically clean.
- Compare matched CPU-oracle and batched-GPU profile rows at the same topology,
  seed shape, collect steps, and simulation count.
- Report renderer time, surface non-render time, payload bytes, pickle time,
  learner/search counts, env steps, wall clock, and GPU utilization.

Pass gates:

- no hidden CPU fallback;
- exact backend name and metadata recorded;
- missing/extra scalar actions rejected;
- partial autoreset and neighboring-row stability covered;
- terminal `final_observation` attached before reset;
- RND latest-frame extraction works through the wrapper when RND is enabled;
- full-loop wall clock improves versus the matched CPU-oracle profile.

Follow-up trigger:

- If real batched GPU render approaches zero-observation throughput, expand the
  matched full-loop grid.
- If real batched GPU render stays far below zero-observation, optimize
  render/stack or prototype a hybrid subprocess-plus-GPU-render-service.
- If real batched GPU render is within about `25-35%` of zero-observation,
  prioritize manager/policy/search architecture over another renderer-only pass.
- If C768 stays flat or worse than C512, stop widening the one-process manager
  and move to architecture work.
- If normal-death rows collapse live roots, track live-root batch size before
  blaming renderer speed.
- If payload/process cost dominates after render is cheap, slim the LightZero
  payload before more renderer work.
- If search dominates after render is cheap, shift to MCTS/root batching or
  topology tuning.
- If the batch is lost before render, stop integration and fix the vector
  boundary first.

## Active Wave: External GPU Architecture Research

Question: are we missing a known high-throughput GPU RL pattern?

Current answer:

- Yes, but the pattern is bigger than "put one render on GPU." Fast systems
  either keep the env/reward/observation tensors on device end to end
  (Isaac Gym, Brax, CuLE), or keep CPU actors very parallel and batch the
  expensive model/search work (Sample Factory, EnvPool, AlphaZero-style actor
  farms).
- PufferLib adds a concrete nearby pattern: native/vector environments write
  observations, rewards, terminals, and actions into contiguous buffers; memory
  allocation is static; rollout buffers and training batches are designed
  together; data transfer is asynchronous and pinned; CUDA graph replay removes
  repeated launch overhead where possible. This is not a tiny LightZero wrapper
  patch. It is a native environment plus trainer-boundary architecture.
- Fresh local clone inspection confirmed the practical shape: `StaticVec` owns
  flat observation/action/reward/terminal/action-mask buffers, assigns envs
  direct pointers into their buffer slices, uses pinned host memory plus device
  buffers in GPU mode, and chunks work by buffer/stream. Treat that as the
  reference pattern for the next CurvyTron compact-buffer falsifier.
- Our current profile lane is a hybrid in the weak sense: render is batched on
  GPU, but every step still returns to host and scalarizes into LightZero
  timestep objects before policy/search.
- The next prototype with enough Amdahl room is a true hybrid: many CPU actor
  workers step compact CurvyTron state, one central batched GPU observation
  service renders large batches, and the stock boundary receives scalar
  timesteps only at the edge.

Follow-up docs:

- `subagent_gpu_architecture_research.md`
- `subagent_host_overhead_code_audit.md`
- `subagent_amdahl_next_priorities.md`
- `hybrid_actor_batched_observation_prototype_plan.md`
- `subagent_gpu_system_patterns_20260521.md`
- `subagent_gpu_renderer_boundary_20260521.md`
- `subagent_gpu_priority_critique_20260521.md`
- `subagent_web_gpu_rl_patterns_20260521.md`
- `subagent_web_gpu_2d_rendering_20260521.md`

## Active Wave: Hybrid Actor Plus Central Observation

Question: can we preserve CPU actor parallelism and still batch observation
work centrally?

Current proof:

- The profile-only in-process hybrid scaffold exists.
- It does not call `train_muzero`, change trainer defaults, touch live runs, or
  write tournament/checkpoint artifacts.
- Local zero-observation rows:
  - B64/A4: about `15425` scalar timesteps/sec.
  - B256/A8: about `21605` scalar timesteps/sec.
  - B512/A16: about `24878` scalar timesteps/sec.
- Modal H100 real-render hybrid rows now run:
  - B256/A8: about `4496` scalar timesteps/sec.
  - B512/A16: about `5447` scalar timesteps/sec.

Interpretation:

- This is not a training speed number. It excludes policy/search/replay/learner,
  RND, real render, and real subprocess IPC.
- The local zero rows proved topology headroom. The Modal real-render rows prove
  the central GPU renderer can be injected at larger batch shapes in a
  no-training harness. These actors are still in-process sequential partitions,
  not true subprocess actor parallelism. A stock bridge is still not justified
  until payload, terminal-row Modal smoke, true IPC/fan-in, and policy/search
  pressure gates pass.

Next implementation wave:

1. Keep the scaffold profile-only.
2. Add a compact-summary CLI path first; done.
3. Add a renderer-backed mode that accepts an injected renderer from the parent
   side; local CPU-oracle smoke and sentinel row/player-order tests now pass.
4. Build the Modal/profile-only wrapper that injects the real dynamic JAX
   renderer. Keep Modal/JAX construction outside `curvyzero.training`.
5. Measure compact actor payload bytes and render-service time separately from
   scalar materialization.
6. Only after the no-training harness wins, design a LightZero bridge. Do not
   wire this into Coach/live training early.

Implementation-risk note:

- See `subagent_hybrid_real_render_risks_20260521.md`.
- Main traps are wrong row/player ordering, terminal final observations rendered
  after autoreset, host-copy timing hiding the win, and accidental CPU fallback.
- The first sentinel renderer test covers row/player ordering. Terminal
  `final_observation` now has local CPU-oracle coverage, but still needs a small
  Modal GPU terminal row before we use it as evidence for any trainer bridge.
- The scalar materializer now accepts a real `action_mask` and the hybrid seam
  passes the merged actor mask through. This still does not make the synthetic
  policy/search probe a real LightZero MCTS row.

## 2026-05-20/21 Reorientation: Long-Trajectory Amdahl

Latest clean profile-only ladder uses dynamic trail slots, B512/A16, no death,
and no synthetic pre-scalar probe:

- 20 measured steps: about `3976` scalar steps/s.
- 100 measured steps: about `1780` scalar steps/s.
- 200 measured steps: about `1149` scalar steps/s.
- 500 measured steps: about `598` scalar steps/s.

The 500-step row spends about `822s` of `856s` in observation and about `802s`
inside renderer render. Plain conclusion: for longer-lived policies, the next
optimizer wave should focus on the long-trail observation renderer or a cheaper
observation representation. CPU scalarization and stack dtype are narrower
secondary wins unless the next bridge consumes batched/device stacks directly.

Active sub-agent wave:

- Mencius: inspect current GPU/browser-lines renderer code and identify the
  clean seam for incremental/dirty long-trail rendering.
- Russell: critique lower-fidelity observation representations that preserve
  game signal while avoiding full long-trail redraw.
- Erdos: research known fast 2D GPU trail/sprite rendering patterns and rank
  practical options for this repo.
- Dirac the 2nd: run an isolated toy benchmark comparing full redraw against
  incremental/persistent updates.

Main-thread rule for this wave: do not touch live Coach training runs. Keep any
implementation profile-only until the seam and fidelity risks are understood.

Wave result:

- Code seam audit confirmed the hot JAX block renderers redraw every selected
  trail slot each frame and run a browser-lines previous-owner prepass.
- Observation-design critique recommends a named persistent policy-space
  surface, ideally 128x128->64x64 as a fidelity/speed compromise and 64x64 as
  the maximum-speed ablation.
- GPU/render research ranks persistent policy-space framebuffer first and
  warns that readback/ownership still matter.
- Two toy/prototype lanes support the cost-model change:
  - H100/L4 synthetic Modal benchmark: `3.67x` to `10.86x` readback-included
    speedup, up to `38.57x` without readback, exact parity against the
    stateless synthetic target.
  - Local direct-64x64 append-only toybench: `10x` to `210x` speedup with
    exact final-frame parity.

Next orchestration move: design and implement a real profile-only persistent
renderer gate behind `SourceStateBatchedObservationRenderer`. It must fail
closed or rebuild on reset/cursor regression/prefix mutation/game clear, report
dirty/rebuild telemetry, preserve controlled-player views, and compare against
a stateless reference before any Coach recommendation.

## 2026-05-21 Active Wave: Divergence Before Promotion

Main-thread objective: do not optimize a proxy blindly. The current wave is a
proof gate, not a trainer change.

Parallel lanes:

- Fidelity critique: identify the missing long-trajectory CPU-vs-GPU
  comparison and the exact semantics it must cover.
- Amdahl critique: decide whether renderer work still matters once full-loop
  manager/search/RND costs are included.
- Local implementation: add the profile-only divergence canary and keep it off
  by default.
- Modal smokes: CPU-vs-CPU exact control, persistent GPU no-death divergence,
  and timeout/autoreset divergence.

Decision rules:

- If CPU-vs-CPU is not exact, fix the canary before trusting any result.
- If persistent GPU diverges catastrophically, keep it as a speed lens only and
  do not recommend it to Coach.
- If persistent GPU stays semantically close over long no-death and reset rows,
  the next proof is matched full-loop A/B against CPU oracle and zero
  observation, with RND measured separately.
- Never compare renderer-only rows to training rows as if they are the same
  workload.

## Background Diagnostic: Seed/Runtime Variance

Question: are repeated stock rows stable enough to act as comparison anchors?

Use only when needed:

- repeat the stock H100 C512/sim4 CPU-oracle no-RND row when a bridge result
  needs a fresh anchor;
- verify workload counts before comparing throughput;
- add a same-seed reset/action trace only if repeated rows disagree beyond
  normal runtime noise;
- record findings as diagnostic context, not as the next optimization lane.

Follow-up trigger:

- If workload counts diverge, fix determinism before trusting A/B rows.
- If only wall time varies, keep using conservative anchors and continue the
  bridge work.

## Separate Axis: RND Cadence

Question: how much does RND cost under each cadence/profile setting?

Keep this modular:

- run no-RND controls beside RND rows when measuring full-loop throughput;
- profile `rnd_update_per_collect` as a named cadence axis;
- keep CUDA-vs-CPU RND comparisons separate from observation renderer results;
- treat positive-weight RND as a launcher/process gate, not a renderer gate.

Follow-up trigger:

- If RND dominates, tune cadence or RND execution separately.
- If RND is disabled, do not use that row to make RND launch claims.

## Not Main Lane

- More surface-only renderer wins without a LightZero bridge.
- Explaining duplicate-seed runtime variance before building the bridge.
- Promoting scalar `policy_observation_backend=jax_gpu`.
- Changing live training, trainer defaults, tournament defaults, or checkpoint
  metadata consumers.

## Active Parallel Wave: Beyond 1.8x Search Boundary

Question:

```text
How do we actually escape the direct_ctree_arrays 1.8x ceiling?
```

Current main-thread read:

- `direct_ctree_arrays` is useful but still keeps LightZero's Python
  per-simulation search loop.
- LightZero CTree has C++ kernels, but its public Cython wrapper still accepts
  lists and returns list-shaped data.
- A root/output-only Cython patch is probably not enough. A deeper Cython
  search-loop boundary or a different batched-search architecture is needed.

Delegated sidecars:

- C++/Cython boundary feasibility:
  `/private/tmp/curvy_ctree_sidecar_handoff.md`.
- Accelerator-native MCTS alternatives:
  `/private/tmp/curvy_accelerator_mcts_handoff.md`.
- Batched actor / leaf-batching architecture:
  `/private/tmp/curvy_batched_actor_arch_handoff.md`.

Decision rule:

- If the C++/Cython sidecar says root/output arrays are small but
  per-simulation array backprop/traverse is feasible, implement the staged
  Cython boundary first.
- If the accelerator sidecar shows an easy MCTX toy canary, run it in parallel
  as a ceiling probe.
- Do not confuse either lane with live Coach training until semantic gates pass.

## 2026-05-22 Current Parallel Wave

The compile-spike falsifier has completed and failed the sim16 gate:

```text
dense_torch_mcts_compile_spike sim16: 4872.70 roots/sec
direct_ctree_gpu_latent sim16:        6153.95 roots/sec
```

That means the main thread should stop polishing this exact compiled helper.
The next work is to price the architecture, not to guess.

Active delegated work:

- Faraday: design a profile-only mock search-service ceiling row. The row
  should consume real batched observations and legal masks, skip real
  CTree/search, return compact legal action/visit/value arrays, and tell us
  whether removing the LightZero-shaped search/output boundary exposes a large
  enough ceiling.
- Ohm: critique array-native fixed-A=3 CTree feasibility. The question is
  whether flat arrays through the CTree boundary are a conservative useful
  bridge or just another small patch.
- Dalton: define the minimum validation harness before any compact search
  service or array-native CTree lane can be trusted.

Main-thread duties while they run:

- keep docs current;
- do not touch live Coach runs;
- avoid another tiny patch unless it directly enables the mock ceiling,
  array-native feasibility, or validation harness;
- keep the denominator language strict: profile-only roots/sec, train-profile
  steps/sec, and real Coach training speed are separate currencies.

## 2026-05-22 Current Wave Supersession

The mock search-service question above is answered enough for now:

```text
mock_search_service sim16:        11648.29 roots/sec
direct_ctree_gpu_latent sim16:     5303.97 roots/sec
```

That is meaningful headroom, but not a 10x proof. The current wave is now:

1. Keep `HybridCompactBatch` as the pre-scalar boundary of truth.
2. Consume that sidecar through real direct CTree arrays, RND-meter-shaped work,
   and replay/target-shaped work before scalar Python objects.
3. Validate sidecar semantics aggressively: binary masks, done roots,
   terminal/autoreset/final rows, `to_play`, active roots, row/player ids.
4. Only after those profile-only gates pass, run matched full-loop A/B rows.

Current closed proof:

```text
ap-RztU5jMKmKBpXuaDY3vZB0 passed the compact sidecar -> direct CTree remote
wiring smoke with persistent policy framebuffer, direct_gray64, no scalar
timestep materialization, compact contract telemetry, and zero illegal actions.
```
