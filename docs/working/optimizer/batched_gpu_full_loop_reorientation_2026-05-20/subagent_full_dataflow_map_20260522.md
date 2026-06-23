# Current CurvyTron Profile Dataflow Map - 2026-05-22

Status: docs-only optimizer subagent note. I did not launch Modal jobs, touch
live Coach training, checkpoints, evals, GIFs, tournament artifacts, or source
code.

## Scope

This map is about the currently profiled compact visual + MCTX loop around:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

The stock LightZero lanes matter only as a guardrail:

- Trusted learning controls use stock `lzero.entry.train_muzero`, especially
  `source_state_fixed_opponent`.
- The compact MCTX loop is an optimizer/profile loop. Its roots/sec numbers are
  not live Coach training speed and not learning proof.
- The old custom two-seat path is not trusted for learning claims unless its
  replay/target path passes parity gates.

## Current Profile Shape

The current mental model is:

```text
H100, B1024/P2, no-death compact loop,
native_actor_buffer=True,
actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack,
no root observation copy,
explicit resident-stack sync off,
exact delta pack,
closed-loop replay-index on for replay-valid rows.
```

At B1024/P2 there are `2048` search roots per closed-loop step.

## One Closed-Loop Step

Plain version:

```text
CPU joint action
-> CPU CurvyTron env step
-> CPU compact sidecars
-> CPU render-state/delta preparation
-> H2D renderer update
-> GPU latest frame
-> GPU resident 4-frame stack
-> CPU root sidecars and masks
-> MCTX search on GPU/JAX
-> small search outputs back to CPU
-> CPU replay-index rows
-> repeat
```

More detailed version:

| Stage | What is produced | Where it lives | Large or small | Main code/bucket |
| --- | --- | --- | --- | --- |
| 1. Joint action | `joint_action[B,2] int16` | CPU | Small, about `4 KiB` | `joint_action_build_sec` in `mctx_synthetic_benchmark.py` |
| 2. Env step | next CurvyTron physics state, rewards, dones, action masks | CPU NumPy/env state | Mechanics state can be large, but true mechanics time is currently small | inside `env_step_sec`; nested `actor_env_runtime_sec`, `actor_env_reward_sec`, `actor_env_post_runtime_bookkeeping_sec` |
| 3. Compact batch sidecars | reward, done, masks, row ids, players, active-root mask | CPU NumPy | Small to medium | `_make_compact_batch(...)`; `compact_batch_build_sec` |
| 4. Render state | visual trail/head/bonus state for latest frame | CPU borrowed state in current profile | Large if copied; borrowing avoids one large copy | nested `actor_render_state_write_sec`, `actor_render_state_write_visual_trail_sec` |
| 5. Persistent render preparation | compact render state and delta/compose arrays | CPU NumPy | Medium/large depending trail span | `renderer_production_to_compact_sec`, `renderer_persistent_delta_pack_sec` |
| 6. Renderer H2D | delta state and compose state | CPU -> GPU/JAX | Medium; repeated every step | `renderer_host_to_device_sec`; `_copy_state_to_device(...)` |
| 7. GPU draw/compose | latest `[B,2,1,64,64] uint8` frame | GPU/JAX | About `8 MiB` | `renderer_persistent_update_sec`, `renderer_device_render_sec` |
| 8. Resident stack | `[B,2,4,64,64] uint8` | GPU/JAX | About `32 MiB` | `resident_stack_update_sec`; explicit block is off in current best rows |
| 9. Compact root batch | root observation metadata, legal mask, row/player ids | CPU, observation is no-copy/view when enabled | Observation would be `32 MiB` if copied; sidecars are small | `root_build_sec`, `root_sidecar_sec`; `build_compact_root_batch_v1(...)` |
| 10. Search input | observation tensor and invalid mask | observation GPU-resident; mask CPU -> GPU | mask is tiny, about `6 KiB`; stack is large but resident | `h2d_sec` |
| 11. MCTX search | action, visit policy, search tree/root values | GPU/JAX during search | search tree size depends sims/depth; outputs are small | `search_sec`; `run_search(...)` / MCTX |
| 12. Search payload readback | selected actions, visit policy, root values | GPU -> CPU | action `~8 KiB`, visit policy `~24 KiB`, root values `~8 KiB` | `d2h_sec`, `root_value_extract_sec` |
| 13. Search validation | `CompactSearchResultV1` | CPU NumPy | Small | `search_result_validate_sec`; `validate_compact_search_result_v1(...)` |
| 14. Replay index | action, policy target, root value, reward/done, ids | CPU NumPy | Small; no full observation copy | `replay_index_sec`; `build_compact_replay_index_rows_v1_from_search_result(...)` |

## What Moves Across Host And Device

Large objects we should keep off the hot CPU/GPU boundary:

| Object | Shape | Approx size | Current status |
| --- | --- | ---: | --- |
| Latest frame | `[1024,2,1,64,64] uint8` | `8 MiB` | GPU-resident in current profile. |
| 4-frame stack | `[1024,2,4,64,64] uint8` | `32 MiB` | GPU-resident in current profile. |
| Root observation copy | `[2048,4,64,64] uint8` | `32 MiB` | Avoided by `copy_observation=False`. |
| Visual trail positions | `[1024,4096,2] float32` | `32 MiB` | Borrowed render state avoids parent copy in the current actor_count=1 profile. |
| Visual trail radius | `[1024,4096] float32` | `16 MiB` | Still read/packed by render prep. |
| Visual trail owner | `[1024,4096] int32` | `16 MiB` | Still read/packed by render prep. |

Small objects that cross the boundary:

| Object | Shape | Approx size | Why it moves |
| --- | --- | ---: | --- |
| Selected actions | `[2048] int32` | `8 KiB` | CPU env cannot step without actions. |
| Invalid mask | `[2048,3] bool` | `6 KiB` | Search needs current legal actions. |
| Visit policy | `[2048,3] float32` | `24 KiB` | Replay target needs it. |
| Root values | `[2048] float32` | `8 KiB` | Replay target/value bookkeeping needs it. |

The byte sizes are important: selected-action readback is semantically required
while the env is CPU-owned, but it is tiny. The bigger issue is repeated
observation/search-input ownership and synchronization, not action bytes.

## Where Synchronization Likely Happens

Hard semantic syncs:

- CPU env step must wait for selected actions from MCTX.
- Search must consume the frame/stack for the same post-step state as the legal
  mask and root sidecars.
- Replay rows must not become visible until action, visit policy, root value,
  reward, done, final flags, row ids, and player ids all match.

Current code sync or wait points:

- `loop_obs.block_until_ready()` and `loop_invalid.block_until_ready()` inside
  `h2d_sec`. In resident mode this can absorb waits from the renderer/stack
  chain, even though the stack was already on GPU.
- `loop_output.action_weights.block_until_ready()` in replay-valid search mode.
  Action-only/deferred diagnostic modes block only on `action`, so their search
  timers are not the same denominator.
- `np.asarray(loop_output.action)` and `np.asarray(loop_output.action_weights)`
  in `d2h_sec`.
- `_extract_mctx_root_values(...)` reads direct root node values. If it falls
  back to `search_tree.summary()`, that is a regression because summary was the
  old expensive materialization path.
- `_copy_state_to_device(...)` blocks by default during renderer H2D. The async
  profile flag can defer this wait, but measured gain was only about `1.05x`.
- `compact_visual_resident_device_stack.block_until_ready()` is explicitly off
  in the current best resident-stack profile shape.

## Bucket Names In Plain English

Top-level closed-loop buckets from `mctx_synthetic_benchmark.py`:

| Bucket | Meaning | Notes |
| --- | --- | --- |
| `root_build_sec` | Build `CompactRootBatchV1`. | With no-copy observation, this is mostly sidecars and checks. |
| `root_sidecar_sec` | Prepare masks, active indices, root ordering checks. | Small arrays, but can force host-visible work. |
| `h2d_sec` | Make search input device-ready. | In resident mode this mostly waits on resident stack readiness and transfers invalid mask. |
| `search_sec` | MCTX search wall. | Blocks on action weights in replay-valid mode. |
| `d2h_sec` | Read action and visit policy to CPU. | Root values are timed separately. |
| `root_value_extract_sec` | Read root values. | Fixed: direct root-node extraction is now small. |
| `search_result_validate_sec` | Validate compact search result. | CPU checks over small arrays. |
| `joint_action_build_sec` | Scatter selected actions into `[B,2]`. | Small. |
| `env_step_sec` | Advance env and prepare next observation. | Very inclusive; not pure game physics. |
| `replay_index_sec` | Build compact replay-index rows. | Measured small in current rows. |

Nested `env_step_sec` buckets:

| Nested bucket | Meaning |
| --- | --- |
| `actor_env_runtime_sec` | Actual CurvyTron physics step. |
| `actor_env_reward_sec` | Reward calculation. |
| `actor_env_public_info_sec` / `actor_env_public_prepare_sec` | Public info and returned batch packaging. |
| `actor_render_state_write_sec` | Copy/write render state into the profile surface. Borrowed state makes this near zero in the current profile. |
| `observation_sec` | Observation/stack update wrapper in the manager. |
| `renderer_production_to_compact_sec` | Convert production state into compact renderer state. |
| `renderer_persistent_delta_pack_sec` | Build the draw delta for the persistent renderer. |
| `renderer_host_to_device_sec` | Copy renderer delta/compose arrays to GPU. |
| `renderer_persistent_update_sec` | Update persistent GPU layer. |
| `renderer_device_render_sec` | Raw GPU draw/compose. This is already small. |
| `resident_stack_update_sec` | Append latest frame to resident stack. |
| `compact_batch_build_sec` | Build `HybridCompactBatch`. Fresh timer shows this is tiny. |
| `batched_stack_probe_wall_sec` | Probe/capture wall. Fresh timer shows this is tiny in the current loop. |

## Measured Evidence To Keep Straight

Root-value extraction fix:

```text
Before direct root-node extraction:
  root_value_extract ~0.26-0.31s on loop24 rows.

After direct root-node extraction:
  sim16 replay-valid loop24:
    roots/sec 54,977, total 0.894s,
    env_step 0.656s, search 0.157s,
    root_value_extract 0.019s, replay_index 0.010s.
  sim32 replay-valid loop24:
    roots/sec 38,122, total 1.289s,
    env_step 0.771s, search 0.418s,
    root_value_extract 0.024s, replay_index 0.012s.
```

Longer stability rows:

```text
loop96 sim16 replay on:
  roots/sec 50,617, total 3.884s,
  env_step 2.963s, search 0.621s,
  root_value_extract 0.069s, replay_index 0.038s.

loop96 sim16 replay off:
  roots/sec 53,579, total 3.669s,
  env_step 2.806s, search 0.613s,
  root_value_extract 0.061s.
```

Compact timer visibility row:

```text
loop48 sim16 after timer patch:
  roots/sec 54,358, total 1.808s,
  compact_batch_build_sec 0.0016s,
  batched_stack_probe_wall_sec 0.0006s.

Nested visible leaves:
  actor_env_runtime 0.2075s,
  actor_env_public_info 0.2576s,
  observation 0.7318s,
  renderer_production_to_compact 0.2251s,
  renderer_persistent_delta_pack 0.2910s,
  renderer_host_to_device 0.1794s,
  renderer_persistent_update 0.0146s,
  renderer_device_render 0.0084s,
  resident_stack_update 0.0347s.
```

Async H2D canary:

```text
loop96 sim16 baseline -> async H2D:
  50,358 -> 53,192 roots/sec, about 1.06x.

loop96 sim32 baseline -> async H2D:
  40,010 -> 41,902 roots/sec, about 1.05x.
```

Plain read: async H2D helps a little. It is not the big move.

## Stock LightZero Lane Relevance

Current stock facts:

- `source_state_fixed_opponent` can call stock `train_muzero` and strictly load
  a frozen checkpoint opponent. CPU and L4 canaries passed.
- Stock fixed/frozen opponent is a trusted plumbing/control lane, not true
  same-current-policy two-seat self-play.
- The old custom `--mode two-seat-selfplay` changed collector, replay rows,
  target building, and learner calls. Treat failed scaled runs there as evidence
  against that custom adapter, not against CurvyTron learning.
- Optimizer profile wins here should not be promoted into Coach training until
  they preserve replay/target contracts or are explicitly labeled as profile
  evidence.

Optimizer implication:

```text
The compact profile loop tells us where the fast array/native shape can go.
The stock lane tells us what semantic contracts must be preserved before a
training-facing optimization is trusted.
```

## Top 5 Amdahl Targets

### 1. Next-search-input handoff inside `env_step_sec`

Evidence:

- loop96 sim16 replay-valid row: `env_step_sec 2.963s / total 3.884s`.
- loop24 sim16 replay-valid row: `env_step_sec 0.656s / total 0.894s`.
- Fresh nested row shows the largest leaves are observation and renderer
  prep/H2D, not `compact_batch_build_sec`.

What to attack:

- production-to-compact conversion;
- delta pack;
- renderer H2D/update waits;
- resident stack/root ownership;
- public packaging around the next state.

### 2. MCTX search itself at higher simulation counts

Evidence:

- sim16 replay-valid loop24: `search_sec 0.157s / total 0.894s`.
- sim32 replay-valid loop24: `search_sec 0.418s / total 1.289s`.
- loop48 sim32 rows had search around `0.754-0.785s`.
- sim32+ search is already large enough that observation-only work cannot give
  a huge whole-loop win by itself.

What to attack:

- search batching and shape stability;
- avoid extra host-visible MCTX summary paths;
- keep replay-needed search payloads array-owned;
- investigate service-owned search/replay boundaries before changing training.

### 3. Renderer compact/delta ownership, not raw drawing

Evidence:

- Fresh nested row: raw `renderer_device_render_sec` was only `0.0084s`, while
  `renderer_production_to_compact_sec + renderer_persistent_delta_pack_sec +
  renderer_host_to_device_sec` was about `0.6955s` over the same loop.
- Earlier fast visual adapter already cut production-to-compact sharply, and
  borrowed render state gave large profile wins. That means ownership and
  packing are real, measurable targets.

What to attack:

- direct visual delta production from the actor/env state;
- compact renderer state owner updated in place;
- fewer copied trail arrays;
- smaller H2D delta payloads.

### 4. Host/device synchronization placement

Evidence:

- Async H2D only gave `~1.05-1.06x`, so sync placement is real but not enough
  alone.
- Resident-stack sync-off helps, but waits can move into `h2d_sec` or search.
- Current timers can label waits differently depending where the first
  `block_until_ready()` happens.

What to attack:

- split queue time from wait time;
- avoid host sync after renderer update unless search or validation truly needs
  it;
- keep selected-action sync because CPU env needs it, but avoid syncing large
  frames/stacks for logging or validation in the hot loop.

### 5. Stock LightZero object topology for trainer-facing speed

Evidence:

- Direct/output-fast stock profiles were only about `1.28-1.31x` faster in the
  full loop, even when compact/profile rows are much faster.
- Stock LightZero still has scalar env objects, dict observations, CTree root
  objects, Python/list-shaped search APIs, GameSegment/replay objects, learner
  batches, and optional RND CPU work.

What to attack:

- array-native collect/search/replay contracts in a proof lane;
- replay/target parity tests before promotion;
- keep stock `train_muzero` canaries as semantic controls.

## Do Not Spend Next Effort Here Unless New Data Changes

- Serial deferred payload flushing. It moved the cost; it did not remove it.
- Python-thread payload overlap. It caused contention in earlier canaries.
- Replay-index construction. Current replay-valid rows show it is small.
- `compact_batch_build_sec`. Fresh timer says it is tiny.
- Raw GPU draw. It is currently a few milliseconds in the compact rows.

## Bottom Line

The current compact loop is not blocked by selected-action bytes, replay-index
rows, or raw frame drawing. The active wall is the repeated construction and
handoff of the next search input, plus MCTX search as simulations increase.

The best next optimizer work should keep the same observation/search/replay
contract, but change ownership so the actor/env, renderer, resident stack, and
search input pass fewer large CPU objects around every step.
