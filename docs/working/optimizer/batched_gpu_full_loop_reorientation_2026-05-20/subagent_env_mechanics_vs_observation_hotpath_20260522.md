# Env Mechanics vs Observation Hotpath, 2026-05-22

Scope: read-only profiler/critic pass over the current compact closed-loop
profile notes and timing code. No source files were changed. No live runs,
Modal jobs, checkpoints, evals, GIFs, or tournament artifacts were touched.

## Short Answer

The current Amdahl wall is **observation/search-input handoff**, not game
mechanics and not GPU pixel drawing by itself.

Plain version:

```text
CurvyTron physics is small.
GPU drawing is small.
The expensive part is moving fresh env state into the shape the next search
call consumes.
```

That handoff includes actor render-state writes, production-state to compact
render-state packing, observation stack update, optional resident-stack update,
root-batch construction, and input readiness for MCTX.

## What The Timers Mean

`env_step_sec` in the compact MCTX closed loop is a broad bucket. It starts just
before:

```text
compact_visual_manager_for_replay.step(loop_joint_action)
```

and ends after the manager step returns and, for resident-stack rows, after the
resident compact visual stack is updated.

So `env_step_sec` means:

```text
apply selected actions
-> advance env actors
-> write compact sidecars
-> write render state
-> refresh observation/stack
-> prepare the next compact batch
```

It is not the same thing as game physics.

`actor_env_runtime_sec` is the closest current leaf to real game mechanics. It
is the actual vector env runtime advance. The plain breakdown also groups
`actor_env_reward_sec` and `actor_env_post_runtime_bookkeeping_sec` with strict
mechanics.

`observation_sec` is the manager's observation-refresh wall time. In
`source_state_hybrid_observation_profile.py`, it wraps:

- shifting the host `[B,P,4,64,64]` stack when host stack update is enabled;
- calling the renderer;
- writing the latest rendered frame into the stack;
- copying terminal rows into `final_observation`;
- doing autoreset observation reset/render work when rows ended.

`observation_sec` is not only the GPU draw kernel. It includes stack ownership
and terminal/reset observation handling around the renderer. Renderer telemetry
such as `renderer_device_render_sec`, `renderer_production_to_compact_sec`, and
`renderer_device_to_host_sec` is nested inside or adjacent to this bucket, so do
not sum every renderer label as if the profile were fully exclusive.

`final_observation`, or "end observation", means the terminal observation saved
for rows that ended before autoreset changes them. It is a replay/target
sidecar. It is not the normal live observation for every row, and it is not the
main wall unless many rows are ending.

## Current Fraction Read

The cleanest current rows are the fresh H100 mechanics-vs-observation grid:

```text
B1024/P2, body4096, hidden64, depth16, loop24,
native actor buffer, root no-copy, replay rows on.
```

The table below shows two views:

- `game/env` and `obs/env` are fractions inside `env_step_sec`.
- `game/total` and `obs/total` multiply those by the top-level env fraction.

| row | roots/sec | env/total | search/total | game/env | obs/env | game/total | obs/total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host stack, sim16 | `23,109` | `68.1%` | `7.5%` | `9.6%` | `80.0%` | `6.5%` | `54.5%` |
| resident stack, sim16 | `30,297` | `68.3%` | `9.6%` | `8.4%` | `79.2%` | `5.7%` | `54.1%` |
| host stack, sim32 | `19,485` | `63.0%` | `15.2%` | `8.8%` | `79.8%` | `5.5%` | `50.3%` |
| resident stack, sim32 | `26,805` | `59.8%` | `19.7%` | `10.8%` | `75.8%` | `6.5%` | `45.3%` |

Read:

```text
In refresh-on rows, strict game mechanics are about 5.5-6.5% of total wall.
Observation/search-input handoff is about 45-55% of total wall.
```

That is the Amdahl answer.

The refresh-off ceiling says the same thing from the other direction:

| row | roots/sec | env/total | search/total | root-build/total | read |
| --- | ---: | ---: | ---: | ---: | --- |
| sim16 refresh on | `20,686` | `71.2%` | `6.8%` | `3.9%` | current matched row before no-copy/fast-visual |
| sim16 refresh off | `48,608` | `21.9%` | `15.2%` | `24.4%` | `2.35x` ceiling |
| sim32 refresh on | `17,855` | `66.7%` | `14.2%` | `3.2%` | current matched sim32 row |
| sim32 refresh off | `32,133` | `18.5%` | `25.7%` | `23.7%` | `1.80x` ceiling |

Deleting observation refresh is not a valid training lane, but it prices the
wall. Once that wall is removed, root build, H2D, and search become visible.
That is exactly what an Amdahl ceiling should do.

Other supporting rows:

- The timing-split matched row reported actual env runtime around `0.089s`,
  while actor render-state write was `0.368s` and observation/stack was
  `0.988s`.
- The latest resident no-copy rows show runtime leaves around `0.055-0.089s`,
  while render-state write plus observation are several times larger.
- GPU draw in the fresh grid was only about `0.005-0.007s` over the measured
  loop. So "rendering" as pixel drawing is not the wall.
- Replay-index rows are about `0.3%` in the matched replay-on grid. They are
  not the wall.

## Critic Read

Do not call the wall "game mechanics." The real mechanics leaf is small enough
that even a heroic mechanics rewrite would have a weak whole-loop ceiling in
the current refresh-on rows.

Do not call the wall "rendering" unless "rendering" means the whole
render-state/observation ownership path. The GPU framebuffer draw is already
tiny. The cost is the state handoff around it.

The useful name is:

```text
observation/search-input handoff
```

That name covers the thing that is actually hot: writing and packing render
state, refreshing the latest stack, keeping host/device copies coherent, and
building the next root input.

## Next Aggressive Moves

### 1. Compact Render-State Ownership

Move the persistent renderer toward owning compact render state directly.
Actors should not keep copying full render-state rows through parent buffers
every hot step if the renderer only needs visual trail, player/head, avatar, and
bonus state.

Targeted buckets:

- `actor_render_state_write_sec`
- `renderer_production_to_compact_sec`
- `renderer_persistent_delta_pack_sec`

Falsifier:

```text
Matched H100 B1024/P2 sim16+sim32 closed-loop rows,
native actor buffer, root no-copy, replay rows on.
```

Keep only if total roots/sec moves by at least about `1.2x` and the above
buckets collapse in the same denominator. Kill or demote if timers improve but
total roots/sec stays flat, because that means another part of the handoff
immediately replaced it.

### 2. Resident Stack As The Hot Search Input

Make the resident compact visual stack the hot source of truth for search. Host
stack materialization should become a sampled validation edge, not mandatory
work in every loop step.

Targeted buckets:

- `observation_sec`
- `stack_shift_sec`
- `stack_latest_update_sec`
- `renderer_device_to_host_sec`
- `h2d_sec`

Falsifier:

```text
Same row shape as above, compare:
host stack no-copy
resident stack no-copy
resident stack with host stack bypassed
refresh-off ceiling
```

Keep if the host-bypassed resident row cuts observation handoff by roughly half
and gives a real whole-loop gain, not just lower H2D. Kill if it recreates the
old "second stack" problem where resident work is added while the host stack is
still maintained.

### 3. Root/Search Input View Instead Of Rebuild

After observation ownership improves, stop rebuilding large root inputs. The
root batch should be a view or device descriptor over the compact/resident
state, with legal masks and row/player sidecars kept compact.

Targeted buckets:

- `root_build_sec`
- `h2d_sec`
- `d2h_sec` for search outputs
- handoff glue around `CompactRootBatchV1`

Falsifier:

```text
Run refresh-on and refresh-off no-copy rows with root batch view/device-handle
mode against the current no-copy root path.
```

Keep if refresh-off rows move materially beyond the current `48.6k-63.6k`
sim16 ceiling band and refresh-on rows also improve. Kill if root-build/H2D
improves only in refresh-off mode, because then the real wall is still earlier
observation ownership.

## Bottom Line

The compact closed-loop wall is not the rules of CurvyTron. It is not the GPU
drawing pixels. It is the handoff that turns fresh game state into the next
search input.

Spend the next serious patches on compact state ownership and resident
observation/search-input ownership. Keep search and replay-index work as
guardrails until those rows say the wall moved back there.
