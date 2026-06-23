# Next State Ownership Patch

Date: 2026-05-22

Scope: design only. I read the current optimizer docs and relevant source. I did
not edit training source, checkpoints, jobs, or live Coach runs.

Current surface checked against
`subagent_latest_renderer_surface_audit_20260522.md`: this proposal is for the
profile-only persistent GPU compact lane, not the trusted production trainer
surface. The relevant fast surface is:

```text
jax_gpu_persistent_policy_framebuffer_profile + direct_gray64
browser-line trail semantics + simple_symbols
[4,64,64], player perspective
```

Do not mix this up with old `body_circles_fast` notes. That mode still exists
as an old/generic render ablation, but it is not the current source-state
policy surface.

## Plain Read

After the resident-stack, no-copy root batch, and fast visual rows, the next
measured wall is narrower and more concrete:

```text
selected action
-> VectorMultiplayerEnv.step
-> actor render-state write into parent buffers
-> persistent renderer
-> host or resident stack handoff
-> root batch / search
```

The newest matched H100 compact rows say the same thing more cleanly:

| source | sims | active roots/sec | total sec | env frac | search frac | mechanics inside env | observation/state handoff inside env |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host stack | 16 | `23.4k` | `1.400s` | `67.9%` | `7.4%` | `9.6%` | `83.5%` |
| resident GPU stack | 16 | `29.6k` | `1.109s` | `66.8%` | `9.6%` | `11.3%` | `80.5%` |
| host stack | 32 | `23.6k` | `1.388s` | `58.3%` | `18.5%` | `10.2%` | `82.1%` |
| resident GPU stack | 32 | `30.5k` | `1.076s` | `56.8%` | `23.4%` | `9.4%` | `80.8%` |

Plain read:

- Resident stack now wins in the matched profile: about `1.26x` at sim16 and
  `1.29x` at sim32.
- The top-level `env_step_sec` bucket is still the wall, but it is mostly not
  game mechanics.
- Actual mechanics are only about `9-11%` of that env bucket in these rows.
- The expensive part is still state/observation handoff around the renderer:
  actor render-state write, production-to-compact, renderer H2D/update/compose,
  and stack handoff.
- MCTS/search is visible, especially at sim32, but it is not yet the dominant
  Amdahl wall in this compact profile.

The next aggressive but clean patch should therefore target ownership, not
search:

```text
single actor owns current env/render state
-> renderer borrows that state directly
-> MCTX consumes resident stack or host stack
-> host scalar/timestep objects remain disabled in the hot profile loop
```

## Proposed Minimal Patch

Add a profile-only `single_actor_borrowed_render_state` mode for the closed
compact visual harness.

Minimal behavior:

1. In `HybridBatchedObservationProfileManager.step`, when all of these are true:
   `native_actor_buffer=True`, `actor_count == 1`, persistent GPU renderer,
   refresh-on profile, and the row has no terminal/autoreset event, do not
   allocate/write parent `_native_render_state` buffers.
2. Let the actor still write compact scalar sidecars (`reward`, `done`,
   `action_mask`, `joint_action`) exactly as today.
3. After the actor step, pass `self.actors[0].env.state` directly to
   `_update_observation(...)` as a borrowed render-state mapping.
4. If any terminal/autoreset row appears, reject the mode with a clear
   profile-only error until a pre-reset snapshot protocol is added. Do not
   silently render borrowed post-reset state as the terminal frame.
5. Keep `compact_root_copy_observation=False` for the hot closed-loop rows.
6. Retest both host-stack and `resident_gpu` stack modes. In resident mode,
   also run one variant that removes the explicit post-render
   `block_until_ready()` and lets the next search consume/synchronize the
   stack. Read only total closed-loop wall, because this may move wait time
   between buckets.

This is intentionally a single-actor patch because the current in-process grid
already showed `actor_count=1` fastest. Sharded/parallel actor ownership is a
different architecture step.

## Files And Functions

Touch only profile-only harness code:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
  - `HybridObservationProfileConfig`: add an opt-in flag such as
    `borrow_single_actor_render_state: bool = False`.
  - `HybridBatchedObservationProfileManager.step`: guard the borrowed path near
    the current native render-state allocation/copy.
  - `HybridBatchedObservationProfileManager._update_observation`: no semantic
    change should be needed; it already accepts a `render_state` mapping.
  - `HybridBatchedObservationProfileManager.contract` and timing fields: report
    the active render-state mode, fallback count, and borrowed/fallback rows.

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
  - Add a CLI/config flag for the borrowed mode.
  - Include `render_state_mode`, `render_state_borrowed_steps`,
    `render_state_copy_fallback_steps`, and resident-stack sync policy in the
    compact JSON summary.
  - Run the mode only for `curvytron_hybrid_compact_visual_sample`.

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  - Prefer no change. `_PersistentJaxPolicyFramebufferRenderer.render` already
    copies the mapping shallowly and the fast visual adapter reads from it.
  - Only add renderer telemetry if tests need proof that borrowed state reached
    the fast visual adapter.

- Tests:
  - `tests/test_source_state_hybrid_observation_profile.py`
  - `tests/test_mctx_synthetic_benchmark_legality.py`

## Why This Is Semantics-Preserving

For non-terminal profile rows, the copied path and borrowed path should expose
the same post-step state to the renderer. The patch removes only this copy:

```text
actor env.state -> parent native_render_state buffers
```

It must not change:

- selected actions or legal masks;
- row/player order;
- `to_play=-1`;
- reward/done sidecars;
- final-observation-before-reset ordering;
- stack FIFO order;
- search input shape `[B*2,4,64,64]`;
- compact replay-index fields.

Terminal/autoreset is the sharp edge. If the actor mutates `env.state` during
autoreset before the renderer captures the final observation, borrowed state is
wrong. That is why the first patch must be no-terminal and fail-closed on any
terminal/autoreset row.

## Expected Speedup And Headroom

Use the current best refresh-on row as the practical baseline:

```text
resident GPU stack + no-copy root, sim16: about 29.6k active roots/sec
resident GPU stack + no-copy root, sim32: about 30.5k active roots/sec
older no-refresh + no-copy ceiling: about 63.6k active roots/sec
```

Expected near-term win:

- Borrowed single-actor render state on top of resident stack:
  plausible `1.10x-1.35x` if `actor_render_state_write_sec` collapses and
  total wall follows.
- Borrowed state plus a deferred resident-stack sync retest:
  plausible `1.20x-1.50x` only if the current renderer/stack waits are partly
  self-inflicted synchronization, not real work.
- Anything near `2x` would be excellent and should be treated as a profile-only
  architecture signal, not trainer advice.

Hard ceiling in this denominator is the no-refresh/no-observation row from the
older fast visual/no-copy batch, roughly `2.1x-2.7x` over the current refresh-on
rows depending which matched baseline is used. The borrowed-state patch cannot
produce a 5-10x result by itself; after this, root/build, H2D/mask transfer,
search, and replay/learner adapters become visible again.

## Validation Tests

Local first:

- Borrowed versus copied render state, same seed/actions, no terminal:
  compare `step.observation`, renderer `last_output_device`, compact sidecars,
  action masks, row/player ids, and `actor_render_state_write_sec == 0`.
- Terminal/autoreset hostile row:
  prove borrowed mode raises with a clear profile-only error; do not allow
  silent post-reset rendering and do not claim copy fallback until a pre-reset
  snapshot protocol exists.
- Resident stack parity:
  compare host stack FIFO against resident `last_output_device` FIFO for
  several steps with the same actions.
- Renderer immutability:
  prove rendering the borrowed mapping does not mutate actor state arrays.
- Summary/contract guards:
  assert `render_state_mode`, fallback counts, host-stack update mode,
  `compact_root_copy_observation`, and resident-stack source are present.

Then Modal profile-only rows:

- H100 B1024/P2/sim16/h64/v16/loop16, host stack, no-copy root:
  copied baseline versus borrowed mode.
- Same row with `resident_gpu` stack, no-copy root:
  copied baseline versus borrowed mode.
- Same resident row with explicit stack-update sync deferred:
  total wall only, not bucket-only claims.
- Repeat sim32 once if sim16 clears the gate.
- One small terminal/autoreset row with borrowed mode enabled and intentionally
  expected to fail closed. A separate copied baseline should still pass.

## Kill Criteria

Kill or revert the patch if any of these happen:

- Any copied-versus-borrowed observation, resident stack, row/player, legal mask,
  reward/done, or replay-index parity check fails.
- Terminal/autoreset cannot fail closed cleanly, or someone tries to make it
  silently fall back while still claiming copied-versus-borrowed timing.
- `actor_render_state_write_sec` does not collapse below noise on no-terminal
  rows.
- Matched H100 closed-loop throughput improves by less than `10%` over the
  fast-visual/no-copy baseline after two comparable runs.
- Resident-stack wall time merely moves into `h2d_sec`, `search_sec`, or
  residual without improving total closed-loop roots/sec.
- The patch requires changing stock LightZero trainer defaults, live training
  launchers, replay semantics, or MuZero search outputs.

## Recommendation

Implement only the single-actor borrowed render-state canary first. It is small,
profile-only, and directly prices the current `actor_render_state_write` bucket.
If it passes, retest resident stack with root-copy already gone. If that still
does not beat the host-stack row, stop calling resident observation the next
speed path and move to a larger compact state owner that keeps render state,
latest frames, and replay/search sidecars in one explicit ownership contract.
