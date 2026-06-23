# Direct Visual Delta Canary Critique - 2026-05-22

Verdict: feasible as a profile-only persistent `direct_gray64` canary, but it is
not a trainer-ready renderer rewrite. The first slice should bypass compact
trail materialization inside the persistent framebuffer renderer only, while
keeping the existing compact path as the oracle/fallback.

## Exact Functions To Change

Primary file: `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.

- `_validate_boundary_config(...)`: add an opt-in flag, e.g.
  `persistent_direct_visual_delta_profile`; require persistent GPU backend,
  `render_surface == direct_gray64`, and an existing canary path.
- `main(...)`: expose the flag in the Modal local entrypoint config.
- `_PersistentJaxPolicyFramebufferRenderer.__init__(...)`: store the flag.
- `_PersistentJaxPolicyFramebufferRenderer.render(...)`: when the flag is on
  and `request.state` has `visual_trail_pos`/`visual_trail_active`, skip
  `_persistent_compact_state_from_production(...)`,
  `_persistent_delta_state(...)`, and `_persistent_compose_state(...)`; instead
  call a new direct helper that returns `delta_state`, `compose_state`,
  `next_cursor`, `delta_stats`, and `avatar_color`. Keep the current path as
  fallback.
- Add `_persistent_visual_delta_and_compose_from_production(...)` or split into
  `_persistent_delta_state_from_visual_production(...)` and
  `_persistent_compose_state_from_production(...)`. It should read directly from
  `visual_trail_pos`, `visual_trail_radius`, `visual_trail_owner`,
  `visual_trail_active`, `visual_trail_break_before`,
  `visual_trail_write_cursor`, `pos`, `radius`, `alive`, `present`,
  `avatar_color`, and `bonus_*`.
- Reuse `_persistent_live_trail_slots_from_production(...)` for cursor/capacity
  clipping. Leave `_persistent_compact_state_from_production(...)`,
  `_persistent_visual_compact_state_from_production_fast(...)`,
  `_persistent_delta_state(...)`, and `_persistent_compose_state(...)` intact as
  reference/fallback.
- Add telemetry, preferably without widening downstream code first:
  `persistent_direct_visual_delta_profile`, `direct_visual_payload_pack_sec`,
  and split/reuse `production_to_compact_sec` and `persistent_delta_pack_sec`
  so existing summaries still price the canary.

Reference file: `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`.

- Do not change it for the first canary. Its
  `_production_to_benchmark_source_state(...)` and
  `_pack_compact_trails_in_owner_draw_order(...)` are the compact reference
  path. Rewriting `_make_jax_render_fn(...)` or `_make_jax_two_view_render_fn(...)`
  would expand scope into the non-persistent benchmark renderer.
- Optional later: add an isolated timing row in
  `_source_state_reference_and_setup_timings_for_benchmark(...)` only after the
  persistent helper has local parity.

## Likely Speed Upside

Current best profile denominator is roughly B1024/P2 loop24 borrowed render
state, resident GPU stack, root-copy off, explicit resident sync off:
about `50.4k` roots/sec at sim16 and `43.3k` at sim32. Raw GPU draw is already
only about `4-6ms`; async H2D recently gave only `1.05-1.06x`.

This canary attacks the still-visible renderer/search-input handoff. The latest
sim16 notes price production-to-compact around `0.078s`, delta pack around
`0.104s`, and renderer H2D around `0.067s` over loop24. A direct helper will not
delete all of that, because it still must build small delta/compose payloads and
move them to device.

Expected upside:

- sim16: likely `1.05x-1.12x`; stretch `1.15x-1.18x` if delta/H2D payloads
  shrink materially.
- sim32: likely `1.00x-1.10x`; search is already about a third of wall and prior
  vectorized delta work regressed a sim32 A/B.
- hard ceiling: about `1.2x` on the current sim16 refresh-on row, because the
  no-refresh ceiling is only about `61.9k` roots/sec and public packaging,
  CPU mechanics, stack/root ownership, and search remain.

## Correctness Risks

- Cursor semantics: active slots past `visual_trail_write_cursor` must stay
  masked, cursor regressions must reset only those rows, and capacity must be
  clipped to live visual/body slots.
- Segment continuity: `previous_owner_pos`, `previous_owner_valid`,
  `visual_trail_break_before`, inactive slots, invalid owners, and interleaved
  owners must match `_persistent_delta_state(...)` exactly.
- Compose semantics: `head_alive` must be `alive & present`; non-identity
  `avatar_color`, radius dtype, `bonus_*`, and missing bonus/present defaults
  must match the compact adapter.
- Async H2D aliasing: do not pass mutable views of `env.state` into deferred
  device transfers if the env can mutate them before DMA finishes. Prefer owning
  small payload arrays.
- JIT churn: varying `max_delta` can grow `_update_fn_cache`; a faster pack that
  changes shapes every step may lose to compile/cache overhead.
- Terminal/autoreset: borrowed-state terminal rows already need a snapshot
  protocol. The direct path must fail closed rather than render post-reset state
  as final observation.
- Scope drift: persistent `direct_gray64` is policy-space profile rendering, not
  browser RGB/pixel parity.

## Smallest Validation Tests

Local first:

- Add helper parity tests in
  `tests/test_source_state_batched_observation_boundary_profile.py`: direct
  helper output equals
  `_persistent_compact_state_from_production(...)` +
  `_persistent_delta_state(...)` + `_persistent_compose_state(...)` for cold
  start, incremental step, row-selective cursor regression, inactive/invalid
  slots, `break_before`, radius changes, avatar-color remap, absent `present`,
  and bonus rows.
- Add one renderer unit test that toggles the flag, verifies the direct path is
  used for visual state, and verifies fallback for non-visual/body state.
- Keep existing benchmark reference tests:
  `tests/test_source_state_gpu_render_benchmark_cpu.py -k "production_to_benchmark or owner_ordered or adversarial"`.

Then one H100 A/B:

- Shape: B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay,
  `actor_count=1`, borrowed render state, resident GPU stack, root-copy off,
  explicit resident sync off.
- Run sim16 and sim32 with flag off/on. Require exact frame/checksum parity
  against current persistent path for the first verified steps, then compare
  total roots/sec and exclusive renderer leaves.

## Kill Criteria

- Any local helper parity mismatch or GPU checksum/frame mismatch.
- Any terminal/autoreset case cannot be guarded with a clear profile-only error.
- Total roots/sec improves less than `5%` at sim16, or sim32 regresses more than
  `5%`, on the matched denominator.
- Combined `renderer_production_to_compact_sec +
  renderer_persistent_delta_pack_sec + renderer_host_to_device_sec` does not
  fall by at least `25-30%`.
- `_update_fn_cache_size` grows steadily, compile spikes dominate, or max-delta
  shape churn explains the apparent win.
- Async-device mode shows stale/torn frames, old stack checksums, or dependence
  on mutating borrowed arrays after render submission.
