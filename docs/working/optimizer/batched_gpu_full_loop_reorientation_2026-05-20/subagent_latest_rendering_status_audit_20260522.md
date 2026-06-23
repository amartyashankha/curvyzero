# Latest Rendering / Observation Status Audit - 2026-05-22

Scope: latest rendering and observation work only. I read the current optimizer docs, the three requested source modules, and recent tests/docs mentioning `browser_lines`, `simple_symbols`, persistent GPU, resident stack, `direct64`/`direct_gray64`, and `body_circles_fast`. I did not edit source and did not touch live runs.

## Plain Answer

The trusted policy surface is still `browser_lines + simple_symbols` through the CPU oracle contract: player-perspective 4-frame grayscale stacks shaped `[4, 64, 64]`, source-state trail geometry, simple symbolic bonuses, heads drawn last, BT.601 grayscale, and 64x64 area-average downsample. This is the production/trainer-safe surface.

The latest fast GPU work is a profile/canary surface, not the trusted default. The main active GPU path is the persistent JAX policy framebuffer backend, `jax_gpu_persistent_policy_framebuffer_profile`, restricted to canaries and `direct_gray64`. It keeps policy-space trail layers on GPU and updates them incrementally, then composes bonuses and heads on GPU. It is fast and increasingly relevant for the MCTX/root pipeline, but it is still documented and guarded as profile-only evidence.

The remaining wall is not raw GPU drawing. Current docs repeatedly show raw GPU draw in the low milliseconds while host observation/search-input handoff, compact/render-state ownership, D2H/H2D, stack update, scalar materialization, and public LightZero/CTree object boundaries dominate.

## Trusted Policy Surface Now

- Contract source: `src/curvyzero/env/observation_surface_contract.py`.
- Policy label: `browser_lines+simple_symbols`.
- Trail mode: `browser_lines`.
- Bonus mode: `simple_symbols`.
- Stack contract: 4 frames, 64x64, raw `uint8` at the env boundary, float32/normalized at model input.
- Trusted backend: `cpu_oracle`. The scalar `jax_gpu` backend is experimental and not the default reliable backend.
- Draw/order contract: background, trails, bonuses, heads; heads last; source-state browser-line segments; simple symbols; BT.601 grayscale; area-average downsample to 64x64.
- Trainer/env tests reject stale fast modes such as `body_circles_fast` and `fast_gray64_direct` as policy/trainer surfaces. `body_circles_fast` remains only historical/generic renderer behavior, not the current policy recommendation.

Important nuance: this trusted policy surface is not claimed to be exact browser-pixel parity. It is the current policy observation contract.

## What Is On GPU

Current GPU rendering work has two related but different lanes.

1. Full-redraw JAX benchmark/profile renderers:

- Implemented in `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`.
- Supports `block_704_gray64` and `direct_gray64` render surfaces.
- Supports two-view fused rendering for both players.
- Supports `browser_lines` trail semantics and `simple_symbols` bonus rendering.
- `block_704_gray64` approximates the high-resolution browser-style block/downsample path without materializing the full source RGB canvas.
- `direct_gray64` samples directly in policy 64x64 space. It is an economics/performance probe and now the basis for persistent profile canaries, not the trusted browser/downsample policy surface.

2. Persistent GPU policy framebuffer profile:

- Implemented in `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
- Backend name: `jax_gpu_persistent_policy_framebuffer_profile`.
- Allowed only behind profile/canary gates such as surface facade, profile env manager, and hybrid observation canaries.
- Requires `renderer_backed_profile` and `render_surface="direct_gray64"`.
- Keeps a device-resident luma trail layer shaped roughly `[B, 2, 64, 64]`.
- JIT update draws only new trail delta segments into the resident device layer.
- JIT compose overlays current bonuses and heads and returns the current two-view frame batch.
- Stores the latest output device-side; `device_only=True` can skip default D2H for the profile path.
- Tracks cursor/owner/avatar/reset bookkeeping in the renderer object, with compact/delta packing still prepared on the host.

Hybrid observation profile support in `src/curvyzero/training/source_state_hybrid_observation_profile.py` can use this persistent renderer with `update_host_observation_stack=False`. In that mode, host stack update is skipped and the latest device output is used as the resident profile evidence path, but stock LightZero training is still not integrated.

## What Still Happens On CPU / Host

- Actual multiplayer env stepping and game mechanics still happen on CPU.
- Actor loops, host action/legal-mask handling, final/autoreset bookkeeping, and replay/collector-facing objects are still host-side.
- Production source-state to compact/render-state conversion is still host work, though recently optimized.
- Persistent delta-state construction is still host work: cursor comparisons, reset detection, owner continuity, and delta segment packing happen before H2D.
- H2D still copies compact/delta/compose state unless the data is already resident for a specific profile probe.
- D2H still happens for default renderer-backed observation paths unless `device_only=True`.
- Host observation stack shift/latest-frame update still happens when `update_host_observation_stack=True`.
- Scalar timestep/pickle/RND/materialization paths are still host/object heavy.
- Public LightZero collect/search/replay boundaries still introduce host/object/list/dict costs outside the renderer itself.

So the system has GPU rendering, and now some GPU-resident observation/profile pieces, but not a fully GPU-resident training loop.

## What Was Optimized Already

- CPU dirty-cache/source-state observation was established as the reliable policy baseline.
- JAX GPU full-redraw benchmark/profile paths were added for `browser_lines` plus `simple_symbols`.
- `direct_gray64` made the policy-space GPU draw much cheaper than the 704-block profile surface.
- Persistent GPU framebuffer avoided full redraws by retaining trail layers and drawing deltas.
- Render-state filtering now copies only renderer-required visual keys for persistent mode.
- Fast visual compact-state adapter cut production-to-compact from roughly `0.37-0.52s` to `0.054-0.057s` in the latest optimizer notes.
- No-copy/root-batch work reduced host rebuild overhead around the observation boundary.
- Resident stack profile retests became positive after no-copy and fast visual compact work.
- Tests now cover many fidelity edges: same-owner interleaved line connection, visual trail masking past cursor, avatar color preservation, all 12 simple-symbol identities, bonus-over-trail, heads-over-bonus, and rejection of stale body-circle policy modes.

## Latest Measured Timings

The most relevant timing read is that rendering has been reduced enough that the wall moved.

- Early H100 B64/256-step surface canary:
  - CPU dirty-cache surface: about `0.237s`.
  - GPU block renderer-backed surface: about `0.144s`.
  - GPU direct surface: about `0.0339s`.
  - Direct device render: about `0.00973s`.

- B512 direct surface canary:
  - Surface step about `0.123s`.
  - Device render about `0.014s`.
  - Payload pickle around `0.020s`.
  - Payload size around `67.1MB`.

- Persistent renderer evidence:
  - First H100 rows around B512/100 improved total from about `81ms` to `45ms`, renderer from `39ms` to `10ms`.
  - B512/500 later showed render about `16ms`, while env/stack were each about `39ms`.
  - Synthetic persistent framebuffer tests reported large speedups, including no-readback cases, but those are profile evidence and not production parity.

- Compact visual MCTX/root evidence:
  - CPU-oracle visual root setup B256: host setup `16.890s`, last render `3.909s`, fresh-boundary roots/s `33,753`.
  - Persistent GPU B256: host setup `2.860s`, last render `0.014s`, fresh-boundary roots/s `70,330`.
  - That is about a `2.1x` fresh-boundary improvement in the cited compact visual gate.

- Latest optimizer board summary:
  - Fast visual compact adapter: production-to-compact about `0.054-0.057s`.
  - No-copy root batch refreshed-on reached about `26.6k` roots/s, up from prior `20.7k`.
  - Resident stack retest:
    - sim16 host no-copy `26.6k` to resident `31.6k`.
    - sim32 host no-copy `21.2k` to resident `28.9k`.
    - loop24 repeat sim16 host `23.1k` to resident `30.3k`.
    - loop24 repeat sim32 host `19.5k` to resident `26.8k`.
    - refresh-off ceiling about `57.9k`.
  - Grouped stdout says actual game mechanics are only about `8-11%` of `env_step_sec`; observation/search-input handoff is about `76-80%`; raw GPU draw is about `5-7ms`.

- Full-loop context:
  - Recent stock-vs-direct full-loop rows showed roughly `433.17` steps/s stock vs `566.19` direct output-fast without RND, about `1.31x`.
  - RND rows were about `351.02` stock vs `448.52` direct, about `1.28x`.
  - Older real-observation vs zero-observation comparisons showed only about `1.25x`, reinforcing that raw rendering is no longer the sole wall.

## What Might Have Been Dropped Or Gone Stale

- `body_circles_fast` should be treated as dropped/stale for policy/trainer work. Current tests reject it for the trainer/env policy surface.
- `fast_gray64_direct` naming is stale. The active canary language is `direct_gray64`, and persistent GPU requires that exact direct surface.
- Scalar `policy_observation_backend="jax_gpu"` is not the new persistent batched GPU profile. It remains experimental and should not be confused with the persistent framebuffer backend.
- `block_704_gray64` is still useful benchmark/profile evidence, but it is no longer the fast-lane surface after `direct_gray64` and persistent framebuffer work.
- Browser sprite parity is intentionally out of scope for `simple_symbols`.
- Full browser RGB canvas parity and browser-pixel exactness are not claimed by the GPU direct/persistent surface.
- `resident_torch_reuse` timing is an upper-bound/stale-input probe unless paired with correctness checks that prove the input is fresh.
- Any phrasing that says "GPU renderer fixed training throughput" is too strong. The current status is: renderer is much faster, but profile-only GPU/resident paths still need integration and correctness gates before promotion.

## Top 3 Rendering / Observation Risks To Verify Next

1. Fidelity drift between persistent `direct_gray64` and the trusted CPU oracle.

   Verify browser-line semantics, same-owner continuity, `break_before`, radius changes, overlap priority, avatar color perspective, bonus/head ordering, and all simple-symbol identities under long rollouts. Existing divergence smokes are useful but not exact enough to promote the path; earlier docs mention nonzero mismatch rates under no-death tests.

2. Resident/device stack correctness around time and resets.

   Verify row/player order, 4-frame FIFO order, `final_observation` before autoreset, partial reset requests, terminal rows, skipped host stack updates, and stale device-output hazards. The device-only path should have checksum/parity probes that cannot pass by reading an old host stack.

3. Host-boundary timing and hidden fallback attribution.

   Keep splitting production-to-compact, actor render-state writes, persistent delta pack, H2D, persistent update, device render, D2H, host stack shift/latest, root-build, scalar materialization, and replay/search object assembly. The danger is declaring a GPU win while the run silently falls back to CPU oracle, uses stale resident input, or moves time into a broader `env_step_sec` bucket.

## Sources Checked

- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/task_board.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/gpu_render_next_phase.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/persistent_framebuffer_next_gate_20260521.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/persistent_gpu_renderer_implementation_checklist_20260521.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/gpu_host_overhead_world_model_20260521.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/current_hot_path_bottleneck_map_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_latest_renderer_surface_audit_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_device_resident_observation_boundary_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_mechanics_vs_observation_audit_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_mechanics_vs_observation_amdahl_20260522.md`
- `src/curvyzero/env/observation_surface_contract.py`
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`
- `tests/test_source_state_gpu_render_benchmark_cpu.py`
- `tests/test_source_state_batched_observation_boundary_profile.py`
- `tests/test_source_state_hybrid_observation_profile.py`
- `tests/test_multiplayer_source_state_trainer_surface.py`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`
- `tests/test_vector_visual_observation.py`
