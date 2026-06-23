# GPU Observation Validation Plan - 2026-05-21

Scope: observation parity and numeric-drift validation only. I did not touch
live training runs, trainer defaults, production code, Modal Volumes,
checkpoints, tournaments, or eval artifacts.

## Plain Read

The current GPU observation lane has enough local shape coverage to keep
working, but not enough semantic evidence to promote any faster path as the
same observation surface.

The safest interpretation is:

- CPU oracle `browser_lines + simple_symbols` remains the trusted training
  observation.
- `direct_gray64` is a fast policy-space approximation. It has exact parity
  against its CPU-direct oracle on focused adversarial/simple-symbol cases, but
  it is not browser-pixel or full CPU-oracle parity.
- Persistent framebuffer is the right cost model for long trails, but its real
  risk is stale state: wrong row cleared, cursor regression missed, palette
  mutation missed, or terminal/final observation taken after reset.
- LightZero consumer probes should consume the pre-scalar `[B,2,4,64,64]`
  stack only after row/player order, action masks, `to_play`, and terminal
  filtering are pinned.

## Existing Coverage Worth Keeping

| Area | Current useful coverage | Residual risk |
| --- | --- | --- |
| `direct_gray64` CPU/JAX parity | `tests/test_source_state_gpu_render_benchmark_cpu.py` checks adversarial two-view JAX parity, CPU-direct parity, previous same-owner trail connection, inactive slots after cursor, and `direct_gray64 + simple_symbols`. | Exact only against CPU-direct policy-space oracle, not the production CPU oracle. Future summaries must not call this browser/canvas parity. |
| `simple_symbols` | All 12 bonus identities stay distinct; bonus symbols overwrite trails; heads overwrite bonus symbols. | Missing one combined fixture with non-identity `avatar_color`, both views, symbols, and head/bonus overlap in the same frame. |
| Row/player order | Batched facade, boundary helper, hybrid manager, and sentinel tests pin row-major order: `(row0,p0), (row0,p1), ...`. | A single fake consumer should assert the exact pixels that LightZero receives after flattening, not only manager metadata. |
| Stack update | FIFO stack shift, latest-frame normalization, selected-row reset, `uint8` stack scalarization, and RND latest-frame extraction have local tests. | Mixed terminal/live rows need one combined fake test so terminal rows do not pollute live roots or post-reset frames. |
| Reset/final observation | Scalar LightZero wrapper and profile manager preserve terminal `final_observation`; materializers attach and validate final stacks. | Existing profile-manager terminal tests tend to terminate all rows. The high-value gap is mixed terminal/live row behavior. |
| Persistent framebuffer | Config fails closed to `direct_gray64`; delta state has cold-start and same-owner append unit tests; synthetic Modal framebuffer has exact stateless/persistent parity. | Missing local cursor-regression/reset-row coverage and missing real persistent-vs-stateless same-surface GPU parity. |
| Tolerance | Boundary config uses exact mode for float64 auto; float32 auto defaults to tolerant `max_abs_diff=2`, `mismatch_fraction=1e-4`; tests reject exact drift and large tolerant drift. | Recent persistent/direct64 rows passed deliberately loose divergence checks with max diff around `61-67` and up to `2.71%` mismatches. That is acceptable as divergence telemetry, not as parity. |

## Smallest High-Value Local Tests

### P0. Fake Consumer Row/Player/Stack Sentinel

Add to `tests/test_source_state_hybrid_observation_profile.py` or
`tests/test_source_state_batched_observation_boundary_profile.py`.

Use a CPU-only renderer whose frame value is unique per `(row, player, step)`,
for example:

```text
value = 10 * (row + 1) + (player + 1) + 40 * step
```

Drive one reset and one step through the renderer-backed hybrid path, then run a
fake `HybridBatchedStackProbe` or fake `_LightZeroCollectForwardStackProbe`.

Pass criteria:

- `observation[b,p,-1,0,0]` equals the current step sentinel.
- `observation[b,p,-2,0,0]` equals the reset/previous-step sentinel.
- Flattened consumer order is `[row0p0, row0p1, row1p0, row1p1, ...]`.
- `policy_env_row`, `policy_player`, `ready_env_id`, and fake consumer first
  pixel all agree.

Why this is high value: it catches the quietest dangerous bug, a transposed
row/player stack that still has plausible shapes and speeds.

### P0. Mixed Terminal/Live Final Observation Fake

Add to `tests/test_source_state_batched_observation_mock_collector.py`.

Use low-level materializer inputs, not a real env, to force:

```text
B=2, P=2
row0 terminal, row1 live
final_observation sentinel only on row0
terminal action mask all false
live action mask all true
```

Pass criteria:

- Terminal scalar infos for row0 players have `final_observation_present=true`
  and carry the terminal sentinel.
- Live row1 infos have `final_observation_present=false`.
- Consumer-ready roots include only live row1 players if zero-mask filtering is
  part of the path.
- Row/player ids for live roots remain `[1,0]`, `[1,1]`, not compacted into
  physical row zero.

Why this is high value: it pins reset/final-observation semantics without
launching training or needing a natural-death rollout.

### P0. Persistent Delta Cursor Regression And Row-Selective Reset

Extend `tests/test_source_state_batched_observation_boundary_profile.py` around
`_persistent_delta_state`.

Fixture:

- two rows;
- row0 previous cursor `5`, current cursor `2` with active prefix slots;
- row1 previous cursor `2`, current cursor `4`;
- previous owner positions are nonzero so stale carryover is visible;
- include `break_before=True` on one active slot.

Pass criteria:

- `reset_mask == [1, 0]`.
- row0 delta starts from its own current prefix and does not connect to stale
  previous owner position.
- row1 delta appends only slots `[2, 4)`.
- `next_owner_pos` and `next_owner_valid` are row-selective.

Why this is high value: persistent speedups fail semantically if cursor
regression or partial reset leaves old trails in a row.

### P0. Persistent Palette Mutation Invalidates Cache

Add a CPU-only fake around the persistent renderer state machine if possible,
or a small test around the invalidation inputs if JAX is unavailable locally.

Fixture:

- first render has `avatar_color=[[0,1], ...]`;
- second render changes only row0 palette to `[[1,0], ...]`;
- trail geometry is unchanged.

Pass criteria:

- row0 is marked for rebuild/reset or cache invalidation.
- row1 remains incremental.
- telemetry exposes the reset/rebuild row count.

Why this is high value: controlled-player self/other luma can silently swap
while the trail layer looks geometrically correct.

### P0. Tolerance Classification Guard

Add to `tests/test_source_state_batched_observation_boundary_profile.py`.

Build one exact comparison, one tolerated comparison, and one loose divergent
comparison. Assert:

- exact comparisons set `all_exact=true`;
- tolerated comparisons set `all_exact=false` and expose sample/bbox data;
- any helper that summarizes speed rows cannot label tolerated or loose
  divergence as `parity_exact`;
- same-surface persistent-vs-stateless tests must use exact parity unless an
  explicit `approximate_surface_drift_ack` flag is present.

Recommended tolerance rule:

- uint8 renderer same-surface tests: `np.testing.assert_array_equal`.
- normalized float stacks from uint8 frames: `rtol=0.0`, `atol=1e-7`.
- GPU float32 dense renderer smoke: at most `max_abs_diff<=2` and
  `mismatch_fraction<=1e-4`, with samples.
- `direct_gray64` versus production CPU oracle: call it divergence, not parity;
  record `max_abs_diff`, mismatch fraction, and sample/connected-component diff.

### P1. Combined Direct-Symbol Perspective Fixture

Add a local CPU/JAX-importskip test beside the existing `direct_gray64`
simple-symbol tests.

Use one asymmetric state with:

- non-identity and duplicate `avatar_color`;
- both controlled-player views;
- one diagonal trail, one close parallel trail, one `break_before` gap;
- at least two bonus symbols overlapping trail pixels;
- one head overlapping a bonus.

Pass criteria:

- CPU-direct and JAX direct outputs match exactly when JAX is available.
- player 0 and player 1 views differ in self/other luma where expected.
- bonus pixels are seat-invariant except where heads overwrite them.

Why this is P1: the components are already tested separately; this catches
composition drift when future kernels refactor draw order.

## Smallest Modal/GPU Smokes

These should be profile-only and detached or artifact-writing only if needed.
They must not resume live runs or write training outputs.

### M0. Direct Gray64 Exact GPU Kernel Smoke

Purpose: verify the current GPU kernel still matches CPU-direct on the
adversarial policy-space contract.

Command shape:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_gpu_render_benchmark \
  --compute gpu-h100 \
  --state-source adversarial_fixture \
  --batch-size 4 \
  --player-count 3 \
  --trail-slots 10 \
  --bonus-count 3 \
  --render-surface direct_gray64 \
  --render-views both \
  --bonus-render-mode simple_symbols \
  --verify-rows 4 \
  --transfer-output true \
  --warmup-runs 1 \
  --steady-runs 2
```

Pass criteria:

- `ok=true`;
- `verification.exact=true` or equivalent exact CPU-direct comparison;
- output order is documented as view-major and then explicitly converted before
  stack ownership.

### M1. Persistent Versus Stateless Same-Surface GPU Parity

Purpose: isolate persistent stale-state bugs from lower-fidelity surface drift.

Add or run a profile-only Modal row that compares:

```text
same compact source-state rollout
-> stateless direct_gray64 GPU render
-> persistent direct_gray64 GPU framebuffer render
-> exact compare latest frames and full stacks
```

Required cases:

- reset/cold start;
- append-only step;
- cursor regression or row reset;
- avatar color mutation;
- bonus/head composition after persistent trail update.

Pass criteria:

- exact uint8 latest-frame parity;
- exact normalized stack parity with `rtol=0`, `atol=1e-7`;
- telemetry reports reset rows, delta slots, cache size, and partial requests.

This is the missing GPU smoke I would add before trusting persistent speed rows.

### M2. Persistent Direct64 Versus CPU Oracle Divergence Smoke

Purpose: measure the approximation against the trusted CPU oracle and keep the
language honest.

Command shape:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --compute gpu-h100 \
  --surface-facade-canary true \
  --surface-facade-divergence-canary true \
  --surface-stack-backend renderer_backed_profile \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --render-surface direct_gray64 \
  --hybrid-stack-storage-dtype uint8 \
  --batch-size 16 \
  --steps 8 \
  --warmup-steps 1 \
  --verify-steps 8 \
  --cpu-reference-interval 1 \
  --max-ticks 2000 \
  --parity-mode tolerant
```

Pass criteria:

- CPU-vs-CPU control for the same harness remains exact.
- Persistent row reports `all_exact=false` unless it truly is exact.
- Result includes `max_abs_diff`, mismatch fraction, mismatch sample/bbox, active
  trail stats, final-observation count if terminal rows occur, and render
  truncation count.
- Summary uses "divergence" or "approximate direct64", not "CPU oracle parity".

### M3. Terminal/Autoreset Divergence Smoke

Purpose: prove terminal final observations are compared before row reuse.

Use the same boundary profile as M2 but with a tiny terminal setup:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --compute gpu-l4-t4 \
  --surface-facade-canary true \
  --surface-facade-divergence-canary true \
  --surface-stack-backend renderer_backed_profile \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --render-surface direct_gray64 \
  --hybrid-stack-storage-dtype uint8 \
  --batch-size 8 \
  --steps 3 \
  --warmup-steps 0 \
  --verify-steps 3 \
  --cpu-reference-interval 1 \
  --max-ticks 1 \
  --parity-mode tolerant
```

Pass criteria:

- terminal rows observed;
- `final_observation` compared separately from post-reset observation;
- no hidden CPU fallback;
- persistent reset-row count is nonzero.

### M4. LightZero Collect-Forward Consumer Smoke

Purpose: verify the real LightZero consumer sees the same batched stack order and
mask semantics, without `train_muzero`.

Command shape:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --compute gpu-h100 \
  --hybrid-observation-canary true \
  --hybrid-lightzero-collect-forward-probe true \
  --hybrid-materialize-scalar-timestep false \
  --surface-stack-backend renderer_backed_profile \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --render-surface direct_gray64 \
  --hybrid-stack-storage-dtype uint8 \
  --batch-size 16 \
  --actor-count 2 \
  --steps 2 \
  --warmup-steps 1 \
  --hybrid-lightzero-consumer-num-simulations 8 \
  --hybrid-lightzero-consumer-temperature 1.0 \
  --hybrid-lightzero-consumer-epsilon 0.0
```

Pass criteria:

- `calls_train_muzero=false`;
- `materialize_scalar_timestep=false`;
- input shape to consumer is `[B*2,4,64,64]`;
- normalized tensor min/max are within `[0,1]`;
- `ready_env_id` is dense and row/player mapping is recoverable;
- fixed-opponent `to_play` is `-1` for all roots unless a two-seat policy config
  explicitly asks for player ids;
- zero-mask roots are filtered;
- illegal action count is zero;
- output action checksum is nonzero or explicitly recorded.

## Promotion Language Gate

Before any speed row is used in a recommendation, require the row or its linked
artifact to answer these yes/no questions:

- Is this CPU oracle, CPU-direct, stateless GPU direct64, or persistent GPU
  direct64?
- Is the comparison exact parity or approximate divergence?
- What are the exact tolerances?
- Did row/player order, latest stack channel, reset rows, and
  `final_observation` pass?
- Were terminal zero-mask roots filtered before LightZero collect forward?
- Was there any hidden CPU fallback?

Small rule of thumb: exact tests protect semantics; tolerant tests quantify
approximation. A tolerant pass should never be the only evidence for "same
observation".

## Recommended Order

1. Add the three P0 CPU-only tests: fake consumer sentinel, mixed terminal/live
   materializer, persistent cursor-regression reset.
2. Add the tolerance classification guard so future summaries cannot turn loose
   divergence into parity by accident.
3. Run M0 exact direct64 GPU kernel smoke.
4. Add/run M1 persistent-vs-stateless same-surface GPU parity.
5. Keep M2/M3 as divergence telemetry and sample artifacts.
6. Run M4 only after row/player and terminal tests are green, because otherwise
   a LightZero speed result is too easy to misread.

