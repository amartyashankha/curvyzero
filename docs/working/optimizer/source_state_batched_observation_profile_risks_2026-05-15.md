# Source-State Batched Observation Profile Risks

Status: profile-only sidecar. Do not wire this into stock LightZero, live runs,
or trainer defaults.

## What the Facade Proves

- A local vector env batch can own `N` rows and update policy-shaped
  `[4,64,64]` stacks outside the trainer.
- The intended renderer boundary can be named as:
  `VectorMultiplayerEnv.state + row_indices + controlled_players -> uint8[B,1,64,64]`.
- Pack/render/readback/stack/reset/final_obs timing slots can be collected even
  while the current implementation is a CPU oracle loop.

## How This Can Mislead Us

- It runs in-process and local. It does not model subprocess env-manager
  serialization, collector scheduling, replay writes, or LightZero policy calls.
- The CPU oracle loop reuses the existing renderer row by row. That proves the
  facade shape, not a batched speedup.
- Reset and terminal handling are much simpler than trainer reality. The facade
  captures final stacks before any reset, but it does not yet prove autoreset
  ordering under an env manager or replay buffer.
- It profiles the current controlled-player observation surface. It does not
  prove tournament/eval policy loading will feed the same player perspective,
  stack dtype, or `action_mask` shape into a checkpoint policy.
- The default profile death mode can keep rows alive for observation timing.
  That is useful for throughput but can hide terminal-path bugs.
- No GPU memory pressure, device transfer batching, stream sync, or host-readback
  contention exists here. The readback timing slot is a placeholder.

## Drift-Sensitive Contract Fields

- `trail_render_mode`: must remain `browser_lines`; old adjacent-slot semantics
  can look plausible while breaking previous-active same-owner connectivity.
- `bonus_render_mode`: must remain `simple_symbols`; browser-sprite or circle
  fallbacks change model pixels.
- `controlled_players`: must be per output row, not a single global player hidden
  inside a renderer closure.
- `avatar_color`: self/other luma must follow color ownership, not raw player
  index, especially after color bonuses.
- `visual_trail_write_cursor`: stale slots after the cursor must stay masked.
- `visual_trail_break_before`: gap semantics must suppress line connection.
- `terminal_final_observation`: final stack must be captured before any reset or
  row reuse.
- `reset_stack_policy`: reset must zero old history then push exactly one fresh
  frame unless the trainer contract explicitly changes.

## Recommended Gates Before Any Integration

1. CPU facade parity: reset, one step, and terminal final_obs stacks must match
   direct `render_source_state_canvas_gray64` for every row and controlled player.
2. GPU/lab parity: compare `uint8[B,1,64,64]` output against the CPU oracle on
   real env rollouts plus adversarial cursor/break/color fixtures.
3. Stack lifecycle gate: prove reset, autoreset, terminal final_obs, and partial
   row reset preserve FIFO stack semantics under the intended env-manager shape.
4. Tournament policy gate: load a real checkpoint policy in eval-only mode and
   assert observation shape, dtype, controlled-player perspective, and action
   masks before any tournament score is trusted.
5. Timing honesty gate: report pack/render/readback/stack/reset/final_obs
   separately; never fold readback or final_obs into render time.
6. Promotion gate: keep `cpu_oracle` as production default until the batched
   backend clears parity and full-loop canary gates without changing trainer
   defaults.
