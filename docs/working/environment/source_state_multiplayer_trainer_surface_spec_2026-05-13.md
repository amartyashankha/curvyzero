# Source-State Multiplayer Trainer Surface Spec - 2026-05-13

Status: v0 surface, in-memory replay arrays, repo-owned target rows,
deterministic sample batches, and fake/injected native `GameSegment` mapping
implemented. The separate opt-in real-LightZero construction helper also
exists, but it is construction-smoke only; real buffer sampled parity is still
open.
Surface id: `source_state_multiplayer_trainer_surface/v0`.
Owner surface: trainer/replay wrapper over product runtime and renderer.

Purpose: define the next trainer-facing multiplayer wrapper without letting
old single-ego, fixed-opponent, scalar, or profile paths masquerade as
multiplayer self-play. The product runtime remains `VectorMultiplayerEnv`.
The next contracts after this surface are implemented replay -> repo-owned
target rows -> deterministic sample batches -> fake/injected native
`GameSegment` mapping. The bridge is injection-only and not a real LightZero
`GameSegment` or buffer claim. A separate opt-in real-LightZero construction
helper now exists, but it remains construction smoke only until real
`MuZeroGameBuffer` sampled-target parity passes.

## Evidence Boundaries

- Source truth: original JS/source oracle behavior and source-env parity.
- Product runtime: `VectorMultiplayerEnv` state, source-frame stepping,
  rewards, done flags, player presence/alive state, and native sidecars.
- Renderer: source-state RGB canvas-like frames, gray64 downsampling, render
  mode metadata, and approximation guards.
- Trainer/replay: this wrapper's per-seat visual stacks, masks, rewards, final
  visual observations, policy-row mapping, metadata, in-memory replay arrays,
  repo-owned target rows, deterministic sample batches, and injection-only
  native `GameSegment` mapping. Real LightZero buffer sampling is not
  implemented by v0.

Do not close a trainer/replay gap with source-state visual evidence alone.
Do not close a renderer gap with metadata-only replay. Do not call a route smoke
or fixed/frozen opponent run true multiplayer self-play.

## Runtime And Control Model

- Runtime: one optimized/product environment, `VectorMultiplayerEnv`.
- Supported v0 player counts: P=2 first, then P=3 and P=4 smoke after the P=2
  contract is stable.
- Source model: real-time controls plus elapsed-ms source frames. The source
  game is not native trainer decisions; the trainer decision cadence is a
  wrapper abstraction.
- Cadence metadata must include `decision_source_frames`, derived
  `decision_ms`, `source_physics_step_ms`, and elapsed source-frame count for
  the decision.
- Actions are player-major at the env boundary: `joint_action int[B,P]`, with
  per-seat masks and sidecars. Dead, absent, or terminal seats should not become
  policy rows.

## Observation Contract

Default visual path:

```text
source-state canvas-like RGB frame at 704x704
-> BT.601/luma gray64 through 11x11 area downsample
-> four-frame FIFO stack
-> float32[B,P,4,64,64]
```

Implemented output fields for reset/step:

- `observation`: `float32[B,P,4,64,64]`, player-perspective source-state
  stack for each seat.
- `legal_action_mask`: `bool[B,P,3]`, legal trainer actions for each seat.
- `lightzero_action_mask`: `bool[B,P,3]` mirror for adapters that need
  LightZero-style masks.
- `reward`: `float32[B,P]`, survival-plus-bonus reward from the trainer
  surface: alive after step plus same-step bonus catches.
- `done`, `terminated`, `truncated`: `bool[B]` row-level episode flags.
- `live_mask`: `bool[B,P]`, true only where a shared policy should receive a
  row this decision.
- `policy_observation`: `float32[R,4,64,64]`, where
  `R = live_policy_row_mask.sum()`.
- `policy_action_mask`: `bool[R,3]`.
- `policy_env_row`: `int32[R]`, mapping flattened policy rows back to env rows.
- `policy_player`: `int16[R]`, zero-based public player index for each policy
  row.
- `joint_action`: `int16[B,P]` on step returns, after mapping selected policy
  rows back to player-major env controls.
- `native_action_sidecar`: source/native control metadata through `info`.
- `info`: metadata described below.

Final observation fields:

- `final_observation`: `float32[B,P,4,64,64]`, populated for final rows before
  any autoreset mutation and zero-filled elsewhere.
- `final_reward_map`: `float32[B,P]`, populated for final rows and zero-filled
  elsewhere.
- `final_observation_row_mask`: `bool[B]`.
- `final_observation_policy`: metadata naming terminal-before-autoreset
  capture. The wrapper must keep reset observations and final observations as
  separate facts.

## Render Mode Rules

Reusable renderer: `SourceStateGray64Stack4` in
`src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`.

- Default: `browser_lines`.
  This is the source-state visual path for v0. It uses canvas-like RGB at 704,
  gray64 downsample, per-seat perspective, and the four-frame stack.
- Explicit approximation: `body_circles_fast`.
  This may be used only when the command and metadata say it is approximate.
  It cannot be cited as source-state visual-gate evidence.
- Profile/custom only: `fast_gray64_direct`.
  This bypasses the raw 704 RGB path. v0 training commands must reject it or
  require an explicit non-training/profile flag and metadata
  `training_visual_surface_allowed=false`.

Every reset/step artifact must record:

- `render_mode`, `default_render_mode`, `supported_render_modes`.
- `render_pipeline`.
- `rgb_source_frame_size`, `downsample_target_frame_size`, and
  `rgb_to_gray64`.
- `trail_renderer_is_approximation`.
- `bonus_renderer_is_approximation`.
- `renderer_impl_id` or equivalent stable renderer schema id.
- `observation_schema_id` and stack owner.

## Metadata Guardrails

Every reset/step and any emitted artifact should include:

- `surface_id: source_state_multiplayer_trainer_surface/v0`.
- `runtime_env_class: VectorMultiplayerEnv`.
- `runtime_env_id` and rules/schema hashes if available.
- `player_count`, `batch_size`, `source_player_ids`, and public player ids.
- `decision_source_frames`, `decision_ms`, `source_physics_step_ms`, and
  elapsed-ms/source-frame accounting.
- `native_control_model_id`, `trainer_control_wrapper_id`, action ids, and
  noop/terminal-padding policy.
- `reward_schema_id`, `final_observation_policy`, and autoreset policy.
- Live-seat flattening maps: `policy_env_row`, `policy_player`, and the exact
  policy-row count.
- Project-only helper metadata when enabled, including `profile_no_death`,
  no-death/profile modes, optimizer modes, and training-helper modes. These are
  valid project additions and must be preserved through replay and target rows,
  but must be labeled as not original CurvyTron/source-fidelity behavior.
- Claim fields: `trainer_observation_claim`,
  `trainer_replay_claim`, `renderer_claim`, and `training_claim`.

Implemented claim values for the first v0 surface:

- `trainer_observation_claim_id: source_state_visual_stack_per_live_seat/v0`.
- `trainer_replay_claim: false` on the surface step object itself.
- `renderer_claim: source_state_canvas_like_rgb704_gray64_stack4`, with
  approximation flags from the selected render mode.
- `training_claim: wrapper_smoke_only`, until replay, target construction,
  sampling, learner, and eval semantics are implemented.

## In-Memory Replay Arrays

Implemented recorder:
`SourceStateMultiplayerTrainerReplayRecorder` in
`src/curvyzero/training/multiplayer_source_state_trainer_replay.py`.

It stores copied arrays over time:

- `observation[T,B,P,4,64,64]`.
- `legal_action_mask[T,B,P,3]`.
- `lightzero_action_mask[T,B,P,3]`.
- `live_mask[T,B,P]`.
- `joint_action[T,B,P]`.
- `reward[T,B,P]`.
- `done`, `terminated`, `truncated`: `bool[T,B]`.
- `final_observation[T,B,P,4,64,64]`.
- `final_observation_row_mask[T,B]`.
- `final_reward_map[T,B,P]`.
- Variable live-policy row arrays per record:
  `policy_observation`, `policy_action_mask`, `policy_env_row`,
  `policy_player`.

The recorder validates that policy row maps match `live_mask` and that
`policy_observation` / `policy_action_mask` match the source batch rows. It
sets `trainer_replay_claim: true` only for this array replay contract and keeps
`lightzero_training_claim`, `native_game_segment_claim`, and native target
claims false. Replay records also carry compact source-state audit metadata
needed to inspect terminal bonus/death cases after the fact: step counters,
bonus support/stack counts, death cause/player/owner arrays, winner/loser
facts, score/alive/present arrays, and final-observation/final-reward policies.
This metadata is an audit trail; the trainer tensors are still the copied
arrays above. `SourceStateMultiplayerTargetRowsV0` now consumes these replay
arrays with explicit `PolicyRowRecordV0` sidecars, validates transition
alignment, preserves project-only mode metadata, and still keeps native
LightZero bridge claims false.
`SourceStateMultiplayerSampleBatchV0` and
`build_source_state_multiplayer_sample_batch_v0` now build deterministic sample
batches on top of those target rows. Focused target-row/sample-batch tests
reported `12 passed` locally per worker.
`src/curvyzero/training/multiplayer_source_state_native_bridge.py` now maps
`SourceStateMultiplayerTargetRowsV0` into injected `GameSegment`-like objects
without importing LightZero. The bridge preserves project-only helper metadata
and keeps native/LightZero/training/buffer/learner claims false. Do not turn
this injection-only bridge into the real-LightZero import path. The separate
opt-in real-LightZero construction helper is construction-smoke only; keep
buffer/training claims false until sampled-target parity is proven.

Event records:

- `remove_player(...)` packages a leave event. It is not a policy action step.
- `advance_warmdown(...)` packages timer/lifecycle progress. It is not a
  policy action step.
- These records may expose live policy rows after the event so the next policy
  call can be made from the new state. Those rows do not mean the event itself
  came from policy/search.
- Event records use `joint_action=-1` padding. Target-row construction must not
  build a training transition whose result record is one of these event records.
  A valid action transition reads the selected action from a following real
  `step(...)` record.

## Shared-Policy Row Mapping

The policy interface is seat-perspective, not single-ego:

1. Build `observation[B,P,4,64,64]` and `legal_action_mask[B,P,3]`.
2. Compute `live_policy_row_mask` from present, alive, legal, and non-terminal
   seats.
3. Flatten only live player seats into `R` policy rows.
4. Run one shared policy over those rows.
5. Map policy actions back with `policy_env_row` and `policy_player`.
6. Fill `joint_action[B,P]` for the env. Non-live seats get the documented
   noop/terminal padding action and must remain absent from policy training
   rows.

Do not feed one player perspective and silently fill the rest as a hidden fixed
opponent when claiming this surface. Fixed/frozen opponents are a separate
single-ego control route.

## No-Overclaim Rules

- This spec does not make current LightZero training true multiplayer
  self-play. Current stock training is fixed/frozen opponent and single-ego.
- This spec now has replay -> target rows -> deterministic sample batches ->
  fake/injected native `GameSegment` mapping implemented. It does not implement
  real LightZero `GameSegment`/buffer sampling, learner updates, or evaluation
  panels.
- This spec does not prove browser canvas pixel parity.
- This spec does not allow fake, blank, scalar, metadata-only, direct-gray, or
  stale debug observations to substitute for source-state visual stacks.
- This spec does not remove project training/profile additions such as
  `profile_no_death`. Those modes must keep working, but artifacts must label
  them as training/profile helpers and not source-fidelity behavior.
- This spec proves focused P=3/P=4 step, terminal, mixed-row warmdown, and
  mixed active/warmdown presence-leave trainer/replay behavior. It does not
  prove broad source-fixture P=3/P=4 presence/leave breadth, bonus stack/death,
  browser pixel parity, or eval behavior.

## Work Plan

1. Done: implement a thin wrapper over `VectorMultiplayerEnv` that reuses
   `SourceStateGray64Stack4` with default `browser_lines`.
2. Done: add render-mode validation that rejects silent `fast_gray64_direct` and
   labels `body_circles_fast` approximate.
3. Done: capture `final_observation` before autoreset and keep it separate
   from reset observations.
4. Done: add a P=2 smoke for per-seat stacks, live-seat flattening, action mapping,
   masks, rewards, done flags, final visual observation, and metadata.
5. Done: add P=3/P=4 step smokes and a P=4 terminal visual final-observation
   smoke.
6. Done: add in-memory trainer replay arrays for the surface.
7. Done: add focused P=3/P=4 mixed-row warmdown lifecycle proof through
   `SourceStateMultiplayerTrainerSurface.advance_warmdown(...)` and
   `SourceStateMultiplayerTrainerReplayRecorder`.
8. Done: add the repo-owned target-row adapter that consumes replay arrays,
   joins policy-row records, preserves no-death/profile/training-helper
   metadata, and keeps native LightZero claims false.
9. Done: add deterministic sample batches on top of
   `SourceStateMultiplayerTargetRowsV0`.
10. Done: add fake/injected native `GameSegment` mapping from target
   rows.
11. Done: add a separate opt-in real-LightZero construction helper, still
    construction-smoke only.
12. Later: prove real `MuZeroGameBuffer` sampled-target parity.
