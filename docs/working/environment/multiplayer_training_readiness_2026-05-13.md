# Multiplayer Training Readiness - 2026-05-13

Status: concise readiness ranking, not a completion claim.
Owner surface: Environment docs/process.

This note ranks what multiplayer training needs next and what is already
proven. It reads the training-facing surface through the current environment
queue: source truth, `VectorMultiplayerEnv`, trainer wrapper, replay/final
observations, and LightZero/coach routes are separate proof surfaces.

Relevant anchors:
[current_queue_2026-05-13.md](current_queue_2026-05-13.md),
[full_game_fidelity_reorientation_2026-05-13.md](full_game_fidelity_reorientation_2026-05-13.md),
[multiplayer_training_interface_2026-05-10.md](multiplayer_training_interface_2026-05-10.md),
[lightzero_env_requirements_2026-05-10.md](lightzero_env_requirements_2026-05-10.md),
[trainer_observation_reward_contract_v0_2026-05-09.md](trainer_observation_reward_contract_v0_2026-05-09.md),
and [source_state_multiplayer_trainer_surface_spec_2026-05-13.md](source_state_multiplayer_trainer_surface_spec_2026-05-13.md).

## Readiness Ranking

1. Keep the new target/bridge scaffolding honest, then return to environment
   fidelity.
   `SourceStateMultiplayerTrainerSurface` now emits per-seat source-state
   visual stacks `float32[B,P,4,64,64]`, live-seat policy-row mappings, masks,
   survival-plus-bonus rewards, terminal visual final observations, and honest
   metadata. `SourceStateMultiplayerTrainerReplayRecorder` now stores copied
   in-memory arrays over time. Repo-owned target rows, deterministic sample
   batches, fake/injected native mapping, and opt-in real-LightZero
   construction smoke exist. Serious training still needs real
   `MuZeroGameBuffer` sampled parity before buffer/training claims.
2. Keep the current stock LightZero route contained as a control, not the
   multiplayer destination.
   The current stock training path is fixed/frozen opponent and single-ego. It
   is useful for route and trainer mechanics, but it is not true multiplayer
   self-play. The next destination is one shared policy reading one seat
   perspective at a time, with live player rows flattened and then mapped back
   into player-major control actions for `VectorMultiplayerEnv`.
3. Expand 2P lifecycle and replay breadth.
   The next useful training proof is not another raw visual gate by itself. It
   is warmdown/match-state behavior, reset/autoreset separation, terminal
   transition capture before reset, and replay rows that can reconstruct the
   trainer transition without guessing.
4. Promote bonus stress only where training will use bonuses.
   Source-default bonus probability, spawned type set, several public seeded
   effects, and focused natural effect slices are now useful. The two promoted
   2P `BonusSelfFast` terminal cases now also survive through trainer
   surface/replay final visual rows, final rewards, death metadata, and bonus
   audit metadata. Training still needs other long-run stack/death cases before
   bonus-enabled multiplayer runs can be called source-faithful.
5. Add raw JS oracle fixtures for hit-owner stress.
   Runtime stress tests cover the important owner-order cases, and focused
   propagation now carries one 3P terminal case and one 4P nonterminal
   two-victim case through public env, trainer surface, replay metadata, and
   debug events. Training still needs raw JS fixtures for those exact 3P/4P
   stress shapes and broader collision edges.
6. Widen to 3P/4P only after the 2P ladder is stable.
   Existing 3P/4P scalar projection and metadata artifacts are useful
   scaffolding, not trainer-ready multiplayer. Before 3P/4P training claims,
   decide the learned-observation schema, ego rotation, opponent-policy
   sidecars, match/round semantics, replay ownership, and evaluation panel.
7. Keep browser/canvas pixels out of the P0 training blocker list.
   The active training visual surface is source-state raw RGB to gray64 stack.
   Browser/canvas pixel parity may matter later for human/browser parity, but
   it should not displace trainer-wrapper/replay/final-observation proof.

## Current Trainer-Surface Plan

1. Done: land `source_state_multiplayer_trainer_surface/v0` as a wrapper over
   `VectorMultiplayerEnv`, reusing `SourceStateGray64Stack4` with default
   `browser_lines`.
2. Done: guard render modes. `browser_lines` is the default source-state visual path,
   `body_circles_fast` is explicit approximate mode, and
   `fast_gray64_direct` is profile-only/custom and must not silently enter
   training runs.
3. Done: add terminal handling that exposes final visual stacks before reset, plus
   final masks/rewards and reset observations as separate facts.
4. Done: smoke P=2 first with one shared policy surface, live-seat flattening,
   policy-row-to-player-major action mapping, rewards, masks, done flags, and
   metadata guardrails.
5. Done: P=3/P=4 step smokes cover live-row filtering and rewards; P=4 terminal
   smoke proves visual final observation after three wall deaths.
6. Done: add in-memory trainer replay arrays for the new surface.
7. Done: add repo-owned target rows, deterministic sample batches,
   fake/injected native mapping, and opt-in real-LightZero construction smoke.
   Next downstream proof is real `MuZeroGameBuffer` sampled parity.

## Already Proven

- There is one public runtime name under hardening:
  `VectorMultiplayerEnv`.
- The active 2P source-state product image path is source-state browser-like
  704x704 RGB raw frame -> 11x11 area-downsampled gray64 -> frame stack.
  `browser_lines` is the default source-state render mode;
  `body_circles_fast` is an explicit approximation for speed/profiling.
  `fast_gray64_direct` exists only as a profile/custom path and must be blocked
  or loudly labeled if a training command asks for it.
- The latest full 2P source-state visual gate reported exact source/vector
  agreement through the native render path:
  `canvas_gray64=35/35`, `typed_bonus=12/12`, `final_obs=pass`,
  `mismatch_pixels=0`, and `max_abs_diff=0.0`. This is source-state visual
  evidence, not browser pixel or full trainer evidence.
- Controls now use `decision_source_frames`, derive `decision_ms`, hold controls
  over source-sized internal frames, and stop early on death. Current proof
  covers 2P/3P/4P mapping slices, held-control parity, release-to-straight,
  invalid/live action rejection, inactive noops, and one direct plus one
  LightZero-facing terminal early-stop trace.
- Direct 2P product-route proof covers raw RGB -> gray64, seeded
  `BonusGameClear`, stale trail/body clear, live ticks, terminal wall death,
  rewards, final observation masks, and metadata replay.
- LightZero-facing wrapper proof covers scalar joint-action decoding, raw RGB
  -> gray64 stack, held source frames, terminal final observation, rewards,
  masks, and native sidecars.
- Bonus defaults are no longer blocked on type-selection weights: source
  default non-`BonusGameClear` bonuses effectively have probability `1`.
  Natural bonus proof pins corrected `BonusGameClear` boundary draws, RNG
  labels/cursors, next-delay scheduling, spawned position, and all 12
  source-default bonus types.
- Hit-owner runtime stress proof covers 4P newest-owner overlap, 4P source-style
  corner island order, 3P own-body latency, and 4P two-victim metadata
  alignment.
- Metadata-only multiplayer wrappers, opponent policy sidecars, 3P/4P scalar
  projections, and replay-shaped scalar artifacts exist as scaffolding. They
  explicitly do not claim learned-observation trainer readiness.
- `SourceStateGray64Stack4` is the reusable source-state visual stack. Its
  default render mode is `browser_lines`; it produces per-seat
  player-perspective stack rows and should be the starting renderer for the new
  trainer surface.
- `SourceStateMultiplayerTrainerSurface` now proves the first trainer-facing
  surface over `VectorMultiplayerEnv`: reset/step visual observations come from
  `SourceStateGray64Stack4`, not metadata-only env observations; live policy
  rows are flattened with env/player maps; `fast_gray64_direct` is rejected;
  `body_circles_fast` is explicit approximate mode; and
  `bonus_render_mode` stays `browser_sprites`.
- `SourceStateMultiplayerTrainerReplayRecorder` now proves the first
  trainer-array replay surface over that wrapper. It stores copied arrays over
  time for observations, masks, joint actions, rewards, done flags, terminal
  visual final observations, final reward maps, and variable live-policy rows.
  The focused P=3/P=4 mixed-row warmdown lifecycle shape is now covered through
  the trainer surface and replay recorder. The focused 2P `BonusSelfFast`
  stack-death and expiry-before-wall-death cases are also covered through
  trainer/replay terminal rows and compact audit metadata. It does not claim
  LightZero
  buffer/training integration.
- Repo-owned target rows, deterministic sample batches, fake/injected native
  mapping, and opt-in real-LightZero construction smoke are implemented
  scaffolding. The real `MuZeroGameBuffer` sampled reward, value, policy,
  action, mask, observation, and `to_play` parity proof is still open and
  downstream of Environment Reconstruction.

## Do Not Overclaim

- Do not infer trainer readiness from gray64, bonus64, or source-state visual
  gates alone.
- Do not call metadata-only replay a production trainer replay shard.
- Do not label the old custom two-seat learner path as the trusted training
  lane unless it calls stock `train_muzero`, feeds native LightZero
  `GameSegment`/`GameBuffer` targets, or has a parity-tested repo-owned target
  contract.
- Do not call the fixed/frozen-opponent single-ego LightZero path true
  multiplayer self-play.
- Do not claim LightZero training readiness from
  `source_state_multiplayer_trainer_surface/v0` or its in-memory replay arrays
  just because target rows, sample batches, fake/injected mapping, or
  construction smoke exist. Real `MuZeroGameBuffer` sampled parity is still
  unproven.
- Do not describe CurvyTron as ALE. It should expose LightZero-compatible
  CurvyTron env shapes directly.
- Do not treat strict `VectorTrainerEnv1v1NoBonus` as the destination. It is a
  proof/profiling boundary.

## Operating Pattern

For larger environment changes, use a spec-first worker pattern:

1. Write a detailed but bounded spec for the target behavior and proof surface.
   Name the source behavior, product route, wrapper/replay/render boundary,
   non-claims, and focused validation commands before implementation starts.
2. Split implementation and tests across workers only after the spec is clear.
   Each worker should own a narrow question, files or commands, evidence,
   remaining risk, and next step.
3. Integrate through `VectorMultiplayerEnv`, then run focused validation on the
   exact surface changed: source truth, product runtime, trainer wrapper,
   replay/final observation, or renderer.
4. Update docs with what was proven, what remains open, and the queue position.
   Keep pass counts as freshness notes, not broad completion claims.

## Path Guardrails

Keep environment, training, and rendering paths simple and guarded:

- One product runtime: `VectorMultiplayerEnv`.
- One trainer-surface direction: a bounded wrapper over `VectorMultiplayerEnv`
  that emits source-state visual stacks and metadata, not a second environment.
- Explicit trainer wrapper: action ids, `decision_source_frames`,
  `decision_ms`, opponent-policy sidecars, masks, rewards, and final
  observations are wrapper/replay facts, not native CurvyTron source facts.
- Explicit render modes: `browser_lines` is the source-state product visual
  path; `body_circles_fast` and other direct/fast paths must carry approximation
  metadata.
- Focused tests should prevent accidental fallback to fake/direct observations,
  stale debug tensors, metadata-only replay, or approximate render modes when a
  product/trainer proof claims source-state visual behavior.
- Approximation metadata belongs in reset/step/replay sidecars so training
  artifacts can be audited after the fact.

## Self-Critique

The work has been going poorly when route evidence gets treated like readiness,
when docs grow into broad catalogs, and when multiple old paths keep similar
names after the product direction has narrowed. The improvement is to keep one
small readiness note, one current queue, and one spec per larger change. Every
claim should say which surface it proves and which validation made it true.

The other weak spot is handoff drift. Workers can produce useful slices that do
not compose unless the integration owner keeps the proof ladder visible:
source -> product runtime -> trainer wrapper -> replay/final observation ->
training route. Use that ladder to accept or reject claims, not the size of the
test run or the number of artifacts produced.

## Queue Recorded

1. Controls source-frame fidelity.
2. End-to-end 2P product route.
3. Bonus probability and source defaults.
4. Hit-owner ordering.
5. Wider multiplayer.

Training-facing interpretation: keep the 2P wrapper/replay/final-observation
proof stable, use the new public 3P/4P mixed-row lifecycle proof as engine
evidence only, then add trainer/replay packaging for lifecycle and bonus stress
before promoting 3P/4P multiplayer training.
