# LightZero Coach-Facing Env Requirements

Status: draft working note
Date: 2026-05-10
Scope: docs-only synthesis for the coach/training side.

This note is a draft. It collects the current best understanding of what a
LightZero-facing CurvyTron environment will need. It should absorb follow-up
findings from LightZero, replay, visual observations, and vector-runtime work.

The project target is one fast, source-faithful CurvyTron runtime for self-play:
`VectorMultiplayerEnv` under hardening. LightZero should be a
trainer/coach adapter on top of that runtime, not the semantic owner of
CurvyTron rules.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Current Best Understanding

LightZero's MuZero path expects an MDP-like env row:

```text
reset() -> observation dict
step(action) -> BaseEnvTimestep(obs, reward, done, info)
```

`done` is the LightZero/DI-engine-facing `terminated or truncated` value. The
env manager handles autoreset, so `env.step()` should not silently reset itself
after a terminal step.

The observation dict should include:

- `observation`
- `action_mask`
- `to_play=-1`
- optional `timestep`

For CurvyTron this probably means a single-ego wrapper first:

- LightZero chooses one ego action.
- The wrapper supplies the other live players' actions through explicit
  versioned opponent policies or snapshot policies.
- The wrapper builds the full CurvyZero wrapper action map, then converts it to
  source CurvyTron real-time control state for elapsed-ms frames.
- The wrapper reports `native_control_model_id`,
  `trainer_control_wrapper_id`, and `decision_ms` in sidecar info.
- The wrapper returns one ego observation, one ego reward, one done flag, and
  enough sidecar info to audit the wrapper decision and source transition.
- The wrapper logs the full `joint_action` as wrapper/replay sidecar. Do not
  start with full wrapper joint-action MCTS; branching is `3^P`.

This is not a claim that LightZero's built-in self-play already matches native
CurvyTron source updates. Native source behavior is current player control
state advanced through elapsed-ms frames. LightZero vector envs are best treated
as many single-ego rows until proven otherwise.
Any single-ego or no-bonus wrapper restriction is an explicit temporary profile
or adapter config, not a reason to avoid reconstructing source-default
CurvyTron behavior.

The scalar/ray path exists and remains useful as a sidecar:

```text
source-backed CurvyTron state -> trainer float32[B,P,106] rays/scalars -> replay-v0 chunks
```

But the project's intended CurvyTron training path is visual LightZero-style
input. Do not treat scalar/ray timing as the main coach-facing optimizer target
unless the visual path is explicitly abandoned. Visual LightZero here means
CurvyTron frames in a LightZero-compatible conv-stack shape, not arcade
emulators, ROMs, or ALE.

## Required Env Surface

The eventual coach-facing environment should provide these fields and policies.

Observations:

- LightZero reset returns the observation dict directly.
- LightZero step returns the next observation dict as the `obs` field inside
  `BaseEnvTimestep`.
- A named observation schema id and hash.
- Fixed shape, dtype, and value range.
- A copy-safe policy observation, not raw mutable env state.
- A scalar/ray path already drafted as `curvyzero_egocentric_rays/v0`
  (`float32[106]` for the LightZero MLP adapter).
- A visual LightZero path for conv-style MuZero is the main target.

Action masks:

- Fixed action ids: `0` left, `1` straight, `2` right.
- Mask shape `(3,)`, preferably explicit `int8` for LightZero.
- Start with a fixed all-ones live action mask unless deliberately testing
  LightZero's `varied_action_space` path.
- Dead, inactive, terminal-padding, or reset-pending rows should expose no
  legal decision.
- Mask order and action schema hash must be recorded in reset/replay metadata.

`to_play` and player identity:

- For the first non-board-game LightZero adapter, use `to_play=-1`.
- Keep CurvyTron ego identity in sidecar metadata: ego player id, env row id,
  player row id, and any seat/opponent policy identity.
- Do not leak stable player index or color into the learned observation unless a
  future schema explicitly chooses that.

Reward:

- Return one scalar ego reward to LightZero.
- Start with the sparse round-outcome reward schema unless a run explicitly
  selects a shaped reward.
- Keep shaped/debug rewards in `info` or sidecar telemetry unless they have a
  named reward schema id and hash.
- Record the all-player final reward map at terminal rows.

Done/truncated:

- Preserve separate `terminated` and `truncated` flags.
- `done = terminated or truncated` for LightZero.
- Terminal outcome should win reward selection if a row is both terminated and
  truncated.
- `info` should include terminal reason, winner ids, loser ids, death player
  ids, draw/timeout flags, tick/step ids, `eval_episode_return`, and schema
  ids/hashes.

Final observation:

- The terminal step must expose the final observation before any reset.
- If autoreset is enabled later, reset observation and final observation must be
  kept separate.
- Replay rows need enough final-observation data to rebuild the terminal
  transition without guessing.

Replay metadata:

- Episode id.
- Reset seed and reset source.
- Ruleset id/hash.
- Observation, action, and reward schema ids/hashes.
- Ego player id and full player list.
- Opponent policy id/version or snapshot checkpoint ref.
- Opponent actions supplied by the wrapper.
- Full wrapper action-map trace or a `joint_action` sidecar ref.
- Native control model id, trainer control wrapper id, and decision window in
  milliseconds.
- Terminal reason, final reward map, and final observation policy.
- Optional source/evidence refs when a rollout is tied to a fixture or scenario.

Reset/autoreset policy:

- `reset(seed)` should create one deterministic episode and return the
  observation dict.
- Any reset info that is needed for replay or audit should be available through
  explicit sidecar metadata or a documented wrapper path.
- DI-engine's env manager handles autoreset.
- `env.step()` should not silently autoreset or replace the terminal
  transition.
- Hidden autoreset should not replace the terminal transition.
- If vector rows autoreset, the order should be:
  terminal transition -> final observation/reward/info/replay capture -> row
  reset -> next reset observation.
- Row-local seed history must survive autoreset.

## Visual LightZero Target

Visual LightZero means LightZero-compatible CurvyTron frames, not arcade
emulators or ROMs.
First target:

- Return one grayscale CurvyTron frame as `float32` in `[0,1]` with shape
  `(1,64,64)`.
- Configure LightZero to stack frames to `(4,64,64)` with
  `frame_stack_num=4`, `image_channel=1`, and `observation_shape=(4,64,64)`.
- If the env returns pre-stacked `(4,64,64)` observations instead, configure
  LightZero with `frame_stack_num=1` and `image_channel=4`.

Open questions:

- Raster source: render from the source-faithful state arrays, a browser/canvas
  reference, or a project-owned renderer?
- Frame stack ownership: let LightZero stack one-frame env observations, or
  return pre-stacked observations from the wrapper?
- Color mode: grayscale for LightZero parity, RGB for easier debugging, or both
  with separate schema ids?
- Shape and range after the first target: keep `64x64` normalized grayscale,
  add RGB/debug schemas, or support another size?
- Scaling/cropping: preserve map aspect ratio, crop around ego, show the full
  arena, or use an ego-centered viewport?
- Action repeat and decision cadence: how long should one wrapper decision
  window (`decision_ms`) hold source controls across elapsed-ms server frames?
- Reward clipping: avoid clipping for the source reward contract, or add an
  explicit trainer-only clipped reward schema if LightZero config needs it?
- Terminal frame: should the final visual observation include death/score state,
  post-event render state, or the last pre-terminal movement frame?
- Diagnostics: how to store a replayable visual sidecar without making pixels
  the only source of truth?

## Current Env Status

- The coach currently uses official LightZero visual Pong only as setup
  validation and as the closest trainer/control pattern for CurvyTron.
- A dummy Pong bridge exists for bridge mechanics, not as a CurvyTron claim.
- A local no-train CurvyZero smoke uses toy ray observations `float32[106]` and
  `int8[3]` masks.
- `vector_runtime` has a supported B>1 batch step for the current fixture-backed
  transition slice.
- Basic terminal flags exist for that narrow vector path.
- Debug observation/reward/mask packing exists for vector shapes, but it is not
  the final learned observation or reward.
- A scalar/ray trainer observation contract exists as a non-visual sidecar.
- The old usable optimizer path was source-backed CurvyTron state -> trainer
  `float32[B,P,106]` rays/scalars -> replay-v0 chunks. Keep it for diagnostics,
  but do not make it the main optimizer target for visual CurvyTron.
- Repo-native self-play can use `obs[B,P,D] -> compact live ego rows ->
  policy/search -> wrapper action_map[B,P] / joint_action sidecar ->
  trainer env.step`.
- A strict 1v1/no-bonus vector trainer handoff now builds pinned
  `float32[106]` ray observations from vector body circles,
  `float32[B,2,106]` final observation arrays, and sparse final reward maps
  before autoreset planning. The strict public env is a fixed-decision wrapper
  over source control state and exposes `native_control_model_id`,
  `trainer_control_wrapper_id`, and `decision_ms`.
- A local LightZero-shaped smoke wrapper exists at
  `src/curvyzero/training/curvyzero_lightzero_smoke.py`, and a thin registered
  wrapper exists at `src/curvyzero/training/curvyzero_lightzero_env.py` as
  `curvyzero_v0_lightzero`. The registered wrapper reuses the local smoke
  semantics and only adds the DI-engine env type/timestep boundary. Installed
  LightZero/DI-engine scalar config-import/no-train smoke passes. Installed
  debug visual config-import/no-train smoke also passes for
  `curvyzero_debug_visual_tensor_lightzero` with a real `BaseEnvTimestep`,
  LightZero-facing `float32[1,64,64]`, model target `(4,64,64)`, action space
  `3`, and no ALE identity. Real training and visual
  collect/search/replay/learner profiling are still unproven.
- The expected CurvyTron bridge is single-ego first: the wrapper chooses
  opponent actions and builds native input/control state for the
  source-faithful env.
- Production replay integration, public full env reset/autoreset semantics,
  visual rendering, and performance integration are not ready. Current
  replay-v0 is 1v1; 3P/4P needs a generalized schema with opponent policy
  ids/actions and wrapper action sidecars.
- No source-faithful visual renderer exists yet.
- A debug visual smoke exists as `debug_visual_tensor` /
  `curvyzero_debug_occupancy_gray64/v0`: raw `uint8[1,64,64]` CHW occupancy,
  normalized to `float32[1,64,64]` CHW for LightZero-facing payloads. It must
  not claim source visual fidelity.
- A separate debug visual LightZero wrapper now exists as
  `curvyzero_debug_visual_tensor_lightzero`. Installed LightZero/DI-engine
  no-train config/import/env-factory reset/step smoke passes on Modal with a
  real `BaseEnvTimestep`, LightZero-facing env payload `float32[1,64,64]`,
  model stack `(4,64,64)`, action space `3`, and no ALE identity. This is
  debug visual plumbing only, not source-fidelity or training evidence.
- Public reset-to-terminal parity is proven for the named strict
  1v1/no-bonus long fixture using the source fixture random tape and warmup
  policy. Do not generalize that to broad lifecycle or full CurvyTron.
- Broader lifecycle, 3P/4P public env reset/warmup/replay/observation parity,
  bonuses, long multiplayer rollout evidence, visual rendering, and visual
  raster contracts still need source-backed work. Multiplayer is still in
  scope; current fast-runtime multiplayer proof is only direct seeded 3P/4P
  no-bonus wall-scoring/order canaries.

## Concrete TODO

1. Build and profile the first visual CurvyTron adapter surface:
   debug occupancy `(1,64,64)` now, eventual source-faithful grayscale
   `(1,64,64)` frame later -> LightZero stack `(4,64,64)` -> conv-style config.
2. Define the LightZero adapter contract in one place: observation dict,
   `action_mask`, `to_play`, optional `timestep`, reward, done, and required
   `info` fields.
3. Keep the strict vector final-observation/reward handoff scoped to the landed
   1v1/no-bonus replay path until broader public env claims exist.
4. Promote replay-v0 integration from shape bridge to production trainer rows:
   real observations, rewards, actions, final observations, seeds, and metadata.
5. Define the first opponent policy contract for single-ego LightZero rows,
   including policy id/version and deterministic seeding.
6. Keep full wrapper joint-action MCTS out of the first adapter. Use one ego
   decision row, versioned opponent policies for non-ego actions, and replay
   sidecars for the full wrapper action map.
7. Decide visual details now: source of pixels, color, resizing/cropping,
   cadence, value range, and LightZero-owned stacking from env `(1,64,64)`
   frames to trainer `(4,64,64)` stacks.
8. Move from installed no-train visual adapter smoke to a bounded whole-loop
   timing harness. The smoke passes; policy/search/replay/learner/eval timing
   and any training claim are still pending.
9. Extend vector/runtime lifecycle support toward full reset, pre-step timers,
   terminal handoff, autoreset, and row-local seed history.
10. Keep speed reports whole-loop: env step, observation packing, policy/search,
   replay staging, reset/autoreset, and completed useful games per minute.
11. Keep claims labeled: fixture-backed runtime, toy-v0 trainer smoke,
    source-fidelity evidence, and LightZero training results are separate facts.

## Related Notes

- `docs/design/environment/training_interface_contract.md`
- `docs/working/environment/trainer_observation_reward_contract_v0_2026-05-09.md`
- `docs/working/environment/replay_terminal_seed_contract_2026-05-09.md`
- `docs/working/environment/vector_state_schema.md`
- `docs/working/environment/vector_lifecycle_plan_2026-05-09.md`
- `docs/working/environment/selfplay_speed_lane_2026-05-09.md`
- `docs/working/training/lightzero_environment_handoff_2026-05-09.md`
- `docs/working/curvytron_visual_lightzero_adapter_plan_2026-05-10.md`
- `docs/working/lightzero_official_visual_pong_pattern_2026-05-09.md`

## Sources

- LightZero repository: https://github.com/opendilab/LightZero
- LightZero docs: https://opendilab.github.io/LightZero/
- DI-engine repository: https://github.com/opendilab/DI-engine
