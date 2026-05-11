# LightZero Environment Handoff

Date: 2026-05-09
Lane: LightZero side-lane environment research
Scope: local no-train adapter smoke plus working-memory notes.

## Short Answer

LightZero is probably the final training setup, but the environment boundary
is not ready to carry source-faithful CurvyTron learning yet.

Update from the local adapter smoke: a no-train, LightZero-shaped wrapper now
exists at `src/curvyzero/training/curvyzero_lightzero_smoke.py`. It wraps
`CurvyTronEnv` for one fixed ego player, uses
`observe_egocentric_rays_v0` for the `float32[106]` observation payload, returns
an `int8[3]` action mask and `to_play=-1`, fills the opponent action with a
fixed deterministic policy, and returns a small local timestep object shaped
like `BaseEnvTimestep`. It now has a narrow optional DI-engine boundary:
`optional_base_env_timestep_cls()` probes for `ding.envs.BaseEnvTimestep`, and
`LocalLightZeroTimestep.to_base_env_timestep(...)` converts the local row when
that real runtime exists.

Update from the registered wrapper pass: a thin env-registration module now
exists at `src/curvyzero/training/curvyzero_lightzero_env.py`. It registers
`curvyzero_v0_lightzero` when DI-engine is available, reuses the local smoke
semantics, exposes scalar reward space `shape=()`, publishes
`LIGHTZERO_CURVYZERO_IMPORT_NAMES =
("curvyzero.training.curvyzero_lightzero_env",)`, and converts steps to
`BaseEnvTimestep`. This is still not a real installed LightZero config/import
smoke, not a trainer run, and not a broader source-fidelity claim.

Important naming correction: ALE means Arcade Learning Environment, the Atari
emulator/API used for real Atari ROMs such as Pong. CurvyTron should not use
ALE. When older notes say "Atari-style CurvyTron," read that as "a future
LightZero visual stacked-frame wrapper for CurvyTron pixels," not Atari
emulation.

Focused local verification after this change:

- Local import probe:
  `uv run python -c "import importlib.util; ..."` -> `ding None`, `lzero None`,
  `lightzero None`.
- `uv run pytest tests/test_curvyzero_lightzero_env.py tests/test_curvyzero_lightzero_smoke.py tests/test_trainer_contract.py -q` ->
  `27 passed, 1 skipped`. The skip is the real
  `ding.envs.BaseEnvTimestep` conversion smoke because DI-engine/LightZero is
  not installed locally.
- `uv run ruff check src/curvyzero/training/curvyzero_lightzero_env.py tests/test_curvyzero_lightzero_env.py` -> `All checks passed!`

Full local verification after this adapter smoke was also green. Keep exact
outputs with the acceptance command logs rather than turning this handoff into a
status-count dashboard.

Next real installed-runtime smoke must prove the actual DI-engine/LightZero
path, not only local fallbacks:

- config says `env.type == "curvyzero_v0_lightzero"`;
- config says `env.import_names ==
  ["curvyzero.training.curvyzero_lightzero_env"]`;
- real `ENV_REGISTRY` constructs the env after import;
- `reset`, `seed(seed, dynamic_seed=False)`, and fixed-action `step` work;
- observation is `float32[106]`, action count is `3`, reward is scalar;
- terminal rows preserve final observation, final reward, zero terminal mask,
  `to_play=-1`, metadata hashes, joint action, and trace hash;
- `step()` returns real `ding.envs.BaseEnvTimestep`, not the local fallback.

LightZero wants an MDP-shaped environment row: one action in, one scalar reward
out, one `done` flag, plus an observation dict with `observation`,
`action_mask`, and `to_play`.

CurvyTron source behavior is held player control state advanced by elapsed-ms
server frames. In this handoff, `step`, `joint_action`, and any fixed decision
cadence are trainer-wrapper/replay abstractions.

The project-owned `CurvyTronEnv` wrapper is not shaped that way: all live-player
wrapper actions are assembled before each trainer step. The smallest sane bridge
is therefore an ego wrapper:

```text
LightZero controls one ego player.
The wrapper supplies the other live players' actions.
The wrapper returns one ego observation, one ego reward, one done flag, and
rich sidecar info.
```

This is good enough for a contained LightZero smoke. It is not full CurvyTron
self-play, not source fidelity, and not joint-action search.

Current environment truth:

- Source guardrails are strong for named fixtures.
- The vector fast path owns 20 fixture-backed transitions and the current
  B>1 speed defaults own 19 supported one-step fixtures.
- One natural multi-step trail-gap source scenario is verified separately. The
  scalar vector comparator also passes that taped full trace once, but B>1
  speed defaults still own only the forced one-step trail-gap fixtures.
- Training remains blocked by production reset/autoreset, lifecycle/spawn
  fidelity, replay writer/reader, row-local RNG, policy/search integration, and
  real LightZero/DI-engine glue. The local CurvyZero adapter smoke proves only
  the project-owned reset/step shape plus an optional timestep conversion
  boundary.
- A first pure policy-row mapping helper now exists at
  `src/curvyzero/training/policy_row_mapping.py`. It maps `obs[B,P,...]`,
  live masks, and legal masks into compact or padded ego policy rows, records
  env row ids and player ids, and maps selected action ids back to
  wrapper `joint_action[B,P]` with dead/padded rows left as no-op. The fixture
  actor bridge now uses this helper at the debug-pack boundary before its
  synthetic NumPy policy/search stand-in, then rehydrates selected actions back
  to wrapper `joint_action[B,P]` for synthetic feedback steps. This is still not
  LightZero, a learned checkpoint, real MCTS, or a source-fidelity claim for
  arbitrary policy moves.
- The sample-only fixture actor bridge can now write one validated local debug
  replay `.npz` chunk through `src/curvyzero/training/debug_actor_loop_replay.py`.
  Timed benchmark runs still stage replay in an in-memory ring and do not
  produce a production replay stream.

## Smallest CurvyZero Adapter Contract

The first real adapter should be `curvyzero-v0` through a project-owned
LightZero/DI-engine `BaseEnv`. It should prove the boundary, not train.

- `reset(seed)` resets one deterministic CurvyZero episode, records seed and
  episode metadata, clears the episode return, and returns a LightZero
  observation dict. It must not call source runners, scenario tools, vector
  actor bridges, or hidden autoreset paths.
- `step(action)` accepts one ego action id in `{0,1,2}`, asks a named
  deterministic opponent policy for the other live-player wrapper actions, builds
  the full wrapper joint action/control snapshot, calls the core env once, and
  returns `BaseEnvTimestep(next_obs, reward, done, info)`.
- `observation` in the local smoke is the pinned
  `curvyzero_egocentric_rays/v0`, flattened for LightZero as `float32[106]`
  with a separate `int8[3]` legal mask. Use
  `docs/working/environment/trainer_observation_reward_contract_v0_2026-05-09.md`
  and `src/curvyzero/env/trainer_contract.py` for the exact schema.
- `action_mask` is `np.int8` shape `(3,)` in `[left, straight, right]` order.
  Live ego rows expose legal actions; dead/inactive terminal rows must not
  silently create another decision.
- `reward` is one scalar ego reward. Use the named sparse round-outcome reward
  first; keep shaped or debug rewards in `info` or sidecar telemetry unless a
  training config explicitly selects them.
- `done` is `terminated[ego] or truncated[ego]`. Preserve both booleans and the
  terminal reason in `info`.
- `info` must include `eval_episode_return` on terminal rows for LightZero,
  plus ego id, wrapper joint action/control snapshot, opponent policy id/version,
  seed/episode ids, tick/step ids, terminal reason, final reward map, schema
  ids/hashes, and a trace hash or sidecar ref.
- Batch/self-play in LightZero means many single-agent env rows. It does not
  automatically mean all CurvyTron players are searched in one simultaneous
  game. The v0 adapter should use one fixed ego row; rotating ego rows,
  independent per-player rows, and full joint-action search are later policy
  choices.
- Fidelity blocks any stronger claim. Until reset/autoreset, final
  observation, observation/reward/info schemas, replay chunks, row-local RNG,
  lifecycle, natural spawn, bonuses, and broader vector semantics are done,
  this adapter can only be called a LightZero plumbing target.

## What Changed Since Previous Pass

- Snoop depth increased from public docs plus local smoke files to current
  upstream LightZero source and examples on `main`: custom env docs, config
  docs, CartPole/Atari/Gomoku/TicTacToe envs, `train_muzero`, collector,
  evaluator, policy, MCTS, `GameSegment`, `GameBuffer`, and MLP model code.
- The collector call order is now concrete:
  `ready_obs -> frame stack -> prepare_observation -> policy.forward ->
  initial_inference + MCTS -> env.step(dict env_id: action) -> GameSegment`.
- Shape expectations are clearer. The current local smoke already uses the
  pinned trainer observation shape: MLP, `frame_stack_num=1`, env observation
  shape `(106,)`, action mask shape `(3,)`, action ids `0,1,2`, scalar reward,
  scalar done, and fixed action-space policy target shape `(3,)`. Older notes
  that mention `(9,)` refer to the debug-observation prototype only.
- MCTS cost is not inside the env. It happens before each env step. The rough
  cost per LightZero decision is one initial model inference plus
  `num_simulations` recurrent model calls, tree traversal/backprop, and
  CPU/GPU conversions, then one env step.
- The smallest adapter recommendation is unchanged but sharper: keep this as a
  no-train optional boundary first, not a fixture actor bridge, not rotating
  ego rows, not joint-action search, and not a main-priority training lane.
- Local dependency truth: `ding`, `lzero`, and `lightzero` are not installed in
  the current `uv run` environment. The real `BaseEnvTimestep` conversion test
  therefore skips gracefully, and upstream/API compatibility remains unverified.
- New blocker: upstream `main` and the pinned package used by our Modal smokes
  may differ. A real implementation must verify the installed LightZero API
  before trusting source from GitHub `main`.

## Deep Source Read: Plain Facts

### 1. How LightZero Environments Plug In

LightZero does not call arbitrary project envs directly. The normal MuZero
path goes through DI-engine:

```text
main_config/create_config
  -> compile_config
  -> get_vec_env_setting(cfg.env)
  -> create_env_manager(...)
  -> collector/evaluator env manager
  -> MuZeroCollector / MuZeroEvaluator
```

The env itself is usually a DI-engine `BaseEnv` registered through
`ENV_REGISTRY.register("name")`. The config points to it through:

```python
create_config.env.type = "curvyzero_v0_lightzero"
create_config.env.import_names = ["curvyzero.training.curvyzero_lightzero_env"]
```

`reset()` returns a LightZero observation dict. `step(action)` returns a
`BaseEnvTimestep`. Gym/Gymnasium envs can be wrapped by
`LightZeroEnvWrapper`, but the CurvyZero path should use a project-owned
`BaseEnv` wrapper because we need ego-player metadata, opponent action
metadata, terminal diagnostics, and trace sidecars.

Required practical surface:

- `reset()` -> `{"observation": obs, "action_mask": mask, "to_play": -1}`
- `step(action)` -> `BaseEnvTimestep(next_obs_dict, reward, done, info)`
- `seed(seed, dynamic_seed=True/False)`
- `observation_space`
- `action_space`
- `reward_space`
- `legal_actions` or an action mask in each observation
- `random_action()`
- `close()`
- `__repr__()`

For non-board games, LightZero examples use `to_play=-1` and an all-ones mask
for fully legal discrete actions. Board-game examples use `to_play` for the
current player and have modes such as `self_play_mode`, `play_with_bot_mode`,
and `eval_mode`. That board-game path is alternating-turn, not simultaneous
multiplayer.

### 2. What the Self-Play Loop Calls Per Step

In the MuZero collector, "self-play" is still a collector loop over env rows.
It is not a general simultaneous multi-agent loop. Per ready env row:

```text
collector reads env.ready_obs
collector extracts action_mask, to_play, optional timestep/chance
collector asks each GameSegment for stacked observation
collector batches observations across ready env ids
prepare_observation(...) reshapes the batch for mlp or conv model
policy.forward(...) runs initial_inference and MCTS
policy returns one scalar action per ready env id
collector calls env.step({env_id: action})
env manager returns BaseEnvTimestep rows
collector stores search stats in GameSegment
collector appends action, next observation, reward, previous mask, previous to_play
collector updates action_mask/to_play from the next observation
on done, collector expects info["eval_episode_return"]
finished GameSegments are pushed into MuZeroGameBuffer
learner samples replay batches from MuZeroGameBuffer
```

The evaluator is the same shape, with eval policy settings and no training
push. It also expects terminal `eval_episode_return`; it may pass optional
`info["episode_info"]` to evaluation logging.

For board games, LightZero's env examples handle the opponent inside
`env.step` in `play_with_bot_mode` and `eval_mode`. In `self_play_mode`, they
advance one current-player action and alternate players. Neither mode matches
CurvyTron's true "all live players choose actions for the same tick" semantics.
For the first adapter, CurvyZero should behave like a non-board ego env:
LightZero controls one ego action; the wrapper fills the rest of the joint
action with a named deterministic opponent policy.

### 3. Observation, Action, Value, and Reward Shapes

Use these shapes for the first CurvyZero adapter:

```text
Env reset/step observation dict:
  observation: np.float32 shape (106,)
  action_mask: np.int8 shape (3,), order [left, straight, right]
  to_play: -1
  timestep: optional int

LightZero batched collector input with frame_stack_num=1:
  raw stack list: [B, 1, 106]
  prepare_observation(..., model_type="mlp"): [B, 106]

Policy/model config:
  model_type: "mlp"
  model.observation_shape: 106
  model.action_space_size: 3
  policy.env_type: "not_board_games"
  policy.action_type: "fixed_action_space"

Action:
  LightZero output action: scalar int in {0, 1, 2}
  env wrapper maps it to ego action id
  wrapper builds the full wrapper joint action/control snapshot internally

Reward:
  env reward returned to LightZero: scalar float
  first smoke reward: sparse ego round reward
  shaped/debug reward: info/sidecar only unless explicitly selected

Done:
  done = terminated[ego] or truncated[ego]
  terminal info must include eval_episode_return

Replay targets:
  action segment: scalar action ids
  reward segment: scalar rewards
  root value segment: scalar searched values
  child visit target: fixed vector length 3
```

Do not use frame stacking in the first smoke. LightZero's MLP helper flattens
`[B, S, O]` to `[B, S*O]`; keeping `frame_stack_num=1` avoids an easy
config/model shape mistake.

LightZero's model does not train on scalar values directly. The env supplies
scalar rewards; MuZero then converts reward/value targets into categorical
support distributions according to the model config. Policy targets come from
MCTS visit distributions. For fixed action spaces, the replay buffer stores a
full-length target vector of size `action_space_size`.

### 4. Where MCTS Cost Sits Relative to Env Step Cost

MCTS sits between observation collection and `env.step`.

For each collector decision over a batch of ready env rows:

```text
model.initial_inference(observation_batch)
for simulation in range(num_simulations):
    C++ tree traversal selects leaves
    model.recurrent_inference(latent_states, last_actions)
    C++ backprop updates roots
select one action per root
env.step(actions)
```

So the rough decision cost is:

```text
initial model inference
+ num_simulations * recurrent model inference
+ tree traversal/backprop
+ tensor/numpy/device transfers
+ one env-manager step
```

For cheap `curvyzero-v0` debug rows, LightZero policy/MCTS overhead will likely
dominate env stepping as soon as `num_simulations` is more than a tiny number.
This is expected and does not prove CurvyTron is slow. The first adapter smoke
should use `num_simulations=2` or no trainer at all. A later performance smoke
should separately time:

- raw `CurvyTronEnv.step`
- LightZero wrapper direct `step`
- env-manager `step`
- policy forward with `collect_with_pure_policy=True`
- policy forward with MCTS enabled

Do not mix those timing claims with source-fidelity claims.

### 5. Minimal Adapter to Build First

Build this first:

```text
CurvyZeroV0LightZeroEnv
  wraps curvyzero-v0
  fixed ego player
  deterministic named opponent policy
  pinned ray/scalar obs float32[106]
  discrete actions 0/1/2
  BaseEnv reset/step
  sidecar terminal trace metadata
  no-train config/import smoke
```

The first smoke should prove only these facts:

- wrapper reset returns a LightZero dict with expected keys and shapes
- wrapper step maps one ego action into one full wrapper joint action/control
  snapshot
- direct `CurvyTronEnv.step` and wrapper `step` agree for the same seed and
  scripted action trace
- terminal info preserves `eval_episode_return`
- compiled LightZero config imports the custom env
- LightZero is pointed at `curvyzero_v0_lightzero`, not CartPole or Atari
- no source-fidelity runner or debug vector actor path is called

Avoid these in the first adapter:

- fixture actor bridge
- vector debug batch/autoreset path
- rotating ego rows
- all-player wrapper joint action as the policy action space
- current-policy opponent loading
- source trace/scenario runners
- training claims

### 6. Exact Unknowns and Blockers

- Upstream source mismatch: this pass read LightZero GitHub `main` and public
  docs. Our Modal smokes pinned `LightZero==0.2.0`. Verify the installed
  package collector/env API before implementing the adapter.
- DI-engine env manager details: need a real config/import smoke to confirm
  how the installed env manager handles terminal rows, `ready_obs`, seeding,
  and final info for this wrapper.
- Final info retention: collector/evaluator require `eval_episode_return`, but
  custom nested diagnostics may be converted, dropped, or stringified by parts
  of DI-engine logging. Use an env-side sidecar for trace metadata.
- Mask dtype: examples use `np.int8`; policy uses `np.nonzero`. We should
  return explicit `np.int8` and avoid bool ambiguity.
- `timestep` handling: collector can read `obs["timestep"]`; not all examples
  rely on it. Include it only if the wrapper can keep it simple and stable.
- Reward support config: sparse ego rewards are likely fine, but value/reward
  support ranges should be checked against CurvyZero's exact reward scale
  before any training run.
- MLP normalization: CartPole config uses batch norm. Tiny batch training may
  be brittle. The no-train env smoke avoids this; training lane should choose
  the smallest stable config separately.
- Episode length: `curvyzero-v0` max-tick and truncation behavior must produce
  clean terminal rows under env-manager control. Direct wrapper smoke must hit
  both terminal and truncation cases eventually.
- Opponent identity: a "random" or "scripted" opponent is not self-play unless
  its source, seed, and version are recorded.
- Independent scorecard remains blocked for learning claims. A LightZero
  checkpoint can be produced before that, but it should not be called useful
  without an independent CurvyZero evaluation path.

## What LightZero Expects

LightZero custom envs are based on DI-engine `BaseEnv`, or a Gym env wrapped
with `LightZeroEnvWrapper`.

For the custom env path, `reset()` returns:

```python
{
    "observation": obs,
    "action_mask": action_mask,
    "to_play": -1,
}
```

For non-board-game discrete envs, `to_play=-1` and `action_mask` is normally an
all-ones `np.int8` array. `step(action)` returns:

```python
BaseEnvTimestep(lightzero_obs_dict, reward, done, info)
```

The env also needs Gym-style `observation_space`, `action_space`, and usually a
`legal_actions` property. Board-game envs add alternating-player ideas such as
`self_play_mode`, `play_with_bot_mode`, `eval_mode`, `legal_actions`,
`bot_action`, and `random_action`. That board-game path is not a natural fit
for CurvyZero's wrapper shape because all live players need a control choice for
the shared transition; native CurvyTron source is held control state over
elapsed-ms frames, not alternating turns.

Config is two-part:

- `main_config` carries environment counts and policy/model settings such as
  `collector_env_num`, `evaluator_env_num`, `n_evaluator_episode`,
  `model_type`, `observation_shape`, `action_space_size`, `num_simulations`,
  `batch_size`, `update_per_collect`, `n_episode`, `eval_freq`, and CUDA.
- `create_config` names the registered env and policy, for example
  `env.type="dummy_pong_lightzero"` and
  `env.import_names=["curvyzero.training.lightzero_dummy_pong_env"]`.

Important source: LightZero's custom env tutorial says the observation is a
dict with `observation`, `action_mask`, and `to_play`, and that `step` returns
`BaseEnvTimestep(...)`.

## Collection, Self-Play, Vector Envs, Replay

LightZero's `train_muzero` flow is:

1. Compile config.
2. Build collector and evaluator env managers through DI-engine.
3. Seed collector and evaluator envs.
4. Create policy, learner, collector, evaluator, and a MuZero-family
   `GameBuffer`.
5. Evaluate, collect episodes, push `GameSegment`s into replay, sample replay,
   and train.

The MuZero collector is episode-based and serial in orchestration, even though
it can manage multiple environment rows. It reads `env.ready_obs`, prepares a
batch of observations, calls `policy.forward(...)`, receives one action per
ready env id, then calls `env.step(actions)`.

Per transition, the collector stores:

- action chosen by LightZero;
- next `obs["observation"]`;
- reward;
- previous action mask;
- previous `to_play`;
- timestep if present;
- MCTS visit distribution and searched value, unless collecting with pure
  policy.

On terminal steps, it expects `info["eval_episode_return"]` for episode-level
logging. Replay/search stats are LightZero-owned, not environment-owned. The
environment should preserve CurvyZero replay/debug metadata in `info` or a
sidecar file; it should not try to produce LightZero's MCTS statistics.

Vector env managers in this path mean "many single-agent env rows." They do not
mean "many player rows inside one shared CurvyTron wrapper transition." If we
want each player to be a trainable ego row, we must create that mapping in our
wrapper or in a project-owned collector, not assume LightZero already does it.

The first mapping bridge is now intentionally small and pure:
`build_policy_row_mapping(...)` produces policy rows from multiplayer env
arrays, and `policy_rows_to_joint_action(...)` rehydrates selected ego action
ids into a joint action grid while leaving non-decision rows as straight/no-op.
The fixture actor bridge now uses this mapping around its synthetic local
policy/search stand-in. This starts the independent ego-row path only; the
search itself is still synthetic and it is not full joint-action search.

LightZero MCTS roots are batched over env roots with legal action lists. The
C++ tree code can batch traversal/backprop and model inference, but it still
selects one action per env row. It does not automatically search every
CurvyTron player's wrapper action/control choice.

## What CurvyTron Must Provide

For a LightZero ego wrapper around `curvyzero-v0`, the environment lane must
provide these pieces:

- `reset(seed)` creates a deterministic episode and does not hide source
  runners or scenario tools in the training step.
- `observe(ego_player)` returns a fixed-shape copied observation. The first
  LightZero smoke should use the pinned trainer target
  `curvyzero_egocentric_rays/v0` with LightZero flat shape `float32[106]`.
  Older `float32[9]` observations are debug-only notes and should not be used
  for new coach-facing work.
- `legal_action_mask(ego_player)` returns action order `[left, straight,
  right]`, with action ids `0`, `1`, `2`.
- The trainer wrapper advances exactly one chosen decision cadence. That
  cadence is a wrapper choice, not a native CurvyTron source concept. The
  wrapper may map one decision to one or more elapsed-ms source frames.
- The wrapper turns one LightZero action into a complete joint action/control
  snapshot for all live players by using a named opponent policy.
- Reward is the scalar ego reward from `StepResult.rewards[ego]`.
- Done is `terminated[ego] or truncated[ego]`, or the env-row equivalent.
- `info` includes terminal reason, winner/loser ids, draw/timeout fields,
  schema ids/hashes, wrapper joint action, opponent policy id, seed, step/tick
  ids, final reward map, and a trace hash.
- Terminal `info` includes `eval_episode_return`, because LightZero's collector
  logs from that field.
- The wrapper records enough sidecar telemetry to replay or diagnose a run
  outside LightZero: episode seed, action trace, opponent policy version,
  observation schema id/hash, action schema id/hash, reward schema id/hash,
  terminal reason, final observations if needed, and source/evidence refs if
  present.

Current `CurvyTronEnv` already has many pieces:

- two-player toy-v0 only;
- `reset(seed) -> dict[player_id, np.ndarray]`;
- `last_reset_info`;
- `observe(ego_player)`;
- `legal_action_mask(ego_player)`;
- wrapper API `step(joint_action) -> StepResult`;
- fixed action ids `0=left`, `1=straight`, `2=right`;
- pinned ray/scalar trainer observation `float32[106]` through the current
  adapter helper, plus older flat privileged debug observation `float32[9]`
  that must stay labeled debug-only;
- sparse round outcome reward;
- per-player `terminated`, `truncated`, and `infos`;
- no hidden autoreset after terminal or truncated step.

The missing piece is a CurvyTron-specific LightZero wrapper and smoke, not a
change to the core environment.

## Current Conflicts

All-live-player wrapper control:
LightZero chooses one action per env row. CurvyZero needs a full wrapper joint
action/control snapshot for all live players. An ego wrapper hides opponent
actions inside `step`; that is useful plumbing, but it is not full multiplayer
search.

Ego rows:
CurvyZero's long-term training shape may want every live player to contribute
an ego transition. LightZero's vector env rows are environment rows, not
player rows. A rotating-ego wrapper can approximate this later, but the first
smoke should use one fixed ego.

Joint actions:
Encoding joint actions as the LightZero action space is possible but dangerous.
With 2 players and 3 actions that is 9 actions; with 6 players it is 729
actions per wrapper decision. Do not start there.

Debug observations:
The current `float32[9]` global observation is privileged debug state. It is
fine for adapter plumbing, but every run must label it as debug. It is not the
future learned CurvyTron observation.

Debug/vector rewards:
The single-env toy reward and vector debug reward are not the same contract.
The LightZero wrapper should use the named sparse outcome reward first and log
any shaped/debug returns as telemetry, not silently train on them.

Source fidelity:
Source runners, scenario JSON, common traces, JS probes, and Modal fidelity
jobs are evidence machinery. The LightZero adapter must not import or call them
from reset, step, search, collection, or evaluation.

Reset/autoreset:
Core `CurvyTronEnv` raises after terminal/truncated step until reset. Some
debug actor/vector paths have internal autoreset behavior for benchmark rows.
Do not build the LightZero adapter on those debug paths first. Preserve the
terminal transition and final info before any env-manager reset.

Policy staleness:
An internal opponent policy must be explicitly named and versioned. If it is a
frozen checkpoint later, the checkpoint ref belongs in info/sidecar metadata.
Do not call that "self-play" unless the opponent source is named: fixed
script, random, latest policy-only, frozen checkpoint, or searched.

Final info loss:
LightZero stores its own replay segments. CurvyZero-specific diagnostics can be
lost unless terminal info or an env sidecar captures them. Sidecar telemetry is
not optional for trust.

## Smallest Adapter Smoke

Already done for dummy Pong:

- `src/curvyzero/training/lightzero_dummy_pong_env.py` implements a
  DI-engine `BaseEnv` wrapper.
- It uses one ego action, a scripted opponent, `to_play=-1`, action mask
  `[1, 1, 1]`, tabular shape `(10,)`, and terminal sidecar telemetry.
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
  patches a CartPole MuZero config to the custom env and passed a no-train
  feature-fit probe.
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py` exists
  for the tiny trainer attempt, but the independent CurvyZero checkpoint
  scorecard is still blocked.

Next smallest environment-lane smoke once a real DI-engine/LightZero runtime is
available:

```text
CurvyZeroLightZeroEnv around curvyzero-v0, config/import only, no trainer.
```

The smoke should:

- construct the wrapper directly;
- call `seed(seed, dynamic_seed=False)` if the wrapper exposes it;
- call `reset()`;
- step one or a few fixed ego actions with a deterministic named opponent;
- compare the wrapper joint-action path against direct wrapper API
  `CurvyTronEnv.step(...)` for the same seed/actions;
- report observation shape `(106,)`, action space `3`, action mask, reward,
  done, `eval_episode_return` on terminal, schema ids/hashes, and trace hash;
- compile a tiny LightZero config that targets only the custom CurvyZero env;
- not call `train_muzero`.

Do not build the fixture actor bridge first. It drags in vector debug packing,
internal autoreset, and source-fidelity concerns before the single-env contract
is proven.

Do not build joint-action search first. It makes the adapter look powerful
while hiding the main semantic mismatch.

After this smoke passes, the training lane can decide whether to run a tiny
LightZero train smoke on `curvyzero-v0`. That is a separate training decision,
not an environment-lane prerequisite, and stronger claims still require
lifecycle, spawn, RNG, reset/autoreset, replay, and source-backed observation
evidence.

## Practical Wrapper Shape

Sketch only:

```python
@ENV_REGISTRY.register("curvyzero_v0_lightzero")
class CurvyZeroV0LightZeroEnv(BaseEnv):
    def reset(self):
        obs_by_player = self._env.reset(seed=self._next_seed())
        self._last_obs_by_player = obs_by_player
        self._episode_return = 0.0
        self._trace = []
        return self._lz_obs()

    def step(self, action):
        ego_action = int(action)
        joint_action = {self.ego: ego_action}
        for player in self._env.agents:
            if player != self.ego and self._env.legal_action_mask(player).any():
                joint_action[player] = self._opponent_policy.action(
                    self._env.observe(player),
                    self._env.legal_action_mask(player),
                )

        result = self._env.step(joint_action)
        reward = float(result.rewards[self.ego])
        done = bool(result.terminated[self.ego] or result.truncated[self.ego])
        self._episode_return += reward

        info = dict(result.infos[self.ego])
        info.update({
            "eval_episode_return": self._episode_return if done else None,
            "ego_player": self.ego,
            "opponent_policy_id": self._opponent_policy.id,
            "joint_action": joint_action,
            "final_rewards": result.rewards if done else None,
            "trace_hash": self._trace_hash(),
        })
        return BaseEnvTimestep(self._lz_obs(), reward, done, info)
```

Use the pinned trainer observation contract:

```python
{
    "observation": observe_egocentric_rays_v0(...).observation.astype(np.float32),
    "action_mask": env.legal_action_mask(ego).astype(np.int8),
    "to_play": -1,
}
```

The older debug-observation prototype used this shape and should stay labeled
as debug-only if it appears in old notes or experiments:

```python
{
    "observation": debug_obs.astype(np.float32),  # shape (9,)
    "action_mask": env.legal_action_mask(ego).astype(np.int8),
    "to_play": -1,
}
```

The exact ray angles, channel order, scalar order, schema hashes, sparse reward,
done/truncated semantics, and terminal info keys are in
`docs/working/environment/trainer_observation_reward_contract_v0_2026-05-09.md`.

Config should patch the proven CartPole MLP path:

```text
policy.type = muzero
policy.model.model_type = mlp
policy.model.observation_shape = 106
policy.model.action_space_size = 3
policy.env_type = not_board_games
collector_env_num = 1
evaluator_env_num = 1
n_evaluator_episode = 1
num_simulations = 2
batch_size = 8
update_per_collect = 1
cuda = false
```

## Checklist Before Training Uses LightZero

- [ ] Target is named: `dummy_pong_lightzero` or `curvyzero-v0-lightzero`, not
      vague "CurvyTron".
- [ ] Wrapper is outside the core simulator. Core env imports no LightZero,
      DI-engine, Gym, or torch.
- [ ] Direct wrapper reset/step works locally or on Modal before any trainer
      call.
- [ ] Compiled LightZero config targets the custom env, not stock CartPole or
      stock Atari Pong.
- [ ] Observation shape, dtype, schema id, and schema hash are written. For the
      first CurvyZero LightZero smoke this should be
      `curvyzero_egocentric_rays/v0` flattened to `float32[106]`. Older
      `float32[9]` rows are debug-only wiring notes.
- [ ] Action ids are fixed at `0=left`, `1=straight`, `2=right`; LightZero
      `action_space_size` is `3`.
- [ ] Legal mask order is `[left, straight, right]`; dtype conversion to
      LightZero-compatible `int8` is explicit.
- [ ] Reward schema is named; shaped/debug rewards are sidecar telemetry unless
      explicitly selected as the training reward.
- [ ] Done means `terminated OR truncated`; terminal and truncation are both
      preserved in info.
- [ ] Terminal info includes `eval_episode_return`.
- [ ] Seed, dynamic seed policy, episode index, wrapper joint-action trace,
      opponent policy id/version, and trace hash are recorded.
- [ ] Opponent action source is deterministic under the recorded seed.
- [ ] The wrapper errors clearly on unknown actions and missing live-player
      actions before calling the core env.
- [ ] No source-fidelity runner, scenario runner, common-trace diff, browser
      probe, or Modal fidelity job is called inside reset/step/search.
- [ ] No hidden autoreset loses the terminal transition or final observation.
- [ ] Sidecar episode JSONL or equivalent is written for scorecard/debug
      recovery.
- [ ] The no-train config/import smoke passes and returns `ok: true`.
- [ ] If a tiny train smoke is run, it mirrors LightZero checkpoints/logs and
      CurvyZero sidecar telemetry into `curvyzero-runs`.
- [ ] Any learning claim waits for an independent CurvyZero scorecard or an
      explicit written blocker saying that scorecard is not ready.
- [ ] Any "self-play" label states the exact opponent source: fixed script,
      random, policy-only latest, frozen checkpoint, or searched.

## Hard Stop Conditions

Stop and hand back to the training/coach lane if any of these happen:

- The CurvyZero wrapper needs a LightZero or DI-engine fork before reset/step.
- The wrapper cannot preserve seed, wrapper joint action, opponent policy,
  terminal rewards, or trace hash.
- LightZero can only work by encoding full wrapper joint actions as the policy
  action space.
- The config/import smoke silently targets CartPole or Atari Pong.
- Final episode info is dropped and the sidecar cannot recover it.
- The wrapper depends on source-fidelity runners or fixture actor/debug
  autoreset paths.
- Integration glue becomes larger than a tiny project-owned trainer would be.

## Speed Handoff Update

Latest local smoke truth for the environment speed lane:

```sh
uv run python scripts/run_environment_fidelity_matrix.py --run smoke --format plain
```

Clean rerun result on 2026-05-09: the smoke matrix passed the focused vector,
mixed comparator, batch-row, and actor-loop checks. Treat it as regression smoke,
not as a LightZero training-readiness claim.

The actor quick is still local/debug timing. With `B=2`, `repeat=1`, debug
events, synthetic policy/search, and in-memory replay staging, the three group
rows reported roughly 4.4k to 5.4k env rows/sec and actor-step p50 latency from
0.324 ms to 0.410 ms. Treat those as regression smoke numbers only.

The latest user-provided larger no-event actor-loop debug run used:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 2 32 128 --repeat 20 --warmup 2 --rollout-steps 4 \
  --hidden-dim 16 --simulations 2 --chunk-steps 16 \
  --event-modes no-event --format plain
```

It passed `summary=passed:19 failed:0 unsupported:0
batch_preflight_failed:False`. At `B=128`, p50 actor-step latency was
`0.333 ms` for `P1`, `0.561 ms` for `P2`, and `0.461 ms` for `P3`; env
rows/sec were about `378k`, `227k`, and `267k`; ego rows/sec were about `378k`,
`453k`, and `801k`. Env step was about `33%`, `47%`, and `47%` of the loop;
synthetic policy/search was about `8%`, `13%`, and `25%`. This is still local
debug timing, not real LightZero training, learned-model inference, or MCTS.

The practical parallelism scout was:

```sh
PYTHONPATH=src python3 scripts/benchmark_selfplay_parallel_bridge.py \
  --batch 128 --steps 100 --warmup 10 --workers 4 \
  --modes serial serial-sharded thread process --format plain
```

Local result: threads were slower than serial for the toy object-env bridge
(`0.865x` steady-loop speedup, p50 `9.946 ms`), while coarse process shards
helped (`3.615x` steady-loop speedup, p50 `1.485 ms`). This measures independent
actor shards with no per-step IPC, no LightZero env manager, no central
inference queue, and no MCTS.

Implication for LightZero:

- Do not put multiprocessing, Modal calls, or source-fidelity tools inside the
  CurvyZero LightZero wrapper.
- If LightZero/DI-engine process env managers are used later, time them as
  coarse env-row sharding and report startup, steady loop, and action latency
  separately.
- The first performance split must measure direct wrapper `step`, env-manager
  `step`, policy forward without MCTS if available, policy forward with MCTS,
  and replay push separately.
- Expect real MCTS/model work to become the bottleneck once `num_simulations`
  is meaningful. Env process sharding helps only until the central
  policy/search batch or queue becomes the slow part.
- Keep Modal GPU for JAX/Mctx search smokes and coarse jobs. Do not use Modal
  Queue/Dict or remote function calls per action or per environment tick.

## Local Files That Matter

- `src/curvyzero/env/core.py`: current `CurvyTronEnv` reset/observe/legal-mask
  and joint-action step contract.
- `docs/design/environment/training_interface_contract.md`: training must use
  `curvyzero.env`; source-fidelity tools are evidence machinery, not trainer
  API.
- `docs/design/environment/observation_reward_contract.md`: current real
  observation/reward/done/info contract and the debug-vs-learned observation
  split.
- `docs/design/deterministic_environment.md`: toy-v0 scope and source-derived
  deviations.
- `docs/research/lightzero_integration.md`: original LightZero fit critique.
- `docs/research/lightzero_feature_fit_for_curvyzero.md`: skeptical feature-fit
  matrix and smoke requirements.
- `docs/research/muzero_framework_vs_project_owned.md`: LightZero-first adapter
  plan with Mctx fallback.
- `docs/decisions/0005-main-pong-repository-library-choice.md`: accepted
  immediate lane: try LightZero custom dummy Pong before project-owned Mctx
  trainer.
- `src/curvyzero/training/lightzero_dummy_pong_env.py`: existing working
  pattern for a single-ego LightZero wrapper.
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`:
  no-train config/import feature-fit smoke.
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`: tiny
  trainer smoke scaffold and artifact mirror.
- `docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md`:
  dummy Pong adapter passed the feature-fit gate; independent checkpoint
  scorecard remains blocked.
- `docs/experiments/2026-05-09-modal-lightzero-cartpole-tiny-train-smoke.md`:
  stock LightZero CartPole MuZero trainer did run on Modal.
- `docs/experiments/2026-05-09-modal-lightzero-pong-env-smoke.md`: stock Atari
  Pong env creation is blocked by ROM management, so it is not the next env
  lane.

There is no `docs/source` directory in this repo right now. For LightZero
source/config references, use upstream docs and GitHub.

## External Sources

- LightZero custom environment tutorial:
  https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
- LightZero config guide:
  https://opendilab.github.io/LightZero/tutorials/config/config.html
- LightZero worker docs for `MuZeroCollector`:
  https://opendilab.github.io/LightZero/api_doc/worker/index.html
- LightZero `MuZeroCollector` source view:
  https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_collector.html
- LightZero MCTS tree-search docs:
  https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html
- LightZero GitHub repo:
  https://github.com/opendilab/LightZero
- Upstream custom env tutorial source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/docs/source/tutorials/envs/customize_envs.md
- Upstream config tutorial source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/docs/source/tutorials/config/config.md
- LightZero `train_muzero.py` upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/entry/train_muzero.py
- LightZero `MuZeroCollector` upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/worker/muzero_collector.py
- LightZero `MuZeroEvaluator` upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/worker/muzero_evaluator.py
- LightZero `MuZeroPolicy` upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/policy/muzero.py
- LightZero `GameSegment` upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/mcts/buffer/game_segment.py
- LightZero `MuZeroGameBuffer` upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/mcts/buffer/game_buffer_muzero.py
- LightZero MCTS C++ tree wrapper upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/mcts/tree_search/mcts_ctree.py
- LightZero observation preparation utility:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/mcts/utils.py
- LightZero MLP MuZero model source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/model/muzero_model_mlp.py
- LightZero `LightZeroEnvWrapper` upstream source:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/envs/wrappers/lightzero_env_wrapper.py
- Upstream CartPole MuZero config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/classic_control/cartpole/config/cartpole_muzero_config.py
- Upstream CartPole LightZero env:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/classic_control/cartpole/envs/cartpole_lightzero_env.py
- Upstream Atari MuZero config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py
- Upstream Atari LightZero env:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/envs/atari_lightzero_env.py
- Upstream TicTacToe board-game env:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/board_games/tictactoe/envs/tictactoe_env.py
- Upstream Gomoku board-game env:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/board_games/gomoku/envs/gomoku_env.py

## Concise Next Actions

1. Keep the current local work at the optional `BaseEnvTimestep` boundary while
   `ding`/`lzero`/`lightzero` are absent.
2. Do not start a train smoke or a larger adapter lane before lifecycle, spawn,
   RNG, reset/autoreset, replay, and source-backed observation blockers are
   explicit.
3. When a real DI-engine/LightZero runtime is available, verify the installed
   package API first, then run only a no-train config/import smoke.
4. Leave fixture actor bridge, joint-action search, rotating ego rows, and
   source-fidelity runners out of the first adapter.
