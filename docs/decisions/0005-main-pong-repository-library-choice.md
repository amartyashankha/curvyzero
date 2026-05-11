# ADR-0005: Try LightZero First For The Immediate Dummy Pong MuZero Run

Date: 2026-05-09
Status: Accepted for the immediate lane

## Context

The user is confused, reasonably, because the docs mention stock LightZero,
Mctx, dummy Pong baselines, Modal smokes, and a possible project-owned trainer.
The practical decision is now:

Try LightZero first for the immediate dummy Pong MuZero run.

That does not mean LightZero owns CurvyZero forever. It means the next useful
experiment should adapt dummy Pong to LightZero's custom environment interface
before we write a project-owned JAX/Mctx trainer. Mctx stays as the fallback
search dependency if LightZero fights the env interface, hides required
metadata, or later becomes too constraining for CurvyTron.

## Plain Decision

The next main implementation lane is:

```text
LightZero custom dummy Pong MuZero smoke
```

This lane should call a real LightZero MuZero-family trainer on a
CurvyZero-owned dummy Pong wrapper, produce a checkpoint, and write a small
CurvyZero summary with eval and survival/loss-delay telemetry.

Do not build the project-owned JAX/Mctx trainer first. Keep it documented as
the fallback path, not the immediate path.

## Why Two Libraries Are Still In Play

Two libraries are in play because they solve different problems:

- LightZero is a full trainer. It already ran stock CartPole MuZero on Modal.
  It can give us replay, learner updates, MCTS integration, logs, and
  checkpoints without writing those pieces ourselves.
- Mctx is a search library. It gives us batched JAX MuZero/Gumbel MuZero search
  calls, but not a trainer, replay buffer, optimizer loop, checkpoint format,
  env wrapper, or Modal run manager.

Using both names is not a mistake if the roles stay separate:

- immediate Pong run: try LightZero custom env first;
- fallback or later ownership lane: project-owned JAX/Mctx if LightZero blocks
  us or CurvyTron needs tighter control.

## What We Reuse From LightZero

Reuse these pieces from LightZero for the immediate smoke:

- MuZero or Gumbel MuZero trainer entrypoint;
- collector/evaluator loop;
- replay and learner/update machinery;
- MCTS integration inside the trainer;
- PyTorch model/update path;
- learner/evaluator logs;
- `.pth.tar` checkpoints;
- config pattern from the already-proven CartPole smoke.

Do not fork LightZero for v0. Add a thin project-side adapter and patch a tiny
config down to a cheap Modal smoke.

## What We Write Ourselves

Write only the pieces LightZero cannot know:

- `CurvyZeroDummyPongLightZeroEnv`, an experiment-only wrapper around
  `PongEnv`;
- a tiny opponent policy inside the wrapper, starting with `random_uniform` or
  `lagged_track_ball_1`;
- seed handling and deterministic action-trace logging;
- observation conversion from dummy Pong ego state to the LightZero model shape;
- sidecar episode telemetry: wins/losses, survival steps, truncation, score
  return, shaped loss-delay return, opponent policy, seed, and trace hash;
- Modal summary copying into the existing `curvyzero-runs` layout;
- a tiny config/training smoke module under `src/curvyzero/infra/modal/`.

This is deliberately smaller than writing replay, targets, model code,
optimizer updates, checkpointing, and eval ourselves.

## Adapter Interface

The adapter should expose dummy Pong as one ego-controlled single-agent env.
LightZero chooses one action for one ego player. The wrapper supplies the
opponent action internally.

Target wrapper behavior:

```python
class CurvyZeroDummyPongLightZeroEnv:
    def __init__(self, cfg):
        self.env = PongEnv(...)
        self.ego = "player_0"
        self.opponent_policy_id = cfg.get("opponent_policy_id", "random_uniform")

    def reset(self, seed=None):
        self.seed = seed
        self.trace = []
        self.last_obs_by_agent = self.env.reset(seed=seed)
        return self._lightzero_obs()

    def step(self, action):
        opponent_action = self._opponent_action()
        joint_action = {"player_0": int(action), "player_1": opponent_action}
        result = self.env.step(joint_action)
        self.trace.append(joint_action)
        obs = self._lightzero_obs()
        reward = float(result.rewards[self.ego])
        done = bool(result.terminated or result.truncated)
        info = self._info(result)
        return obs, reward, done, info
```

LightZero observation:

```python
{
    "observation": obs,                 # float32 shape [10] for v0
    "action_mask": np.ones(3, np.int8), # A=3: up, stay, down
    "to_play": -1,                      # single-agent MuZero convention
}
```

Use tabular ego observations first because that is closest to the already
working CartPole MLP smoke. Convert `PongObservation` to this fixed float32
vector:

```text
ego_paddle_y / height
opponent_paddle_y / height
ego_paddle_x / width
opponent_paddle_x / width
ball_dx_forward / width
ball_dy_from_ego_center / height
ball_vx_forward
ball_vy
ball_y / height
step / max_steps
```

So v0 uses `observation_shape = 10`, `feature_mode = tabular_ego`, and
`model_type = "mlp"`. Raster is a later feature mode after the trainer path
works; do not make visual quality the first framework-integration blocker.

Required `info` fields:

- `seed`;
- `ruleset_id`;
- `observation_schema_id`;
- `reward_schema_id`;
- `action_schema_id`;
- `ego_agent`;
- `opponent_policy_id`;
- `joint_action`;
- `step`;
- `terminated`;
- `truncated`;
- `winner`;
- `score_return`;
- `survival_steps`;
- `shaped_loss_delay_return`;
- `trace_hash`.

The exact base class can follow the smallest LightZero/DI-engine custom env
path that works. Prefer a Gym-like wrapper if LightZero accepts it; use
DI-engine `BaseEnv` registration only if needed.

## Config Shape

Start from the stock CartPole MuZero config because it already runs on Modal
and uses an MLP model. Patch it instead of starting from Atari Pong. At the time
of this decision, the ROM/ALE setup had not been solved yet, and Atari Pong
brought image/segment assumptions we did not need for dummy Pong. Later work did get installed
`LightZero==0.2.0` Atari Pong running through ALE; that does not change the
reason dummy Pong existed as the first custom-env smoke.

Tiny config surface:

```text
env_id: curvyzero-dummy-pong-v0
env_type: curvyzero_dummy_pong_lightzero
policy_type: muzero
model_type: mlp
observation_schema_id: dummy_pong_lightzero_tabular_v0
feature_mode: tabular_ego
observation_shape: 10
action_space_size: 3
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
cuda: false
num_simulations: 2 first, then 4 or 8
batch_size: 8
update_per_collect: 1
max_env_step: 64
max_train_iter: 2
eval_freq: 1
seed: 0
```

Use MuZero first. Add a Gumbel MuZero switch only after the plain config starts
and writes a checkpoint.

## Modal Image

Use the same LightZero image family as the working CartPole smoke:

```python
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "LightZero==0.2.0",
)
```

Do not add AutoROM or Atari ROM handling for this lane. This is a custom dummy
Pong env, not stock Atari Pong.

Only add extra packages if the adapter truly needs them. If the wrapper needs
Gym spaces and they are not already present through LightZero/DI-engine, add
the smallest compatible Gym dependency and record it in the returned package
versions.

## Modal And Artifact Integration

Add two Modal modules:

```text
src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py
src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py
```

The config/import smoke should:

- build the image;
- import LightZero, DI-engine, torch, and the dummy Pong adapter;
- create/reset/step the adapter for fixed seeds;
- capture the patched LightZero config without training;
- return observation shape, action count, env type, policy type, package
  versions, and a direct-env versus wrapper summary.

The train smoke should:

- call LightZero's trainer, not a project-owned loop;
- cap env steps, train iterations, simulations, batch size, and evaluator
  episodes;
- scan the LightZero experiment directory for learner logs, evaluator logs,
  TensorBoard events, and `.pth.tar` checkpoints;
- run or extract a tiny eval summary;
- write a CurvyZero summary under the `curvyzero-runs` Volume with:
  `run.json`, `attempt.json`, `train/summary.json`, copied LightZero artifact
  refs, and final eval telemetry.

Checkpoint rule: keep the original LightZero checkpoint format for the smoke.
Copy or reference it from CurvyZero's run layout; do not invent a new
checkpoint conversion step yet.

## Smallest Commands

### 0. LightZero Feature-Fit Audit Gate

Work item: add this either as
`src/curvyzero/infra/modal/lightzero_dummy_pong_feature_fit_smoke.py` or as
`--mode feature-fit` on
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`.
This is a dry Modal audit that exposes LightZero fit gaps before trainer noise.

Command after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode feature-fit \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

The report must cover env reset/step, observation shape and schema, legal
action handling or a clear all-actions-legal note, reward/info telemetry,
trainer entrypoint fit, checkpoint discovery patterns, and the independent
CurvyZero scorecard path.

Pass if every required feature is present or explicitly not needed, with no
`missing`, `unknown`, or `hidden_by_framework` fields.

Fail if reset/step cannot run, observation shape is ambiguous, legal actions or
telemetry cannot be represented, the trainer would not target the custom env,
checkpoint discovery is unclear, or no independent scorecard path exists.

### 1. LightZero Custom-Env Config/Import Smoke

Work item: add
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`.
This is a dry smoke. It imports the framework and adapter, patches the tiny
config, creates/resets/steps one custom dummy Pong env, and captures the config
without calling the trainer.

Command after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

Pass if LightZero/DI-engine can import, see the custom dummy Pong env,
create/reset/step it, preserve `A=3`, tabular observation shape, and legal-action
surface if needed, include the feature-fit audit fields, and return the patched
MuZero config as clear JSON without calling the trainer.

Fail if the adapter needs invasive DI-engine work, hides CurvyZero
seed/action/reward traces, cannot create/reset/step before training, or falls
back to stock CartPole/Pong instead of the custom env.

### 2. LightZero Custom Dummy Pong Tiny MuZero Train Smoke

Prerequisite: the config/import smoke passes.

Work item: add
`src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`.
This calls LightZero's own MuZero trainer on the same custom dummy Pong adapter
with brutal caps, scans LightZero logs/checkpoints, and writes a small
CurvyZero summary. It must not use stock Atari Pong or Atari ROM setup.

Command after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --seed 0
```

Pass if:

- the command calls a real LightZero MuZero trainer;
- `ok: true`;
- at least one LightZero checkpoint exists;
- the summary includes eval plus wins/losses, survival steps, truncation rate,
  score return, and shaped loss-delay return;
- seed/action trace metadata is preserved or sidecar-logged;
- checkpoint discovery finds the LightZero checkpoint path;
- an independent CurvyZero scorecard runs or is scheduled outside LightZero's
  evaluator;
- artifacts are visible from the CurvyZero Volume layout.

Fail if:

- LightZero cannot create/reset/step the custom env cleanly;
- the trainer needs invasive DI-engine or LightZero forks;
- required Pong telemetry cannot be recovered;
- seeds/action traces are hidden;
- checkpoint discovery fails;
- no independent CurvyZero scorecard path exists;
- the smoke takes more glue than a tiny project-owned Mctx trainer would.

### 3. Project-Owned Mctx Tiny Trainer Fallback/Comparison

Work item: only after LightZero fails, or as a comparison after the two
LightZero smokes, add
`src/curvyzero/infra/modal/mctx_known_env_tiny_train_smoke.py` and the smallest
owned trainer behind it. This is not first anymore. It must include
representation, dynamics, prediction, Mctx search, replay rows, target
building, optimizer updates, checkpoint write/load, and eval in one container.

Command after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_known_env_tiny_train_smoke \
  --env dummy_line_duel \
  --iterations 2 \
  --episodes-per-iter 4 \
  --num-simulations 4 \
  --batch-size 8 \
  --seed 0
```

Pass if it returns `ok: true`, has finite normalized Mctx action weights,
reloads a checkpoint for eval, and clearly labels the result
`project-owned MuZero/Mctx`.

Fail if it is only a search benchmark, omits replay/update/checkpoint/eval, or
grows into a general owned framework before one tiny trainer result.

## Mctx Fallback

If LightZero fails those gates, consider the project-owned JAX/Mctx fallback.
That fallback would reuse Mctx search but write the missing trainer pieces:
env batching, replay rows, model, targets, optimizer updates, checkpoints,
eval, and Modal run management. Do not start it before recording the exact
LightZero blocker.

Mctx also remains the later choice if CurvyTron needs tighter control over
simultaneous self-play, replay metadata, stochastic rules, or checkpoint
format than LightZero can comfortably provide.

## Other Repos

- Muax: possible later JAX/Mctx helper if direct Mctx would require too much
  trainer code. Do not use before the LightZero smoke.
- muzero-general: useful reading for replay and target structure, but not the
  immediate Modal backbone.
- EfficientZero: later Atari/sample-efficiency reference. Too heavy for the
  first dummy Pong custom-env smoke.

## Consequences

- The next practical direction is LightZero-first.
- Do not call the LightZero custom dummy Pong smoke "project-owned MuZero."
  It is `LightZero MuZero` on a project-owned env wrapper.
- Do not describe Mctx benchmarks as training.
- Do not start stock Atari Pong training in this lane; avoid ROM work until the
  custom dummy Pong LightZero path has been tested.
- If LightZero passes, continue toward a slightly larger dummy Pong run and
  then a Curvy-style wrapper.
- If LightZero fails, write down the exact failure and then use Mctx as the
  fallback search dependency for a project-owned trainer.

## Links

- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/research/lightzero_feature_fit_for_curvyzero.md`
- `docs/research/muzero_framework_vs_project_owned.md`
- `docs/research/muzero_reference_examples.md`
- `docs/research/muzero_repo_baseline_options.md`
- `docs/research/mctx_integration.md`
- `docs/design/muzero_modal_architecture.md`
- `docs/runbooks/training_smokes.md`
